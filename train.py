"""
train.py

Main entry point. Pure orchestration - no algorithm logic lives here:
    1. build vectorized env (env_utils)
    2. build ActorCritic model (networks)
    3. collect a rollout into RolloutBuffer (buffer)
    4. compute GAE, run a PPO update (ppo)
    5. log diagnostics, repeat
    6. after training: save two diagnostic plots (learning curve + crash rate)

Run with:
    python train.py
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from buffer import RolloutBuffer
from config import GAMMA, GAE_LAMBDA, LR, N_ENVS, N_STEPS, NET_ARCH, SEED, TOTAL_TIMESTEPS
from env_utils import make_vec_env
from networks import ActorCritic
from ppo import PPOAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many of the most recently finished episodes to average over when
# reporting mean_ep_return / crash_rate at each update
ROLLING_WINDOW = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a .pt checkpoint to resume training from")
    parser.add_argument("--save-every", type=int, default=20_000,
                         help="save a checkpoint every N timesteps (default: 20000)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                         help="directory to save periodic checkpoints in")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                         help="override the total number of timesteps from config.py")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(args.save_dir, exist_ok=True)

    env = make_vec_env(n_envs=N_ENVS, seed=SEED)

    obs = env.reset()
    obs_dim = int(np.prod(obs.shape[1:]))
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions, hidden_sizes=NET_ARCH).to(DEVICE)

    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    agent = PPOAgent(model, DEVICE)
    buffer = RolloutBuffer(n_steps=N_STEPS, n_envs=N_ENVS, obs_dim=obs_dim)

    done = np.zeros(N_ENVS, dtype=np.float32)

    n_updates = args.total_timesteps // (N_STEPS * N_ENVS)
    global_step = 0
    start_time = time.time()
    next_save_at = args.save_every

    episode_returns = np.zeros(N_ENVS, dtype=np.float32)
    episode_speed_sum = np.zeros(N_ENVS, dtype=np.float32)
    episode_overtakes = np.zeros(N_ENVS, dtype=np.int64)
    episode_steps = np.zeros(N_ENVS, dtype=np.int64)
    finished_returns = []
    finished_crashed = []
    finished_speeds = []
    finished_overtakes = []
    finished_llm_scores = []

    # history recorded once per update, used for the plots at the end
    history = {
        "update": [],
        "step": [],
        "mean_return": [],
        "crash_rate": [],
        "mean_speed": [],
        "mean_overtakes": [],
        "entropy": [],
        "explained_variance": [],
        "approx_kl": [],
        "lr": [],
    }

    header = (f"{'update':>10} {'step':>9} {'fps':>5} {'return':>9} "
              f"{'crash%':>7} {'speed':>7} {'overtk':>7} {'entropy':>8} {'kl':>7} "
              f"{'exp_var':>8} {'lr':>9}")
    print(header)
    print("-" * len(header))

    for update in range(1, n_updates + 1):
        buffer.reset()

        # linear LR annealing: 1.0 at the start of training -> 0.0 at the end.
        # Keeps early updates fast/large while shrinking step size later on,
        # so the policy can settle into a stable optimum instead of
        # continuing to take large, potentially destabilizing steps.
        progress_remaining = 1.0 - (update - 1) / n_updates
        current_lr = LR * progress_remaining
        agent.set_lr(current_lr)

        for _ in range(N_STEPS):
            obs_flat = obs.reshape(N_ENVS, -1)
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            next_obs, reward, next_done, infos = env.step(action_np)

            buffer.add(
                obs=obs_flat,
                action=action_np,
                log_prob=log_prob.cpu().numpy(),
                reward=reward,
                done=done,
                value=value.cpu().numpy(),
            )

            episode_returns += reward
            episode_speed_sum += np.array([info.get("speed", 0.0) for info in infos], dtype=np.float32)
            episode_overtakes += np.array([info.get("n_overtakes", 0) for info in infos], dtype=np.int64)
            episode_steps += 1

            for i, d in enumerate(next_done):
                if d:
                    finished_returns.append(episode_returns[i])
                    finished_crashed.append(bool(infos[i].get("crashed", False)))
                    finished_speeds.append(episode_speed_sum[i] / max(episode_steps[i], 1))
                    finished_overtakes.append(episode_overtakes[i])
                    if infos[i].get("llm_judge_score") is not None:
                        finished_llm_scores.append(infos[i]["llm_judge_score"])
                    episode_returns[i] = 0.0
                    episode_speed_sum[i] = 0.0
                    episode_overtakes[i] = 0
                    episode_steps[i] = 0

            obs = next_obs
            done = next_done.astype(np.float32)
            global_step += N_ENVS

            if global_step >= next_save_at:
                checkpoint_path = os.path.join(args.save_dir, f"ppo_highway_step{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  [checkpoint saved: {checkpoint_path}]")
                next_save_at += args.save_every

        # bootstrap value for the observation *after* the last stored step
        with torch.no_grad():
            obs_flat = obs.reshape(N_ENVS, -1)
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE)
            _, last_values = model.forward(obs_t)
            last_values = last_values.cpu().numpy()

        advantages, returns = buffer.compute_gae(last_values, done, gamma=GAMMA, lam=GAE_LAMBDA)
        b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values = buffer.get_flat(
            advantages, returns
        )

        stats = agent.update(b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values)

        fps = int(global_step / (time.time() - start_time))
        recent_returns = finished_returns[-ROLLING_WINDOW:]
        recent_crashed = finished_crashed[-ROLLING_WINDOW:]
        recent_speeds = finished_speeds[-ROLLING_WINDOW:]
        recent_overtakes = finished_overtakes[-ROLLING_WINDOW:]
        recent_llm_scores = finished_llm_scores[-ROLLING_WINDOW:]
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        crash_rate = float(np.mean(recent_crashed)) * 100 if recent_crashed else float("nan")
        mean_speed = float(np.mean(recent_speeds)) if recent_speeds else float("nan")
        mean_overtakes = float(np.mean(recent_overtakes)) if recent_overtakes else float("nan")
        mean_llm_score = float(np.mean(recent_llm_scores)) if recent_llm_scores else float("nan")

        history["update"].append(update)
        history["step"].append(global_step)
        history["mean_return"].append(mean_return)
        history["crash_rate"].append(crash_rate)
        history["mean_speed"].append(mean_speed)
        history["mean_overtakes"].append(mean_overtakes)
        history["entropy"].append(stats["entropy"])
        history["explained_variance"].append(stats["explained_variance"])
        history["approx_kl"].append(stats["approx_kl"])
        history["lr"].append(current_lr)

        print(f"{update:>10} {global_step:>9} {fps:>5} {mean_return:>9.2f} "
              f"{crash_rate:>6.1f}% {mean_speed:>6.2f} {mean_overtakes:>7.2f} "
              f"{stats['entropy']:>8.3f} {stats['approx_kl']:>7.4f} "
              f"{stats['explained_variance']:>8.3f} {current_lr:>9.2e} llm={mean_llm_score:.2f}")

    env.close()
    torch.save(model.state_dict(), "ppo_highway_scratch.pt")
    print("\nTraining finished. Model saved to ppo_highway_scratch.pt")

    _save_plots(history)


def _save_plots(history: dict):
    """
    Saves the two most useful diagnostic plots:
      1. learning curve - mean episode return over training
      2. crash rate over training (the safety metric that matters most
         for a driving policy - a policy can have a decent return while
         still crashing often, so this is tracked separately)
    """
    steps = history["step"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, history["mean_return"], color="tab:blue")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"mean episode return (rolling {ROLLING_WINDOW})")
    ax.set_title("Learning curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("learning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, history["crash_rate"], color="tab:red")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"crash rate % (rolling {ROLLING_WINDOW})")
    ax.set_title("Crash rate over training")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("crash_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, history["mean_speed"], color="tab:green")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"mean speed, m/s (rolling {ROLLING_WINDOW})")
    ax.set_title("Speed over training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("speed.png", dpi=150)
    plt.close(fig)

    print("Saved plots: learning_curve.png, crash_rate.png, speed.png")


if __name__ == "__main__":
    main()

