"""
evaluate.py

Loads a trained ActorCritic checkpoint and runs a number of evaluation
episodes on a single (non-vectorized) env.

Key difference from training: actions are chosen *greedily* (argmax over
logits) rather than sampled from the Categorical distribution. During
training, sampling gives exploration; during evaluation we want the
policy's best guess, not a random draw.

Run with:
    python evaluate.py
    python evaluate.py --episodes 20 --render
"""

import argparse

import numpy as np
import torch

from config import ENV_CONFIG, ENV_ID, NET_ARCH, OVERTAKE_BONUS, TTC_THRESHOLD, TTC_WEIGHT
from networks import ActorCritic
from reward_wrapper import RewardShapingWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False, deterministic: bool = True):
    import gymnasium as gym
    import highway_env  # noqa: F401  (registers highway-fast-v0)

    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(ENV_CONFIG)
    env = RewardShapingWrapper(
        env,
        ttc_threshold=TTC_THRESHOLD,
        ttc_weight=TTC_WEIGHT,
        overtake_bonus=OVERTAKE_BONUS,
    )

    # NOTE: observation_space can still reflect the default config until the
    # first reset() happens, so we derive obs_dim from an actual reset
    # observation rather than from env.observation_space.shape directly.
    first_obs, _ = env.reset(seed=0)
    obs_dim = int(np.prod(first_obs.shape))
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions, hidden_sizes=NET_ARCH).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    episode_returns = []
    episode_lengths = []
    episode_overtakes = []
    crash_count = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=1000 + episode)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_overtakes = 0

        while not done:
            obs_flat = obs.reshape(1, -1)
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, _ = model.forward(obs_t)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    from torch.distributions import Categorical
                    action = Categorical(logits=logits).sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1
            ep_overtakes += info.get("n_overtakes", 0)

            if render:
                env.render()

        if info.get("crashed", False):
            crash_count += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_overtakes.append(ep_overtakes)
        print(f"episode {episode + 1}/{n_episodes}: return={ep_return:.3f} length={ep_length} "
              f"overtakes={ep_overtakes} crashed={info.get('crashed', False)}")

    env.close()

    print()
    print("--- evaluation summary ---")
    print(f"episodes:        {n_episodes}")
    print(f"mean return:     {np.mean(episode_returns):.3f} +/- {np.std(episode_returns):.3f}")
    print(f"mean length:     {np.mean(episode_lengths):.1f}")
    print(f"mean overtakes:  {np.mean(episode_overtakes):.2f}")
    print(f"crash rate:      {crash_count / n_episodes:.1%}")

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_overtakes": float(np.mean(episode_overtakes)),
        "crash_rate": crash_count / n_episodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_highway_scratch.pt",
                         help="path to the saved model checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="open a window and render episodes")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of taking argmax (off by default)")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )
