"""
train_candidate.py

Runs a short PPO training run for ONE reward candidate, reusing the shared
ActorCritic / RolloutBuffer / PPOAgent stack (networks.py / buffer.py /
ppo.py) with a smaller env count and a shorter timestep budget, since the
goal here is comparing candidates against each other, not producing the
best possible final policy.

Logs compact training progress (steps, return, crash rate, entropy) so long
candidate runs are not silent. Optional shaping_components from the candidate
are accumulated into rolling-window snapshots for reward reflection.
"""

import json
import os
import time

import numpy as np
import torch

from buffer import RolloutBuffer
from config import GAMMA, GAE_LAMBDA, LR, N_STEPS, NET_ARCH
from networks import ActorCritic
from ppo import PPOAgent
from eureka.env_factory import make_candidate_vec_env
from eureka.eureka_config import EUREKA_N_ENVS
from eureka.logging_utils import get_logger, TrainProgressTable

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROLLING_WINDOW = 20


def train_candidate(module_path: str, total_timesteps: int, seed: int = 0) -> str:
    """
    module_path: dotted import path of the candidate's code, e.g.
                 "eureka.candidates.gen0_cand2" (must define shaping_reward)

    Returns the path to the saved model checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_candidate_vec_env(module_path, n_envs=EUREKA_N_ENVS, seed=seed)

    obs = env.reset()
    obs_dim = int(np.prod(obs.shape[1:]))
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions, hidden_sizes=NET_ARCH).to(DEVICE)
    agent = PPOAgent(model, DEVICE)
    buffer = RolloutBuffer(n_steps=N_STEPS, n_envs=EUREKA_N_ENVS, obs_dim=obs_dim)

    done = np.zeros(EUREKA_N_ENVS, dtype=np.float32)
    n_updates = total_timesteps // (N_STEPS * EUREKA_N_ENVS)
    global_step = 0
    start_time = time.time()
    log_every = max(1, n_updates // 10)

    episode_returns = np.zeros(EUREKA_N_ENVS, dtype=np.float32)
    episode_speed_sum = np.zeros(EUREKA_N_ENVS, dtype=np.float32)
    episode_overtakes = np.zeros(EUREKA_N_ENVS, dtype=np.int64)
    episode_steps = np.zeros(EUREKA_N_ENVS, dtype=np.int64)
    finished_returns = []
    finished_crashed = []
    finished_speeds = []
    finished_overtakes = []

    # Lazily keyed by component name once a non-empty shaping_components dict
    # is observed. Empty for legacy bare-float candidates.
    episode_component_sums: dict[str, np.ndarray] = {}
    finished_components: dict[str, list[float]] = {}
    component_history: dict[str, list[float]] = {}

    short_name = module_path.split(".")[-1]
    progress = TrainProgressTable(short_name, n_updates)
    logger.info(
        "candidate training started",
        extra={
            "event": "train_start",
            "candidate_module": short_name,
            "device": str(DEVICE),
            "updates": n_updates,
            "envs": EUREKA_N_ENVS,
            "seed": seed,
        },
    )

    for update in range(1, n_updates + 1):
        buffer.reset()

        progress_remaining = 1.0 - (update - 1) / n_updates
        current_lr = LR * progress_remaining
        agent.set_lr(current_lr)

        for _ in range(N_STEPS):
            obs_flat = obs.reshape(EUREKA_N_ENVS, -1)
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
            episode_speed_sum += np.array(
                [info.get("speed", 0.0) for info in infos], dtype=np.float32
            )
            episode_overtakes += np.array(
                [info.get("n_overtakes", 0) for info in infos], dtype=np.int64
            )
            episode_steps += 1

            for i, info in enumerate(infos):
                components = info.get("shaping_components") or {}
                if not components:
                    continue
                for key, value in components.items():
                    if key not in episode_component_sums:
                        episode_component_sums[key] = np.zeros(
                            EUREKA_N_ENVS, dtype=np.float64
                        )
                        finished_components.setdefault(key, [])
                        component_history.setdefault(key, [])
                    episode_component_sums[key][i] += float(value)

            for i, d in enumerate(next_done):
                if d:
                    finished_returns.append(episode_returns[i])
                    finished_crashed.append(bool(infos[i].get("crashed", False)))
                    finished_speeds.append(episode_speed_sum[i] / max(episode_steps[i], 1))
                    finished_overtakes.append(episode_overtakes[i])
                    steps_i = max(episode_steps[i], 1)
                    for key, sums in episode_component_sums.items():
                        finished_components[key].append(float(sums[i] / steps_i))
                        sums[i] = 0.0
                    episode_returns[i] = 0.0
                    episode_speed_sum[i] = 0.0
                    episode_overtakes[i] = 0
                    episode_steps[i] = 0

            obs = next_obs
            done = next_done.astype(np.float32)
            global_step += EUREKA_N_ENVS

        with torch.no_grad():
            obs_flat = obs.reshape(EUREKA_N_ENVS, -1)
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE)
            _, last_values = model.forward(obs_t)
            last_values = last_values.cpu().numpy()

        advantages, returns = buffer.compute_gae(last_values, done, gamma=GAMMA, lam=GAE_LAMBDA)
        b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values = buffer.get_flat(
            advantages, returns
        )
        stats = agent.update(b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values)

        if update == 1 or update % log_every == 0 or update == n_updates:
            elapsed = time.time() - start_time
            fps = int(global_step / elapsed) if elapsed > 0 else 0
            recent_returns = finished_returns[-ROLLING_WINDOW:]
            recent_crashed = finished_crashed[-ROLLING_WINDOW:]
            recent_speeds = finished_speeds[-ROLLING_WINDOW:]
            recent_overtakes = finished_overtakes[-ROLLING_WINDOW:]
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            crash_rate = float(np.mean(recent_crashed)) * 100 if recent_crashed else float("nan")
            mean_speed = float(np.mean(recent_speeds)) if recent_speeds else float("nan")
            mean_overtakes = float(np.mean(recent_overtakes)) if recent_overtakes else float("nan")

            component_snapshots: dict[str, float] = {}
            for key, values in finished_components.items():
                recent = values[-ROLLING_WINDOW:]
                if recent:
                    mean_val = float(np.mean(recent))
                    component_snapshots[key] = mean_val
                    component_history[key].append(mean_val)

            progress.add_row(
                update=update,
                global_step=global_step,
                fps=fps,
                mean_return=mean_return,
                crash_rate_pct=crash_rate,
                mean_speed=mean_speed,
                mean_overtakes=mean_overtakes,
            )
            logger.debug(
                "training update",
                extra={
                    "event": "train_update",
                    "candidate_module": short_name,
                    "update": update,
                    "global_step": global_step,
                    "fps": fps,
                    "mean_return": mean_return,
                    "crash_rate_pct": crash_rate,
                    "mean_speed": mean_speed,
                    "mean_overtakes": mean_overtakes,
                    "component_snapshots": component_snapshots,
                    "entropy": stats["entropy"],
                    "approx_kl": stats["approx_kl"],
                },
            )

    env.close()

    module_name = module_path.split(".")[-1]
    checkpoint_dir = os.path.join("eureka", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{module_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    if component_history:
        components_path = os.path.join(checkpoint_dir, f"{module_name}_components.json")
        with open(components_path, "w", encoding="utf-8") as f:
            json.dump({"component_history": component_history}, f)

    elapsed = time.time() - start_time
    logger.info(
        "candidate training finished",
        extra={
            "event": "train_complete",
            "candidate_module": short_name,
            "duration_s": round(elapsed, 4),
            "checkpoint": checkpoint_path,
        },
    )

    return checkpoint_path
