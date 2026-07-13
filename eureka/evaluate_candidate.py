"""
evaluate_candidate.py

Runs deterministic (argmax) evaluation episodes for one trained candidate
and reports crash_rate / mean_speed / mean_overtakes / mean_raw_return.

Deliberately reports mean_raw_return (the env's own built-in reward,
info["raw_reward"]) alongside the candidate's ground-truth behavior
metrics, and NOT the candidate's shaped return - fitness.py only uses
crash_rate/mean_speed/mean_overtakes specifically so a candidate can't
inflate its own fitness by writing a shaping function that just returns
large values regardless of actual driving quality.
"""

import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401

from config import ENV_CONFIG, ENV_ID, NET_ARCH
from networks import ActorCritic
from eureka.candidate_wrapper import CandidateRewardWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_candidate(model_path: str, module_path: str, n_episodes: int = 10) -> dict:
    import importlib

    module = importlib.import_module(module_path)
    shaping_fn = module.shaping_reward

    env = gym.make(ENV_ID)
    env.unwrapped.configure(ENV_CONFIG)
    env = CandidateRewardWrapper(env, shaping_fn)

    first_obs, _ = env.reset(seed=0)
    obs_dim = int(np.prod(first_obs.shape))
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions, hidden_sizes=NET_ARCH).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    crash_count = 0
    speeds, overtakes_list, raw_returns = [], [], []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=2000 + episode)
        done = False
        speed_sum, steps, overtakes, raw_return = 0.0, 0, 0, 0.0

        while not done:
            obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                logits, _ = model.forward(obs_t)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            speed_sum += info.get("speed", 0.0)
            overtakes += info.get("n_overtakes", 0)
            raw_return += info.get("raw_reward", 0.0)
            steps += 1

        crashed = bool(info.get("crashed", False))
        if crashed:
            crash_count += 1

        mean_ep_speed = speed_sum / max(steps, 1)
        speeds.append(mean_ep_speed)
        overtakes_list.append(overtakes)
        raw_returns.append(raw_return)

        print(f"      ep {episode + 1}/{n_episodes}: "
              f"speed={mean_ep_speed:.1f} overtakes={overtakes} "
              f"raw_return={raw_return:.2f} crashed={crashed}", flush=True)

    env.close()

    return {
        "crash_rate": crash_count / n_episodes,
        "mean_speed": float(np.mean(speeds)),
        "mean_overtakes": float(np.mean(overtakes_list)),
        "mean_raw_return": float(np.mean(raw_returns)),
    }
