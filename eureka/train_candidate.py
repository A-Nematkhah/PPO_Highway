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


def component_sidecar_path(candidate_name: str) -> str:
    """Return the path for a candidate's component sidecar JSON file."""
    checkpoint_dir = os.path.join("eureka", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{candidate_name}_components.json")


def _remove_stale_component_sidecar(candidate_name: str) -> None:
    """Remove a stale sidecar file before candidate training begins."""
    path = component_sidecar_path(candidate_name)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning(
                "failed to remove stale component sidecar",
                extra={"event": "component_sidecar_remove_failed", "path": path},
            )


def train_candidate(module_path: str, total_timesteps: int, seed: int = 0) -> str:
    """
    module_path: dotted import path of the candidate's code, e.g.
                 "eureka.candidates.gen0_cand2" (must define shaping_reward)

    Returns the path to the saved model checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    short_name = module_path.split(".")[-1]

    checkpoint_dir = os.path.join("eureka", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{short_name}.pt")
    components_path = os.path.join(checkpoint_dir, f"{short_name}_components.json")

    # A prior run of this same candidate module (e.g. a re-run, or a stale
    # checkpoints/ directory reused across experiments) may have left a
    # components sidecar on disk. If THIS run doesn't produce fresh
    # component data (e.g. the candidate uses the legacy bare-float
    # shaping_reward contract with no components), the stale sidecar would
    # otherwise still be picked up by loop.py and silently attributed to
    # this run's reflection feedback. Clear it up front so "no sidecar"
    # always means "no component data from this run."
    if os.path.isfile(components_path):
        os.remove(components_path)

    env = make_candidate_vec_env(module_path, n_envs=EUREKA_N_ENVS, seed=seed)
    try:
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

                # Bootstrap value at time-limit truncation. highway-env's
                # 30s "duration" cutoff ends an episode via truncation, not
                # a genuine terminal state (e.g. a collision). Treating a
                # truncated step the same as a true terminal step (as
                # SyncVectorEnv/AsyncVectorEnv's merged `done` flag alone
                # would suggest) tells GAE the episode's future value is
                # exactly zero at that boundary, which is wrong and biases
                # every value target near an episode's natural time limit
                # low - exactly the common case here since most episodes
                # run the full 30s. Standard fix (as in SB3): add
                # gamma * V(terminal_observation) to the reward at the
                # truncated step before it enters the buffer, so GAE's
                # existing terminal-masking logic (which zeroes the
                # bootstrap for `dones[t+1]==1`) produces the correct
                # target without any change to buffer.py itself.
                truncated_mask = np.array(
                    [bool(info.get("TimeLimit.truncated", False)) for info in infos],
                    dtype=bool,
                )
                if truncated_mask.any():
                    terminal_obs = np.stack(
                        [
                            infos[i]["terminal_observation"] if truncated_mask[i] else next_obs[i]
                            for i in range(EUREKA_N_ENVS)
                        ]
                    )
                    terminal_obs_flat = terminal_obs.reshape(EUREKA_N_ENVS, -1)
                    terminal_obs_t = torch.as_tensor(
                        terminal_obs_flat, dtype=torch.float32, device=DEVICE
                    )
                    with torch.no_grad():
                        _, terminal_values = model.forward(terminal_obs_t)
                    terminal_values_np = terminal_values.cpu().numpy()
                    reward = reward + GAMMA * terminal_values_np * truncated_mask

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
                    for key, component_value in components.items():
                        if key not in episode_component_sums:
                            episode_component_sums[key] = np.zeros(
                                EUREKA_N_ENVS, dtype=np.float64
                            )
                            finished_components.setdefault(key, [])
                            component_history.setdefault(key, [])
                        episode_component_sums[key][i] += float(component_value)

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
    finally:
        # Must run even if env.step()/env.reset() raises (e.g. a worker
        # timeout RuntimeError from AsyncVectorEnv). Previously env.close()
        # was only called after the training loop completed successfully,
        # so any mid-training failure - which loop.py explicitly catches
        # and treats as "reject this candidate, continue the search" -
        # left that candidate's worker processes and pipes running for the
        # remainder of the run, accumulating across every subsequent
        # rejected candidate.
        env.close()

    torch.save(model.state_dict(), checkpoint_path)

    if component_history:
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
