"""
env_utils.py

Responsible for:
    - creating a single highway-fast-v0 environment
    - seeding
    - a simple synchronous vectorized wrapper (no external deps beyond gymnasium)

Run this file directly to inspect observation/action space shapes:
    python env_utils.py
"""

import traceback

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-fast-v0)

from config import (
    ENV_CONFIG,
    ENV_ID,
    GROQ_MODEL,
    LLM_JUDGE_EVERY_N_EPISODES,
    LLM_JUDGE_WEIGHT,
    N_ENVS,
    OVERTAKE_BONUS,
    SEED,
    TTC_THRESHOLD,
    TTC_WEIGHT,
    USE_LLM_JUDGE,
)
from reward_wrapper import RewardShapingWrapper

# Blocking recv() on a dead/hung worker used to hang the entire EUREKA loop with
# no error output if env_fn() failed during worker startup. Poll before recv.
WORKER_INIT_TIMEOUT_S = 30.0
WORKER_OP_TIMEOUT_S = 60.0


def _env_fn_label(env_fn) -> str:
    """Human-readable context for worker failure messages (module path, seed, ...)."""
    parts = []
    module_path = getattr(env_fn, "module_path", None)
    if module_path is not None:
        parts.append(f"module_path={module_path!r}")
    seed = getattr(env_fn, "seed", None)
    if seed is not None:
        parts.append(f"seed={seed}")
    return ", ".join(parts) if parts else repr(env_fn)


def _recv_with_timeout(remote, timeout_s: float, context: str = ""):
    if not remote.poll(timeout_s):
        suffix = f" ({context})" if context else ""
        raise RuntimeError(f"Worker timed out after {timeout_s}s{suffix}")

    message = remote.recv()
    if isinstance(message, tuple) and len(message) == 2:
        tag, payload = message
        if tag == "error":
            suffix = f" ({context})" if context else ""
            raise RuntimeError(f"Worker failed{suffix}:\n{payload}")
        if tag == "ok":
            return payload
    return message


class _EnvFactory:
    """
    Picklable zero-arg callable that builds and seeds a single env.

    A plain nested function (closure) would NOT be picklable, which breaks
    multiprocessing's "spawn" start method (required on Windows). A class
    with __call__ defined at module level works fine because it can be
    pickled by reference + its (picklable) __init__ arguments.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self):
        env = gym.make(ENV_ID)
        env.unwrapped.configure(ENV_CONFIG)

        llm_judge_fn = None
        if USE_LLM_JUDGE:
            # imported lazily so the `groq` package is only required when
            # this feature is actually turned on
            from functools import partial
            from llm_judge import judge_episode
            llm_judge_fn = partial(judge_episode, model=GROQ_MODEL)

        env = RewardShapingWrapper(
            env,
            ttc_threshold=TTC_THRESHOLD,
            ttc_weight=TTC_WEIGHT,
            overtake_bonus=OVERTAKE_BONUS,
            llm_judge_fn=llm_judge_fn,
            llm_judge_weight=LLM_JUDGE_WEIGHT,
            llm_judge_every_n_episodes=LLM_JUDGE_EVERY_N_EPISODES,
        )
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return env


def make_env(seed: int):
    """
    Returns a picklable zero-arg callable that builds and seeds a single env.
    Using a callable (rather than the env itself) lets us create envs lazily
    in worker processes for AsyncVectorEnv.
    """
    return _EnvFactory(seed)


class SyncVectorEnv:
    """
    Minimal synchronous vectorized environment.

    Runs N_ENVS copies of the env in a simple Python loop (no multiprocessing).
    Auto-resets any env that terminates/truncates, matching the convention
    used by gymnasium.vector and Stable-Baselines3 (so rollouts never contain
    "dead" environments waiting to be reset manually).
    """

    def __init__(self, env_fns):
        self.envs = []
        for env_fn in env_fns:
            try:
                self.envs.append(env_fn())
            except Exception as e:
                label = _env_fn_label(env_fn)
                raise RuntimeError(f"Failed to initialize env for {label}: {e}") from e
        self.n = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        """
        actions: array-like of length n (one discrete action per env)

        Returns:
            obs:     (n, *obs_shape)
            rewards: (n,)
            dones:   (n,)  -- True if that env's episode ended this step
            infos:   list[dict] of length n
        """
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if done:
                # keep the true terminal observation available for anyone who
                # wants to bootstrap value estimates correctly, then reset
                info["terminal_observation"] = obs
                obs, _ = env.reset()

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=np.bool_),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()


def _worker(remote, parent_remote, env_fn):
    """
    Runs in a separate process. Owns one env instance and services commands
    sent over a pipe from the main process. This is the same pattern used
    by Stable-Baselines3's SubprocVecEnv.
    """
    parent_remote.close()
    try:
        env = env_fn()
    except Exception:
        remote.send(("error", traceback.format_exc()))
        remote.close()
        return

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(int(data))
                done = terminated or truncated
                if done:
                    info["terminal_observation"] = obs
                    obs, _ = env.reset()
                remote.send(("ok", (obs, reward, done, info)))

            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(("ok", obs))

            elif cmd == "close":
                env.close()
                remote.close()
                break

            elif cmd == "spaces":
                remote.send(("ok", (env.observation_space, env.action_space)))

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception:
        remote.send(("error", traceback.format_exc()))
        remote.close()


class AsyncVectorEnv:
    """
    True multiprocessing vector env: each sub-env runs in its own OS process,
    so N_ENVS envs actually step in parallel across CPU cores instead of
    being looped over sequentially in one process.

    API-compatible with SyncVectorEnv (reset / step / close), so it's a
    drop-in replacement in train.py.
    """

    def __init__(self, env_fns):
        import multiprocessing as mp

        self.n = len(env_fns)
        self._env_labels = [_env_fn_label(env_fn) for env_fn in env_fns]
        ctx = mp.get_context("spawn")  # required for reliability on Windows

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = ctx.Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        for worker_idx, remote in enumerate(self.remotes):
            remote.send(("spaces", None))
            context = f"worker {worker_idx} initializing ({self._env_labels[worker_idx]})"
            spaces = _recv_with_timeout(remote, WORKER_INIT_TIMEOUT_S, context)
            if worker_idx == 0:
                self.observation_space, self.action_space = spaces

    def reset(self):
        for worker_idx, remote in enumerate(self.remotes):
            remote.send(("reset", None))
        obs = []
        for worker_idx, remote in enumerate(self.remotes):
            context = f"worker {worker_idx} reset ({self._env_labels[worker_idx]})"
            obs.append(_recv_with_timeout(remote, WORKER_OP_TIMEOUT_S, context))
        return np.stack(obs)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        results = []
        for worker_idx, remote in enumerate(self.remotes):
            context = f"worker {worker_idx} step ({self._env_labels[worker_idx]})"
            results.append(_recv_with_timeout(remote, WORKER_OP_TIMEOUT_S, context))
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            list(infos),
        )

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()


def make_vec_env(n_envs: int = N_ENVS, seed: int = SEED, parallel: bool = True):
    """
    parallel=True  -> AsyncVectorEnv (multiprocessing, faster on multi-core CPUs)
    parallel=False -> SyncVectorEnv (single process, easier to debug)
    """
    env_fns = [make_env(seed + i) for i in range(n_envs)]
    if parallel:
        return AsyncVectorEnv(env_fns)
    return SyncVectorEnv(env_fns)


if __name__ == "__main__":
    # quick inspection script - run this to see obs/action space before
    # designing the network input size
    env = gym.make(ENV_ID)
    env.unwrapped.configure(ENV_CONFIG)
    obs, info = env.reset()

    print("observation_space:", env.observation_space)
    print("obs.shape:", obs.shape)
    print("obs.dtype:", obs.dtype)
    print("action_space:", env.action_space)
    print()
    print("sample obs:")
    print(obs)

    env.close()
