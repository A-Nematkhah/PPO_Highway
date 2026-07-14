"""
env_factory.py

Builds a vectorized env for one specific reward candidate.

Same Windows-multiprocessing constraint as env_utils._EnvFactory: a closure
capturing the candidate's shaping_fn directly would not be picklable for
the "spawn" start method. So instead we pass the candidate's MODULE PATH
(a plain string, trivially picklable) and load it fresh inside each
worker process. loop.py writes each candidate's code to eureka/candidates/
so workers can read the source file from disk.

Training-time load uses eureka.sandbox (AST allowlist + restricted exec) —
the same path as smoke_test.py — NOT importlib.import_module. Workers still
run inside the parent OS process (no container yet); see docs/SECURITY.md.
"""

import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-fast-v0)

from config import ENV_CONFIG, ENV_ID
from env_utils import AsyncVectorEnv, SyncVectorEnv
from eureka.candidate_wrapper import CandidateRewardWrapper
from eureka.sandbox import load_shaping_reward_from_module_path


class _CandidateEnvFactory:
    def __init__(self, seed: int, module_path: str):
        self.seed = seed
        self.module_path = module_path

    def __call__(self):
        shaping_fn = load_shaping_reward_from_module_path(self.module_path)

        env = gym.make(ENV_ID)
        env.unwrapped.configure(ENV_CONFIG)
        env = CandidateRewardWrapper(env, shaping_fn)
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return env


def make_candidate_vec_env(module_path: str, n_envs: int, seed: int, parallel: bool = True):
    env_fns = [_CandidateEnvFactory(seed + i, module_path) for i in range(n_envs)]
    if parallel:
        return AsyncVectorEnv(env_fns)
    return SyncVectorEnv(env_fns)
