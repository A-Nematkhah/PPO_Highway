"""
env_factory.py

Builds a vectorized env for one specific reward candidate.

Same Windows-multiprocessing constraint as env_utils._EnvFactory: a closure
capturing the candidate's shaping_fn directly would not be picklable for
the "spawn" start method. So instead we pass the candidate's MODULE PATH
(a plain string, trivially picklable) and re-import it fresh inside each
worker process via importlib. This is exactly why loop.py writes each
candidate's code to an actual .py file under eureka/candidates/ instead of
keeping it as an in-memory string.
"""

import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-fast-v0)

from config import ENV_CONFIG, ENV_ID
from env_utils import AsyncVectorEnv, SyncVectorEnv
from eureka.candidate_wrapper import CandidateRewardWrapper


class _CandidateEnvFactory:
    def __init__(self, seed: int, module_path: str):
        self.seed = seed
        self.module_path = module_path

    def __call__(self):
        import importlib

        # TODO(security): training-time import executes candidate module code with
        # full worker-process privileges (no AST gate, no restricted builtins).
        # smoke_test.py validates candidates in a subprocess before write, but
        # a malicious or compromised .py file on disk would still run unrestricted
        # here. Longer-term: load candidates in an isolated container/subprocess
        # with no filesystem/network access, or use a declarative reward DSL
        # instead of arbitrary Python import.
        module = importlib.import_module(self.module_path)
        shaping_fn = module.shaping_reward

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
