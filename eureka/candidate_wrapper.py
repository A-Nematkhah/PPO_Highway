"""
candidate_wrapper.py

CandidateRewardWrapper: same shape as reward_wrapper.RewardShapingWrapper,
but instead of a hardcoded TTC penalty + overtake bonus, it applies an
arbitrary shaping_fn(ego, road, info) -> float | (float, dict) supplied by
the caller. This is what lets eureka/loop.py plug in a different,
LLM-generated reward function per candidate without touching the wrapper
class itself.

n_overtakes is still computed deterministically here (reusing
reward_wrapper.compute_overtakes) and exposed via info, for two reasons:
    1. candidates can reference info["n_overtakes"] directly instead of
       re-deriving it, reducing the chance of a broken candidate
    2. it's also used, independently of whatever the candidate does with
       it, as one of the ground-truth fitness metrics in
       eureka/evaluate_candidate.py

A candidate that raises an exception, times out, or returns a non-finite/invalid
value degrades to a zero shaping bonus for that step rather than crashing
training - a bad candidate should train into a "does nothing extra" policy
and simply score poorly, not take down the whole run.

Optional reward_components (when the candidate returns a 2-tuple) are stored
in info["shaping_components"] for logging/reflection only — they never alter
the scalar reward used for RL updates.
"""

import gymnasium as gym

from eureka.shaping_call import call_shaping_fn
from reward_wrapper import compute_overtakes


class CandidateRewardWrapper(gym.Wrapper):
    def __init__(self, env, shaping_fn):
        super().__init__(env)
        self.shaping_fn = shaping_fn
        self._prev_relative_x = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_relative_x = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ego = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road

        n_overtakes, self._prev_relative_x = compute_overtakes(road, ego, self._prev_relative_x)
        info["n_overtakes"] = n_overtakes
        info["raw_reward"] = reward

        candidate_info = {
            "crashed": bool(info.get("crashed", False)),
            "speed": info.get("speed", 0.0),
            "raw_reward": reward,
            "n_overtakes": n_overtakes,
        }

        shaping_value, shaping_components = call_shaping_fn(
            self.shaping_fn, ego, road, candidate_info
        )

        shaped_reward = reward + shaping_value
        info["shaping_value"] = shaping_value
        info["shaping_components"] = shaping_components

        return obs, shaped_reward, terminated, truncated, info
