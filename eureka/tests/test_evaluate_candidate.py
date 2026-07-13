"""Unit tests for eureka/evaluate_candidate.py with mocked env/model."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from eureka.evaluate_candidate import evaluate_candidate


def _make_mock_env(n_episode_steps: int = 3):
    env = MagicMock()
    obs = np.zeros(75, dtype=np.float32)

    step_count = {"n": 0}

    def reset(seed=None):
        step_count["n"] = 0
        return obs.copy(), {}

    def step(action):
        step_count["n"] += 1
        done = step_count["n"] >= n_episode_steps
        info = {
            "speed": 20.0 + step_count["n"],
            "n_overtakes": 1,
            "raw_reward": 0.5,
            "crashed": done and step_count["n"] == n_episode_steps,
            "shaping_components": {},
        }
        return obs.copy(), 1.0, done, False, info

    env.reset.side_effect = reset
    env.step.side_effect = step
    env.observation_space.shape = obs.shape
    env.action_space.n = 5
    env.unwrapped = MagicMock()
    return env


@patch("eureka.evaluate_candidate.CandidateRewardWrapper")
@patch("eureka.evaluate_candidate.load_shaping_reward_from_module_path")
@patch("eureka.evaluate_candidate.gym.make")
@patch("eureka.evaluate_candidate.torch.load")
@patch("eureka.evaluate_candidate.ActorCritic")
def test_evaluate_candidate_computes_averages(
    mock_ac_cls, mock_load, mock_gym_make, mock_load_fn, mock_wrapper_cls,
):
    env = _make_mock_env(n_episode_steps=2)
    mock_gym_make.return_value = env
    mock_wrapper_cls.side_effect = lambda e, fn: e
    mock_load_fn.return_value = lambda ego, road, info: 0.0
    mock_load.return_value = {}

    class _FakeModel:
        def load_state_dict(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

        def forward(self, obs):
            return torch.zeros(1, 5), torch.zeros(1)

    mock_ac_cls.return_value = _FakeModel()

    metrics = evaluate_candidate("fake.pt", "eureka.candidates.test", n_episodes=2)

    assert metrics["crash_rate"] == pytest.approx(1.0)
    assert metrics["mean_speed"] == pytest.approx((21.0 + 22.0) / 2)
    assert metrics["mean_overtakes"] == pytest.approx(2.0)
    assert metrics["mean_raw_return"] == pytest.approx(1.0)
    assert "component_means" in metrics
    assert isinstance(metrics["component_means"], dict)
    assert mock_load.call_args.kwargs.get("weights_only") is True
