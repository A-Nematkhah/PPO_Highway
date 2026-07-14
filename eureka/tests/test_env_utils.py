"""Unit tests for AsyncVectorEnv worker failure handling (no live highway env)."""

import pytest

from env_utils import AsyncVectorEnv


class _FailingEnvFactory:
    """Picklable factory that simulates import_module / env construction failure."""

    module_path = "eureka.candidates.missing"
    seed = 7

    def __call__(self):
        raise ImportError("simulated candidate import failure")


def test_async_vector_env_raises_on_worker_init_failure():
    with pytest.raises(RuntimeError, match="module_path='eureka.candidates.missing'"):
        AsyncVectorEnv([_FailingEnvFactory()])
