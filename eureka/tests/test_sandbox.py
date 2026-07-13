"""Unit tests for restricted candidate loader (training-time path)."""

import os
import tempfile

import pytest

from eureka.sandbox import (
    exec_shaping_reward,
    load_shaping_reward_from_code,
    load_shaping_reward_from_module_path,
    module_path_to_source_path,
    validate_candidate_ast,
)

_VALID = """
def shaping_reward(ego, road, info):
    return float(info.get("n_overtakes", 0)) * 0.1
"""


def test_load_shaping_reward_from_code_runs_in_restricted_namespace():
    fn = load_shaping_reward_from_code(_VALID)
    assert fn(None, None, {"n_overtakes": 2}) == pytest.approx(0.2)


def test_load_shaping_reward_rejects_import():
    code = "import os\n" + _VALID
    with pytest.raises(ValueError, match="Import"):
        load_shaping_reward_from_code(code)


def test_exec_shaping_reward_rejects_tampered_file_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)
    path = module_path_to_source_path("eureka.candidates.evil")
    with open(path, "w", encoding="utf-8") as f:
        f.write("import os\n" + _VALID)

    with pytest.raises(ValueError, match="Import"):
        load_shaping_reward_from_module_path("eureka.candidates.evil")


def test_validate_candidate_ast_requires_exactly_one_shaping_reward():
    code = "def other(ego, road, info):\n    return 0.0\n"
    passed, message = validate_candidate_ast(code)
    assert passed is False
    assert "shaping_reward" in message
