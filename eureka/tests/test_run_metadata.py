"""Unit tests for eureka/run_metadata.py."""

import subprocess

import pytest

from eureka.run_metadata import (
    _get_cpu_model,
    _get_git_branch,
    _get_git_commit_hash,
    _get_git_dirty,
    _get_total_ram_gb,
    collect_run_metadata,
)


def test_collect_run_metadata_returns_expected_types(tmp_path):
    metadata = collect_run_metadata("test-model", repo_root=str(tmp_path))

    assert isinstance(metadata.timestamp_utc, str)
    assert metadata.python_version.count(".") >= 1
    assert metadata.os_name
    assert metadata.llm_model == "test-model"
    assert metadata.cpu_cores_logical is None or metadata.cpu_cores_logical >= 1


def test_collect_run_metadata_on_non_git_directory_degrades_gracefully(tmp_path):
    """A directory that isn't a git checkout must not raise - git fields
    just come back None."""
    metadata = collect_run_metadata("test-model", repo_root=str(tmp_path))
    assert metadata.git_commit is None
    assert metadata.git_branch is None
    assert metadata.git_dirty is None


def test_git_commit_hash_returns_none_when_git_binary_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _get_git_commit_hash() is None
    assert _get_git_branch() is None
    assert _get_git_dirty() is None


def test_git_helpers_return_none_on_subprocess_timeout(monkeypatch):
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _get_git_commit_hash() is None


def test_to_dict_is_json_serializable(tmp_path):
    import json

    metadata = collect_run_metadata("test-model", repo_root=str(tmp_path))
    json.dumps(metadata.to_dict(), default=str)  # must not raise


def test_cpu_model_never_raises():
    # Just confirm it returns str or None across whatever this test
    # machine actually is - the real assertion is "doesn't raise."
    result = _get_cpu_model()
    assert result is None or isinstance(result, str)


def test_total_ram_never_raises():
    result = _get_total_ram_gb()
    assert result is None or (isinstance(result, float) and result > 0)


def test_collect_run_metadata_on_real_repo_finds_git_info():
    """Sanity check against the actual repo checkout this test runs
    from (not tmp_path) - confirms the happy path works when git really
    is available, not just that failures degrade gracefully."""
    metadata = collect_run_metadata("test-model", repo_root=".")
    # This suite only runs inside a real git checkout in CI/dev, but skip
    # gracefully rather than fail if that assumption doesn't hold somewhere.
    if metadata.git_commit is None:
        pytest.skip("not running inside a git checkout")
    assert len(metadata.git_commit) == 40
    assert isinstance(metadata.git_dirty, bool)
