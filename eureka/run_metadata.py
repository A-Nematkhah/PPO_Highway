"""
run_metadata.py

Best-effort collection of system/environment metadata for one EUREKA run,
used by experiment.py to populate each run directory's metadata.json.

Every field here is collected defensively: a missing git binary, an
unreadable /proc/cpuinfo, or any other environment quirk degrades that
one field to None rather than raising and aborting the whole run. This
module has no new hard dependencies - `psutil` is used opportunistically
if already installed (for RAM), with a ctypes/proc-file fallback on
Windows/Linux respectively, and no dependency at all on other platforms.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional


def _get_git_commit_hash(cwd: Optional[str] = None) -> Optional[str]:
    """Returns the current git commit hash, or None if this isn't a git
    checkout, git isn't installed, or the call fails for any reason."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _get_git_branch(cwd: Optional[str] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _get_git_dirty(cwd: Optional[str] = None) -> Optional[bool]:
    """True if there are uncommitted changes - useful for reproducibility:
    a run's commit hash alone doesn't tell you if the working tree matched
    it exactly. None if this isn't a git checkout."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _get_cpu_model() -> Optional[str]:
    """platform.processor() is often an empty string on Linux; fall back
    to reading /proc/cpuinfo's "model name" field there."""
    model = platform.processor()
    if model:
        return model
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _get_total_ram_gb() -> Optional[float]:
    """Best-effort total system RAM in GiB. Tries, in order: psutil (only
    if already installed - never added as a new requirement), then a
    platform-specific zero-dependency fallback (ctypes on Windows,
    /proc/meminfo on Linux), then gives up and returns None."""
    try:
        import psutil  # optional; not in requirements.txt
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass

    if sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024 ** 3), 2)
        except (OSError, AttributeError, ValueError):
            pass
    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    match = re.match(r"MemTotal:\s+(\d+)\s+kB", line)
                    if match:
                        return round(int(match.group(1)) / (1024 ** 2), 2)
        except OSError:
            pass

    return None


@dataclass
class RunMetadata:
    """Everything Step 4 of the experiment-manager brief asks to record
    for one run. Every field defaults to None/unknown gracefully rather
    than failing the run if it can't be collected - metadata is a nice-to-
    have for reproducibility, never a reason to abort an experiment."""

    timestamp_utc: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: Optional[bool]
    python_version: str
    os_name: str
    os_version: str
    cpu_model: Optional[str]
    cpu_cores_logical: Optional[int]
    ram_total_gb: Optional[float]
    llm_model: str
    hostname: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


def collect_run_metadata(llm_model: str, repo_root: Optional[str] = None) -> RunMetadata:
    """
    Collects everything in RunMetadata for the current process/machine.

    repo_root: directory to run git commands from (defaults to the current
    working directory, which is what `python -m eureka.loop` is normally
    invoked from).
    """
    from datetime import datetime, timezone

    try:
        hostname = platform.node() or None
    except OSError:
        hostname = None

    return RunMetadata(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_get_git_commit_hash(repo_root),
        git_branch=_get_git_branch(repo_root),
        git_dirty=_get_git_dirty(repo_root),
        python_version=sys.version.split()[0],
        os_name=platform.system(),
        os_version=platform.version(),
        cpu_model=_get_cpu_model(),
        cpu_cores_logical=os.cpu_count(),
        ram_total_gb=_get_total_ram_gb(),
        llm_model=llm_model,
        hostname=hostname,
    )
