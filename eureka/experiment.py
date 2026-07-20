"""
experiment.py

Experiment/run management for the EUREKA search: every `python -m
eureka.loop` invocation gets its own numbered, self-contained directory
under runs/ (run_0001, run_0002, ...) so nothing overwrites a previous
run and every artifact from one experiment lives in one place.

Deliberately does NOT change where LIVE, load-bearing files are read from
during a run:
    - candidate reward source (eureka/candidates/genX_candY.py) is loaded
      by eureka.sandbox.load_shaping_reward_from_module_path(), which
      derives its file path directly from the dotted module path
      ("eureka.candidates.gen0_cand0" -> that exact on-disk path). Moving
      the live copy would require changing sandbox.py's path resolution,
      which is explicitly out of scope (sandbox logic must not change).
    - training checkpoints (eureka/checkpoints/*.pt) are written by
      train_candidate.py to a hardcoded directory that confirmation runs,
      evaluate_cli.py, and re-evaluation all also expect.

Instead, this module ARCHIVES copies of both into the run directory
(reward_candidates/, checkpoints/) for a clean, self-contained experiment
record, while the live copies that the running search actually reads from
stay exactly where they've always been. Telemetry and plots, on the other
hand, already accept an explicit output path/directory as a parameter
(see telemetry.Telemetry.__init__ and plots.generate_run_plots), so those
are pointed directly at the run directory with zero changes to either
module.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO

from eureka.run_metadata import RunMetadata, collect_run_metadata

_RUN_DIR_PATTERN = re.compile(r"^run_(\d{4,})$")


def _next_run_number(runs_root: Path) -> int:
    """
    Scans runs_root for existing run_NNNN directories and returns the next
    number. Starts at 1 if none exist. Deliberately tolerant of unrelated
    or malformed directory names under runs_root (e.g. a stray file, or a
    directory that doesn't match the run_NNNN pattern) - those are just
    ignored rather than raising, since they're not this module's concern.
    """
    if not runs_root.is_dir():
        return 1
    numbers = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        match = _RUN_DIR_PATTERN.match(entry.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


class _Tee:
    """Minimal stdout/stderr duplicator: every write goes to both the
    original stream and a log file, so console.log ends up with an exact
    transcript of what the terminal showed - including raw
    sys.stdout.write() calls (e.g. logging_utils.TrainProgressTable) that
    never go through the logging module at all."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._original.isatty()


@dataclass
class ExperimentRun:
    """
    One numbered run directory and everything needed to locate its
    artifacts. Construct via ExperimentRun.start(), not directly, so run
    numbering and directory creation happen together atomically.
    """

    run_id: int
    run_dir: Path

    # --- subdirectories (Step 3's suggested structure) ---
    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def reflection_dir(self) -> Path:
        return self.run_dir / "reflection"

    @property
    def reward_candidates_dir(self) -> Path:
        return self.run_dir / "reward_candidates"

    # --- files ---
    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "eureka_log.json"

    @property
    def telemetry_path(self) -> Path:
        return self.run_dir / "telemetry.jsonl"

    @property
    def pareto_archive_csv_path(self) -> Path:
        return self.run_dir / "pareto_archive.csv"

    @property
    def generation_summary_csv_path(self) -> Path:
        return self.run_dir / "generation_summary.csv"

    @property
    def candidate_metrics_csv_path(self) -> Path:
        return self.run_dir / "candidate_metrics.csv"

    @property
    def report_html_path(self) -> Path:
        return self.run_dir / "report.html"

    @property
    def final_reward_path(self) -> Path:
        return self.run_dir / "final_reward.py"

    @property
    def console_log_path(self) -> Path:
        return self.run_dir / "console.log"

    @property
    def run_name(self) -> str:
        return self.run_dir.name

    # --- construction ---
    @classmethod
    def start(cls, runs_root: str = "runs") -> "ExperimentRun":
        """
        Allocates the next run_NNNN directory and creates its full
        subdirectory tree. Retries on a rare TOCTOU collision (two
        processes starting at the same instant) by trying the next
        number, rather than crashing or silently reusing a directory.
        """
        root = Path(runs_root)
        root.mkdir(parents=True, exist_ok=True)

        for _ in range(1000):
            run_id = _next_run_number(root)
            run_dir = root / f"run_{run_id:04d}"
            try:
                run_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            break
        else:
            raise RuntimeError(f"Could not allocate a new run directory under {root} after 1000 attempts")

        run = cls(run_id=run_id, run_dir=run_dir)
        for subdir in (run.plots_dir, run.checkpoints_dir, run.reflection_dir, run.reward_candidates_dir):
            subdir.mkdir(parents=True, exist_ok=True)
        return run

    # --- config / metadata ---
    def write_config_snapshot(self, config: dict[str, Any]) -> None:
        """Dumps the resolved search configuration (not the whole Python
        module - just the plain-data values relevant to reproducing this
        run) to config.json."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    def write_metadata(self, metadata: RunMetadata, execution_time_s: Optional[float] = None) -> dict:
        """Writes metadata.json and returns the exact enriched payload
        (metadata fields + run_id/run_name/execution_time_s) that was
        written, so callers that also need this data (e.g. the HTML
        report) use the same values rather than reconstructing a
        second, potentially-drifting copy via metadata.to_dict()."""
        payload = metadata.to_dict()
        payload["run_id"] = self.run_id
        payload["run_name"] = self.run_name
        if execution_time_s is not None:
            payload["execution_time_s"] = round(execution_time_s, 3)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    # --- archival copies (live files stay where the algorithm needs them) ---
    def archive_candidate_code(self, short_name: str, code: str) -> Path:
        """Writes one candidate's exact source (already available in-memory
        as `result["code"]`) into reward_candidates/ for this run - purely
        an archival copy; the live copy under eureka/candidates/ that
        training/eval actually load is untouched."""
        path = self.reward_candidates_dir / f"{short_name}.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path

    def archive_final_reward(self, code: str) -> Path:
        with open(self.final_reward_path, "w", encoding="utf-8") as f:
            f.write(code)
        return self.final_reward_path

    def archive_checkpoint(self, source_path: str) -> Optional[Path]:
        """Copies a trained candidate's .pt checkpoint into this run's
        checkpoints/ directory. Best-effort: a missing/unreadable source
        (e.g. a confirmation run whose checkpoint path was already cleaned
        up) logs nothing and returns None rather than aborting the run -
        archival copies are a convenience, never a reason to fail a search
        that has already produced a real result."""
        src = Path(source_path)
        if not src.is_file():
            return None
        dest = self.checkpoints_dir / src.name
        try:
            shutil.copy2(src, dest)
        except OSError:
            return None
        return dest

    def archive_reflection_prompt(self, generation: int, index: int, role: Optional[str], prompt: str) -> Path:
        """Saves one LLM reflection/generation prompt for the report's
        'Reflection' section and for offline inspection."""
        role_part = f"_{role}" if role else ""
        path = self.reflection_dir / f"gen{generation}_req{index}{role_part}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt)
        return path

    # --- console capture ---
    @contextlib.contextmanager
    def capture_console(self) -> Iterator[None]:
        """
        Tees stdout/stderr into console.log for the duration of the run,
        AND forces eureka.logging_utils to rebind its console handler to
        the teed stream - the console handler is normally created once,
        bound to whatever sys.stdout was at the time `setup_logging()`
        first ran (typically at module-import time, before this context
        manager can possibly run), so without the rebind, log lines would
        keep going to the original stdout and never reach console.log.

        Both the plain `print()`/raw `sys.stdout.write()` calls (e.g.
        logging_utils.TrainProgressTable) and everything the structured
        logger writes end up in console.log, matching exactly what a
        person watching the terminal would have seen.
        """
        import eureka.logging_utils as logging_utils

        original_stdout, original_stderr = sys.stdout, sys.stderr
        with open(self.console_log_path, "a", encoding="utf-8") as log_file:
            sys.stdout = _Tee(original_stdout, log_file)
            sys.stderr = _Tee(original_stderr, log_file)
            try:
                # Force the console logging handler to rebind to the new
                # (teed) sys.stdout - see docstring above for why this is
                # necessary rather than merely reassigning sys.stdout.
                logging_utils._configured = False
                logging_utils.setup_logging()
                yield
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                logging_utils._configured = False
                logging_utils.setup_logging()


def build_experiment_config_snapshot() -> dict[str, Any]:
    """
    Collects the plain-data configuration values relevant to reproducing
    a run (Step 4: "Configuration", "Training budget", "Evaluation
    budget", "Random seeds") from eureka_config.py, without importing
    anything that would trigger heavier dependencies (torch/gym) just to
    write a JSON snapshot.
    """
    from eureka import eureka_config as cfg

    return {
        "n_generations": cfg.N_GENERATIONS,
        "k_candidates": cfg.K_CANDIDATES,
        "train_steps_per_candidate": cfg.TRAIN_STEPS_PER_CANDIDATE,
        "eureka_n_envs": cfg.EUREKA_N_ENVS,
        "n_eval_episodes": cfg.N_EVAL_EPISODES,
        "max_concurrent_candidates": cfg.MAX_CONCURRENT_CANDIDATES,
        "multi_objective_mode": cfg.MULTI_OBJECTIVE_MODE,
        "objective_specs": list(cfg.OBJECTIVE_SPECS),
        "pareto_archive_size": cfg.PARETO_ARCHIVE_SIZE,
        "reflection_elites": cfg.REFLECTION_ELITES,
        "confirmation_seeds": list(cfg.CONFIRMATION_SEEDS),
        "legacy_fitness_weights": cfg.FITNESS_WEIGHTS,
        "seed_generation_0_with_human_reward": cfg.SEED_GENERATION_0_WITH_HUMAN_REWARD,
        "screening_second_seed_enabled": cfg.SCREENING_SECOND_SEED_ENABLED,
        "shaping_fn_timeout_s": cfg.SHAPING_FN_TIMEOUT_S,
        "groq_model": cfg.GROQ_MODEL,
        "llm_temperature": cfg.LLM_TEMPERATURE,
    }
