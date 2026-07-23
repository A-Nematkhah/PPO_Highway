"""
evaluate_run.py

Convenience CLI: re-evaluate (and optionally render) THE WINNING CANDIDATE
of one specific past `runs/run_NNNN/` experiment - "just show me how run
6's winner drives", without having to dig through eureka_log.json by hand.

--------------------------------------------------------------------------
Why this can't just be `python -m eureka.evaluate_cli <short_name>`
--------------------------------------------------------------------------
Candidate module names (gen0_cand0, gen1_cand3, ...) are reused by EVERY
run. The LIVE, load-bearing locations `eureka/candidates/*.py` and
`eureka/checkpoints/*.pt` only ever reflect the MOST RECENT run - by the
time you're on run_0009, run_0006's `gen0_cand0.py` / `gen0_cand0.pt` at
those live paths have long since been overwritten by run_0007/0008/0009's
own gen0_cand0. Looking a candidate up by name the normal way would
silently evaluate whichever run most recently used that name, not run 6's
actual winner.

This script instead reads the ARCHIVED copies experiment.py wrote inside
runs/run_0006/ itself (reward_candidates/*.py, checkpoints/*.pt, and
pareto_archive.csv to identify the winner) - frozen at exactly what that
run produced - and TEMPORARILY stages the winning candidate's source at
the live dotted-module-path location only for the duration of evaluation
(eureka/sandbox.py's loader has no other way to resolve a module path; see
experiment.py's module docstring for the same constraint). Whatever was at
that live path before is restored in a `finally` block immediately
afterward, so your actual latest/live run state is never left altered.

Usage:
    python -m eureka.evaluate_run 6                      # run_0006's winner, default episodes
    python -m eureka.evaluate_run run_0006 --episodes 100
    python -m eureka.evaluate_run 6 --render             # + save a GIF into runs/run_0006/renders/
    python -m eureka.evaluate_run 6 --render-live         # + open a live window
    python -m eureka.evaluate_run 6 --render-live --render-only   # skip numeric eval, just watch it drive
    python -m eureka.evaluate_run --list                 # which runs exist and who won each
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from eureka.eureka_config import FITNESS_WEIGHTS, N_EVAL_EPISODES
from eureka.evaluate_candidate import evaluate_candidate
from eureka.evaluate_cli import CandidateRef, RenderJob, print_single_result, render_candidate
from eureka.fitness import compute_fitness
from eureka.loop import CANDIDATES_DIR, RUNS_ROOT
from eureka.logging_utils import get_logger

logger = get_logger(__name__)

_RUN_DIR_PATTERN = re.compile(r"^run_(\d{4,})$")


# --------------------------------------------------------------------------- #
# Run resolution
# --------------------------------------------------------------------------- #


def resolve_run_dir(raw: str) -> Path:
    """
    Accepts "6", "06", "0006", "run_0006", a relative path like
    "runs/run_0006", or an absolute path - all resolve to the same
    directory when they refer to the same run.
    """
    raw = raw.strip()
    if raw.isdigit():
        return Path(RUNS_ROOT) / f"run_{int(raw):04d}"
    if _RUN_DIR_PATTERN.match(raw):
        return Path(RUNS_ROOT) / raw
    return Path(raw)


def list_available_runs() -> list[Path]:
    root = Path(RUNS_ROOT)
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and _RUN_DIR_PATTERN.match(p.name)),
        key=lambda p: p.name,
    )


# --------------------------------------------------------------------------- #
# Winner identification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunWinner:
    run_dir: Path
    candidate_id: str
    short_name: str
    module_path: str
    archived_source_path: Path
    archived_checkpoint_path: Path
    metrics: Optional[dict] = None
    fitness: Optional[float] = None


class RunWinnerNotFound(Exception):
    """Raised with a human-readable explanation of exactly what was
    missing/inconsistent about the run, rather than a bare KeyError."""


def _short_name_from_module_path(module_path: str) -> str:
    return module_path.rsplit(".", 1)[-1]


def _find_winner_from_csv(run_dir: Path) -> Optional[tuple[str, str]]:
    """Returns (candidate_id, module_path) of the row marked is_winner in
    pareto_archive.csv, or None if the file is missing/has no winner row."""
    csv_path = run_dir / "pareto_archive.csv"
    if not csv_path.is_file():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_winner", "").strip().lower() == "true":
                return row.get("candidate_id", ""), row.get("module_path", "")
    return None


def _find_winner_from_log(run_dir: Path) -> Optional[tuple[str, str, dict, float]]:
    """
    Fallback for runs made before pareto_archive.csv existed, or if CSV
    export failed for some reason: reads eureka_log.json's final archive
    directly and picks the highest legacy_fitness rank-0 candidate.
    """
    log_path = run_dir / "eureka_log.json"
    if not log_path.is_file():
        return None
    try:
        full_log = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not full_log:
        return None

    archive = full_log[-1].get("final_archive") or []
    front = [c for c in archive if c.get("pareto_rank") == 0] or archive
    if not front:
        return None
    best = max(front, key=lambda c: c.get("legacy_fitness", c.get("fitness", float("-inf"))))
    return (
        best.get("candidate_id", ""),
        str(best.get("module_path", "")),
        best.get("metrics", {}),
        best.get("legacy_fitness", best.get("fitness")),
    )


def find_run_winner(run_dir: Path) -> RunWinner:
    if not run_dir.is_dir():
        raise RunWinnerNotFound(
            f"no such run directory: {run_dir} (use --list to see available runs)"
        )

    metrics: Optional[dict] = None
    fitness: Optional[float] = None

    from_csv = _find_winner_from_csv(run_dir)
    if from_csv:
        candidate_id, module_path = from_csv
    else:
        from_log = _find_winner_from_log(run_dir)
        if not from_log:
            raise RunWinnerNotFound(
                f"{run_dir} has neither a pareto_archive.csv with a winner row nor a "
                "readable eureka_log.json final archive - was this run interrupted "
                "before completing, or is this not a EUREKA run directory at all?"
            )
        candidate_id, module_path, metrics, fitness = from_log

    if not module_path:
        raise RunWinnerNotFound(f"{run_dir}'s winner row has no module_path recorded.")

    short_name = _short_name_from_module_path(module_path)
    archived_source = run_dir / "reward_candidates" / f"{short_name}.py"
    if not archived_source.is_file():
        # final_reward.py is guaranteed to be exactly the winner's code,
        # even if the per-candidate archive copy is missing for some reason.
        fallback = run_dir / "final_reward.py"
        if fallback.is_file():
            archived_source = fallback
        else:
            raise RunWinnerNotFound(
                f"could not find archived source for {short_name} under {run_dir} "
                "(checked reward_candidates/ and final_reward.py) - this run may "
                "predate the experiment archiving feature."
            )

    archived_checkpoint = run_dir / "checkpoints" / f"{short_name}.pt"
    if not archived_checkpoint.is_file():
        raise RunWinnerNotFound(
            f"no archived checkpoint at {archived_checkpoint} - the live "
            f"eureka/checkpoints/{short_name}.pt may still exist, but it could "
            "belong to a LATER run that reused the same candidate name, so it "
            "is deliberately not used as a fallback here."
        )

    return RunWinner(
        run_dir=run_dir,
        candidate_id=candidate_id,
        short_name=short_name,
        module_path=module_path,
        archived_source_path=archived_source,
        archived_checkpoint_path=archived_checkpoint,
        metrics=metrics,
        fitness=fitness,
    )


def print_available_runs() -> None:
    runs = list_available_runs()
    if not runs:
        print(f"No runs found under {RUNS_ROOT}/.")
        return
    print(f"\nAvailable runs under {RUNS_ROOT}/:\n")
    for run_dir in runs:
        try:
            winner = find_run_winner(run_dir)
        except RunWinnerNotFound as e:
            print(f"  {run_dir.name}: (no winner found - {e})")
            continue
        metrics_text = ""
        if winner.metrics:
            metrics_text = (
                f" crash={winner.metrics.get('crash_rate', float('nan')):.2f} "
                f"speed={winner.metrics.get('mean_speed', float('nan')):.1f} "
                f"overtakes={winner.metrics.get('mean_overtakes', float('nan')):.2f}"
            )
        print(f"  {run_dir.name}: winner = {winner.short_name}{metrics_text}")
    print()


# --------------------------------------------------------------------------- #
# Temporary live staging (required for eureka.sandbox's dotted-path loader)
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def stage_live_candidate(short_name: str, archived_source_path: Path) -> Iterator[str]:
    """
    Temporarily copies an archived candidate's source to the LIVE
    eureka/candidates/{short_name}.py location - required because
    eureka.sandbox.load_shaping_reward_from_module_path() (used by both
    evaluate_candidate.py and the rendering path) resolves a dotted module
    path directly to that on-disk location, with no way to point it
    elsewhere without changing sandbox.py itself.

    Whatever was already at the live path (if anything - it may belong to
    the actual most recent run) is backed up in memory and restored in the
    `finally` block, so this is safe to run at any time without disturbing
    your current/latest run's live candidate files.

    Yields the dotted module path to use for evaluate_candidate() calls.
    """
    candidates_dir = Path(CANDIDATES_DIR)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    live_path = candidates_dir / f"{short_name}.py"

    had_previous = live_path.is_file()
    previous_content = live_path.read_bytes() if had_previous else None

    live_path.write_bytes(archived_source_path.read_bytes())
    try:
        yield f"eureka.candidates.{short_name}"
    finally:
        if had_previous:
            live_path.write_bytes(previous_content)
        else:
            live_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eureka.evaluate_run",
        description="Re-evaluate (and optionally render) a specific past run's winning candidate.",
    )
    parser.add_argument(
        "run", nargs="?", default=None,
        help="Run identifier: 6, 0006, run_0006, or a path to a run directory.",
    )
    parser.add_argument("--list", action="store_true", help="List available runs and their winners, then exit.")
    parser.add_argument(
        "--episodes", type=int, default=N_EVAL_EPISODES,
        help=f"Deterministic eval episodes (default: {N_EVAL_EPISODES}).",
    )
    parser.add_argument("--render", action="store_true", help="Save a GIF of one episode into the run's renders/ dir.")
    parser.add_argument(
        "--render-only", action="store_true",
        help="Skip the numeric evaluation entirely and only render. Implies --render.",
    )
    parser.add_argument(
        "--render-live", action="store_true",
        help="Open a live window instead of saving a GIF (needs a local display). Implies --render.",
    )
    parser.add_argument("--render-episodes", type=int, default=1, help="Episodes to render (default: 1).")
    parser.add_argument("--render-fps", type=int, default=5, help="GIF frames per second (default: 5).")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_available_runs()
        return 0

    if not args.run:
        parser.error("a run identifier is required unless --list is given (e.g. `python -m eureka.evaluate_run 6`)")

    render = args.render or args.render_only or args.render_live
    if args.render_only:
        args.render = True
    if args.render_episodes < 1:
        parser.error("--render-episodes must be >= 1")

    run_dir = resolve_run_dir(args.run)
    try:
        winner = find_run_winner(run_dir)
    except RunWinnerNotFound as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Run {winner.run_dir.name}: winner = {winner.short_name} ({winner.module_path})")

    with stage_live_candidate(winner.short_name, winner.archived_source_path) as staged_module_path:
        if not args.render_only:
            print(f"  Evaluating {winner.short_name} ({args.episodes} episodes)...")
            try:
                metrics = evaluate_candidate(
                    str(winner.archived_checkpoint_path), staged_module_path, n_episodes=args.episodes,
                )
            except Exception as e:
                print(f"  FAILED to evaluate {winner.short_name}: {e}")
                return 1
            fitness = compute_fitness(metrics, FITNESS_WEIGHTS)
            print_single_result(winner.short_name, metrics, fitness, args.episodes)
        else:
            print(f"  Skipping numeric evaluation for {winner.short_name} (--render-only).")

        if render:
            render_dir = winner.run_dir / "renders"
            job = RenderJob(
                episodes=args.render_episodes, live=args.render_live,
                render_dir=render_dir, fps=args.render_fps,
            )
            if not job.live:
                render_dir.mkdir(parents=True, exist_ok=True)
            ref = CandidateRef.from_raw(staged_module_path)
            render_candidate(ref, winner.archived_checkpoint_path, job)

    return 0


if __name__ == "__main__":
    sys.exit(main())
