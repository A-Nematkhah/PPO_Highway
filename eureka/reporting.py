"""
reporting.py

Human-readable per-generation / end-of-run console reports, CSV export of
the Pareto archive, and reproducibility-metadata capture.

Deliberately split out of loop.py: loop.py owns *decisions* (selection,
archiving, confirmation); this module only owns *presentation* of
decisions loop.py has already made. It reads the same result/archive dicts
loop.py already builds (candidate_id, metrics, pareto_rank, archive_member,
...) and never mutates them or influences selection - so extracting this
does not change search behavior, only what a human sees about it.

All functions here are pure (str in, str out) or write a single output
file, which makes them trivial to unit-test without mocking the whole
training pipeline.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #

def _num(value, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _pct(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _short_name(module_path) -> str:
    return str(module_path or "?").rsplit(".", 1)[-1]


# --------------------------------------------------------------------------- #
# per-generation report
# --------------------------------------------------------------------------- #

def render_generation_table(generation: int, results: list[dict]) -> str:
    """
    Renders the "Generation N / Pareto Front" table for one generation's
    results (as produced by loop.py's _build_result / annotate_population -
    each entry already carries metrics + pareto_rank).

    "Smoothness" is included as a column because it is frequently requested
    reward-quality diagnostic, but this codebase does not currently compute
    it anywhere (evaluate_candidate.py reports crash_rate / mean_speed /
    mean_overtakes / mean_raw_return only). Rather than inventing a number,
    the column prints "n/a" unless a future evaluate_candidate.py change
    populates metrics["mean_smoothness"] - see the final summary note.
    """
    front = sorted(
        (r for r in results if r.get("pareto_rank") == 0),
        key=lambda r: (
            -float(r.get("crowding_distance", 0.0))
            if r.get("crowding_distance", 0.0) != float("inf")
            else float("-inf")
        ),
    )
    if not front:
        front = sorted(results, key=lambda r: r.get("fitness", 0.0), reverse=True)

    lines = [
        "=" * 80,
        "",
        f"Generation {generation}",
        "",
        "Pareto Front",
        "",
        f"{'Rank':<5}{'Candidate':<20}{'Crash':>8}{'Speed':>9}"
        f"{'Overtakes':>11}{'Smoothness':>12}{'Raw Return':>12}",
    ]
    for idx, result in enumerate(front, start=1):
        metrics = result.get("metrics", {})
        lines.append(
            f"{idx:<5}{_short_name(result.get('module_path')):<20}"
            f"{_pct(metrics.get('crash_rate')):>8}"
            f"{_num(metrics.get('mean_speed'), 1):>9}"
            f"{_num(metrics.get('mean_overtakes'), 2):>11}"
            f"{_num(metrics.get('mean_smoothness'), 2):>12}"
            f"{_num(metrics.get('mean_raw_return'), 2):>12}"
        )
    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def _winner_reason(winner: dict, mode: str) -> tuple[str, str]:
    metrics = winner.get("metrics", {})
    if mode == "pareto":
        reason = (
            "Unweighted knee point of this generation's non-dominated front - "
            "closest candidate to the ideal (0% crash, max speed, max overtakes) "
            "point in normalized objective space, with no cross-objective weighting."
        )
    else:
        reason = (
            "Highest legacy scalar fitness (diagnostic weighted sum of "
            "crash_rate/mean_speed/mean_overtakes) - see eureka/fitness.py."
        )
    tradeoff = (
        f"crash_rate={_pct(metrics.get('crash_rate'))}, "
        f"mean_speed={_num(metrics.get('mean_speed'), 1)} m/s, "
        f"mean_overtakes={_num(metrics.get('mean_overtakes'), 2)}/ep"
    )
    return reason, tradeoff


def render_generation_winner(generation: int, winner: dict | None, mode: str) -> str:
    if winner is None:
        return (
            "Generation Winner\n\n"
            "Candidate: (none - all candidates rejected or failed this generation)\n"
        )
    reason, tradeoff = _winner_reason(winner, mode)
    return (
        "Generation Winner\n\n"
        f"Candidate: {_short_name(winner.get('module_path'))}\n"
        f"Reason: {reason}\n"
        f"Trade-off: {tradeoff}\n"
        + "=" * 80
    )


# --------------------------------------------------------------------------- #
# end-of-run report
# --------------------------------------------------------------------------- #

def render_final_summary(pareto_archive: list[dict], winner: dict | None, mode: str) -> str:
    lines = [
        "",
        "=" * 52,
        "FINAL RESULTS",
        "=" * 52,
        "",
        "Global Pareto Archive",
        "",
        f"{'Rank':<5}{'Candidate':<20}{'Gen':>5}{'Crash':>8}{'Speed':>9}{'Overtakes':>11}{'Smoothness':>12}",
    ]
    ranked = sorted(pareto_archive, key=lambda c: c.get("pareto_rank", 10**9))
    for candidate in ranked:
        metrics = candidate.get("metrics", {})
        lines.append(
            f"{candidate.get('pareto_rank', '?'):<5}"
            f"{_short_name(candidate.get('module_path')):<20}"
            f"{str(candidate.get('generation', '?')):>5}"
            f"{_pct(metrics.get('crash_rate')):>8}"
            f"{_num(metrics.get('mean_speed'), 1):>9}"
            f"{_num(metrics.get('mean_overtakes'), 2):>11}"
            f"{_num(metrics.get('mean_smoothness'), 2):>12}"
        )
    lines.append("")
    if winner is None:
        lines.append("Winner: (none - no candidate survived the full run)")
    else:
        reason, tradeoff = _winner_reason(winner, mode)
        lines.append(f"Winner: {_short_name(winner.get('module_path'))}")
        lines.append(f"Reason: {reason}")
        lines.append(f"Trade-off: {tradeoff}")
    lines.append("")
    lines.append(
        "Note: 'Smoothness' is reported for forward-compatibility but is not "
        "currently computed anywhere in this codebase (evaluate_candidate.py "
        "only reports crash_rate/mean_speed/mean_overtakes/mean_raw_return); "
        "it will read 'n/a' until a smoothness metric (e.g. mean |action "
        "change| or jerk per episode) is added there."
    )
    lines.append("=" * 52)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #

_CSV_FIELDS = (
    "generation", "candidate_id", "module_path", "crash_rate", "mean_speed",
    "mean_overtakes", "mean_smoothness", "mean_raw_return", "pareto_rank",
    "dominated_by", "archive_member", "winner",
)


def export_pareto_archive_csv(path: str, archive: list[dict], winner_id: str | None = None) -> str:
    """
    Writes eureka/results/pareto_archive.csv. Never raises on I/O failure
    to the training loop's control flow - callers should wrap this the same
    way plots.generate_run_plots() is already wrapped in loop.py, since a
    reporting failure must not abort a multi-hour search run.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for candidate in archive:
            metrics = candidate.get("metrics", {})
            candidate_id = candidate.get("candidate_id")
            writer.writerow({
                "generation": candidate.get("generation"),
                "candidate_id": candidate_id,
                "module_path": candidate.get("module_path"),
                "crash_rate": metrics.get("crash_rate"),
                "mean_speed": metrics.get("mean_speed"),
                "mean_overtakes": metrics.get("mean_overtakes"),
                "mean_smoothness": metrics.get("mean_smoothness"),
                "mean_raw_return": metrics.get("mean_raw_return"),
                "pareto_rank": candidate.get("pareto_rank"),
                "dominated_by": "" if candidate.get("pareto_rank") == 0 else "rank>0",
                "archive_member": True,
                "winner": candidate_id == winner_id,
            })
    return path


# --------------------------------------------------------------------------- #
# reproducibility metadata
# --------------------------------------------------------------------------- #

def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def capture_reproducibility_metadata(
    *,
    env_config: dict,
    eureka_config: dict,
    seed: int,
    llm_model: str,
    train_steps_per_candidate: int,
    n_eval_episodes: int,
) -> dict:
    """
    Snapshot of everything needed to reproduce (or at least explain) one
    EUREKA run. Written once per run to eureka/run_metadata.json alongside
    the existing eureka_log.json / eureka_metrics.jsonl.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": seed,
        "llm_model": llm_model,
        "train_steps_per_candidate": train_steps_per_candidate,
        "n_eval_episodes": n_eval_episodes,
        "env_config": env_config,
        "eureka_config": eureka_config,
    }


def write_run_metadata(metadata: dict, path: str = os.path.join("eureka", "run_metadata.json")) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path
