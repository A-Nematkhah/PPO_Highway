"""
csv_export.py

CSV exports for one EUREKA run's results (Step 7 of the experiment-manager
brief). Consumes exactly the data structures loop.py already builds
(archive entries from eureka.objectives, per-generation full_log records)
- no new metrics, no changes to what gets computed, just three flat views
of the same numbers for spreadsheet/pandas consumption alongside the JSON
log.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]

# Metric keys pulled from each candidate's `metrics` dict. `smoothness` is
# deliberately included here as an OPTIONAL column: no current evaluation
# code computes it (evaluate_candidate.py reports crash_rate / mean_speed /
# mean_overtakes / mean_raw_return / component_means only), and adding a
# new evaluation metric is out of scope for an experiment-infrastructure
# change - see the accompanying report for why this wasn't invented here.
# If a future candidate's metrics dict does contain a "smoothness" key
# (e.g. a shaping-reward component happens to be named that), it will
# simply appear in this column; otherwise the column is written empty.
_METRIC_KEYS = ("crash_rate", "mean_speed", "mean_overtakes", "smoothness", "mean_raw_return")


def _metric_row(metrics: dict) -> dict[str, Any]:
    return {key: metrics.get(key, "") for key in _METRIC_KEYS}


def export_pareto_archive_csv(
    archive: list[dict], path: PathLike, representative_id: Optional[str] = None
) -> Path:
    """
    One row per candidate in the final Pareto archive.

    Columns match the report's Pareto Archive table (Step 5): rank,
    candidate, generation, the metric columns, archive membership, and
    whether this candidate is the run's overall winner.
    """
    path = Path(path)
    fieldnames = [
        "rank", "candidate_id", "module_path", "generation",
        *_METRIC_KEYS, "archive_member", "is_winner",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in archive:
            module_path = str(candidate.get("module_path", ""))
            row = {
                "rank": candidate.get("pareto_rank", ""),
                "candidate_id": candidate.get("candidate_id", ""),
                "module_path": module_path,
                "generation": candidate.get("generation", ""),
                **_metric_row(candidate.get("metrics", {})),
                "archive_member": True,
                "is_winner": candidate.get("candidate_id") == representative_id,
            }
            writer.writerow(row)
    return path


def export_generation_summary_csv(full_log: list[dict], path: PathLike) -> Path:
    """
    One row per generation: how many candidates survived, the Pareto
    front size, archive size at that point, the legacy scalar winner, and
    whether that scalar winner actually made the Pareto front (the same
    "scalar vs. Pareto disagreement" signal loop.py already logs).
    """
    path = Path(path)
    fieldnames = [
        "generation", "n_results", "pareto_front_size", "archive_size",
        "selection_mode", "legacy_scalar_winner_id", "legacy_scalar_winner_on_front",
        "representative_id", "all_candidates_failed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in full_log:
            writer.writerow({
                "generation": record.get("generation", ""),
                "n_results": len(record.get("results") or []),
                "pareto_front_size": record.get("pareto_front_size", ""),
                "archive_size": record.get("archive_size", ""),
                "selection_mode": record.get("selection_mode", ""),
                "legacy_scalar_winner_id": record.get("legacy_scalar_winner_id", ""),
                "legacy_scalar_winner_on_front": record.get("legacy_scalar_winner_on_front", ""),
                "representative_id": record.get("representative_id", ""),
                "all_candidates_failed": record.get("all_candidates_failed", False),
            })
    return path


def export_candidate_metrics_csv(full_log: list[dict], path: PathLike) -> Path:
    """
    One row per EVALUATED candidate across the whole run (not just the
    final archive) - the full population loop.py trained and scored,
    generation by generation. This is the raw material behind both other
    CSVs and behind the report's per-generation tables.
    """
    path = Path(path)
    fieldnames = [
        "generation", "candidate_index", "candidate_id", "module_path", "source",
        *_METRIC_KEYS, "legacy_fitness", "pareto_rank", "crowding_distance",
        "archive_member",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in full_log:
            for result in record.get("results") or []:
                writer.writerow({
                    "generation": result.get("generation", record.get("generation", "")),
                    "candidate_index": result.get("candidate_index", ""),
                    "candidate_id": result.get("candidate_id", ""),
                    "module_path": result.get("module_path", ""),
                    "source": result.get("source", ""),
                    **_metric_row(result.get("metrics", {})),
                    "legacy_fitness": result.get("legacy_fitness", result.get("fitness", "")),
                    "pareto_rank": result.get("pareto_rank", ""),
                    "crowding_distance": result.get("crowding_distance", ""),
                    "archive_member": result.get("archive_member", ""),
                })
    return path


def export_all(
    full_log: list[dict], archive: list[dict], run_dir: PathLike,
    representative_id: Optional[str] = None,
) -> dict[str, Path]:
    """Convenience wrapper: writes all three CSVs into run_dir and returns
    their paths keyed by name, for logging/reporting."""
    run_dir = Path(run_dir)
    return {
        "pareto_archive": export_pareto_archive_csv(
            archive, run_dir / "pareto_archive.csv", representative_id
        ),
        "generation_summary": export_generation_summary_csv(
            full_log, run_dir / "generation_summary.csv"
        ),
        "candidate_metrics": export_candidate_metrics_csv(
            full_log, run_dir / "candidate_metrics.csv"
        ),
    }
