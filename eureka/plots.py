"""
plots.py

End-of-run visualization for one EUREKA search. Purely a reporting
convenience - never touches selection/archive logic - so the caller wraps
generate_run_plots() in a try/except: a plotting failure must never abort
or corrupt a multi-hour search run.

Reads the SAME result/archive dicts loop.py already produces (metrics,
pareto_rank, archive_member, generation, ...) and writes several
publication-quality PNGs to `output_dir` (default eureka/results/):

    1. pareto_crash_vs_speed.png   - crash_rate vs mean_speed, Pareto/winner highlighted
    2. speed_vs_overtakes.png      - mean_speed vs mean_overtakes, Pareto highlighted
    3. generation_progress.png     - best crash/speed/overtakes per generation
    4. archive_evolution.png       - archive size + front size per generation
    5. all_candidates_scatter.png  - every evaluated candidate: dominated vs
                                      Pareto vs winner, distinct colors/markers
    6. final_archive_bar.png       - grouped bar chart of final archive metrics

generate_run_plots() returns a dict {plot_name: path}. Uses matplotlib's
non-interactive "Agg" backend so it works headlessly on a server with no
display attached.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

_STYLE = {
    "dominated": dict(c="#b0b0b0", marker="x", s=30, alpha=0.6, label="dominated"),
    "pareto": dict(c="#2b6cb0", marker="o", s=60, alpha=0.85, label="Pareto front"),
    "winner": dict(c="#e53e3e", marker="*", s=260, alpha=1.0, label="winner"),
}


def _all_candidate_rows(full_log: list[dict]) -> list[dict]:
    rows = []
    for record in full_log:
        rows.extend(record.get("results") or [])
    return rows


def _generation_summary(full_log: list[dict]) -> dict:
    generations, best_crash, best_speed, best_overtakes = [], [], [], []
    archive_sizes, front_sizes = [], []

    for record in full_log:
        results = record.get("results") or []
        generations.append(record.get("generation"))

        if results:
            best_crash.append(min(r["metrics"]["crash_rate"] for r in results))
            best_speed.append(max(r["metrics"]["mean_speed"] for r in results))
            best_overtakes.append(max(r["metrics"]["mean_overtakes"] for r in results))
        else:
            best_crash.append(float("nan"))
            best_speed.append(float("nan"))
            best_overtakes.append(float("nan"))

        archive_sizes.append(record.get("archive_size", 0))
        front_sizes.append(record.get("pareto_front_size", 0))

    return {
        "generations": generations,
        "best_crash": best_crash,
        "best_speed": best_speed,
        "best_overtakes": best_overtakes,
        "archive_sizes": archive_sizes,
        "front_sizes": front_sizes,
    }


def _classify(row: dict, winner_id: str | None) -> str:
    if winner_id is not None and row.get("candidate_id") == winner_id:
        return "winner"
    if row.get("pareto_rank") == 0 or row.get("archive_member"):
        return "pareto"
    return "dominated"


def _scatter_by_class(ax, rows: list[dict], x_key: str, y_key: str, winner_id: str | None) -> None:
    buckets: dict[str, list[dict]] = {"dominated": [], "pareto": [], "winner": []}
    for row in rows:
        buckets[_classify(row, winner_id)].append(row)

    # draw dominated first so Pareto/winner markers sit on top
    for label in ("dominated", "pareto", "winner"):
        group = buckets[label]
        if not group:
            continue
        xs = [row["metrics"][x_key] for row in group]
        ys = [row["metrics"][y_key] for row in group]
        ax.scatter(xs, ys, **_STYLE[label])


def _plot_pareto_2d(rows, x_key, y_key, x_label, y_label, title, winner_id, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _scatter_by_class(ax, rows, x_key, y_key, winner_id)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_generation_progress(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    gens = summary["generations"]

    axes[0].plot(gens, summary["best_crash"], marker="o", color="#c53030")
    axes[0].set_title("Best crash_rate per generation (lower better)")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("crash_rate")

    axes[1].plot(gens, summary["best_speed"], marker="o", color="#2f855a")
    axes[1].set_title("Best mean_speed per generation")
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("m/s")

    axes[2].plot(gens, summary["best_overtakes"], marker="o", color="#b7791f")
    axes[2].set_title("Best mean_overtakes per generation")
    axes[2].set_xlabel("generation")
    axes[2].set_ylabel("per episode")

    for ax in axes:
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Generation progress (per-generation best, not archive-wide)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_archive_evolution(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    gens = summary["generations"]
    ax.plot(gens, summary["archive_sizes"], marker="o", label="archive size", color="#2b6cb0")
    ax.plot(gens, summary["front_sizes"], marker="s", label="Pareto front size (rank 0)", color="#c53030")
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.set_title("Archive evolution across generations")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_final_archive_bar(pareto_archive, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not pareto_archive:
        return None

    ranked = sorted(pareto_archive, key=lambda c: c.get("pareto_rank", 10**9))
    names = [str(c.get("module_path", "?")).rsplit(".", 1)[-1] for c in ranked]
    crash = [c["metrics"]["crash_rate"] for c in ranked]
    speed_norm = [c["metrics"]["mean_speed"] / 40.0 for c in ranked]  # normalized vs OBJECTIVE_SPECS bound
    overtakes_norm = [c["metrics"]["mean_overtakes"] / 10.0 for c in ranked]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    ax.bar(x - width, crash, width, label="crash_rate", color="#c53030")
    ax.bar(x, speed_norm, width, label="mean_speed / 40", color="#2f855a")
    ax.bar(x + width, overtakes_norm, width, label="mean_overtakes / 10", color="#b7791f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("normalized value")
    ax.set_title("Final Pareto archive: metric comparison (normalized for shared axis)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def generate_run_plots(
    full_log: list[dict],
    pareto_archive: list[dict],
    output_dir: str,
    winner_id: str | None = None,
) -> dict:
    """
    Renders all publication-quality plots and returns {name: path}.
    Raises on failure (e.g. matplotlib missing) - callers should wrap this
    in try/except so a plotting problem never aborts the search itself.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_rows = _all_candidate_rows(full_log)
    summary = _generation_summary(full_log)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    paths = {}

    paths["pareto_crash_vs_speed"] = _plot_pareto_2d(
        all_rows, "crash_rate", "mean_speed",
        "crash_rate (lower better)", "mean_speed (m/s, higher better)",
        "Pareto front: crash rate vs speed", winner_id,
        os.path.join(output_dir, f"pareto_crash_vs_speed_{timestamp}.png"),
    )
    paths["speed_vs_overtakes"] = _plot_pareto_2d(
        all_rows, "mean_speed", "mean_overtakes",
        "mean_speed (m/s)", "mean_overtakes (per episode, higher better)",
        "Pareto front: speed vs overtakes", winner_id,
        os.path.join(output_dir, f"speed_vs_overtakes_{timestamp}.png"),
    )
    paths["generation_progress"] = _plot_generation_progress(
        summary, os.path.join(output_dir, f"generation_progress_{timestamp}.png"),
    )
    paths["archive_evolution"] = _plot_archive_evolution(
        summary, os.path.join(output_dir, f"archive_evolution_{timestamp}.png"),
    )
    paths["all_candidates_scatter"] = _plot_pareto_2d(
        all_rows, "crash_rate", "mean_overtakes",
        "crash_rate (lower better)", "mean_overtakes (per episode)",
        "All evaluated candidates: dominated vs Pareto vs winner", winner_id,
        os.path.join(output_dir, f"all_candidates_scatter_{timestamp}.png"),
    )
    bar_path = _plot_final_archive_bar(
        pareto_archive, os.path.join(output_dir, f"final_archive_bar_{timestamp}.png"),
    )
    if bar_path:
        paths["final_archive_bar"] = bar_path

    return paths