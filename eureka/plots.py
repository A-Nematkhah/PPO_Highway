"""
plots.py

End-of-run visualization for one EUREKA search. Purely a reporting
convenience — never touches selection/archive logic — so the caller
wraps generate_run_plots() in a try/except: a plotting failure must
never abort or corrupt a multi-hour search run.

Produces one PNG with six panels:
    1. best legacy fitness per generation (diagnostic only)
    2. mean crash_rate per generation
    3. mean speed per generation
    4. mean overtakes per generation
    5. archive size + Pareto front size per generation
    6. final archive scatter: crash_rate vs mean_speed (color = overtakes,
       larger markers = Pareto rank 0)

Uses matplotlib's non-interactive "Agg" backend so it works headlessly on
a server with no display attached.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def _generation_summary(full_log: list[dict]) -> dict:
    generations, best_fitness = [], []
    mean_crash, mean_speed, mean_overtakes = [], [], []
    archive_sizes, front_sizes = [], []

    for record in full_log:
        results = record.get("results") or []
        generations.append(record.get("generation"))

        if results:
            best_fitness.append(max(r["fitness"] for r in results))
            mean_crash.append(sum(r["metrics"]["crash_rate"] for r in results) / len(results))
            mean_speed.append(sum(r["metrics"]["mean_speed"] for r in results) / len(results))
            mean_overtakes.append(sum(r["metrics"]["mean_overtakes"] for r in results) / len(results))
        else:
            best_fitness.append(float("nan"))
            mean_crash.append(float("nan"))
            mean_speed.append(float("nan"))
            mean_overtakes.append(float("nan"))

        archive_sizes.append(record.get("archive_size", 0))
        front_sizes.append(record.get("pareto_front_size", 0))

    return {
        "generations": generations,
        "best_fitness": best_fitness,
        "mean_crash": mean_crash,
        "mean_speed": mean_speed,
        "mean_overtakes": mean_overtakes,
        "archive_sizes": archive_sizes,
        "front_sizes": front_sizes,
    }


def generate_run_plots(
    full_log: list[dict],
    pareto_archive: list[dict],
    output_dir: str,
) -> str:
    """
    Renders the summary PNG and returns its path. Raises on failure (e.g.
    matplotlib missing) — callers should wrap this in try/except so a
    plotting problem never aborts the search itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _generation_summary(full_log)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"run_summary_{timestamp}.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    gens = summary["generations"]

    axes[0, 0].plot(gens, summary["best_fitness"], marker="o")
    axes[0, 0].set_title("Best legacy fitness per generation (diagnostic)")
    axes[0, 0].set_xlabel("generation")
    axes[0, 0].set_ylabel("fitness")

    axes[0, 1].plot(gens, summary["mean_crash"], marker="o", color="crimson")
    axes[0, 1].set_title("Mean crash_rate per generation")
    axes[0, 1].set_xlabel("generation")
    axes[0, 1].set_ylabel("crash_rate")

    axes[0, 2].plot(gens, summary["mean_speed"], marker="o", color="seagreen")
    axes[0, 2].set_title("Mean speed per generation")
    axes[0, 2].set_xlabel("generation")
    axes[0, 2].set_ylabel("m/s")

    axes[1, 0].plot(gens, summary["mean_overtakes"], marker="o", color="darkorange")
    axes[1, 0].set_title("Mean overtakes per generation")
    axes[1, 0].set_xlabel("generation")
    axes[1, 0].set_ylabel("per episode")

    axes[1, 1].plot(gens, summary["archive_sizes"], marker="o", label="archive size")
    axes[1, 1].plot(gens, summary["front_sizes"], marker="s", label="pareto front size")
    axes[1, 1].set_title("Archive growth per generation")
    axes[1, 1].set_xlabel("generation")
    axes[1, 1].legend()

    ax = axes[1, 2]
    if pareto_archive:
        crash = [c["metrics"]["crash_rate"] for c in pareto_archive]
        speed = [c["metrics"]["mean_speed"] for c in pareto_archive]
        overtakes = [c["metrics"]["mean_overtakes"] for c in pareto_archive]
        sizes = [80 if c.get("pareto_rank") == 0 else 35 for c in pareto_archive]
        scatter = ax.scatter(crash, speed, c=overtakes, s=sizes, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="mean_overtakes")
        ax.set_title("Final archive: crash_rate vs mean_speed")
        ax.set_xlabel("crash_rate (lower better)")
        ax.set_ylabel("mean_speed (higher better)")
    else:
        ax.set_title("Final archive: crash_rate vs mean_speed")
        ax.text(0.5, 0.5, "no archive data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("EUREKA run summary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
