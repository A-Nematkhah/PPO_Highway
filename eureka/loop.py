"""
loop.py

The main EUREKA orchestrator. For each generation:
    1. ask the LLM for K reward-shaping code candidates (informed by the
       best candidate + its metrics from the previous generation)
    2. smoke-test each candidate; reject anything that fails
    3. train each surviving candidate for a short budget
    4. run a deterministic evaluation to get objective metrics
    5. rank objective trade-offs with epsilon/NSGA-II-lite selection
    6. maintain a bounded Pareto archive across ALL generations
    7. log everything to eureka/eureka_log.json

Run with:
    python -m eureka.loop

Requires: GROQ_API_KEY environment variable set (see llm_reward_designer.py)
"""

import json
import math
import os
import time

from eureka.eureka_config import (
    CONFIRMATION_SEEDS,
    EUREKA_N_ENVS,
    FITNESS_WEIGHTS,
    GROQ_MODEL,
    K_CANDIDATES,
    LLM_TEMPERATURE,
    MULTI_OBJECTIVE_MODE,
    N_EVAL_EPISODES,
    N_GENERATIONS,
    OBJECTIVE_SPECS,
    PARETO_ARCHIVE_SIZE,
    REFLECTION_ELITES,
    TRAIN_STEPS_PER_CANDIDATE,
    candidate_base_seed,
)
from eureka.evaluate_candidate import evaluate_candidate
from eureka.fitness import compute_fitness
from eureka.llm_reward_designer import generate_candidates
from eureka.logging_utils import get_logger
from eureka.objectives import (
    annotate_population,
    candidate_id,
    select_reflection_elites,
    select_representative,
    update_archive,
)
from eureka.smoke_test import smoke_test
from eureka.telemetry import Telemetry
from eureka.train_candidate import train_candidate

logger = get_logger(__name__)

CANDIDATES_DIR = os.path.join("eureka", "candidates")
LOG_PATH = os.path.join("eureka", "eureka_log.json")


def _log_banner():
    logger.info(
        "EUREKA reward search starting",
        extra={
            "event": "run_start",
            "generations": N_GENERATIONS,
            "candidates_per_gen": K_CANDIDATES,
            "train_steps_per_candidate": TRAIN_STEPS_PER_CANDIDATE,
            "parallel_envs": EUREKA_N_ENVS,
            "eval_episodes": N_EVAL_EPISODES,
            "llm_model": GROQ_MODEL,
            "multi_objective_mode": MULTI_OBJECTIVE_MODE,
            "objective_specs": OBJECTIVE_SPECS,
            "pareto_archive_size": PARETO_ARCHIVE_SIZE,
            "legacy_fitness_weights": FITNESS_WEIGHTS,
            "log_path": LOG_PATH,
        },
    )


def _aggregate_metrics(runs: list[dict]) -> dict:
    """Mean objective metrics across independent confirmation runs."""
    keys = ("crash_rate", "mean_speed", "mean_overtakes", "mean_raw_return")
    return {
        key: sum(float(run[key]) for run in runs) / len(runs)
        for key in keys
    }


def _json_safe(value):
    """Convert non-finite NSGA-II boundary distances to strict JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _confirm_archive(
    archive: list[dict],
    telemetry: Telemetry,
) -> list[dict]:
    """
    Optionally retrain rank-zero finalists on independent seeds. Disabled by
    default because confirmation adds one full train/eval per configured seed.
    """
    if not CONFIRMATION_SEEDS:
        return archive

    confirmed = []
    for candidate in archive:
        if candidate.get("pareto_rank") != 0:
            confirmed.append(candidate)
            continue

        runs = [candidate["metrics"]]
        confirmation_records = []
        for seed in CONFIRMATION_SEEDS:
            try:
                with telemetry.timed(
                    "confirmation",
                    candidate_id=candidate_id(candidate),
                    module_path=candidate["module_path"],
                    seed=seed,
                ) as context:
                    checkpoint = train_candidate(
                        candidate["module_path"],
                        total_timesteps=TRAIN_STEPS_PER_CANDIDATE,
                        seed=seed,
                    )
                    metrics = evaluate_candidate(
                        checkpoint,
                        candidate["module_path"],
                        n_episodes=N_EVAL_EPISODES,
                    )
                    context.update(metrics)
                    runs.append(metrics)
                    confirmation_records.append(
                        {"seed": seed, "checkpoint": checkpoint, "metrics": metrics}
                    )
            except Exception as error:
                logger.warning(
                    "candidate confirmation failed",
                    extra={
                        "event": "confirmation_failed",
                        "candidate_id": candidate_id(candidate),
                        "seed": seed,
                        "reason": str(error),
                    },
                )

        item = dict(candidate)
        item["screening_metrics"] = candidate["metrics"]
        item["confirmation_runs"] = confirmation_records
        item["metrics"] = _aggregate_metrics(runs)
        confirmed.append(item)

    return update_archive([], confirmed, OBJECTIVE_SPECS, PARETO_ARCHIVE_SIZE)


def main():
    if MULTI_OBJECTIVE_MODE not in {"shadow", "pareto"}:
        raise ValueError(
            "MULTI_OBJECTIVE_MODE must be 'shadow' or 'pareto', "
            f"got {MULTI_OBJECTIVE_MODE!r}"
        )
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    telemetry = Telemetry()
    _log_banner()

    best = None  # legacy scalar winner, used only while mode == "shadow"
    pareto_archive = []
    full_log = []
    run_start = time.perf_counter()

    for generation in range(N_GENERATIONS):
        gen_start = time.perf_counter()
        logger.info("generation started", extra={"event": "generation_start", "generation": generation})

        logger.info(
            "requesting LLM candidates",
            extra={"event": "llm_request", "generation": generation, "k": K_CANDIDATES},
        )
        if MULTI_OBJECTIVE_MODE == "pareto":
            reflection_context = select_reflection_elites(
                pareto_archive, OBJECTIVE_SPECS, REFLECTION_ELITES
            )
        else:
            reflection_context = best
        with telemetry.timed("llm_generation", generation=generation, k=K_CANDIDATES) as llm_ctx:
            candidates_code = generate_candidates(
                reflection_context, k=K_CANDIDATES, generation=generation,
                model=GROQ_MODEL, temperature=LLM_TEMPERATURE,
            )
            llm_ctx["n_received"] = len(candidates_code)

        if not candidates_code:
            logger.warning(
                "no candidates returned from LLM",
                extra={"event": "llm_empty", "generation": generation},
            )
            full_log.append({
                "generation": generation,
                "results": [],
                "pareto_front_size": sum(
                    item.get("pareto_rank") == 0 for item in pareto_archive
                ),
                "archive_size": len(pareto_archive),
                "selection_mode": MULTI_OBJECTIVE_MODE,
            })
            telemetry.record(
                "generation_complete",
                generation=generation,
                duration_s=round(time.perf_counter() - gen_start, 4),
                n_results=0,
            )
            continue

        generation_results = []

        for k, code in enumerate(candidates_code):
            cand_start = time.perf_counter()
            logger.info(
                "candidate started",
                extra={"event": "candidate_start", "generation": generation, "candidate": k},
            )

            with telemetry.timed("smoke_test", generation=generation, candidate=k) as smoke_ctx:
                passed, message = smoke_test(code)
                smoke_ctx["passed"] = passed
                if not passed:
                    smoke_ctx["reason"] = message

            if not passed:
                logger.warning(
                    "candidate rejected by smoke test",
                    extra={
                        "event": "candidate_rejected",
                        "generation": generation,
                        "candidate": k,
                        "stage": "smoke_test",
                        "reason": message,
                    },
                )
                continue

            module_name = f"gen{generation}_cand{k}"
            module_path = f"eureka.candidates.{module_name}"
            file_path = os.path.join(CANDIDATES_DIR, f"{module_name}.py")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(
                "candidate code saved",
                extra={
                    "event": "candidate_saved",
                    "generation": generation,
                    "candidate": k,
                    "path": file_path,
                },
            )

            checkpoint_path = None
            try:
                with telemetry.timed("train", generation=generation, candidate=k, module_path=module_path) as train_ctx:
                    train_ctx["total_timesteps"] = TRAIN_STEPS_PER_CANDIDATE
                    checkpoint_path = train_candidate(
                        module_path,
                        total_timesteps=TRAIN_STEPS_PER_CANDIDATE,
                        seed=candidate_base_seed(generation, k),
                    )
                    train_ctx["checkpoint"] = checkpoint_path
            except RuntimeError as e:
                logger.warning(
                    "candidate rejected during training",
                    extra={
                        "event": "candidate_rejected",
                        "generation": generation,
                        "candidate": k,
                        "stage": "train",
                        "reason": str(e),
                    },
                )
                continue

            try:
                with telemetry.timed("eval", generation=generation, candidate=k, module_path=module_path) as eval_ctx:
                    metrics = evaluate_candidate(
                        checkpoint_path, module_path, n_episodes=N_EVAL_EPISODES
                    )
                    eval_ctx.update(metrics)
            except Exception as e:
                logger.warning(
                    "candidate rejected during evaluation",
                    extra={
                        "event": "candidate_rejected",
                        "generation": generation,
                        "candidate": k,
                        "stage": "eval",
                        "reason": str(e),
                    },
                )
                continue

            # Retained only for shadow comparison and backward-compatible logs.
            # Pareto mode never uses this weighted value for selection.
            fitness = compute_fitness(metrics, FITNESS_WEIGHTS)
            cand_duration = round(time.perf_counter() - cand_start, 4)

            logger.info(
                "candidate completed",
                extra={
                    "event": "candidate_complete",
                    "generation": generation,
                    "candidate": k,
                    "legacy_fitness": fitness,
                    "duration_s": cand_duration,
                    **metrics,
                },
            )
            telemetry.record(
                "candidate_complete",
                generation=generation,
                candidate=k,
                module_path=module_path,
                legacy_fitness=fitness,
                duration_s=cand_duration,
                **metrics,
            )

            result = {
                "module_path": module_path,
                "code": code,
                "metrics": metrics,
                "fitness": fitness,
                "legacy_fitness": fitness,
                "checkpoint": checkpoint_path,
                "timing_s": {"total": cand_duration},
                "generation": generation,
                "candidate_index": k,
            }
            result["candidate_id"] = candidate_id(result)
            generation_results.append(result)

        gen_duration = round(time.perf_counter() - gen_start, 4)

        if not generation_results:
            full_log.append({
                "generation": generation,
                "results": [],
                "pareto_front_size": sum(
                    item.get("pareto_rank") == 0 for item in pareto_archive
                ),
                "archive_size": len(pareto_archive),
                "selection_mode": MULTI_OBJECTIVE_MODE,
            })
            telemetry.record(
                "generation_complete",
                generation=generation,
                duration_s=gen_duration,
                n_results=0,
                archive_size=len(pareto_archive),
            )
            logger.warning(
                "all candidates rejected this generation",
                extra={"event": "generation_empty", "generation": generation},
            )
            continue

        # Compute within-generation metadata for both modes. Shadow mode keeps
        # legacy behavior while making disagreement with Pareto visible.
        generation_fronts = annotate_population(generation_results, OBJECTIVE_SPECS)
        generation_front_ids = [
            generation_results[index]["candidate_id"]
            for index in generation_fronts[0]
        ]
        generation_best = max(generation_results, key=lambda r: r["fitness"])
        scalar_winner_on_front = generation_best["candidate_id"] in generation_front_ids

        pareto_archive = update_archive(
            pareto_archive,
            generation_results,
            OBJECTIVE_SPECS,
            PARETO_ARCHIVE_SIZE,
        )
        archive_by_id = {
            candidate["candidate_id"]: candidate for candidate in pareto_archive
        }
        for result in generation_results:
            archived = archive_by_id.get(result["candidate_id"])
            result["archive_member"] = archived is not None
            result["archive_pareto_rank"] = (
                archived["pareto_rank"] if archived is not None else None
            )
            result["archive_crowding_distance"] = (
                archived["crowding_distance"] if archived is not None else None
            )
        representative = select_representative(pareto_archive, OBJECTIVE_SPECS)
        pareto_front_size = sum(
            candidate.get("pareto_rank") == 0 for candidate in pareto_archive
        )

        generation_record = {
            "generation": generation,
            "results": generation_results,
            "selection_mode": MULTI_OBJECTIVE_MODE,
            "pareto_front_size": pareto_front_size,
            "archive_size": len(pareto_archive),
            "archive_candidate_ids": [
                candidate["candidate_id"] for candidate in pareto_archive
            ],
            "generation_front_candidate_ids": generation_front_ids,
            "legacy_scalar_winner_id": generation_best["candidate_id"],
            "legacy_scalar_winner_on_front": scalar_winner_on_front,
            "representative_id": (
                representative["candidate_id"] if representative else None
            ),
            "objective_specs": OBJECTIVE_SPECS,
        }
        full_log.append(generation_record)

        telemetry.record(
            "generation_complete",
            generation=generation,
            duration_s=gen_duration,
            n_results=len(generation_results),
            selection_mode=MULTI_OBJECTIVE_MODE,
            pareto_front_size=pareto_front_size,
            archive_size=len(pareto_archive),
            rank_distribution={
                str(rank): sum(
                    candidate.get("pareto_rank") == rank
                    for candidate in pareto_archive
                )
                for rank in sorted({
                    candidate.get("pareto_rank") for candidate in pareto_archive
                })
            },
            scalar_winner_on_front=scalar_winner_on_front,
        )

        logger.info(
            "generation selection complete",
            extra={
                "event": "generation_selection",
                "generation": generation,
                "selection_mode": MULTI_OBJECTIVE_MODE,
                "legacy_fitness": generation_best["fitness"],
                "legacy_winner": generation_best["module_path"],
                "legacy_winner_on_front": scalar_winner_on_front,
                "pareto_front_size": pareto_front_size,
                "archive_size": len(pareto_archive),
                "representative": (
                    representative["module_path"] if representative else None
                ),
                "duration_s": gen_duration,
            },
        )

        if MULTI_OBJECTIVE_MODE == "shadow":
            if best is None or generation_best["fitness"] > best["fitness"]:
                best = generation_best
                logger.info(
                    "new legacy scalar best (shadow mode)",
                    extra={
                        "event": "new_legacy_best",
                        "legacy_fitness": best["fitness"],
                        "module_path": best["module_path"],
                    },
                )
        else:
            # No weighted score is authoritative in Pareto mode.
            best = representative

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_json_safe(full_log), f, indent=2, default=str, allow_nan=False)
        logger.info("log updated", extra={"event": "log_write", "path": LOG_PATH})

    total_duration = round(time.perf_counter() - run_start, 4)
    pareto_archive = _confirm_archive(pareto_archive, telemetry)
    representative = select_representative(pareto_archive, OBJECTIVE_SPECS)
    if MULTI_OBJECTIVE_MODE == "pareto":
        best = representative
    if full_log:
        full_log[-1]["final_archive"] = pareto_archive
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_json_safe(full_log), f, indent=2, default=str, allow_nan=False)

    telemetry.record(
        "run_complete",
        duration_s=total_duration,
        had_best=best is not None,
        selection_mode=MULTI_OBJECTIVE_MODE,
        archive_size=len(pareto_archive),
        pareto_front_size=sum(
            candidate.get("pareto_rank") == 0 for candidate in pareto_archive
        ),
        pareto_front=[
            {
                "candidate_id": candidate["candidate_id"],
                "module_path": candidate["module_path"],
                "metrics": candidate["metrics"],
            }
            for candidate in pareto_archive
            if candidate.get("pareto_rank") == 0
        ],
    )

    if best is None:
        logger.warning("run finished with no successful candidate", extra={"event": "run_empty"})
        return

    logger.info(
        "EUREKA run finished",
        extra={
            "event": "run_complete",
            "duration_s": total_duration,
            "selection_mode": MULTI_OBJECTIVE_MODE,
            "representative_module": best["module_path"],
            "representative_checkpoint": best["checkpoint"],
            "representative_metrics": best["metrics"],
            "legacy_fitness": best.get("fitness"),
            "pareto_front_size": sum(
                candidate.get("pareto_rank") == 0 for candidate in pareto_archive
            ),
            "archive_size": len(pareto_archive),
        },
    )


if __name__ == "__main__":
    main()
