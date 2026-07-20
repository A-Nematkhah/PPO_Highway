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
    8. save an end-of-run summary plot to eureka/plots/

Run with:
    python -m eureka.loop

Requires: GROQ_API_KEY environment variable set (see llm_reward_designer.py)
"""

import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from eureka.csv_export import export_all
from eureka.eureka_config import (
    CONFIRMATION_SEEDS,
    EUREKA_N_ENVS,
    FITNESS_WEIGHTS,
    GROQ_MODEL,
    K_CANDIDATES,
    LLM_TEMPERATURE,
    MAX_CONCURRENT_CANDIDATES,
    MULTI_OBJECTIVE_MODE,
    N_EVAL_EPISODES,
    N_GENERATIONS,
    OBJECTIVE_SPECS,
    PARETO_ARCHIVE_SIZE,
    REFLECTION_ELITES,
    SCREENING_SECOND_SEED_ENABLED,
    SCREENING_SECOND_SEED_OFFSET,
    SEED_GENERATION_0_WITH_HUMAN_REWARD,
    TORCH_THREADS_PER_WORKER,
    TRAIN_STEPS_PER_CANDIDATE,
    candidate_base_seed,
)
from eureka.evaluate_candidate import evaluate_candidate
from eureka.experiment import ExperimentRun, build_experiment_config_snapshot
from eureka.fitness import compute_fitness
from eureka.llm_reward_designer import REFLECTION_TARGET_ROLES, generate_candidates
from eureka.logging_utils import get_logger, print_final_results_banner, print_generation_table
from eureka.objectives import (
    annotate_population,
    candidate_id,
    select_reflection_elites,
    select_representative,
    update_archive,
)
from eureka.reflection import build_reflection
from eureka.report_html import compute_execution_stats, generate_html_report, generation_reason
from eureka.run_metadata import collect_run_metadata
from eureka.smoke_test import smoke_test
from eureka.telemetry import Telemetry
from eureka.train_candidate import component_sidecar_path, train_candidate

logger = get_logger(__name__)

CANDIDATES_DIR = os.path.join("eureka", "candidates")
# LOG_PATH / PLOTS_DIR: retained as module constants for backward
# compatibility (anything importing them directly still gets a sensible
# value), but main() no longer writes to them directly - every run now
# gets its own numbered directory under RUNS_ROOT (see experiment.py),
# and main() uses that run's log_path/plots_dir instead.
LOG_PATH = os.path.join("eureka", "eureka_log.json")
PLOTS_DIR = os.path.join("eureka", "plots")
RUNS_ROOT = "runs"


def _log_banner():
    logger.info(
        "EUREKA reward search starting",
        extra={
            "event": "run_start",
            "generations": N_GENERATIONS,
            "candidates_per_gen": K_CANDIDATES,
            "train_steps_per_candidate": TRAIN_STEPS_PER_CANDIDATE,
            "parallel_envs": EUREKA_N_ENVS,
            "max_concurrent_candidates": MAX_CONCURRENT_CANDIDATES,
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
    keys = ("crash_rate", "mean_speed", "mean_overtakes", "mean_raw_return")
    return {
        key: sum(float(run[key]) for run in runs) / len(runs)
        for key in keys
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_component_history(short_name: str, generation: int, k: int) -> dict | None:
    """
    Best-effort read of a candidate's component-history sidecar (written by
    train_candidate.py). Returns None on any missing/unreadable/corrupt
    file - deliberately NOT re-raised to the caller's training try/except.

    component_history is purely diagnostic (LLM reflection context, see
    reflection.py); it must never be able to reject an otherwise
    successfully-trained-and-checkpointed candidate. Before this helper
    existed, a corrupt sidecar's json.JSONDecodeError propagated out of the
    same try/except that wraps train_candidate() itself, so a bad sidecar
    silently discarded a good checkpoint - see
    eureka/tests/test_component_sidecar.py::test_corrupt_component_sidecar_does_not_abort_loop.
    """
    path = component_sidecar_path(short_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "component sidecar unreadable, continuing without component_history",
            extra={
                "event": "component_sidecar_read_failed",
                "generation": generation, "candidate": k, "path": path, "reason": str(e),
            },
        )
        return None
    history = sidecar.get("component_history") or {}
    return history or None


def _pin_worker_threads() -> None:
    threads = str(max(1, TORCH_THREADS_PER_WORKER))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    import torch
    torch.set_num_threads(max(1, TORCH_THREADS_PER_WORKER))


def _run_candidate_worker(job: dict) -> dict:
    _pin_worker_threads()
    from eureka.evaluate_candidate import evaluate_candidate as _evaluate_candidate
    from eureka.train_candidate import train_candidate as _train_candidate

    start = time.perf_counter()

    try:
        checkpoint_path = _train_candidate(
            job["module_path"], total_timesteps=job["total_timesteps"], seed=job["seed"],
        )
    except Exception as e:
        return {
            "k": job["k"], "stage": "train", "error": str(e),
            "duration_s": round(time.perf_counter() - start, 4),
        }

    # Deliberately its own try/except via _load_component_history, NOT
    # folded into the training try/except above: a corrupt sidecar is
    # diagnostic-data-only and must never discard an otherwise
    # successfully-trained checkpoint (see _load_component_history's
    # docstring and test_component_sidecar.py).
    component_history = _load_component_history(
        job["module_path"].split(".")[-1], job.get("generation", -1), job["k"]
    )

    try:
        metrics = _evaluate_candidate(checkpoint_path, job["module_path"], n_episodes=job["n_eval_episodes"])
    except Exception as e:
        return {
            "k": job["k"], "stage": "eval", "error": str(e),
            "duration_s": round(time.perf_counter() - start, 4),
        }

    return {
        "k": job["k"], "stage": "done", "error": None,
        "checkpoint": checkpoint_path, "metrics": metrics,
        "component_history": component_history,
        "duration_s": round(time.perf_counter() - start, 4),
    }


def _run_confirmation_worker(job: dict) -> dict:
    _pin_worker_threads()
    from eureka.evaluate_candidate import evaluate_candidate as _evaluate_candidate
    from eureka.train_candidate import train_candidate as _train_candidate

    start = time.perf_counter()
    try:
        checkpoint = _train_candidate(
            job["module_path"], total_timesteps=job["total_timesteps"], seed=job["seed"],
        )
        metrics = _evaluate_candidate(checkpoint, job["module_path"], n_episodes=job["n_eval_episodes"])
    except Exception as e:
        return {
            "candidate_id": job["candidate_id"], "seed": job["seed"],
            "error": str(e), "duration_s": round(time.perf_counter() - start, 4),
        }

    return {
        "candidate_id": job["candidate_id"], "seed": job["seed"], "error": None,
        "checkpoint": checkpoint, "metrics": metrics,
        "duration_s": round(time.perf_counter() - start, 4),
    }


def _smoke_test_and_save(
    candidates_code: list[str],
    generation: int,
    human_seed_index: int | None,
    telemetry: Telemetry,
) -> list[dict]:
    survivors = []
    for k, code in enumerate(candidates_code):
        source = "human_seed" if k == human_seed_index else "llm"
        logger.info(
            "candidate started",
            extra={"event": "candidate_start", "generation": generation, "candidate": k, "source": source},
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
                    "event": "candidate_rejected", "generation": generation,
                    "candidate": k, "stage": "smoke_test", "reason": message,
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
            extra={"event": "candidate_saved", "generation": generation, "candidate": k, "path": file_path},
        )
        survivors.append({"k": k, "code": code, "module_path": module_path, "source": source})

    return survivors


def _build_result(
    survivor: dict,
    generation: int,
    metrics: dict,
    checkpoint_path: str,
    component_history: dict | None,
    cand_duration: float,
) -> dict:
    fitness = compute_fitness(metrics, FITNESS_WEIGHTS)
    result = {
        "module_path": survivor["module_path"],
        "code": survivor["code"],
        "metrics": metrics,
        "fitness": fitness,
        "legacy_fitness": fitness,
        "checkpoint": checkpoint_path,
        "timing_s": {"total": cand_duration},
        "generation": generation,
        "candidate_index": survivor["k"],
        "source": survivor["source"],
    }
    if component_history:
        result["component_history"] = component_history
    result["candidate_id"] = candidate_id(result)
    return result


def _log_candidate_complete(generation: int, k: int, source: str, fitness: float,
                             duration_s: float, metrics: dict, telemetry: Telemetry,
                             module_path: str) -> None:
    logger.info(
        "candidate completed",
        extra={
            "event": "candidate_complete", "generation": generation, "candidate": k,
            "source": source, "legacy_fitness": fitness, "duration_s": duration_s, **metrics,
        },
    )
    telemetry.record(
        "candidate_complete", generation=generation, candidate=k, source=source,
        module_path=module_path, legacy_fitness=fitness, duration_s=duration_s, **metrics,
    )


def _log_candidate_rejected(generation: int, k: int, stage: str, reason: str) -> None:
    logger.warning(
        "candidate rejected during training" if stage == "train" else "candidate rejected during evaluation",
        extra={"event": "candidate_rejected", "generation": generation, "candidate": k, "stage": stage, "reason": reason},
    )


def _train_and_evaluate_sequential(
    survivors: list[dict], generation: int, telemetry: Telemetry,
) -> list[dict]:
    generation_results = []

    for survivor in survivors:
        k = survivor["k"]
        cand_start = time.perf_counter()
        checkpoint_path = None
        component_history = None

        try:
            with telemetry.timed("train", generation=generation, candidate=k, module_path=survivor["module_path"]) as train_ctx:
                train_ctx["total_timesteps"] = TRAIN_STEPS_PER_CANDIDATE
                checkpoint_path = train_candidate(
                    survivor["module_path"], total_timesteps=TRAIN_STEPS_PER_CANDIDATE,
                    seed=candidate_base_seed(generation, k),
                )
                train_ctx["checkpoint"] = checkpoint_path
        except Exception as e:
            _log_candidate_rejected(generation, k, "train", str(e))
            continue

        # Deliberately OUTSIDE the training try/except above: this is
        # diagnostic-only data (see _load_component_history docstring) and
        # must never cause a successfully-trained candidate to be rejected.
        component_history = _load_component_history(f"gen{generation}_cand{k}", generation, k)

        try:
            with telemetry.timed("eval", generation=generation, candidate=k, module_path=survivor["module_path"]) as eval_ctx:
                metrics = evaluate_candidate(checkpoint_path, survivor["module_path"], n_episodes=N_EVAL_EPISODES)
                eval_ctx.update(metrics)
        except Exception as e:
            _log_candidate_rejected(generation, k, "eval", str(e))
            continue

        cand_duration = round(time.perf_counter() - cand_start, 4)
        result = _build_result(survivor, generation, metrics, checkpoint_path, component_history, cand_duration)
        _log_candidate_complete(generation, k, survivor["source"], result["fitness"], cand_duration, metrics, telemetry, survivor["module_path"])
        generation_results.append(result)

    return generation_results


def _train_and_evaluate_parallel(
    survivors: list[dict], generation: int, telemetry: Telemetry,
) -> list[dict]:
    jobs = [
        {
            "k": s["k"],
            "generation": generation,
            "module_path": s["module_path"],
            "seed": candidate_base_seed(generation, s["k"]),
            "total_timesteps": TRAIN_STEPS_PER_CANDIDATE,
            "n_eval_episodes": N_EVAL_EPISODES,
        }
        for s in survivors
    ]
    survivors_by_k = {s["k"]: s for s in survivors}
    generation_results = []

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_CANDIDATES, mp_context=ctx) as pool:
        futures = {pool.submit(_run_candidate_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            k = job["k"]
            survivor = survivors_by_k[k]
            try:
                worker_result = future.result()
            except Exception as e:
                worker_result = {"k": k, "stage": "pool_error", "error": str(e), "duration_s": 0.0}

            if worker_result.get("error"):
                stage = worker_result.get("stage", "unknown")
                _log_candidate_rejected(generation, k, stage if stage in ("train", "eval") else "train", worker_result["error"])
                continue

            metrics = worker_result["metrics"]
            cand_duration = worker_result["duration_s"]
            result = _build_result(
                survivor, generation, metrics, worker_result["checkpoint"],
                worker_result.get("component_history"), cand_duration,
            )
            _log_candidate_complete(generation, k, survivor["source"], result["fitness"], cand_duration, metrics, telemetry, survivor["module_path"])
            generation_results.append(result)

    generation_results.sort(key=lambda r: r["candidate_index"])
    return generation_results


def _train_and_evaluate_generation(
    survivors: list[dict], generation: int, telemetry: Telemetry,
) -> list[dict]:
    if not survivors:
        return []
    if MAX_CONCURRENT_CANDIDATES <= 1:
        return _train_and_evaluate_sequential(survivors, generation, telemetry)
    return _train_and_evaluate_parallel(survivors, generation, telemetry)


def _screen_confirm_front(
    generation_results: list[dict],
    front_indices: list[int],
    generation: int,
    telemetry: Telemetry,
) -> list[dict]:
    """
    Retrain+reevaluate this generation's Pareto-front candidates on one extra
    independent seed and average into their screening metrics, IN PLACE.

    CONFIRMATION_SEEDS only re-checks the whole run's FINAL rank-0 archive
    members at the very end - a candidate that got unlucky on its single
    screening seed can be evicted from the archive by update_archive() long
    before confirmation ever sees it. This is a cheaper, earlier version of
    the same idea: only the current generation's front (typically a handful
    of candidates, not all K_CANDIDATES) gets a second seed, right before
    this generation's results are folded into the cross-generation archive.

    Mutates and returns `generation_results` (only entries at `front_indices`
    are touched) so callers can simply re-run nondominated sorting afterward
    on the same list.
    """
    if not SCREENING_SECOND_SEED_ENABLED:
        return generation_results

    for index in front_indices:
        result = generation_results[index]
        k = result["candidate_index"]
        second_seed = candidate_base_seed(generation, k) + SCREENING_SECOND_SEED_OFFSET

        try:
            with telemetry.timed(
                "screening_second_seed", generation=generation, candidate=k,
                module_path=result["module_path"], seed=second_seed,
            ) as ctx:
                checkpoint = train_candidate(
                    result["module_path"], total_timesteps=TRAIN_STEPS_PER_CANDIDATE,
                    seed=second_seed,
                )
                second_metrics = evaluate_candidate(
                    checkpoint, result["module_path"], n_episodes=N_EVAL_EPISODES,
                )
                ctx.update(second_metrics)
        except Exception as e:
            logger.warning(
                "screening second-seed run failed, keeping single-seed metrics",
                extra={
                    "event": "screening_second_seed_failed",
                    "generation": generation, "candidate": k, "reason": str(e),
                },
            )
            continue

        result["screening_seed_1_metrics"] = result["metrics"]
        result["screening_seed_2_metrics"] = second_metrics
        result["metrics"] = _aggregate_metrics([result["metrics"], second_metrics])
        result["fitness"] = compute_fitness(result["metrics"], FITNESS_WEIGHTS)
        result["legacy_fitness"] = result["fitness"]

    return generation_results


def _confirm_archive_sequential(finalists: list[dict], telemetry: Telemetry) -> tuple[dict, dict]:
    runs_by_candidate = {candidate_id(c): [c["metrics"]] for c in finalists}
    records_by_candidate: dict[str, list[dict]] = {candidate_id(c): [] for c in finalists}

    for candidate in finalists:
        cid = candidate_id(candidate)
        for seed in CONFIRMATION_SEEDS:
            try:
                with telemetry.timed(
                    "confirmation", candidate_id=cid, module_path=candidate["module_path"], seed=seed,
                ) as context:
                    checkpoint = train_candidate(
                        candidate["module_path"], total_timesteps=TRAIN_STEPS_PER_CANDIDATE, seed=seed,
                    )
                    metrics = evaluate_candidate(checkpoint, candidate["module_path"], n_episodes=N_EVAL_EPISODES)
                    context.update(metrics)
                    runs_by_candidate[cid].append(metrics)
                    records_by_candidate[cid].append({"seed": seed, "checkpoint": checkpoint, "metrics": metrics})
            except Exception as error:
                logger.warning(
                    "candidate confirmation failed",
                    extra={"event": "confirmation_failed", "candidate_id": cid, "seed": seed, "reason": str(error)},
                )

    return runs_by_candidate, records_by_candidate


def _confirm_archive_parallel(finalists: list[dict], telemetry: Telemetry) -> tuple[dict, dict]:
    jobs = [
        {
            "candidate_id": candidate_id(candidate),
            "module_path": candidate["module_path"],
            "seed": seed,
            "total_timesteps": TRAIN_STEPS_PER_CANDIDATE,
            "n_eval_episodes": N_EVAL_EPISODES,
        }
        for candidate in finalists
        for seed in CONFIRMATION_SEEDS
    ]
    runs_by_candidate = {candidate_id(c): [c["metrics"]] for c in finalists}
    records_by_candidate: dict[str, list[dict]] = {candidate_id(c): [] for c in finalists}

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_CANDIDATES, mp_context=ctx) as pool:
        futures = {pool.submit(_run_confirmation_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"candidate_id": job["candidate_id"], "seed": job["seed"], "error": str(e)}

            if result.get("error"):
                logger.warning(
                    "candidate confirmation failed",
                    extra={
                        "event": "confirmation_failed", "candidate_id": job["candidate_id"],
                        "seed": job["seed"], "reason": result["error"],
                    },
                )
                continue

            telemetry.record(
                "confirmation", candidate_id=job["candidate_id"], module_path=job["module_path"],
                seed=job["seed"], duration_s=result["duration_s"], **result["metrics"],
            )
            runs_by_candidate[job["candidate_id"]].append(result["metrics"])
            records_by_candidate[job["candidate_id"]].append(
                {"seed": job["seed"], "checkpoint": result["checkpoint"], "metrics": result["metrics"]}
            )

    return runs_by_candidate, records_by_candidate


def _confirm_archive(archive: list[dict], telemetry: Telemetry) -> list[dict]:
    if not CONFIRMATION_SEEDS:
        return archive

    finalists = [c for c in archive if c.get("pareto_rank") == 0]
    if not finalists:
        return archive

    if MAX_CONCURRENT_CANDIDATES <= 1:
        runs_by_candidate, records_by_candidate = _confirm_archive_sequential(finalists, telemetry)
    else:
        runs_by_candidate, records_by_candidate = _confirm_archive_parallel(finalists, telemetry)

    confirmed = []
    for candidate in archive:
        if candidate.get("pareto_rank") != 0:
            confirmed.append(candidate)
            continue
        cid = candidate_id(candidate)
        item = dict(candidate)
        item["screening_metrics"] = candidate["metrics"]
        item["confirmation_runs"] = records_by_candidate.get(cid, [])
        item["metrics"] = _aggregate_metrics(runs_by_candidate.get(cid, [candidate["metrics"]]))

        # Recompute fitness from the CONFIRMED metrics, not the pre-confirmation
        # screening metrics, and point the checkpoint at the last confirmation
        # run. Without this, the final archive shows a fitness score and a
        # checkpoint file that don't correspond to the metrics reported next
        # to them (both were still the screening-run values).
        item["fitness"] = compute_fitness(item["metrics"], FITNESS_WEIGHTS)
        item["legacy_fitness"] = item["fitness"]
        if records_by_candidate.get(cid):
            item["screening_checkpoint"] = candidate["checkpoint"]
            item["checkpoint"] = records_by_candidate[cid][-1]["checkpoint"]

        confirmed.append(item)

    return update_archive([], confirmed, OBJECTIVE_SPECS, PARETO_ARCHIVE_SIZE)


def _record_empty_generation(
    generation: int, pareto_archive: list[dict], full_log: list[dict],
    telemetry: Telemetry, gen_start: float,
) -> None:
    """
    Records a generation where the LLM returned zero usable candidates
    (e.g. every call failed with RequestTooLargeError even after the one
    retry in generate_candidates, or every response was unparseable).

    Escalated to ERROR-level plus a dedicated telemetry event (distinct
    from the generic "generation_complete" row) so a run silently losing a
    third of its search budget to LLM failures is impossible to miss in an
    hours-long log - `grep generation_all_candidates_failed` finds it
    immediately instead of requiring someone to notice the archive ended
    up smaller than expected.
    """
    logger.error(
        "ALL LLM CANDIDATES FAILED THIS GENERATION - generation "
        "budget lost entirely (check llm_call_error / "
        "llm_call_request_too_large events above for the cause)",
        extra={
            "event": "generation_all_candidates_failed",
            "generation": generation,
            "k_requested": K_CANDIDATES,
        },
    )
    telemetry.record(
        "generation_all_candidates_failed",
        generation=generation,
        k_requested=K_CANDIDATES,
    )
    full_log.append({
        "generation": generation,
        "results": [],
        "pareto_front_size": sum(item.get("pareto_rank") == 0 for item in pareto_archive),
        "archive_size": len(pareto_archive),
        "selection_mode": MULTI_OBJECTIVE_MODE,
        "all_candidates_failed": True,
    })
    telemetry.record(
        "generation_complete",
        generation=generation,
        duration_s=round(time.perf_counter() - gen_start, 4),
        n_results=0,
    )


def _record_generation_with_no_survivors(
    generation: int, pareto_archive: list[dict], full_log: list[dict],
    telemetry: Telemetry, gen_duration: float,
) -> None:
    """Records a generation where the LLM returned candidates, but every one
    was rejected by the smoke test / training / evaluation."""
    full_log.append({
        "generation": generation,
        "results": [],
        "pareto_front_size": sum(item.get("pareto_rank") == 0 for item in pareto_archive),
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


@dataclass
class GenerationOutcome:
    """What main()'s generation loop needs back after a successful
    generation: the updated cross-generation archive plus the two
    candidates for "best" (legacy scalar winner and Pareto representative)
    - main() picks between them based on MULTI_OBJECTIVE_MODE."""

    pareto_archive: list[dict]
    generation_best: dict
    representative: Optional[dict]


def _finalize_successful_generation(
    generation: int,
    generation_results: list[dict],
    pareto_archive: list[dict],
    full_log: list[dict],
    telemetry: Telemetry,
    gen_duration: float,
) -> GenerationOutcome:
    """
    Pareto-ranks this generation's results, re-screens its front on a
    second seed, merges it into the cross-generation archive, builds and
    appends the generation's full_log record, and emits the
    generation_complete / generation_selection log+telemetry events.

    This is the "happy path" body of the per-generation loop in main() -
    split out because it was, by a wide margin, the single longest and
    most deeply-nested block in the whole module (originally ~115 lines
    inline). Splitting it doesn't change any of its logic or ordering,
    only where it lives.
    """
    generation_fronts = annotate_population(generation_results, OBJECTIVE_SPECS)

    # Re-screen this generation's front on a second independent seed
    # BEFORE it is locked into the cross-generation archive or used to
    # pick the legacy scalar winner - otherwise a genuinely good
    # candidate can be evicted purely from single-seed noise before
    # CONFIRMATION_SEEDS (which only runs at the very end) ever gets a
    # chance to re-check it.
    generation_results = _screen_confirm_front(
        generation_results, generation_fronts[0], generation, telemetry
    )
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
    archive_by_id = {candidate["candidate_id"]: candidate for candidate in pareto_archive}
    for result in generation_results:
        archived = archive_by_id.get(result["candidate_id"])
        result["archive_member"] = archived is not None
        result["archive_pareto_rank"] = archived["pareto_rank"] if archived is not None else None
        result["archive_crowding_distance"] = (
            archived["crowding_distance"] if archived is not None else None
        )
    representative = select_representative(pareto_archive, OBJECTIVE_SPECS)
    pareto_front_size = sum(candidate.get("pareto_rank") == 0 for candidate in pareto_archive)

    generation_record = {
        "generation": generation,
        "results": generation_results,
        "selection_mode": MULTI_OBJECTIVE_MODE,
        "pareto_front_size": pareto_front_size,
        "archive_size": len(pareto_archive),
        "archive_candidate_ids": [candidate["candidate_id"] for candidate in pareto_archive],
        "generation_front_candidate_ids": generation_front_ids,
        "legacy_scalar_winner_id": generation_best["candidate_id"],
        "legacy_scalar_winner_on_front": scalar_winner_on_front,
        "representative_id": representative["candidate_id"] if representative else None,
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
            str(rank): sum(candidate.get("pareto_rank") == rank for candidate in pareto_archive)
            for rank in sorted({candidate.get("pareto_rank") for candidate in pareto_archive})
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
            "representative": representative["module_path"] if representative else None,
            "duration_s": gen_duration,
        },
    )

    winner_for_display = representative if MULTI_OBJECTIVE_MODE == "pareto" else generation_best
    print_generation_table(
        generation=generation,
        front_candidates=[generation_results[index] for index in generation_fronts[0]],
        winner_module_path=winner_for_display["module_path"] if winner_for_display else None,
        reason=generation_reason(generation_record),
    )

    return GenerationOutcome(
        pareto_archive=pareto_archive,
        generation_best=generation_best,
        representative=representative,
    )


@dataclass
class RunFinalization:
    """What main() needs back from _finalize_run to drive the new
    experiment-manager artifacts step (CSV export, HTML report, checkpoint
    archiving) - the CONFIRMED final archive and representative, not the
    pre-confirmation values main() already had."""

    pareto_archive: list[dict]
    representative: Optional[dict]
    best: Optional[dict]


def _finalize_run(
    pareto_archive: list[dict],
    full_log: list[dict],
    telemetry: Telemetry,
    best: Optional[dict],
    run_start: float,
    empty_generations: list[int],
    log_path: str,
    plots_dir: str,
) -> RunFinalization:
    """
    Everything that happens once the generation loop is done: multi-seed
    confirmation of the final archive, writing the final log, the
    end-of-run plot, and the final summary log lines (including the
    impossible-to-miss warning when one or more generations lost their
    entire LLM candidate budget).

    log_path/plots_dir are passed explicitly (rather than read from
    module-level LOG_PATH/PLOTS_DIR constants) so each run can be pointed
    at its own experiment directory - see ExperimentRun in experiment.py.
    """
    pareto_archive = _confirm_archive(pareto_archive, telemetry)
    representative = select_representative(pareto_archive, OBJECTIVE_SPECS)
    if MULTI_OBJECTIVE_MODE == "pareto":
        best = representative
    if full_log:
        full_log[-1]["final_archive"] = pareto_archive
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(full_log), f, indent=2, default=str, allow_nan=False)

    total_duration = round(time.perf_counter() - run_start, 4)

    telemetry.record(
        "run_complete",
        duration_s=total_duration,
        had_best=best is not None,
        selection_mode=MULTI_OBJECTIVE_MODE,
        archive_size=len(pareto_archive),
        pareto_front_size=sum(candidate.get("pareto_rank") == 0 for candidate in pareto_archive),
        pareto_front=[
            {
                "candidate_id": candidate["candidate_id"],
                "module_path": candidate["module_path"],
                "metrics": candidate["metrics"],
            }
            for candidate in pareto_archive
            if candidate.get("pareto_rank") == 0
        ],
        empty_generations=empty_generations,
    )

    try:
        from eureka.plots import generate_run_plots
        plot_path = generate_run_plots(full_log, pareto_archive, plots_dir)
        logger.info("run plots saved", extra={"event": "plots_saved", "path": plot_path})
        telemetry.record("plots_saved", path=plot_path)
    except Exception as e:
        logger.warning("failed to generate run plots", extra={"event": "plots_failed", "reason": str(e)})

    if best is None:
        logger.warning("run finished with no successful candidate", extra={"event": "run_empty"})
        return RunFinalization(pareto_archive=pareto_archive, representative=representative, best=best)

    # A run can "finish" with a usable archive while still having silently
    # lost one or more generations to LLM failures. Surface that loudly in
    # the final summary log line instead of requiring someone to grep the
    # whole run for generation_all_candidates_failed.
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
            "pareto_front_size": sum(candidate.get("pareto_rank") == 0 for candidate in pareto_archive),
            "archive_size": len(pareto_archive),
            "generations_with_zero_candidates": empty_generations,
        },
    )
    if empty_generations:
        logger.error(
            f"RUN COMPLETED BUT {len(empty_generations)}/{N_GENERATIONS} "
            f"GENERATIONS RETURNED ZERO CANDIDATES (generations: "
            f"{empty_generations}) - search budget was silently reduced; "
            "see generation_all_candidates_failed events above",
            extra={
                "event": "run_completed_with_empty_generations",
                "empty_generation_count": len(empty_generations),
                "total_generations": N_GENERATIONS,
                "empty_generations": empty_generations,
            },
        )

    return RunFinalization(pareto_archive=pareto_archive, representative=representative, best=best)


def _archive_reflection_prompts(run: ExperimentRun, generation: int, reflection_context) -> None:
    """
    Archives the LLM reflection prompt(s) that will be used to produce
    this generation's candidates, for the report's "Reflection" section
    and offline inspection - mirrors generate_candidates()'s own role-
    cycling (see REFLECTION_TARGET_ROLES in llm_reward_designer.py)
    exactly, WITHOUT calling into or modifying generate_candidates()
    itself: this rebuilds the same prompt text via the same
    build_reflection() call generate_candidates() makes internally, using
    the exact same reflection_context this generation actually received.

    Best-effort: archiving a prompt is a reporting nicety, never a reason
    to fail a generation that would otherwise succeed.
    """
    try:
        if not reflection_context:
            run.archive_reflection_prompt(generation, 0, None, build_reflection(None))
            return
        roles = REFLECTION_TARGET_ROLES[: min(K_CANDIDATES, len(REFLECTION_TARGET_ROLES))]
        for index, role in enumerate(roles):
            prompt = build_reflection(reflection_context, target_role=role)
            run.archive_reflection_prompt(generation, index, role, prompt)
    except Exception as e:
        logger.warning(
            "failed to archive reflection prompt",
            extra={"event": "reflection_archive_failed", "generation": generation, "reason": str(e)},
        )


def _finalize_experiment_artifacts(
    run: ExperimentRun,
    finalization: RunFinalization,
    full_log: list[dict],
    telemetry: Telemetry,
    metadata,
    run_start: float,
) -> None:
    """
    New, additive experiment-manager finalization, run after
    _finalize_run's algorithm-level finalization (confirmation, plots,
    summary logs): archives final checkpoints/reward code, exports CSVs,
    writes the self-contained HTML report, updates metadata.json with the
    final execution time, and prints the console FINAL RESULTS banner.

    Every step here is best-effort and independently wrapped: a failure
    generating the HTML report (say) must never look like the search
    itself failed, and must never prevent the other artifacts (CSVs,
    checkpoints, metadata) from still being written.
    """
    pareto_archive = finalization.pareto_archive
    representative = finalization.representative
    best = finalization.best
    representative_id = representative["candidate_id"] if representative else None

    for candidate in pareto_archive:
        checkpoint = candidate.get("checkpoint")
        if checkpoint:
            run.archive_checkpoint(checkpoint)

    if best is not None and best.get("code"):
        run.archive_final_reward(best["code"])

    try:
        export_all(full_log, pareto_archive, run.run_dir, representative_id=representative_id)
    except Exception as e:
        logger.warning("CSV export failed", extra={"event": "csv_export_failed", "reason": str(e)})

    total_duration = round(time.perf_counter() - run_start, 4)
    metadata_payload = run.write_metadata(metadata, execution_time_s=total_duration)

    try:
        execution_stats = compute_execution_stats(run.telemetry_path, total_runtime_s=total_duration)
        generate_html_report(
            run_dir=run.run_dir,
            full_log=full_log,
            archive=pareto_archive,
            representative_id=representative_id,
            metadata=metadata_payload,
            config=build_experiment_config_snapshot(),
            execution_stats=execution_stats,
            plots_dir=run.plots_dir,
            reflection_dir=run.reflection_dir,
        )
        logger.info(
            "HTML report generated",
            extra={"event": "report_generated", "path": str(run.report_html_path)},
        )
    except Exception as e:
        logger.warning("HTML report generation failed", extra={"event": "report_failed", "reason": str(e)})

    summary = f"{len(pareto_archive)} candidates in final archive, {N_GENERATIONS} generations requested."
    print_final_results_banner(pareto_archive, representative_id, summary)


def main():
    if MULTI_OBJECTIVE_MODE not in {"shadow", "pareto"}:
        raise ValueError(
            "MULTI_OBJECTIVE_MODE must be 'shadow' or 'pareto', "
            f"got {MULTI_OBJECTIVE_MODE!r}"
        )
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    # Every invocation gets its own numbered runs/run_NNNN/ directory -
    # nothing overwrites a previous run. See experiment.py for exactly
    # which files stay in their existing shared locations (candidate
    # source under eureka/candidates/, checkpoints under
    # eureka/checkpoints/ - both required by sandbox.py's dotted-module-
    # path loading and train_candidate.py's confirmation-run reuse) versus
    # which are archived into / written directly to the run directory.
    run = ExperimentRun.start(runs_root=RUNS_ROOT)
    metadata = collect_run_metadata(GROQ_MODEL)
    run.write_config_snapshot(build_experiment_config_snapshot())
    run.write_metadata(metadata)

    with run.capture_console():
        telemetry = Telemetry(path=str(run.telemetry_path))
        _log_banner()
        logger.info(
            "run directory created",
            extra={"event": "run_dir_created", "run": run.run_name, "path": str(run.run_dir)},
        )

        best = None
        pareto_archive: list[dict] = []
        full_log: list[dict] = []
        run_start = time.perf_counter()
        # Tracks how many generations returned zero candidates from the LLM
        # (e.g. every call failed with RequestTooLargeError even after the one
        # retry in generate_candidates, or every response was unparseable) so
        # a run silently losing part of its search budget is surfaced loudly
        # at the end instead of requiring someone to notice the archive was
        # smaller than expected.
        empty_generations: list[int] = []

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
            _archive_reflection_prompts(run, generation, reflection_context)

            with telemetry.timed("llm_generation", generation=generation, k=K_CANDIDATES) as llm_ctx:
                candidates_code = generate_candidates(
                    reflection_context, k=K_CANDIDATES, generation=generation,
                    model=GROQ_MODEL, temperature=LLM_TEMPERATURE,
                )
                llm_ctx["n_received"] = len(candidates_code)

            human_seed_index = None
            if generation == 0 and SEED_GENERATION_0_WITH_HUMAN_REWARD:
                from eureka.human_seed import HUMAN_SEED_CODE
                candidates_code = [HUMAN_SEED_CODE] + list(candidates_code)
                human_seed_index = 0

            if not candidates_code:
                empty_generations.append(generation)
                _record_empty_generation(generation, pareto_archive, full_log, telemetry, gen_start)
                continue

            survivors = _smoke_test_and_save(candidates_code, generation, human_seed_index, telemetry)
            for survivor in survivors:
                run.archive_candidate_code(f"gen{generation}_cand{survivor['k']}", survivor["code"])

            generation_results = _train_and_evaluate_generation(survivors, generation, telemetry)
            gen_duration = round(time.perf_counter() - gen_start, 4)

            if not generation_results:
                _record_generation_with_no_survivors(
                    generation, pareto_archive, full_log, telemetry, gen_duration
                )
                continue

            outcome = _finalize_successful_generation(
                generation, generation_results, pareto_archive, full_log, telemetry, gen_duration,
            )
            pareto_archive = outcome.pareto_archive

            if MULTI_OBJECTIVE_MODE == "shadow":
                if best is None or outcome.generation_best["fitness"] > best["fitness"]:
                    best = outcome.generation_best
                    logger.info(
                        "new legacy scalar best (shadow mode)",
                        extra={
                            "event": "new_legacy_best",
                            "legacy_fitness": best["fitness"],
                            "module_path": best["module_path"],
                        },
                    )
            else:
                best = outcome.representative

            with open(run.log_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(full_log), f, indent=2, default=str, allow_nan=False)
            logger.info("log updated", extra={"event": "log_write", "path": str(run.log_path)})

        finalization = _finalize_run(
            pareto_archive, full_log, telemetry, best, run_start, empty_generations,
            log_path=str(run.log_path), plots_dir=str(run.plots_dir),
        )
        _finalize_experiment_artifacts(run, finalization, full_log, telemetry, metadata, run_start)


if __name__ == "__main__":
    main()