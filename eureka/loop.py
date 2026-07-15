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

--------------------------------------------------------------------------
Changes in this revision
--------------------------------------------------------------------------
P0 fixes:
  - total_duration is now measured AFTER _confirm_archive finishes, not
    before. Previously a run that logged "run finished (183m53s)" had
    actually taken ~248 minutes wall-clock — confirmation (~65 min here)
    ran after the timer was already read.
  - Training/eval failures are now caught as `except Exception` instead
    of `except RuntimeError` only. A single unexpected error (anything
    other than RuntimeError) used to be able to kill an entire multi-hour
    run with nothing saved from that generation.
  - eureka/objectives.py: epsilon-box tie-break is now quality-based
    instead of code-hash-based (see that file's docstring).

New capability: optional concurrent candidate training. Set
MAX_CONCURRENT_CANDIDATES > 1 in eureka_config.py (or via the
EUREKA_MAX_CONCURRENT_CANDIDATES env var) to train+evaluate multiple
candidates simultaneously in separate OS processes — useful on a
many-vCPU, no-GPU machine where the bottleneck is environment-stepping
throughput, not matrix multiplication. Default stays at 1 (fully
sequential) so the existing test suite, which monkeypatches
train_candidate/evaluate_candidate/smoke_test in-process, keeps working
unchanged: those monkeypatches are invisible to a separate spawned
process, so parallel execution is only exercised in real (unmocked) runs.
"""

import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    SEED_GENERATION_0_WITH_HUMAN_REWARD,
    TORCH_THREADS_PER_WORKER,
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
PLOTS_DIR = os.path.join("eureka", "plots")


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


def _pin_worker_threads() -> None:
    """
    Must run BEFORE importing torch inside a worker process. Without this,
    every concurrent candidate process tries to use ALL visible cores,
    causing oversubscription that can make parallel execution slower than
    sequential. Called at the top of every ProcessPoolExecutor worker.
    """
    threads = str(max(1, TORCH_THREADS_PER_WORKER))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    import torch
    torch.set_num_threads(max(1, TORCH_THREADS_PER_WORKER))


# --------------------------------------------------------------------------- #
# Worker functions for ProcessPoolExecutor (must be module-level to pickle).
# Each one NEVER raises across the process boundary: failures are captured
# and returned as data so one candidate's crash cannot take down its
# siblings or the pool itself.
# --------------------------------------------------------------------------- #

def _run_candidate_worker(job: dict) -> dict:
    """Trains + evaluates ONE already-smoke-tested, already-saved candidate."""
    _pin_worker_threads()
    from eureka.evaluate_candidate import evaluate_candidate as _evaluate_candidate
    from eureka.train_candidate import train_candidate as _train_candidate

    start = time.perf_counter()
    checkpoint_path = None
    component_history = None

    try:
        checkpoint_path = _train_candidate(
            job["module_path"], total_timesteps=job["total_timesteps"], seed=job["seed"],
        )
        components_sidecar = os.path.join(
            "eureka", "checkpoints", f"{job['module_path'].split('.')[-1]}_components.json"
        )
        if os.path.isfile(components_sidecar):
            with open(components_sidecar, encoding="utf-8") as f:
                sidecar = json.load(f)
            history = sidecar.get("component_history") or {}
            if history:
                component_history = history
    except Exception as e:
        return {
            "k": job["k"], "stage": "train", "error": str(e),
            "duration_s": round(time.perf_counter() - start, 4),
        }

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
    """Retrains + re-evaluates ONE (candidate, independent seed) pair."""
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


# --------------------------------------------------------------------------- #
# Smoke test + save (always sequential — cheap, and it's the gate that
# decides which candidates are even worth spending training compute on).
# --------------------------------------------------------------------------- #

def _smoke_test_and_save(
    candidates_code: list[str],
    generation: int,
    human_seed_index: int | None,
    telemetry: Telemetry,
) -> list[dict]:
    """Returns a list of surviving candidates: [{"k", "code", "module_path", "source"}, ...]."""
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
    """Original, fully in-process control flow. Used when
    MAX_CONCURRENT_CANDIDATES <= 1 — this is also the path the test suite's
    monkeypatching of train_candidate/evaluate_candidate exercises."""
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
            components_sidecar = os.path.join(
                "eureka", "checkpoints", f"gen{generation}_cand{k}_components.json"
            )
            if os.path.isfile(components_sidecar):
                with open(components_sidecar, encoding="utf-8") as f:
                    sidecar = json.load(f)
                history = sidecar.get("component_history") or {}
                if history:
                    component_history = history
        except Exception as e:
            # P0 fix: was `except RuntimeError` only — any other exception
            # used to propagate and kill the entire multi-hour run.
            _log_candidate_rejected(generation, k, "train", str(e))
            continue

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
    """
    Trains + evaluates up to MAX_CONCURRENT_CANDIDATES survivors at once,
    each in its own OS process. See the eureka_config.py module docstring
    for sizing guidance and why this is opt-in rather than default.
    """
    jobs = [
        {
            "k": s["k"],
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
                # Should be rare — the worker itself already catches training/
                # eval exceptions internally. This covers pool-level failures
                # (e.g. a worker process crashing outright).
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

    # Keep result ordering deterministic (by candidate index) regardless of
    # which worker happened to finish first.
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


# --------------------------------------------------------------------------- #
# Confirmation: retrain Pareto rank-0 finalists on independent seeds
# --------------------------------------------------------------------------- #

def _confirm_archive_sequential(finalists: list[dict], telemetry: Telemetry) -> tuple[dict, dict]:
    """Original in-process control flow — exercised by test_confirmation.py."""
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
    """Retrains all (finalist, seed) pairs concurrently across processes."""
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
    """
    Optionally retrain rank-zero finalists on independent seeds. Disabled by
    default because confirmation adds one full train/eval per configured seed
    (though with MAX_CONCURRENT_CANDIDATES > 1 those runs happen concurrently).
    """
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

        # Extra generation-0 slot: human-authored baseline (EUREKA Sec 4.4).
        # Not counted against K_CANDIDATES; only prepended once in generation 0.
        human_seed_index = None
        if generation == 0 and SEED_GENERATION_0_WITH_HUMAN_REWARD:
            from eureka.human_seed import HUMAN_SEED_CODE
            candidates_code = [HUMAN_SEED_CODE] + list(candidates_code)
            human_seed_index = 0

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

        survivors = _smoke_test_and_save(candidates_code, generation, human_seed_index, telemetry)
        generation_results = _train_and_evaluate_generation(survivors, generation, telemetry)

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

    pareto_archive = _confirm_archive(pareto_archive, telemetry)
    representative = select_representative(pareto_archive, OBJECTIVE_SPECS)
    if MULTI_OBJECTIVE_MODE == "pareto":
        best = representative
    if full_log:
        full_log[-1]["final_archive"] = pareto_archive
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_json_safe(full_log), f, indent=2, default=str, allow_nan=False)

    # P0 fix: total_duration is measured here, AFTER _confirm_archive, so it
    # reflects the run's real wall-clock time. Previously this line ran
    # before confirmation (which can take tens of minutes), so the reported
    # duration silently undercounted every run that had CONFIRMATION_SEEDS set.
    total_duration = round(time.perf_counter() - run_start, 4)

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

    # Best-effort plotting: a failure here (e.g. matplotlib missing) must
    # never take down a completed multi-hour search run.
    try:
        from eureka.plots import generate_run_plots
        plot_path = generate_run_plots(full_log, pareto_archive, PLOTS_DIR)
        logger.info("run plots saved", extra={"event": "plots_saved", "path": plot_path})
        telemetry.record("plots_saved", path=plot_path)
    except Exception as e:
        logger.warning("failed to generate run plots", extra={"event": "plots_failed", "reason": str(e)})

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