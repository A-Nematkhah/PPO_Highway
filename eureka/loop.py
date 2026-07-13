"""
loop.py

The main Phase 4 orchestrator. For each generation:
    1. ask the LLM for K reward-shaping code candidates (informed by the
       best candidate + its metrics from the previous generation)
    2. smoke-test each candidate; reject anything that fails
    3. train each surviving candidate for a short budget
    4. run a deterministic evaluation to get objective metrics
    5. compute fitness from those metrics
    6. keep the best candidate seen across ALL generations so far
    7. log everything to eureka/eureka_log.json

Run with:
    python -m eureka.loop

Requires: GROQ_API_KEY environment variable set (see llm_reward_designer.py)
"""

import json
import os
import time

from eureka.eureka_config import (
    EUREKA_N_ENVS,
    FITNESS_WEIGHTS,
    GROQ_MODEL,
    K_CANDIDATES,
    LLM_TEMPERATURE,
    N_EVAL_EPISODES,
    N_GENERATIONS,
    TRAIN_STEPS_PER_CANDIDATE,
)
from eureka.evaluate_candidate import evaluate_candidate
from eureka.fitness import compute_fitness
from eureka.llm_reward_designer import generate_candidates
from eureka.smoke_test import smoke_test
from eureka.train_candidate import train_candidate

CANDIDATES_DIR = os.path.join("eureka", "candidates")
LOG_PATH = os.path.join("eureka", "eureka_log.json")


def _print_banner():
    print("=== EUREKA reward search ===", flush=True)
    print(f"  generations:      {N_GENERATIONS}", flush=True)
    print(f"  candidates/gen:   {K_CANDIDATES}", flush=True)
    print(f"  train steps:      {TRAIN_STEPS_PER_CANDIDATE:,} per candidate", flush=True)
    print(f"  parallel envs:    {EUREKA_N_ENVS}", flush=True)
    print(f"  eval episodes:    {N_EVAL_EPISODES}", flush=True)
    print(f"  llm model:        {GROQ_MODEL}", flush=True)
    print(f"  fitness weights:  {FITNESS_WEIGHTS}", flush=True)
    print(f"  log file:         {LOG_PATH}", flush=True)


def main():
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    _print_banner()

    best = None  # dict: code, metrics, fitness, module_path, checkpoint
    full_log = []
    run_start = time.time()

    for generation in range(N_GENERATIONS):
        gen_start = time.time()
        print(f"\n=== Generation {generation} ===", flush=True)

        print(f"  [1/4] requesting {K_CANDIDATES} candidates from LLM...", flush=True)
        llm_start = time.time()
        candidates_code = generate_candidates(
            best, k=K_CANDIDATES, generation=generation,
            model=GROQ_MODEL, temperature=LLM_TEMPERATURE,
        )
        print(f"  [1/4] received {len(candidates_code)}/{K_CANDIDATES} candidates "
              f"in {time.time() - llm_start:.1f}s", flush=True)

        if not candidates_code:
            print("  no candidates returned from LLM this generation - skipping", flush=True)
            full_log.append({"generation": generation, "results": []})
            continue

        generation_results = []

        for k, code in enumerate(candidates_code):
            cand_start = time.time()
            print(f"\n  --- candidate {k} ({k + 1}/{len(candidates_code)}) ---", flush=True)

            print(f"  [2/4] smoke testing...", flush=True)
            # smoke_test validates the exact `code` string written below; no
            # sanitize-and-diverge — AST + subprocess probe gate what gets saved.
            passed, message = smoke_test(code)
            if not passed:
                print(f"  candidate {k}: REJECTED ({message})", flush=True)
                continue
            print(f"  [2/4] smoke test passed", flush=True)

            module_name = f"gen{generation}_cand{k}"
            module_path = f"eureka.candidates.{module_name}"
            file_path = os.path.join(CANDIDATES_DIR, f"{module_name}.py")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"  saved candidate code -> {file_path}", flush=True)

            print(f"  [3/4] training ({TRAIN_STEPS_PER_CANDIDATE:,} steps)...", flush=True)
            train_start = time.time()
            checkpoint_path = train_candidate(
                module_path,
                total_timesteps=TRAIN_STEPS_PER_CANDIDATE,
                seed=generation * 100 + k,
            )
            print(f"  [3/4] training finished in {time.time() - train_start:.1f}s", flush=True)

            print(f"  [4/4] evaluating ({N_EVAL_EPISODES} episodes)...", flush=True)
            eval_start = time.time()
            metrics = evaluate_candidate(checkpoint_path, module_path, n_episodes=N_EVAL_EPISODES)
            print(f"  [4/4] evaluation finished in {time.time() - eval_start:.1f}s", flush=True)
            fitness = compute_fitness(metrics, FITNESS_WEIGHTS)

            print(f"  candidate {k}: fitness={fitness:.3f} "
                  f"crash_rate={metrics['crash_rate']:.1%} "
                  f"mean_speed={metrics['mean_speed']:.2f} "
                  f"mean_overtakes={metrics['mean_overtakes']:.2f} "
                  f"raw_return={metrics['mean_raw_return']:.2f} "
                  f"(total {time.time() - cand_start:.1f}s)", flush=True)

            generation_results.append({
                "module_path": module_path,
                "code": code,
                "metrics": metrics,
                "fitness": fitness,
                "checkpoint": checkpoint_path,
            })

        full_log.append({"generation": generation, "results": generation_results})

        if not generation_results:
            print("  all candidates were rejected or failed this generation - skipping",
                  flush=True)
            continue

        generation_best = max(generation_results, key=lambda r: r["fitness"])
        print(f"\n  generation {generation} best: fitness={generation_best['fitness']:.3f} "
              f"({generation_best['module_path']})", flush=True)
        print(f"  generation {generation} elapsed: {time.time() - gen_start:.1f}s", flush=True)

        if best is None or generation_best["fitness"] > best["fitness"]:
            best = generation_best
            print(f"  new best overall! fitness={best['fitness']:.3f} "
                  f"(module={best['module_path']})")
        else:
            print(f"  no improvement this generation "
                  f"(best remains fitness={best['fitness']:.3f} from {best['module_path']})")

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(full_log, f, indent=2, default=str)
        print(f"  log updated -> {LOG_PATH}", flush=True)

    print(f"\n=== EUREKA finished (total {time.time() - run_start:.1f}s) ===", flush=True)
    if best is None:
        print("No candidate ever passed smoke testing - nothing to report.")
        return

    print(f"Best fitness:     {best['fitness']:.3f}")
    print(f"Best module:      {best['module_path']}")
    print(f"Best checkpoint:  {best['checkpoint']}")
    print(f"Best metrics:     {best['metrics']}")
    print(f"\nBest code:\n{best['code']}")


if __name__ == "__main__":
    main()
