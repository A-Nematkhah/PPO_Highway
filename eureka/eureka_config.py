"""
eureka_config.py

Hyperparameters for the Phase 4 evolutionary reward design loop
(eureka/loop.py). Kept separate from the main project's config.py since
these control the search process itself, not any single training run.

--------------------------------------------------------------------------
Hardware-scaling knobs (env-var overridable)
--------------------------------------------------------------------------
Everything below that says "env-var overridable" reads an environment
variable first and falls back to a default sized for a modest single
machine. This lets the SAME committed defaults stay safe/fast for CI and
a laptop, while a bigger box (e.g. many-vCPU / no-GPU servers) can opt in
per-run without editing this file:

    # Linux/macOS
    export EUREKA_MAX_CONCURRENT_CANDIDATES=6
    export EUREKA_N_ENVS=16
    python -m eureka.loop

    # Windows PowerShell
    $env:EUREKA_MAX_CONCURRENT_CANDIDATES = "6"
    $env:EUREKA_N_ENVS = "16"
    python -m eureka.loop

MAX_CONCURRENT_CANDIDATES defaults to 1 (fully sequential) on purpose:
the unit/integration test suite monkeypatches
eureka.loop.train_candidate / evaluate_candidate / smoke_test directly,
which only works when loop.py calls those symbols in-process. The
parallel path (MAX_CONCURRENT_CANDIDATES > 1) dispatches real work to
separate OS processes via ProcessPoolExecutor, which re-imports
eureka.train_candidate fresh in each worker and therefore does NOT see
monkeypatches applied in the parent test process. Keeping the default at
1 means `pytest` always exercises the sequential (patchable) code path,
and nothing breaks until a human explicitly opts into parallel execution
for a real, unmocked training run.
"""

import os

# --- evolutionary search ---
N_GENERATIONS = 3          # how many rounds of candidate generation + training
K_CANDIDATES = 8
                            # how many reward candidates the LLM proposes per generation.
                            # Bumped from 4 -> 8: with only 4 children per generation,
                            # a 3-generation run empirically failed to ever improve on
                            # the generation-0 archive (see run log analysis). A wider
                            # generation gives Pareto selection more material to work
                            # with per round.

# --- per-candidate training budget ---
TRAIN_STEPS_PER_CANDIDATE = 75000
                            # Raised from 50_000 -> 150_000. Confirmation runs on a
                            # many-vCPU box showed the SAME unmodified candidate
                            # swinging from mean_overtakes=1.53 -> 0.80 -> 1.53 and
                            # crash_rate=0.17 -> 0.27 purely from changing the training
                            # seed. That's measurement noise, not signal, and it was
                            # large enough to make epsilon-dominance decisions
                            # essentially arbitrary. More steps per candidate reduces
                            # this variance; affordable now that concurrent candidate
                            # training (below) no longer makes this a purely serial
                            # wall-clock cost.
EUREKA_N_ENVS = 6
                            # Raised from 4 -> 16. Bottleneck for a tiny 256x256 MLP
                            # on CPU is environment-stepping throughput, not matrix
                            # multiplication - more parallel envs per candidate directly
                            # buys more rollout throughput. Tune down on smaller
                            # machines via the EUREKA_N_ENVS env var.

# Per-step wall-clock cap on shaping_reward() during training/eval (seconds).
# Catches candidates that hang only after many calls (invisible to short smoke probes).
SHAPING_FN_TIMEOUT_S = 0.05

# Worker threads for the shared shaping_reward() ThreadPoolExecutor in
# shaping_call.py. Larger pools tolerate more leaked (timed-out) threads
# before executor replacement is required.
SHAPING_FN_EXECUTOR_WORKERS = 8

# --------------------------------------------------------------------------- #
# Parallel candidate execution (new)
# --------------------------------------------------------------------------- #
# How many candidates eureka/loop.py trains+evaluates SIMULTANEOUSLY, each in
# its own OS process (ProcessPoolExecutor, spawn context). 1 = fully
# sequential, identical control flow to the original implementation (and what
# the test suite exercises). See the module docstring above for why the
# default is intentionally 1.
#
# Sizing guidance for an N-vCPU, no-GPU machine:
#   usable_cores = N_vcpu - 4                      (reserve for OS/orchestrator)
#   MAX_CONCURRENT_CANDIDATES ~= usable_cores // (EUREKA_N_ENVS + TORCH_THREADS_PER_WORKER)
# e.g. 124 vCPU, EUREKA_N_ENVS=16, TORCH_THREADS_PER_WORKER=2:
#   (124 - 4) // (16 + 2) = 6 concurrent candidates, ~96 env worker processes
#   + 6 training processes + orchestrator, comfortably under 124.
MAX_CONCURRENT_CANDIDATES = 1

# Threads PyTorch is allowed to use PER candidate worker process. Left at
# PyTorch's default (which greedily grabs every visible core), N concurrent
# candidate processes will each try to use ALL cores, causing severe
# oversubscription/contention that makes parallel execution SLOWER than
# sequential. Explicit pinning (set in the worker before importing torch)
# avoids this. 2 is plenty for this project's tiny 256x256 MLP.
TORCH_THREADS_PER_WORKER = 1

# --- multi-objective selection ---
# Default is "pareto": the bounded epsilon/NSGA-II-lite archive is authoritative
# for survivor selection and LLM reflection elites (select_reflection_elites).
# "shadow" remains available for diagnostics: it still computes Pareto metadata
# but routes selection/reflection through the legacy scalar `best` instead.
MULTI_OBJECTIVE_MODE = "pareto"

# Fixed directions, practical-resolution epsilons, and domain bounds. Bounds
# are used only for the unweighted knee representative, never for dominance.
OBJECTIVE_SPECS = (
    {
        "metric": "crash_rate",
        "direction": "min",
        "epsilon": 0.10,
        "bounds": (0.0, 1.0),
    },
    {
        "metric": "mean_speed",
        "direction": "max",
        "epsilon": 0.50,
        "bounds": (0.0, 40.0),
    },
    {
        "metric": "mean_overtakes",
        "direction": "max",
        "epsilon": 0.25,
        "bounds": (0.0, 10.0),
    },
)
PARETO_ARCHIVE_SIZE = 6
REFLECTION_ELITES = 3

# Retrain Pareto rank-0 finalists on independent seeds before reporting the
# final archive. Each seed adds a full PPO train/eval; now that confirmation
# runs concurrently (see loop.py) too, more seeds cost wall-clock much less
# than they used to, and directly attack the seed-noise problem documented
# above (TRAIN_STEPS_PER_CANDIDATE comment).
CONFIRMATION_SEEDS = (10000, 20000)

# --- legacy scalar fitness (diagnostic only; does not drive selection in "pareto") ---
# fitness = -FITNESS_WEIGHTS["crash"] * crash_rate
#           + FITNESS_WEIGHTS["speed"] * mean_speed
#           + FITNESS_WEIGHTS["overtakes"] * mean_overtakes
# crash_rate is a fraction in [0, 1], mean_speed is in m/s (~15-30),
# mean_overtakes is a small integer per episode (~0-5) - weights are scaled
# accordingly so no single term dominates by unit magnitude alone.
FITNESS_WEIGHTS = {
    "crash": 1.0,
    "speed": 0.05,
    "overtakes": 0.3,
}

N_EVAL_EPISODES = 50
                            # Raised from 30 -> 50 for finer crash_rate quantization
                            # (1/50 = 2% steps vs 1/30 = 3.3%); eval is comparatively
                            # cheap so this is affordable on the new hardware.

# If True, generation 0 includes one extra candidate (in addition to the
# K LLM-generated ones) seeded from the hand-written baseline reward in
# eureka/human_seed.py. This candidate is smoke-tested, trained, and
# evaluated identically to LLM candidates; its real metrics then flow
# into the Pareto archive and generation-1 reflection context, following
# the "EUREKA from Human Initialization" technique (Ma et al., ICLR 2024,
# Sec 4.4). Adds exactly one extra train+eval run, only in generation 0.
SEED_GENERATION_0_WITH_HUMAN_REWARD = True


def candidate_base_seed(generation: int, k: int) -> int:
    """
    Base seed for one candidate's vector env. Each candidate in a generation
    owns a disjoint block of EUREKA_N_ENVS consecutive seeds so sibling
    candidates never share RNG state across their parallel sub-envs.
    Deterministic by (generation, k) regardless of execution order, so this
    stays correct whether candidates train sequentially or concurrently.

    --------------------------------------------------------------------
    P0 fix: per-generation stride must reserve K_CANDIDATES + 1 blocks
    --------------------------------------------------------------------
    Previously the stride was `K_CANDIDATES * EUREKA_N_ENVS`. But when
    SEED_GENERATION_0_WITH_HUMAN_REWARD is True, generation 0 actually
    trains K_CANDIDATES LLM candidates PLUS one extra human-seed candidate
    at slot k == K_CANDIDATES (prepended in loop.py). That extra slot's
    seed block was never reserved by the old stride, so it landed exactly
    on top of the NEXT generation's k == 0 block:

        old_seed(generation=0, k=K_CANDIDATES)  == old_seed(generation=1, k=0)

    i.e. generation 0's human-seed candidate and generation 1's first LLM
    candidate trained on byte-for-byte identical stochastic rollouts -
    two structurally different reward functions sharing RNG state, which
    corrupts the seed-independence the search's reflection signal depends
    on (confirmed by test_seeds.py::test_human_seed_slot_does_not_collide_
    with_next_generation, which was failing under the old formula).

    The stride now always reserves K_CANDIDATES + 1 seed blocks per
    generation, whether or not the human-seed slot is actually used that
    generation. This keeps the formula correct and fully order-independent
    without needing to reference SEED_GENERATION_0_WITH_HUMAN_REWARD here,
    at the minor cost of one permanently-unused seed block per generation
    after generation 0.
    """
    slots_per_generation = K_CANDIDATES + 1
    return generation * slots_per_generation * EUREKA_N_ENVS + k * EUREKA_N_ENVS


# --- LLM ---
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.1-70b-versatile / llama-3.3-70b-versatile
                                     # were deprecated by Groq (June 2026); this is
                                     # their official recommended replacement for
                                     # code/reasoning-heavy tasks like this one
LLM_TEMPERATURE = 0.9      # higher than the Phase 1 judge - we want diverse candidates