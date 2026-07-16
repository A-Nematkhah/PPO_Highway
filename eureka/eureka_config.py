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
TRAIN_STEPS_PER_CANDIDATE = 150000
                            # Raised from 75_000 -> 150_000 (this comment previously
                            # claimed the raise had already happened, but the value
                            # had drifted back down to 75_000 - now corrected).
                            # Confirmation runs showed the SAME unmodified candidate
                            # swinging crash_rate from 8% -> 18% -> 86% and other
                            # metrics just as wildly, purely from changing the
                            # training seed. That's measurement noise, not signal,
                            # and it was large enough to make epsilon-dominance
                            # decisions essentially arbitrary. More steps per
                            # candidate is the single biggest lever to reduce this
                            # variance (at ~2x wall-clock cost per candidate);
                            # affordable now that concurrent candidate training
                            # (below) no longer makes this a purely serial cost.
EUREKA_N_ENVS = 12
                            # Raised from 6 -> 12 (comment previously said 4 -> 16,
                            # but the value had drifted to 6 - now corrected).
                            # Bottleneck for a tiny 256x256 MLP on CPU is
                            # environment-stepping throughput, not matrix
                            # multiplication - more parallel envs per candidate
                            # directly buys more rollout throughput AND more
                            # parallel rollout diversity per PPO update, which
                            # reduces per-update variance independent of total
                            # steps (a second, cheaper lever against the same
                            # seed-noise problem as TRAIN_STEPS_PER_CANDIDATE
                            # above). Tune down on smaller machines via the
                            # EUREKA_N_ENVS env var.

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

# --------------------------------------------------------------------------- #
# Screening-stage second seed (new)
# --------------------------------------------------------------------------- #
# CONFIRMATION_SEEDS only re-checks the FINAL archive's rank-0 candidates at
# the very end of the whole run. That leaves a gap: a candidate that got
# unlucky on its single screening seed within a single generation can be
# evicted from the Pareto archive by update_archive() before confirmation
# ever gets a chance to see it - the seed-noise problem shows up one step
# earlier than CONFIRMATION_SEEDS can catch it.
#
# When enabled, loop.py retrains+reevaluates only THIS generation's Pareto
# front (typically a handful of candidates, not all K_CANDIDATES) on one
# extra independent seed, and averages the two runs into the candidate's
# metrics before update_archive() locks in this generation's archive
# membership. Much cheaper than doubling TRAIN_STEPS_PER_CANDIDATE globally,
# and complements it rather than replacing it.
SCREENING_SECOND_SEED_ENABLED = True
# Offset added to candidate_base_seed(generation, k) for the second seed.
# Chosen far outside any candidate_base_seed(...) stride so it can never
# collide with another candidate's reserved seed block.
SCREENING_SECOND_SEED_OFFSET = 10_000_000

# --------------------------------------------------------------------------- #
# Legacy scalar fitness (P2 fix #6 - rebalanced weighting)
# --------------------------------------------------------------------------- #
# fitness = -FITNESS_WEIGHTS["crash"] * crash_rate
#           + FITNESS_WEIGHTS["speed"] * mean_speed
#           + FITNESS_WEIGHTS["overtakes"] * mean_overtakes
#
# IMPORTANT: with MULTI_OBJECTIVE_MODE="pareto" (the default), this scalar
# does NOT drive survivor selection or LLM reflection elites - that is
# handled entirely by eureka.objectives' epsilon/NSGA-II-lite archive, which
# treats crash_rate/mean_speed/mean_overtakes as independent objectives with
# no arbitrary cross-weighting. `legacy_fitness` is diagnostic-only, surfaced
# in eureka_log.json / plots.py / the console for a *quick* sanity read.
#
# That diagnostic value is still misleading if badly calibrated, though.
# The previous weights (crash=1.0, speed=0.05, overtakes=0.3) let a modest
# speed improvement mask a much worse crash rate: a candidate going from
# crash_rate=0.27 -> 0.52 (nearly doubling its crash rate) while also
# gaining ~5 m/s of speed showed fitness *improving* (1.38 -> 1.76), because
# 5 m/s * 0.05 (=0.25) outweighed 0.25 * 1.0 (=0.25) crash penalty almost
# exactly, before overtakes tipped it further into "looks better."
#
# Rebalanced so that crash_rate dominates the diagnostic score by a wide
# margin, matching how a human would actually read "is this reward function
# safe": the crash weight is raised well above what any plausible speed or
# overtake gain could offset, while speed/overtakes weights are trimmed
# proportionally so the score doesn't blow up in magnitude. Concretely, the
# same example above now nets a LARGE fitness drop (0.25 * 3.0 = 0.75)
# instead of a small net-positive move, even after adding back the ~5 m/s
# speed gain (5 * 0.03 = 0.15).
#
# This is still a diagnostic-only heuristic, not a substitute for the Pareto
# archive - it has no epsilon deadband, no notion of trade-off fronts, and a
# single scalar can never fully represent a 3-objective trade-off. Treat
# `legacy_fitness` in logs as "rough, safety-weighted sanity check", not as
# a ranking to optimize against.
FITNESS_WEIGHTS = {
    "crash": 3.0,
    "speed": 0.03,
    "overtakes": 0.2,
}

# --------------------------------------------------------------------------- #
# Evaluation episode budget (P2 fix #7)
# --------------------------------------------------------------------------- #
# Previously 50. Combined with CONFIRMATION_SEEDS having 2 entries, every
# Pareto-front finalist paid for 1 (screening) + 2 (confirmation) x 50 = 150
# deterministic eval episodes in total - a large, and largely wasted, cost.
# The actual source of instability documented above (TRAIN_STEPS_PER_CANDIDATE
# comment) is PPO training-seed variance, not evaluation-episode sampling
# noise: more eval episodes make a single noisy policy's metrics measured
# more *precisely*, but they do nothing to reduce the variance *between*
# training seeds, which is the real problem. SCREENING_SECOND_SEED_ENABLED
# (above) and CONFIRMATION_SEEDS already spend budget on multi-seed
# averaging, which directly attacks that variance instead. Trading some of
# the eval-episode budget back (50 -> 20) frees up wall-clock without
# reintroducing the crash_rate quantization noise problem that originally
# motivated raising N_EVAL_EPISODES (20 episodes = 5% resolution, still far
# finer than the 10% CONFIRMATION_SEEDS/OBJECTIVE_SPECS epsilon-box width
# for crash_rate, so this does not blur Pareto dominance decisions).
N_EVAL_EPISODES = 20

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