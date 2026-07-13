"""
eureka_config.py

Hyperparameters for the Phase 4 evolutionary reward design loop
(eureka/loop.py). Kept separate from the main project's config.py since
these control the search process itself, not any single training run.
"""

# --- evolutionary search ---
N_GENERATIONS = 3          # how many rounds of candidate generation + training
K_CANDIDATES = 4           # how many reward candidates the LLM proposes per generation
                            # (EUREKA's paper uses 16 - kept small here for feasibility
                            # on a single machine; increase if you have time/compute)

# --- per-candidate training budget ---
TRAIN_STEPS_PER_CANDIDATE = 50_000
EUREKA_N_ENVS = 4          # fewer parallel envs than main training (config.N_ENVS=6),
                            # since we're running many short trainings back-to-back

# Per-step wall-clock cap on shaping_reward() during training/eval (seconds).
# Catches candidates that hang only after many calls (invisible to short smoke probes).
SHAPING_FN_TIMEOUT_S = 0.05

# --- multi-objective selection ---
# "shadow": compute/log Pareto metadata but preserve legacy scalar selection.
# "pareto": bounded epsilon/NSGA-II-lite archive is authoritative.
MULTI_OBJECTIVE_MODE = "shadow"

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

# Optional confirmation is deliberately off by default because each seed adds
# a full PPO train/eval. Populate (for example, (10000, 20000)) after shadow
# analysis to re-check archive finalists before reporting a final front.
CONFIRMATION_SEEDS = ()

# --- legacy scalar fitness (shadow comparison only) ---
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

N_EVAL_EPISODES = 10       # deterministic evaluation episodes used to compute objectives


def candidate_base_seed(generation: int, k: int) -> int:
    """
    Base seed for one candidate's vector env. Each candidate in a generation
    owns a disjoint block of EUREKA_N_ENVS consecutive seeds so sibling
    candidates never share RNG state across their parallel sub-envs.
    """
    return generation * K_CANDIDATES * EUREKA_N_ENVS + k * EUREKA_N_ENVS


# --- LLM ---
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.1-70b-versatile / llama-3.3-70b-versatile
                                     # were deprecated by Groq (June 2026); this is
                                     # their official recommended replacement for
                                     # code/reasoning-heavy tasks like this one
LLM_TEMPERATURE = 0.9      # higher than the Phase 1 judge - we want diverse candidates
