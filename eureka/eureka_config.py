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

# --- fitness function ---
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

N_EVAL_EPISODES = 10       # deterministic evaluation episodes used to compute fitness

# --- LLM ---
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.1-70b-versatile / llama-3.3-70b-versatile
                                     # were deprecated by Groq (June 2026); this is
                                     # their official recommended replacement for
                                     # code/reasoning-heavy tasks like this one
LLM_TEMPERATURE = 0.9      # higher than the Phase 1 judge - we want diverse candidates
