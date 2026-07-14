"""
config.py

All hyperparameters in one place. Extend as networks.py / buffer.py / ppo.py
get written.
"""

ENV_ID = "highway-fast-v0"
SEED = 0

# vectorization
N_ENVS = 6

# --------------------------------------------------------------------------- #
# Environment-specific config (passed to env.unwrapped.configure())
#
# These are highway-env's own parameters (scene, reward shaping, observation
# type, action type, simulation timing) - NOT PPO hyperparameters.
#
# Values below match highway-fast-v0's own defaults; kept explicit here so
# you can see and tweak every knob in one place instead of relying on
# hidden library defaults.
# --------------------------------------------------------------------------- #
ENV_CONFIG = {
    # --- observation ---
    "observation": {
        "type": "Kinematics",          # (presence, x, y, vx, vy) per nearby vehicle
        "vehicles_count": 15,          # how many nearby vehicles are observed
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,             # positions relative to ego vehicle
        "normalize": True,
        "order": "sorted",             # sort observed vehicles by distance
    },

    # --- action ---
    "action": {
        "type": "DiscreteMetaAction",  # {LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}
    },

    # --- scene / traffic ---
    "lanes_count": 4,
    "vehicles_count": 50,              # total vehicles in the scene
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "ego_spacing": 2,
    "vehicles_density": 1,

    # --- episode / timing ---
    "duration": 30,                    # episode length in seconds
    "simulation_frequency": 15,        # physics steps per second (Hz)
    "policy_frequency": 5,             # agent decisions per second (Hz)

    # --- reward shaping ---
    "collision_reward": -3,            # reduced from -5: -5 combined with gamma=0.95 made
                                        # the agent overly risk-averse (crawled slowly, never
                                        # overtook). -3 still penalizes crashes much more than
                                        # the original -1, but leaves room for high_speed_reward
                                        # to make overtaking worthwhile.
    "right_lane_reward": 0.1,          # bonus for driving on the rightmost lane
    "high_speed_reward": 0.7,          # increased from 0.4: speed needs to be worth the risk
                                        # relative to collision_reward, or the policy just
                                        # learns to drive slowly and never overtake.
    "lane_change_reward": 0,           # extra penalty per lane change (0 = none)
    "reward_speed_range": [20, 30],    # m/s range mapped to [0, 1] for high_speed_reward
    "normalize_reward": True,          # squash total reward roughly into [0, 1]

    # --- rendering (irrelevant for headless training, kept for completeness) ---
    "offscreen_rendering": True,
    "show_trajectories": False,
    "render_agent": False,  # no visual rendering during training -> saves compute
}

# Shared PPO roll-step hyperparameters (consumed by eureka/train_candidate.py
# and the library modules networks.py / buffer.py / ppo.py).
N_STEPS = 128
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.95  # increased from 0.8: gives the agent a longer effective planning
              # horizon (~20 steps vs ~5), important for anticipating collisions
              # further ahead rather than only reacting to immediate risk
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
LR = 5e-4
ENT_COEF = 0.01  # increased from 0.0: keeps a small amount of exploration pressure
                 # alive throughout training so the policy doesn't collapse early
                 # into one overly-cautious behavior (e.g. "always slow down")
                 # without ever trying riskier-but-better maneuvers like overtaking
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NET_ARCH = [256, 256]
TOTAL_TIMESTEPS = 200_000

# --------------------------------------------------------------------------- #
# Custom reward shaping (reward_wrapper.py) - applied ON TOP of the built-in
# highway-env reward defined in ENV_CONFIG above.
# --------------------------------------------------------------------------- #
TTC_THRESHOLD = 3.0    # seconds - below this, the TTC penalty starts applying
TTC_WEIGHT = 0.1       # max penalty magnitude as TTC -> 0
OVERTAKE_BONUS = 0.2   # reward per vehicle overtaken in a step

# --------------------------------------------------------------------------- #
# ORPHAN / BROKEN PATH — do not enable without restoring llm_judge.py
#
# Historically (Phase 1), an LLM judged finished episodes and returned a 0/1
# score added to the terminal reward. The entry module `llm_judge.py` was
# removed in the EUREKA-only cleanup, but these flags and the import site in
# env_utils._EnvFactory remain. Setting USE_LLM_JUDGE=True currently raises
# ImportError. Leave False. EUREKA LLM calls use eureka_config.GROQ_MODEL
# via key_manager, not this block.
# --------------------------------------------------------------------------- #
USE_LLM_JUDGE = False
LLM_JUDGE_WEIGHT = 0.5            # weight applied to the judge's 0/1 score
LLM_JUDGE_EVERY_N_EPISODES = 5    # only judge every Nth finished episode (per env)
GROQ_MODEL = "openai/gpt-oss-20b"  # unused unless USE_LLM_JUDGE is somehow restored
