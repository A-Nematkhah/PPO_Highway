"""
human_seed.py

Human-authored baseline reward, ported from reward_wrapper.py's TTC
penalty + overtake bonus, used only as a generation-0 seed candidate.

Follows the EUREKA paper's "Human Initialization" experiment (Ma et al.,
ICLR 2024, Section 4.4): seeding the evolutionary search with a human
reward function uniformly improves final results over both human-only and
EUREKA-only baselines. The seed participates in the same smoke/train/eval
pipeline as LLM candidates; only generation 0 receives this extra slot.
"""

# Literal source text, smoke-tested and sandboxed like LLM output.
HUMAN_SEED_CODE = """
def shaping_reward(ego, road, info):
    ttc_threshold = 3.0
    ttc_weight = 0.1
    overtake_bonus = 0.2

    min_ttc = 1e9
    for vehicle in road.vehicles:
        if vehicle is ego:
            continue
        if vehicle.lane_index[2] != ego.lane_index[2]:
            continue
        dx = vehicle.position[0] - ego.position[0]
        if dx <= 0:
            continue
        closing_speed = ego.speed - vehicle.speed
        if closing_speed <= 0:
            continue
        ttc = dx / closing_speed
        if ttc < min_ttc:
            min_ttc = ttc

    ttc_penalty = 0.0
    if min_ttc < ttc_threshold:
        severity = (ttc_threshold - min_ttc) / ttc_threshold
        ttc_penalty = -ttc_weight * severity

    n_overtakes = info.get("n_overtakes", 0)
    overtake_reward = overtake_bonus * float(n_overtakes)
    return ttc_penalty + overtake_reward
""".strip()
