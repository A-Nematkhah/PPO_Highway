"""Unit tests for eureka/reflection.py."""

from eureka.reflection import build_reflection


def test_build_reflection_best_none_returns_initial_prompt():
    prompt = build_reflection(None)
    assert "initial reward shaping" in prompt.lower()
    assert "overtaking" in prompt.lower()
    assert "```" not in prompt


def test_build_reflection_includes_code_block_and_metrics():
    best = {
        "code": "def shaping_reward(ego, road, info):\n    return 0.1\n",
        "fitness": 1.234,
        "metrics": {
            "crash_rate": 0.25,
            "mean_speed": 22.5,
            "mean_overtakes": 3.0,
            "mean_raw_return": 0.8,
        },
    }
    prompt = build_reflection(best)
    assert "```python" in prompt
    assert best["code"].strip() in prompt
    assert "crash_rate: 25.00%" in prompt
    assert "mean_speed: 22.50 m/s" in prompt
    assert "mean_overtakes: 3.00 per episode" in prompt
    assert "legacy scalar score (diagnostic only): 1.234" in prompt
    assert "IMPROVED" in prompt


def test_build_reflection_describes_multiple_pareto_elites_and_target():
    elites = [
        {
            "code": "def shaping_reward(ego, road, info):\n    return 0.1\n",
            "metrics": {
                "crash_rate": 0.0,
                "mean_speed": 20.0,
                "mean_overtakes": 1.0,
            },
            "pareto_rank": 0,
            "reflection_role": "safest",
        },
        {
            "code": "def shaping_reward(ego, road, info):\n    return 0.2\n",
            "metrics": {
                "crash_rate": 0.1,
                "mean_speed": 30.0,
                "mean_overtakes": 3.0,
            },
            "pareto_rank": 0,
            "reflection_role": "fastest_safe",
        },
    ]
    prompt = build_reflection(elites, target_role="fastest_safe")
    assert "non-dominated trade-off elites" in prompt
    assert "Candidate 1 (safest)" in prompt
    assert "Candidate 2 (fastest_safe)" in prompt
    assert "preserve its speed" in prompt


def test_build_reflection_includes_component_means_when_present():
    best = {
        "code": "def shaping_reward(ego, road, info):\n    return 0.1, {}\n",
        "fitness": 1.0,
        "metrics": {
            "crash_rate": 0.1,
            "mean_speed": 20.0,
            "mean_overtakes": 1.0,
            "component_means": {
                "ttc_penalty": -0.05,
                "overtake_bonus": 0.12,
            },
        },
        "component_history": {
            "ttc_penalty": [-0.08, -0.06, -0.05],
            "overtake_bonus": [0.05, 0.10, 0.12],
        },
    }
    prompt = build_reflection(best)
    assert "Reward component means over this evaluation:" in prompt
    assert "ttc_penalty: -0.0500" in prompt
    assert "overtake_bonus: 0.1200" in prompt
    assert "Reward component values at checkpoints during training" in prompt
    assert "ttc_penalty: [-0.08, -0.06, -0.05]" in prompt


def test_build_reflection_omits_component_section_when_absent():
    best = {
        "code": "def shaping_reward(ego, road, info):\n    return 0.1\n",
        "fitness": 1.234,
        "metrics": {
            "crash_rate": 0.25,
            "mean_speed": 22.5,
            "mean_overtakes": 3.0,
        },
    }
    prompt = build_reflection(best)
    assert "Reward component means" not in prompt
    assert "Reward component values at checkpoints" not in prompt
    assert "crash_rate: 25.00%" in prompt
    assert "mean_speed: 22.50 m/s" in prompt
    assert "mean_overtakes: 3.00 per episode" in prompt