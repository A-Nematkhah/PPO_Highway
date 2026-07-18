"""
End-to-end integration test: mocked LLM, trivial candidate, minimal train budget.

No real Groq API key or GPU required. Exercises generate -> smoke -> train ->
eval -> fitness -> log/metrics write.
"""

import json
import os

import pytest

TRIVIAL_CANDIDATE = """
def shaping_reward(ego, road, info):
    return 0.0
"""


@pytest.mark.integration
def test_loop_end_to_end_with_mocked_llm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)

    monkeypatch.setattr("eureka.loop.N_GENERATIONS", 1)
    monkeypatch.setattr("eureka.loop.K_CANDIDATES", 1)
    monkeypatch.setattr("eureka.loop.TRAIN_STEPS_PER_CANDIDATE", 512)
    monkeypatch.setattr("eureka.loop.N_EVAL_EPISODES", 2)
    monkeypatch.setattr("eureka.loop.MULTI_OBJECTIVE_MODE", "shadow")
    monkeypatch.setattr("eureka.loop.CONFIRMATION_SEEDS", ())
    monkeypatch.setattr("eureka.loop.SEED_GENERATION_0_WITH_HUMAN_REWARD", False)
    monkeypatch.setattr("eureka.loop.LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr(
        "eureka.loop.generate_candidates",
        lambda best, k, generation, model, temperature: [TRIVIAL_CANDIDATE],
    )

    import eureka.env_factory as env_factory

    _orig_make = env_factory.make_candidate_vec_env

    def _sync_only(module_path, n_envs, seed, parallel=True):
        return _orig_make(module_path, n_envs, seed, parallel=False)

    monkeypatch.setattr(env_factory, "make_candidate_vec_env", _sync_only)

    from eureka.loop import main

    main()

    log_path = tmp_path / "eureka" / "eureka_log.json"
    metrics_path = tmp_path / "eureka" / "eureka_metrics.jsonl"
    assert log_path.is_file()

    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(log_data) == 1
    assert len(log_data[0]["results"]) == 1
    result = log_data[0]["results"][0]
    assert "fitness" in result
    assert "metrics" in result
    assert "pareto_rank" in result
    assert "crowding_distance" in result
    assert log_data[0]["selection_mode"] == "shadow"
    assert log_data[0]["archive_size"] == 1
    assert result["metrics"]["crash_rate"] >= 0.0

    assert metrics_path.is_file()
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_names = {row["event"] for row in events}
    assert "llm_generation" in event_names
    assert "smoke_test" in event_names
    assert "train" in event_names
    assert "eval" in event_names
    assert "candidate_complete" in event_names
    assert "run_complete" in event_names


def test_pareto_mode_keeps_tradeoffs_and_uses_unweighted_representative(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)

    candidates = [
        "def shaping_reward(ego, road, info):\n    return 0.1\n",
        "def shaping_reward(ego, road, info):\n    return 0.2\n",
    ]
    metric_rows = iter([
        {
            "crash_rate": 0.1,
            "mean_speed": 30.0,
            "mean_overtakes": 5.0,
            "mean_raw_return": 1.0,
        },
        {
            "crash_rate": 1.0,
            "mean_speed": 40.0,
            "mean_overtakes": 10.0,
            "mean_raw_return": 1.0,
        },
    ])

    monkeypatch.setattr("eureka.loop.N_GENERATIONS", 1)
    monkeypatch.setattr("eureka.loop.K_CANDIDATES", 2)
    monkeypatch.setattr("eureka.loop.MULTI_OBJECTIVE_MODE", "pareto")
    monkeypatch.setattr("eureka.loop.CONFIRMATION_SEEDS", ())
    monkeypatch.setattr("eureka.loop.SEED_GENERATION_0_WITH_HUMAN_REWARD", False)
    monkeypatch.setattr("eureka.loop.LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr(
        "eureka.loop.generate_candidates",
        lambda elites, k, generation, model, temperature: candidates,
    )
    monkeypatch.setattr("eureka.loop.smoke_test", lambda code: (True, "ok"))
    monkeypatch.setattr(
        "eureka.loop.train_candidate",
        lambda module_path, total_timesteps, seed: f"{module_path}.pt",
    )
    monkeypatch.setattr(
        "eureka.loop.evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: next(metric_rows),
    )

    from eureka.loop import main

    main()

    log_data = json.loads(
        (tmp_path / "eureka" / "eureka_log.json").read_text(encoding="utf-8")
    )
    generation = log_data[0]
    assert generation["selection_mode"] == "pareto"
    assert generation["pareto_front_size"] == 2
    assert generation["archive_size"] == 2

    scalar_winner_id = generation["legacy_scalar_winner_id"]
    representative_id = generation["representative_id"]
    assert scalar_winner_id != representative_id
    assert all(result["pareto_rank"] == 0 for result in generation["results"])


def test_screening_second_seed_averages_front_metrics_before_archiving(
    tmp_path, monkeypatch
):
    """
    Regression test for the seed-variance fix: the single candidate that
    reaches this generation's Pareto front must be retrained+reevaluated on
    one extra independent seed, and its final metrics/fitness must reflect
    the AVERAGE of both seeds, not just the first (screening) run.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)

    monkeypatch.setattr("eureka.loop.N_GENERATIONS", 1)
    monkeypatch.setattr("eureka.loop.K_CANDIDATES", 1)
    monkeypatch.setattr("eureka.loop.MULTI_OBJECTIVE_MODE", "shadow")
    monkeypatch.setattr("eureka.loop.CONFIRMATION_SEEDS", ())
    monkeypatch.setattr("eureka.loop.SEED_GENERATION_0_WITH_HUMAN_REWARD", False)
    monkeypatch.setattr("eureka.loop.SCREENING_SECOND_SEED_ENABLED", True)
    monkeypatch.setattr("eureka.loop.LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr(
        "eureka.loop.generate_candidates",
        lambda context, k, generation, model, temperature: [TRIVIAL_CANDIDATE],
    )
    monkeypatch.setattr("eureka.loop.smoke_test", lambda code: (True, "ok"))

    train_calls = []

    def fake_train(module_path, total_timesteps, seed):
        train_calls.append(seed)
        return f"{module_path}_{seed}.pt"

    # First (screening) eval: crash_rate=0.8 (bad luck). Second (second-seed)
    # eval: crash_rate=0.0. Average should be 0.4, not 0.8 or 0.0 alone.
    eval_rows = iter([
        {"crash_rate": 0.8, "mean_speed": 20.0, "mean_overtakes": 1.0, "mean_raw_return": 0.5},
        {"crash_rate": 0.0, "mean_speed": 20.0, "mean_overtakes": 1.0, "mean_raw_return": 0.5},
    ])

    monkeypatch.setattr("eureka.loop.train_candidate", fake_train)
    monkeypatch.setattr(
        "eureka.loop.evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: next(eval_rows),
    )

    from eureka.loop import main

    main()

    log_data = json.loads(
        (tmp_path / "eureka" / "eureka_log.json").read_text(encoding="utf-8")
    )
    result = log_data[0]["results"][0]

    # Second seed was actually used to retrain (two distinct seeds called).
    assert len(train_calls) == 2
    assert train_calls[0] != train_calls[1]

    # Final metrics/fitness reflect the AVERAGE of both seeds.
    assert result["metrics"]["crash_rate"] == pytest.approx(0.4)
    assert "screening_seed_1_metrics" in result
    assert "screening_seed_2_metrics" in result
    assert result["screening_seed_1_metrics"]["crash_rate"] == pytest.approx(0.8)
    assert result["screening_seed_2_metrics"]["crash_rate"] == pytest.approx(0.0)


def test_shadow_mode_preserves_legacy_scalar_reflection_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)
    codes = [
        "def shaping_reward(ego, road, info):\n    return 0.1\n",
        "def shaping_reward(ego, road, info):\n    return 0.2\n",
    ]
    contexts = []
    rows = iter([
        {
            "crash_rate": 0.0,
            "mean_speed": 10.0,
            "mean_overtakes": 0.0,
            "mean_raw_return": 0.0,
        },
        {
            "crash_rate": 0.5,
            "mean_speed": 40.0,
            "mean_overtakes": 5.0,
            "mean_raw_return": 0.0,
        },
        {
            "crash_rate": 0.0,
            "mean_speed": 10.0,
            "mean_overtakes": 0.0,
            "mean_raw_return": 0.0,
        },
        {
            "crash_rate": 0.5,
            "mean_speed": 40.0,
            "mean_overtakes": 5.0,
            "mean_raw_return": 0.0,
        },
    ])

    def generate(context, k, generation, model, temperature):
        contexts.append(context)
        return codes

    monkeypatch.setattr("eureka.loop.N_GENERATIONS", 2)
    monkeypatch.setattr("eureka.loop.K_CANDIDATES", 2)
    monkeypatch.setattr("eureka.loop.MULTI_OBJECTIVE_MODE", "shadow")
    monkeypatch.setattr("eureka.loop.CONFIRMATION_SEEDS", ())
    monkeypatch.setattr("eureka.loop.SEED_GENERATION_0_WITH_HUMAN_REWARD", False)
    monkeypatch.setattr("eureka.loop.LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr("eureka.loop.generate_candidates", generate)
    monkeypatch.setattr("eureka.loop.smoke_test", lambda code: (True, "ok"))
    monkeypatch.setattr(
        "eureka.loop.train_candidate",
        lambda module_path, total_timesteps, seed: f"{module_path}.pt",
    )
    monkeypatch.setattr(
        "eureka.loop.evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: next(rows),
    )

    from eureka.loop import main

    main()
    assert contexts[0] is None
    assert isinstance(contexts[1], dict)
    assert contexts[1]["metrics"]["mean_speed"] == 40.0