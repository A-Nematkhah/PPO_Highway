"""Regression tests for component-history sidecar lifecycle."""

import json
import os

from eureka.train_candidate import (
    _remove_stale_component_sidecar,
    component_sidecar_path,
)


_CANDIDATE = "def shaping_reward(ego, road, info):\n    return 0.0\n"


def test_stale_component_sidecar_is_removed_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = component_sidecar_path("gen0_cand0")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"component_history": {"stale": [1.0]}}, f)

    _remove_stale_component_sidecar("gen0_cand0")

    assert not os.path.exists(path)


def test_corrupt_component_sidecar_does_not_abort_loop(tmp_path, monkeypatch):
    import eureka.loop as loop

    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)
    os.makedirs("eureka/checkpoints", exist_ok=True)

    monkeypatch.setattr(loop, "N_GENERATIONS", 1)
    monkeypatch.setattr(loop, "K_CANDIDATES", 1)
    monkeypatch.setattr(loop, "MULTI_OBJECTIVE_MODE", "shadow")
    monkeypatch.setattr(loop, "CONFIRMATION_SEEDS", ())
    monkeypatch.setattr(loop, "SEED_GENERATION_0_WITH_HUMAN_REWARD", False)
    monkeypatch.setattr(loop, "LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr(
        loop,
        "generate_candidates",
        lambda context, k, generation, model, temperature: [_CANDIDATE],
    )
    monkeypatch.setattr(loop, "smoke_test", lambda code: (True, "ok"))

    def fake_train(module_path, total_timesteps, seed):
        path = component_sidecar_path(module_path.split(".")[-1])
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        return f"{module_path}.pt"

    monkeypatch.setattr(loop, "train_candidate", fake_train)
    monkeypatch.setattr(
        loop,
        "evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: {
            "crash_rate": 0.0,
            "mean_speed": 20.0,
            "mean_overtakes": 0.0,
            "mean_raw_return": 1.0,
            "component_means": {},
        },
    )

    loop.main()

    data = json.loads(
        (tmp_path / "eureka" / "eureka_log.json").read_text(encoding="utf-8")
    )
    result = data[0]["results"][0]
    assert "component_history" not in result
