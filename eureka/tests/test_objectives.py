"""Unit tests for epsilon/Pareto/NSGA-II-lite objective selection."""

import math

import pytest

from eureka.eureka_config import OBJECTIVE_SPECS
from eureka.objectives import (
    annotate_population,
    crowding_distance,
    dominates,
    epsilon_box,
    nondominated_sort,
    select_reflection_elites,
    select_representative,
    update_archive,
    validate_metrics,
)


def _candidate(name, crash, speed, overtakes):
    return {
        "module_path": name,
        "code": (
            f"def shaping_reward(ego, road, info):\n"
            f"    # candidate: {name}\n"
            f"    return {len(name) / 100}\n"
        ),
        "metrics": {
            "crash_rate": crash,
            "mean_speed": speed,
            "mean_overtakes": overtakes,
            "mean_raw_return": 0.0,
        },
    }


def test_mixed_direction_dominance():
    stronger = _candidate("stronger", 0.1, 25.0, 3.0)
    weaker = _candidate("weaker", 0.3, 20.0, 1.0)
    assert dominates(stronger, weaker, OBJECTIVE_SPECS)
    assert not dominates(weaker, stronger, OBJECTIVE_SPECS)


def test_epsilon_box_ignores_sub_resolution_difference():
    a = _candidate("a", 0.11, 20.10, 1.01)
    b = _candidate("b", 0.19, 20.40, 1.20)
    assert epsilon_box(a["metrics"], OBJECTIVE_SPECS) == epsilon_box(
        b["metrics"], OBJECTIVE_SPECS
    )
    assert not dominates(a, b, OBJECTIVE_SPECS)
    assert not dominates(b, a, OBJECTIVE_SPECS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metric_rejected(value):
    candidate = _candidate("bad", 0.1, value, 1.0)
    with pytest.raises(ValueError, match="finite"):
        validate_metrics(candidate["metrics"], OBJECTIVE_SPECS)


def test_nondominated_sort_builds_multiple_fronts():
    population = [
        _candidate("safe", 0.0, 20.0, 1.0),
        _candidate("fast", 0.2, 30.0, 2.0),
        _candidate("dominated", 0.5, 10.0, 0.0),
    ]
    fronts = nondominated_sort(population, OBJECTIVE_SPECS)
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]


def test_sort_is_input_order_invariant_by_candidate_identity():
    candidates = [
        _candidate("a", 0.0, 20.0, 1.0),
        _candidate("b", 0.2, 30.0, 2.0),
        _candidate("c", 0.6, 10.0, 0.0),
    ]
    first = nondominated_sort(candidates, OBJECTIVE_SPECS)
    reverse_population = list(reversed(candidates))
    second = nondominated_sort(reverse_population, OBJECTIVE_SPECS)
    first_names = [{candidates[i]["module_path"] for i in front} for front in first]
    second_names = [
        {reverse_population[i]["module_path"] for i in front} for front in second
    ]
    assert first_names == second_names


def test_crowding_small_front_and_constant_objective():
    population = [
        _candidate("a", 0.1, 20.0, 1.0),
        _candidate("b", 0.1, 20.0, 2.0),
    ]
    distances = crowding_distance(population, [0, 1], OBJECTIVE_SPECS)
    assert all(math.isinf(value) for value in distances.values())


def test_archive_deduplicates_code_and_epsilon_boxes_and_caps_capacity():
    a = _candidate("a", 0.11, 20.1, 1.01)
    duplicate_box = _candidate("box_duplicate", 0.19, 20.4, 1.2)
    code_duplicate = dict(a, module_path="a_second")
    b = _candidate("b", 0.3, 30.0, 2.0)
    archive = update_archive(
        [],
        [a, duplicate_box, code_duplicate, b],
        OBJECTIVE_SPECS,
        capacity=2,
    )
    assert len(archive) == 2
    assert len({tuple(item["epsilon_box"]) for item in archive}) == 2
    assert all("pareto_rank" in item for item in archive)


def test_representative_and_reflection_elites_are_deterministic():
    archive = [
        _candidate("safe", 0.0, 18.0, 1.0),
        _candidate("fast", 0.1, 30.0, 2.0),
        _candidate("overtake", 0.1, 24.0, 5.0),
    ]
    annotate_population(archive, OBJECTIVE_SPECS)
    representative = select_representative(archive, OBJECTIVE_SPECS)
    elites = select_reflection_elites(archive, OBJECTIVE_SPECS, count=3)
    assert representative in archive
    assert len({elite["candidate_id"] for elite in elites}) == len(elites)
    assert elites[0]["reflection_role"] == "balanced_knee"


def test_improving_all_objectives_cannot_worsen_rank():
    base = _candidate("base", 0.3, 20.0, 1.0)
    improved = _candidate("improved", 0.1, 25.0, 2.0)
    population = [base, improved]
    annotate_population(population, OBJECTIVE_SPECS)
    assert improved["pareto_rank"] <= base["pareto_rank"]
