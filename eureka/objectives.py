"""
objectives.py

Multi-objective selection utilities for EUREKA's safety/speed/overtaking
trade-off. This is deliberately an NSGA-II-lite environmental selector:
the LLM supplies variation, while this module owns epsilon dominance,
nondominated sorting, crowding diversity, and a bounded elite archive.

All ordering is deterministic. That matters because PPO evaluations are
expensive: rerunning the same recorded metrics must produce the same archive.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable


def candidate_id(candidate: dict) -> str:
    """Stable identity used for deduplication and final tie-breaking."""
    code = candidate.get("code", "")
    if code:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    return str(candidate.get("module_path", ""))


def validate_metrics(metrics: dict, specs: tuple[dict, ...]) -> None:
    """Reject missing/non-finite objective values before Pareto ranking."""
    for spec in specs:
        key = spec["metric"]
        if key not in metrics:
            raise ValueError(f"missing objective metric: {key}")
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"objective metric {key!r} must be finite, got {value!r}")


def objective_vector(metrics: dict, specs: tuple[dict, ...]) -> tuple[float, ...]:
    """Raw objective values in configured order (directions remain explicit)."""
    validate_metrics(metrics, specs)
    return tuple(float(metrics[spec["metric"]]) for spec in specs)


def epsilon_box(metrics: dict, specs: tuple[dict, ...]) -> tuple[int, ...]:
    """
    Quantize objectives into transitive epsilon boxes, all in minimization
    orientation. Differences smaller than practical measurement resolution
    therefore cannot create a spurious dominance edge.
    """
    values = objective_vector(metrics, specs)
    coordinates = []
    for value, spec in zip(values, specs):
        oriented = value if spec["direction"] == "min" else -value
        coordinates.append(math.floor(oriented / float(spec["epsilon"])))
    return tuple(coordinates)


def dominates(a: dict, b: dict, specs: tuple[dict, ...]) -> bool:
    """Return True when candidate a epsilon-dominates candidate b."""
    box_a = epsilon_box(a["metrics"], specs)
    box_b = epsilon_box(b["metrics"], specs)
    return all(x <= y for x, y in zip(box_a, box_b)) and any(
        x < y for x, y in zip(box_a, box_b)
    )


def nondominated_sort(population: list[dict], specs: tuple[dict, ...]) -> list[list[int]]:
    """
    O(N^2) nondominated sorting. N is intentionally small (archive + K),
    making this implementation easier to audit than canonical NSGA-II code.
    """
    for candidate in population:
        validate_metrics(candidate["metrics"], specs)

    dominates_set = [set() for _ in population]
    dominated_count = [0 for _ in population]
    fronts: list[list[int]] = [[]]

    for p in range(len(population)):
        for q in range(len(population)):
            if p == q:
                continue
            if dominates(population[p], population[q], specs):
                dominates_set[p].add(q)
            elif dominates(population[q], population[p], specs):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    rank = 0
    while rank < len(fronts) and fronts[rank]:
        next_front = []
        for p in fronts[rank]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(sorted(next_front, key=lambda i: candidate_id(population[i])))
        rank += 1

    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(
    population: list[dict],
    front: Iterable[int],
    specs: tuple[dict, ...],
) -> dict[int, float]:
    """
    Standard NSGA-II crowding distance with per-front normalization.
    Constant objectives contribute zero; fronts of size <=2 preserve both
    boundary candidates with infinite distance.
    """
    indices = list(front)
    distances = {index: 0.0 for index in indices}
    if len(indices) <= 2:
        return {index: math.inf for index in indices}

    for spec in specs:
        key = spec["metric"]
        ordered = sorted(indices, key=lambda i: (population[i]["metrics"][key], candidate_id(population[i])))
        low = float(population[ordered[0]]["metrics"][key])
        high = float(population[ordered[-1]]["metrics"][key])
        if high == low:
            continue
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        span = high - low
        for position in range(1, len(ordered) - 1):
            if math.isinf(distances[ordered[position]]):
                continue
            previous = float(population[ordered[position - 1]]["metrics"][key])
            following = float(population[ordered[position + 1]]["metrics"][key])
            distances[ordered[position]] += (following - previous) / span
    return distances


def annotate_population(population: list[dict], specs: tuple[dict, ...]) -> list[list[int]]:
    """Attach objective/rank/crowding fields in place and return fronts."""
    fronts = nondominated_sort(population, specs)
    for rank, front in enumerate(fronts):
        distances = crowding_distance(population, front, specs)
        for index in front:
            candidate = population[index]
            candidate["candidate_id"] = candidate_id(candidate)
            candidate["objective_vector"] = list(objective_vector(candidate["metrics"], specs))
            candidate["epsilon_box"] = list(epsilon_box(candidate["metrics"], specs))
            candidate["pareto_rank"] = rank
            candidate["crowding_distance"] = distances[index]
            candidate["on_pareto_front"] = rank == 0
    return fronts


def _selection_key(candidate: dict) -> tuple:
    crowding = candidate.get("crowding_distance", 0.0)
    crowding_key = float("-inf") if math.isinf(crowding) else -float(crowding)
    return (
        int(candidate.get("pareto_rank", 10**9)),
        crowding_key,
        candidate_id(candidate),
    )


def update_archive(
    archive: list[dict],
    newcomers: list[dict],
    specs: tuple[dict, ...],
    capacity: int,
) -> list[dict]:
    """
    Combine, code-deduplicate, epsilon-box-deduplicate, rank, and truncate.
    One deterministic representative per epsilon box avoids retaining metric
    differences below the configured practical resolution.
    """
    if capacity <= 0:
        return []

    by_code: dict[str, dict] = {}
    for source_candidate in archive + newcomers:
        validate_metrics(source_candidate["metrics"], specs)
        candidate = dict(source_candidate)
        cid = candidate_id(source_candidate)
        previous = by_code.get(cid)
        if previous is None or str(candidate.get("module_path", "")) < str(previous.get("module_path", "")):
            by_code[cid] = candidate

    by_box: dict[tuple[int, ...], dict] = {}
    for candidate in sorted(by_code.values(), key=candidate_id):
        box = epsilon_box(candidate["metrics"], specs)
        by_box.setdefault(box, candidate)

    candidates = list(by_box.values())
    fronts = annotate_population(candidates, specs)
    selected: list[dict] = []
    for front in fronts:
        ranked_front = sorted((candidates[i] for i in front), key=_selection_key)
        remaining = capacity - len(selected)
        if remaining <= 0:
            break
        selected.extend(ranked_front[:remaining])
        if len(ranked_front) > remaining:
            break

    annotate_population(selected, specs)
    return sorted(selected, key=_selection_key)


def select_representative(archive: list[dict], specs: tuple[dict, ...]) -> dict | None:
    """
    Pick an unweighted knee-like representative from Pareto rank zero.
    Fixed domain bounds avoid population-dependent normalization drift.
    The entire front remains the primary result; this is only a compatibility
    representative for summaries and single-parent callers.
    """
    front = [candidate for candidate in archive if candidate.get("pareto_rank") == 0]
    if not front:
        return None

    def distance(candidate: dict) -> tuple[float, str]:
        squared = 0.0
        for spec in specs:
            value = float(candidate["metrics"][spec["metric"]])
            lower, upper = map(float, spec["bounds"])
            span = upper - lower
            normalized = (value - lower) / span if span else 0.0
            normalized = min(1.0, max(0.0, normalized))
            loss = normalized if spec["direction"] == "min" else 1.0 - normalized
            squared += loss * loss
        return math.sqrt(squared), candidate_id(candidate)

    return min(front, key=distance)


def select_reflection_elites(
    archive: list[dict],
    specs: tuple[dict, ...],
    count: int,
) -> list[dict]:
    """Select knee/safe/fast/overtake representatives without duplicates."""
    if not archive or count <= 0:
        return []
    front = [candidate for candidate in archive if candidate.get("pareto_rank") == 0] or archive
    knee = select_representative(archive, specs)
    roles: list[tuple[str, dict | None]] = [("balanced_knee", knee)]

    metric_specs = {spec["metric"]: spec for spec in specs}
    safety_epsilon = float(metric_specs["crash_rate"]["epsilon"])
    safest = min(front, key=lambda c: (c["metrics"]["crash_rate"], candidate_id(c)))
    eligible = [
        c for c in front
        if c["metrics"]["crash_rate"] <= safest["metrics"]["crash_rate"] + safety_epsilon
    ]
    fastest = max(eligible, key=lambda c: (c["metrics"]["mean_speed"], candidate_id(c)))
    overtaker = max(eligible, key=lambda c: (c["metrics"]["mean_overtakes"], candidate_id(c)))
    roles.extend([("safest", safest), ("fastest_safe", fastest), ("overtaking_safe", overtaker)])

    selected = []
    seen = set()
    for role, candidate in roles:
        if candidate is None:
            continue
        cid = candidate_id(candidate)
        if cid in seen:
            continue
        item = dict(candidate)
        item["reflection_role"] = role
        selected.append(item)
        seen.add(cid)
        if len(selected) == count:
            return selected

    for candidate in sorted(front, key=_selection_key):
        cid = candidate_id(candidate)
        if cid not in seen:
            item = dict(candidate)
            item["reflection_role"] = "diverse_front"
            selected.append(item)
            seen.add(cid)
        if len(selected) == count:
            break
    return selected
