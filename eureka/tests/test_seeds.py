"""Unit tests for non-overlapping candidate seed assignment."""

from eureka.eureka_config import EUREKA_N_ENVS, K_CANDIDATES, candidate_base_seed


def _seed_range(generation: int, k: int) -> set[int]:
    base = candidate_base_seed(generation, k)
    return set(range(base, base + EUREKA_N_ENVS))


def test_candidate_seeds_do_not_overlap_within_generation():
    generation = 2
    ranges = [_seed_range(generation, k) for k in range(K_CANDIDATES)]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            assert ranges[i].isdisjoint(ranges[j])


def test_candidate_seeds_do_not_overlap_across_generations():
    gen0_last = _seed_range(0, K_CANDIDATES - 1)
    gen1_first = _seed_range(1, 0)
    assert gen0_last.isdisjoint(gen1_first)
