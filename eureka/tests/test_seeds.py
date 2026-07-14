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


def test_human_seed_slot_does_not_collide_with_next_generation():
    """
    Regression test for the P0 seed-collision bug: when
    SEED_GENERATION_0_WITH_HUMAN_REWARD is True, generation 0 trains
    K_CANDIDATES + 1 candidates (human seed prepended at index 0, LLM
    candidates at k=1..K_CANDIDATES). With the old stride
    (K_CANDIDATES * EUREKA_N_ENVS), the last generation-0 slot
    (k == K_CANDIDATES) computed the exact same base seed as generation
    1's first candidate (k == 0), so two structurally different
    candidates trained on identical stochastic rollouts.
    """
    human_seed_slot = _seed_range(0, K_CANDIDATES)
    gen1_first = _seed_range(1, 0)
    assert human_seed_slot.isdisjoint(gen1_first)


def test_human_seed_slot_does_not_collide_with_other_gen0_candidates():
    ranges = [_seed_range(0, k) for k in range(K_CANDIDATES + 1)]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            assert ranges[i].isdisjoint(ranges[j])
