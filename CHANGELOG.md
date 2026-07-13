# Changelog

## Unreleased

### Component-level reward reflection (EUREKA paper Sec 3.3 / Prompt 3)
- shaping_reward() may now optionally return (total: float, components:
  dict[str, float]) instead of a bare float, mirroring the EUREKA
  paper's reward_components output format. Backward compatible: bare
  float returns still work unchanged.
- eureka/sandbox.py: added normalize_shaping_output() to unify both
  return forms with graceful degradation on malformed output.
- eureka/train_candidate.py: accumulates per-component rolling-window
  snapshots during training, written to a checkpoint sidecar JSON file.
- eureka/evaluate_candidate.py: reports component_means alongside
  existing crash_rate/mean_speed/mean_overtakes/mean_raw_return metrics.
- eureka/reflection.py: includes component means and chronological
  training snapshots in the LLM feedback prompt when available, per
  "Reward reflection enables targeted improvement" (paper Sec 4.3,
  reports 28.6% score drop without this granularity).
- No change to training budget, candidate count, or LLM call count.

### Human reward initialization (EUREKA paper Sec 4.4)
- Added `eureka/human_seed.py`: hand-written shaping_reward ported from
  reward_wrapper.py's TTC penalty + overtake bonus, used as an extra
  generation-0 candidate (not counted against K_CANDIDATES) so its real
  trained metrics seed the Pareto archive/reflection context, per
  "EUREKA can improve and benefit from human reward functions" (Ma et
  al., ICLR 2024). Adds one extra train+eval run, generation 0 only.
  Toggle: `SEED_GENERATION_0_WITH_HUMAN_REWARD` in eureka_config.py.

### Multi-objective Pareto / NSGA-II-lite selection
- Added `eureka/objectives.py`: epsilon-box dominance, nondominated sorting,
  normalized crowding distance, deterministic bounded archive, unweighted knee
  representative, and diverse reflection-elite selection.
- Added `MULTI_OBJECTIVE_MODE`: `shadow` logs Pareto metadata while preserving
  legacy scalar selection; `pareto` makes the cross-generation archive authoritative.
- Candidate logs now include objective vectors, epsilon boxes, Pareto rank,
  crowding distance, archive membership, and scalar/Pareto disagreement.
- Reflection now accepts multiple Pareto elites and schedules balanced, safest,
  fastest-safe, and overtaking-safe LLM mutation targets.
- Optional `CONFIRMATION_SEEDS` retrains rank-zero finalists and rebuilds the
  final archive from aggregate metrics.
- Added objective, confirmation, reflection, and mocked Pareto-loop tests.

### EUREKA-only cleanup (phase 1)
- Removed baseline entrypoints: `train.py`, `evaluate.py`, `llm_judge.py`.
- EUREKA pipeline (`python -m eureka.loop`) unchanged; all 12 tests pass.

### Phase 2 — Reliability, observability, test coverage
- **eureka/logging_utils.py**: structured logging (text or JSON via `EUREKA_LOG_JSON=1`).
- **eureka/telemetry.py**: `eureka_metrics.jsonl` timing/events for plotting.
- **loop.py**, **llm_reward_designer.py**, **train_candidate.py**, **evaluate_candidate.py**:
  migrated from `print` to structured logger + telemetry.
- **Unit tests**: fitness, reflection, `_extract_code`, evaluate_candidate (mocked),
  logging/telemetry.
- **Integration test**: `test_integration_loop.py` — mocked LLM, 512 train steps.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) + `.coveragerc` (80% on non-loop modules).

### Phase 1 — Sandbox & security hardening
- **eureka/sandbox.py**: AST allowlist (whitelist) gate + restricted `exec()` loader
  shared by smoke test and training-time load (replaces `importlib.import_module`).
- **eureka/env_factory.py** + **evaluate_candidate.py**: load candidates via sandbox.
- **eureka/shaping_call.py**: per-step `shaping_reward()` timeout (`SHAPING_FN_TIMEOUT_S`).
- **eureka/candidate_wrapper.py**: uses timed shaping calls.
- **docs/SECURITY.md**: remaining risks + Phase 3 DSL roadmap.
- **eureka/tests/test_redteam_sandbox.py**: 15 red-team escape payloads (all rejected).
- **eureka/tests/test_sandbox.py**, **test_shaping_call.py**: loader + timeout tests.

### Concurrency, sandbox, reproducibility (Phase 0)
- **env_utils.py**: Worker env init wrapped in try/except; `AsyncVectorEnv` uses
  `poll()` timeouts on all blocking `recv()` calls; failures raise `RuntimeError`
  with module_path/seed context instead of hanging silently.
- **eureka/loop.py**: Training/eval failures (`RuntimeError` from worker env init)
  reject the candidate and continue the search.
- **eureka/smoke_test.py**: Reject all `.format()` calls (runtime dunder bypass);
  POSIX `resource.setrlimit` in probe worker as defense-in-depth.
- **eureka/eureka_config.py**: `candidate_base_seed()` — disjoint seed blocks per
  candidate (`k * EUREKA_N_ENVS`) so sibling candidates no longer share RNG state.

### Security (critical)
- **smoke_test.py**: Removed `_sanitize_candidate_code()` which stripped `import`
  lines before exec but let the raw code through to disk/import. `smoke_test()`
  now validates the **exact** string that `loop.py` writes and `env_factory.py`
  imports: AST rejection first, then an unmodified exec in a **spawn subprocess**
  runtime probe with restricted builtins.
- **env_factory.py**: Training-time code now uses the restricted sandbox loader
  rather than `importlib`; OS-level container isolation remains future work.

### Correctness / maintainability
- **ppo.py**: Removed duplicate dead method `set_learning_rate()`; `set_lr()` kept.
- **reward_wrapper.py**: `_compute_overtake_bonus()` now delegates to shared
  `compute_overtakes()` (same logic as `eureka/candidate_wrapper.py`).
- **smoke_test.py**: Runtime probe varies `n_overtakes` (from `compute_overtakes()`
  plus explicit nonzero probes) so candidates that only work at `n_overtakes == 0`
  are caught.

### Tests
- Added `eureka/tests/test_smoke_test.py` (AST rejection + malicious-code rejection).
- Added `eureka/tests/test_reward_wrapper.py` (shared overtake counting).
