# Changelog

## Unreleased

### EUREKA-only cleanup (phase 1)
- Removed baseline entrypoints: `train.py`, `evaluate.py`, `llm_judge.py`.
- EUREKA pipeline (`python -m eureka.loop`) unchanged; all 12 tests pass.

### Security (critical)
- **smoke_test.py**: Removed `_sanitize_candidate_code()` which stripped `import`
  lines before exec but let the raw code through to disk/import. `smoke_test()`
  now validates the **exact** string that `loop.py` writes and `env_factory.py`
  imports: AST rejection first, then an unmodified exec in a **spawn subprocess**
  runtime probe with restricted builtins.
- **env_factory.py**: Documented remaining risk — training-time `importlib` still
  runs candidate modules with full worker privileges (TODO: container/isolation).

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
