# Documentation audit summary (2026-07-14)

Cross-check of docs/comments vs current repo state after Pareto default switch,
sandbox hardening, and EUREKA-only cleanup.

## Inconsistencies found and fixes applied

### Critical / structural

1. **`docs/PROJECT_ARCHITECTURE.md` listed `train.py`, `evaluate.py`, `llm_judge.py` as active modules**
   - **Reality:** these files are absent (matches CHANGELOG "EUREKA-only cleanup").
   - **Fix:** removed those module entries; updated execution flow so the only runnable
     search entrypoint is `python -m eureka.loop`.

2. **Dead `llm_judge` code path still present while module is gone**
   - **Reality:** `config.USE_LLM_JUDGE` / `env_utils._EnvFactory` still do
     `from llm_judge import judge_episode` when enabled; `reward_wrapper` still accepts
     `llm_judge_fn`. Enabling `USE_LLM_JUDGE=True` would raise `ImportError`.
   - **Fix:** documented as a known broken/orphan path in architecture + README;
     added explicit warning comments in `config.py`, `env_utils.py`, and
     `reward_wrapper.py`. **Did not delete the hooks** (per task instructions).

3. **`MULTI_OBJECTIVE_MODE` default prose lagged behind config**
   - **Reality:** `eureka_config.py` sets `"pareto"`.
   - **Fix:** section 5/11 already partially updated; section 11 prose now states
     Pareto is the active default and clarifies shadow vs reflection_context.

4. **`eureka/fitness.py` module docstring still said scalar fitness "ranks candidates"**
   - **Reality:** always computed/logged; drives selection only when mode is `"shadow"`.
   - **Fix:** docstring clarified.

5. **In-code comments still referenced removed `train.py` / `llm_judge.py`**
   - Files: `env_utils.py` (AsyncVectorEnv docstring), `eureka/train_candidate.py`,
     `reward_wrapper.py`, `config.py`.
   - **Fix:** comments/docstrings corrected or marked orphan.

6. **`docs/SECURITY.md` understated current shaping executor sizing**
   - **Reality:** `SHAPING_FN_EXECUTOR_WORKERS` default is **8** (was hardcoded 2);
     leak detection + executor replacement already documented.
   - **Fix:** mitigate table / remaining-risk row mention configurable worker count.

7. **`CHANGELOG.md` removal claim is accurate; orphan path not called out**
   - **Fix:** follow-up note under Unreleased documenting the leftover
     `USE_LLM_JUDGE` / `llm_judge` import stub.

8. **Architecture claimed training-time `env_factory` still uses privilege-heavy
   `importlib` / "TODO"**
   - **Reality:** loads via `eureka.sandbox` restricted exec (changelog + code).
   - **Fix:** sandbox subsection updated; remaining risk is OS-level isolation only.

9. **Missing documentation of eureka modules that now exist**
   - `sandbox.py`, `shaping_call.py`, `logging_utils.py`, `telemetry.py`.
   - **Fix:** added to architecture section 3.

10. **Roadmap item "Confirm Pareto finalists across seeds"** is now default-on
    (`CONFIRMATION_SEEDS = (10000, 20000)`).
    - **Fix:** marked as configured default in architecture / README known limitations.

### Docstrings flagged but left (style-only / historical naming, not wrong enough to rewrite)

- `eureka/eureka_config.py` / `llm_reward_designer.py` still say "Phase 4" — naming of
  the EUREKA code-gen track; functionally accurate.
- `eureka/telemetry.py` mentions "Phase 5" plotting — aspirational consumer, not false.
- `eureka/sandbox.py` "Phase 1 hardening" / "Phase 3 DSL" — matches SECURITY.md roadmap.

### Verified accurate (no change needed)

- CHANGELOG entries claiming removal of `train.py` / `evaluate.py` / `llm_judge.py`.
- Smoke-test AST allowlist + subprocess probe; training load via sandbox.
- Shaping timeout + leak recovery behavior in `shaping_call.py`.
- Human-seed generation-0 behavior; component-level reflection contract.
- `.coveragerc` fail_under=80 omitting `loop.py`, `train_candidate.py`,
  `smoke_test.py`, `env_factory.py`, and tests.

## Deliverables from this audit

- Updated `docs/PROJECT_ARCHITECTURE.md`, `docs/SECURITY.md`, `CHANGELOG.md`
- Corrected/annotated comments in `config.py`, `env_utils.py`, `reward_wrapper.py`,
  `eureka/fitness.py`, `eureka/train_candidate.py`
- New root `README.md`
- This file (`docs/DOC_AUDIT.md`)
