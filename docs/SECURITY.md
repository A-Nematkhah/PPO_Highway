# Security model — EUREKA candidate reward code

## What runs untrusted code?

LLM-generated Python is written to `eureka/candidates/genX_candY.py`, then:

1. **Smoke test** (`eureka/smoke_test.py`) — AST allowlist + restricted `exec()` in a spawn subprocess (POSIX rlimits).
2. **Training / eval** (`eureka/env_factory.py`, `eureka/evaluate_candidate.py`) — same AST allowlist + restricted `exec()` via `eureka/sandbox.py` (no `importlib.import_module`).
3. **Per-step calls** (`eureka/candidate_wrapper.py`) — `shaping_reward()` wrapped with a wall-clock timeout (`SHAPING_FN_TIMEOUT_S`).

## What is mitigated (Phase 1)

| Risk | Mitigation |
|------|------------|
| Arbitrary imports / `open` / `eval` | AST allowlist + restricted builtins |
| Dunder attribute traversal | Blocked in AST; `.format()` blocked |
| `str.format()` runtime bypass | Explicitly rejected in AST |
| Smoke-test-only sandboxing | Training uses same `sandbox.py` loader |
| Hang after many steps | Per-step timeout in `shaping_call.py` |
| Worker env init hang | `AsyncVectorEnv` poll timeouts (Phase 0) |
| Tampered file on disk after smoke test | Re-validate AST on every load |

## Remaining risk (honest assessment)

| Risk | Status | Planned fix |
|------|--------|-------------|
| Unknown `exec()` escape vectors | **Partial** — red-team suite + allowlist; not provably secure | Phase 3: Declarative Reward DSL (no `exec`) |
| OS-level isolation (filesystem, network) | **Not implemented** — worker shares parent UID | Container (nsjail/firejail) or separate VM — future |
| Timed-out shaping thread keeps running | **Accepted** — CPython cannot kill threads; worker process recycled per env | Documented limitation |
| Windows lacks POSIX rlimits in smoke probe | **Accepted** — subprocess still limits blast radius vs in-process | Container on Linux training hosts |

## Roadmap

- **Phase 1 (current):** AST allowlist, restricted training-time load, per-step timeout, red-team tests.
- **Phase 3 (proposed):** Declarative Reward DSL (JSON/YAML + safe expression evaluator) — LLM emits DSL, not Python; security risk approaches zero.

## Running red-team tests

```bash
python -m pytest eureka/tests/test_redteam_sandbox.py -v
```

At least 15 known escape payloads must be rejected.
