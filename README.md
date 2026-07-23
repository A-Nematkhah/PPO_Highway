# PPO Highway + EUREKA Reward Search

Highway-driving reinforcement learning for `highway-env`, built around an
**EUREKA-inspired** loop: an LLM proposes Python reward-shaping functions, a
sandbox smoke-tests them, short PPO runs train each candidate, behavior metrics
rank them, and Pareto elites are reflected back into the next generation.
Shared library code (`ppo.py`, `networks.py`, `buffer.py`, `reward_wrapper.py`,
`env_utils.py`) provides the PPO stack and handcrafted TTC/overtake shaping
used by the search; baseline CLI scripts (`train.py` / `evaluate.py`) were
removed.

Deep detail: [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) ·
security: [docs/SECURITY.md](docs/SECURITY.md) · doc/code audit:
[docs/DOC_AUDIT.md](docs/DOC_AUDIT.md).

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
# source .venv/bin/activate

pip install -r requirements.txt
```

Create `groq_keys.json` in the repository root (gitignored):

```json
{"keys": ["gsk_...", "gsk_..."]}
```

Run the search (this immediately starts generation 0 — there is no `--help` flag):

```bash
python -m eureka.loop
```

Every invocation creates its own numbered directory under `runs/`
(`runs/run_0001/`, `runs/run_0002/`, ...) containing that run's
`config.json`, `metadata.json` (git commit, hardware, timing), `eureka_log.json`,
`telemetry.jsonl`, three CSV exports, a self-contained `report.html`, archived
candidate source / reflection prompts / final checkpoints, and a `console.log`
transcript. Nothing overwrites a previous run. See
[`eureka/experiment.py`](eureka/experiment.py) for exactly what lives in the
run directory versus the existing shared `eureka/candidates/` /
`eureka/checkpoints/` locations (still load-bearing — see its module
docstring for why those specifically aren't moved).

Other inspectable entrypoints:

```bash
python env_utils.py                        # print obs/action space shapes
python -m eureka.evaluate_cli --list       # inspect trained candidates outside the search loop
python -m eureka.evaluate_run 6 --render   # re-evaluate/render a specific past run's winner
python -m pytest eureka/tests/ -m "not integration"
```

Requires GPU-capable `torch` for comfortable training times; CUDA is used when available.

## Repository layout

```text
config.py                 # env / PPO / manual shaping constants
env_utils.py              # Sync/Async vector envs (spawn-safe on Windows)
reward_wrapper.py         # handcrafted TTC + overtake shaping
networks.py / buffer.py / ppo.py   # shared ActorCritic + GAE + PPO update
key_manager.py            # Groq key pool + 429 cooldown
requirements.txt
eureka/
  loop.py                 # orchestrator (python -m eureka.loop)
  eureka_config.py        # search hyperparameters
  llm_reward_designer.py  # LLM prompts + code extraction
  human_seed.py           # optional gen-0 human reward seed
  smoke_test.py           # AST + subprocess runtime probe
  sandbox.py              # restricted exec loader
  shaping_call.py         # per-step timeout + executor leak recovery
  train_candidate.py      # short-budget PPO per candidate
  evaluate_candidate.py   # deterministic behavior metrics
  objectives.py           # epsilon / NSGA-II-lite archive
  fitness.py              # legacy scalar score (diagnostic in pareto mode)
  reflection.py           # multi-elite LLM feedback prompts
  candidate_wrapper.py    # applies LLM shaping each step
  env_factory.py          # picklable candidate env factory
  logging_utils.py        # structured console (+ optional JSONL, generation/final-results tables)
  telemetry.py            # per-run telemetry.jsonl event stream
  evaluate_cli.py         # standalone CLI: evaluate/render any trained candidate
  evaluate_run.py         # re-evaluate/render a SPECIFIC past run's winner (uses that
                          # run's archived candidate/checkpoint, not the live shared ones)
  experiment.py           # ExperimentRun: numbered runs/run_NNNN/ directories
  run_metadata.py         # git/OS/CPU/RAM metadata collection (best-effort)
  csv_export.py           # pareto_archive / generation_summary / candidate_metrics CSVs
  report_html.py          # self-contained per-run report.html
  candidates/             # generated reward programs (live, load-bearing - see sandbox.py)
  checkpoints/            # per-candidate .pt (live, load-bearing)
  tests/                  # unit + integration tests
docs/
  PROJECT_ARCHITECTURE.md
  SECURITY.md
  DOC_AUDIT.md
runs/
  run_NNNN/                 # one per invocation of `python -m eureka.loop` (see Quick start above)
```

## How the EUREKA loop works

1. **Generate** — Groq returns up to `K_CANDIDATES` `shaping_reward` functions
   (plus optional human seed in generation 0).
2. **Smoke-test / sandbox** — AST allowlist + restricted subprocess probe;
   invalid code never trains.
3. **Train** — short PPO budget with `CandidateRewardWrapper`.
4. **Evaluate** — deterministic rollouts → `crash_rate`, `mean_speed`,
   `mean_overtakes` (shaped return is *not* the ranking objective).
5. **Pareto-rank** — epsilon boxes + nondominated sorting into a bounded archive.
6. **Reflect** — diverse elites + target roles feed the next LLM prompts.
7. **Repeat** for `N_GENERATIONS`; then **confirmation** seeds retrain rank-0
   finalists and rebuild the final archive.

**Default selection mode is `MULTI_OBJECTIVE_MODE="pareto"`.** Survivor selection
and LLM reflection elites come from the live archive via
`select_reflection_elites`. Legacy `compute_fitness()` is still logged but does
not lock the mutation parent. Mode `"shadow"` keeps Pareto metadata but routes
reflection through a single scalar `best` (can freeze onto an early human seed).

## Configuration

| File | Role |
|------|------|
| `config.py` | Env scene, PPO hyperparams, manual TTC/overtake weights |
| `eureka/eureka_config.py` | Evolutionary search knobs |

Settings you are most likely to change (`eureka_config.py` defaults today):

| Setting | Default | Tradeoff |
|---------|---------|----------|
| `N_GENERATIONS` | `3` | More generations improve search but multiply wall-clock ~linearly |
| `K_CANDIDATES` | `4` | More LLM variants per gen (paper uses larger K); more GPU/API cost |
| `TRAIN_STEPS_PER_CANDIDATE` | `50_000` | Higher → more reliable ranking, much slower |
| `MULTI_OBJECTIVE_MODE` | `"pareto"` | `"pareto"` = archive drives selection/reflection; `"shadow"` = scalar `best` |
| `N_EVAL_EPISODES` | `30` | Higher → less crash_rate quantization noise; eval is still cheap vs train |
| `CONFIRMATION_SEEDS` | `(10000, 20000)` | Extra full train/eval per rank-0 finalist; set `()` to skip |

Also: `SEED_GENERATION_0_WITH_HUMAN_REWARD=True`, `SHAPING_FN_TIMEOUT_S=0.05`,
`SHAPING_FN_EXECUTOR_WORKERS=8`, `GROQ_MODEL="openai/gpt-oss-120b"`.

**Do not set `config.USE_LLM_JUDGE=True`.** That path imports removed `llm_judge.py`
and will `ImportError` (see DOC_AUDIT.md).

## Safety / sandboxing

LLM reward code is **untrusted**. Mitigations include an AST allowlist, restricted
builtins + injected `math`, spawn-subprocess smoke probes, re-validation on
training load, and per-step timeouts with executor leak recovery. This is
**not** a container sandbox. Details and remaining risks:
[docs/SECURITY.md](docs/SECURITY.md).

## Testing

```bash
# Fast unit / component tests
python -m pytest eureka/tests/ -m "not integration"

# Slower end-to-end loop (mocked LLM, tiny train budget)
python -m pytest eureka/tests/ -m integration

# Coverage gate used in CI (.coveragerc fail_under=80)
python -m pytest eureka/tests/ -m "not integration" --cov=eureka --cov-report=term-missing
```

`.coveragerc` measures the `eureka` package but **omits** `eureka/loop.py`,
`train_candidate.py`, `smoke_test.py`, `env_factory.py`, and `eureka/tests/*`
from the 80% fail-under threshold.

## Known limitations

- Default search budget is small (`3 × 4` candidates; 50k steps each).
- Restricted `exec` is defense-in-depth, not OS isolation / a no-exec DSL.
- Short ranking budgets can disagree with long-horizon policy quality.
- Orphan `USE_LLM_JUDGE` hooks remain in `config.py` / `env_utils.py` /
  `reward_wrapper.py` but cannot run without restoring `llm_judge.py`.
- Hung shaping threads cannot be killed in CPython; the pool replaces itself
  when all workers leak (see SECURITY.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for sandbox hardening, Pareto defaults,
component reflection, human seed init, and related history.
