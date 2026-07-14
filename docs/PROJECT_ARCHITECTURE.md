# PROJECT ARCHITECTURE

## 1) Project Overview

This repository implements a highway-driving RL system oriented around an
**EUREKA-inspired** evolutionary reward-design pipeline, with shared PPO/env
building blocks that originally supported a separate baseline track.

- **Shared RL stack**: PPO (`ppo.py`, `networks.py`, `buffer.py`) + `highway-fast-v0`
  via `env_utils.py`, with handcrafted TTC/overtake shaping in `reward_wrapper.py`.
- **EUREKA track (active entrypoint)**: LLM-generated `shaping_reward` programs are
  smoke-tested, trained briefly, evaluated on behavior metrics, ranked with
  epsilon/NSGA-II-lite Pareto selection, and reflected back to the LLM.

Baseline CLI scripts (`train.py`, `evaluate.py`) and the Phase-1 episode judge
module (`llm_judge.py`) were **removed** (see CHANGELOG). Library code that
supported them remains (manual shaping wrapper, vector envs). Enabling
`USE_LLM_JUDGE=True` is currently a **broken orphan path** — it imports the
missing `llm_judge` module (see §3 and `docs/DOC_AUDIT.md`).

### Research goal

Automate reward design for autonomous highway behavior (safety + speed +
overtaking skill), reducing manual reward engineering burden.

### Relation to EUREKA (ICLR 2024)

1. LLM proposes reward code candidates.
2. Candidates are validated in a sandbox.
3. Each candidate is trained with RL for a short budget.
4. Candidates are evaluated on behavior metrics.
5. Pareto elites are fed back to the LLM as diverse reflection context.

---

## 2) High-Level Architecture

```text
+--------------------+      +---------------------+      +----------------------+
|  highway-fast-v0   | ---> |  PPO Agent          | ---> |  Behavior Rollouts   |
|  (Gymnasium env)   |      |  (ActorCritic+PPO)  |      |  (returns, crashes)  |
+--------------------+      +---------------------+      +----------------------+
          ^                           |                               |
          |                           v                               v
+--------------------+      +---------------------+      +----------------------+
| Reward Wrappers    | <--- | Reward Function     | <--- | LLM Reward Generator |
| - manual shaping   |      | (Python function)   |      | (Groq model)         |
| - candidate shaping|      +---------------------+      +----------------------+
+--------------------+                  ^                              |
                                        |                              v
                              +---------------------+      +----------------------+
                              | Eval + Pareto archive| ---> | Reflection Prompt    |
                              | crash/speed/overtake |      | (elites + metrics)   |
                              +---------------------+      +----------------------+
```

Pipeline: `Environment -> RL Agent -> Reward System -> LLM -> Reward Generation -> Evaluation -> Feedback/Critique -> Reward Improvement`

---

## 3) Repository Analysis

### Top-level modules

- `config.py`
  - **Purpose**: env / PPO / manual shaping constants (also orphaned LLM-judge toggles).
  - **Key contents**: `ENV_CONFIG`, PPO hyperparameters, `TTC_*` / `OVERTAKE_BONUS`,
    `USE_LLM_JUDGE` (default `False`; **do not enable** — `llm_judge.py` is removed).

- `env_utils.py`
  - **Purpose**: environment factory + vectorized env abstractions.
  - **Main classes/functions**: `_EnvFactory`, `SyncVectorEnv`, `AsyncVectorEnv`,
    `make_vec_env(...)`.
  - **Note:** `_EnvFactory` still contains a conditional import of missing
    `llm_judge` when `USE_LLM_JUDGE` is True (dead/broken path).

- `reward_wrapper.py`
  - **Purpose**: handcrafted TTC + overtake shaping; optional `llm_judge_fn` hook
    retained but unusable without the removed module.
  - **Main elements**: `compute_overtakes(...)`, `RewardShapingWrapper`.

- `networks.py` — `build_mlp`, `ActorCritic`, `flatten_obs` (shared PPO policy/value).
- `buffer.py` — `RolloutBuffer` + GAE.
- `ppo.py` — `PPOAgent` update step (used by `eureka/train_candidate.py`).
- `key_manager.py` — `GroqKeyManager` (API key pool / 429 rotation for EUREKA LLM calls).
- `requirements.txt`, `.gitignore` — deps and ignored artifacts (keys, checkpoints, logs).

**Removed (historical — do not document as runnable):** `train.py`, `evaluate.py`,
`llm_judge.py`.

### EUREKA package (`eureka/`)

- `eureka_config.py` — search hyperparameters (`N_GENERATIONS`, `K_CANDIDATES`,
  budgets, `MULTI_OBJECTIVE_MODE`, `CONFIRMATION_SEEDS`, LLM settings).
- `human_seed.py` — hand-written `shaping_reward` (TTC + overtake) as optional gen-0
  candidate (`SEED_GENERATION_0_WITH_HUMAN_REWARD`).
- `loop.py` — orchestrator: generate → smoke → train → eval → Pareto archive →
  reflect → log; optional confirmation of rank-0 finalists.
- `llm_reward_designer.py` — Groq prompts + `_extract_code` / `generate_candidates`.
- `smoke_test.py` — AST allowlist + restricted subprocess runtime probe.
- `sandbox.py` — shared AST whitelist + restricted `exec` loader +
  `normalize_shaping_output`.
- `shaping_call.py` — per-step timeout via shared `ThreadPoolExecutor`, with leak
  detection / executor replacement (`SHAPING_FN_EXECUTOR_WORKERS`, default 8).
- `train_candidate.py` — short-budget PPO; optional component snapshot sidecars.
- `evaluate_candidate.py` — deterministic eval → `crash_rate`, `mean_speed`,
  `mean_overtakes`, `mean_raw_return`, `component_means`.
- `objectives.py` — epsilon-box NSGA-II-lite archive / reflection elites.
- `fitness.py` — legacy scalar `compute_fitness` (diagnostic in `"pareto"` mode;
  drives `best` / reflection only in `"shadow"`).
- `reflection.py` — multi-elite / role-targeted LLM prompts (+ component history).
- `candidate_wrapper.py` — applies candidate shaping each step (degrades to 0).
- `env_factory.py` — picklable candidate env factory; loads code via `sandbox`.
- `logging_utils.py` — structured console (+ optional JSONL run log).
- `telemetry.py` — `eureka_metrics.jsonl` event stream.
- `candidates/*.py`, `eureka_log.json`, checkpoints — runtime artifacts.

---

## 4) Full Execution Flow (Complete Experiment)

1. **Start:** `python -m eureka.loop` (requires `groq_keys.json` at repo root).
2. **Config:** `config.py` (env/PPO) + `eureka/eureka_config.py` (search).
3. **Env:** `highway-fast-v0` + `CandidateRewardWrapper`, vectorized
   (`AsyncVectorEnv` / spawn on Windows).
4. **LLM:** `build_reflection` + `SYSTEM_PROMPT` → Groq via `key_manager`.
5. **Validate:** `smoke_test` AST + runtime probe; reject before train.
6. **Train:** rollouts → GAE → `PPOAgent.update` → checkpoint.
7. **Evaluate:** deterministic policy; metrics independent of shaped return.
8. **Select:** annotate objectives; update Pareto archive. With default
   `MULTI_OBJECTIVE_MODE="pareto"`, reflection elites come from the archive
   (`select_reflection_elites`), not a single locked scalar `best`.
9. **Confirm (default on):** `CONFIRMATION_SEEDS` retrain rank-0 finalists and
   rebuild the final archive from aggregate metrics.
10. **Persist:** `eureka/eureka_log.json` + `eureka_metrics.jsonl`.

---

## 5) Reward Engineering System

- **Manual shaping:** `RewardShapingWrapper` (TTC + overtakes) over highway-env base reward.
- **EUREKA shaping:** LLM `shaping_reward(ego, road, info)` → float or
  `(total, components: dict)`.
- **Anti-hacking:** ranking uses external metrics, not shaped return; smoke/sandbox;
  wrapper degrades invalid outputs to `0.0`.

### Current limitations

- `MULTI_OBJECTIVE_MODE="pareto"` (current default) makes the epsilon/NSGA-II-lite
  archive authoritative for survivor selection and LLM reflection elites.
  `"shadow"` remains available for diagnostics (scalar `best` still drives
  reflection context).
- Candidate code still runs via restricted Python `exec` (not a container / DSL).
- Limited novelty pressure beyond prompt roles / crowding.
- Orphan `USE_LLM_JUDGE` path cannot be enabled without restoring `llm_judge.py`.

---

## 6) LLM Module

- **API:** Groq Chat Completions via `groq` + `key_manager`.
- **EUREKA model default:** `openai/gpt-oss-120b` (`eureka_config.GROQ_MODEL`).
- **Prompt / parse:** `SYSTEM_PROMPT` + fenced/`def shaping_reward` extraction.
- **Artifacts:** `eureka/candidates/genX_candY.py`, loaded through `sandbox`.
- **Sandbox:** AST allowlist; no imports / dunder / eval/exec/open; subprocess
  probe; training/eval re-validate AST on load. Remaining risk: OS-level isolation
  and unknown `exec` escapes — see `docs/SECURITY.md`.

---

## 7) Reinforcement Learning Pipeline

- **Env:** `highway-fast-v0`, kinematics obs, `DiscreteMetaAction` (5 actions).
- **Algo:** PPO (clip, GAE, entropy, value loss, grad clip, LR anneal).
- **Shared defaults (`config.py`):** `N_ENVS=6`, `N_STEPS=128`, `N_EPOCHS=10`,
  `BATCH_SIZE=64`, `GAMMA=0.95`, `GAE_LAMBDA=0.95`, `CLIP_RANGE=0.2`, `LR=5e-4`,
  `ENT_COEF=0.01`, `TOTAL_TIMESTEPS=200_000`.
- **EUREKA train defaults:** `EUREKA_N_ENVS=4`, `TRAIN_STEPS_PER_CANDIDATE=50_000`,
  `N_EVAL_EPISODES=30`.
- **Logging:** structured console (`logging_utils`), optional
  `EUREKA_LOG_JSON=1` → `eureka/eureka_run.jsonl`, telemetry JSONL for metrics.

---

## 8) EUREKA Comparison

| EUREKA Feature | Current Implementation | Missing | Possible Improvement |
|---|---|---|---|
| LLM reward generation | Yes (`llm_reward_designer.py`) | Multi-model ensemble | Evaluate multiple model families |
| Automatic reward coding | Yes (Python synthesis + sandbox) | OS-level isolation | Containerize worker processes |
| Evolutionary optimization | Yes (generation loop + archive) | Explicit crossover operators | Operator-based derivation |
| Reflection/critique | Yes (`reflection.py`) | Multi-perspective critics | Critic prompts per failure mode |
| Reward archive/RAG | Partial (`eureka_log.json` + in-memory archive) | Embedding retrieval | Searchable historical archive |
| Sandbox execution | AST + restricted exec + smoke probe | Container / no-exec DSL | Phase 3 declarative DSL |
| Parallel training | Intra-candidate env parallelism | Inter-candidate concurrent train | Multi-device scheduler |
| Experiment management | Logs + checkpoints + telemetry | W&B/MLflow | Run IDs, config snapshots |

---

## 9) Research Evaluation

### Weak points

- Small default search budget (`3 × 4` candidates) vs paper-scale EUREKA.
- Short per-candidate train budgets can mismatch long-horizon policy quality.
- Sandbox is defense-in-depth, not a hard security boundary.
- Historical scalar fitness still logged; in `"shadow"` it can lock reflection
  onto an early human seed (observed in a completed shadow-mode run).

### Technical risks

- API rate limits / network failures reduce candidate diversity.
- Eval noise (mitigated somewhat by `N_EVAL_EPISODES=30` and confirmation seeds).
- Training reward vs evaluation metric mismatch.

---

## 10) Future Roadmap

1. Inter-candidate parallel training scheduler.
2. Archive retrieval / novelty-aware selection.
3. Stricter AST complexity limits (and/or Phase 3 DSL).
4. Multi-seed scientific protocols beyond the two default confirmation seeds.
5. Richer experiment tracking and report generation.
6. Distill best rewards into interpretable parametric forms.
7. Restore or remove the orphan `USE_LLM_JUDGE` / `llm_judge` hooks cleanly.

---

## 11) Multi-objective Selection

Objectives (epsilon boxes): minimize `crash_rate` (ε=0.10), maximize
`mean_speed` (ε=0.5), maximize `mean_overtakes` (ε=0.25). `mean_raw_return` is
reporting-only.

**Default:** `MULTI_OBJECTIVE_MODE="pareto"` — archive is authoritative for
survival and reflection elites; scalar fitness is diagnostic only.

**Optional:** `"shadow"` — Pareto metadata is still logged, but generation-to-
generation reflection follows legacy scalar `best` (can freeze mutation parents).

Confirmation: `CONFIRMATION_SEEDS = (10000, 20000)` retrains rank-0 finalists
before the final archive is reported.

---

## 12) Running Instructions

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Groq keys at repo root:

```json
{"keys": ["gsk_...", "gsk_..."]}
```

```bash
python -m eureka.loop
```

See root `README.md` for tests and configuration tradeoffs.

---

## Notes on completeness

- Generated candidates, checkpoints, and log files are runtime artifacts.
- Prefer `docs/DOC_AUDIT.md` for the 2026-07-14 code/doc drift reconciliation.
