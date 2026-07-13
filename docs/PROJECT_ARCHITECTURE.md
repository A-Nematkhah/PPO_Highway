# PROJECT ARCHITECTURE

## 1) Project Overview

This repository implements a highway-driving RL system with two parallel tracks:

- **Baseline RL track**: PPO training with handcrafted shaping (`TTC` penalty + overtake bonus).
- **EUREKA-inspired track**: LLM-generated reward programs are iteratively trained, evaluated, and improved.

### Research goal

The main goal is to automate reward design for autonomous highway behavior (safety + speed + overtaking skill), reducing manual reward engineering burden.

### Relation to EUREKA (ICLR 2024)

The `eureka/` pipeline follows the core EUREKA pattern:

1. LLM proposes reward code candidates.
2. Candidates are validated in a sandbox.
3. Each candidate is trained with RL for a short budget.
4. Candidates are evaluated on behavior metrics.
5. Best candidate is fed back to the LLM as reflection context for the next generation.

### Expected contribution

- A practical, Windows-compatible implementation of code-level reward evolution for `highway-env`.
- Safety-conscious reward shaping with explicit anti-gaming fitness computation.
- Incremental LLM integration roadmap:
  - Phase 1: LLM episode judge (`llm_judge.py`)
  - Phase 4: LLM reward code generation (`eureka/`)

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
                              | Evaluation & Fitness| ---> | Reflection Prompt    |
                              | crash/speed/overtake|      | (best code + metrics)|
                              +---------------------+      +----------------------+
```

Pipeline abstraction requested:

`Environment -> RL Agent -> Reward System -> LLM -> Reward Generation -> Evaluation -> Feedback/Critique -> Reward Improvement`

---

## 3) Repository Analysis

## Top-level modules

- `config.py`
  - **Purpose**: global configuration for baseline PPO and manual reward shaping.
  - **Key contents**: `ENV_CONFIG`, PPO hyperparameters, LLM judge toggles (`USE_LLM_JUDGE`), reward shaping constants.
  - **Inputs/Outputs**: imported by training, env construction, wrappers, judge.
  - **Dependencies**: none (constants only).
  - **Connections**: central parameter hub.

- `train.py`
  - **Purpose**: baseline PPO orchestration.
  - **Main functions**: `parse_args()`, `main()`, `_save_plots()`.
  - **I/O**:
    - Inputs: env observations, config constants.
    - Outputs: model checkpoints (`.pt`), plots (`learning_curve.png`, `crash_rate.png`, `speed.png`), console logs.
  - **Dependencies**: `env_utils`, `networks`, `buffer`, `ppo`.
  - **Connections**: baseline training entrypoint.

- `evaluate.py`
  - **Purpose**: evaluate trained baseline policy (deterministic or stochastic).
  - **Main function**: `evaluate(...)`.
  - **I/O**: reads model checkpoint, outputs per-episode and summary metrics.
  - **Dependencies**: `ActorCritic`, `RewardShapingWrapper`, Gymnasium.
  - **Connections**: post-training policy validation.

- `env_utils.py`
  - **Purpose**: environment factory + vectorized env abstractions.
  - **Main classes/functions**: `_EnvFactory`, `SyncVectorEnv`, `AsyncVectorEnv`, `make_vec_env(...)`.
  - **I/O**: vectorized `reset()`/`step()` API returning stacked arrays.
  - **Dependencies**: `gymnasium`, `highway_env`, `RewardShapingWrapper`, `config`.
  - **Connections**: used by both baseline and candidate training.

- `reward_wrapper.py`
  - **Purpose**: manual reward shaping over env reward.
  - **Main elements**: `compute_overtakes(...)`, `RewardShapingWrapper`.
  - **I/O**:
    - Input: env state (`ego`, `road`, `info`).
    - Output: shaped scalar reward + diagnostic keys in `info`.
  - **Dependencies**: `gymnasium`.
  - **Connections**: baseline reward path; reused utility for candidate wrapper.

- `networks.py`
  - **Purpose**: actor-critic neural architecture.
  - **Main elements**: `build_mlp(...)`, `ActorCritic`, `flatten_obs(...)`.
  - **I/O**: flattened obs -> policy logits + value.
  - **Dependencies**: `torch`.
  - **Connections**: shared by baseline and eureka candidate training/eval.

- `buffer.py`
  - **Purpose**: rollout storage + GAE computation.
  - **Main class**: `RolloutBuffer`.
  - **I/O**: stores transitions; outputs flattened training tensors.
  - **Dependencies**: `numpy`.
  - **Connections**: feeds PPO update.

- `ppo.py`
  - **Purpose**: PPO optimization step.
  - **Main class**: `PPOAgent`.
  - **I/O**: rollout minibatches in, loss stats out.
  - **Dependencies**: `torch`, `numpy`.
  - **Connections**: used by `train.py` and `eureka/train_candidate.py`.

- `llm_judge.py`
  - **Purpose**: optional phase-1 LLM binary scoring of episode quality.
  - **Main function**: `judge_episode(stats, model=...) -> int`.
  - **I/O**: episode summary -> `0`/`1`, with graceful failure returning `0`.
  - **Dependencies**: `key_manager`.
  - **Connections**: called by `RewardShapingWrapper` on episode end when enabled.

- `key_manager.py`
  - **Purpose**: Groq API key pooling and 429 cooldown rotation.
  - **Main class**: `GroqKeyManager`.
  - **I/O**: wraps `chat.completions.create`.
  - **Dependencies**: `groq` (runtime), `json`, `time`.
  - **Connections**: shared by LLM judge and eureka reward generation.

- `requirements.txt`
  - **Purpose**: runtime dependencies.

- `.gitignore`
  - **Purpose**: ignores keys, checkpoints, generated candidate code, logs, artifacts.

## EUREKA package (`eureka/`)

- `eureka/eureka_config.py`
  - **Purpose**: search-loop hyperparameters (generations, candidates, train budget, fitness weights, model).

- `eureka/loop.py`
  - **Purpose**: evolutionary orchestration loop.
  - **Main flow**:
    - generate candidates with LLM
    - smoke-test each candidate
    - write candidate code to module file
    - train candidate
    - evaluate candidate
    - compute fitness and select generation/global best
    - persist log (`eureka/eureka_log.json`)
  - **Connections**: hub connecting all eureka components.

- `eureka/llm_reward_designer.py`
  - **Purpose**: prompt + parse LLM reward function code.
  - **Main elements**: `SYSTEM_PROMPT`, `_extract_code`, `generate_candidates(...)`.
  - **Behavior**: one API call per candidate (not one call for all K), with parsing fallbacks.

- `eureka/smoke_test.py`
  - **Purpose**: pre-train safety/validity check for generated code.
  - **Mechanisms**:
    - AST gate on the exact code string that will be written/imported (rejects
      imports, dunder access, exec/eval/open/compile/__import__, global/nonlocal)
    - restricted builtins + injected `math` for in-subprocess exec probe
    - function existence check
    - runtime probe on real environment states in an isolated subprocess
    - `n_overtakes` varied across trials (computed + explicit nonzero probes)
    - finite scalar + value-range checks

- `eureka/train_candidate.py`
  - **Purpose**: short-budget PPO training per candidate.
  - **Behavior**: same PPO stack as baseline, smaller env count (`EUREKA_N_ENVS`) and budget.
  - **Output**: candidate checkpoint in `eureka/checkpoints/`.

- `eureka/evaluate_candidate.py`
  - **Purpose**: deterministic evaluation for candidate ranking.
  - **Output metrics**: `crash_rate`, `mean_speed`, `mean_overtakes`, `mean_raw_return`.
  - **Important**: fitness does not use shaped return.

- `eureka/fitness.py`
  - **Purpose**: scalar ranking score:
    - `-w_crash*crash_rate + w_speed*mean_speed + w_overtakes*mean_overtakes`.

- `eureka/reflection.py`
  - **Purpose**: builds LLM feedback prompt from best code + metrics.

- `eureka/candidate_wrapper.py`
  - **Purpose**: executes candidate reward function in environment step loop.
  - **Safety**: invalid candidate outputs degrade to `0.0` shaping instead of crashing training.

- `eureka/env_factory.py`
  - **Purpose**: picklable factory for candidate environments (Windows `spawn` compatible).
  - **Key design**: uses module path string, imports candidate function inside worker.

- `eureka/candidates/*.py`
  - **Purpose**: generated candidate reward programs (artifacts).
  - **Role**: executable candidate modules for multiprocessing workers.

- `eureka/eureka_log.json`
  - **Purpose**: serialized generation results archive.

---

## 4) Full Execution Flow (Complete Experiment)

1. **Starting command**
   - Baseline: `python train.py`
   - EUREKA loop: `python -m eureka.loop`

2. **Configuration loading**
   - `config.py` for PPO/env/reward.
   - `eureka/eureka_config.py` for search settings.

3. **Environment initialization**
   - Base env `highway-fast-v0` configured via `ENV_CONFIG`.
   - Wrapped with:
     - baseline: `RewardShapingWrapper`
     - eureka: `CandidateRewardWrapper`
   - Vectorized through `AsyncVectorEnv` (spawn workers on Windows).

4. **Reward creation**
   - Baseline: handcrafted TTC + overtake shaping (+ optional LLM judge terminal bonus).
   - EUREKA: generated function `shaping_reward(ego, road, info)`.

5. **LLM interaction (EUREKA path)**
   - `build_reflection(best)` creates user prompt.
   - `SYSTEM_PROMPT` enforces contract and constraints.
   - Groq API called via `key_manager`.

6. **Reward validation**
   - Candidate code passes `smoke_test` sandbox/runtime checks.
   - Failed candidates are rejected before expensive RL training.

7. **RL training**
   - Rollout collection (`RolloutBuffer`) -> GAE -> PPO updates (`PPOAgent.update`).
   - Candidate run saves checkpoint.

8. **Evaluation**
   - Deterministic argmax policy over fixed number of episodes.
   - Metrics computed independently from shaped reward accumulation.

9. **Reward update/evolution**
   - Fitness computed from crash/speed/overtakes.
   - Generation best and global best tracked.
   - Best candidate used in next generation reflection prompt.

10. **Final result**
   - Printed best module/checkpoint/metrics/code.
   - Full generation logs persisted to `eureka/eureka_log.json`.

---

## 5) Reward Engineering System

### Reward representation

- **Baseline**: parameterized handcrafted shaping inside `RewardShapingWrapper`.
- **EUREKA**: executable Python program (`def shaping_reward(...) -> float`) generated by LLM.

### Reward components in current baseline

- Environment-native reward from `highway-env` (`collision_reward`, `high_speed_reward`, etc.).
- Additional shaping:
  - TTC penalty (continuous risk-based).
  - Overtake bonus (event-based).
- Optional terminal LLM judge bonus (phase 1).

### Reward calculation flow

`env.step(action)` -> base reward -> wrapper computes shaping terms -> shaped reward returned to PPO -> diagnostics put in `info`.

### Reward shaping strategy

- Combine safety (TTC), efficiency (speed), and maneuver progress (overtake).
- Keep shaping magnitudes bounded and interpretable.
- In EUREKA, LLM explores alternative shaping programs under constrained function contract.

### Reward hacking prevention mechanisms

- Candidate fitness excludes candidate-shaped return.
- Fitness based on external behavior metrics (`crash_rate`, `mean_speed`, `mean_overtakes`).
- Sandbox and runtime validation in `smoke_test`.
- Candidate wrapper falls back to zero shaping on invalid outputs.

### Current limitations

- Fitness still scalarized; trade-offs depend heavily on chosen weights.
- Candidate code execution is constrained but still Python execution.
- No explicit novelty/diversity pressure beyond prompt instruction.

---

## 6) LLM Module

### Model/API used

- Groq Chat Completions API via `groq` SDK.
- Current models:
  - Phase-1 judge default: `openai/gpt-oss-20b`
  - EUREKA generation: `openai/gpt-oss-120b`

### Prompt design

- Strong system prompt defines:
  - exact function signature
  - available object attributes
  - output range expectations
  - formatting constraints (single fenced Python block)
  - no import statements
- Reflection prompt injects previous best code + quantitative outcomes.

### Input/output format

- Input: system + user prompt (best candidate context if available).
- Output expected: Python function text; parser extracts fenced block or fallback from `def shaping_reward`.

### Generated reward handling

- Parsed code stored in `eureka/candidates/genX_candY.py`.
- Imported by module path in worker processes.

### Error handling

- API errors per candidate are caught and logged; loop continues.
- Key manager rotates keys on 429 and retries across key pool.
- Missing/invalid candidate outputs are skipped.

### Validation/sandbox

- `smoke_test` AST-gates the exact code string saved to disk (no import stripping).
- Forbidden: imports, dunder attribute access, exec/eval/open/compile/__import__,
  global/nonlocal.
- Runtime probe runs in a spawn subprocess with restricted builtins + injected `math`.
- Probes real env states with varied `n_overtakes` (computed + explicit nonzero).
- Ensures callable exists and runtime behavior is numeric/finite/bounded.
- Training-time import in `env_factory.py` still runs with full worker privileges
  (documented TODO for container/isolation).

---

## 7) Reinforcement Learning Pipeline

### Environment

- `highway-fast-v0` from `highway-env`.
- Observation: `Kinematics` with `vehicles_count=15`, features `[presence, x, y, vx, vy]`, normalized and sorted.
- Action space: `DiscreteMetaAction` (5 actions).

### RL algorithm

- PPO with:
  - clipped surrogate objective
  - advantage normalization
  - entropy bonus
  - value loss
  - gradient clipping
  - linear LR annealing

### Training parameters (baseline defaults)

- `N_ENVS=6`, `N_STEPS=128`, `N_EPOCHS=10`, `BATCH_SIZE=64`
- `GAMMA=0.95`, `GAE_LAMBDA=0.95`, `CLIP_RANGE=0.2`
- `LR=5e-4`, `ENT_COEF=0.01`, `TOTAL_TIMESTEPS=200000`

### Evaluation metrics

- Baseline: return, crash rate, speed, overtakes.
- EUREKA ranking: crash rate, mean speed, mean overtakes (and mean raw return for reporting only).

### Logging system

- Console tables for updates.
- Baseline plot artifacts.
- EUREKA per-stage logs + `eureka/eureka_log.json`.

---

## 8) EUREKA Comparison

| EUREKA Feature | Current Implementation | Missing | Possible Improvement |
|---|---|---|---|
| LLM reward generation | Yes (`eureka/llm_reward_designer.py`) | Multi-model ensemble | Evaluate multiple model families per generation |
| Automatic reward coding | Yes (Python function synthesis) | Full training-time isolation | Containerize candidate import in workers |
| Evolutionary optimization | Yes (generation loop + best feedback) | Explicit population operators (mutation/crossover) | Add operator-based candidate derivation |
| Reflection/critique | Yes (`reflection.py`) | Multi-perspective critiques | Add critic prompts per metric failure mode |
| Reward archive/RAG | Partial (`eureka_log.json`) | Retrieval over historical candidates | Build searchable archive + embedding retrieval |
| Sandbox execution | Yes (AST gate + subprocess smoke probe) | Training-time import still full-privilege | Container/jail for env_factory worker import |
| Parallel training | Intra-candidate env parallelism | Inter-candidate concurrent training | Train multiple candidates concurrently across devices |
| Experiment management | Basic logs/checkpoints | Reproducible run tracking (e.g., W&B/MLflow) | Add run IDs, config snapshots, artifacts, seeds matrix |

---

## 9) Research Evaluation

### Novel aspects

- Practical integration of LLM code generation with PPO reward optimization in highway driving.
- Explicit anti-reward-hacking ranking metric design.
- Windows-friendly multiprocessing architecture for candidate code modules.

### Weak points

- Small search budget (`3 x 4` candidates) relative to EUREKA-style exploration.
- Fitness scalarization may over-reward aggressive policies depending on weights.
- Limited safety sandboxing depth.

### Technical risks

- API rate limits and network failures can reduce candidate diversity.
- Overfitting to short training budgets for candidate ranking.
- Potential mismatch between training reward and evaluation behavior.

### Missing experiments

- Baseline vs EUREKA comparative table across multiple seeds.
- Long-horizon retraining of best candidate reward.
- Robustness under changed traffic density/lanes/weather-like perturbations (if supported).

### Required ablation studies

- Remove TTC term / overtake term / LLM judge effects.
- Vary fitness weights and evaluate safety-speed Pareto tradeoff.
- Compare deterministic vs stochastic evaluation policy.
- Compare number of generations/candidates and train budget.

### Toward publishable research quality

- Add multi-seed statistical significance.
- Add stronger benchmark protocol and baselines (manual reward tuning, random search, Bayesian optimization).
- Add reproducibility package (fixed seeds, manifests, version pinning, run registry).

---

## 10) Future Roadmap

1. Add inter-candidate parallel training scheduler.
2. Introduce archive retrieval and novelty-aware candidate selection.
3. Add AST-level policy for candidate code safety and complexity limits.
4. Add multi-objective optimization (safety vs speed Pareto front).
5. Add automated regression suite for reward functions and environment variants.
6. Integrate structured experiment tracking and report generation.
7. Distill best evolved reward into interpretable parametric form.

---

## 11) Running Instructions

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

- Required: `torch`, `gymnasium`, `highway-env`, `numpy`, `matplotlib`
- Optional LLM: `groq`

### Configuration

- Baseline and env params: `config.py`
- EUREKA search params: `eureka/eureka_config.py`
- Groq keys file at repo root:

```json
{"keys": ["gsk_...", "gsk_..."]}
```

### Training commands

- Baseline PPO:

```bash
python train.py
```

- Resume baseline training:

```bash
python train.py --resume checkpoints/ppo_highway_stepXXXX.pt
```

- EUREKA reward search:

```bash
python -m eureka.loop
```

### Evaluation commands

- Baseline model eval:

```bash
python evaluate.py --model ppo_highway_scratch.pt --episodes 10
```

- Rendered eval:

```bash
python evaluate.py --model ppo_highway_scratch.pt --episodes 10 --render
```

---

## Notes on completeness

- This document reflects actual behavior of the current codebase.
- Generated artifacts (`eureka/candidates/*.py`, checkpoints, and logs) are runtime products, not stable source modules.
- Several comments in code mention phased roadmap features not fully expanded yet (e.g., richer critique/retrieval loops).

