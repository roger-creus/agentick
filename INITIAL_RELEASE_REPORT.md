# Agentick Initial Public Release — Change Report

**Date**: 2026-03-17
**Commit**: `cd01e73`
**Stats**: 145 files changed, 1,035 insertions, 17,808 deletions

---

## Phase 0: Git Branching

- Created `dev` branch from `main` HEAD (`fa95128`) and pushed to remote
- All release work done on `main`

---

## Phase 1: Critical Bug Fixes

### 1A. BacktrackPuzzle double-activation bug
- **File**: `agentick/tasks/planning/backtrack_puzzle.py`
- **Bug**: `on_agent_interact()` incremented `_switches_activated` even when switch was already activated (metadata >= 100). Interacting with same switch N times could open the gate on medium+ difficulties.
- **Fix**: Added guard `if grid.metadata[sy, sx] < 100:` before the activation block.

### 1B. scoring.py stale TASK_CAPABILITY_MAP
- **File**: `agentick/leaderboard/scoring.py`
- **Bug**: Used old 11-category system (30 tasks). Missing 8 tasks. Wrong categorizations.
- **Fix**: Replaced with correct 6-category mapping (38 tasks): navigation(8), planning(9), reasoning(9), memory(4), generalization(3), multi_agent(5).

### 1C. Leaderboard site v1 -> v2
- **File**: `agentick/leaderboard/site/generate.py`
- **Fix**: Changed `"agentick-full-v1"` to `"agentick-full-v2"`.

### 1D. Task correctness audit
- All 38 tasks verified via `test_all_tasks_behavioral.py`: **420 passed, 0 failed**
- Oracle tests: **417/456 = 91.4%** (sub-90% tasks are stochastic NPC tasks — expected)
- All 5 render modes verified on GoToGoal-v0: ascii, language, language_structured, rgb_array, state_dict

---

## Phase 2: Removed Stale Code

### 2A. Deleted stale packages
| Package | Contents | Reason |
|---------|----------|--------|
| `agentick/curriculum/` | AdaptiveCurriculum, ManualCurriculum | Never used by tasks |
| `agentick/benchmark/` | Duplicate of leaderboard/ | Superseded |
| `agentick/rewards/` | CompositeReward, PotentialBasedReward | Never used by tasks |
| `agentick/vector/` | AsyncVectorAgentickEnv | Not needed for release |

### 2B. Deleted training modules
| File/Dir | Contents | Reason |
|----------|----------|--------|
| `agentick/training/tinker/` | TinkerSFTTrainer, TinkerRLTrainer | Moved to dev branch |
| `agentick/training/behavior_cloning.py` | BehaviorCloningTrainer | Moved to dev branch |
| `agentick/training/README.md` | Stale docs | Referenced deleted modules |

**Updated**:
- `agentick/training/__init__.py`: Removed BehaviorCloningTrainer, TinkerSFTTrainer lazy imports. Removed CurriculumCallback import. Kept AgentickSFTTrainer + SFTAgent.
- `agentick/training/callbacks.py`: Removed CurriculumCallback class (~100 lines).

**Deleted examples**: `behavior_cloning_training.py`, `tinker_sft_training.py`, `tinker_rl_training.py`

### 2C. Deleted Anthropic backend
| File | Action |
|------|--------|
| `agentick/agents/backends/anthropic_backend.py` | Deleted |
| `agentick/agents/backends/__init__.py` | Removed `"anthropic"` from `_BACKEND_REGISTRY` |
| `agentick/agents/factory.py` | Removed `"anthropic"` from model key check |
| `agentick/leaderboard/cost_tracker.py` | Removed 5 Claude model pricing entries |
| `agentick/experiments/runner.py` | Removed `"anthropic"` from `_API_BACKENDS` |
| `examples/llm/anthropic_text_agent.py` | Deleted |
| `examples/llm/anthropic_vision_agent.py` | Deleted |
| 4x `examples/experiments/configs/claude_haiku_*.yaml` | Deleted |

### 2D. Removed rgb_array_flat / SimpleGridRenderer
| File | Action |
|------|--------|
| `agentick/core/simple_grid_renderer.py` | Deleted (609 lines) |
| `agentick/core/renderer.py` | Removed SimpleGridRenderer branch, removed PixelRenderer alias. `"human"` mode now maps to IsometricRenderer. |
| `agentick/core/env.py` | Removed `"rgb_array_flat"` from metadata render_modes. Removed observation space branch for `rgb_array_flat`/`rgb_array_2d`. Fixed `get_pixel_observation()` to use `_render_isometric()` instead of deleted PixelRenderer. Removed unused `ObjectType` import. |
| `agentick/experiments/config.py` | Removed `"rgb_array_flat"` from valid_modes |
| `agentick/experiments/training_runner.py` | Changed 3x `"rgb_array_flat"` defaults to `"rgb_array"` |
| `agentick/wrappers/atari_preprocessing.py` | Changed default render mode to `"rgb_array"` |
| `agentick/tasks/registry.py` | Updated docstring |
| `docs/showcase/videos/flat/` | Deleted directory |
| 2x `examples/experiments/configs/ppo_pixels_*_rgbFlat.yaml` | Deleted |

### 2E. Removed NonMarkovian harnesses
- **File**: `agentick/agents/harness.py` — Rewrote from 475 to 168 lines.
  - Removed: `NonMarkovianZeroShot`, `NonMarkovianReasoner`, `COT_SYSTEM_SUFFIX_COMPACT`
  - Removed helper functions: `_content_to_text`, `_compute_diff`, `_estimate_tokens_content`, `_estimate_tokens_msg`, `_build_sliding_history`, `_render_history_slice`, `_compact_cot_response`
  - Removed `difflib` import (only used by NonMarkovian)
  - Updated `HARNESS_REGISTRY`: kept `markovian_zero_shot` + `markovian_reasoner`
- **File**: `agentick/agents/__init__.py` — Removed NonMarkovianZeroShot from imports and `__all__`
- **Deleted**: 10 nonmarkov YAML configs (`qwen3_4b_*_nonmarkov*.yaml`, `qwen35_4b_*_nonmarkov*.yaml`)

### 2F. Cleaned examples
| Deleted | Count |
|---------|-------|
| `examples/debug/` directory | 3 files |
| `examples/llm/` stale files | 5 files (base_llm_agent, compare_llms, openai_cot_agent, openai_vision_agent, openai_agent) |
| `examples/rl/` stale files | 4 files (dqn_cleanrl, ppo_cleanrl, ppo_pixels, dqn_pixels) |

**Kept**: `openai_text_agent.py` (rewritten), `huggingface_local_agent.py`, `sb3_ppo.py`, `sb3_dqn.py`

### 2G. Cleaned leaderboard adapters
- **Moved**: `agentick/leaderboard/adapters/prompt_templates.py` -> `agentick/agents/prompt_templates.py`
- **Updated imports** in: `agents/harness.py`, `agents/observation.py`, `agents/base.py`, `experiments/batched_runner.py`
- **Deleted entire `adapters/` directory**: api_adapter.py, code_adapter.py, docker_adapter.py, git_adapter.py, huggingface_adapter.py, local_weights_adapter.py, protocol.py, `__init__.py`
- **Deleted**: `evaluator.py`, `submission.py`, `comparison.py`
- **Fixed broken imports in**: `result.py` (removed SubmissionSpec dependency), `cli.py` (rewrote evaluate/verify commands), `integrity.py` (removed verify_reproducibility function)

---

## Phase 3: Standardized Experiment Runner

- **Episode filenames**: Changed from `seed_{idx}_ep_{idx}.json` to `diff_{difficulty}_seed_{idx}_ep_{idx}.json` (both in worker and _run_episode paths)
- **metadata.json**: Added fields: `agent_name`, `agent_type`, `model`, `backend`, `observation_modes`, `harness`

---

## Phase 4: Reworked Logging, Visualization & Analysis

### Logging (`agentick/logging/`)
- **Deleted**: `llm_logger.py`, `agent_logger.py`, `browser.py`, `replay.py`, `README.md`
- **Kept**: `experiment_logger.py`, `episode_logger.py`
- **Updated**: `__init__.py` (exports EpisodeLogger + ExperimentLogger)

### Visualization (`agentick/visualization/`)
- **Deleted**: `training_plots.py`, `interactive.py`, `report.py`, `tables.py`, `video.py`, `plots.py`, `README.md`
- **Kept**: `experiment_plots.py`, `comparison_plots.py`, `style.py`
- **Created**: `agentick/experiments/_video_utils.py` (extracted `_has_ffmpeg`, `_save_mp4`, `_save_gif` from deleted video.py — used by runner.py and training_runner.py)

### Analysis (`agentick/analysis/`)
- **Deleted**: `comparisons.py`, `learning_curves.py`, `README.md`
- **Kept**: `statistics.py`, `metrics.py`

---

## Phase 5: Leaderboard Rework

### 5A. Simplified database
- **File**: `agentick/leaderboard/database.py` — Replaced complex class with 4 functions: `load_entries()`, `save_entries()`, `add_entry()`, `get_entries()`
- Schema: JSON file at `leaderboard_data/entries.json`

### 5B. Created `scripts/validate_submission.py`
- Public script for users to validate their results directory
- Checks: 38 tasks present, 4 difficulties each, seeds match official eval seeds, episode counts correct
- Computes scores, prints detailed report, packages into submission zip
- Prints email instructions for `roger.creus-castanyer@mila.quebec`

### 5C. Created `scripts/publish_to_leaderboard.py`
- Admin-only script: unpacks zip, validates, adds to entries.json, regenerates site

### 5D. Updated site templates
- `index.html`: Sortable ranking table with score bars, per-category tabs
- `submit.html`: Zip-based submission instructions
- `generate.py`: Updated to use new `load_entries` API

### 5E. Created `leaderboard_data/entries.json`
- Empty initial structure for first entries

---

## Phase 6: pyproject.toml & Dependency Cleanup

### Core dependencies (always installed)
`gymnasium>=1.0`, `numpy>=1.24`, `pygame>=2.5`, `Pillow>=10.0`, `pydantic>=2.0`, `rich>=13.0`, `scipy`, `pyyaml`, `python-dotenv`, `tqdm`

### Extras
| Extra | Dependencies |
|-------|-------------|
| `rl` | torch>=2.0, stable-baselines3>=2.0, tensorboard |
| `llm` | openai>=1.0, google-genai>=1.66.0, transformers>=4.40, torch>=2.0, accelerate |
| `vllm` | vllm>=0.12.0 |
| `finetune` | trl>=0.8, peft, datasets |
| `tracking` | wandb>=0.17 |
| `viz` | matplotlib>=3.8, seaborn>=0.13, plotly>=5.0 |
| `webapp` | flask>=3.1 (NEW) |
| `docs` | mkdocs>=1.5, mkdocs-material>=9.0, mkdocstrings[python] |
| `all` | All of the above |

**Removed**: `anthropic>=0.30` from llm, `docker` from leaderboard, `leaderboard` extra entirely, `notebooks` extra

---

## Phase 7: Documentation Overhaul

| File | Changes |
|------|---------|
| `docs/getting_started/quickstart.md` | Fixed extras list (removed local-llm/train-llm), removed rgb_array_flat example |
| `docs/agents/finetuning.md` | Removed BehaviorCloningTrainer and Tinker sections |
| `docs/agents/llm_agents.md` | Rewrote: removed Anthropic, NonMarkovian refs. Updated backends table, harness table, YAML example, direct usage example |
| `docs/agents/rl_agents.md` | Changed rgb_array_flat to rgb_array |
| `docs/concepts/observations.md` | Removed "RGB Array Flat Mode" section |
| `docs/concepts/architecture.md` | Replaced rgb_array_flat row with human mode |
| `docs/index.md` | Removed rgb_array_flat refs, fixed LLM example (removed APIAgent) |
| `docs/leaderboard.md` | **NEW** — Submission flow, scoring methodology, category scores |
| `docs/results_format.md` | **NEW** — Canonical results directory schema |
| `mkdocs.yml` | Added Results Format + Leaderboard pages to nav |

---

## Phase 8: README Overhaul

- Removed rgb_array_flat from render modes table
- Updated dependency groups to match new extras
- Added "Try It First" section with webapp command at top
- Added "Experiment Runner" section showing YAML config approach
- Added "Leaderboard" section with submission instructions
- Added "Roadmap" section listing features on dev branch
- Updated project structure (removed rendering/, updated training/ description)
- Updated documentation links

---

## Phase 9: Examples Verification

- **Rewrote** `examples/llm/openai_text_agent.py` — replaced deleted `APIAgent` import with new harness system (`BaseAgent` + `OpenAIBackend` + `MarkovianZeroShot`)
- **Verified** remaining examples have no stale imports

---

## Phase 10: Final Validation

### Test Results
| Metric | Value |
|--------|-------|
| Tests passed | 1186 |
| Tests failed | 14 (all pre-existing) |
| Tests skipped | 17 |

### Pre-existing failures (unchanged)
- 9x gymnasium compliance RNG tests (DistributionShift, DynamicObstacles, EmergentStrategy, FogOfWar, GoToGoal, Herding, NoisyObservation, SequenceMemory, TagHunt)
- 2x ChaseEvade difficulty level tests (stochastic)
- 1x ChaseEvade recording test
- 1x recording wrapper test
- 1x test_episode_recorder

### Deleted tests (for removed modules)
- `test_profiler.py`, `test_profiling.py`, `test_throughput.py` (benchmark/)
- `test_random_baseline.py` (learning_curves/)
- `test_submission.py`, `test_evaluator.py`, `test_comparison.py`, `test_adapters.py` (leaderboard/)
- `test_plots.py` (visualization/)

### Fixed tests
- `test_renderer.py`: Removed PixelRenderer tests and import
- `test_all_render_modes.py`: Removed rgb_array_flat from RENDER_MODES list
- `test_visual_regression.py`: Changed rgb_array_flat to rgb_array
- `test_docs_code_blocks.py`: Updated incorrect import patterns (adapters removed, agents is correct)
- `test_cli.py`: Updated evaluate/verify command tests to match new CLI
- `test_database.py`: Rewrote for new JSON I/O API
- `test_full_pipeline.py`: Removed imports of deleted plots/tables modules

### Lint
- Zero F821 (undefined name) errors
- All new/modified files pass ruff format

---

## Files Summary

### New files (5)
- `agentick/agents/prompt_templates.py` (moved from adapters/)
- `agentick/experiments/_video_utils.py` (extracted from deleted video.py)
- `docs/leaderboard.md`
- `docs/results_format.md`
- `leaderboard_data/entries.json`

### New scripts (2)
- `scripts/validate_submission.py`
- `scripts/publish_to_leaderboard.py`

### Deleted files (107)
- 4 packages: curriculum/ (4), benchmark/ (6), rewards/ (5), vector/ (3)
- Training: tinker/ (4), behavior_cloning.py, README.md
- Agents: anthropic_backend.py
- Core: simple_grid_renderer.py, README.md
- Leaderboard: adapters/ (9), evaluator.py, submission.py, comparison.py, README.md
- Logging: 4 stale files + README.md
- Visualization: 6 stale files + README.md
- Analysis: 2 stale files + README.md
- Examples: 18 files (debug/, stale llm/, stale rl/, stale training/)
- Configs: 16 YAML files (claude, nonmarkov, rgbFlat)
- Tests: 8 files for deleted modules

### Modified files (33)
- Core: env.py, renderer.py
- Agents: __init__.py, backends/__init__.py, base.py, factory.py, harness.py, observation.py
- Experiments: batched_runner.py, config.py, runner.py, training_runner.py
- Leaderboard: cli.py, cost_tracker.py, database.py, integrity.py, result.py, scoring.py, site/generate.py
- Training: __init__.py, callbacks.py
- Logging: __init__.py
- Wrappers: atari_preprocessing.py
- Docs: 7 files
- Examples: openai_text_agent.py
- Config: mkdocs.yml, pyproject.toml
- Tests: 7 files
- CLAUDE.md, README.md
