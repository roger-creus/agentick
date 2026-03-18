# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentick is a universal benchmark for evaluating AI agents (RL, LLM, VLM, hybrid, human) across 38 procedurally generated gridworld tasks. It implements the Gymnasium API and supports multi-modal observations (ASCII, language, pixels, state dict).

## Common Commands

```bash
# Install dependencies
uv sync --extra all          # all extras (rl, llm, viz, etc.)
uv sync --group dev          # dev tools (pytest, ruff, mypy)

# Run all tests
uv run pytest tests/ -v

# Run a single test file or test
uv run pytest tests/test_tasks/test_all_tasks_behavioral.py -v
uv run pytest tests/test_core/test_renderer.py::TestASCIIRenderer::test_basic_render -v

# Lint and format
uv run ruff check agentick/ tests/
uv run ruff format agentick/ tests/

# Type check
uv run mypy agentick/

# CLI
uv run agentick list-tasks
uv run agentick list-suites
```

## Architecture

### Core Environment Stack

`AgentickEnv` (in `core/env.py`) is the Gymnasium `gym.Env` base class. It owns the `Grid`, `Agent`, renderer, and action space. Tasks don't subclass `AgentickEnv` directly — they subclass `TaskSpec` (in `tasks/base.py`), and the `TaskEnv` wrapper (in `tasks/registry.py`) bridges them:

```
agentick.make("GoToGoal-v0")
  → registry looks up TaskSpec subclass
  → TaskSpec.generate(seed) → (Grid, config)
  → TaskEnv(task, grid, config) wraps AgentickEnv
```

`TaskEnv` delegates reward, success checking, movement rules, and per-step hooks back to the `TaskSpec`:
- `_compute_reward()` → calls `task.compute_sparse_reward()` or `task.compute_dense_reward()`
- `_check_success()` → calls `task.check_done()` and `task.check_success()`
- `_move_agent()` → calls `task.can_agent_enter()` and `task.on_agent_moved()` if defined
- `step()` → calls `task.on_env_step()` for NPC/obstacle movement

### Grid Data Model

`Grid` (in `core/grid.py`) has four numpy `int8`/`int16` layers: `terrain`, `objects`, `agents`, `metadata`. Positions are `(x, y)` tuples but array indexing is `[y, x]`. The enum types in `core/types.py` (`CellType`, `ObjectType`, `ActionType`, `Direction`) define the valid integer values.

### Task Registration

Tasks use the `@register_task("Name-v0", tags=[...])` decorator (from `tasks/registry.py`). Task modules are auto-imported via `tasks/__init__.py`, which imports all category subpackages. Each category (navigation, planning, reasoning, memory, generalization, multi_agent) is a subpackage under `tasks/`.

### Creating a New Task

1. Create `agentick/tasks/<category>/your_task.py`
2. Subclass `TaskSpec` and implement:
   - `difficulty_configs` dict mapping "easy"/"medium"/"hard"/"expert" → `DifficultyConfig`
   - `generate(seed)` → `(Grid, config_dict)` with at least `agent_start`, `goal_positions`, `max_steps`
   - `compute_dense_reward(old_state, action, new_state, info)` → float
   - `check_success(state)` → bool (check `state["grid"]` and `state["agent"]`)
3. Decorate with `@register_task("YourTask-v0", tags=[...])`
4. Import in the category's `__init__.py`
5. Use `generation/validation.py` (`verify_solvable`, `find_optimal_path`) to ensure solvability

### Rendering Modes

- `"ascii"` — ANSI-colored text grid (`ASCIIRenderer`)
- `"language"` / `"language_structured"` — natural language descriptions (`EnhancedLanguageRenderer` wrapping `AdvancedLanguageRenderer` from `core/language.py`)
- `"rgb_array"` — **Isometric pixel sprites** via `IsometricRenderer` (Kenney assets, fixed 512×512 output). Default visual mode.
- `"state_dict"` — structured dict with numpy arrays (use `fast_mode=True` to skip `.tolist()` conversions)

### Observation Flow

`env.step()` → `env._get_observation()` → `env.render()` → dispatches to the configured renderer. Each renderer implements the `Renderer` protocol: `render(grid, entities, agent, info) → Any`.

### Agent Harness System

`agents/` provides a composable harness for LLM/VLM agents:

- `BaseAgent` (in `agents/base.py`) composes a `ModelBackend` + `HarnessPreset` and conforms to `AgentProtocol`
- **Backends** (`agents/backends/`): OpenAI, Gemini, HuggingFaceLLM, HuggingFaceVLM, vLLM — lazy-loaded to avoid import overhead
- **Harness presets** (`agents/harness.py`): MarkovianZeroShot, MarkovianReasoner — control prompting strategy
- **Factory**: `create_agent(AgentConfig)` builds an agent from a YAML config
- **Circular import note**: `experiments/__init__.py` uses `__getattr__` lazy import for agent classes to break `agents/` ↔ `experiments/` circular dependency

### Oracle System

`oracles/` provides hand-coded optimal or near-optimal policies for each task, organized by category (one file per category: `navigation_oracles.py`, `planning_oracles.py`, `reasoning_oracles.py`, `memory_oracles.py`, `generalization_oracles.py`, `multi_agent_oracles.py`). Used for:

- Generating expert trajectories for behavior cloning / SFT
- Verifying task solvability
- Establishing score upper bounds

API: `get_oracle(task_name, env)` returns an oracle instance; `list_oracles()` lists all available oracles. Oracle base class is in `oracles/base.py`.

### Experiment System

`experiments/` provides reproducible evaluation:

- `ExperimentRunner` (in `experiments/runner.py`) runs episodes with any agent
- `experiments/config.py` defines YAML-based experiment configs
- CLI: `python -m agentick.experiments.run --config path/to/config.yaml`

### Training Infrastructure

`training/` provides training callbacks (EvalCallback, CheckpointCallback). For SFT, use TRL directly — see `examples/data_and_finetuning/sft_with_trl.py`.

### Data Collection

`data/collector.py` provides `DataCollector` for recording agent trajectories and exporting to HuggingFace datasets format.

## Code Style

- Line length: 100 characters
- Ruff rules: E, F, W, I, N, UP
- Python 3.11+ (uses `from __future__ import annotations`, `X | Y` unions, etc.)
- Google-style docstrings
- Pydantic v2 for config models (`DifficultyConfig`, `GridConfig`)

### INTERACT Action

Some tasks use the INTERACT action (`ActionType.INTERACT`). When dispatched, `TaskEnv` calls `task.on_agent_interact(pos, agent, grid)` on the `TaskSpec`. Tasks using INTERACT include GraphColoring, RuleInduction, ToolUse, and others.

## Known Issues

- `test_analysis/test_statistics.py` is flaky due to p-value randomness — not a real failure
- `test_integration/test_all_reward_modes.py` DelayedGratification test is flaky
- `test_tasks/test_all_tasks_behavioral.py::test_cooperative_transport_can_succeed` is flaky
- Pre-existing N806 ruff errors in `env.py`, `renderer.py` etc. — do not fix these
- `PixelRenderer` alias in `core/renderer.py` has suppressed ruff warnings (`# noqa: E402, F401, I001, N812`) — do not remove
