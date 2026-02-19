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

Tasks use the `@register_task("Name-v0", tags=[...])` decorator (from `tasks/registry.py`). Task modules are auto-imported via `tasks/__init__.py`, which imports all category subpackages. Each category (navigation, memory, reasoning, skill, control, combinatorial, adversarial, meta, multi_agent, compositional) is a subpackage under `tasks/`.

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
- `"rgb_array"` — 2D pixel sprites via `SimpleGridRenderer` (in `core/simple_grid_renderer.py`)
- `"rgb_array_2d"` — 2D sprite renderer with isometric perspective
- `"state_dict"` — structured dict with numpy arrays (use `fast_mode=True` to skip `.tolist()` conversions)

### Observation Flow

`env.step()` → `env._get_observation()` → `env.render()` → dispatches to the configured renderer. Each renderer implements the `Renderer` protocol: `render(grid, entities, agent, info) → Any`.

## Code Style

- Line length: 100 characters
- Ruff rules: E, F, W, I, N, UP
- Python 3.11+ (uses `from __future__ import annotations`, `X | Y` unions, etc.)
- Google-style docstrings
- Pydantic v2 for config models (`DifficultyConfig`, `GridConfig`)

## Known Issues

- `test_analysis/test_statistics.py` is flaky due to p-value randomness — not a real failure
