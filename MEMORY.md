# Agentick — Codebase Memory

> Living reference document. Updated as edits are made.

---

## 1. Project Overview

**Agentick** is a universal benchmark for evaluating AI agents (RL, LLM, VLM, hybrid, human) across **38 procedurally generated gridworld tasks**. Implements the Gymnasium API with multi-modal observations.

| Field | Value |
|-------|-------|
| Python | 3.11+ (`from __future__ import annotations`, PEP 604 unions) |
| Build backend | hatchling |
| Package manager | uv |
| License | MIT |
| Entry point | `agentick = agentick.leaderboard.cli:main` |

---

## 2. Directory Layout

```
agentick/                  # Main package (~160 modules, 18 subpackages)
  core/                    # Base env, grid, renderers, types (10 files, ~2,605 lines)
  tasks/                   # 38 tasks across 10 categories (52 files, ~8,180 lines)
  wrappers/                # Gymnasium wrappers (6 files)
  generation/              # Procedural generation & validation (6 files)
  analysis/                # Statistical analysis (5 files)
  vector/                  # Vectorized environments (2 files)
  data/                    # Trajectory collection & export (4 files)
  training/                # Callbacks & loggers (3 files)
  visualization/           # Plots, tables, video (8 files)
  logging/                 # Agent/episode/experiment logging (7 files)
  human/                   # Human baselines & player (6 files)
  curriculum/              # Curriculum learning (3 files)
  rewards/                 # Reward shaping (4 files)
  benchmark/               # Baselines & metrics (6 files)
  leaderboard/             # Evaluation, submissions, adapters (25 files)
  experiments/             # Config, runner, reproduce (5 files)
  utils/                   # Video recording helpers (2 files)
tests/                     # 17 categories, 53 files, ~6,786 lines
examples/                  # 55+ scripts, 9 categories
docs/                      # MkDocs documentation
scripts/                   # Dev utility scripts
paper/                     # Research paper materials
leaderboard_site/          # Web frontend
.github/workflows/         # CI/CD (tests, docs, leaderboard)
```

---

## 3. Core Module (`agentick/core/`)

### `types.py` (136 lines)
- **`Position`** = `tuple[int, int]` — (x, y) coordinates
- **`CellType`** (IntEnum): EMPTY(0), WALL(1), HAZARD(2), WATER(3), ICE(4), HOLE(5)
- **`ObjectType`** (IntEnum): NONE(0), GOAL(1), KEY(2), DOOR(3), SWITCH(4), BOX(5), TARGET(6), TOOL(7), RESOURCE(8), BREADCRUMB(9), NPC(10), ENEMY(11), SHEEP(12), BLOCKER(13)
- **`AgentType`** (IntEnum): NONE(0), AGENT(1), NPC(2), ENEMY(3)
- **`Direction`** (IntEnum): NORTH(0), EAST(1), SOUTH(2), WEST(3) — has `opposite()`, `rotate_left()`, `rotate_right()`, `to_delta()`
- **`ActionType`** (IntEnum): NOOP, MOVE_UP/DOWN/LEFT/RIGHT, PICKUP, DROP, USE, INTERACT, ROTATE_LEFT/RIGHT, MOVE_FORWARD
- **`COLORS`**: dict mapping entity names to RGB tuples
- **`ASCII_CHARS`**: dict mapping types to display characters

### `entity.py` (217 lines)
- **`Entity`** (dataclass): id, entity_type, position, properties; copy(), to_dict(), from_dict()
- **`Agent`** (extends Entity): orientation, inventory, energy, health; add_to_inventory(), remove_from_inventory(), has_item(), get_item()
- **`EntityRegistry`**: register/create custom entity types; global `_global_registry`

### `grid.py` (247 lines)
- **`Grid`**: height, width + 4 numpy layers:
  - `terrain` (int8) — CellType values
  - `objects` (int8) — ObjectType values
  - `agents` (int8) — AgentType values
  - `metadata` (int16) — custom per-cell data
- Methods: `in_bounds()`, `is_walkable()`, `get_neighbors()`, `line_of_sight()` (Bresenham), `flood_fill()`, `bfs()`, `manhattan_distance()`, `copy()`, serialization
- **Coordinate convention**: positions are `(x, y)` but array indexing is `[y, x]`

### `actions.py` (234 lines)
- **`ActionSpace`**: 9 basic or 12 extended actions; gym_space, bidirectional idx-ActionType mappings, `sample()`, `parse_action_name()`
- **`compute_action_mask()`**: binary mask based on grid walkability, inventory, objects
- Helpers: `get_move_delta()`, `is_movement_action()`, `action_to_direction()`

### `env.py` (460 lines)
- **`AgentickEnv`** (extends `gym.Env`): the base environment
  - 7 render_modes: ascii, language, language_structured, rgb_array, rgb_array_2d, human, state_dict
  - reward_mode: sparse / dense / custom
  - Gymnasium API: `reset()`, `step()`, `render()`, `close()`
  - Multi-modal access: `get_text_observation()`, `get_pixel_observation()`, `get_state_dict()`
  - `get_valid_actions()` with caching
  - Lifecycle hooks for subclasses: `_reset_state()`, `_execute_action()`, `_move_agent()`, `_pickup()`, `_drop()`, `_compute_reward()`, `_check_success()`, `_get_observation()`, `_get_info()`, `_get_state_for_reward()`
  - `fast_mode=True` skips `.tolist()` conversions in state_dict

### `renderer.py` (426 lines)
- **`Renderer`** (Protocol): `render(grid, entities, agent, info) -> Any`
- **`RenderConfig`**: tile_size=32, show_grid, show_fog, show_hud, use_ansi_colors, font_size
- **`ASCIIRenderer`**: ANSI-colored terminal grid with legend, agent orientation (^ v < >)
- **`EnhancedLanguageRenderer`**: wraps AdvancedLanguageRenderer; structured/text, verbosity, perspective
- **`StateDictRenderer`**: fast_mode support; returns dict with grid/agent/entities/info
- **`create_renderer(mode, ...)`**: factory function

### `language.py` (576 lines)
- **`LanguageConfig`**: verbosity (minimal/standard/verbose), perspective (first_person/third_person/omniscient), flags for memory/goals/threats/spatial
- **`AdvancedLanguageRenderer`**: rich NL observations with spatial reasoning, relative directions, inventory, status, goals, threats, memory tracking (100 entries)

### `simple_grid_renderer.py` (182 lines)
- **`SimpleGridRenderer`**: PIL-based 2D sprites for rgb_array mode
  - tile_size auto-aligned to 16 (FFMPEG compatible), HEADER_H=32
  - Squares for collectibles, circles for movable entities
  - Contrast-aware text color (black on light, white on dark)

### `feature_extractor.py` (117 lines)
- **`extract_state_features(state_dict, grid_size=(15,15))`** -> flat float32 array (456 dims for 15x15)
- **`get_state_feature_space(grid_size)`** -> `spaces.Box`

---

## 4. Tasks Module (`agentick/tasks/`)

### Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `base.py` | 199 | **TaskSpec** ABC: `generate(seed)`, `compute_dense_reward()`, `check_success()`, `check_done()`, optional hooks `on_env_reset/step()`, `on_agent_moved()`, `can_agent_enter()` |
| `registry.py` | 377 | **TaskEnv** (wraps AgentickEnv + TaskSpec), `make()`, `register_task()`, `list_tasks()`, `make_suite()`, `EnvSpec` |
| `configs.py` | 50 | **DifficultyConfig** (name, grid_size, max_steps, params), **GridConfig**, **TaskMetadata** (Pydantic v2) |

### 38 Registered Tasks

| Category | # | Tasks (registered names) |
|----------|---|--------------------------|
| **Navigation** | 5 | GoToGoal-v0, MazeNavigation-v0, MultiGoalRoute-v0, DynamicObstacles-v0, FogOfWarExploration-v0 |
| **Memory** | 5 | KeyDoorPuzzle-v0, SequenceMemory-v0, BreadcrumbTrail-v0, DelayedGratification-v0, BacktrackPuzzle-v0 |
| **Reasoning** | 5 | SokobanPush-v0, SwitchCircuit-v0, SymbolMatching-v0, CausalChain-v0, RuleInduction-v0 |
| **Skill** | 5 | ToolUse-v0, RecipeAssembly-v0, MultiRoomEscape-v0, ResourceManagement-v0, EmergentStrategy-v0 |
| **Control** | 4 | PreciseNavigation-v0, TimingChallenge-v0, ChaseEvade-v0, Herding-v0 |
| **Combinatorial** | 4 | LightsOut-v0, TileSorting-v0, GraphColoring-v0, PackingPuzzle-v0 |
| **Compositional** | 3 | RecursiveRooms-v0, ProgramSynthesis-v0, InstructionFollowing-v0 |
| **Adversarial** | 3 | NoisyObservation-v0, DeceptiveReward-v0, DistributionShift-v0 |
| **Multi-Agent** | 2 | CooperativeTransport-v0, CompetitiveTag-v0 |
| **Meta** | 2 | FewShotAdaptation-v0, TaskInterference-v0 |

### Difficulty Scaling Pattern
Each task defines 4 levels via `difficulty_configs`:
- **easy**: ~5x5 grid, ~20 max_steps, minimal obstacles
- **medium**: ~10x10 grid, ~50 max_steps, moderate obstacles
- **hard**: ~15x15 grid, ~100 max_steps, significant obstacles
- **expert**: ~20x20 grid, ~200 max_steps, dense obstacles

Task-specific `params` dict keys: `wall_density`, `n_guards`, `n_hazards`, `n_keys`, `n_doors`, `n_boxes`, `n_targets`, `n_decoys`, etc.

### Common Task Patterns
1. **Goal Reaching** — Navigate to goal position (GoToGoal, MazeNavigation, DynamicObstacles)
2. **Key-Door Chaining** — Collect keys to unlock doors (KeyDoorPuzzle, ToolUse) via `can_agent_enter()` + `on_agent_moved()`
3. **Box Pushing** — Sokoban-style mechanics (SokobanPush) via `can_agent_enter()`
4. **Multi-Objective Control** — Manage NPCs/entities (ChaseEvade, Herding) via `on_env_step()`
5. **Adversarial** — True goal hidden among decoys (NoisyObservation, DeceptiveReward)

### Task Implementation Notes
- All tasks use `numpy.random.default_rng(seed)` with 10-attempt retry loops
- Validation via `verify_solvable()` and `find_optimal_path()` from `generation/validation.py`
- Dynamic state stored in `config` dict (NPC positions, door status, collected items)
- Dense rewards typically: -0.01 step cost + distance shaping + achievement bonuses + 1.0 on success

---

## 5. Supporting Modules

### `wrappers/` — Gymnasium Wrappers
- `observation_wrappers.py`: TextObservationWrapper, PixelObservationWrapper, DictObservationWrapper, FlattenObservationWrapper, LanguageActionWrapper
- `reward_wrappers.py`: DenseRewardWrapper, SparseRewardWrapper, RewardScaleWrapper, CurriculumWrapper
- `recording_wrappers.py`: EpisodeRecorder (all modalities + timestamps), TrajectoryWrapper
- `atari_preprocessing.py`: ResizeObservation, GrayscaleObservation, FrameStack, `make_atari_env()` (84x84 pipeline)
- `state_features_wrapper.py`: StateFeaturesWrapper (uses feature_extractor)

### `generation/` — Procedural Generation
- `maze.py`: **MazeGenerator** — 5 algorithms: recursive_backtracker, prims, kruskals, binary_tree, recursive_division; post-processing (add loops, remove dead ends)
- `room.py`: **RoomGenerator** — BSP rooms, random_rooms_with_corridors, place_key_door_sequence
- `validation.py`: **SolvabilityValidator** — validates navigation, key-door, box-pushing, switch, inventory puzzles; `find_optimal_path()`, `compute_solution_stats()`
- `difficulty.py`: **DifficultyEstimator** — weighted metrics (solution_length 0.3, branching 0.2, subgoals 0.25, wall_density 0.1, turns 0.15)
- `seed.py`: **SeedManager** — benchmark seeds, multi-stage spawning, reproducible generation

### `analysis/` — Statistical Analysis
- `statistics.py`: bootstrap_ci, permutation_test, welch_t_test, mann_whitney_u, cohens_d, cliff_delta, holm_bonferroni, benjamini_hochberg, iqr_outlier_detection
- `metrics.py`: normalized_score, agentick_score, capability_profile, learning_curves
- `comparisons.py`: compare_agents (full statistical comparison)

### `vector/` — Vectorized Environments
- `vector_env.py`: SyncVectorAgentickEnv, AsyncVectorAgentickEnv, `make_vec_env()`

### `data/` — Trajectory Collection & Export
- `collector.py`: Trajectory (dataclass), TrajectoryCollector
- `demonstrations.py`: `collect_oracle_trajectories()` with random fallback
- `formats.py`: `export_to_format()` -> JSONL, HuggingFace Datasets, D4RL HDF5, conversation (LLM fine-tuning)

### `training/` — Training Utilities
- `callbacks.py`: EvalCallback, CurriculumCallback, CheckpointCallback
- `logger.py`: MultiBackendLogger (stdout, JSON, wandb, tensorboard)

### `visualization/` — Plots & Video
- `plots.py`: plot_bar_comparison, plot_learning_curves, plot_heatmap, plot_sample_efficiency
- `style.py`: set_style, agent color/marker/linestyle, save_figure
- `tables.py`: render_results_table (Markdown/LaTeX)
- `video.py`, `interactive.py`, `experiment_plots.py`

### `logging/` — Agent & Episode Logging
- `agent_logger.py`: LoggableAgent (Protocol), LLMAgentLogger, RLAgentLogger, SearchAgentLogger
- `episode_logger.py`, `experiment_logger.py`, `llm_logger.py`, `replay.py`

### `human/` — Human Evaluation
- `baselines.py`: HUMAN_BASELINES dict, `get_human_baseline()`, `compare_to_human()`, `estimate_human_performance()`
- `player.py`: HumanPlayer; `recorder.py`: HumanDataRecorder; `analysis.py`: HumanBaselineAnalyzer

### `curriculum/` — Curriculum Learning
- `adaptive.py`: AdaptiveCurriculum — sliding window success, auto-advance/regress (easy->medium->hard->expert)
- `manual.py`: ManualCurriculum — predefined stages

### `rewards/` — Reward Shaping
- `intrinsic.py`: ExplorationBonus (visit counts), CuriosityReward
- `potential.py`: PotentialBasedReward (Ng et al. 1999)
- `composite.py`: CompositeReward — weighted combination with component breakdown

### `benchmark/` — Baseline Agents & Metrics
- `baselines.py`: RandomAgent, GreedyAgent, OracleAgent
- `metrics.py`: success_rate, average_return, normalized_score, sample_efficiency, generalization_score, capability_profile, agentick_score
- `suite.py`: BenchmarkRunner, `get_suite()`

### `leaderboard/` — Official Evaluation System
- `suites.py`: CORE_TASKS (27), FULL_TASKS (35), per-category task lists, BenchmarkSuite, ScoringConfig
- `evaluator.py`: LeaderboardEvaluator — agents from api/code/huggingface/local_weights/docker/git_repo
- `submission.py`: SubmissionSpec (from_yaml, validate, to_dict/yaml)
- `result.py`: EpisodeResult, EvaluationResult
- `scoring.py`, `integrity.py`, `cost_tracker.py`
- `cli.py`: cmd_evaluate, cmd_verify, cmd_list_tasks, cmd_list_suites, cmd_info
- `adapters/`: api, code, huggingface, local_weights, docker, git adapters

### `experiments/` — Experiment Management
- `config.py`, `registry.py`, `runner.py`, `reproduce.py`

### `utils/`
- `video.py`: `record_episode()`, `record_episodes()` (imageio MP4/GIF)

---

## 6. Tests

**53 test files across 17 categories** (~6,786 total lines)

| Category | Files | Coverage Focus |
|----------|-------|----------------|
| test_tasks | 4 | Task behavior, registry, individual tasks, all_tasks_behavioral |
| test_core | 5 | Environment, grid, actions, renderer, entity |
| test_performance | 4 | Throughput, profiling, step speed benchmarks |
| test_leaderboard | 10 | Submissions, evaluation, rankings, database, CLI |
| test_logging | 2 | Episode logging, wandb integration |
| test_wrappers | 4 | Observation, reward, atari preprocessing, recording |
| test_human | 2 | Human baselines, web interface |
| test_integration | 6 | Render modes, reward modes, determinism, compliance |
| test_visualization | 2 | Experiment plots, visualization tools |
| test_rendering | 1 | Visual regression tests |
| test_experiments | 2 | Config loading, experiment runner |
| test_learning_curves | 4 | Solvability, greedy beats random, oracle optimal |
| test_e2e | 2 | Full pipeline, reproducibility |
| test_examples | 4 | Example imports and execution |
| test_docs | 1 | Documentation code blocks |
| test_analysis | 1 | Statistics and metrics (**flaky** — p-value randomness) |

---

## 7. Examples

**55+ scripts across 9 categories:**

| Category | Count | Highlights |
|----------|-------|-----------|
| basics | 5 | make/step, render modes, reward modes, task listing, difficulty |
| rl | 8 | SB3 PPO/DQN, cleanRL, pixel/curriculum training |
| llm | 10 | Anthropic, OpenAI, HuggingFace; vision & text variants |
| plotting | 7 | Radar charts, heatmaps, learning curves, tables |
| data_and_finetuning | 6 | Trajectory collection, oracle/random, TRL SFT, HF export |
| leaderboard | 5 | Submissions, evaluation, comparison, validation |
| experiments | 5 | Config runners, benchmarks, paper figures |
| advanced | 4 | Custom rewards, tasks, human play, parallel evaluation |

---

## 8. Commands

```bash
# Install
uv sync --extra all          # all extras
uv sync --group dev          # dev tools

# Test
uv run pytest tests/ -v
uv run pytest tests/test_tasks/test_all_tasks_behavioral.py -v
uv run pytest tests/test_core/test_renderer.py::TestASCIIRenderer::test_basic_render -v

# Lint & format
uv run ruff check agentick/ tests/
uv run ruff format agentick/ tests/

# Type check
uv run mypy agentick/

# CLI
uv run agentick list-tasks
uv run agentick list-suites

# Makefile
make test          # run all tests
make test-cov      # tests with coverage
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy
make docs          # build MkDocs
make build         # build distribution
make benchmark     # performance benchmarks
make all           # lint + typecheck + test + build
```

---

## 9. Code Style

| Rule | Value |
|------|-------|
| Line length | 100 |
| Ruff rules | E, F, W, I, N, UP |
| Python target | 3.11 |
| Docstrings | Google style |
| Config models | Pydantic v2 |
| Naming | snake_case functions, CamelCase classes, UPPER_CASE constants |
| Type hints | Required for public APIs |

---

## 10. Architecture Notes

### Coordinate System
- Positions: `(x, y)` — x is column, y is row
- Array indexing: `grid[y, x]` (numpy row-major)
- Origin: (0, 0) at top-left
- Directions: NORTH=(0,-1), EAST=(1,0), SOUTH=(0,1), WEST=(-1,0)

### Environment Stack
```
agentick.make("GoToGoal-v0")
  -> registry looks up TaskSpec subclass
  -> TaskSpec.generate(seed) -> (Grid, config)
  -> TaskEnv(task, grid, config) wraps AgentickEnv
```

### Rendering Pipeline
```
env.step() -> env._get_observation() -> env.render()
  -> create_renderer(mode) -> Renderer.render(grid, entities, agent, info)
```

### Grid Layers
```
terrain (int8)  -> CellType enum (walls, hazards, water, ice, holes)
objects (int8)  -> ObjectType enum (goals, keys, doors, boxes, NPCs)
agents  (int8)  -> AgentType enum (agent, NPC, enemy)
metadata(int16) -> Custom per-cell data (task-specific)
```

### Public API (`agentick/__init__.py`)
```python
make, list_tasks, register_task, make_suite
AgentickEnv, Grid, Entity, Agent, ActionSpace, ActionType, TaskSpec
```

---

## 11. Known Issues / Gotchas

- `test_analysis/test_statistics.py` is flaky (p-value randomness) — not a real failure
- 3D rendering was fully removed — no `render_3d`, no `agentick/rendering/` module
- Grid coordinates: positions are `(x, y)` but array indexing is `[y, x]` — easy to mix up
- `fast_mode=True` in state_dict keeps numpy arrays (no `.tolist()`) — important for RL throughput

---

## 12. Changelog

_Track all subsequent edits below._

| Date | Summary |
|------|---------|
| 2026-02-19 | Initial MEMORY.md created from full codebase read |
