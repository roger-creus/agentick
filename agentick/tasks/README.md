# Agentick Tasks

Procedurally generated gridworld tasks for benchmarking AI agents across 10 capability categories.

## TaskSpec Base Class

Every task subclasses `TaskSpec` (in `base.py`) and implements three core methods:

- **`generate(seed) -> (Grid, config_dict)`** -- Procedurally create a grid and return it with a config dict containing at least `agent_start`, `goal_positions`, and `max_steps`.
- **`compute_dense_reward(old_state, action, new_state, info) -> float`** -- Shaped reward signal for RL training.
- **`check_success(state) -> bool`** -- Whether the agent has achieved the task objective.

Optional overrides include `compute_sparse_reward` (defaults to 1.0 on success), `check_done` (for episodes that can end without success), `validate_instance` (solvability check), and hooks like `on_env_step`, `on_env_reset`, `can_agent_enter`, and `on_agent_moved`.

## Task Registration

Tasks register themselves with the `@register_task` decorator from `registry.py`:

```python
from agentick.tasks.base import TaskSpec
from agentick.tasks.registry import register_task

@register_task("MyTask-v0", tags=["navigation", "memory"])
class MyTask(TaskSpec):
    ...
```

The decorator adds the class to a global registry. All task modules are auto-imported via `tasks/__init__.py`, which imports each category subpackage. The registry provides `list_tasks()`, `get_task_class()`, `make()`, and `make_suite()`.

## DifficultyConfig

Each task defines a `difficulty_configs` dict mapping four standard levels to `DifficultyConfig` objects (from `configs.py`):

| Level    | Typical grid | Typical max_steps |
|----------|-------------|-------------------|
| easy     | 5x5         | 20                |
| medium   | 10x10       | 50                |
| hard     | 15x15       | 100               |
| expert   | 20x20       | 200               |

`DifficultyConfig` is a Pydantic model with `name`, `grid_size`, `max_steps`, and a `params` dict for task-specific scaling parameters (wall density, number of obstacles, etc.).

## TaskEnv Bridge

`TaskEnv` (in `registry.py`) subclasses `AgentickEnv` and bridges it to a `TaskSpec`. It delegates:

- `_compute_reward()` to `task.compute_sparse_reward()` or `task.compute_dense_reward()`
- `_check_success()` to `task.check_done()` and `task.check_success()`
- `_move_agent()` to `task.can_agent_enter()` and `task.on_agent_moved()`
- `step()` calls `task.on_env_step()` for NPC/obstacle movement
- `_reset_state()` calls `task.generate()` to regenerate the grid on reset

The entry point is `agentick.make("TaskName-v0")`, which looks up the task class, instantiates it, calls `generate(seed)`, validates the instance, and wraps everything in a `TaskEnv`.

## Task Categories

| Category         | Tasks |
|------------------|-------|
| **navigation**   | `go_to_goal`, `maze_navigation`, `multi_goal_route`, `dynamic_obstacles`, `fog_of_war` |
| **memory**       | `key_door_puzzle`, `sequence_memory`, `breadcrumb_trail`, `delayed_gratification`, `backtrack_puzzle` |
| **reasoning**    | `sokoban_push`, `switch_circuit`, `symbol_matching`, `rule_induction` |
| **skill**        | `tool_use`, `recipe_assembly`, `resource_management`, `emergent_strategy` |
| **control**      | `precise_navigation`, `timing_challenge`, `chase_evade`, `herding` |
| **combinatorial**| `lights_out`, `tile_sorting`, `graph_coloring`, `packing_puzzle` |
| **multi_agent**  | `cooperative_transport`, `tag_hunt` |
| **compositional**| `recursive_rooms`, `program_synthesis`, `instruction_following` |
| **adversarial**  | `noisy_observation`, `deceptive_reward`, `distribution_shift` |
| **meta**         | `few_shot_adaptation`, `task_interference` |

## Creating a New Task

1. Create `agentick/tasks/<category>/your_task.py`.
2. Subclass `TaskSpec` and define `difficulty_configs` for all four levels.
3. Implement `generate(seed)` returning `(Grid, config_dict)`. Use `generation/validation.py` (`verify_solvable`, `find_optimal_path`) to ensure solvability.
4. Implement `compute_dense_reward(old_state, action, new_state, info)`.
5. Implement `check_success(state)` -- check `state["grid"]` and `state["agent"]`.
6. Decorate with `@register_task("YourTask-v0", tags=[...])`.
7. Import your class in the category's `__init__.py`.

Minimal example:

```python
@register_task("MyTask-v0", tags=["navigation"])
class MyTask(TaskSpec):
    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=5,  max_steps=20,  params={}),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=50,  params={}),
        "hard":   DifficultyConfig(name="hard",   grid_size=15, max_steps=100, params={}),
        "expert": DifficultyConfig(name="expert", grid_size=20, max_steps=200, params={}),
    }

    def generate(self, seed):
        cfg = self.difficulty_config
        grid = Grid(cfg.grid_size, cfg.grid_size)
        # ... procedural generation ...
        return grid, {"agent_start": (1, 1), "goal_positions": [(8, 8)], "max_steps": cfg.max_steps}

    def compute_dense_reward(self, old_state, action, new_state, info):
        # ... shaped reward ...
        return 0.0

    def check_success(self, state):
        return state["agent"].position in state["config"].get("goal_positions", [])
```

## Task Descriptions

`descriptions.py` provides `get_task_description(task_name)` and `get_all_task_descriptions()` for extracting human-readable descriptions from the registry. Descriptions are pulled from each task class's docstring and metadata, returned as `TaskDescription` dataclasses with name, category, capability tags, and available difficulties.
