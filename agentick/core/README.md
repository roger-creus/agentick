# agentick/core

Core engine for the Agentick gridworld environment: grid representation, agent mechanics, actions, and multi-modal rendering.

## Module Overview

| File | Role |
|------|------|
| `types.py` | Enum definitions (`CellType`, `ObjectType`, `ActionType`, `Direction`, `AgentType`), color/ASCII constants |
| `grid.py` | `Grid` class -- multi-layer numpy grid representation |
| `entity.py` | `Entity` (base) and `Agent` (with inventory, orientation, status) dataclasses |
| `actions.py` | `ActionSpace` (discrete action space with masking), `compute_action_mask()`, `get_move_delta()` |
| `env.py` | `AgentickEnv` -- Gymnasium `gym.Env` subclass that owns the grid, agent, renderer, and step loop |
| `renderer.py` | `Renderer` protocol, `ASCIIRenderer`, `EnhancedLanguageRenderer`, `StateDictRenderer`, `create_renderer()` factory |
| `language.py` | `AdvancedLanguageRenderer` -- rich natural language descriptions with spatial reasoning, configurable via `LanguageConfig` |
| `simple_grid_renderer.py` | `SimpleGridRenderer` -- top-down 2D pixel renderer using Pillow (used for `rgb_array` and `rgb_array_2d` modes) |
| `feature_extractor.py` | `extract_state_features()` -- converts `state_dict` observations into flat numpy vectors for RL training |

## Enum Types

All enums are `IntEnum` subclasses so they work directly as numpy array values.

- **`CellType`** -- terrain layer values: `EMPTY`, `WALL`, `HAZARD`, `WATER`, `ICE`, `HOLE`
- **`ObjectType`** -- object layer values: `NONE`, `GOAL`, `KEY`, `DOOR`, `SWITCH`, `BOX`, `TARGET`, `TOOL`, `RESOURCE`, `BREADCRUMB`, `NPC`, `ENEMY`, `SHEEP`, `BLOCKER`, `GEM`, `LEVER`, `POTION`, `SCROLL`, `COIN`, `ORB`
- **`ActionType`** -- discrete actions: movement (4 cardinal + forward), `NOOP`, `PICKUP`, `DROP`, `USE`, `INTERACT`, `ROTATE_LEFT`, `ROTATE_RIGHT`
- **`Direction`** -- cardinal directions: `NORTH`, `EAST`, `SOUTH`, `WEST` (with `opposite()`, `rotate_left()`, `rotate_right()`, `to_delta()`)

## Grid Layer Model

`Grid` stores the world state in four numpy arrays, all indexed as `[y, x]`:

| Layer | dtype | Contents |
|-------|-------|----------|
| `terrain` | `int8` | Base terrain (`CellType` values) |
| `objects` | `int8` | Collectible/interactive objects (`ObjectType` values) |
| `agents` | `int8` | Agent/NPC positions (`AgentType` values) |
| `metadata` | `int16` | Per-cell auxiliary data (colors, IDs, states) |

**Coordinate convention:** positions throughout the codebase are `(x, y)` tuples, but numpy array indexing is `[y, x]`. For example, to read the terrain at position `(3, 5)`, access `grid.terrain[5, 3]`.

Key methods: `Grid.in_bounds(pos)`, `Grid.copy()`, `Grid.set_cell()`, `Grid.get_cell()`.

## Data Flow

```
agentick.make("TaskName-v0")
    |
    v
AgentickEnv.__init__(grid, render_mode, reward_mode, max_steps)
    |-- creates Grid (or receives one from TaskSpec.generate())
    |-- creates Agent at starting position
    |-- creates ActionSpace (9 basic or 12 extended actions)
    |-- calls create_renderer(render_mode) to build the renderer
    |
    v
env.step(action)
    |-- resolves ActionType from action index
    |-- computes move delta, checks walkability
    |-- updates agent position and grid layers
    |-- calls task hooks (on_env_step, on_agent_moved, etc.)
    |-- computes reward via task.compute_sparse_reward() or compute_dense_reward()
    |-- calls env._get_observation() -> env.render() -> renderer.render(grid, entities, agent, info)
    |
    v
observation (str | np.ndarray | dict)
```

## Rendering Pipeline

All renderers conform to the `Renderer` protocol:

```python
def render(self, grid: Grid, entities: list[Entity], agent: Agent, info: dict) -> Any
```

`env.render()` dispatches to the configured renderer based on `render_mode`:

| Mode | Renderer | Output |
|------|----------|--------|
| `ascii` | `ASCIIRenderer` | ANSI-colored string |
| `language` | `EnhancedLanguageRenderer` | Natural language string |
| `language_structured` | `EnhancedLanguageRenderer` | JSON-structured string |
| `rgb_array` | `SimpleGridRenderer` | `np.ndarray` (H, W, 3) |
| `rgb_array_2d` | `SimpleGridRenderer` (isometric) | `np.ndarray` (H, W, 3) |
| `state_dict` | `StateDictRenderer` | `dict` with numpy arrays |

## Action Space and Masking

`ActionSpace` wraps a `gym.spaces.Discrete` and provides:

- Bidirectional mapping between integer indices and `ActionType` enums
- Name-based parsing for language agents (`parse_action_name("move_up")`)
- `compute_action_mask()` returns a boolean array of valid actions given the current grid state and agent capabilities

The basic action set has 9 actions; the extended set adds `ROTATE_LEFT`, `ROTATE_RIGHT`, and `MOVE_FORWARD` for partial-observability tasks (12 total).
