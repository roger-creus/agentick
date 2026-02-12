# Observations

Agentick supports multiple observation modes for different agent types and use cases. Each mode represents the same gridworld state in a different format optimized for specific agent architectures.

## Overview

The six observation modes allow a single environment to be used with:
- **LLMs**: ASCII or Language modes
- **Vision models**: RGB array mode
- **Bots**: State dict mode
- **Humans**: Human mode (Pygame UI)

All modes can be accessed from the same environment by selecting different `render_mode` values during environment creation.

## 1. ASCII Mode (`render_mode="ascii"`)

Text-based colored grid representation, ideal for LLMs.

### Basic Symbols

```
#########
#^ .....#
#..#....#
#..#..G.#
#.......#
#########

--- Legend ---
^ v < >: Agent (facing direction)
#: Wall
G: Goal
K: Key
D: Door
B: Box
X: Hazard
~: Water
```

### Symbol Reference

| Symbol | Meaning | ANSI Color |
|--------|---------|------------|
| `^` `v` `<` `>` | Agent facing North/South/West/East | Blue |
| `#` | Wall | Dark Gray |
| `.` | Empty space | White |
| `G` | Goal | Green |
| `K` | Key | Yellow |
| `D` | Door | Orange |
| `B` | Box | Magenta |
| `X` | Hazard | Red |
| `~` | Water | Cyan |
| `i` | Ice | Bright White |
| `O` | Hole | Gray |
| `S` | Switch | Yellow |
| `t` | Tool | Yellow |
| `r` | Resource | Yellow |
| `*` | Breadcrumb | Yellow |

### Example: GoToGoal-v0 (Easy)

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
obs, info = env.reset(seed=42)
print(obs)
```

Output:
```
###########
#^  .  .  #
# . . . . #
# . . . . #
# . . . . #
###########

--- Legend ---
^ v < >: Agent (facing direction)
#: Wall
G: Goal
K: Key
D: Door
B: Box
X: Hazard
~: Water
```

### When to Use

- **LLM prompting**: Feed ASCII grids to language models
- **Text-based agents**: Agents that parse grid strings
- **Debugging**: Visualize environments in console
- **Low bandwidth**: Minimal data transfer

### Observation Space

```python
env.observation_space  # gymnasium.spaces.Text
# min_length=1, max_length=100000
# charset: printable ASCII + ANSI codes
```

### Key Methods

```python
env = agentick.make("GoToGoal-v0", render_mode="ascii")
obs, info = env.reset()

# Current observation
print(obs)  # str with ANSI colors

# Get text observation from any render mode
ascii_obs = env.get_text_observation()  # Forces ASCII regardless of render_mode

# Info dict always available
print(info["valid_actions"])  # ["move_up", "move_left", ...]
print(info["step_count"])     # Current step in episode
```

## 2. Language Mode (`render_mode="language"`)

Natural language description of the gridworld state. Rich semantic information suitable for LLM instruction following.

### Example: GoToGoal-v0 (Easy)

```python
env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
obs, info = env.reset(seed=42)
print(obs)
```

Output:
```
You are in a 5×5 room at position (1, 1). You are facing north. To your
north is a wall. To your east is empty space. To your south is empty space.
To your west is a wall. The goal is visible to your southeast at distance 5.
You are carrying nothing. Your energy is 100%. Actions: move_up, move_down,
move_left, move_right, noop.
```

### Perspective Options

Language renderer supports three perspectives:

```python
# First-person (default)
env = agentick.make(
    "GoToGoal-v0",
    render_mode="language",
    verbosity="standard",
    perspective="first_person"
)
# "You are at position..."

# Third-person
env = agentick.make(
    "GoToGoal-v0",
    render_mode="language",
    perspective="third_person"
)
# "The agent is at position..."

# Omniscient
env = agentick.make(
    "GoToGoal-v0",
    render_mode="language",
    perspective="omniscient"
)
# Complete world state
```

### Verbosity Levels

```python
# Minimal: Just essential info
render_mode="language", verbosity="minimal"
# Output: "At (1,1) facing north. Goal southeast at (4,4)."

# Standard: Balanced information (default)
render_mode="language", verbosity="standard"
# Output: "You are at (1,1) facing north. To your south: empty space..."

# Verbose: All available information
render_mode="language", verbosity="verbose"
# Output: Complete description with memory, threat assessment, etc.
```

### Features

- **Spatial reasoning**: Relative directions (north, southeast, etc.)
- **Memory tracking**: References to previously visited locations
- **Threat awareness**: Mentions hazards and enemies
- **Goal guidance**: Describes distance and direction to goals
- **Inventory description**: What the agent is carrying
- **Status**: Energy and health levels

### Example: KeyDoorPuzzle-v0 (Medium)

```python
env = agentick.make("KeyDoorPuzzle-v0", difficulty="medium", render_mode="language")
obs, info = env.reset(seed=42)
print(obs)
```

Output:
```
You are in a 10×10 room at position (1, 1). You are facing north. You can see
a key to your northeast at distance 4 and a locked door to the south at
distance 9. The goal is beyond the door to the south. You are carrying nothing.
Your energy is 100%. Actions: move_up, move_down, move_left, move_right,
pickup, drop, noop.
```

### When to Use

- **LLM instruction following**: Language models that follow natural language instructions
- **Semantic understanding**: Agents that benefit from semantic descriptions
- **Human-readable evaluation**: Understand what agents see
- **Interpretability**: Audit agent decisions

### Observation Space

```python
env.observation_space  # gymnasium.spaces.Text
# charset: printable ASCII
```

## 3. Language Structured Mode (`render_mode="language_structured"`)

Dictionary representation of the state with structured semantic information.

### Example Output

```python
env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language_structured")
obs, info = env.reset(seed=42)
print(obs)
```

Output:
```python
{
    "description": "You are in a 5x5 room...",  # Natural language summary
    "position": {"x": 1, "y": 1},
    "orientation": "north",
    "surroundings": {
        "north": "wall",
        "south": "empty",
        "east": "empty",
        "west": "wall"
    },
    "inventory": [],
    "visible_goals": [
        {
            "type": "goal",
            "position": {"x": 4, "y": 4},
            "direction": "southeast",
            "distance": 5
        }
    ],
    "visible_threats": [
        # Hazards, enemies, obstacles
    ],
    "energy": 1.0,  # 0.0 to 1.0
    "health": 1.0,  # 0.0 to 1.0
    "valid_actions": ["move_up", "move_down", "move_left", "move_right", "noop"],
    "step_count": 0,
    "max_steps": 20
}
```

### When to Use

- **JSON-parsing agents**: Models that work with structured input
- **Hybrid approaches**: Combine structured data with language
- **Easier parsing**: More reliable than free-form text parsing
- **API compatibility**: Integrate with services expecting JSON

### Observation Space

```python
env.observation_space  # gymnasium.spaces.Dict
# Keys: description, position, orientation, surroundings, inventory,
#       visible_goals, visible_threats, energy, health, valid_actions, etc.
```

## 4. RGB Array Mode (`render_mode="rgb_array"`)

Pixel-based visual observation as numpy array. Renders gridworld as colored sprites.

### Example: GoToGoal-v0

```python
import agentick
import matplotlib.pyplot as plt

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
obs, info = env.reset(seed=42)

print(obs.shape)  # (160, 160, 3) for 5x5 grid with 32px tiles
print(obs.dtype)  # uint8
print(obs.min(), obs.max())  # 0, 255

# Visualize
plt.imshow(obs)
plt.show()
```

### Visual Elements

- **Agent**: Colored triangle pointing in facing direction
  - North: Points up
  - South: Points down
  - East: Points right
  - West: Points left
- **Goal**: Gold star shape
- **Keys**: Golden key shapes
- **Doors**: Brown door rectangles
- **Walls**: Gray wall tiles
- **Empty**: Light gray background
- **Hazards**: Red X pattern
- **Water**: Cyan waves
- **Ice**: Light blue frosted pattern

### Grid Overlay

Optional grid lines separating tiles:

```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="rgb_array",
    show_grid=True  # Show grid lines
)
```

### HUD Overlay

Optional heads-up display with information:

```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="rgb_array",
    show_hud=True  # Show HUD (step count, reward, inventory, etc.)
)
```

### Tile Size Options

```python
# Different tile sizes for different resolutions
env = agentick.make(
    "GoToGoal-v0",
    render_mode="rgb_array",
    tile_size=8   # 8x8 pixels per tile (small)
)

# grid 5x5 -> image 40x40
# grid 10x10 -> image 80x80
# grid 15x15 -> image 120x120
# grid 20x20 -> image 160x160

env = agentick.make(
    "GoToGoal-v0",
    render_mode="rgb_array",
    tile_size=64  # 64x64 pixels per tile (large, detailed)
)

# grid 5x5 -> image 320x320
# grid 10x10 -> image 640x640
```

### When to Use

- **Vision models**: CNN, Vision Transformers, CLIP
- **Deep reinforcement learning**: DQN, PPO, etc.
- **Multi-modal models**: Vision-language models
- **Visual debugging**: See what agents see
- **Video recording**: Create visualizations of episodes

### Observation Space

```python
env.observation_space  # gymnasium.spaces.Box
# shape: (H, W, 3) where H = grid_height * tile_size, W = grid_width * tile_size
# dtype: uint8
# low: 0, high: 255
```

### Performance

```python
# Pixel rendering is slower than text
# Benchmark (10,000 steps, 10x10 grid):
# ASCII:      ~100 steps/sec
# Language:   ~50 steps/sec
# RGB (8px):  ~30 steps/sec
# RGB (32px): ~20 steps/sec
# RGB (64px): ~10 steps/sec

# Use fast_mode for state_dict to avoid expensive conversions
env = agentick.make("GoToGoal-v0", render_mode="rgb_array", fast_mode=False)
```

## 5. State Dict Mode (`render_mode="state_dict"`)

Complete structured state representation. Full access to game state for programmatic reasoning.

### Example Output

```python
env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="state_dict")
obs, info = env.reset(seed=42)
print(obs)
```

Output:
```python
{
    "grid": {
        "height": 5,
        "width": 5,
        "terrain": [[1, 1, 1, 1, 1],   # 0=empty, 1=wall, 2=hazard, etc.
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]],
        "objects": [[0, 0, 0, 0, 0],   # 0=none, 1=goal, 2=key, 3=door, etc.
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0],
                    [0, 0, 0, 0, 0]],
        "agents": [[0, 0, 0, 0, 0],    # Agent count per cell
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
    },
    "agent": {
        "position": (1, 1),
        "orientation": "north",
        "inventory": [],
        "energy": 1.0,
        "health": 1.0
    },
    "entities": [],  # Other agents/NPCs
    "info": {
        "step_count": 0,
        "max_steps": 20,
        "episode_reward": 0.0,
        "valid_actions": ["move_down", "move_right", "noop"],
        "agent_position": (1, 1)
    }
}
```

### Fast Mode

For high-frequency sampling (e.g., planning algorithms), use fast mode:

```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="state_dict",
    fast_mode=True  # Skip expensive conversions
)

# Returns numpy arrays instead of lists:
obs["grid"]["terrain"]  # numpy array (not list)
obs["grid"]["objects"]  # numpy array (not list)
obs["agent"]["orientation"]  # Direction enum (not string)
obs["agent"]["inventory"]  # list[Entity] (not serialized)

# Much faster for iterative access:
# Fast mode: ~1000 steps/sec
# Standard: ~100 steps/sec
```

### Accessing State Programmatically

```python
env = agentick.make("GoToGoal-v0", render_mode="state_dict")
obs, info = env.reset()

# Agent position
x, y = obs["agent"]["position"]

# Agent facing direction
facing = obs["agent"]["orientation"]  # Direction enum

# Grid dimensions
height = obs["grid"]["height"]
width = obs["grid"]["width"]

# Terrain at position (2, 3)
terrain_type = obs["grid"]["terrain"][3][2]  # Note: [y][x] indexing

# What object is at (2, 3)?
obj_type = obs["grid"]["objects"][3][2]

# All valid actions this step
valid_actions = info["valid_actions"]

# Goal positions from earlier info
goal_pos = obs["info"].get("goal_positions", [])
```

### When to Use

- **Bots and search agents**: Pathfinding, planning algorithms
- **Perfect information agents**: Agents that need complete state
- **Curriculum learning**: Custom reward computation
- **Visualization**: Build custom visualization tools
- **Analysis**: Post-hoc analysis of episodes

### Observation Space

```python
env.observation_space  # gymnasium.spaces.Dict
# Contains all state data as nested dicts and arrays
```

## 6. Human Mode (`render_mode="human"`)

Interactive Pygame window for human play and data collection.

### Usage

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="human")
obs, info = env.reset()

for step in range(100):
    # Window handles input
    obs, reward, terminated, truncated, info = env.step(0)  # Dummy action
    if terminated or truncated:
        break

env.close()
```

### Controls

- **Arrow keys**: Move up/down/left/right
- **Space**: Interact / pickup / drop
- **R**: Rotate left
- **L**: Rotate right
- **Q**: Quit

### Display Elements

- Large sprite-based rendering
- Grid overlay
- HUD with status (steps, reward, inventory)
- Real-time agent position
- All game objects visible

### When to Use

- **Human baselines**: Collect human gameplay data
- **Debugging**: Understand task difficulty
- **Data collection**: Record human solutions
- **Demonstration**: Show task to others

## Comparing Observation Modes

### Example: Same State, All Modes

All these representations describe the same gridworld state:

```python
import agentick

# ASCII
env_ascii = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
obs_ascii, _ = env_ascii.reset(seed=42)
print("ASCII:\n", obs_ascii)

# Language
env_lang = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
obs_lang, _ = env_lang.reset(seed=42)
print("Language:", obs_lang)

# Language Structured
env_lang_struct = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language_structured")
obs_struct, _ = env_lang_struct.reset(seed=42)
print("Structured:", obs_struct)

# RGB
env_rgb = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
obs_rgb, _ = env_rgb.reset(seed=42)
print("RGB shape:", obs_rgb.shape)

# State Dict
env_state = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="state_dict")
obs_state, _ = env_state.reset(seed=42)
print("State Dict keys:", obs_state.keys())
```

All environments reset with same seed produce equivalent internal state!

### Selection Guide

| Agent Type | Best Modes | Reason |
|---|---|---|
| **LLM** | ASCII, Language | Processes text naturally |
| **Vision Transformer** | RGB Array | Processes images |
| **CNN-based RL** | RGB Array | Designed for pixels |
| **Semantic understanding** | Language, Language Structured | Rich semantic info |
| **Programmatic bot** | State Dict | Full programmatic access |
| **Search/Planning** | State Dict (fast mode) | Speed + full information |
| **Human baseline** | Human | Interactive play |
| **Hybrid models** | Language + RGB (convert) | Multiple inputs |

## Wrappers: Transforming Observations

Modify observations using wrapper middleware:

```python
from gymnasium import Wrapper
import numpy as np

class NormalizeObservationWrapper(Wrapper):
    """Normalize RGB observations to [0, 1]."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32) / 255.0
        return obs, reward, terminated, truncated, info

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
wrapped = NormalizeObservationWrapper(env)
obs, info = wrapped.reset()
print(obs.min(), obs.max())  # 0.0, 1.0
```

## Performance Notes

### Rendering Speed (estimated, 10x10 grid)

| Mode | Speed | Notes |
|---|---|---|
| **State Dict** | Fastest | Direct conversion, minimal computation |
| **ASCII** | Fast | Simple character assignment |
| **Language** | Medium | Spatial reasoning computation |
| **RGB** | Slow | Pygame sprite rendering |
| **Human** | Variable | Depends on display refresh |

### Memory Usage

| Mode | Size (10x10 grid) |
|---|---|
| State Dict | ~5 KB |
| ASCII | ~200 bytes |
| Language | ~500 bytes |
| RGB (32px) | ~30 KB |
| RGB (64px) | ~120 KB |

### Best Practices

1. **Use fast_mode=True** with state_dict for planning algorithms
2. **Use small tile_size** (8-16px) for CNN models to reduce memory
3. **Batch rendering** for multiple environments with VectorEnv
4. **Cache observations** if access multiple times per step

## Advanced: Creating Custom Observation Modes

Implement the Renderer protocol:

```python
from agentick.core.renderer import Renderer

class MyCustomRenderer:
    def render(self, grid, entities, agent, info):
        # Your custom rendering logic
        return my_observation_format

# Use in environment (requires modifying renderer factory)
```

See [Custom Renderers](../extending/custom_renderers.md) for complete guide.

## Summary

Agentick's multi-modal observation system enables:
- **Flexibility**: Same environment works with different agent types
- **Efficiency**: Choose observation mode optimized for your agent
- **Transparency**: Different modes reveal different aspects of state
- **Interoperability**: Support LLM, vision, bot, and human agents

Choose the observation mode that best matches your agent's perception capabilities!
