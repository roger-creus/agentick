# Observations

Agentick supports multiple observation modes optimized for different agent types.

## Overview

Seven modes via `render_mode`:
- **ASCII**: Colored text grid
- **Language**: Natural language descriptions
- **Language Structured**: Dictionary with semantic info
- **RGB Array**: 3D isometric pixel observations (default) or 2D sprites
- **RGB Array 2D**: Flat 2D sprite-based pixel observations
- **State Dict**: Full state access
- **Human**: Pygame window

## 1. ASCII Mode

```python
env = agentick.make("GoToGoal-v0", render_mode="ascii")
obs, info = env.reset(seed=42)
```

Output: `#^..G#` (colored grid with legend)

Space: `gymnasium.spaces.Text`

## 2. Language Mode

```python
env = agentick.make("GoToGoal-v0", render_mode="language")
obs, info = env.reset()
```

Output: "You are at (1,1) facing north. Goal southeast at distance 5."

**Options**:
- Perspective: `first_person`, `third_person`, `omniscient`
- Verbosity: `minimal`, `standard`, `verbose`

```python
env = agentick.make("GoToGoal-v0", render_mode="language", verbosity="minimal")
```

Space: `gymnasium.spaces.Text`

## 3. Language Structured Mode

```python
env = agentick.make("GoToGoal-v0", render_mode="language_structured")
obs, info = env.reset()
```

Output:
```python
{
    "description": "You are in a 5x5 room...",
    "position": {"x": 1, "y": 1},
    "orientation": "north",
    "surroundings": {"north": "wall", "south": "empty"},
    "visible_goals": [{"type": "goal", "position": {"x": 4, "y": 4}}],
    "valid_actions": ["move_up", "move_right"]
}
```

Space: `gymnasium.spaces.Dict`

## 4. RGB Array Mode (3D Isometric)

```python
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
obs, info = env.reset()
print(obs.shape)  # (512, 512, 3), uint8
```

Agentick includes a **3D isometric renderer** powered by GLB models generated via
Meshy AI. Enable it with `render_3d=True` for visually rich observations with an
orthographic isometric camera, warm key lighting, and cool fill lighting.

**3D rendering options**:
```python
# 3D isometric (opt-in)
env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=True)

# Force 2D sprites (faster, for RL training)
env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=False)

# Custom GLB models directory
env = agentick.make("GoToGoal-v0", render_mode="rgb_array", asset_dir="my_models/")
```

If `trimesh` and `pyrender` are not installed, the renderer automatically falls back
to 2D sprites. Install 3D dependencies with: `uv sync --extra render3d`

Space: `gymnasium.spaces.Box` shape `(512, 512, 3)`, dtype `uint8`, range `[0, 255]`

## 4b. RGB Array 2D Mode (Flat Sprites)

For maximum speed (e.g., during RL training), use the flat 2D sprite renderer:

```python
env = agentick.make("GoToGoal-v0", render_mode="rgb_array_2d")
obs, info = env.reset()
print(obs.shape)  # (H*32, W*32, 3), uint8
```

**Visual elements**: Agent (triangle), Goal (star), Keys (key icon), Doors (rectangle), Walls (gray), Hazards (red X)

**Options**:
```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="rgb_array_2d",
    tile_size=32,  # 8, 16, 32, 64
    show_grid=True,
    show_hud=True
)
```

**Performance comparison**:

| Mode | Speed | Use Case |
|---|---|---|
| `rgb_array` (3D) | ~10-50 FPS | LLM evaluation, human play, video recording |
| `rgb_array_2d` (2D) | ~1000+ FPS | RL training, batch evaluation |

Space: `gymnasium.spaces.Box` shape `(H, W, 3)`, dtype `uint8`, range `[0, 255]`

## 5. State Dict Mode

```python
env = agentick.make("GoToGoal-v0", render_mode="state_dict")
obs, info = env.reset()
```

Output:
```python
{
    "grid": {
        "height": 5, "width": 5,
        "terrain": [[1,1,1],[1,0,0],...],  # 0=empty, 1=wall
        "objects": [[0,0,0],[0,0,1],...],  # 0=none, 1=goal
        "agents": [[0,1,0],[0,0,0],...]
    },
    "agent": {
        "position": (1, 1),
        "orientation": "north",
        "inventory": [],
        "energy": 1.0
    },
    "info": {
        "step_count": 0,
        "max_steps": 20,
        "valid_actions": ["move_down", "move_right"]
    }
}
```

**Fast mode** for planning:
```python
env = agentick.make("GoToGoal-v0", render_mode="state_dict", fast_mode=True)
# Returns numpy arrays, ~10x faster
```

Space: `gymnasium.spaces.Dict`

## 6. Human Mode

```python
env = agentick.make("GoToGoal-v0", render_mode="human")
obs, info = env.reset()
```

**Controls**: Arrow keys (move), Space (interact), R/L (rotate), Q (quit)

**Display**: Large sprites, grid overlay, HUD

## Selection Guide

| Agent Type | Best Mode |
|---|---|
| LLM | ASCII, Language |
| VLM (e.g. GPT-4o) | RGB Array (3D for richer visuals) |
| Vision Transformer | RGB Array |
| CNN-based RL | RGB Array 2D (fast) or RGB Array |
| Programmatic bot | State Dict |
| Search/Planning | State Dict (fast mode) |
| Human baseline | Human |

## Multi-Modal Access

```python
# Access any mode programmatically
env = agentick.make("GoToGoal-v0", render_mode="ascii")
ascii_obs = env.get_text_observation()
pixel_obs = env.get_pixel_observation()
state_obs = env.get_state_dict()
```

Same seed produces equivalent internal state across modes:
```python
env1 = agentick.make("GoToGoal-v0", render_mode="ascii")
env2 = agentick.make("GoToGoal-v0", render_mode="rgb_array")
obs1, _ = env1.reset(seed=42)
obs2, _ = env2.reset(seed=42)
# Same internal state, different representations
```

## Observation Wrappers

```python
from gymnasium import Wrapper

class NormalizeObservationWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.astype(np.float32) / 255.0  # Normalize to [0,1]
        return obs, reward, terminated, truncated, info

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
wrapped = NormalizeObservationWrapper(env)
```

See [Architecture](architecture.md) and [Tasks](tasks.md) for more details.
