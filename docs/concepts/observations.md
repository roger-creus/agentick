# Observations

Agentick supports multiple observation modes optimized for different agent types.

## Overview

Six modes via `render_mode`:
- **ASCII**: Colored text grid
- **Language**: Natural language descriptions
- **Language Structured**: Dictionary with semantic info
- **RGB Array**: Isometric sprite-based pixel observations (512x512)
- **State Dict**: Full state access
- **Human**: Isometric display (same as rgb_array, for interactive use)

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

## 4. RGB Array Mode (Isometric)

```python
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
obs, info = env.reset()
print(obs.shape)  # (512, 512, 3), uint8
```

The default visual mode uses an **isometric sprite renderer** powered by Kenney tile assets. Produces visually rich 512x512 images with an isometric diamond perspective.

Space: `gymnasium.spaces.Box` shape `(512, 512, 3)`, dtype `uint8`, range `[0, 255]`

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

See [Architecture](architecture.md) and [Tasks](../tasks.md) for more details.
