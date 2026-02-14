# Custom Renderers

Learn how to add new observation modalities to Agentick environments.

## Renderer Interface

All renderers follow the same protocol:

```python
from typing import Any, Protocol
import gymnasium as gym

class Renderer(Protocol):
    """Protocol for custom renderers."""

    def render(
        self,
        grid: "Grid",
        entities: list["Entity"],
        agent: "Agent",
        info: dict[str, Any],
    ) -> Any:
        """Convert environment state to observation.

        Args:
            grid: The grid state
            entities: List of entities in the environment
            agent: The agent entity
            info: Additional info dict

        Returns:
            Observation in the custom format
        """
        ...

    @property
    def observation_space(self) -> gym.Space:
        """Define the observation space.

        Returns:
            Gymnasium space object describing the observation format
        """
        ...
```

## Complete Working Example: JSON-LD Renderer

A new renderer that outputs observations as structured JSON-LD (JSON for Linked Data):

```python
"""JSON-LD Renderer: Structured semantic observations."""

from typing import Any
import gymnasium as gym
import json
import numpy as np
from agentick.core.grid import Grid
from agentick.core.entity import Agent, Entity
from agentick.core.types import ObjectType, CellType


class JSONLDRenderer:
    """Render environment state as JSON-LD (semantic/structured format).

    JSON-LD provides structured, machine-readable observations with
    semantic meaning, suitable for LLM agents.
    """

    def __init__(
        self,
        include_spatial_relations=True,
        include_history=False,
        max_history_length=5,
    ):
        """Initialize JSON-LD renderer.

        Args:
            include_spatial_relations: Add relative position descriptions
            include_history: Track action/observation history
            max_history_length: How many steps to remember
        """
        self.include_spatial_relations = include_spatial_relations
        self.include_history = include_history
        self.max_history_length = max_history_length
        self.history = []

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str:
        """Render state as JSON-LD string.

        Args:
            grid: Environment grid
            entities: All entities
            agent: Agent entity
            info: Additional info

        Returns:
            JSON-LD string representation
        """
        # Build semantic representation
        observation = {
            "@context": "https://agentick.org/context.jsonld",
            "@type": "EnvironmentState",
            "timestamp": info.get("step", 0),
            "agent": self._render_agent(agent),
            "grid": self._render_grid(grid),
            "entities": self._render_entities(entities, agent),
            "task_info": self._render_task_info(info),
        }

        if self.include_spatial_relations:
            observation["spatial_relations"] = self._compute_spatial_relations(
                grid, agent, entities
            )

        if self.include_history:
            observation["history"] = self.history[-self.max_history_length :]

        # Convert to JSON string
        json_str = json.dumps(observation, indent=2)

        # Add to history if enabled
        if self.include_history:
            self.history.append({
                "action": info.get("last_action", None),
                "observation": observation,
            })

        return json_str

    def _render_agent(self, agent: Agent) -> dict[str, Any]:
        """Render agent as JSON-LD entity."""
        return {
            "@type": "Agent",
            "id": "agent_0",
            "position": {
                "x": agent.position[0],
                "y": agent.position[1],
            },
            "orientation": agent.orientation.name if hasattr(agent, "orientation") else None,
            "inventory": [
                {
                    "@type": "Item",
                    "type": item.entity_type,
                    "id": item.id,
                }
                for item in getattr(agent, "inventory", [])
            ],
            "health": getattr(agent, "health", None),
        }

    def _render_grid(self, grid: Grid) -> dict[str, Any]:
        """Render grid as JSON-LD structure."""
        return {
            "@type": "Grid",
            "width": grid.width,
            "height": grid.height,
            "terrain_summary": self._summarize_terrain(grid),
            "object_summary": self._summarize_objects(grid),
        }

    def _summarize_terrain(self, grid: Grid) -> dict[str, int]:
        """Summarize terrain types and counts."""
        unique, counts = np.unique(grid.terrain, return_counts=True)
        summary = {}

        terrain_names = {
            CellType.EMPTY: "empty",
            CellType.WALL: "wall",
            CellType.WATER: "water",
            CellType.LAVA: "lava",
            CellType.GOAL: "goal",
        }

        for cell_type, count in zip(unique, counts):
            name = terrain_names.get(int(cell_type), f"type_{cell_type}")
            summary[name] = int(count)

        return summary

    def _summarize_objects(self, grid: Grid) -> dict[str, list]:
        """Summarize objects and their locations."""
        summary = {}
        object_names = {
            ObjectType.GOAL: "goals",
            ObjectType.KEY: "keys",
            ObjectType.DOOR: "doors",
            ObjectType.BOX: "boxes",
            ObjectType.HAZARD: "hazards",
        }

        for obj_type in object_names:
            positions = np.argwhere(grid.objects == obj_type)
            if len(positions) > 0:
                summary[object_names[obj_type]] = [
                    {"x": int(pos[1]), "y": int(pos[0])}
                    for pos in positions
                ]

        return summary

    def _render_entities(
        self,
        entities: list[Entity],
        agent: Agent,
    ) -> list[dict[str, Any]]:
        """Render non-agent entities."""
        entity_list = []

        for i, entity in enumerate(entities):
            if entity is not agent:  # Skip agent (already rendered)
                entity_list.append({
                    "@type": "Entity",
                    "id": f"entity_{i}",
                    "type": entity.entity_type,
                    "position": {
                        "x": entity.position[0],
                        "y": entity.position[1],
                    },
                })

        return entity_list

    def _render_task_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Render task-specific information."""
        return {
            "step": info.get("step", 0),
            "episode_return": info.get("episode_return", 0.0),
            "success": info.get("success", False),
            "goals": info.get("goals", []),
        }

    def _compute_spatial_relations(
        self,
        grid: Grid,
        agent: Agent,
        entities: list[Entity],
    ) -> dict[str, Any]:
        """Compute spatial relations (left, right, nearby, etc)."""
        agent_pos = np.array(agent.position)

        relations = {
            "nearby_walls": [],
            "nearby_goals": [],
            "nearby_entities": [],
        }

        # Check nearby cells (distance <= 2)
        for y in range(max(0, agent_pos[1] - 2), min(grid.height, agent_pos[1] + 3)):
            for x in range(max(0, agent_pos[0] - 2), min(grid.width, agent_pos[0] + 3)):
                if (x, y) == tuple(agent_pos):
                    continue  # Skip agent position

                # Check terrain
                if grid.terrain[y, x] == CellType.WALL:
                    direction = self._get_direction(agent_pos, np.array([x, y]))
                    relations["nearby_walls"].append(direction)

                # Check objects
                if grid.objects[y, x] == ObjectType.GOAL:
                    direction = self._get_direction(agent_pos, np.array([x, y]))
                    relations["nearby_goals"].append(direction)

        return relations

    def _get_direction(self, from_pos: np.ndarray, to_pos: np.ndarray) -> str:
        """Get relative direction (north, south, etc)."""
        dy = to_pos[1] - from_pos[1]
        dx = to_pos[0] - from_pos[0]

        if dy < 0:
            if dx < 0:
                return "northwest"
            elif dx > 0:
                return "northeast"
            else:
                return "north"
        elif dy > 0:
            if dx < 0:
                return "southwest"
            elif dx > 0:
                return "southeast"
            else:
                return "south"
        else:
            if dx < 0:
                return "west"
            else:
                return "east"

    @property
    def observation_space(self) -> gym.Space:
        """Observation space for JSON-LD.

        Returns Dict space with JSON string.
        """
        return gym.spaces.Dict({
            "observation": gym.spaces.Text(
                max_length=10000,  # Max JSON-LD string length
            ),
        })

    def reset(self):
        """Reset history."""
        if self.include_history:
            self.history.clear()
```

## Integration with AgentickEnv

### Register Custom Renderer

```python
# Custom renderers can be used by passing them to environments directly.
# The built-in render modes are: ascii, language, language_structured,
# rgb_array, state_dict.
#
# To use a custom renderer, override the render() method in your task class:
import agentick

env = agentick.make(
    "GoToGoal-v0",
    render_mode="ascii",  # Use built-in renderer
)

obs, info = env.reset()
print(obs)  # JSON-LD formatted observation
```

### Using Multiple Renderers

```python
# Get observations in multiple formats
env = agentick.make(
    "GoToGoal-v0",
    render_modes=["ascii", "language", "jsonld"],  # Multiple modes
)

obs, info = env.reset()

# Different observation types
ascii_obs = obs["ascii"]  # ASCII art
lang_obs = obs["language"]  # Natural language
json_obs = obs["jsonld"]  # Structured JSON-LD
```

## Custom Renderer Examples

### Example 1: Graph Representation Renderer

```python
"""Render environment as graph structure."""

import json
import numpy as np
from agentick.core.grid import Grid


class GraphRenderer:
    """Render as node-link graph representation."""

    def render(self, grid: Grid, entities, agent, info) -> str:
        """Render as graph JSON."""
        graph = {
            "nodes": [],
            "links": [],
        }

        # Add agent as node
        graph["nodes"].append({
            "id": "agent",
            "type": "agent",
            "x": agent.position[0],
            "y": agent.position[1],
        })

        # Add entities as nodes
        for i, entity in enumerate(entities):
            graph["nodes"].append({
                "id": f"entity_{i}",
                "type": entity.entity_type,
                "x": entity.position[0],
                "y": entity.position[1],
            })

        # Add edges (spatial relationships)
        agent_pos = np.array(agent.position)
        for node in graph["nodes"]:
            if node["id"] != "agent":
                entity_pos = np.array([node["x"], node["y"]])
                distance = np.linalg.norm(agent_pos - entity_pos)

                if distance < 5:  # Only link nearby entities
                    graph["links"].append({
                        "source": "agent",
                        "target": node["id"],
                        "distance": float(distance),
                    })

        return json.dumps(graph)

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Text(max_length=5000)
```

### Example 2: Symbolic Representation Renderer

```python
"""Render as symbolic logic representation."""

class SymbolicRenderer:
    """Render observations as first-order logic facts."""

    def render(self, grid: Grid, entities, agent, info) -> str:
        """Render as logic facts."""
        facts = []

        # Agent facts
        ax, ay = agent.position
        facts.append(f"at(agent, {ax}, {ay})")

        # Entity facts
        for entity in entities:
            ex, ey = entity.position
            facts.append(f"at({entity.entity_type}, {ex}, {ey})")
            facts.append(f"type({entity.entity_type})")

        # Terrain facts
        for y in range(grid.height):
            for x in range(grid.width):
                terrain = grid.terrain[y, x]
                if terrain != CellType.EMPTY:
                    facts.append(f"terrain({terrain.name.lower()}, {x}, {y})")

        # Adjacency facts
        adjacent_objects = self._find_adjacent_objects(grid, agent.position)
        for obj_type in adjacent_objects:
            facts.append(f"adjacent(agent, {obj_type})")

        # Combine facts
        representation = "\n".join(facts)
        return representation

    def _find_adjacent_objects(self, grid: Grid, pos):
        """Find objects adjacent to position."""
        x, y = pos
        adjacent = set()

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.height and 0 <= nx < grid.width:
                    if grid.objects[ny, nx] != ObjectType.EMPTY:
                        adjacent.add(grid.objects[ny, nx].name)

        return adjacent

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Text(max_length=5000)
```

### Example 3: Feature Vector Renderer

```python
"""Render as numerical feature vector."""

import numpy as np
from gymnasium import spaces


class FeatureVectorRenderer:
    """Render as fixed-size feature vector."""

    def __init__(self, grid_size=15):
        self.grid_size = grid_size

    def render(self, grid: Grid, entities, agent, info) -> np.ndarray:
        """Render as feature vector."""
        features = []

        # Agent position (normalized 0-1)
        features.extend([
            agent.position[0] / grid.width,
            agent.position[1] / grid.height,
        ])

        # Entity features
        for entity in entities[:10]:  # Max 10 entities
            features.extend([
                entity.position[0] / grid.width,
                entity.position[1] / grid.height,
                float(entity.entity_type.value),
            ])

        # Pad to fixed size
        max_features = 2 + 10 * 3  # Agent + 10 entities
        while len(features) < max_features:
            features.append(0.0)

        return np.array(features[:max_features], dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Space:
        max_features = 2 + 10 * 3
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(max_features,),
            dtype=np.float32,
        )
```

## Testing Custom Renderers

### Unit Tests for Renderers

```python
import pytest
import numpy as np
from agentick.core.grid import Grid


def test_renderer_output_format():
    """Test renderer produces expected format."""
    renderer = JSONLDRenderer()

    # Create mock state
    grid = Grid(10, 10)
    entities = []
    agent = MockAgent(position=(5, 5))
    info = {"step": 0}

    # Render
    obs = renderer.render(grid, entities, agent, info)

    # Verify format
    assert isinstance(obs, str)
    parsed = json.loads(obs)  # Should be valid JSON
    assert "@context" in parsed
    assert "@type" in parsed


def test_renderer_observation_space():
    """Test observation space is valid."""
    renderer = JSONLDRenderer()
    space = renderer.observation_space

    # Space should be gymnasium Space
    import gymnasium as gym
    assert isinstance(space, gym.Space)

    # Should be able to sample (for some spaces)
    assert hasattr(space, "sample")


def test_renderer_determinism():
    """Test renderer is deterministic."""
    renderer = JSONLDRenderer()

    grid = Grid(10, 10)
    entities = []
    agent = MockAgent(position=(5, 5))
    info = {"step": 0}

    obs1 = renderer.render(grid, entities, agent, info)
    obs2 = renderer.render(grid, entities, agent, info)

    assert obs1 == obs2, "Renderer not deterministic"


def test_renderer_with_environment():
    """Test renderer works with actual environment."""
    import agentick

    # Custom renderers are used by overriding render() in task classes
    env = agentick.make("GoToGoal-v0", render_mode="ascii")

    obs, info = env.reset(seed=42)

    # Obs should be string (JSON-LD)
    assert isinstance(obs, str)

    # Take a step
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, str)

    env.close()
```

## Best Practices

1. **Document format clearly**: Explain what your renderer outputs
2. **Use gymnasium spaces**: Inherit from `gym.Space` for compatibility
3. **Handle edge cases**: Empty grids, max entity counts, etc.
4. **Test thoroughly**: Unit tests for different states
5. **Optimize performance**: Rendering is called every step
6. **Use standard formats**: JSON, numpy arrays, text when possible
7. **Support configuration**: Allow renderer customization
8. **Backward compatible**: Don't break existing code

## Troubleshooting

### Renderer Not Found
```python
# Custom renderers are used by overriding the render() method
# in your task class. Built-in modes: ascii, language,
# language_structured, rgb_array, state_dict
env = agentick.make("Task-v0", render_mode="ascii")
```

### Large Output Size
```python
# Compress observations
class CompressedRenderer:
    def render(self, ...):
        obs = self._create_full_obs(...)
        # Compress
        import gzip
        compressed = gzip.compress(obs.encode())
        return compressed
```

### Performance Issues
```python
# Cache results if unchanged
class CachedRenderer:
    def __init__(self):
        self.last_obs = None
        self.last_state_hash = None

    def render(self, grid, entities, agent, info):
        state_hash = hash((id(grid), tuple(e.id for e in entities)))
        if state_hash == self.last_state_hash:
            return self.last_obs

        obs = self._render_impl(grid, entities, agent, info)
        self.last_obs = obs
        self.last_state_hash = state_hash
        return obs
```
