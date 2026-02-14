# Creating Custom Tasks

Learn how to create your own Agentick tasks from scratch.

## Task Structure Overview

Every task in Agentick inherits from `TaskSpec` and must implement:

```python
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.core.actions import ActionType

class MyTask(TaskSpec):
    # 1. Metadata
    name = "MyTask-v0"
    description = "Description of the task"
    capability_tags = ["navigation", "custom"]

    # 2. Difficulty configurations
    difficulty_configs = {
        "easy": DifficultyConfig(...),
        "medium": DifficultyConfig(...),
        "hard": DifficultyConfig(...),
    }

    # 3. Required methods
    def generate(self, seed: int) -> tuple[Grid, dict]:
        """Generate task instance"""
        pass

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Compute shaped reward"""
        pass

    def check_success(self, state: dict) -> bool:
        """Check if task is complete"""
        pass
```

## Step-by-Step Guide

### Step 1: Define Metadata

```python
from agentick.tasks.base import TaskSpec

class FindGoalTask(TaskSpec):
    # Task identifier (must be unique and end with -v0)
    name = "FindGoal-v0"

    # Human-readable description
    description = "Agent must navigate to a goal location"

    # Capability tags for categorization
    capability_tags = ["navigation", "spatial_reasoning"]

    # Additional metadata
    class Config:
        json_schema_extra = {
            "author": "Your Name",
            "license": "MIT",
        }
```

### Step 2: Define Difficulty Configurations

```python
from agentick.tasks.configs import DifficultyConfig

difficulty_configs = {
    "easy": DifficultyConfig(
        name="easy",
        grid_size=7,        # 7x7 grid
        max_steps=100,      # Plenty of steps
        num_obstacles=3,    # Few obstacles
    ),
    "medium": DifficultyConfig(
        name="medium",
        grid_size=10,
        max_steps=75,
        num_obstacles=8,
    ),
    "hard": DifficultyConfig(
        name="hard",
        grid_size=15,
        max_steps=50,
        num_obstacles=20,
    ),
    "expert": DifficultyConfig(
        name="expert",
        grid_size=20,
        max_steps=30,
        num_obstacles=40,
    ),
}
```

### Step 3: Implement Task Generation

```python
import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType

def generate(self, seed: int) -> tuple[Grid, dict]:
    """Generate a task instance."""
    # Get RNG from seed
    rng = np.random.default_rng(seed)

    # Get difficulty config
    config = self.difficulty_configs[self.difficulty]
    size = config.grid_size

    # Create empty grid
    grid = Grid(size, size)

    # Add walls around border
    grid.terrain[0, :] = CellType.WALL
    grid.terrain[-1, :] = CellType.WALL
    grid.terrain[:, 0] = CellType.WALL
    grid.terrain[:, -1] = CellType.WALL

    # Add random obstacles
    for _ in range(config.num_obstacles):
        while True:
            x, y = rng.integers(1, size-1, 2)
            if grid.terrain[y, x] == CellType.EMPTY:
                grid.terrain[y, x] = CellType.WALL
                break

    # Place agent
    while True:
        agent_x, agent_y = rng.integers(1, size-1, 2)
        if grid.terrain[agent_y, agent_x] == CellType.EMPTY:
            break

    # Place goal (far from agent)
    while True:
        goal_x, goal_y = rng.integers(1, size-1, 2)
        if (grid.terrain[goal_y, goal_x] == CellType.EMPTY and
            abs(goal_x - agent_x) + abs(goal_y - agent_y) > size / 2):
            grid.objects[goal_y, goal_x] = ObjectType.GOAL
            break

    # Return grid and task config
    return grid, {
        "agent_start": (agent_x, agent_y),
        "goal_position": (goal_x, goal_y),
        "max_steps": config.max_steps,
    }
```

### Step 4: Implement Reward Function

```python
def compute_dense_reward(
    self,
    old_state: dict,
    action: ActionType,
    new_state: dict,
    info: dict,
) -> float:
    """Compute shaped reward for learning."""
    reward = 0.0

    # Step penalty (encourages efficiency)
    reward -= 0.01

    # Distance-to-goal reward (encourages progress)
    if "agent" in old_state and "agent" in new_state:
        old_pos = old_state["agent"].position
        new_pos = new_state["agent"].position

        # Goal position from info
        goal_pos = info.get("goal_position", (0, 0))

        # Manhattan distance improvement
        old_dist = abs(old_pos[0] - goal_pos[0]) + abs(old_pos[1] - goal_pos[1])
        new_dist = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])

        distance_reward = (old_dist - new_dist) * 0.1
        reward += distance_reward

    # Success bonus (achieved in check_success)
    if info.get("success", False):
        reward += 1.0

    return reward
```

### Step 5: Implement Success Check

```python
def check_success(self, state: dict) -> bool:
    """Check if task is successfully completed."""
    if "grid" not in state or "agent" not in state:
        return False

    agent_pos = state["agent"].position
    grid = state["grid"]

    # Check if agent is on a goal
    x, y = agent_pos
    return grid.objects[y, x] == ObjectType.GOAL
```

### Step 6: Implement Baseline Methods

```python
def get_optimal_return(self, difficulty: str | None = None) -> float:
    """Get theoretical optimal return."""
    if difficulty is None:
        difficulty = self.difficulty

    # Optimal: Take shortest path + success bonus
    config = self.difficulty_configs[difficulty]
    min_steps = config.grid_size  # Approximate

    # Optimal return = success bonus - steps * step penalty
    return 1.0 - min_steps * 0.01

def get_random_baseline(self, difficulty: str | None = None) -> float:
    """Get expected return for random agent."""
    if difficulty is None:
        difficulty = self.difficulty

    config = self.difficulty_configs[difficulty]

    # Rough estimate: random agent rarely succeeds
    # Expected steps: grid_size^2 / 4 (quarter of grid)
    expected_steps = (config.grid_size ** 2) / 4
    random_return = -expected_steps * 0.01  # Just step penalties

    return random_return
```

## Complete Working Example (40+ lines)

```python
"""Complete example of a custom task."""

import numpy as np
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.core.actions import ActionType, Direction
from agentick.tasks.registry import register_task


@register_task("TreasureHunt-v0", tags=["custom", "navigation", "exploration"])
class TreasureHuntTask(TaskSpec):
    """
    Agent must find treasure in a maze.
    Treasure locations change each episode.
    """

    name = "TreasureHunt-v0"
    description = "Navigate maze to find treasure"
    capability_tags = ["navigation", "exploration"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=8,
            max_steps=150,
            num_obstacles=5,
            num_treasures=1,
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=100,
            num_obstacles=15,
            num_treasures=2,
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=16,
            max_steps=75,
            num_obstacles=30,
            num_treasures=3,
        ),
    }

    def generate(self, seed: int) -> tuple[Grid, dict]:
        """Generate treasure hunt task."""
        rng = np.random.default_rng(seed)
        config = self.difficulty_configs[self.difficulty]
        size = config.grid_size

        # Create grid
        grid = Grid(size, size)

        # Add border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Add random obstacles
        for _ in range(config.num_obstacles):
            while True:
                x, y = rng.integers(1, size-1, 2)
                if grid.terrain[y, x] == CellType.EMPTY:
                    grid.terrain[y, x] = CellType.WALL
                    break

        # Place agent
        while True:
            ax, ay = rng.integers(1, size-1, 2)
            if grid.terrain[ay, ax] == CellType.EMPTY:
                break

        # Place treasure(s)
        treasures = []
        for _ in range(config.num_treasures):
            while True:
                tx, ty = rng.integers(1, size-1, 2)
                if (grid.terrain[ty, tx] == CellType.EMPTY and
                    (tx, ty) not in treasures):
                    grid.objects[ty, tx] = ObjectType.GOAL
                    treasures.append((tx, ty))
                    break

        return grid, {
            "agent_start": (ax, ay),
            "treasures": treasures,
            "max_steps": config.max_steps,
            "found_treasures": [],
        }

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Reward for finding treasure and exploring."""
        reward = 0.0

        # Step penalty
        reward -= 0.01

        # Distance to nearest treasure
        if "agent" in new_state and "treasures" in info:
            agent_pos = new_state["agent"].position
            treasures = info["treasures"]

            if treasures:
                distances = [
                    abs(agent_pos[0] - tx) + abs(agent_pos[1] - ty)
                    for tx, ty in treasures
                ]
                nearest = min(distances)

                # Reward for getting closer
                old_pos = old_state["agent"].position
                old_distances = [
                    abs(old_pos[0] - tx) + abs(old_pos[1] - ty)
                    for tx, ty in treasures
                ]
                old_nearest = min(old_distances)

                if nearest < old_nearest:
                    reward += 0.1

        # Treasure finding bonus
        if info.get("success", False):
            reward += 1.0

        return reward

    def check_success(self, state: dict) -> bool:
        """Success when on treasure."""
        if "grid" not in state or "agent" not in state:
            return False

        agent_pos = state["agent"].position
        x, y = agent_pos
        grid = state["grid"]

        return grid.objects[y, x] == ObjectType.GOAL

    def get_optimal_return(self, difficulty: str | None = None) -> float:
        """Optimal: shortest path + treasure bonus."""
        if difficulty is None:
            difficulty = self.difficulty

        config = self.difficulty_configs[difficulty]
        min_steps = config.grid_size

        return 1.0 - min_steps * 0.01

    def get_random_baseline(self, difficulty: str | None = None) -> float:
        """Random agent baseline."""
        if difficulty is None:
            difficulty = self.difficulty

        config = self.difficulty_configs[difficulty]
        expected_steps = (config.grid_size ** 2) / 4

        return -expected_steps * 0.01
```

## Registration and Testing

### Register the Task

```python
# In your_task.py file with @register_task decorator

from agentick.tasks.registry import register_task

@register_task("MyTask-v0", tags=["custom", "navigation"])
class MyTask(TaskSpec):
    ...

# The decorator approach above is the recommended way to register tasks
```

### Test Your Task

```python
import agentick

# Create environment
env = agentick.make("TreasureHunt-v0", difficulty="easy")

# Run episode
obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")

# Take steps
total_reward = 0.0
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:
        break

print(f"Episode finished:")
print(f"  Total steps: {step}")
print(f"  Total reward: {total_reward:.3f}")
print(f"  Success: {info.get('success', False)}")
```

### Unit Tests

```python
import pytest
import numpy as np
import agentick


def test_task_creation():
    """Test task can be created."""
    env = agentick.make("TreasureHunt-v0", difficulty="easy")
    assert env is not None


def test_determinism():
    """Test same seed produces same episode."""
    env = agentick.make("TreasureHunt-v0", difficulty="easy")

    obs1, _ = env.reset(seed=42)
    env.close()

    env = agentick.make("TreasureHunt-v0", difficulty="easy")
    obs2, _ = env.reset(seed=42)

    assert np.array_equal(obs1, obs2)


def test_episode_completion():
    """Test episode can run to completion."""
    env = agentick.make("TreasureHunt-v0", difficulty="easy")
    obs, _ = env.reset(seed=42)

    max_steps = 200
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    assert step < max_steps - 1  # Should finish before max


def test_success_detection():
    """Test success is detected."""
    env = agentick.make("TreasureHunt-v0", difficulty="easy")

    # Run many episodes
    successes = 0
    for seed in range(10):
        obs, info = env.reset(seed=seed)

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                if info.get("success", False):
                    successes += 1
                break

    assert successes > 0  # Should have at least some successes


def test_reward_computation():
    """Test rewards are computed correctly."""
    env = agentick.make("TreasureHunt-v0", difficulty="easy")
    obs, info = env.reset(seed=42)

    # Take one step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Reward should be reasonable
    assert isinstance(reward, (int, float))
    assert -1.0 <= reward <= 1.0  # Reasonable bounds
```

## Best Practices

1. **Determinism**: Use `np.random.default_rng(seed)` for reproducibility
2. **Clear rewards**: Shape should guide learning without being too specific
3. **Documentation**: Document task objective and parameters clearly
4. **Testing**: Write tests for generation, rewards, and success conditions
5. **Difficulty progression**: Easy → Medium → Hard should increase challenge
6. **Balanced objectives**: Avoid conflicting goals that confuse learning
7. **Meaningful metadata**: Use capability_tags for task categorization
8. **Efficient generation**: Grid generation should be O(n²) or better

## Advanced: Custom State Representations

```python
class CustomTask(TaskSpec):
    def get_state_representation(self) -> dict:
        """Define custom state representation as a dictionary."""
        return {
            "agent_position": "tuple[int, int]",
            "agent_orientation": "Direction",
            "grid_terrain": "np.ndarray",
            "grid_objects": "np.ndarray",
            "treasures_collected": "int",
        }
```

## Advanced: Custom Rewards

```python
from agentick.rewards import PotentialBasedReward

class AdvancedTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Use potential-based reward shaping
        def goal_potential(state):
            if "agent" in state:
                agent_pos = state["agent"].position
                goal_pos = self.goal_position
                distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
                return -distance
            return 0.0

        self.reward_shaper = PotentialBasedReward(goal_potential)

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        base_reward = -0.01  # Step cost
        shaped = self.reward_shaper.shape_reward(
            base_reward,
            new_state,
            info.get("terminated", False)
        )
        return shaped
```
