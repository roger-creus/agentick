"""Example of creating a custom task."""

import agentick
from agentick import TaskSpec, register_task
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.configs import DifficultyConfig


@register_task("MyCustomTask-v0", tags=["custom"])
class MyCustomTask(TaskSpec):
    name = "MyCustomTask-v0"
    description = "My custom task"
    capability_tags = ["custom"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=5, max_steps=50),
    }

    def generate(self, seed):
        size = 5
        grid = Grid(size, size)

        # Add walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Add goal
        grid.objects[3, 3] = ObjectType.GOAL

        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [(3, 3)],
            "max_steps": 50,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        return -0.01

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return state["grid"].objects[y, x] == ObjectType.GOAL

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0


# Use the custom task
env = agentick.make("MyCustomTask-v0", seed=42)
obs, info = env.reset()
print("Custom task created successfully!")
print(env.render())
