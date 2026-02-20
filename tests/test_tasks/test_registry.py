"""Tests for task registry."""

import pytest

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import get_task_class, list_tasks, make, register_task


# Create a simple test task
@register_task("TestTask-v0", tags=["test", "navigation"])
class TestTask(TaskSpec):
    """Simple test task for registry testing."""

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=5,
            max_steps=50,
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=100,
        ),
    }

    def generate(self, seed):
        """Generate simple grid with goal."""
        import numpy as np

        rng = np.random.default_rng(seed)

        size = self.difficulty_config.grid_size
        grid = Grid(size, size)

        # Add border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Random agent start
        agent_x = rng.integers(1, size - 1)
        agent_y = rng.integers(1, size - 1)

        # Random goal
        goal_x = rng.integers(1, size - 1)
        goal_y = rng.integers(1, size - 1)
        grid.objects[goal_y, goal_x] = ObjectType.GOAL

        config = {
            "agent_start": (agent_x, agent_y),
            "goal_positions": [(goal_x, goal_y)],
            "max_steps": self.get_max_steps(),
        }

        return grid, config

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Simple distance-based reward."""
        return -0.01  # Small step penalty

    def check_success(self, state):
        """Check if agent reached goal."""
        if "grid" not in state or "agent" not in state:
            return False

        grid = state["grid"]
        agent = state["agent"]
        x, y = agent.position

        return grid.objects[y, x] == ObjectType.GOAL


def test_task_registration():
    """Test that task was registered."""
    assert "TestTask-v0" in list_tasks()


def test_list_tasks():
    """Test listing all tasks."""
    tasks = list_tasks()
    assert isinstance(tasks, list)
    assert "TestTask-v0" in tasks


def test_list_tasks_by_capability():
    """Test filtering tasks by capability."""
    tasks = list_tasks(capability="navigation")
    assert "TestTask-v0" in tasks

    tasks = list_tasks(capability="nonexistent_capability")
    assert "TestTask-v0" not in tasks


def test_get_task_class():
    """Test getting task class by name."""
    task_class = get_task_class("TestTask-v0")
    assert task_class == TestTask
    assert task_class.name == "TestTask-v0"
    assert "test" in task_class.capability_tags


def test_get_task_class_invalid():
    """Test getting non-existent task."""
    with pytest.raises(ValueError, match="not found"):
        get_task_class("NonExistentTask-v0")


def test_task_instance_creation():
    """Test creating task instance."""
    task = TestTask(difficulty="easy")
    assert task.difficulty == "easy"
    assert task.get_max_steps() == 50


def test_task_instance_invalid_difficulty():
    """Test creating task with invalid difficulty."""
    with pytest.raises(ValueError, match="Unknown difficulty"):
        TestTask(difficulty="impossible")


def test_task_generation():
    """Test task generation."""
    task = TestTask(difficulty="easy")
    grid, config = task.generate(seed=42)

    assert isinstance(grid, Grid)
    assert grid.height == 5
    assert grid.width == 5
    assert "agent_start" in config
    assert "goal_positions" in config
    assert len(config["goal_positions"]) == 1


def test_task_generation_reproducibility():
    """Test that same seed produces same grid."""
    task1 = TestTask(difficulty="easy")
    task2 = TestTask(difficulty="easy")

    grid1, config1 = task1.generate(seed=42)
    grid2, config2 = task2.generate(seed=42)

    assert grid1 == grid2
    assert config1["agent_start"] == config2["agent_start"]
    assert config1["goal_positions"] == config2["goal_positions"]


def test_make_environment():
    """Test creating environment via make()."""
    env = make("TestTask-v0", difficulty="easy", render_mode="ascii", seed=42)

    assert env is not None
    assert env.render_mode == "ascii"
    assert env.max_steps == 50


def test_make_environment_reset():
    """Test that created environment can be reset."""
    env = make("TestTask-v0", difficulty="easy", seed=42)
    obs, info = env.reset()

    assert obs is not None
    assert isinstance(info, dict)


def test_make_environment_step():
    """Test that created environment can step."""
    env = make("TestTask-v0", difficulty="easy", seed=42)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)  # NOOP

    assert obs is not None
    assert isinstance(reward, (int, float))
    assert terminated in (True, False)  # Accept bool or np.bool_
    assert truncated in (True, False)


def test_task_validation():
    """Test task instance validation."""
    task = TestTask(difficulty="easy")
    grid, config = task.generate(seed=42)

    # Should be valid (goal is reachable)
    assert task.validate_instance(grid, config)


def test_reward_computation():
    """Test reward computation."""
    env = make("TestTask-v0", difficulty="easy", reward_mode="dense", seed=42)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    # Dense reward should be -0.01 (step penalty)
    assert abs(reward - (-0.01)) < 0.001


def test_success_checking():
    """Test success condition."""
    env = make("TestTask-v0", difficulty="easy", seed=42)
    env.reset()

    # Move agent to goal
    goal_pos = env.task_config["goal_positions"][0]
    env.agent.position = goal_pos

    # Check success
    state = env._get_state_for_reward()
    state.update(
        {
            "grid": env.grid,
            "agent": env.agent,
            "config": env.task_config,
        }
    )

    assert env.task.check_success(state)


def test_sparse_vs_dense_rewards():
    """Test that sparse and dense rewards differ."""
    env_sparse = make("TestTask-v0", difficulty="easy", reward_mode="sparse", seed=42)
    env_dense = make("TestTask-v0", difficulty="easy", reward_mode="dense", seed=42)

    # Use seed=0 which is known to not place agent on goal for this 5x5 TestTask
    env_sparse.reset(seed=0)
    env_dense.reset(seed=0)

    # Take same action
    _, reward_sparse, *_ = env_sparse.step(0)
    _, reward_dense, *_ = env_dense.step(0)

    # Dense reward should have step penalty, sparse should be 0
    assert reward_sparse == 0.0
    assert reward_dense < 0.0
