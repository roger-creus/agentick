"""Integration tests for world model evaluation with environment."""

import pytest

from agentick import make
from agentick.tasks.registry import list_tasks
from agentick.worldmodel.evaluator import WorldModelEvaluator


def test_env_generate_worldmodel_tests_exists():
    """Test that env.generate_worldmodel_tests() method exists."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii")
    assert hasattr(env, "generate_worldmodel_tests")
    assert callable(env.generate_worldmodel_tests)


def test_env_generate_worldmodel_tests_returns_factories():
    """Test that generate_worldmodel_tests() returns proper factory functions."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii")
    factories = env.generate_worldmodel_tests()

    # Check that it returns a dict with expected keys
    assert isinstance(factories, dict)
    assert "env_factory" in factories
    assert "modified_env_factory" in factories
    assert "change_env_factory" in factories

    # Check that all values are callable
    assert callable(factories["env_factory"])
    assert callable(factories["modified_env_factory"])
    assert callable(factories["change_env_factory"])


def test_env_factories_create_working_environments():
    """Test that the generated factories create working environments."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii")
    factories = env.generate_worldmodel_tests()

    # Test env_factory
    test_env = factories["env_factory"]()
    assert test_env is not None
    obs, info = test_env.reset(seed=42)
    assert obs is not None
    assert info is not None

    # Test modified_env_factory
    modified_env = factories["modified_env_factory"]()
    assert modified_env is not None
    obs, info = modified_env.reset(seed=42)
    assert obs is not None

    # Test change_env_factory
    change_env = factories["change_env_factory"]()
    assert change_env is not None
    obs, info = change_env.reset(seed=42)
    assert obs is not None


def test_world_model_evaluator_with_generated_factories():
    """Test that WorldModelEvaluator can use the generated factories."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii")
    factories = env.generate_worldmodel_tests()

    # Create evaluator with generated factories
    evaluator = WorldModelEvaluator(
        env_factory=factories["env_factory"],
        modified_env_factory=factories["modified_env_factory"],
        change_env_factory=factories["change_env_factory"],
    )

    assert evaluator is not None
    assert evaluator.env_factory is not None
    assert evaluator.modified_env_factory is not None
    assert evaluator.change_env_factory is not None


@pytest.mark.parametrize(
    "task_id",
    [
        "KeyDoorPuzzle-v0",
        "BacktrackPuzzle-v0",
        "DelayedGratification-v0",
        "EnvironmentShift-v0",
        "PhysicsDiscovery-v0",
    ],
)
def test_generate_worldmodel_tests_for_multiple_tasks(task_id):
    """Test that generate_worldmodel_tests() works for various tasks."""
    env = make(task_id, difficulty="easy", render_mode="ascii")
    factories = env.generate_worldmodel_tests()

    # Verify structure
    assert isinstance(factories, dict)
    assert "env_factory" in factories
    assert "modified_env_factory" in factories
    assert "change_env_factory" in factories

    # Verify factories work
    test_env = factories["env_factory"]()
    obs, _ = test_env.reset(seed=42)
    assert obs is not None


def test_generated_factories_preserve_task_config():
    """Test that factories preserve the original task configuration."""
    env = make("KeyDoorPuzzle-v0", difficulty="hard", render_mode="language")
    factories = env.generate_worldmodel_tests()

    # Create new env from factory
    new_env = factories["env_factory"]()

    # Check that key config is preserved
    assert new_env.render_mode == env.render_mode
    assert new_env.reward_mode == env.reward_mode
    # max_steps might differ for modified_env_factory


def test_modified_env_factory_creates_different_env():
    """Test that modified_env_factory creates a slightly different environment."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii")
    factories = env.generate_worldmodel_tests()

    base_env = factories["env_factory"]()
    modified_env = factories["modified_env_factory"]()

    # Modified env should have different max_steps (uses different difficulty level)
    # The exact difference depends on the task's difficulty configurations
    assert modified_env.max_steps != base_env.max_steps
    # Should be larger (harder difficulty typically has more steps)
    assert modified_env.max_steps > base_env.max_steps


def test_all_registered_tasks_have_worldmodel_tests():
    """Test that all registered tasks support generate_worldmodel_tests()."""
    all_tasks = list_tasks()

    for task_id in all_tasks:
        try:
            env = make(task_id, difficulty="easy", render_mode="ascii")
            factories = env.generate_worldmodel_tests()

            # Verify structure
            assert isinstance(factories, dict)
            assert "env_factory" in factories
            assert "modified_env_factory" in factories
            assert "change_env_factory" in factories

            # Verify at least env_factory works
            test_env = factories["env_factory"]()
            obs, _ = test_env.reset(seed=42)
            assert obs is not None

        except Exception as e:
            pytest.fail(f"Task {task_id} failed worldmodel test generation: {e}")
