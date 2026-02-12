"""Tests for experiment configuration."""

import pytest
import yaml

from agentick.experiments.config import ExperimentConfig


def test_basic_config():
    """Test basic config creation."""
    config = ExperimentConfig(
        name="test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=10,
    )

    assert config.name == "test"
    assert config.agent.type == "random"
    assert config.tasks == ["GoToGoal-v0"]
    assert config.n_episodes == 10


def test_config_defaults():
    """Test config defaults."""
    config = ExperimentConfig(
        name="test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
    )

    assert config.n_episodes == 10
    assert config.n_seeds == 3
    assert config.render_modes == ["ascii"]


def test_config_yaml_serialization(tmp_path):
    """Test YAML serialization."""
    config = ExperimentConfig(
        name="test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=50,
    )

    # Save to YAML
    yaml_path = tmp_path / "config.yaml"
    config.to_yaml(yaml_path)

    # Load back
    loaded = ExperimentConfig.from_yaml(yaml_path)

    assert loaded.name == config.name
    assert loaded.agent == config.agent
    assert loaded.tasks == config.tasks
    assert loaded.n_episodes == config.n_episodes


def test_config_validation():
    """Test config validation."""
    from pydantic import ValidationError

    # No tasks should fail
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="test",
            agent={"type": "random"},
            tasks=[],
        )


def test_config_inheritance(tmp_path):
    """Test config inheritance."""
    # Base config
    base_config = ExperimentConfig(
        name="base",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=100,
    )

    base_path = tmp_path / "base.yaml"
    base_config.to_yaml(base_path)

    # Override config
    override = {
        "name": "derived",
        "n_episodes": 50,
    }

    override_path = tmp_path / "override.yaml"
    with open(override_path, "w") as f:
        yaml.dump(override, f)

    # Merge
    merged = ExperimentConfig.from_yaml(base_path)
    # Apply overrides manually
    for key, value in override.items():
        setattr(merged, key, value)

    assert merged.name == "derived"
    assert merged.n_episodes == 50
    assert merged.tasks == base_config.tasks
