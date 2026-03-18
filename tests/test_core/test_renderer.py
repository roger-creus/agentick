"""Tests for rendering system."""

import numpy as np
import pytest

from agentick.core.entity import Agent, Entity
from agentick.core.grid import Grid
from agentick.core.renderer import (
    ASCIIRenderer,
    LanguageRenderer,
    StateDictRenderer,
    create_renderer,
)
from agentick.core.types import CellType, Direction, ObjectType


@pytest.fixture
def simple_grid():
    """Create a simple 5x5 grid."""
    grid = Grid(5, 5)
    grid.terrain[0, :] = CellType.WALL
    grid.terrain[4, :] = CellType.WALL
    grid.terrain[:, 0] = CellType.WALL
    grid.terrain[:, 4] = CellType.WALL
    grid.objects[3, 3] = ObjectType.GOAL
    return grid


@pytest.fixture
def agent():
    """Create a test agent."""
    return Agent(
        id="agent_1",
        entity_type="agent",
        position=(2, 2),
        orientation=Direction.NORTH,
    )


@pytest.fixture
def info_dict():
    """Create test info dict."""
    return {
        "valid_actions": ["move_up", "move_down", "move_left", "move_right"],
        "step_count": 5,
        "max_steps": 100,
    }


def test_ascii_renderer(simple_grid, agent, info_dict):
    """Test ASCII rendering."""
    renderer = ASCIIRenderer()
    output = renderer.render(simple_grid, [], agent, info_dict)

    assert isinstance(output, str)
    assert "\n" in output
    # New renderer uses directional arrows instead of "A"
    assert any(char in output for char in ["^", "v", "<", ">"])  # Agent character
    assert "G" in output  # Goal character
    assert "#" in output  # Wall character

    lines = output.split("\n")
    # New renderer includes legend, so more than 5 lines
    assert len(lines) >= 5  # At least 5x5 grid plus legend
    assert "Legend" in output  # Legend is included


def test_language_renderer_narrative(simple_grid, agent, info_dict):
    """Test narrative language rendering."""
    renderer = LanguageRenderer(structured=False)
    output = renderer.render(simple_grid, [], agent, info_dict)

    assert isinstance(output, str)
    # New renderer uses natural language with relative descriptions
    assert "facing north" in output.lower()
    assert "goal" in output.lower()
    # Check for action information
    assert "move" in output.lower() or "Actions" in output


def test_language_renderer_structured(simple_grid, agent, info_dict):
    """Test structured language rendering."""
    renderer = LanguageRenderer(structured=True)
    output = renderer.render(simple_grid, [], agent, info_dict)

    assert isinstance(output, dict)
    assert "position" in output
    assert output["position"]["x"] == 2
    assert output["position"]["y"] == 2
    assert "orientation" in output
    assert output["orientation"] == "north"
    # New renderer uses different keys
    assert "visible_entities" in output or "goals" in output
    assert "inventory" in output
    assert "valid_actions" in output


def test_language_renderer_inventory(simple_grid, agent, info_dict):
    """Test language rendering with inventory."""
    key = Entity(id="key_1", entity_type="key", position=(0, 0))
    agent.add_to_inventory(key)

    renderer = LanguageRenderer(structured=False)
    output = renderer.render(simple_grid, [], agent, info_dict)

    assert "carrying: key" in output


def test_state_dict_renderer(simple_grid, agent, info_dict):
    """Test state dict rendering."""
    renderer = StateDictRenderer()
    output = renderer.render(simple_grid, [], agent, info_dict)

    assert isinstance(output, dict)
    assert "grid" in output
    assert "agent" in output
    assert "entities" in output
    assert "info" in output

    # Check grid structure
    assert output["grid"]["height"] == 5
    assert output["grid"]["width"] == 5
    assert "terrain" in output["grid"]
    assert "objects" in output["grid"]

    # Check agent structure
    assert output["agent"]["position"] == (2, 2)
    assert output["agent"]["orientation"] == "north"


def test_create_renderer_ascii():
    """Test renderer factory for ASCII."""
    renderer = create_renderer("ascii")
    assert isinstance(renderer, ASCIIRenderer)


def test_create_renderer_language():
    """Test renderer factory for language."""
    renderer = create_renderer("language")
    assert isinstance(renderer, LanguageRenderer)
    assert not renderer.structured


def test_create_renderer_language_structured():
    """Test renderer factory for structured language."""
    renderer = create_renderer("language_structured")
    assert isinstance(renderer, LanguageRenderer)
    assert renderer.structured


def test_create_renderer_state_dict():
    """Test renderer factory for state dict."""
    renderer = create_renderer("state_dict")
    assert isinstance(renderer, StateDictRenderer)


def test_create_renderer_invalid():
    """Test renderer factory with invalid mode."""
    with pytest.raises(ValueError):
        create_renderer("invalid_mode")
