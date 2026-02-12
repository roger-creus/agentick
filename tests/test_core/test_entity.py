"""Tests for Entity system."""

from agentick.core.entity import Agent, Entity, EntityRegistry, create_entity
from agentick.core.types import Direction


def test_entity_creation():
    """Test basic entity creation."""
    entity = Entity(
        id="key_1",
        entity_type="key",
        position=(3, 4),
        properties={"color": "red"},
    )

    assert entity.id == "key_1"
    assert entity.entity_type == "key"
    assert entity.position == (3, 4)
    assert entity.properties["color"] == "red"


def test_entity_copy():
    """Test entity deep copy."""
    entity = Entity(
        id="box_1",
        entity_type="box",
        position=(1, 2),
        properties={"weight": 10},
    )

    copied = entity.copy()
    assert copied.id == entity.id
    assert copied.position == entity.position
    assert copied.properties == entity.properties
    assert copied is not entity
    assert copied.properties is not entity.properties


def test_entity_serialization():
    """Test entity serialization."""
    entity = Entity(
        id="goal_1",
        entity_type="goal",
        position=(5, 5),
        properties={"score": 100},
    )

    data = entity.to_dict()
    restored = Entity.from_dict(data)

    assert restored.id == entity.id
    assert restored.entity_type == entity.entity_type
    assert restored.position == entity.position
    assert restored.properties == entity.properties


def test_agent_creation():
    """Test agent creation with defaults."""
    agent = Agent(
        id="agent_1",
        entity_type="agent",
        position=(0, 0),
    )

    assert agent.id == "agent_1"
    assert agent.orientation == Direction.NORTH
    assert len(agent.inventory) == 0
    assert agent.energy == 1.0
    assert agent.health == 1.0


def test_agent_inventory():
    """Test agent inventory management."""
    agent = Agent(
        id="agent_1",
        entity_type="agent",
        position=(0, 0),
    )

    key = Entity(id="key_1", entity_type="key", position=(1, 1))
    tool = Entity(id="tool_1", entity_type="tool", position=(2, 2))

    # Add items
    assert agent.add_to_inventory(key)
    assert agent.add_to_inventory(tool)
    assert len(agent.inventory) == 2

    # Check for items
    assert agent.has_item("key")
    assert agent.has_item("tool")
    assert not agent.has_item("sword")

    # Get item
    found_key = agent.get_item("key")
    assert found_key is not None
    assert found_key.id == "key_1"

    # Remove item
    removed = agent.remove_from_inventory("key_1")
    assert removed is not None
    assert removed.id == "key_1"
    assert len(agent.inventory) == 1
    assert not agent.has_item("key")


def test_agent_inventory_limit():
    """Test agent inventory size limit."""
    agent = Agent(
        id="agent_1",
        entity_type="agent",
        position=(0, 0),
        properties={"max_inventory": 2},
    )

    item1 = Entity(id="item_1", entity_type="item", position=(0, 0))
    item2 = Entity(id="item_2", entity_type="item", position=(0, 0))
    item3 = Entity(id="item_3", entity_type="item", position=(0, 0))

    assert agent.add_to_inventory(item1)
    assert agent.add_to_inventory(item2)
    assert not agent.add_to_inventory(item3)  # Over limit
    assert len(agent.inventory) == 2


def test_agent_copy():
    """Test agent deep copy."""
    agent = Agent(
        id="agent_1",
        entity_type="agent",
        position=(2, 3),
        orientation=Direction.EAST,
        energy=0.8,
        health=0.9,
    )

    key = Entity(id="key_1", entity_type="key", position=(0, 0))
    agent.add_to_inventory(key)

    copied = agent.copy()
    assert copied.id == agent.id
    assert copied.orientation == agent.orientation
    assert copied.energy == agent.energy
    assert copied.health == agent.health
    assert len(copied.inventory) == len(agent.inventory)
    assert copied is not agent
    assert copied.inventory is not agent.inventory


def test_agent_serialization():
    """Test agent serialization."""
    agent = Agent(
        id="agent_1",
        entity_type="agent",
        position=(1, 2),
        orientation=Direction.SOUTH,
        energy=0.7,
        health=0.6,
    )

    key = Entity(id="key_1", entity_type="key", position=(0, 0))
    agent.add_to_inventory(key)

    data = agent.to_dict()
    restored = Agent.from_dict(data)

    assert restored.id == agent.id
    assert restored.orientation == agent.orientation
    assert restored.energy == agent.energy
    assert restored.health == agent.health
    assert len(restored.inventory) == 1
    assert restored.inventory[0].id == "key_1"


def test_entity_registry():
    """Test entity registry."""
    registry = EntityRegistry()

    # Default types are registered
    assert registry.is_registered("entity")
    assert registry.is_registered("agent")

    # Create entities
    entity = registry.create("entity", id="e1", position=(0, 0))
    assert isinstance(entity, Entity)

    agent = registry.create("agent", id="a1", position=(0, 0))
    assert isinstance(agent, Agent)

    # Unknown types default to Entity
    unknown = registry.create("unknown", id="u1", position=(0, 0))
    assert isinstance(unknown, Entity)
    assert unknown.entity_type == "unknown"


def test_entity_registry_custom_type():
    """Test registering custom entity type."""
    from dataclasses import dataclass

    @dataclass
    class CustomEntity(Entity):
        custom_property: int = 0

    registry = EntityRegistry()
    registry.register("custom", CustomEntity)

    assert registry.is_registered("custom")

    custom = registry.create(
        "custom",
        id="c1",
        position=(0, 0),
        custom_property=42,
    )
    assert isinstance(custom, CustomEntity)
    assert custom.custom_property == 42


def test_global_entity_creation():
    """Test global entity creation helper."""
    entity = create_entity("entity", id="e1", position=(5, 5))
    assert isinstance(entity, Entity)
    assert entity.id == "e1"

    agent = create_entity("agent", id="a1", position=(3, 3))
    assert isinstance(agent, Agent)
    assert agent.id == "a1"
