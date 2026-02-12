"""Entity system for gridworld agents and objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentick.core.types import Direction, Position


@dataclass(slots=True)
class Entity:
    """
    Base entity class for all objects in the gridworld.

    Attributes:
        id: Unique identifier for this entity
        entity_type: String type identifier (e.g., "key", "goal")
        position: (x, y) position on the grid
        properties: Arbitrary properties dictionary
    """

    id: str
    entity_type: str
    position: Position
    properties: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> Entity:
        """Create a deep copy of this entity."""
        return Entity(
            id=self.id,
            entity_type=self.entity_type,
            position=self.position,
            properties=self.properties.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize entity to dictionary."""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "position": self.position,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Deserialize entity from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data["entity_type"],
            position=tuple(data["position"]),
            properties=data.get("properties", {}),
        )


@dataclass(slots=True)
class Agent(Entity):
    """
    Agent entity with orientation, inventory, and status.

    Additional attributes:
        orientation: Direction the agent is facing
        inventory: List of entities the agent is carrying
        energy: Energy level (0.0 to 1.0)
        health: Health level (0.0 to 1.0)
    """

    orientation: Direction = Direction.NORTH
    inventory: list[Entity] = field(default_factory=list)
    energy: float = 1.0
    health: float = 1.0

    def __post_init__(self):
        """Ensure entity_type is set correctly."""
        if self.entity_type == "":
            self.entity_type = "agent"

    def copy(self) -> Agent:
        """Create a deep copy of this agent."""
        return Agent(
            id=self.id,
            entity_type=self.entity_type,
            position=self.position,
            properties=self.properties.copy(),
            orientation=self.orientation,
            inventory=[item.copy() for item in self.inventory],
            energy=self.energy,
            health=self.health,
        )

    def add_to_inventory(self, item: Entity) -> bool:
        """
        Add an item to inventory.

        Args:
            item: Entity to add

        Returns:
            True if added successfully
        """
        max_inventory = self.properties.get("max_inventory", 10)
        if len(self.inventory) < max_inventory:
            self.inventory.append(item)
            return True
        return False

    def remove_from_inventory(self, item_id: str) -> Entity | None:
        """
        Remove an item from inventory by ID.

        Args:
            item_id: ID of item to remove

        Returns:
            Removed entity, or None if not found
        """
        for i, item in enumerate(self.inventory):
            if item.id == item_id:
                return self.inventory.pop(i)
        return None

    def has_item(self, entity_type: str) -> bool:
        """Check if agent has an item of given type in inventory."""
        return any(item.entity_type == entity_type for item in self.inventory)

    def get_item(self, entity_type: str) -> Entity | None:
        """Get first item of given type from inventory."""
        for item in self.inventory:
            if item.entity_type == entity_type:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent to dictionary."""
        base = Entity.to_dict(self)
        base.update(
            {
                "orientation": int(self.orientation),
                "inventory": [item.to_dict() for item in self.inventory],
                "energy": self.energy,
                "health": self.health,
            }
        )
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Deserialize agent from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data["entity_type"],
            position=tuple(data["position"]),
            properties=data.get("properties", {}),
            orientation=Direction(data.get("orientation", 0)),
            inventory=[Entity.from_dict(item) for item in data.get("inventory", [])],
            energy=data.get("energy", 1.0),
            health=data.get("health", 1.0),
        )


class EntityRegistry:
    """
    Registry for custom entity types.

    Allows tasks to define and register custom entity behaviors.
    """

    def __init__(self):
        self._types: dict[str, type] = {
            "entity": Entity,
            "agent": Agent,
        }

    def register(self, entity_type: str, entity_class: type) -> None:
        """
        Register a custom entity type.

        Args:
            entity_type: String identifier for the entity type
            entity_class: Class to use for entities of this type
        """
        self._types[entity_type] = entity_class

    def create(self, entity_type: str, **kwargs: Any) -> Entity:
        """
        Create an entity of the given type.

        Args:
            entity_type: Type of entity to create
            **kwargs: Arguments passed to entity constructor

        Returns:
            New entity instance
        """
        entity_class = self._types.get(entity_type, Entity)
        if "entity_type" not in kwargs:
            kwargs["entity_type"] = entity_type
        return entity_class(**kwargs)

    def is_registered(self, entity_type: str) -> bool:
        """Check if entity type is registered."""
        return entity_type in self._types


# Global registry instance
_global_registry = EntityRegistry()


def register_entity_type(entity_type: str, entity_class: type) -> None:
    """Register a custom entity type globally."""
    _global_registry.register(entity_type, entity_class)


def create_entity(entity_type: str, **kwargs: Any) -> Entity:
    """Create an entity using the global registry."""
    return _global_registry.create(entity_type, **kwargs)
