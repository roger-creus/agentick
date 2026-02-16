"""Mapping from game entity types to 3D model slugs and color tints."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType

# Map entity identifiers to GLB model slug names.
# Keys can be CellType/ObjectType enum values or string entity_type values.
# A value of None means the entity is handled specially (e.g. fog overlay).

ENTITY_MODEL_MAP: dict[str, str | None] = {
    # CellType terrain → model slug
    "wall": "wall",
    "floor": "floor",
    "hazard": "hazard",
    "water": "water",
    "ice": "ice",
    "hole": "hole",
    # ObjectType → model slug
    "goal": "goal",
    "key": "key",
    "key_red": "key",
    "key_blue": "key",
    "key_yellow": "key",
    "key_green": "key",
    "door": "door",
    "door_open": "door",
    "switch": "switch",
    "switch_off": "switch",
    "switch_on": "switch",
    "box": "box",
    "box_on_target": "box",
    "target": "target",
    "tool": "tool",
    "resource": "gem",
    "breadcrumb": "breadcrumb",
    # Agent/entity types → model slug
    "agent": "player",
    "player": "player",
    "agent_2": "npc",
    "npc": "npc",
    "enemy": "enemy",
    # Collectible/effect types
    "heart": "heart",
    "gem": "gem",
    "lightning": "lightning",
    # Fog of war — handled via overlay, not a model
    "fog": None,
    "fog_partial": None,
}

# CellType enum → model slug (for iterating terrain layer)
CELLTYPE_MODEL_MAP: dict[int, str | None] = {
    CellType.EMPTY: "floor",
    CellType.WALL: "wall",
    CellType.HAZARD: "hazard",
    CellType.WATER: "water",
    CellType.ICE: "ice",
    CellType.HOLE: "hole",
}

# ObjectType enum → model slug (for iterating object layer)
OBJECTTYPE_MODEL_MAP: dict[int, str | None] = {
    ObjectType.NONE: None,
    ObjectType.GOAL: "goal",
    ObjectType.KEY: "key",
    ObjectType.DOOR: "door",
    ObjectType.SWITCH: "switch",
    ObjectType.BOX: "box",
    ObjectType.TARGET: "target",
    ObjectType.TOOL: "tool",
    ObjectType.RESOURCE: "gem",
    ObjectType.BREADCRUMB: "breadcrumb",
}

# Color tints for key/door variants (RGB float multipliers).
COLOR_TINTS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.3, 0.3),
    "blue": (0.3, 0.3, 1.0),
    "yellow": (1.0, 0.9, 0.2),
    "green": (0.3, 1.0, 0.3),
}

# Objects that should float above the ground plane
FLOATING_OBJECTS: set[str] = {
    "key",
    "heart",
    "gem",
    "lightning",
    "breadcrumb",
    "goal",
    "resource",
}
