"""Core type definitions for Agentick."""

from enum import IntEnum

# Type aliases
Position = tuple[int, int]


class CellType(IntEnum):
    """Types of cells in the grid terrain layer."""

    EMPTY = 0
    WALL = 1
    HAZARD = 2
    WATER = 3
    ICE = 4
    HOLE = 5


class ObjectType(IntEnum):
    """Types of objects that can be placed on the grid."""

    NONE = 0
    GOAL = 1
    KEY = 2
    DOOR = 3
    SWITCH = 4
    BOX = 5
    TARGET = 6
    TOOL = 7
    RESOURCE = 8
    BREADCRUMB = 9
    NPC = 10  # neutral/allied NPC (rendered as cyan circle)
    ENEMY = 11  # enemy / adversary (rendered as red circle)
    SHEEP = 12  # herding target (rendered as white circle)
    BLOCKER = 13  # moving obstacle / blocker (rendered as orange square)
    # New object types for task differentiation
    GEM = 14  # collectible gem/crystal (purple diamond)
    LEVER = 15  # physical lever/button (different from switch)
    POTION = 16  # consumable potion/flask (teal)
    SCROLL = 17  # readable scroll/book (parchment)
    COIN = 18  # collectible coin/token (gold circle)
    ORB = 19  # magical orb (glowing pink)


# Objects that block movement (agent must face + INTERACT instead of walking onto them).
# DOOR, LEVER, SWITCH are solid; KEY, GEM, COIN, ORB etc. remain walkable auto-pickup.
NON_WALKABLE_OBJECTS = frozenset({ObjectType.DOOR, ObjectType.LEVER, ObjectType.SWITCH})


class AgentType(IntEnum):
    """Types of agents on the grid."""

    NONE = 0
    AGENT = 1
    NPC = 2
    ENEMY = 3


class Direction(IntEnum):
    """Cardinal directions for agent orientation."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def opposite(self) -> "Direction":
        """Return the opposite direction."""
        return Direction((self + 2) % 4)

    def rotate_left(self) -> "Direction":
        """Rotate 90 degrees counter-clockwise."""
        return Direction((self - 1) % 4)

    def rotate_right(self) -> "Direction":
        """Rotate 90 degrees clockwise."""
        return Direction((self + 1) % 4)

    def to_delta(self) -> Position:
        """Convert direction to (dx, dy) delta."""
        deltas = {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
        }
        return deltas[self]


class ActionType(IntEnum):
    """Discrete action types."""

    NOOP = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    INTERACT = 5
    # Extended actions for partial observability
    ROTATE_LEFT = 6
    ROTATE_RIGHT = 7
    MOVE_FORWARD = 8


# Color constants for rendering
COLORS = {
    "wall": (64, 64, 64),
    "empty": (240, 240, 240),
    "agent": (0, 100, 255),
    "goal": (0, 200, 0),
    "hazard": (255, 0, 0),
    "key": (255, 215, 0),
    "door": (139, 69, 19),
    "switch": (128, 128, 255),
    "box": (210, 180, 140),
    "target": (0, 255, 0),
    "water": (0, 191, 255),
    "ice": (173, 216, 230),
    "breadcrumb": (255, 192, 203),
    "fog": (128, 128, 128),
    "gem": (148, 0, 211),
    "lever": (180, 140, 60),
    "potion": (0, 180, 180),
    "scroll": (210, 180, 140),
    "coin": (255, 200, 0),
    "orb": (255, 105, 180),
}


# ASCII characters for rendering
ASCII_CHARS = {
    CellType.EMPTY: ".",
    CellType.WALL: "#",
    CellType.HAZARD: "X",
    CellType.WATER: "~",
    CellType.ICE: "i",
    CellType.HOLE: "O",
    ObjectType.GOAL: "G",
    ObjectType.KEY: "K",
    ObjectType.DOOR: "D",
    ObjectType.SWITCH: "S",
    ObjectType.BOX: "B",
    ObjectType.TARGET: "T",
    ObjectType.TOOL: "t",
    ObjectType.RESOURCE: "r",
    ObjectType.BREADCRUMB: "*",
    ObjectType.GEM: "d",
    ObjectType.LEVER: "L",
    ObjectType.POTION: "P",
    ObjectType.SCROLL: "?",
    ObjectType.COIN: "c",
    ObjectType.ORB: "O",
    AgentType.AGENT: "A",
    AgentType.NPC: "N",
    AgentType.ENEMY: "E",
}
