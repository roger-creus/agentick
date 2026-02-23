"""Load, cache, and map Kenney isometric tiles to game entities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance

# Base tile dimensions (new dedicated sprites are 118x128; old 111px Kenney tiles
# get scaled to the same target width, so the ~6% stretch is negligible)
BASE_TILE_W = 118
BASE_TILE_H = 128
# The diamond base occupies roughly the bottom 64px of the 128px tile
DIAMOND_H = 64
# The cube depth (vertical extent above the diamond base) is ~64px
TILE_DEPTH = 64


def _default_tiles_dir() -> Path:
    """Return the default tiles directory shipped with the package."""
    return Path(__file__).parent / "tiles"


def _draw_iso_diamond(
    color: tuple[int, int, int, int],
    width: int,
    height: int,
) -> Image.Image:
    """Draw a colored isometric diamond as fallback when PNG is missing.

    Creates a simple isometric cube with top face, left face, and right face.
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = width // 2
    # Diamond base midpoint
    base_y = height * 3 // 4
    diamond_h = height // 4
    cube_depth = height // 4

    # Top face (lighter)
    r, g, b, a = color
    top_color = (min(r + 40, 255), min(g + 40, 255), min(b + 40, 255), a)
    top_points = [
        (cx, base_y - diamond_h - cube_depth),
        (width, base_y - cube_depth),
        (cx, base_y - cube_depth + diamond_h),
        (0, base_y - cube_depth),
    ]
    draw.polygon(top_points, fill=top_color)

    # Left face (darker)
    left_color = (max(r - 30, 0), max(g - 30, 0), max(b - 30, 0), a)
    left_points = [
        (0, base_y - cube_depth),
        (cx, base_y - cube_depth + diamond_h),
        (cx, base_y + diamond_h),
        (0, base_y),
    ]
    draw.polygon(left_points, fill=left_color)

    # Right face (medium)
    right_color = (max(r - 15, 0), max(g - 15, 0), max(b - 15, 0), a)
    right_points = [
        (cx, base_y - cube_depth + diamond_h),
        (width, base_y - cube_depth),
        (width, base_y),
        (cx, base_y + diamond_h),
    ]
    draw.polygon(right_points, fill=right_color)

    return img


# Default fallback colors for entity types (RGBA)
_FALLBACK_COLORS: dict[str, tuple[int, int, int, int]] = {
    "floor": (220, 210, 190, 255),
    "wall": (80, 80, 90, 255),
    "agent": (30, 120, 255, 255),
    "agent_up": (30, 120, 255, 255),
    "agent_down": (30, 120, 255, 255),
    "agent_left": (30, 120, 255, 255),
    "agent_right": (30, 120, 255, 255),
    "goal": (50, 200, 50, 255),
    "hazard": (230, 60, 20, 255),
    "water": (30, 130, 230, 255),
    "ice": (180, 220, 240, 255),
    "hole": (20, 20, 20, 255),
    "key": (240, 200, 40, 255),
    "golden_key": (240, 200, 40, 255),
    "red_key": (220, 50, 50, 255),
    "blue_key": (50, 100, 220, 255),
    "door": (140, 80, 30, 255),
    "door_open": (200, 180, 140, 255),
    "golden_door_ud": (180, 140, 40, 255),
    "golden_door_rl": (180, 140, 40, 255),
    "golden_door_open_ud": (220, 200, 120, 255),
    "golden_door_open_rl": (220, 200, 120, 255),
    "red_door_ud": (180, 40, 40, 255),
    "red_door_rl": (180, 40, 40, 255),
    "red_door_open_ud": (220, 120, 120, 255),
    "red_door_open_rl": (220, 120, 120, 255),
    "blue_door_ud": (40, 80, 180, 255),
    "blue_door_rl": (40, 80, 180, 255),
    "blue_door_open_ud": (120, 160, 220, 255),
    "blue_door_open_rl": (120, 160, 220, 255),
    "switch": (100, 100, 180, 255),
    "switch_on": (80, 200, 80, 255),
    "switch_off": (100, 100, 180, 255),
    "lever_on": (180, 180, 60, 255),
    "lever_off": (140, 120, 60, 255),
    "cage": (140, 80, 30, 255),
    "box": (180, 140, 80, 255),
    "target": (40, 220, 40, 255),
    "tool": (160, 160, 160, 255),
    "resource": (200, 180, 60, 255),
    "breadcrumb": (220, 180, 200, 255),
    "npc": (60, 180, 180, 255),
    "npc_up": (60, 180, 180, 255),
    "npc_down": (60, 180, 180, 255),
    "npc_left": (60, 180, 180, 255),
    "npc_right": (60, 180, 180, 255),
    "enemy": (220, 40, 40, 255),
    "enemy_up": (220, 40, 40, 255),
    "enemy_down": (220, 40, 40, 255),
    "enemy_left": (220, 40, 40, 255),
    "enemy_right": (220, 40, 40, 255),
    "sheep": (240, 240, 240, 255),
    "sheep_up": (240, 240, 240, 255),
    "sheep_down": (240, 240, 240, 255),
    "sheep_left": (240, 240, 240, 255),
    "sheep_right": (240, 240, 240, 255),
    "blocker": (200, 120, 40, 255),
    "gem": (148, 0, 211, 255),
    "lever": (180, 140, 60, 255),
    "potion": (0, 180, 180, 255),
    "scroll": (210, 180, 140, 255),
    "coin": (255, 200, 0, 255),
    "orb": (255, 105, 180, 255),
    "fog": (30, 30, 35, 255),
    "fog_partial": (80, 80, 85, 255),
}

# Color tint mapping for variants (applied as multiplicative blend)
_VARIANT_TINTS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.3, 0.3),
    "blue": (0.3, 0.4, 1.0),
    "yellow": (1.0, 0.9, 0.3),
    "green": (0.3, 1.0, 0.4),
    "purple": (0.7, 0.3, 1.0),
    "cyan": (0.3, 0.9, 1.0),
    "white": (1.0, 1.0, 1.0),
    "orange": (1.0, 0.6, 0.2),
    "pink": (1.0, 0.5, 0.7),
}


class TileAtlas:
    """Manages isometric tile images and entity-to-tile mapping.

    Loads tiles on first access and caches them. Falls back to colored
    diamond primitives if tile PNGs are missing.
    """

    def __init__(
        self,
        tiles_dir: Path | str | None = None,
        tile_scale: float = 1.0,
    ):
        self._tiles_dir = Path(tiles_dir) if tiles_dir else _default_tiles_dir()
        self._tile_scale = tile_scale
        self._cache: dict[str, Image.Image] = {}
        self._mapping = self._load_mapping()

        # Compute scaled dimensions
        self._scaled_w = max(1, int(BASE_TILE_W * tile_scale))
        self._scaled_h = max(1, int(BASE_TILE_H * tile_scale))
        self._scaled_diamond_h = max(1, int(DIAMOND_H * tile_scale))
        self._scaled_depth = max(1, int(TILE_DEPTH * tile_scale))

    def _load_mapping(self) -> dict[str, str | None]:
        """Load entity-to-tile filename mapping from JSON."""
        mapping_path = self._tiles_dir / "mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                data = json.load(f)
            # Remove comment keys
            return {k: v for k, v in data.items() if not k.startswith("_")}
        return {}

    def get_tile(
        self, entity_type: str | int, variant: str = "default"
    ) -> Image.Image:
        """Get the tile image for a game entity.

        Args:
            entity_type: Entity type name ("floor", "wall", etc.) or int enum value
            variant: Color variant ("default", "red", "blue", etc.)

        Returns:
            RGBA PIL Image of the tile, scaled to current tile_scale
        """
        # Convert int enum values to string names
        if isinstance(entity_type, int):
            entity_type = self._int_to_name(entity_type)

        cache_key = f"{entity_type}:{variant}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load the base tile
        tile = self._load_base_tile(entity_type)

        # Apply variant tint if not default
        if variant != "default" and variant in _VARIANT_TINTS:
            tile = self._apply_tint(tile, _VARIANT_TINTS[variant])

        # Scale if needed
        if self._tile_scale != 1.0:
            tile = tile.resize(
                (self._scaled_w, self._scaled_h), Image.LANCZOS
            )

        self._cache[cache_key] = tile
        return tile

    def _load_base_tile(self, entity_type: str) -> Image.Image:
        """Load a tile PNG from disk or create fallback."""
        # Check mapping
        filename = self._mapping.get(entity_type)
        if filename is None:
            return self._create_fallback(entity_type)

        tile_path = self._tiles_dir / filename
        if tile_path.exists():
            img = Image.open(tile_path).convert("RGBA")
            return img

        # PNG missing — use fallback
        return self._create_fallback(entity_type)

    def _create_fallback(self, entity_type: str) -> Image.Image:
        """Draw a colored isometric diamond as fallback."""
        color = _FALLBACK_COLORS.get(entity_type, (180, 180, 180, 255))
        return _draw_iso_diamond(color, BASE_TILE_W, BASE_TILE_H)

    def _apply_tint(
        self, img: Image.Image, tint: tuple[float, float, float]
    ) -> Image.Image:
        """Apply a color tint via per-channel multiply."""
        img = img.copy()
        r, g, b, a = img.split()

        # Multiply each channel
        r = r.point(lambda p: int(p * tint[0]))
        g = g.point(lambda p: int(p * tint[1]))
        b = b.point(lambda p: int(p * tint[2]))

        return Image.merge("RGBA", (r, g, b, a))

    @staticmethod
    def _int_to_name(val: int) -> str:
        """Convert integer CellType to entity name string.

        NOTE: Only use for terrain/cell values. For object values, use
        :meth:`_obj_int_to_name` instead — CellType and ObjectType share
        integer values 0-5 which causes mis-mapping.
        """
        from agentick.core.types import CellType

        _CELL_NAMES = {
            CellType.EMPTY: "floor",
            CellType.WALL: "wall",
            CellType.HAZARD: "hazard",
            CellType.WATER: "water",
            CellType.ICE: "ice",
            CellType.HOLE: "hole",
        }
        return _CELL_NAMES.get(CellType(val), "floor") if val in CellType._value2member_map_ else "floor"

    @staticmethod
    def _obj_int_to_name(val: int) -> str:
        """Convert integer ObjectType to entity name string.

        Unlike :meth:`_int_to_name` this only checks ObjectType, avoiding
        collisions with CellType values 0-5.
        """
        from agentick.core.types import ObjectType

        _OBJ_NAMES = {
            ObjectType.NONE: "empty",
            ObjectType.GOAL: "goal",
            ObjectType.KEY: "key",
            ObjectType.DOOR: "door",
            ObjectType.SWITCH: "switch",
            ObjectType.BOX: "box",
            ObjectType.TARGET: "target",
            ObjectType.TOOL: "tool",
            ObjectType.RESOURCE: "resource",
            ObjectType.BREADCRUMB: "breadcrumb",
            ObjectType.NPC: "npc",
            ObjectType.ENEMY: "enemy",
            ObjectType.SHEEP: "sheep",
            ObjectType.BLOCKER: "blocker",
            ObjectType.GEM: "gem",
            ObjectType.LEVER: "lever",
            ObjectType.POTION: "potion",
            ObjectType.SCROLL: "scroll",
            ObjectType.COIN: "coin",
            ObjectType.ORB: "orb",
        }
        return _OBJ_NAMES.get(ObjectType(val), "empty") if val in ObjectType._value2member_map_ else "empty"

    def clear_cache(self) -> None:
        """Clear all cached tile images."""
        self._cache.clear()

    def rescale(self, new_scale: float) -> None:
        """Change tile scale and clear cache."""
        self._tile_scale = new_scale
        self._scaled_w = max(1, int(BASE_TILE_W * new_scale))
        self._scaled_h = max(1, int(BASE_TILE_H * new_scale))
        self._scaled_diamond_h = max(1, int(DIAMOND_H * new_scale))
        self._scaled_depth = max(1, int(TILE_DEPTH * new_scale))
        self._cache.clear()

    @property
    def tile_width(self) -> int:
        """Scaled tile image width."""
        return self._scaled_w

    @property
    def tile_height(self) -> int:
        """Scaled diamond-base height (used for isometric projection)."""
        return self._scaled_diamond_h

    @property
    def tile_depth(self) -> int:
        """Scaled vertical extent of tile cube above diamond base."""
        return self._scaled_depth
