"""Simple 2D grid renderer — clean top-down view, visually distinct objects.

Each tile fills with its terrain/object color. Objects dominate the tile.
Entities are circles drawn on top. Task name shown as header.
"""

from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentick.core.entity import Agent, Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType

# Object labels (one-letter, drawn inside the tile)
OBJECT_LABELS = {
    ObjectType.GOAL:      ("G", "#00DD00"),
    ObjectType.KEY:       ("K", "#FFD700"),
    ObjectType.DOOR:      ("D", "#8B4513"),
    ObjectType.BOX:       ("B", "#CD853F"),
    ObjectType.SWITCH:    ("S", "#32CD32"),
    ObjectType.RESOURCE:  ("R", "#00CED1"),
    ObjectType.TARGET:    ("T", "#AAFFAA"),
    ObjectType.TOOL:      ("T", "#FFA500"),
    ObjectType.BREADCRUMB:("·", "#777777"),
    ObjectType.NONE:      (None, None),
}

# These are drawn as circles (like entities), not squares
CIRCLE_OBJECTS = {
    ObjectType.NPC:     ("N", "#11AAEE"),
    ObjectType.ENEMY:   ("E", "#EE1111"),
    ObjectType.SHEEP:   ("S", "#FFFFFF"),
    ObjectType.BLOCKER: ("X", "#FF6600"),
}

TERRAIN_COLORS = {
    CellType.EMPTY:  "#DCDCDC",
    CellType.WALL:   "#3A3A3A",
    CellType.HAZARD: "#FF4500",
    CellType.WATER:  "#1E90FF",
    CellType.ICE:    "#B0E0E6",
    CellType.HOLE:   "#111111",
}

ENTITY_COLORS = {
    "player": "#0044FF",
    "enemy":  "#EE1111",
    "npc":    "#11AAEE",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _contrast_text(bg_hex: str) -> str:
    """Return black or white for best contrast on bg."""
    r, g, b = _hex_to_rgb(bg_hex)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if lum > 140 else "#FFFFFF"


class SimpleGridRenderer:
    """Top-down 2D grid renderer with prominent, visually distinct tiles."""

    HEADER_H = 32   # pixels for task-name header (must be divisible by 16 for FFMPEG)

    def __init__(self, tile_size: int = 32):
        # Ensure tile_size divisible by 16 for FFMPEG compatibility
        tile_size = ((tile_size + 15) // 16) * 16
        self.tile_size = tile_size
        try:
            self._font_big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", tile_size // 2)
            self._font_small= ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            self._font_big  = ImageFont.load_default()
            self._font_small= ImageFont.load_default()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> np.ndarray:
        H, W = grid.height, grid.width
        ts = self.tile_size
        header = self.HEADER_H

        img = Image.new("RGB", (W * ts, H * ts + header), (15, 15, 25))
        draw = ImageDraw.Draw(img)

        # ── Header: task name + step info ─────────────────────────────────
        task_name = info.get("task_name", info.get("task", ""))
        step      = info.get("step", "")
        hdr_text  = f"{task_name}  step={step}" if step != "" else task_name
        draw.text((4, 3), hdr_text, font=self._font_small, fill=(220, 220, 100))

        # ── Terrain ───────────────────────────────────────────────────────
        for y in range(H):
            for x in range(W):
                px, py = x * ts, y * ts + header
                cell = CellType(grid.terrain[y, x]) if grid.terrain[y, x] in CellType._value2member_map_ else CellType.EMPTY
                bg = TERRAIN_COLORS.get(cell, TERRAIN_COLORS[CellType.EMPTY])
                draw.rectangle([px, py, px + ts - 1, py + ts - 1],
                               fill=bg, outline="#555555", width=1)

                # ── Object on tile ────────────────────────────────────────
                obj_val = grid.objects[y, x]
                try:
                    obj = ObjectType(obj_val)
                except ValueError:
                    obj = ObjectType.NONE

                if obj == ObjectType.BREADCRUMB:
                    # Small dot
                    cx2 = px + ts // 2; cy2 = py + ts // 2; r2 = max(3, ts // 6)
                    draw.ellipse([cx2-r2, cy2-r2, cx2+r2, cy2+r2], fill="#888888")
                elif obj in CIRCLE_OBJECTS:
                    # Draw as entity circle
                    letter, color = CIRCLE_OBJECTS[obj]
                    self._draw_entity(draw, x, y, color, header, letter=letter)
                elif obj != ObjectType.NONE and obj in OBJECT_LABELS:
                    label, obj_color = OBJECT_LABELS[obj]
                    if label is not None:
                        # Fill ~75% of tile with object color
                        m = max(2, ts // 8)
                        draw.rectangle([px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                                       fill=obj_color, outline="#000000", width=2)
                        # Draw label in center
                        text_color = _contrast_text(obj_color)
                        bbox = draw.textbbox((0, 0), label, font=self._font_big)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                        draw.text((px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                                  label, font=self._font_big, fill=text_color)

        # ── Non-player entities ────────────────────────────────────────────
        for entity in entities:
            if entity is agent:
                continue
            ex, ey = entity.position
            color = ENTITY_COLORS.get(getattr(entity, "entity_type", "npc"), ENTITY_COLORS["npc"])
            self._draw_entity(draw, ex, ey, color, header, letter="E")

        # ── Player ────────────────────────────────────────────────────────
        ax, ay = agent.position
        self._draw_entity(draw, ax, ay, ENTITY_COLORS["player"], header, letter="A")

        return np.array(img)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_entity(
        self,
        draw: ImageDraw,
        gx: int, gy: int,
        color: str,
        header: int,
        letter: str = "A",
    ):
        ts = self.tile_size
        px = gx * ts
        py = gy * ts + header
        cx = px + ts // 2
        cy = py + ts // 2
        r  = ts * 5 // 12

        # Circle with border
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=color, outline="#FFFFFF", width=2)
        # Letter inside
        text_color = _contrast_text(color)
        bbox = draw.textbbox((0, 0), letter, font=self._font_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2 - 1), letter,
                  font=self._font_big, fill=text_color)
