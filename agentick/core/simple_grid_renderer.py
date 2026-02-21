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
    ObjectType.GOAL: ("G", "#00DD00"),
    ObjectType.KEY: ("K", "#FFD700"),
    ObjectType.DOOR: ("D", "#8B4513"),
    ObjectType.BOX: ("B", "#CD853F"),
    ObjectType.SWITCH: ("S", "#32CD32"),
    ObjectType.RESOURCE: ("R", "#00CED1"),
    ObjectType.TARGET: ("T", "#AAFFAA"),
    ObjectType.TOOL: ("T", "#FFA500"),
    ObjectType.BREADCRUMB: ("·", "#777777"),
    ObjectType.GEM: ("d", "#9400D3"),
    ObjectType.LEVER: ("L", "#B48C3C"),
    ObjectType.POTION: ("P", "#00B4B4"),
    ObjectType.SCROLL: ("?", "#D2B48C"),
    ObjectType.COIN: ("c", "#FFC800"),
    ObjectType.ORB: ("O", "#FF69B4"),
    ObjectType.NONE: (None, None),
}

# These are drawn as circles (like entities), not squares
CIRCLE_OBJECTS = {
    ObjectType.NPC: ("N", "#11AAEE"),
    ObjectType.ENEMY: ("E", "#EE1111"),
    ObjectType.SHEEP: ("S", "#FFFFFF"),
    ObjectType.BLOCKER: ("X", "#FF6600"),
}

TERRAIN_COLORS = {
    CellType.EMPTY: "#DCDCDC",
    CellType.WALL: "#3A3A3A",
    CellType.HAZARD: "#FF4500",
    CellType.WATER: "#1E90FF",
    CellType.ICE: "#B0E0E6",
    CellType.HOLE: "#111111",
}

# Special metadata values used by tasks for visual rendering
META_FOG = -1  # FogOfWar: cell hidden under fog (render as dark)
META_LIT = 1  # LightsOut: cell is lit (render as bright yellow)
META_LIGHT_POS = 2  # LightsOut: unlit light position (render as dark gray)
META_CAGE = 3  # InstructionFollowing: cage border cell

# Object-type metadata on TARGET/BOX cells — typed slot / tile rendering
# Metadata = ObjectType int → show that object's color/label
_META_OBJ_COLORS = {
    5: "#CD853F",  # BOX — tan
    14: "#9400D3",  # GEM — purple
    15: "#B48C3C",  # LEVER — brown
    16: "#00B4B4",  # POTION — teal
    17: "#D2B48C",  # SCROLL — parchment
    18: "#FFC800",  # COIN — gold
    19: "#FF69B4",  # ORB — pink
}
_META_OBJ_LABELS = {
    5: "B",  # BOX
    14: "d",  # GEM
    15: "L",  # LEVER
    16: "P",  # POTION
    17: "?",  # SCROLL
    18: "c",  # COIN
    19: "O",  # ORB
}

# GraphColoring node colors (metadata 1-4 on SWITCH objects = color assigned)
_META_GC_COLORS = {
    0: "#888888",  # uncolored → gray
    1: "#FF4444",  # color 1 → red
    2: "#4488FF",  # color 2 → blue
    3: "#44BB44",  # color 3 → green
    4: "#FFDD00",  # color 4 → yellow
}

# ResourceManagement energy stations (RESOURCE object, metadata = energy 0-100)
def _energy_color(level: int) -> str:
    """Return hex color for energy level 0-100."""
    if level >= 80:
        return "#00DD44"  # bright green
    elif level >= 60:
        return "#88DD00"  # yellow-green
    elif level >= 40:
        return "#FFDD00"  # yellow
    elif level >= 20:
        return "#FF8800"  # orange
    else:
        return "#FF2200"  # red / critical

ENTITY_COLORS = {
    "player": "#0044FF",
    "enemy": "#EE1111",
    "npc": "#11AAEE",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _contrast_text(bg_hex: str) -> str:
    """Return black or white for best contrast on bg."""
    r, g, b = _hex_to_rgb(bg_hex)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if lum > 140 else "#FFFFFF"


class SimpleGridRenderer:
    """Top-down 2D grid renderer with prominent, visually distinct tiles."""

    HEADER_H = 32  # pixels for task-name header (must be divisible by 16 for FFMPEG)

    def __init__(self, tile_size: int = 32):
        # Ensure tile_size divisible by 16 for FFMPEG compatibility
        tile_size = ((tile_size + 15) // 16) * 16
        self.tile_size = tile_size
        try:
            self._font_big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", tile_size // 2
            )
            self._font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
            )
        except Exception:
            self._font_big = ImageFont.load_default()
            self._font_small = ImageFont.load_default()

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
        step = info.get("step", "")
        hdr_text = f"{task_name}  step={step}" if step != "" else task_name
        draw.text((4, 3), hdr_text, font=self._font_small, fill=(220, 220, 100))

        # ── Terrain ───────────────────────────────────────────────────────
        for y in range(H):
            for x in range(W):
                px, py = x * ts, y * ts + header
                meta = int(grid.metadata[y, x])

                # Fog of war: hidden cells render as dark
                if meta == META_FOG:
                    draw.rectangle(
                        [px, py, px + ts - 1, py + ts - 1],
                        fill="#0A0A15",
                        outline="#111122",
                        width=1,
                    )
                    continue

                cell = (
                    CellType(grid.terrain[y, x])
                    if grid.terrain[y, x] in CellType._value2member_map_
                    else CellType.EMPTY
                )

                # LightsOut special rendering via metadata
                if meta == META_LIT:
                    bg = "#FFD700"  # bright yellow for lit cells
                elif meta == META_LIGHT_POS:
                    bg = "#2A2A3A"  # dark gray for unlit light positions
                elif meta == META_CAGE:
                    bg = "#4A2A10"  # brown cage border
                else:
                    bg = TERRAIN_COLORS.get(cell, TERRAIN_COLORS[CellType.EMPTY])

                # Phase-shift indicator (DistributionShift): terrain changes color
                obj_here = int(grid.objects[y, x])
                # (handled below in object rendering via metadata)

                draw.rectangle(
                    [px, py, px + ts - 1, py + ts - 1], fill=bg, outline="#555555", width=1
                )

                # ── Object on tile ────────────────────────────────────────
                obj_val = grid.objects[y, x]
                try:
                    obj = ObjectType(obj_val)
                except ValueError:
                    obj = ObjectType.NONE

                if obj == ObjectType.BREADCRUMB:
                    # Small dot
                    cx2 = px + ts // 2
                    cy2 = py + ts // 2
                    r2 = max(3, ts // 6)
                    draw.ellipse([cx2 - r2, cy2 - r2, cx2 + r2, cy2 + r2], fill="#888888")
                elif obj in CIRCLE_OBJECTS:
                    # Draw as entity circle
                    letter, color = CIRCLE_OBJECTS[obj]
                    self._draw_entity(draw, x, y, color, header, letter=letter)
                elif obj == ObjectType.SWITCH and meta in _META_GC_COLORS:
                    # GraphColoring: colored node — show with assigned color
                    node_color = _META_GC_COLORS[meta]
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill=node_color,
                        outline="#000000",
                        width=2,
                    )
                    label = str(meta) if meta > 0 else "N"
                    text_color = _contrast_text(node_color)
                    bbox = draw.textbbox((0, 0), label, font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        label, font=self._font_big, fill=text_color,
                    )
                elif obj == ObjectType.TARGET and meta in _META_OBJ_LABELS:
                    # Typed target slot: show expected piece type color + label
                    slot_color = _META_OBJ_COLORS[meta]
                    # Draw slot as translucent-ish background (slightly lighter)
                    m = max(2, ts // 8)
                    # Outer border = lighter shade, inner = object color
                    import colorsys
                    r2, g2, b2 = _hex_to_rgb(slot_color)
                    light = "#{:02X}{:02X}{:02X}".format(
                        min(255, r2 + 60), min(255, g2 + 60), min(255, b2 + 60)
                    )
                    draw.rectangle(
                        [px, py, px + ts - 1, py + ts - 1], fill=light, outline=slot_color, width=3
                    )
                    label = _META_OBJ_LABELS[meta]
                    text_color = _contrast_text(light)
                    bbox = draw.textbbox((0, 0), label, font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        label, font=self._font_big, fill=text_color,
                    )
                elif obj == ObjectType.BOX and meta != 0:
                    # Numbered tile (TileSorting) — show tile number instead of "B"
                    _, obj_color = OBJECT_LABELS[obj]
                    # Vary the shade slightly by tile number for more distinction
                    hue_shift = (meta * 37) % 60  # small hue variation
                    r2, g2, b2 = _hex_to_rgb(obj_color)
                    shifted = "#{:02X}{:02X}{:02X}".format(
                        min(255, r2 + hue_shift - 30),
                        min(255, g2 + hue_shift - 30),
                        b2,
                    )
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill=shifted, outline="#000000", width=2,
                    )
                    tile_label = str(meta) if meta <= 9 else chr(ord("A") + meta - 10)
                    text_color = _contrast_text(shifted)
                    bbox = draw.textbbox((0, 0), tile_label, font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        tile_label, font=self._font_big, fill=text_color,
                    )
                elif obj == ObjectType.RESOURCE and 1 <= meta <= 100:
                    # Energy station (ResourceManagement) — energy level shown as color
                    station_color = _energy_color(meta)
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill=station_color, outline="#000000", width=2,
                    )
                    energy_label = str(meta)
                    text_color = _contrast_text(station_color)
                    bbox = draw.textbbox((0, 0), energy_label, font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        energy_label, font=self._font_big, fill=text_color,
                    )
                elif obj != ObjectType.NONE and obj in OBJECT_LABELS:
                    label, obj_color = OBJECT_LABELS[obj]
                    if label is not None:
                        # Fill ~75% of tile with object color
                        m = max(2, ts // 8)
                        draw.rectangle(
                            [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                            fill=obj_color,
                            outline="#000000",
                            width=2,
                        )
                        # Draw label in center
                        text_color = _contrast_text(obj_color)
                        bbox = draw.textbbox((0, 0), label, font=self._font_big)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                        draw.text(
                            (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                            label,
                            font=self._font_big,
                            fill=text_color,
                        )

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
        gx: int,
        gy: int,
        color: str,
        header: int,
        letter: str = "A",
    ):
        ts = self.tile_size
        px = gx * ts
        py = gy * ts + header
        cx = px + ts // 2
        cy = py + ts // 2
        r = ts * 5 // 12

        # Circle with border
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="#FFFFFF", width=2)
        # Letter inside
        text_color = _contrast_text(color)
        bbox = draw.textbbox((0, 0), letter, font=self._font_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2 - 1), letter, font=self._font_big, fill=text_color)
