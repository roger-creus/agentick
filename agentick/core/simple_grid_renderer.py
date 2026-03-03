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

# GraphColoring / SwitchCircuit node colors (metadata 1-5 on SWITCH objects)
_META_GC_COLORS = {
    0: "#888888",  # uncolored → gray
    1: "#FF4444",  # color 1 → red
    2: "#4488FF",  # color 2 → blue
    3: "#44BB44",  # color 3 → green
    4: "#FFDD00",  # color 4 → yellow
    5: "#AA44FF",  # color 5 → purple
}

# EmergentStrategy NPC colors (NPC metadata 1,3-5)
_EMERGENT_NPC_COLORS = {
    1: "#11AAEE",  # Follower — cyan
    3: "#44CC44",  # Fearful — green
    4: "#AA44FF",  # Mirror — purple
    5: "#FFCC00",  # Contrarian — gold
}
_EMERGENT_NPC_LABELS = {
    1: "F",  # Follower
    3: "X",  # Fearful
    4: "M",  # Mirror
    5: "C",  # Contrarian
}

# SwitchCircuit colored walls (metadata 1-5 on WALL terrain = gate color)
_META_COLORED_WALL = {
    1: "#8B2222",  # red wall
    2: "#22448B",  # blue wall
    3: "#228B22",  # green wall
    4: "#8B8B00",  # yellow wall
    5: "#5522AA",  # purple wall
}

# KeyDoorPuzzle color-coded keys/doors (metadata 0-2)
_META_KEY_COLORS = {
    0: "#FFD700",  # gold key
    1: "#FF4444",  # red key
    2: "#4488FF",  # blue key
}
_META_DOOR_COLORS = {
    0: "#B8860B",  # gold door (dark goldenrod)
    1: "#8B2222",  # red door
    2: "#22448B",  # blue door
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

        # ── InstructionFollowing: show target type icon (bottom-right) ──
        task_config = info.get("task_config", {})
        if "target_type" in task_config and "InstructionFollowing" in task_name:
            target_obj = ObjectType(task_config["target_type"])
            label_entry = OBJECT_LABELS.get(target_obj)
            if label_entry and label_entry[0]:
                tgt_label, tgt_color = label_entry
                # Draw a prominent colored icon tile in the bottom-right corner
                icon_size = max(ts, 28)
                icon_x = W * ts - icon_size - 6
                icon_y = H * ts + header - icon_size - 6
                draw.rectangle(
                    [icon_x, icon_y, icon_x + icon_size - 1, icon_y + icon_size - 1],
                    fill=tgt_color, outline="#000000", width=2,
                )
                text_color = _contrast_text(tgt_color)
                bbox = draw.textbbox((0, 0), tgt_label, font=self._font_big)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                draw.text(
                    (icon_x + (icon_size - lw) // 2,
                     icon_y + (icon_size - lh) // 2 - 1),
                    tgt_label, font=self._font_big, fill=text_color,
                )

        # ── TaskInterference: show GEM/ORB meters ────────────────────────
        if "TaskInterference" in task_name:
            red = task_config.get("_red_meter", 0.0)
            blue = task_config.get("_blue_meter", 0.0)
            bar_w, bar_h = 60, 8
            bar_y = header - bar_h - 4
            # Gem meter
            draw.rectangle([4, bar_y, 4 + bar_w, bar_y + bar_h], outline=(180, 60, 60))
            red_x1 = max(5, 4 + int(red * (bar_w - 1)))
            draw.rectangle(
                [5, bar_y + 1, red_x1, bar_y + bar_h - 1],
                fill=(220, 50, 50),
            )
            draw.text((4 + bar_w + 4, bar_y - 1), f"GEM {red:.0%}",
                       font=self._font_small, fill=(220, 80, 80))
            # Orb meter
            bx = W * ts // 2
            draw.rectangle([bx, bar_y, bx + bar_w, bar_y + bar_h], outline=(60, 60, 180))
            blue_x1 = max(bx + 1, bx + int(blue * (bar_w - 1)))
            draw.rectangle(
                [bx + 1, bar_y + 1, blue_x1, bar_y + bar_h - 1],
                fill=(50, 50, 220),
            )
            draw.text((bx + bar_w + 4, bar_y - 1), f"ORB {blue:.0%}",
                       font=self._font_small, fill=(80, 80, 220))

        # ── TreasureHunt: show discovered clues ──────────────────────────
        if "TreasureHunt" in task_name:
            clue_info = task_config.get("_clue_info", {})
            clues_read = task_config.get("_clues_read", [])
            dir_labels = {0: "N", 1: "E", 2: "S", 3: "W"}
            clue_parts = []
            for cpos in clues_read:
                key = (
                    f"{cpos[0]},{cpos[1]}"
                    if isinstance(cpos, (list, tuple)) else cpos
                )
                ci = clue_info.get(key, clue_info.get(tuple(cpos), {}))
                if ci:
                    d = dir_labels.get(ci.get("direction", 0), "?")
                    dist = ci.get("distance", "?")
                    clue_parts.append(f"{d}{dist}")
            if clue_parts:
                # Wrap clues: max 5 per line to avoid running off-screen.
                # Stack rows upward from the bottom of the header area.
                max_per_line = 5
                n_rows = (len(clue_parts) + max_per_line - 1) // max_per_line
                base_y = header - 14
                for ri in range(n_rows):
                    start = ri * max_per_line
                    row = clue_parts[start:start + max_per_line]
                    clue_text = "Clue: " + "  ".join(row)
                    y_pos = base_y - (n_rows - 1 - ri) * 12
                    if y_pos < 0:
                        break
                    draw.text(
                        (4, y_pos), clue_text,
                        font=self._font_small, fill=(200, 200, 100),
                    )

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
                elif cell == CellType.WALL and meta in _META_COLORED_WALL:
                    bg = _META_COLORED_WALL[meta]  # SwitchCircuit colored gate wall
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
                elif (
                    obj == ObjectType.NPC
                    and meta in _EMERGENT_NPC_COLORS
                    and "EmergentStrategy" in task_name
                ):
                    # EmergentStrategy: color-coded NPC by behavior type
                    npc_color = _EMERGENT_NPC_COLORS[meta]
                    npc_letter = _EMERGENT_NPC_LABELS.get(meta, "N")
                    self._draw_entity(draw, x, y, npc_color, header, letter=npc_letter)
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
                elif obj == ObjectType.TARGET and meta >= 200:
                    # TileSorting: goal slot with tile number = meta - 200
                    slot_tile = meta - 200
                    tile_label = (
                        str(slot_tile) if slot_tile <= 9
                        else chr(ord("A") + slot_tile - 10)
                    )
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill="#AAFFAA", outline="#44AA44", width=2,
                    )
                    bbox = draw.textbbox(
                        (0, 0), tile_label, font=self._font_big
                    )
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        tile_label, font=self._font_big, fill="#226622",
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
                    # meta >= 100 means tile is in its correct position (green tint)
                    correct_pos = meta >= 100
                    real_tile = meta - 100 if correct_pos else meta
                    if correct_pos:
                        # Green tint for correctly placed tiles
                        hue_shift = (real_tile * 37) % 40
                        shifted = "#{:02X}{:02X}{:02X}".format(
                            40 + hue_shift,
                            min(255, 180 + hue_shift),
                            40 + hue_shift,
                        )
                    else:
                        _, obj_color = OBJECT_LABELS[obj]
                        # Vary the shade slightly by tile number for more distinction
                        hue_shift = (real_tile * 37) % 60  # small hue variation
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
                    tile_label = (
                        str(real_tile) if real_tile <= 9
                        else chr(ord("A") + real_tile - 10)
                    )
                    text_color = _contrast_text(shifted)
                    bbox = draw.textbbox((0, 0), tile_label, font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        tile_label, font=self._font_big, fill=text_color,
                    )
                elif obj == ObjectType.SCROLL and meta > 0:
                    # TreasureHunt directional scroll clue.
                    # Metadata encoding: direction * 10 + distance
                    # direction: 0=N, 1=E, 2=S, 3=W
                    # distance: 1-9 (Manhattan distance to nearest treasure, capped)
                    scroll_dir = meta // 10
                    scroll_dist = meta % 10
                    arrow_chars = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}
                    arrow = arrow_chars.get(scroll_dir, "?")

                    # Color intensity based on distance: closer = brighter green
                    if scroll_dist <= 1:
                        arrow_bg = "#00EE00"  # bright green (very close)
                    elif scroll_dist <= 2:
                        arrow_bg = "#00CC00"
                    elif scroll_dist <= 3:
                        arrow_bg = "#00AA00"
                    elif scroll_dist <= 4:
                        arrow_bg = "#008800"
                    else:
                        arrow_bg = "#006600"  # dim green (far)

                    # Draw parchment background with distance-colored inner rect
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill=arrow_bg,
                        outline="#000000",
                        width=2,
                    )
                    # Draw the arrow character
                    text_color = _contrast_text(arrow_bg)
                    bbox = draw.textbbox((0, 0), arrow, font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        arrow,
                        font=self._font_big,
                        fill=text_color,
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
                elif obj == ObjectType.KEY and meta in _META_KEY_COLORS:
                    key_color = _META_KEY_COLORS[meta]
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill=key_color, outline="#000000", width=2,
                    )
                    text_color = _contrast_text(key_color)
                    bbox = draw.textbbox((0, 0), "K", font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        "K", font=self._font_big, fill=text_color,
                    )
                elif obj == ObjectType.DOOR and meta in _META_DOOR_COLORS:
                    door_color = _META_DOOR_COLORS[meta]
                    m = max(2, ts // 8)
                    draw.rectangle(
                        [px + m, py + m, px + ts - m - 1, py + ts - m - 1],
                        fill=door_color, outline="#000000", width=2,
                    )
                    text_color = _contrast_text(door_color)
                    bbox = draw.textbbox((0, 0), "D", font=self._font_big)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text(
                        (px + (ts - tw) // 2, py + (ts - th) // 2 - 1),
                        "D", font=self._font_big, fill=text_color,
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

        # Resize to fixed 512x512 for consistent observation space
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.LANCZOS)
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
