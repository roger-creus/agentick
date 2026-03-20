"""Isometric 2D sprite renderer for Agentick environments.

Produces beautiful isometric renders using Kenney Isometric Blocks sprites
via pure Pillow 2D compositing. No 3D engine needed.

Renders rich visual information:
- Metadata-driven visuals (fog, lit/unlit, energy levels, colored nodes, etc.)
- Text labels on tiles for numbered tiles, energy levels, graph coloring, etc.
- Entity elevation: objects that sit on the floor are drawn one cube height above
- Alpha occlusion: elevated tiles in front of the agent are semi-transparent
- Direction arrows: labeled UP/DOWN/LEFT/RIGHT outside the diamond map
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentick.core.entity import Agent, Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, Direction, ObjectType
from agentick.rendering.iso_math import (
    calculate_canvas_size,
    calculate_offset,
    grid_to_screen,
)
from agentick.rendering.tile_atlas import TileAtlas

# --------------------------------------------------------------------------- #
# Metadata constants for task-specific rendering
# --------------------------------------------------------------------------- #
META_FOG = -1       # FogOfWar: cell hidden under fog
META_LIT = 1        # LightsOut: cell is lit (bright yellow)
META_LIGHT_POS = 2  # LightsOut: unlit light position (dark gray)
META_CAGE = 3       # InstructionFollowing: cage border cell

# GraphColoring node colors (SWITCH metadata 0-4)
_GC_TINTS: dict[int, tuple[float, float, float]] = {
    0: (0.6, 0.6, 0.6),   # uncolored -> gray
    1: (1.0, 0.27, 0.27),  # red
    2: (0.27, 0.53, 1.0),  # blue
    3: (0.27, 0.73, 0.27),  # green
    4: (1.0, 0.87, 0.0),   # yellow
}

# EmergentStrategy NPC colors by metadata (1,3-5)
_EMERGENT_NPC_TINTS: dict[int, tuple[float, float, float]] = {
    1: (0.07, 0.67, 0.93),  # Follower — cyan
    3: (0.27, 0.80, 0.27),  # Fearful — green
    4: (0.67, 0.27, 1.0),   # Mirror — purple
    5: (1.0, 0.80, 0.0),    # Contrarian — gold
}
_EMERGENT_NPC_LABELS: dict[int, str] = {
    1: "F", 3: "X", 4: "M", 5: "C",
}

# Door color names by metadata value (KeyDoorPuzzle color-coding)
_DOOR_COLOR_NAMES: dict[int, str] = {0: "golden", 1: "red", 2: "blue"}

# Direction enum -> directional sprite suffix
_DIRECTION_SUFFIX: dict[int, str] = {0: "up", 1: "right", 2: "down", 3: "left"}

# Ghost tile mapping: TARGET metadata (ObjectType int) -> tile name to ghost
_GHOST_TILE_MAP: dict[int, str] = {
    5: "box",      # BOX
    14: "gem",     # GEM
    15: "lever",   # LEVER
    16: "potion",  # POTION
    17: "scroll",  # SCROLL
    18: "coin",    # COIN
    19: "orb",     # ORB
}

# --------------------------------------------------------------------------- #
# Elevation: which entities sit ON TOP of floor (elevated by one cube height)
# --------------------------------------------------------------------------- #
_ELEVATED_ENTITIES: set[str] = {
    "agent", "agent_up", "agent_down", "agent_left", "agent_right",
    "goal", "key", "golden_key", "red_key", "blue_key",
    "door", "door_open",
    "golden_door_ud", "golden_door_rl", "golden_door_open_ud", "golden_door_open_rl",
    "red_door_ud", "red_door_rl", "red_door_open_ud", "red_door_open_rl",
    "blue_door_ud", "blue_door_rl", "blue_door_open_ud", "blue_door_open_rl",
    "box", "enemy", "enemy_up", "enemy_down", "enemy_left", "enemy_right",
    "switch", "switch_on", "switch_off",
    "npc", "npc_up", "npc_down", "npc_left", "npc_right",
    "tool", "resource",
    "sheep", "sheep_up", "sheep_down", "sheep_left", "sheep_right",
    "blocker", "gem", "lever", "lever_on", "lever_off",
    "potion", "scroll", "coin", "orb", "breadcrumb",
}

# Extra floating offset for pickups (on top of elevation)
_FLOAT_EXTRA: dict[str, int] = {
    "key": 4, "gem": 5, "coin": 3, "orb": 6, "potion": 3, "heart": 5,
}

# Ground-level entities (NOT elevated — drawn at floor level)
_GROUND_LEVEL: set[str] = {
    "floor", "wall", "border_wall", "hazard", "water", "ice", "hole",
    "fog", "fog_partial", "target", "empty",
}



class IsometricRenderer:
    """Renders game state as isometric sprite image using Kenney tiles.

    Pure 2D compositing with Pillow. No 3D engine.
    Produces (H, W, 3) uint8 numpy arrays compatible with Gymnasium.

    Features:
    - Entity elevation: objects/agent sit ON TOP of floor tiles
    - Wall elevation: walls drawn one cube height above floor
    - Alpha occlusion: elevated tiles in front of agent become semi-transparent
    - Direction arrows: labeled UP/DOWN/LEFT/RIGHT at diamond edges
    - Metadata-driven rendering: fog, lit/unlit, energy, colored nodes, labels
    - Ghost tiles: target slots show faded version of the expected object
    - Comprehensive visual information for all tasks
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (512, 512),
        tiles_dir: str | Path | None = None,
        background_color: tuple[int, int, int] = (40, 44, 52),
        show_hud: bool = True,
        hud_height: int = 40,
    ):
        self.output_size = output_size
        self.tiles_dir = Path(tiles_dir) if tiles_dir else None
        self.background_color = background_color
        self.show_hud = show_hud
        self.hud_height = hud_height if show_hud else 0

        # Atlas is lazily initialized on first render to auto-scale for grid size
        self._atlas: TileAtlas | None = None
        self._last_grid_shape: tuple[int, int] | None = None

        # Font for tile labels
        self._label_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

    def _get_label_font(self, size: int = 12) -> Any:
        """Get or create a font for tile labels."""
        if self._label_font is None:
            try:
                self._label_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
                )
            except (OSError, IOError):
                self._label_font = ImageFont.load_default()
        return self._label_font

    def _ensure_atlas(self, rows: int, cols: int) -> TileAtlas:
        """Initialize or re-scale atlas to fit the grid in output_size."""
        grid_shape = (rows, cols)
        if self._atlas is not None and self._last_grid_shape == grid_shape:
            return self._atlas

        out_w, out_h = self.output_size
        available_h = out_h - self.hud_height

        from agentick.rendering.tile_atlas import BASE_TILE_W, DIAMOND_H, TILE_DEPTH

        # Raw canvas size at scale=1.0 — include extra cube_height for elevation
        raw_w = (rows + cols) * (BASE_TILE_W // 2)
        raw_h = (rows + cols) * (DIAMOND_H // 2) + TILE_DEPTH * 2  # extra for elevation

        scale_x = out_w / raw_w if raw_w > 0 else 1.0
        scale_y = available_h / raw_h if raw_h > 0 else 1.0
        tile_scale = min(scale_x, scale_y) * 0.90

        tile_scale = max(0.05, min(tile_scale, 1.5))

        if self._atlas is None:
            self._atlas = TileAtlas(
                tiles_dir=self.tiles_dir, tile_scale=tile_scale
            )
        else:
            self._atlas.rescale(tile_scale)

        # Reset label font when scale changes
        label_size = max(8, int(12 * tile_scale))
        try:
            self._label_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", label_size
            )
        except (OSError, IOError):
            self._label_font = ImageFont.load_default()

        self._last_grid_shape = grid_shape
        return self._atlas

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Render current game state as isometric image.

        Args:
            grid: The game Grid object
            entities: List of entities on the grid
            agent: The Agent object
            info: Dict with step, reward, etc. for HUD

        Returns:
            (height, width, 3) uint8 numpy array
        """
        rows, cols = grid.height, grid.width
        atlas = self._ensure_atlas(rows, cols)

        tile_w = atlas.tile_width
        tile_h = atlas.tile_height  # diamond height
        tile_d = atlas.tile_depth   # cube side height = elevation offset
        cube_h = tile_d             # elevation for entities sitting on floor

        # Calculate canvas dimensions — extra room for elevated entities at top
        canvas_w, canvas_h = calculate_canvas_size(
            rows, cols, tile_w, tile_h, tile_d, self.hud_height + cube_h
        )
        canvas_w = max(canvas_w, 1)
        canvas_h = max(canvas_h, 1)

        canvas = Image.new(
            "RGBA", (canvas_w, canvas_h), (*self.background_color, 255)
        )

        # Store task name for _draw_object to access
        self._render_task_name = info.get("task_name", "")

        # Offsets — extra cube_h padding at top for elevated entities
        offset_x = calculate_offset(rows, tile_w)
        offset_y = self.hud_height + cube_h

        # Build entity lookup (x, y) -> entity
        entity_map: dict[tuple[int, int], Entity] = {}
        for ent in entities:
            entity_map[ent.position] = ent

        ax, ay = agent.position

        # Track cage positions for background target annotation (FIX 7)
        cage_goal_pos: tuple[int, int] | None = None

        # Painter's order: back to front
        for row in range(rows):
            for col in range(cols):
                sx, sy = grid_to_screen(row, col, tile_w, tile_h)
                draw_x = sx + offset_x
                draw_y = sy + offset_y

                meta = int(grid.metadata[row, col])

                # --- Fog of war (metadata == -1) ---
                if meta == META_FOG:
                    fog_tile = atlas.get_tile("fog")
                    self._safe_paste(canvas, fog_tile, draw_x, draw_y)
                    continue

                # --- Draw floor / terrain tile ---
                terrain_val = int(grid.terrain[row, col])
                terrain_name = atlas._int_to_name(terrain_val)

                # InstructionFollowing special terrain via metadata
                if meta == META_CAGE:
                    # Cage border — use cage tile (InstructionFollowing)
                    floor_tile = atlas.get_tile("cage")
                elif terrain_name in ("water", "ice", "hazard", "hole"):
                    floor_tile = atlas.get_tile(terrain_name)
                else:
                    floor_tile = atlas.get_tile("floor")

                # Floor is always at ground level, no alpha occlusion
                self._safe_paste(canvas, floor_tile, draw_x, draw_y)

                # --- Walls ---
                if terrain_name == "wall":
                    is_border = (
                        row == 0 or row == rows - 1
                        or col == 0 or col == cols - 1
                    )
                    if is_border:
                        # Border walls: flat gray solid block (no elevation)
                        border_tile = atlas.get_tile("border_wall")
                        self._safe_paste(canvas, border_tile, draw_x, draw_y)
                    else:
                        wall_tile = atlas.get_tile("wall")
                        wall_lift = int(cube_h * 0.2)
                        self._safe_paste(
                            canvas, wall_tile, draw_x, draw_y - wall_lift,
                        )

                # --- Track cage goal for background annotation (FIX 7) ---
                obj_val = int(grid.objects[row, col])
                if meta == META_CAGE and obj_val == ObjectType.LEVER:
                    cage_goal_pos = (col, row)
                    # Skip rendering this GOAL in the grid — show in background
                    continue

                # --- Draw objects (with elevation + metadata) ---
                if obj_val != ObjectType.NONE:
                    self._draw_object(
                        canvas, atlas, obj_val, meta, draw_x, draw_y,
                        cube_h, grid, col, row,
                    )

                # --- Draw entities (NPCs, enemies, sheep from entity list) ---
                ent = entity_map.get((col, row))
                if ent is not None:
                    ent_name = ent.entity_type
                    # Use directional sprite if entity has orientation or metadata
                    if ent_name in ("npc", "enemy", "sheep"):
                        ent_dir = getattr(ent, "orientation", None)
                        if ent_dir is not None:
                            suffix = _DIRECTION_SUFFIX.get(int(ent_dir), "down")
                        else:
                            # Fall back to metadata direction
                            suffix = _DIRECTION_SUFFIX.get(meta, "down")
                        ent_tile = atlas.get_tile(f"{ent_name}_{suffix}")
                    else:
                        ent_tile = atlas.get_tile(ent_name)
                    elev = cube_h if ent_name in _ELEVATED_ENTITIES else 0
                    self._safe_paste(canvas, ent_tile, draw_x, draw_y - elev)

                # --- Draw agent (always elevated, never alpha-reduced) ---
                if col == ax and row == ay:
                    dir_suffix = _DIRECTION_SUFFIX.get(
                        int(agent.orientation), "down"
                    )
                    agent_tile = atlas.get_tile(f"agent_{dir_suffix}")
                    self._safe_paste(canvas, agent_tile, draw_x, draw_y - cube_h)

        # --- Draw direction arrows on canvas border (FIX 2) ---
        self._draw_direction_arrows(
            canvas, rows, cols, tile_w, tile_h, offset_x, offset_y
        )

        # --- Draw instruction target in background (FIX 7) ---
        if cage_goal_pos is not None:
            self._draw_instruction_target(canvas, atlas, cube_h)

        # --- Draw HUD ---
        if self.show_hud and info:
            self._draw_hud(canvas, info)

        # --- Resize to output_size ---
        canvas_rgb = canvas.convert("RGB")
        canvas_rgb = canvas_rgb.resize(self.output_size, Image.LANCZOS)

        return np.array(canvas_rgb, dtype=np.uint8)

    def _draw_object(
        self,
        canvas: Image.Image,
        atlas: TileAtlas,
        obj_val: int,
        meta: int,
        draw_x: int,
        draw_y: int,
        cube_h: int,
        grid: Grid | None = None,
        col: int = 0,
        row: int = 0,
    ) -> None:
        """Draw a grid object with metadata-driven visuals and elevation."""
        obj_name = atlas._obj_int_to_name(obj_val)
        is_elevated = obj_name in _ELEVATED_ENTITIES
        elev = cube_h if is_elevated else 0
        float_extra = int(_FLOAT_EXTRA.get(obj_name, 0) * atlas._tile_scale)

        # --- SWITCH rendering (no color tinting — use sprites as-is) ---
        if obj_val == ObjectType.SWITCH:
            # GraphColoring: meta 0-4 (0=uncolored, 1-4=colors)
            # LightsOut: meta 1 = on (lit), meta 2 = off (unlit position)
            # SwitchCircuit: meta >= 100 = on, meta < 100 = off
            # FewShotAdaptation: meta 0 = default
            if meta >= 100 or meta == 1:
                tile_name = "switch_on"
            else:
                tile_name = "switch_off"
            tile = atlas.get_tile(tile_name)
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            # GraphColoring label (meta 0-4)
            if 0 <= meta <= 4:
                self._draw_tile_label(
                    canvas, str(meta), draw_x, draw_y - elev, atlas,
                )
            return

        # --- DOOR (color-coded + orientation-aware, supports open state) ---
        if obj_val == ObjectType.DOOR:
            orientation = (
                self._detect_door_orientation(grid, col, row) if grid is not None
                else "ud"
            )
            if meta >= 10:
                # Open door: meta = color + 10
                color = _DOOR_COLOR_NAMES.get(meta - 10, "golden")
                tile_name = f"{color}_door_open_{orientation}"
            else:
                color = _DOOR_COLOR_NAMES.get(meta, "golden")
                tile_name = f"{color}_door_{orientation}"
            tile = atlas.get_tile(tile_name)
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            return

        # --- KEY (color-coded) ---
        if obj_val == ObjectType.KEY:
            color = _DOOR_COLOR_NAMES.get(meta, "golden")
            tile = atlas.get_tile(f"{color}_key")
            self._safe_paste(
                canvas, tile, draw_x, draw_y - elev - float_extra
            )
            return

        # --- LEVER on/off ---
        if obj_val == ObjectType.LEVER:
            tile_name = "lever_on" if meta > 0 else "lever_off"
            tile = atlas.get_tile(tile_name)
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            return

        # --- EmergentStrategy NPC (metadata 1-5 = behavior type) ---
        task_name = getattr(self, "_render_task_name", "")
        if (
            obj_val == ObjectType.NPC
            and meta in _EMERGENT_NPC_TINTS
            and "EmergentStrategy" in task_name
        ):
            tile = atlas.get_tile("npc_down")
            # Tint the tile to the behavior color
            tint = _EMERGENT_NPC_TINTS[meta]
            tile = self._tint_tile(tile, tint)
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            label = _EMERGENT_NPC_LABELS.get(meta, "")
            if label:
                self._draw_tile_label(canvas, label, draw_x, draw_y - elev, atlas)
            return

        # --- NPC / ENEMY / SHEEP (directional from metadata) ---
        if obj_val in (ObjectType.NPC, ObjectType.ENEMY, ObjectType.SHEEP):
            suffix = _DIRECTION_SUFFIX.get(meta, "down")
            tile = atlas.get_tile(f"{obj_name}_{suffix}")
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            return

        # --- TARGET: TileSorting goal slot (meta >= 200) ---
        if obj_val == ObjectType.TARGET and meta >= 200:
            slot_tile = meta - 200
            label = (
                str(slot_tile) if slot_tile <= 9
                else chr(ord("A") + slot_tile - 10)
            )
            tile = atlas.get_tile("target")
            tile = self._reduce_alpha(tile, 120)
            self._safe_paste(canvas, tile, draw_x, draw_y)
            self._draw_tile_label(canvas, label, draw_x, draw_y, atlas)
            return

        # --- TARGET with ghost tiles (typed slots — PackingPuzzle, etc.) ---
        if obj_val == ObjectType.TARGET and meta in _GHOST_TILE_MAP:
            ghost_name = _GHOST_TILE_MAP[meta]
            ghost_tile = atlas.get_tile(ghost_name)
            # Transparent ghost at elevated level (on-floor, not buried)
            ghost_tile = self._reduce_alpha(ghost_tile, 100)  # ~40%
            self._safe_paste(canvas, ghost_tile, draw_x, draw_y - cube_h)
            return

        # --- TARGET (default rendering — transparent ghost block) ---
        if obj_val == ObjectType.TARGET:
            # If terrain is HOLE, the hole_block tile IS the target visual
            if grid is not None and int(grid.terrain[row, col]) == int(CellType.HOLE):
                return  # hole_block already rendered as floor
            tile = atlas.get_tile("target")
            tile = self._reduce_alpha(tile, 120)  # ~47% ghost
            self._safe_paste(canvas, tile, draw_x, draw_y)  # at floor level
            return

        # --- BOX with numbered tile (TileSorting, meta != 0) ---
        if obj_val == ObjectType.BOX and meta != 0:
            tile = atlas.get_tile("box")
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            real_tile = meta - 100 if meta >= 100 else meta
            label = (
                str(real_tile) if real_tile <= 9
                else chr(ord("A") + real_tile - 10)
            )
            self._draw_tile_label(canvas, label, draw_x, draw_y - elev, atlas)
            return

        # --- RESOURCE with energy level (1-100) ---
        if obj_val == ObjectType.RESOURCE and 1 <= meta <= 100:
            tile = atlas.get_tile("resource")
            self._safe_paste(canvas, tile, draw_x, draw_y - elev)
            self._draw_tile_label(canvas, str(meta), draw_x, draw_y - elev, atlas)
            return

        # --- Default object rendering ---
        obj_tile = atlas.get_tile(obj_name)
        self._safe_paste(
            canvas, obj_tile, draw_x, draw_y - elev - float_extra
        )

    @staticmethod
    def _detect_door_orientation(grid: Grid, col: int, row: int) -> str:
        """Detect door orientation from adjacent walls.

        Returns "ud" (up-down passage, walls east+west) or "rl" (right-left
        passage, walls north+south).
        """
        rows, cols = grid.height, grid.width
        wall_east = (
            col + 1 < cols
            and int(grid.terrain[row, col + 1]) == int(CellType.WALL)
        )
        wall_west = (
            col - 1 >= 0
            and int(grid.terrain[row, col - 1]) == int(CellType.WALL)
        )
        wall_north = (
            row - 1 >= 0
            and int(grid.terrain[row - 1, col]) == int(CellType.WALL)
        )
        wall_south = (
            row + 1 < rows
            and int(grid.terrain[row + 1, col]) == int(CellType.WALL)
        )

        if wall_north and wall_south:
            return "rl"
        if wall_east and wall_west:
            return "ud"
        # Default: up-down passage
        return "ud"

    def _draw_tile_label(
        self,
        canvas: Image.Image,
        label: str,
        tile_x: int,
        tile_y: int,
        atlas: TileAtlas,
    ) -> None:
        """Draw a text label centered on a tile (for numbered tiles, energy, etc.)."""
        draw = ImageDraw.Draw(canvas)
        font = self._get_label_font()

        # Center the label on the top face of the isometric tile
        tw = atlas.tile_width
        th = atlas._scaled_h

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        cx = tile_x + tw // 2 - text_w // 2
        cy = tile_y + int(th * 0.3) - text_h // 2

        # Draw outline for readability
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((cx + dx, cy + dy), label, fill=(0, 0, 0, 200), font=font)
        draw.text((cx, cy), label, fill=(255, 255, 255, 255), font=font)

    def _draw_direction_arrows(
        self,
        canvas: Image.Image,
        rows: int,
        cols: int,
        tile_w: int,
        tile_h: int,
        offset_x: int,
        offset_y: int,
    ) -> None:
        """Draw labeled direction arrows outside the diamond map.

        In the isometric projection:
        - MOVE_UP (row-1) -> upper-right on screen
        - MOVE_DOWN (row+1) -> lower-left on screen
        - MOVE_LEFT (col-1) -> upper-left on screen
        - MOVE_RIGHT (col+1) -> lower-right on screen
        """
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                max(10, int(14 * (self._atlas._tile_scale if self._atlas else 1.0))),
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Compute diamond corner positions (center of each corner tile)
        half_tw = tile_w // 2
        half_th = tile_h // 2

        def screen_center(r: int, c: int) -> tuple[int, int]:
            sx, sy = grid_to_screen(r, c, tile_w, tile_h)
            return sx + offset_x + half_tw, sy + offset_y + half_th

        top = screen_center(0, 0)
        right = screen_center(0, cols - 1)
        bottom = screen_center(rows - 1, cols - 1)
        left = screen_center(rows - 1, 0)

        # Edge midpoints (arrows placed at midpoints, pushed well into background)
        margin = max(20, int(85 * (self._atlas._tile_scale if self._atlas else 1.0)))

        # UP arrow: top-right edge midpoint (between top and right corners)
        up_x = (top[0] + right[0]) // 2 + margin
        up_y = (top[1] + right[1]) // 2 - margin

        # DOWN arrow: bottom-left edge midpoint (between left and bottom corners)
        # Extra offset so the label doesn't crowd the bottom-left map edge
        down_margin = int(margin * 1.5)
        down_x = (left[0] + bottom[0]) // 2 - down_margin
        down_y = (left[1] + bottom[1]) // 2 + down_margin

        # LEFT arrow: top-left edge midpoint (between top and left corners)
        left_x = (top[0] + left[0]) // 2 - margin
        left_y = (top[1] + left[1]) // 2 - margin

        # RIGHT arrow: bottom-right edge midpoint (between right and bottom corners)
        right_x = (right[0] + bottom[0]) // 2 + margin
        right_y = (right[1] + bottom[1]) // 2 + margin

        labels = [
            (up_x, up_y, "UP"),
            (down_x, down_y, "DOWN"),
            (left_x, left_y, "LEFT"),
            (right_x, right_y, "RIGHT"),
        ]

        for lx, ly, text in labels:
            # Clamp to canvas bounds
            lx = max(2, min(lx, canvas.size[0] - 60))
            ly = max(2, min(ly, canvas.size[1] - 16))
            # Draw outline (dark background for readability)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            (lx + dx, ly + dy), text,
                            fill=(0, 0, 0, 200), font=font,
                        )
            draw.text((lx, ly), text, fill=(255, 255, 255, 220), font=font)

    def _draw_instruction_target(
        self,
        canvas: Image.Image,
        atlas: TileAtlas,
        cube_h: int,
    ) -> None:
        """Draw the InstructionFollowing target indicator in the background canvas.

        Renders a 'Target:' label with the goal tile in the top-right area
        of the canvas, outside the diamond map.
        """
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                max(10, int(13 * (atlas._tile_scale if atlas else 1.0))),
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Position: top-right corner with generous spacing
        # Label on top, tile below — stacked vertically for clarity
        bbox = draw.textbbox((0, 0), "Target:", font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Scale down tile for annotation
        small_w = max(1, atlas.tile_width * 2 // 3)
        small_h = max(1, atlas._scaled_h * 2 // 3)

        # Right-align the block: need space for max(text_w, small_w) + padding
        block_w = max(text_w, small_w)
        label_x = canvas.size[0] - block_w - 12
        label_y = self.hud_height + 4

        # Clamp to canvas
        label_x = max(4, label_x)

        # Draw "Target:" label
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text(
                        (label_x + dx, label_y + dy), "Target:",
                        fill=(0, 0, 0, 200), font=font,
                    )
        draw.text((label_x, label_y), "Target:", fill=(255, 255, 255, 230), font=font)

        # Draw the lever tile below the label, centered
        lever_tile = atlas.get_tile("lever_on")
        small_tile = lever_tile.resize((small_w, small_h), Image.LANCZOS)
        tile_x = label_x + (text_w - small_w) // 2
        tile_y = label_y + text_h + 4
        self._safe_paste(canvas, small_tile, tile_x, tile_y)

    @staticmethod
    def _tint_tile(
        tile: Image.Image, tint: tuple[float, float, float]
    ) -> Image.Image:
        """Apply a color tint to a tile (multiply RGB channels by tint factors)."""
        tile = tile.copy()
        r, g, b, a = tile.split()
        r = r.point(lambda p: min(255, int(p * tint[0])))
        g = g.point(lambda p: min(255, int(p * tint[1])))
        b = b.point(lambda p: min(255, int(p * tint[2])))
        return Image.merge("RGBA", (r, g, b, a))

    @staticmethod
    def _reduce_alpha(tile: Image.Image, alpha_val: int) -> Image.Image:
        """Return a copy of tile with alpha channel capped at alpha_val.

        Used for alpha occlusion (semi-transparent tiles in front of agent)
        and ghost tile rendering.
        """
        tile = tile.copy()
        r, g, b, a = tile.split()
        a = a.point(lambda p: min(p, alpha_val))
        return Image.merge("RGBA", (r, g, b, a))

    @staticmethod
    def _safe_paste(
        canvas: Image.Image,
        tile: Image.Image,
        x: int,
        y: int,
    ) -> None:
        """Paste tile onto canvas handling out-of-bounds gracefully."""
        try:
            canvas.paste(tile, (x, y), tile)
        except ValueError:
            cw, ch = canvas.size
            tw, th = tile.size
            src_x = max(0, -x)
            src_y = max(0, -y)
            dst_x = max(0, x)
            dst_y = max(0, y)
            vis_w = min(tw - src_x, cw - dst_x)
            vis_h = min(th - src_y, ch - dst_y)
            if vis_w > 0 and vis_h > 0:
                cropped = tile.crop((src_x, src_y, src_x + vis_w, src_y + vis_h))
                canvas.paste(cropped, (dst_x, dst_y), cropped)

    def _draw_hud(self, canvas: Image.Image, info: dict[str, Any]) -> None:
        """Draw step counter and reward at top of image."""
        draw = ImageDraw.Draw(canvas)
        step = info.get("step", info.get("step_count", 0))
        max_step = info.get("max_steps", "?")
        reward = info.get("episode_reward", info.get("total_reward", 0.0))
        task_name = info.get("task_name", "")

        if task_name:
            text = f"{task_name}  Step: {step}/{max_step}  Reward: {reward:.2f}"
        else:
            text = f"Step: {step}/{max_step}  Reward: {reward:.2f}"

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw.text((8, 8), text, fill=(255, 255, 255, 255), font=font)

        # ── Task-specific HUD elements ──────────────────────────────────
        task_config = info.get("task_config", {})
        task_name_str = info.get("task_name", "")

        # TaskInterference meters (GEM/ORB)
        if "TaskInterference" in task_name_str:
            red = task_config.get("_red_meter", 0.0)
            blue = task_config.get("_blue_meter", 0.0)
            bar_w, bar_h, bar_y = 80, 10, 28
            draw.rectangle([8, bar_y, 8 + bar_w, bar_y + bar_h],
                           outline=(180, 60, 60, 255))
            red_x1 = max(9, 8 + int(red * (bar_w - 1)))
            draw.rectangle(
                [9, bar_y + 1, red_x1, bar_y + bar_h - 1],
                fill=(220, 50, 50, 255),
            )
            draw.text((8 + bar_w + 6, bar_y - 2), f"GEM {red:.0%}",
                      fill=(220, 80, 80, 255), font=font)
            bx = canvas.width // 2
            draw.rectangle([bx, bar_y, bx + bar_w, bar_y + bar_h],
                           outline=(60, 60, 180, 255))
            blue_x1 = max(bx + 1, bx + int(blue * (bar_w - 1)))
            draw.rectangle(
                [bx + 1, bar_y + 1, blue_x1, bar_y + bar_h - 1],
                fill=(50, 50, 220, 255),
            )
            draw.text((bx + bar_w + 6, bar_y - 2), f"ORB {blue:.0%}",
                      fill=(80, 80, 220, 255), font=font)

        # InstructionFollowing target indicator (bottom-right — show sprite)
        if "InstructionFollowing" in task_name_str and "target_type" in task_config:
            obj_sprite_names = {14: "gem", 17: "scroll", 19: "orb", 18: "coin"}
            sprite_name = obj_sprite_names.get(
                int(task_config["target_type"]), "goal"
            )
            if self._atlas is not None:
                target_tile = self._atlas.get_tile(sprite_name)
                # Scale sprite to 48x48 for the annotation
                sprite_size = 48
                small_tile = target_tile.resize(
                    (sprite_size, sprite_size), Image.LANCZOS
                )
                sprite_x = canvas.width - sprite_size - 8
                sprite_y = canvas.height - sprite_size - 8
                self._safe_paste(canvas, small_tile, sprite_x, sprite_y)

        # RuleInduction trial + target indicator
        if "RuleInduction" in task_name_str and "_target_type" in task_config:
            trial = task_config.get("_current_trial", 0) + 1
            n_trials = task_config.get("_n_trials", 1)
            trial_text = f"Trial: {trial}/{n_trials}"
            draw.text((10, canvas.height - 30), trial_text,
                      fill=(255, 200, 50, 255), font=font)
            obj_sprite_names = {14: "gem", 15: "potion", 17: "scroll", 18: "coin", 19: "orb"}
            sprite_name = obj_sprite_names.get(
                int(task_config["_target_type"]), "goal"
            )
            if self._atlas is not None:
                target_tile = self._atlas.get_tile(sprite_name)
                sprite_size = 48
                small_tile = target_tile.resize(
                    (sprite_size, sprite_size), Image.LANCZOS
                )
                tx = canvas.width - sprite_size - 10
                ty = canvas.height - sprite_size - 10
                canvas.paste(small_tile, (tx, ty), small_tile)
                draw.text((tx - 60, ty + 15), "Target:",
                          fill=(255, 255, 255, 200), font=font)

        # DistributionShift phase indicator
        if "DistributionShift" in task_name_str:
            phase = task_config.get("_phases_completed", 0) + 1
            n_phases = task_config.get("_n_phases", 3)
            phase_type = task_config.get("_current_phase_type", "goal_reach")
            phase_type_names = {
                "goal_reach": "Navigate", "key_door": "Key+Door",
                "lever_barrier": "Lever", "collection": "Collect",
                "box_push": "BoxPush",
            }
            type_label = phase_type_names.get(phase_type, phase_type)
            phase_text = f"Phase: {phase}/{n_phases}  [{type_label}]"
            draw.text((10, canvas.height - 30), phase_text,
                      fill=(100, 200, 255, 255), font=font)

        # TreasureHunt: show discovered clues
        if "TreasureHunt" in task_name_str:
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
                max_per_line = 5
                base_y = 28 + 16
                for ri in range(0, len(clue_parts), max_per_line):
                    row = clue_parts[ri:ri + max_per_line]
                    clue_text = "Clue: " + "  ".join(row)
                    y_pos = base_y + (ri // max_per_line) * 14
                    draw.text((8, y_pos), clue_text,
                              fill=(200, 200, 100, 255), font=font)
