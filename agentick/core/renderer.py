"""Enhanced multi-modal rendering engine for gridworld environments.

This module provides production-quality renderers with:
- ASCII: ANSI color support, legend, multi-character cells
- Language: Full spatial reasoning, memory awareness, goals, threats
- Pixel: Distinct programmatic sprites, HUD overlay, multiple tile sizes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from agentick.core.entity import Agent, Entity
from agentick.core.grid import Grid
from agentick.core.language import AdvancedLanguageRenderer, LanguageConfig
from agentick.core.types import CellType, Direction, ObjectType

# Set pygame to headless by default
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class Renderer(Protocol):
    """Protocol for renderer implementations."""

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> Any:
        """Render the current state."""
        ...


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    tile_size: int = 32
    show_grid: bool = True
    show_fog: bool = False
    show_hud: bool = True
    use_ansi_colors: bool = True
    font_size: int = 12


# ANSI color codes
ANSI_COLORS = {
    "reset": "\033[0m",
    "agent": "\033[94m",  # Blue
    "wall": "\033[90m",  # Dark gray
    "goal": "\033[92m",  # Green
    "key": "\033[93m",  # Yellow
    "door": "\033[33m",  # Orange/brown
    "hazard": "\033[91m",  # Red
    "box": "\033[95m",  # Magenta
    "empty": "\033[37m",  # White
    "water": "\033[96m",  # Cyan
    "ice": "\033[97m",  # Bright white
    "npc": "\033[96m",  # Cyan
    "enemy": "\033[91m",  # Red
    "gem": "\033[35m",  # Purple
    "lever": "\033[33m",  # Yellow/brown
    "potion": "\033[96m",  # Cyan
    "scroll": "\033[33m",  # Yellow
    "coin": "\033[93m",  # Bright yellow
    "orb": "\033[95m",  # Magenta
}


class ASCIIRenderer:
    """Render grid as ASCII art with ANSI colors and legend."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str:
        """
        Render grid as ASCII string with ANSI colors and legend.

        Returns:
            Multi-line ASCII representation with colors and legend
        """
        output = []
        task_config = info.get("task_config", {})
        task_name = info.get("task_name", "")

        # ── Task-specific annotation headers ────────────────────────────
        if "InstructionFollowing" in task_name and "target_type" in task_config:
            target_val = task_config["target_type"]
            _OBJ_NAMES = {
                int(ObjectType.GEM): "GEM", int(ObjectType.SCROLL): "SCROLL",
                int(ObjectType.ORB): "ORB", int(ObjectType.COIN): "COIN",
            }
            tname = _OBJ_NAMES.get(int(target_val), f"OBJ_{target_val}")
            output.append(f"Target: [{tname}]")

        if "TaskInterference" in task_name:
            red = task_config.get("_red_meter", 0.0)
            blue = task_config.get("_blue_meter", 0.0)
            r_bar = "=" * int(red * 10) + " " * (10 - int(red * 10))
            b_bar = "=" * int(blue * 10) + " " * (10 - int(blue * 10))
            output.append(f"GEM:[{r_bar}] {red:.2f}  ORB:[{b_bar}] {blue:.2f}")

        if "TreasureHunt" in task_name:
            clue_info = task_config.get("_clue_info", {})
            clues_read = task_config.get("_clues_read", [])
            dirs = {0: "N", 1: "E", 2: "S", 3: "W"}
            clue_parts = []
            for cpos in clues_read:
                key = f"{cpos[0]},{cpos[1]}" if isinstance(cpos, (list, tuple)) else cpos
                ci = clue_info.get(key, clue_info.get(tuple(cpos), {}))
                if ci:
                    d = dirs.get(ci.get("direction", 0), "?")
                    dist = ci.get("distance", "?")
                    clue_parts.append(f"{d}{dist}")
            # Wrap clues: max 5 per line to avoid overflow
            max_per_line = 5
            for row_start in range(0, len(clue_parts), max_per_line):
                row = clue_parts[row_start:row_start + max_per_line]
                output.append("Clue: " + "  ".join(row))

        if "RuleInduction" in task_name:
            trial = task_config.get("_current_trial", 0) + 1
            n_trials = task_config.get("_n_trials", 1)
            target_type = task_config.get("_target_type", 0)
            obj_names = {
                int(ObjectType.GEM): "GEM", int(ObjectType.POTION): "POTION",
                int(ObjectType.SCROLL): "SCROLL", int(ObjectType.COIN): "COIN",
                int(ObjectType.ORB): "ORB",
            }
            tname = obj_names.get(int(target_type), f"OBJ_{target_type}")
            output.append(f"Trial: {trial}/{n_trials}  Target: [{tname}]")

        if "DistributionShift" in task_name:
            phase = task_config.get("_phases_completed", 0) + 1
            n_phases = task_config.get("_n_phases", 3)
            phase_type = task_config.get("_current_phase_type", "goal_reach")
            phase_type_names = {
                "goal_reach": "Navigate", "key_door": "Key+Door",
                "lever_barrier": "Lever", "collection": "Collect",
                "box_push": "BoxPush",
            }
            type_label = phase_type_names.get(phase_type, phase_type)
            remap = task_config.get("_action_remap")
            line = f"Phase: {phase}/{n_phases}  Task: {type_label}"
            if remap:
                line += "  [REMAPPED]"
            output.append(line)

        if "LightsOut" in task_name:
            output.append("LightsOut: 1=lit, 2=unlit")

        # Build character grid
        char_grid = np.full((grid.height, grid.width), ".", dtype=object)
        color_grid = np.full((grid.height, grid.width), "empty", dtype=object)

        # Terrain layer
        for y in range(grid.height):
            for x in range(grid.width):
                # Fog of war: metadata == -1 means fogged/unexplored
                if int(grid.metadata[y, x]) == -1:
                    char_grid[y, x] = " "
                    color_grid[y, x] = "wall"
                    continue

                terrain_val = int(grid.terrain[y, x])
                if terrain_val == CellType.EMPTY:
                    char_grid[y, x] = "."
                    color_grid[y, x] = "empty"
                elif terrain_val == CellType.WALL:
                    char_grid[y, x] = "#"
                    # Barrier walls: metadata encodes color group
                    _BARRIER_COLORS = {
                        1: "hazard", 2: "agent", 3: "goal", 4: "coin", 5: "scroll",
                    }
                    wall_meta = int(grid.metadata[y, x])
                    color_grid[y, x] = _BARRIER_COLORS.get(wall_meta, "wall")
                elif terrain_val == CellType.HAZARD:
                    char_grid[y, x] = "X"
                    color_grid[y, x] = "hazard"
                elif terrain_val == CellType.WATER:
                    char_grid[y, x] = "~"
                    color_grid[y, x] = "water"
                elif terrain_val == CellType.ICE:
                    char_grid[y, x] = "i"
                    color_grid[y, x] = "ice"
                elif terrain_val == CellType.HOLE:
                    char_grid[y, x] = "O"
                    color_grid[y, x] = "wall"

        # Objects layer
        for y in range(grid.height):
            for x in range(grid.width):
                # Skip fogged cells
                if int(grid.metadata[y, x]) == -1:
                    continue
                obj_val = int(grid.objects[y, x])
                if obj_val == ObjectType.GOAL:
                    char_grid[y, x] = "G"
                    color_grid[y, x] = "goal"
                elif obj_val == ObjectType.KEY:
                    meta_val = int(grid.metadata[y, x])
                    _KEY_ANSI = {0: "key", 1: "hazard", 2: "agent", 3: "goal"}
                    _KEY_SUFFIX = {0: "g", 1: "r", 2: "b", 3: "n"}  # gold/red/blue/green
                    char_grid[y, x] = "K" + _KEY_SUFFIX.get(meta_val, "")
                    color_grid[y, x] = _KEY_ANSI.get(meta_val, "key")
                elif obj_val == ObjectType.DOOR:
                    meta_val = int(grid.metadata[y, x])
                    _DOOR_ANSI = {0: "door", 1: "hazard", 2: "agent", 3: "goal"}
                    _DOOR_SUFFIX = {0: "g", 1: "r", 2: "b", 3: "n"}
                    if meta_val >= 10:
                        color_idx = meta_val - 10
                        char_grid[y, x] = "d" + _DOOR_SUFFIX.get(color_idx, "")
                        color_grid[y, x] = _DOOR_ANSI.get(color_idx, "goal")
                    else:
                        char_grid[y, x] = "D" + _DOOR_SUFFIX.get(meta_val, "")
                        color_grid[y, x] = _DOOR_ANSI.get(meta_val, "door")
                elif obj_val == ObjectType.SWITCH:
                    meta_val = int(grid.metadata[y, x])
                    _GC_COLORS = {0: "key", 1: "hazard", 2: "agent", 3: "goal", 4: "coin"}
                    if meta_val >= 100:
                        # Activated switch (BacktrackPuzzle, SwitchCircuit)
                        char_grid[y, x] = "s"
                        color_grid[y, x] = "goal"
                    elif meta_val > 0:
                        # GraphColoring color digit
                        char_grid[y, x] = str(meta_val)
                        color_grid[y, x] = _GC_COLORS.get(meta_val, "key")
                    else:
                        # Off / uncolored
                        char_grid[y, x] = "S"
                        color_grid[y, x] = "wall"
                elif obj_val == ObjectType.BOX:
                    meta_val = int(grid.metadata[y, x])
                    if meta_val >= 100:
                        # Correctly placed tile (TileSorting): green tint
                        real_tile = meta_val - 100
                        char_grid[y, x] = (
                            str(real_tile) if real_tile <= 9
                            else chr(ord("A") + real_tile - 10)
                        )
                        color_grid[y, x] = "goal"  # green color for correct
                    elif meta_val != 0:
                        # Numbered tile (TileSorting): show tile number
                        char_grid[y, x] = (
                            str(meta_val) if meta_val <= 9
                            else chr(ord("A") + meta_val - 10)
                        )
                        color_grid[y, x] = "box"
                    else:
                        char_grid[y, x] = "B"
                        color_grid[y, x] = "box"
                elif obj_val == ObjectType.TARGET:
                    meta_val = int(grid.metadata[y, x])
                    if meta_val >= 200:
                        # TileSorting: goal slot — show expected tile number
                        slot_tile = meta_val - 200
                        char_grid[y, x] = (
                            str(slot_tile) if slot_tile <= 9
                            else chr(ord("A") + slot_tile - 10)
                        )
                        color_grid[y, x] = "goal"
                    else:
                        # Typed target: show expected object type character
                        _TYPED_TARGET_CHARS = {
                            5: ("B", "box"),    # BOX slot
                            14: ("d", "gem"),   # GEM slot
                            15: ("L", "lever"), # LEVER slot
                            16: ("P", "potion"),# POTION slot
                            17: ("?", "scroll"),# SCROLL slot
                            18: ("c", "coin"),  # COIN slot
                            19: ("O", "orb"),   # ORB slot
                        }
                        if meta_val in _TYPED_TARGET_CHARS:
                            ch, col = _TYPED_TARGET_CHARS[meta_val]
                            char_grid[y, x] = ch
                            color_grid[y, x] = col
                        else:
                            char_grid[y, x] = "T"
                            color_grid[y, x] = "goal"
                elif obj_val == ObjectType.TOOL:
                    char_grid[y, x] = "t"
                    color_grid[y, x] = "key"
                elif obj_val == ObjectType.RESOURCE:
                    energy = int(grid.metadata[y, x])
                    if energy > 0:
                        digit = min(9, max(1, energy // 11 + 1))
                        char_grid[y, x] = str(digit)
                        if energy > 60:
                            color_grid[y, x] = "goal"
                        elif energy > 30:
                            color_grid[y, x] = "coin"
                        else:
                            color_grid[y, x] = "hazard"
                    else:
                        char_grid[y, x] = "r"
                        color_grid[y, x] = "key"
                elif obj_val == ObjectType.BREADCRUMB:
                    char_grid[y, x] = "*"
                    color_grid[y, x] = "key"
                elif obj_val == ObjectType.NPC:
                    npc_meta = int(grid.metadata[y, x])
                    _NPC_TYPE_CHARS = {1: "F", 3: "X", 4: "M", 5: "C"}
                    _NPC_TYPE_COLORS = {
                        1: "water", 3: "goal", 4: "scroll", 5: "coin",
                    }
                    char_grid[y, x] = _NPC_TYPE_CHARS.get(npc_meta, "N")
                    color_grid[y, x] = _NPC_TYPE_COLORS.get(npc_meta, "npc")
                elif obj_val == ObjectType.ENEMY:
                    char_grid[y, x] = "E"
                    color_grid[y, x] = "enemy"
                elif obj_val == ObjectType.SHEEP:
                    char_grid[y, x] = "o"
                    color_grid[y, x] = "goal"
                elif obj_val == ObjectType.BLOCKER:
                    char_grid[y, x] = "X"
                    color_grid[y, x] = "hazard"
                elif obj_val == ObjectType.GEM:
                    char_grid[y, x] = "d"
                    color_grid[y, x] = "gem"
                elif obj_val == ObjectType.LEVER:
                    lever_meta = int(grid.metadata[y, x])
                    if lever_meta > 0:
                        char_grid[y, x] = "l"
                        color_grid[y, x] = "goal"
                    else:
                        char_grid[y, x] = "L"
                        color_grid[y, x] = "lever"
                elif obj_val == ObjectType.POTION:
                    char_grid[y, x] = "P"
                    color_grid[y, x] = "potion"
                elif obj_val == ObjectType.SCROLL:
                    scroll_meta = int(grid.metadata[y, x])
                    if scroll_meta > 0:
                        direction = scroll_meta // 10
                        _DIR_CHARS = {0: "^", 1: ">", 2: "v", 3: "<"}
                        char_grid[y, x] = _DIR_CHARS.get(direction, "?")
                    else:
                        char_grid[y, x] = "?"
                    color_grid[y, x] = "scroll"
                elif obj_val == ObjectType.COIN:
                    char_grid[y, x] = "c"
                    color_grid[y, x] = "coin"
                elif obj_val == ObjectType.ORB:
                    char_grid[y, x] = "O"
                    color_grid[y, x] = "orb"

        # Agent layer
        ax, ay = agent.position
        # Multi-character cell: show agent orientation
        direction_chars = {
            Direction.NORTH: "^",
            Direction.EAST: ">",
            Direction.SOUTH: "v",
            Direction.WEST: "<",
        }
        char_grid[ay, ax] = direction_chars[agent.orientation]
        color_grid[ay, ax] = "agent"

        # If agent is on an object, show both
        obj_val = int(grid.objects[ay, ax])
        if obj_val != ObjectType.NONE:
            # Multi-character representation
            if obj_val == ObjectType.DOOR:
                door_meta = int(grid.metadata[ay, ax])
                obj_char = "d" if door_meta >= 10 else "D"
            else:
                obj_char = {
                    ObjectType.GOAL: "G",
                    ObjectType.KEY: "K",
                }.get(obj_val, "")
            if obj_char:
                char_grid[ay, ax] = f"{direction_chars[agent.orientation]}{obj_char}"

        # Add other agents/NPCs from entities
        for entity in entities:
            if entity.entity_type in ("npc", "enemy"):
                ex, ey = entity.position
                char_grid[ey, ex] = "N" if entity.entity_type == "npc" else "E"
                color_grid[ey, ex] = "hazard" if entity.entity_type == "enemy" else "agent"

        # Convert to string with ANSI colors
        for y in range(grid.height):
            row_parts = []
            for x in range(grid.width):
                char = char_grid[y, x]
                # Pad multi-character cells
                if len(str(char)) > 1:
                    char_str = str(char)[:2].ljust(2)
                else:
                    char_str = str(char).ljust(2)

                if self.config.use_ansi_colors:
                    color = ANSI_COLORS.get(color_grid[y, x], "")
                    reset = ANSI_COLORS["reset"]
                    row_parts.append(f"{color}{char_str}{reset}")
                else:
                    row_parts.append(char_str)

            output.append("".join(row_parts))

        # Build dynamic legend based on object types actually present
        from agentick.core.annotations import extract_annotations

        ann = extract_annotations(grid, info)

        present_objs = set()
        for y in range(grid.height):
            for x in range(grid.width):
                obj = int(grid.objects[y, x])
                if obj != 0:
                    present_objs.add(obj)
                terrain = int(grid.terrain[y, x])
                if terrain != 0:
                    present_objs.add(-terrain)  # negative for terrain

        has_keys = ObjectType.KEY in present_objs
        has_doors = ObjectType.DOOR in present_objs
        has_scrolls = ObjectType.SCROLL in present_objs

        _LEGEND_MAP: list[tuple[str, str, str, set]] = [
            # (symbol, description, color_key, required_objs — empty = always)
            ("A", "Agent (facing ^v<>)", "agent", set()),
            ("#", "Wall", "wall", {-CellType.WALL}),
            (".", "Empty", "", set()),
            (" ", "Fog (unexplored)", "wall", set()),
            ("G", "Goal", "goal", {ObjectType.GOAL}),
            ("K", "Key", "key", {ObjectType.KEY}),
            ("D", "Door (closed)", "door", {ObjectType.DOOR}),
            ("d", "Door (open)", "goal", {ObjectType.DOOR}),
            ("S", "Switch (off)", "wall", {ObjectType.SWITCH}),
            ("s", "Switch (on)", "goal", {ObjectType.SWITCH}),
            ("L", "Lever (off)", "lever", {ObjectType.LEVER}),
            ("l", "Lever (on)", "goal", {ObjectType.LEVER}),
            ("B", "Box", "box", {ObjectType.BOX}),
            ("T", "Target", "goal", {ObjectType.TARGET}),
            ("~", "Water", "water", {-CellType.WATER}),
            ("X", "Hazard", "hazard", {-CellType.HAZARD}),
            ("i", "Ice", "ice", {-CellType.ICE}),
            ("E", "Enemy", "enemy", {ObjectType.ENEMY}),
            ("N", "NPC", "npc", {ObjectType.NPC}),
            ("o", "Sheep", "goal", {ObjectType.SHEEP}),
            ("?", "Scroll", "scroll", {ObjectType.SCROLL}),
            ("c", "Coin", "coin", {ObjectType.COIN}),
            ("P", "Potion", "potion", {ObjectType.POTION}),
            ("O", "Orb", "orb", {ObjectType.ORB}),
        ]

        legend_parts = ["\n--- Legend ---"]
        for symbol, desc, color_key, req in _LEGEND_MAP:
            if req and not (req & present_objs):
                continue
            line = f"{symbol}: {desc}"
            if self.config.use_ansi_colors and color_key:
                color = ANSI_COLORS.get(color_key, "")
                reset = ANSI_COLORS["reset"]
                line = f"{color}{symbol}{reset}: {desc}"
            legend_parts.append(line)

        # Key/door color guide (only when keys or doors are present)
        if has_keys or has_doors:
            legend_parts.append(
                "Color suffixes: g=gold, r=red, b=blue, n=green "
                "(e.g. Kr=red key, Db=blue door)"
            )

        # Scroll direction guide (only when scrolls present)
        if has_scrolls and ann.scroll_directions:
            legend_parts.append("Scroll arrows: ^ N, > E, v S, < W (direction + distance clue)")

        output.extend(legend_parts)

        return "\n".join(output)


class EnhancedLanguageRenderer:
    """Wrapper for AdvancedLanguageRenderer with configuration options."""

    def __init__(
        self,
        structured: bool = False,
        verbosity: str = "standard",
        perspective: str = "first_person",
    ):
        """
        Initialize language renderer.

        Args:
            structured: Return structured dict instead of text
            verbosity: "minimal", "standard", or "verbose"
            perspective: "first_person", "third_person", or "omniscient"
        """
        self.structured = structured  # Keep for backward compatibility
        config = LanguageConfig(
            verbosity=verbosity,
            perspective=perspective,
            structured=structured,
        )
        self.renderer = AdvancedLanguageRenderer(config)

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str | dict[str, Any]:
        """
        Render grid as language description.

        Returns:
            Natural language string or structured dict
        """
        return self.renderer.render(grid, entities, agent, info)


class StateDictRenderer:
    """Render grid as structured state dictionary with metadata annotations."""

    def __init__(self, fast_mode: bool = False):
        """
        Initialize renderer.

        Args:
            fast_mode: If True, skip expensive conversions (tolist(), etc.)
        """
        self.fast_mode = fast_mode

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Render grid as complete state dictionary.

        Returns:
            Dictionary with full grid state, entity state, and annotations.
        """
        from agentick.core.annotations import extract_annotations

        ann = extract_annotations(grid, info)

        # Build annotations dict (serialisable)
        annotations: dict[str, Any] = {}
        if ann.key_colors:
            annotations["key_colors"] = {f"{p[0]},{p[1]}": c for p, c in ann.key_colors.items()}
        if ann.door_colors:
            annotations["door_colors"] = {f"{p[0]},{p[1]}": c for p, c in ann.door_colors.items()}
        if ann.door_states:
            annotations["door_states"] = {f"{p[0]},{p[1]}": s for p, s in ann.door_states.items()}
        if ann.switch_states:
            annotations["switch_states"] = {
                f"{p[0]},{p[1]}": s for p, s in ann.switch_states.items()
            }
        if ann.switch_colors:
            annotations["switch_colors"] = {
                f"{p[0]},{p[1]}": c for p, c in ann.switch_colors.items()
            }
        if ann.lever_states:
            annotations["lever_states"] = {
                f"{p[0]},{p[1]}": s for p, s in ann.lever_states.items()
            }
        if ann.npc_types:
            annotations["npc_types"] = {f"{p[0]},{p[1]}": t for p, t in ann.npc_types.items()}
        if ann.enemy_types:
            annotations["enemy_types"] = {f"{p[0]},{p[1]}": t for p, t in ann.enemy_types.items()}
        if ann.resource_energy:
            annotations["resource_energy"] = {
                f"{p[0]},{p[1]}": e for p, e in ann.resource_energy.items()
            }
        if ann.tile_numbers:
            annotations["tile_numbers"] = {
                f"{p[0]},{p[1]}": n for p, n in ann.tile_numbers.items()
            }
        if ann.target_slots:
            annotations["target_slots"] = {
                f"{p[0]},{p[1]}": n for p, n in ann.target_slots.items()
            }
        if ann.scroll_directions:
            annotations["scroll_directions"] = {
                f"{p[0]},{p[1]}": d for p, d in ann.scroll_directions.items()
            }

        # Task-level annotations
        task_ann: dict[str, Any] = {}
        if ann.target_object:
            task_ann["target_object"] = ann.target_object
        if ann.gem_meter is not None:
            task_ann["gem_meter"] = ann.gem_meter
            task_ann["orb_meter"] = ann.orb_meter
            task_ann["meter_threshold"] = ann.meter_threshold
        if ann.trial_info:
            task_ann["trial_info"] = ann.trial_info
        if ann.phase_info:
            task_ann["phase_info"] = ann.phase_info
        if ann.clues:
            task_ann["clues"] = ann.clues
        if ann.lights_out:
            task_ann["lights_out"] = True
        if task_ann:
            annotations["task"] = task_ann

        if self.fast_mode:
            return {
                "grid": {
                    "height": grid.height,
                    "width": grid.width,
                    "terrain": grid.terrain,
                    "objects": grid.objects,
                    "agents": grid.agents,
                    "metadata": grid.metadata,
                },
                "agent": {
                    "position": agent.position,
                    "orientation": agent.orientation,
                    "inventory": agent.inventory,
                    "energy": agent.energy,
                    "health": agent.health,
                },
                "entities": entities,
                "annotations": annotations,
                "info": info,
            }
        else:
            return {
                "grid": {
                    "height": grid.height,
                    "width": grid.width,
                    "terrain": grid.terrain.tolist(),
                    "objects": grid.objects.tolist(),
                    "agents": grid.agents.tolist(),
                    "metadata": grid.metadata.tolist(),
                },
                "agent": {
                    "position": agent.position,
                    "orientation": agent.orientation.name.lower(),
                    "inventory": [item.to_dict() for item in agent.inventory],
                    "energy": agent.energy,
                    "health": agent.health,
                },
                "entities": [entity.to_dict() for entity in entities],
                "annotations": annotations,
                "info": info,
            }


def create_renderer(
    mode: str,
    tile_size: int = 32,
    show_hud: bool = True,
    verbosity: str = "standard",
    perspective: str = "first_person",
    fast_mode: bool = False,
    use_isometric: bool = False,
    **kwargs,
) -> Renderer:
    """
    Factory function to create renderer for given mode.

    Args:
        mode: Render mode ("ascii", "language", "language_structured", "rgb_array", "state_dict")
        tile_size: Tile size for pixel rendering (8, 16, 32, or 64)
        show_hud: Show HUD overlay in pixel mode
        verbosity: Language verbosity ("minimal", "standard", "verbose")
        perspective: Language perspective ("first_person", "third_person", "omniscient")
        fast_mode: Enable fast mode for state_dict (skip expensive conversions)
        **kwargs: Additional arguments for renderer config

    Returns:
        Renderer instance
    """
    if mode == "ascii":
        config = RenderConfig(tile_size=tile_size, **kwargs)
        return ASCIIRenderer(config)
    elif mode == "language":
        return EnhancedLanguageRenderer(
            structured=False,
            verbosity=verbosity,
            perspective=perspective,
        )
    elif mode == "language_structured":
        return EnhancedLanguageRenderer(
            structured=True,
            verbosity=verbosity,
            perspective=perspective,
        )
    elif mode == "human":
        # Human mode uses isometric renderer (lazy import to avoid circular deps)
        from agentick.rendering.iso_renderer import IsometricRenderer

        return IsometricRenderer()
    elif mode == "state_dict":
        return StateDictRenderer(fast_mode=fast_mode)
    else:
        raise ValueError(f"Unknown render mode: {mode}")


# Backward compatibility aliases
LanguageRenderer = EnhancedLanguageRenderer
