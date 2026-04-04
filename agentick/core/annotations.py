"""Shared annotation extraction for multimodal rendering parity.

All 5 render modes (ascii, language, language_structured, rgb_array, state_dict)
must expose identical semantic information. This module centralises metadata
interpretation so every renderer consumes the same facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentick.core.grid import Grid
from agentick.core.types import ObjectType, Position

# ── Constants ──────────────────────────────────────────────────────────────────

_COLOR_NAMES = {0: "gold", 1: "red", 2: "blue", 3: "green"}
_DIRECTION_NAMES = {0: "north", 1: "east", 2: "south", 3: "west"}
_DIRECTION_SHORT = {0: "N", 1: "E", 2: "S", 3: "W"}
_OBJ_NAMES = {
    int(ObjectType.GEM): "gem",
    int(ObjectType.POTION): "potion",
    int(ObjectType.SCROLL): "scroll",
    int(ObjectType.COIN): "coin",
    int(ObjectType.ORB): "orb",
}
_NPC_TYPES = {1: "follower", 3: "fearful", 4: "mirror", 5: "contrarian"}
_ENEMY_TYPES = {0: "chaser", 1: "ambusher", 2: "patrol", 3: "erratic"}


@dataclass
class TaskAnnotations:
    """Task-specific annotations extracted from grid + config."""

    # ── Per-cell metadata ──────────────────────────────────────────────────
    key_colors: dict[Position, str] = field(default_factory=dict)
    door_colors: dict[Position, str] = field(default_factory=dict)
    door_states: dict[Position, str] = field(default_factory=dict)
    switch_states: dict[Position, str] = field(default_factory=dict)
    switch_colors: dict[Position, str] = field(default_factory=dict)
    lever_states: dict[Position, str] = field(default_factory=dict)
    npc_types: dict[Position, str] = field(default_factory=dict)
    enemy_types: dict[Position, str] = field(default_factory=dict)
    resource_energy: dict[Position, int] = field(default_factory=dict)
    tile_numbers: dict[Position, int] = field(default_factory=dict)
    target_slots: dict[Position, int] = field(default_factory=dict)
    scroll_directions: dict[Position, dict] = field(default_factory=dict)
    fog_cells: set[Position] = field(default_factory=set)

    # ── Task-level annotations (from task_config) ─────────────────────────
    target_object: str | None = None
    gem_meter: float | None = None
    orb_meter: float | None = None
    meter_threshold: float | None = None
    trial_info: str | None = None
    phase_info: str | None = None
    clues: list[dict] | None = None
    lights_out: bool = False


def extract_annotations(
    grid: Grid,
    info: dict[str, Any],
) -> TaskAnnotations:
    """Extract all semantic annotations from the grid and task config.

    Every renderer should call this once and use the result, guaranteeing
    that all observation modalities expose identical information.
    """
    ann = TaskAnnotations()
    task_config = info.get("task_config", {})
    task_name = info.get("task_name", "")

    # ── Scan the grid for per-cell metadata ────────────────────────────────
    for y in range(grid.height):
        for x in range(grid.width):
            meta = int(grid.metadata[y, x])
            pos: Position = (x, y)

            # Fog of war
            if meta == -1:
                ann.fog_cells.add(pos)
                continue

            obj = int(grid.objects[y, x])

            # Keys
            if obj == ObjectType.KEY:
                ann.key_colors[pos] = _COLOR_NAMES.get(meta, "gold")

            # Doors
            elif obj == ObjectType.DOOR:
                if meta >= 10:
                    ann.door_states[pos] = "open"
                    ann.door_colors[pos] = _COLOR_NAMES.get(meta - 10, "gold")
                else:
                    ann.door_states[pos] = "closed"
                    ann.door_colors[pos] = _COLOR_NAMES.get(meta, "gold")

            # Switches
            elif obj == ObjectType.SWITCH:
                if meta >= 100:
                    ann.switch_states[pos] = "on"
                elif meta > 0:
                    ann.switch_states[pos] = "off"
                    ann.switch_colors[pos] = _COLOR_NAMES.get(meta, str(meta))
                else:
                    ann.switch_states[pos] = "off"

            # Levers
            elif obj == ObjectType.LEVER:
                ann.lever_states[pos] = "on" if meta > 0 else "off"

            # NPCs
            elif obj == ObjectType.NPC:
                npc_type = meta & 0x0F
                ann.npc_types[pos] = _NPC_TYPES.get(npc_type, "npc")

            # Enemies
            elif obj == ObjectType.ENEMY:
                enemy_type = meta & 0x0F
                ann.enemy_types[pos] = _ENEMY_TYPES.get(enemy_type, "enemy")

            # Resources
            elif obj == ObjectType.RESOURCE:
                ann.resource_energy[pos] = meta

            # Boxes (TileSorting tile numbers)
            elif obj == ObjectType.BOX:
                if meta >= 100:
                    ann.tile_numbers[pos] = meta - 100
                elif meta > 0:
                    ann.tile_numbers[pos] = meta

            # Targets (TileSorting goal slots)
            elif obj == ObjectType.TARGET:
                if meta >= 200:
                    ann.target_slots[pos] = meta - 200

            # Scrolls (TreasureHunt direction+distance)
            elif obj == ObjectType.SCROLL:
                if meta > 0:
                    direction = meta // 10
                    distance = meta % 10
                    ann.scroll_directions[pos] = {
                        "direction": _DIRECTION_NAMES.get(direction, "unknown"),
                        "direction_short": _DIRECTION_SHORT.get(direction, "?"),
                        "distance": distance,
                    }

    # ── Task-level annotations ─────────────────────────────────────────────

    # InstructionFollowing: target object type
    if "InstructionFollowing" in task_name and "target_type" in task_config:
        target_val = int(task_config["target_type"])
        ann.target_object = _OBJ_NAMES.get(target_val, f"object_{target_val}")

    # TaskInterference: meters
    if "TaskInterference" in task_name:
        ann.gem_meter = task_config.get("_red_meter")
        ann.orb_meter = task_config.get("_blue_meter")
        ann.meter_threshold = task_config.get("threshold", 0.5)

    # TreasureHunt: collected clues
    if "TreasureHunt" in task_name:
        clue_info = task_config.get("_clue_info", {})
        clues_read = task_config.get("_clues_read", [])
        clue_list = []
        for cpos in clues_read:
            key = f"{cpos[0]},{cpos[1]}" if isinstance(cpos, (list, tuple)) else cpos
            ci = clue_info.get(key, clue_info.get(tuple(cpos), {}))
            if ci:
                clue_list.append({
                    "direction": _DIRECTION_NAMES.get(ci.get("direction", 0), "unknown"),
                    "direction_short": _DIRECTION_SHORT.get(ci.get("direction", 0), "?"),
                    "distance": ci.get("distance", 0),
                })
        ann.clues = clue_list if clue_list else None

    # RuleInduction: trial info
    if "RuleInduction" in task_name:
        trial = task_config.get("_current_trial", 0) + 1
        n_trials = task_config.get("_n_trials", 1)
        target_type = task_config.get("_target_type", 0)
        tname = _OBJ_NAMES.get(int(target_type), f"object_{target_type}")
        ann.trial_info = f"Trial {trial}/{n_trials}, target: {tname}"

    # DistributionShift: phase info
    if "DistributionShift" in task_name:
        phase = task_config.get("_phases_completed", 0) + 1
        n_phases = task_config.get("_n_phases", 3)
        phase_type = task_config.get("_current_phase_type", "goal_reach")
        remap = task_config.get("_action_remap")
        label = f"Phase {phase}/{n_phases}: {phase_type}"
        if remap:
            label += " [remapped]"
        ann.phase_info = label

    # LightsOut
    if "LightsOut" in task_name:
        ann.lights_out = True

    return ann
