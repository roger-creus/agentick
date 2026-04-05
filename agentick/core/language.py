"""Advanced natural language rendering for gridworld environments.

This module provides rich, context-aware language descriptions with:
- Spatial reasoning with relative directions
- Memory-aware descriptions (references to past states)
- Goal-aware and threat-aware descriptions
- Configurable verbosity and perspective
- Structured JSON variant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agentick.core.entity import Agent, Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, Direction, ObjectType

# Color name maps for keys/doors (metadata → color prefix)
_KEY_COLOR_NAMES = {0: "", 1: "red ", 2: "blue ", 3: "green "}
_DOOR_COLOR_NAMES = {0: "", 1: "red ", 2: "blue ", 3: "green "}


@dataclass
class LanguageConfig:
    """Configuration for language rendering."""

    verbosity: Literal["minimal", "standard", "verbose"] = "standard"
    perspective: Literal["first_person", "third_person", "omniscient"] = "first_person"
    include_memory: bool = True
    include_goals: bool = True
    include_threats: bool = True
    include_spatial_reasoning: bool = True
    structured: bool = False


class AdvancedLanguageRenderer:
    """Render grid as rich natural language description."""

    def __init__(self, config: LanguageConfig | None = None):
        """
        Initialize language renderer.

        Args:
            config: Language rendering configuration
        """
        self.config = config or LanguageConfig()
        self.memory: list[dict[str, Any]] = []  # Track visited locations and events

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str | dict[str, Any]:
        """
        Render grid as language description.

        Args:
            grid: Grid state
            entities: List of entities
            agent: Agent state
            info: Additional info dict

        Returns:
            Natural language string or structured dict
        """
        if self.config.structured:
            return self._render_structured(grid, entities, agent, info)
        else:
            return self._render_narrative(grid, entities, agent, info)

    def _render_narrative(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str:
        """Render as narrative description."""
        # Stash info for _describe_surroundings_relative (needs task_config)
        self._last_info = info
        parts = []

        # Opening based on perspective
        if self.config.perspective == "first_person":
            parts.extend(self._describe_first_person(grid, agent))
        elif self.config.perspective == "third_person":
            parts.extend(self._describe_third_person(grid, agent))
        else:  # omniscient
            parts.extend(self._describe_omniscient(grid, agent, entities))

        # ── Task-specific annotations ───────────────────────────────────
        task_config = info.get("task_config", {})
        task_name = info.get("task_name", "")
        if "InstructionFollowing" in task_name and "target_type" in task_config:
            obj_names = {
                int(ObjectType.GEM): "GEM", int(ObjectType.SCROLL): "SCROLL",
                int(ObjectType.ORB): "ORB", int(ObjectType.COIN): "COIN",
            }
            tname = obj_names.get(
                int(task_config["target_type"]), "UNKNOWN"
            )
            parts.append(f"Your target is: {tname}.")
        if "TaskInterference" in task_name:
            red = task_config.get("_red_meter", 0.0)
            blue = task_config.get("_blue_meter", 0.0)
            threshold = task_config.get("threshold", 0.5)
            parts.append(
                f"Gem meter: {red:.2f}/{threshold:.1f}. "
                f"Orb meter: {blue:.2f}/{threshold:.1f}."
            )
        if "TreasureHunt" in task_name:
            clue_info = task_config.get("_clue_info", {})
            clues_read = task_config.get("_clues_read", [])
            dir_names = {0: "North", 1: "East", 2: "South", 3: "West"}
            for cpos in clues_read:
                key = f"{cpos[0]},{cpos[1]}" if isinstance(cpos, (list, tuple)) else cpos
                ci = clue_info.get(key, clue_info.get(tuple(cpos), {}))
                if ci:
                    d = dir_names.get(ci.get("direction", 0), "unknown")
                    dist = ci.get("distance", "?")
                    parts.append(
                        f"A scroll pointed {d}, {dist} tiles away."
                    )
        if "RuleInduction" in task_name:
            trial = task_config.get("_current_trial", 0) + 1
            n_trials = task_config.get("_n_trials", 1)
            target_type = task_config.get("_target_type", 0)
            obj_names = {
                int(ObjectType.GEM): "GEM", int(ObjectType.POTION): "POTION",
                int(ObjectType.SCROLL): "SCROLL", int(ObjectType.COIN): "COIN",
                int(ObjectType.ORB): "ORB",
            }
            tname = obj_names.get(int(target_type), "UNKNOWN")
            parts.append(f"Trial {trial} of {n_trials}. Target object: {tname}.")
        if "DistributionShift" in task_name:
            phase = task_config.get("_phases_completed", 0) + 1
            n_phases = task_config.get("_n_phases", 3)
            phase_type = task_config.get("_current_phase_type", "goal_reach")
            phase_desc = {
                "goal_reach": "Navigate to the goal.",
                "key_door": "Collect the key and open the door to reach the goal.",
                "lever_barrier": "Activate the lever to open the barrier, then reach the goal.",
                "collection": "Collect all gems, then reach the goal.",
                "box_push": "Push the box onto the target, then reach the goal.",
            }
            parts.append(f"Phase {phase} of {n_phases}. {phase_desc.get(phase_type, '')}")

        # Spatial surroundings with relative directions
        if self.config.include_spatial_reasoning:
            surroundings = self._describe_surroundings_relative(grid, agent, entities)
            if surroundings:
                parts.append(surroundings)

        # Inventory
        parts.append(self._describe_inventory(agent))

        # Status (energy/health)
        if self.config.verbosity != "minimal":
            status = self._describe_status(agent)
            if status:
                parts.append(status)

        # Goals
        if self.config.include_goals:
            goals = self._describe_goals(grid, agent, info)
            if goals:
                parts.append(goals)

        # Threats/dangers
        if self.config.include_threats:
            threats = self._describe_threats(grid, agent, entities)
            if threats:
                parts.append(threats)

        # Memory (if enabled and verbose)
        if self.config.include_memory and self.config.verbosity == "verbose":
            memory = self._describe_memory(agent)
            if memory:
                parts.append(memory)

        # Valid actions
        if "valid_actions" in info and self.config.verbosity != "minimal":
            action_names = info["valid_actions"]
            if self.config.verbosity == "verbose":
                parts.append(f"Available actions: {', '.join(action_names)}.")
            else:
                actions_str = ", ".join(action_names[:5])
                ellipsis = "..." if len(action_names) > 5 else ""
                parts.append(f"Actions: {actions_str}{ellipsis}.")

        return " ".join(parts)

    def _describe_first_person(self, grid: Grid, agent: Agent) -> list[str]:
        """First-person perspective description."""
        parts = []

        if self.config.verbosity == "verbose":
            parts.append(f"You are in a {grid.width}×{grid.height} room.")

        ax, ay = agent.position

        if self.config.verbosity == "minimal":
            parts.append(f"Position: ({ax}, {ay}).")
        else:
            # Describe location in spatial terms
            rel_x = "left" if ax < grid.width // 2 else "right"
            rel_y = "top" if ay < grid.height // 2 else "bottom"

            if ax == grid.width // 2 and ay == grid.height // 2:
                parts.append("You are in the center of the room.")
            elif ax < grid.width // 3:
                parts.append(f"You are near the western edge of a {grid.width}×{grid.height} room.")
            elif ax > 2 * grid.width // 3:
                parts.append(f"You are near the eastern edge of a {grid.width}×{grid.height} room.")
            elif ay < grid.height // 3:
                parts.append(
                    f"You are near the northern edge of a {grid.width}×{grid.height} room."
                )
            elif ay > 2 * grid.height // 3:
                parts.append(
                    f"You are near the southern edge of a {grid.width}×{grid.height} room."
                )
            else:
                parts.append(
                    f"You are in the {rel_y}-{rel_x} area of a {grid.width}×{grid.height} room."
                )

        # Orientation
        direction_names = {
            Direction.NORTH: "north",
            Direction.EAST: "east",
            Direction.SOUTH: "south",
            Direction.WEST: "west",
        }
        parts.append(f"You are facing {direction_names[agent.orientation]}.")

        return parts

    def _describe_third_person(self, grid: Grid, agent: Agent) -> list[str]:
        """Third-person perspective description."""
        parts = []

        ax, ay = agent.position
        direction_names = {
            Direction.NORTH: "north",
            Direction.EAST: "east",
            Direction.SOUTH: "south",
            Direction.WEST: "west",
        }

        if self.config.verbosity == "minimal":
            parts.append(f"Agent at ({ax}, {ay}) facing {direction_names[agent.orientation]}.")
        else:
            room_info = f"The agent is located at position ({ax}, {ay})"
            room_size = f"in a {grid.width}×{grid.height} room"
            orientation = f"facing {direction_names[agent.orientation]}."
            parts.append(f"{room_info} {room_size}, {orientation}")

        return parts

    def _describe_omniscient(self, grid: Grid, agent: Agent, entities: list[Entity]) -> list[str]:
        """Omniscient perspective with full information."""
        parts = []

        ax, ay = agent.position

        parts.append(f"Grid: {grid.width}×{grid.height}. Agent at ({ax}, {ay}).")

        # Count entities
        goals = sum(
            1
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.GOAL
        )
        keys = sum(
            1
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.KEY
        )
        doors = sum(
            1
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.DOOR
        )

        entity_counts = []
        if goals > 0:
            entity_counts.append(f"{goals} goal(s)")
        if keys > 0:
            entity_counts.append(f"{keys} key(s)")
        if doors > 0:
            entity_counts.append(f"{doors} door(s)")
        if len(entities) > 0:
            entity_counts.append(f"{len(entities)} other entity(ies)")

        if entity_counts:
            parts.append(f"Contains: {', '.join(entity_counts)}.")

        return parts

    def _describe_surroundings_relative(
        self, grid: Grid, agent: Agent, entities: list[Entity]
    ) -> str:
        """Describe surroundings using relative directions and distances.

        Objects are ALWAYS included regardless of distance — an agent using
        text observations must see everything a pixel-observation agent sees.
        Terrain (walls, water) is only mentioned when nearby.
        """
        from agentick.core.annotations import extract_annotations

        ax, ay = agent.position

        # Build annotations for metadata-aware descriptions
        # (we need info dict for task_config; stash a minimal one)
        ann_info = getattr(self, "_last_info", {})
        ann = extract_annotations(grid, ann_info)

        # Compute the cell directly ahead for wall-blocking check
        fdx, fdy = agent.orientation.to_delta()
        faced_pos = (ax + fdx, ay + fdy)

        high: list[str] = []  # objects, hazards, entities
        low: list[str] = []  # walls (only blocking), water (close)

        # Terrain visibility range (walls/water are low-priority)
        terrain_range = 2

        # Scan the ENTIRE grid for objects (full parity with pixel mode)
        for ny in range(grid.height):
            for nx in range(grid.width):
                if nx == ax and ny == ay:
                    continue
                if not grid.in_bounds((nx, ny)):
                    continue

                pos = (nx, ny)

                # Skip fogged cells
                if pos in ann.fog_cells:
                    continue

                dx = nx - ax
                dy = ny - ay
                dist = abs(dx) + abs(dy)
                rel_dir = self._get_relative_direction(agent.orientation, dx, dy)
                dist_desc = f"{dist} steps"

                # --- Terrain (nearby only) ---
                if dist <= terrain_range:
                    terrain_type = CellType(grid.terrain[ny, nx])
                    if terrain_type == CellType.WALL and (nx, ny) == faced_pos:
                        low.append("a wall ahead (blocking)")
                    elif terrain_type == CellType.HAZARD:
                        high.append(f"a hazard {rel_dir} ({dist_desc})")
                    elif terrain_type == CellType.WATER:
                        low.append(f"water {rel_dir} ({dist_desc})")

                # --- Objects (FULL GRID scan) ---
                obj_type = ObjectType(grid.objects[ny, nx])
                if obj_type == ObjectType.NONE:
                    continue

                if obj_type == ObjectType.GOAL:
                    high.append(f"a goal {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.KEY:
                    color = ann.key_colors.get(pos, "gold")
                    high.append(f"a {color} key {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.DOOR:
                    color = ann.door_colors.get(pos, "gold")
                    state = ann.door_states.get(pos, "closed")
                    high.append(f"a {state} {color} door {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.BOX:
                    tile_num = ann.tile_numbers.get(pos)
                    if tile_num is not None:
                        high.append(f"tile {tile_num} {rel_dir} ({dist_desc})")
                    else:
                        high.append(f"a box {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.TARGET:
                    slot_num = ann.target_slots.get(pos)
                    if slot_num is not None:
                        high.append(f"target slot {slot_num} {rel_dir} ({dist_desc})")
                    else:
                        high.append(f"a target {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.SWITCH:
                    state = ann.switch_states.get(pos, "off")
                    color = ann.switch_colors.get(pos)
                    label = f"{color} " if color else ""
                    high.append(f"a {label}switch ({state}) {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.LEVER:
                    state = ann.lever_states.get(pos, "off")
                    high.append(f"a lever ({state}) {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.RESOURCE:
                    energy = ann.resource_energy.get(pos, 0)
                    if energy > 0:
                        high.append(
                            f"a resource station (energy: {energy}) "
                            f"{rel_dir} ({dist_desc})"
                        )
                    else:
                        high.append(
                            f"an empty resource station {rel_dir} ({dist_desc})"
                        )
                elif obj_type == ObjectType.NPC:
                    npc_type = ann.npc_types.get(pos, "npc")
                    high.append(f"a {npc_type} NPC {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.ENEMY:
                    enemy_type = ann.enemy_types.get(pos, "enemy")
                    high.append(f"a {enemy_type} enemy {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.SCROLL:
                    sd = ann.scroll_directions.get(pos)
                    if sd:
                        high.append(
                            f"a scroll (points {sd['direction']}, "
                            f"distance {sd['distance']}) {rel_dir} ({dist_desc})"
                        )
                    else:
                        high.append(f"a scroll {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.GEM:
                    high.append(f"a gem {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.ORB:
                    high.append(f"an orb {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.COIN:
                    high.append(f"a coin {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.POTION:
                    high.append(f"a potion {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.SHEEP:
                    high.append(f"a sheep {rel_dir} ({dist_desc})")
                elif obj_type == ObjectType.BLOCKER:
                    high.append(f"a blocker {rel_dir} ({dist_desc})")

        # Entities (NPCs/enemies not on grid objects layer)
        for entity in entities:
            ex, ey = entity.position
            dx = ex - ax
            dy = ey - ay
            dist = abs(dx) + abs(dy)
            rel_dir = self._get_relative_direction(agent.orientation, dx, dy)
            dist_desc = f"{dist} steps"
            etype = entity.entity_type
            pos = (ex, ey)
            npc_type = ann.npc_types.get(pos) or ann.enemy_types.get(pos)
            if npc_type:
                etype = f"{npc_type} {etype}"
            high.append(f"a {etype} {rel_dir} ({dist_desc})")

        # Combine: high-priority first, then low-priority
        descriptions = high + low

        if descriptions:
            if self.config.verbosity == "minimal":
                descriptions = descriptions[:5]
            return "You see: " + ", ".join(descriptions) + "."
        elif self.config.verbosity != "minimal":
            return "The area around you is empty."
        else:
            return ""

    def _get_relative_direction(self, orientation: Direction, dx: int, dy: int) -> str:
        """Get relative direction description based on agent orientation."""
        # Determine absolute direction
        if abs(dx) > abs(dy):
            abs_dir = "east" if dx > 0 else "west"
        elif abs(dy) > abs(dx):
            abs_dir = "south" if dy > 0 else "north"
        else:
            # Diagonal
            ns = "south" if dy > 0 else "north"
            ew = "east" if dx > 0 else "west"
            abs_dir = f"{ns}{ew}"

        # Convert to relative (ahead, behind, left, right)
        if orientation == Direction.NORTH:
            rel_map = {
                "north": "ahead",
                "south": "behind",
                "east": "to your right",
                "west": "to your left",
                "northeast": "ahead and to your right",
                "northwest": "ahead and to your left",
                "southeast": "behind and to your right",
                "southwest": "behind and to your left",
            }
        elif orientation == Direction.EAST:
            rel_map = {
                "east": "ahead",
                "west": "behind",
                "south": "to your right",
                "north": "to your left",
                "southeast": "ahead and to your right",
                "northeast": "ahead and to your left",
                "southwest": "behind and to your right",
                "northwest": "behind and to your left",
            }
        elif orientation == Direction.SOUTH:
            rel_map = {
                "south": "ahead",
                "north": "behind",
                "west": "to your right",
                "east": "to your left",
                "southwest": "ahead and to your right",
                "southeast": "ahead and to your left",
                "northwest": "behind and to your right",
                "northeast": "behind and to your left",
            }
        else:  # WEST
            rel_map = {
                "west": "ahead",
                "east": "behind",
                "north": "to your right",
                "south": "to your left",
                "northwest": "ahead and to your right",
                "southwest": "ahead and to your left",
                "northeast": "behind and to your right",
                "southeast": "behind and to your left",
            }

        return rel_map.get(abs_dir, abs_dir)

    def _get_distance_description(self, dist: int) -> str:
        """Convert Manhattan distance to descriptive term."""
        if dist == 1:
            return "1 step"
        elif dist == 2:
            return "2 steps"
        elif dist <= 4:
            return f"{dist} steps"
        else:
            return "far away"

    def _describe_inventory(self, agent: Agent) -> str:
        """Describe agent's inventory."""
        if agent.inventory:
            items = [item.entity_type for item in agent.inventory]
            if self.config.verbosity == "minimal":
                return f"Inventory: {len(items)} items."
            else:
                return f"You are carrying: {', '.join(items)}."
        else:
            if self.config.verbosity == "minimal":
                return "Inventory: empty."
            else:
                return "Your inventory is empty."

    def _describe_status(self, agent: Agent) -> str:
        """Describe agent status (energy, health)."""
        if agent.energy < 1.0 or agent.health < 1.0:
            if self.config.verbosity == "verbose":
                return f"Energy: {agent.energy:.0%}, Health: {agent.health:.0%}."
            else:
                status_parts = []
                if agent.energy < 0.3:
                    status_parts.append("low energy")
                if agent.health < 0.3:
                    status_parts.append("low health")
                if status_parts:
                    return f"Warning: {', '.join(status_parts)}."
        return ""

    def _describe_goals(self, grid: Grid, agent: Agent, info: dict[str, Any]) -> str:
        """Describe goals and objectives."""
        ax, ay = agent.position

        # Find nearest goal
        min_dist = float("inf")
        goal_pos = None

        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GOAL:
                    dist = abs(x - ax) + abs(y - ay)
                    if dist < min_dist:
                        min_dist = dist
                        goal_pos = (x, y)

        if goal_pos:
            dx = goal_pos[0] - ax
            dy = goal_pos[1] - ay
            rel_dir = self._get_relative_direction(agent.orientation, dx, dy)

            if self.config.verbosity == "verbose":
                steps = int(min_dist)
                return (
                    f"Your objective is to reach the goal. "
                    f"The nearest goal is {rel_dir}, approximately {steps} steps away."
                )
            else:
                return f"Goal: {rel_dir}, ~{int(min_dist)} steps."
        return ""

    def _describe_threats(self, grid: Grid, agent: Agent, entities: list[Entity]) -> str:
        """Describe nearby threats or dangers."""
        ax, ay = agent.position
        threats = []

        # Check for hazards nearby
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = ax + dx, ay + dy
                if not grid.in_bounds((nx, ny)):
                    continue

                if grid.terrain[ny, nx] == CellType.HAZARD:
                    dist = abs(dx) + abs(dy)
                    rel_dir = self._get_relative_direction(agent.orientation, dx, dy)
                    threats.append(f"hazard {rel_dir} ({dist} step{'s' if dist > 1 else ''})")

        # Check for enemy entities
        for entity in entities:
            if entity.entity_type == "enemy":
                ex, ey = entity.position
                dx = ex - ax
                dy = ey - ay
                dist = abs(dx) + abs(dy)

                if dist <= 3:
                    rel_dir = self._get_relative_direction(agent.orientation, dx, dy)
                    threats.append(f"enemy {rel_dir} ({dist} step{'s' if dist > 1 else ''})")

        if threats:
            if self.config.verbosity == "verbose":
                return f"Warning: Detected {', '.join(threats)}."
            else:
                return f"Threat: {threats[0]}."
        return ""

    def _describe_memory(self, agent: Agent) -> str:
        """Describe relevant memories (visited locations, past events)."""
        # For now, return empty - this requires trajectory tracking
        # Will be implemented when episode recorder is added
        return ""

    def _render_structured(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Render as structured JSON-like dict with full metadata parity."""
        from agentick.core.annotations import extract_annotations

        ax, ay = agent.position
        ann = extract_annotations(grid, info)

        # Collect visible objects with metadata interpretation
        visible_entities = []
        for y in range(grid.height):
            for x in range(grid.width):
                pos = (x, y)
                if pos in ann.fog_cells:
                    continue
                obj_type = ObjectType(grid.objects[y, x])
                if obj_type == ObjectType.NONE:
                    continue
                entry: dict[str, Any] = {
                    "type": obj_type.name.lower(),
                    "position": [x, y],
                    "distance": abs(x - ax) + abs(y - ay),
                }
                # Enrich with metadata from annotations
                if pos in ann.key_colors:
                    entry["color"] = ann.key_colors[pos]
                if pos in ann.door_colors:
                    entry["color"] = ann.door_colors[pos]
                if pos in ann.door_states:
                    entry["state"] = ann.door_states[pos]
                if pos in ann.switch_states:
                    entry["state"] = ann.switch_states[pos]
                if pos in ann.switch_colors:
                    entry["color"] = ann.switch_colors[pos]
                if pos in ann.lever_states:
                    entry["state"] = ann.lever_states[pos]
                if pos in ann.npc_types:
                    entry["behavior"] = ann.npc_types[pos]
                if pos in ann.enemy_types:
                    entry["behavior"] = ann.enemy_types[pos]
                if pos in ann.resource_energy:
                    entry["energy"] = ann.resource_energy[pos]
                if pos in ann.tile_numbers:
                    entry["tile_number"] = ann.tile_numbers[pos]
                if pos in ann.target_slots:
                    entry["slot_number"] = ann.target_slots[pos]
                if pos in ann.scroll_directions:
                    sd = ann.scroll_directions[pos]
                    entry["direction"] = sd["direction"]
                    entry["clue_distance"] = sd["distance"]
                visible_entities.append(entry)

        # Add other entities (NPCs not on grid objects layer)
        for entity in entities:
            ex, ey = entity.position
            entry = {
                "type": entity.entity_type,
                "position": [ex, ey],
                "distance": abs(ex - ax) + abs(ey - ay),
            }
            pos = (ex, ey)
            if pos in ann.npc_types:
                entry["behavior"] = ann.npc_types[pos]
            if pos in ann.enemy_types:
                entry["behavior"] = ann.enemy_types[pos]
            visible_entities.append(entry)

        goals = [e for e in visible_entities if e["type"] == "goal"]
        threats = [e for e in visible_entities if e["type"] in ("hazard", "enemy")]

        # Task-specific annotations
        task_annotations: dict[str, Any] = {}
        if ann.target_object:
            task_annotations["target_object"] = ann.target_object
        if ann.gem_meter is not None:
            task_annotations["gem_meter"] = ann.gem_meter
            task_annotations["orb_meter"] = ann.orb_meter
            task_annotations["meter_threshold"] = ann.meter_threshold
        if ann.trial_info:
            task_annotations["trial_info"] = ann.trial_info
        if ann.phase_info:
            task_annotations["phase_info"] = ann.phase_info
        if ann.clues:
            task_annotations["clues"] = ann.clues
        if ann.lights_out:
            task_annotations["lights_out"] = True

        return {
            "description": f"A {grid.width}x{grid.height} gridworld environment",
            "position": {"x": ax, "y": ay},
            "orientation": agent.orientation.name.lower(),
            "visible_entities": visible_entities,
            "inventory": [item.entity_type for item in agent.inventory],
            "energy": agent.energy,
            "health": agent.health,
            "goals": goals,
            "threats": threats,
            "task_annotations": task_annotations,
            "valid_actions": info.get("valid_actions", []),
            "step_count": info.get("step_count", 0),
            "max_steps": info.get("max_steps", 0),
        }

    def update_memory(self, position: tuple[int, int], event: str):
        """
        Update memory with visited location or event.

        Args:
            position: Position where event occurred
            event: Description of event
        """
        self.memory.append(
            {
                "position": position,
                "event": event,
                "timestamp": len(self.memory),
            }
        )

        # Keep memory bounded
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def clear_memory(self):
        """Clear memory (e.g., at episode reset)."""
        self.memory.clear()
