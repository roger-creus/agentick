"""Structured task descriptions for all Agentick tasks.

Provides rich, structured descriptions with objects, goals, actions, and
per-difficulty summaries.  Used by LLM agent system prompts via
``get_task_description()`` and by the showcase webapp.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DifficultyDescription:
    """Per-difficulty description (20-30 words)."""

    level: str
    description: str


@dataclass
class TaskDescription:
    """Rich structured description of an Agentick task."""

    name: str
    category: str
    summary: str
    objects: str
    goal: str
    actions: str
    difficulty_descriptions: list[DifficultyDescription] = field(default_factory=list)
    # Kept for backward compat with get_all_task_descriptions()
    capability_tags: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def description(self) -> str:
        return self.summary

    @property
    def detailed_description(self) -> str:
        return self.to_prompt_text()

    def to_prompt_text(self) -> str:
        """Format as rich text for LLM system prompts."""
        lines = [self.summary]
        lines.append(f"\nObjects: {self.objects}")
        lines.append(f"Goal: {self.goal}")
        lines.append(f"Actions: {self.actions}")
        if self.difficulty_descriptions:
            lines.append("\nDifficulty levels:")
            for dd in self.difficulty_descriptions:
                lines.append(f"- {dd.level}: {dd.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured description store — one entry per registered task
# ---------------------------------------------------------------------------

_TASK_DESCRIPTIONS: dict[str, TaskDescription] = {}


def _td(
    name: str,
    category: str,
    summary: str,
    objects: str,
    goal: str,
    actions: str,
    diffs: list[tuple[str, str]],
    tags: list[str] | None = None,
) -> None:
    """Register a structured task description."""
    _TASK_DESCRIPTIONS[name] = TaskDescription(
        name=name,
        category=category,
        summary=summary,
        objects=objects,
        goal=goal,
        actions=actions,
        difficulty_descriptions=[
            DifficultyDescription(level=d[0], description=d[1]) for d in diffs
        ],
        capability_tags=tags or [],
        difficulties=[d[0] for d in diffs],
    )


# ── NAVIGATION ────────────────────────────────────────────────────────────

_td(
    "GoToGoal-v0",
    "navigation",
    "Navigate a grid to reach a visible GOAL while avoiding walls, hazards, "
    "and patrolling guard NPCs.",
    "GOAL (destination), WALL (obstacles), HAZARD (lava, fatal), Guard NPCs "
    "(patrolling, collision penalty), Decoy GOALs (false targets).",
    "Reach the GOAL position. Stepping on HAZARD or colliding with a guard "
    "ends the episode.",
    "Move in 4 directions only (actions 1-4). Walk onto the GOAL cell to complete the task. "
    "Stepping on HAZARD or colliding with a guard NPC ends the episode. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "5x5 open grid, no walls, guards, or hazards. Straightforward path to goal."),
        ("medium", "10x10 grid, 1 guard, no decoys. Obstacle avoidance needed."),
        ("hard", "15x15 maze, 2 guards, 2 decoys, 4 hazards. Complex navigation."),
        ("expert", "20x20 complex maze with 3 guards, 4 decoys, and 8 hazards. Maximum challenge."),
    ],
    tags=["basic_navigation", "navigation"],
)

_td(
    "MazeNavigation-v0",
    "navigation",
    "Solve procedurally generated mazes (binary tree or recursive backtracker) "
    "to reach the exit GOAL. At medium+, colored key/door pairs block the optimal path "
    "at choke points and must be collected/opened. At hard+, hazard terrain appears in "
    "dead ends. No guard NPCs.",
    "GOAL (maze exit), WALL (maze structure), KEY objects (collect to unlock doors, "
    "medium+), DOOR (blocks path, opens with matching key, medium+), HAZARD (dead ends, hard+).",
    "Navigate the maze to reach the GOAL exit. Collect keys and open doors blocking the path.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Walk onto a KEY "
    "cell to auto-collect it. DOORs are solid — move toward a DOOR to face it "
    "(orientation updates even when blocked), then INTERACT (action 5) to unlock with "
    "the matching-color key. Walk through the opened doorway. Walk onto the GOAL cell "
    "(maze exit) to complete the task.",
    [
        ("easy", "7x7 binary-tree maze. No keys or hazards. Simple pathfinding exercise."),
        ("medium", "11x11 recursive-backtracker maze, 1 key-door pair blocking the path."),
        ("hard", "15x15 maze, 2 key-door pairs, hazards in dead ends."),
        ("expert", "21x21 dense recursive maze, 3 key-door pairs, hazards. Maximum branching."),
    ],
    tags=["planning", "spatial_reasoning", "navigation"],
)

_td(
    "ShortestPath-v0",
    "navigation",
    "Visit multiple GOAL objects scattered across the grid in any order. "
    "Tests multi-objective pathfinding similar to the traveling salesman problem. "
    "Success requires visiting all goals within a step budget relative to the optimal path length.",
    "GOAL objects (multiple, consumed on visit), WALL (obstacles), "
    "TARGET objects (decoy false targets, medium+).",
    "Visit all real GOAL objects within the step budget (optimal path × budget multiplier).",
    "Move in 4 directions only (actions 1-4). Walk onto a GOAL cell to auto-collect it "
    "(goals are consumed on visit). Avoid stepping on decoy TARGET cells. Visit all real "
    "GOAL objects within the step budget to win. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7 open grid with 2 goals and no obstacles. Simple multi-stop navigation."),
        ("medium", "10x10, 3 goals, 3 obstacles, 1 decoy. Route planning needed."),
        ("hard", "13x13, 4 goals, 5 obstacles, 2 decoys. Complex routing."),
        ("expert", "15x15 maze-like grid with 5 goals, 8 obstacles, and 3 decoy TARGETs."),
    ],
    tags=["planning", "optimization", "navigation"],
)

_td(
    "DynamicObstacles-v0",
    "navigation",
    "Navigate toward a GOAL while avoiding moving NPC obstacles that patrol "
    "the grid. NPCs move probabilistically each step; expert mode adds pursuit behavior.",
    "GOAL (static destination), NPC objects (moving obstacles, fatal on collision), "
    "WALL (static barriers).",
    "Reach GOAL without colliding with any NPC. Collision ends episode.",
    "Move in 4 directions only (actions 1-4). Walk onto the GOAL cell to complete the task. "
    "Stepping on or colliding with any NPC ends the episode immediately. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7 grid with 2 NPCs (50% move chance). Open layout, easy avoidance."),
        ("medium", "10x10 grid with 3 NPCs (75% move chance) and 3 scattered walls."),
        ("hard", "13x13 grid with 5 NPCs (100% move) and 6 wall obstacles."),
        ("expert", "15x15 grid with 7 NPCs (90% move, 15% pursuit), 9 walls. Pursuing behavior."),
    ],
    tags=["reactive_planning", "navigation"],
)

_td(
    "FogOfWarExploration-v0",
    "memory",
    "Navigate a grid where fog never clears — only the agent's immediate neighbors "
    "(4 orthogonal cells) are visible at any time. The agent must remember the map "
    "layout to find the hidden GOAL.",
    "GOAL (hidden behind fog), WALL (revealed only when adjacent), HAZARD (dangerous), "
    "Guard NPCs (hard+), Decoy GOALs (medium+), FOG (all non-adjacent cells).",
    "Find and reach the GOAL using spatial memory — previously seen cells re-fog.",
    "Move in 4 directions only (actions 1-4). Walk onto the GOAL cell to complete the task. "
    "Only your 4 orthogonal neighbor cells are visible; everything else is hidden by fog. "
    "Previously seen cells re-fog when you move away — you must memorize the layout. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7, no hazards/guards/decoys. Build spatial memory of layout."),
        ("medium", "10x10 grid, 2 decoy goals. Must remember which goals are real."),
        ("hard", "13x13 grid, 1 guard, 3 decoys. Navigate blind with hostile NPCs."),
        ("expert", "15x15 grid, 2 guards, 4 decoys. Extreme spatial memory demand."),
    ],
    tags=["memory", "spatial_memory"],
)

_td(
    "TreasureHunt-v0",
    "memory",
    "Find invisible hidden treasures by reading directional SCROLL clues. "
    "Scrolls encode the direction (N/E/S/W) and distance to the nearest "
    "hidden treasure as metadata (direction * 10 + distance). Stepping on "
    "a scroll reads and consumes it. Some clues are misleading at harder "
    "difficulties.",
    "SCROLL objects (directional clues, consumed on contact), invisible "
    "treasures (not rendered — step on exact cell to collect), WALL (obstacles).",
    "Read scroll clues, triangulate hidden treasure positions, and step on "
    "each treasure cell to collect all treasures.",
    "Move in 4 directions only (actions 1-4). Walk onto a SCROLL cell to read and consume "
    "it (the scroll clue tells you the direction and distance to a treasure). Walk onto "
    "a hidden treasure cell to collect it (treasures are invisible — you must triangulate "
    "from scroll clues). No INTERACT action needed — all interactions "
    "are automatic on step.",
    [
        ("easy", "9x9 grid, 2 treasures, 6 scrolls, 0 misleading. Simple triangulation."),
        ("medium", "12x12 grid, 3 treasures, 8 scrolls, 1 misleading. Must cross-check clues."),
        ("hard", "15x15 grid, 4 treasures, 10 scrolls, 3 misleading. Significant deception."),
        ("expert", "18x18 grid, 5 treasures, 12 scrolls, 5 misleading. Nearly half the clues lie."),
    ],
    tags=["exploration", "memory", "reasoning"],
)

_td(
    "CuriosityMaze-v0",
    "navigation",
    "Explore a procedurally generated maze and visit a target percentage of all "
    "reachable cells within a step budget. No targets or landmarks — success is "
    "purely coverage-based. The agent must remember where it has been.",
    "WALL (border and random interior walls forming maze structure). No target "
    "objects or landmarks on the grid.",
    "Visit at least the required percentage of all reachable cells before the "
    "step budget runs out.",
    "Move in 4 directions only (actions 1-4). Each new cell you step on counts toward "
    "coverage. There are no visual markers for visited vs unvisited cells — you must "
    "track your own path from memory. No INTERACT action needed "
    "— only movement.",
    [
        ("easy", "9x9 maze, 70% coverage needed, 80 steps. Low wall density."),
        ("medium", "13x13 maze, 75% coverage needed, 150 steps. Moderate wall density."),
        ("hard", "17x17 maze, 80% coverage needed, 280 steps. High wall density."),
        ("expert", "21x21 maze, 85% coverage needed, 450 steps. Dense complex maze."),
    ],
    tags=["exploration", "memory", "navigation"],
)

# ── MEMORY / PLANNING ────────────────────────────────────────────────────

_td(
    "KeyDoorPuzzle-v0",
    "planning",
    "Collect color-coded keys and unlock ALL matching doors to reach the GOAL in the "
    "innermost room. One-key-at-a-time inventory limit: picking up a new key drops any "
    "held key. At hard+, keys are placed in previous rooms, forcing backtracking "
    "through already-opened doors.",
    "KEY objects (color-coded: gold/red/blue/green, one-at-a-time inventory), DOOR "
    "(opens with matching-color key), GOAL (in final room behind ALL doors), Guard NPCs "
    "(hard+), WALL (room separators and corridors).",
    "Reach GOAL after unlocking ALL doors with matching keys. Every door must be opened.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Walk onto a KEY "
    "cell to auto-pick it up (one key at a time — picking up a new key drops any "
    "currently held key). DOORs are solid — move toward a DOOR to face it (orientation "
    "updates even when blocked), then INTERACT (action 5) to unlock with the matching-"
    "color key. Walk through the opened doorway. Walk onto the GOAL cell after all "
    "doors are unlocked to win.",
    [
        ("easy", "9x9 grid with 1 color-coded key-door pair. No guards. Key in hub, "
         "goal behind the single door."),
        ("medium", "11x11 grid with 2 color-coded key-door pairs. No guards. "
         "Key for door 2 is behind door 1, forcing backtracking."),
        ("hard", "13x13 grid with 3 chained color-coded locks, 1 guard. "
         "Keys placed in previous rooms, forcing backtracking."),
        ("expert", "15x15 grid with 4 chained color-coded locks, 2 guards. "
         "Complex key dependencies with keys in earlier rooms requiring multi-room backtracking."),
    ],
    tags=["memory", "sequential_reasoning"],
)

_td(
    "DelayedGratification-v0",
    "memory",
    "Resist nearby decoy KEY objects (traps) and navigate to a distant true GOAL. "
    "Collecting a decoy KEY blocks future success and ends the episode. Tests impulse control.",
    "GOAL (true objective, distant), KEY objects (decoy traps, closer to start), "
    "WALL (optional obstacles), HAZARD (hard+, blocks paths).",
    "Reach GOAL without collecting decoy KEYs. Collecting a decoy ends episode.",
    "Move in 4 directions only (actions 1-4). Walk onto the GOAL cell to complete the task. "
    "KEYs are auto-collected when you step on them — collecting any KEY ends the episode "
    "in failure. Plan your path to avoid all KEY cells. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7 grid with 2 decoy KEYs. Open layout, clear path to goal."),
        ("medium", "10x10 grid with 4 decoy KEYs, 4 walls. Must navigate around traps."),
        ("hard", "13x13 maze with 6 decoys, 8 walls, 3 hazards. Hazards block shortcuts."),
        ("expert", "15x15 maze, 8 decoys, 12 walls, 6 hazards. Careful planning."),
    ],
    tags=["credit_assignment", "long_horizon"],
)

_td(
    "BacktrackPuzzle-v0",
    "planning",
    "GOAL is blocked by a wall gate. Navigate past it to reach a distant SWITCH, "
    "which opens the gate, then backtrack to collect the now-accessible GOAL.",
    "GOAL (initially blocked), SWITCH objects (trigger gate opening), "
    "WALL (gate barrier, opens after switch), dead-end paths.",
    "Activate the correct SWITCH to open the gate, then backtrack to reach GOAL.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Switches are solid "
    "— move toward a SWITCH to stand adjacent and face it (orientation updates even when "
    "blocked), then INTERACT (action 5) to activate it. After all switches are activated, "
    "the wall gate opens and the GOAL appears — walk onto the GOAL cell to win. "
    "Only actions 1-4 (movement) and 5 (INTERACT) are useful in this task.",
    [
        ("easy", "9x9 L-shaped layout with 1 switch. No dead ends. Clear backtrack path."),
        ("medium", "11x11 T-shaped layout with 2 switches and 1 dead end."),
        ("hard", "13x13 zigzag layout with 3 switches and 2 dead ends."),
        ("expert", "15x15 complex layout with 4 switches and 3 dead ends. Multiple gates."),
    ],
    tags=["memory", "planning"],
)

_td(
    "SequenceMemory-v0",
    "memory",
    "Two-phase task: show phase displays target positions as GEM objects one by one. "
    "Reproduce phase requires visiting those memorized positions in exact order. "
    "No visual markers during reproduce — pure spatial memory.",
    "GEM objects (shown during show phase, then removed), distractor positions "
    "(invalid positions, medium+, no visual marker — tracked internally only).",
    "Memorize shown GEM positions, then visit them in exact order during reproduce phase.",
    "Move in 4 directions only (actions 1-4). During the show phase, observe which cells "
    "display GEM objects and memorize their positions and order. During the reproduce phase, "
    "walk onto each memorized position in the exact order they were shown. Positions are "
    "auto-registered when you step on them. No INTERACT action "
    "needed — all interactions are automatic on step.",
    [
        ("easy", "7x7 grid, 3 targets shown for 4 steps each. No distractors. Short sequence."),
        ("medium", "10x10, 4 targets (3 steps each), 2 distractors, 4 obstacles."),
        ("hard", "13x13 grid, 5 targets shown for 2 steps each, 3 distractors, shuffled order."),
        ("expert", "15x15 grid, 6 targets shown for 1 step each, 4 distractors, shuffled order."),
    ],
    tags=["memory", "pattern_recognition"],
)

# ── REASONING ─────────────────────────────────────────────────────────────

_td(
    "SokobanPush-v0",
    "planning",
    "Classic Sokoban: push BOX objects onto TARGET positions by walking into them. "
    "Boxes move in the direction the agent pushes. Cannot push into walls.",
    "BOX objects (pushable), TARGET positions (where boxes must go), "
    "WALL (immovable obstacles).",
    "Push all BOX objects onto matching TARGET positions.",
    "Move in 4 directions only (actions 1-4). Walk into a BOX cell to push it one cell "
    "in your movement direction. Cannot push boxes into walls or other boxes. "
    "No INTERACT action needed — pushing is automatic on movement.",
    [
        ("easy", "7x7 grid with 1 box and 1 target. Open layout, no dead-end traps."),
        ("medium", "10x10 grid with 2 boxes, 2 targets, 3 obstacles creating push constraints."),
        ("hard", "13x13 grid with 3 boxes, 3 targets, 5 obstacles, 3 hazards."),
        ("expert", "15x15 grid with 4 boxes, 4 targets, 8 obstacles, 5 hazards. Complex mazes."),
    ],
    tags=["reasoning", "planning"],
)

_td(
    "SwitchCircuit-v0",
    "reasoning",
    "Non-linear switch dependency puzzle. An open arena has N switches toggled via "
    "INTERACT action, each controlling wall-segment barriers. Switches have mutual "
    "exclusion dependencies: activating one may deactivate another. No number annotations "
    "on switches — the agent must discover dependencies through experimentation.",
    "SWITCH objects (color-coded, toggled via INTERACT), WALL barrier segments "
    "(1-3 cells, metadata = barrier index), GOAL (behind final barrier).",
    "Plan switch activation order to open all barriers blocking the path to GOAL.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Switches are solid "
    "— move toward a SWITCH to stand adjacent and face it (orientation updates even when "
    "blocked), then INTERACT (action 5) to toggle it ON/OFF. Mutual exclusion: toggling "
    "one switch may deactivate another. "
    "Only actions 1-4 (movement) and 5 (INTERACT) are useful in this task.",
    [
        ("easy", "9x9 grid, 2 switches. Simple dependency: A opens path to B, B opens goal."),
        ("medium", "11x11 grid, 3 switches. One mutual exclusion requires planning order."),
        ("hard", "13x13 grid, 4 switches. Multiple mutual exclusions, backtracking needed."),
        ("expert", "15x15 grid, 5 switches. Complex dependency graph with mutual exclusions."),
    ],
    tags=["combinatorial_logic", "reasoning"],
)

_td(
    "RuleInduction-v0",
    "reasoning",
    "XLand-style combination discovery. Objects (GEM, POTION, SCROLL, COIN, ORB) are "
    "scattered on the grid. Hidden rules map pairs of objects to result objects. Pick up "
    "one object, walk onto another to combine. Valid combos produce a result; invalid "
    "combos destroy both objects. Craft the TARGET object shown on screen. Episode has "
    "N resetting trials — learn from failed combinations across trials.",
    "Various objects (GEM, POTION, SCROLL, COIN, ORB), hidden combination rules, "
    "TARGET indicator (bottom-right), trial counter.",
    "Discover combination rules through experimentation and craft the target object.",
    "Move in 4 directions (actions 1-4). Walk onto an object to pick it up (one item "
    "at a time). While carrying an item, walk onto another object to combine them. If "
    "the combination is valid, a result object appears. If invalid, both objects are "
    "destroyed. Craft and collect the target object to succeed. Each trial resets the "
    "grid — use knowledge from failed trials to plan better combinations.",
    [
        ("easy", "9x9 grid, 6 objects, 2 valid combos, 5 trials."),
        ("medium", "11x11 grid, 8 objects, 3 valid combos, 4 trials."),
        ("hard", "13x13 grid, 10 objects, 4 valid combos, 3 trials."),
        ("expert", "15x15 grid, 12 objects, 5 valid combos, 2 trials."),
    ],
    tags=["reasoning", "rule_learning"],
)

_td(
    "SymbolMatching-v0",
    "reasoning",
    "Match symbol items (GEM, POTION, SCROLL, COIN, ORB, LEVER) to their matching "
    "targets of the same type. Items are on the left half; matching targets (same "
    "ObjectType) are on the right half. Pick up an item and deliver it to the matching "
    "target. Placing on wrong type = mismatch penalty.",
    "Symbol items on left side (GEM/POTION/SCROLL/COIN/ORB/LEVER), matching targets "
    "on right side (same ObjectType as their item), fake items (type with no matching "
    "target, medium+).",
    "Pick up each symbol item and deliver it to the matching target of the same type "
    "on the right side of the grid.",
    "Move in 4 directions only (actions 1-4). Walk onto a symbol item cell to auto-pick "
    "it up (one at a time). Walk onto a matching target cell (same symbol type) to deliver. "
    "Placing on wrong type = mismatch penalty. No INTERACT action "
    "needed — all interactions are automatic on step.",
    [
        ("easy", "7x7 grid with 2 symbol-target pairs. No fake items. Clear pairing."),
        ("medium", "10x10 grid with 3 pairs, 1 fake item, 3 obstacles."),
        ("hard", "13x13 maze with 4 pairs, 2 fake items, 5 obstacles."),
        ("expert", "15x15 grid with 5 pairs, 3 fake items, 8 obstacles."),
    ],
    tags=["reasoning", "pattern_recognition"],
)

# ── PLANNING (cont.) / MULTI-AGENT ────────────────────────────────────────

_td(
    "RecipeAssembly-v0",
    "planning",
    "Follow a recipe: collect ingredients in exact order and deliver each to the "
    "crafting station. Recipe is displayed visually in corner zone using TARGET markers.",
    "Recipe zone (TARGET objects with ingredient type metadata), ingredient items "
    "(GEM, SCROLL, ORB, COIN), crafting station (NPC object), decoy ingredients, WALL.",
    "Collect and deliver all ingredients in recipe order to the crafting station.",
    "Move in 4 directions only (actions 1-4). Walk onto an ingredient cell to auto-pick "
    "it up (one at a time). Walk onto the crafting station (NPC) cell to deliver the held "
    "ingredient. Ingredients must be delivered in recipe order (left-to-right in the recipe "
    "zone). No INTERACT action needed — all interactions are "
    "automatic on step.",
    [
        ("easy", "9x9 grid, 2-ingredient recipe. No decoys, open layout. Simple collect-deliver."),
        ("medium", "11x11, 3-ingredient recipe, 2 decoys, 3 obstacles. Route planning."),
        ("hard", "14x14 maze, 4-ingredient recipe, 3 decoys, 5 obstacles."),
        ("expert", "16x16 dense maze, 5-ingredient recipe, 4 decoys, 7 obstacles."),
    ],
    tags=["compositional_logic", "planning"],
)

_td(
    "ToolUse-v0",
    "planning",
    "GOAL is behind a horizontal river (WATER band). Agent can cross the river but only "
    "gets partial reward without the ORB tool. SCROLL objects are scattered on the agent's "
    "side. Collecting ALL required scrolls spawns an ORB. Picking up the ORB and then "
    "reaching the GOAL gives full reward. COIN decoys look different and have no effect.",
    "SCROLL (collectible clue), COIN (decoy, no effect), ORB (spawns after all scrolls "
    "collected), WATER (river band), GOAL (on far side of river).",
    "Collect all SCROLLs to spawn the ORB, pick up the ORB, cross the river, "
    "and reach the GOAL for full reward (1.0). Without the ORB, reward is only 0.2.",
    "Move in 4 directions only (actions 1-4). Walk onto SCROLL, COIN, or ORB cells to "
    "auto-collect them. The ORB only appears after all scrolls are collected. Collect "
    "the ORB, then cross the WATER river and walk onto the GOAL cell. Without the ORB, "
    "reaching GOAL gives only partial reward (0.2). No INTERACT action "
    "needed — all pickups are automatic on step.",
    [
        ("easy", "9x9 grid, 2 scrolls, 0 decoys, 1 river (2 cells wide)."),
        ("medium", "11x11 grid, 3 scrolls, 1 decoy, 1 river (3 cells wide)."),
        ("hard", "13x13 grid, 4 scrolls, 2 decoys, 2 river crossings (2 cells wide)."),
        ("expert", "15x15 grid, 5 scrolls, 3 decoys, 2 river crossings (3 cells wide)."),
    ],
    tags=["tool_use", "discovery"],
)

_td(
    "ResourceManagement-v0",
    "planning",
    "Keep energy stations alive by recharging them before they drain to zero. "
    "Stations drain energy over time. Any station dying ends the episode in failure. "
    "Success = surviving all max_steps.",
    "RESOURCE stations (energy 0-100, drain over time), WALL (obstacles between stations).",
    "Keep ALL stations above 0 energy for the entire episode (survive max_steps). "
    "Any station reaching 0 = failure.",
    "Move in 4 directions only (actions 1-4). Walk onto a RESOURCE station cell to "
    "recharge it to full energy (100). You can step on the same station multiple times. "
    "Monitor all stations and revisit them before they drain to 0. "
    "No INTERACT action needed — recharging is automatic on step.",
    [
        ("easy", "9x9 grid, 2 stations, slow drain (every 8 steps). No obstacles."),
        ("medium", "12x12 grid, 3 stations, moderate drain (every 5 steps), 3 obstacles."),
        ("hard", "15x15 grid, 4 stations, fast drain (every 3 steps), 5 obstacles."),
        ("expert", "18x18, 5 stations, fast drain (every 2 steps), 7 obstacles."),
    ],
    tags=["planning", "resource_allocation"],
)

_td(
    "EmergentStrategy-v0",
    "multi_agent",
    "GOAL is behind horizontal WALL barriers opened by locking pressure plates (TARGET). "
    "Four color-coded NPC types with unified discoverable behaviors: Follower (cyan, BFS "
    "toward cell behind agent), Fearful (green, flees from agent within distance 2), "
    "Mirror (purple, moves toward mirrored agent position), Contrarian (gold, moves "
    "opposite to agent's last action). All NPC types are controllable/influenceable by "
    "the agent's movement. When an NPC steps on a plate, it LOCKS permanently and the "
    "barrier stays open.",
    "TARGET (locking pressure plates, metadata stores barrier index), WALL barriers "
    "(full-width horizontal rows, open permanently when plate locked), color-coded NPCs "
    "(Follower/Fearful/Contrarian/Mirror), GOAL (behind last barrier).",
    "Exploit NPC behaviors to lure/scare them onto locking pressure plates, permanently "
    "opening barriers to reach the GOAL in the final zone.",
    "Move in 4 directions only (actions 1-4). Cannot walk through NPCs. Follower (cyan) stays "
    "1 tile behind you; Fearful (green) flees from you; Mirror (purple) mirrors your position; "
    "Contrarian (gold) moves opposite to your last action. When an NPC steps onto a TARGET "
    "plate, it locks permanently and opens the corresponding barrier. You do NOT interact with "
    "plates directly — guide NPCs onto them by exploiting their behaviors. Walk onto the GOAL "
    "cell after all barriers are open. No INTERACT action needed "
    "— only movement.",
    [
        ("easy", "9x9 grid, 1 locking plate, 1 follower. Single barrier."),
        ("medium", "12x12 grid, 2 locking plates, 1 follower + 1 fearful. Two barriers."),
        ("hard", "15x15 grid, 3 locking plates, 1 follower + 1 fearful + 1 contrarian. "
         "Three barriers."),
        ("expert", "18x18 grid, 4 locking plates, 1 follower + 1 fearful + 1 contrarian "
         "+ 1 mirror. Four barriers."),
    ],
    tags=["skill_composition", "long_horizon"],
)

# ── PLANNING (cont.) / MULTI-AGENT (cont.) / NAVIGATION (cont.) ──────────

_td(
    "PreciseNavigation-v0",
    "planning",
    "Ice sliding puzzle on larger grids with interior L/T/I-shaped wall segments. "
    "The interior is filled with ICE terrain; the agent slides until hitting a WALL, "
    "an EMPTY cell (stopping point), or a BOX. Fewer stopping points and minimum "
    "solution path length validation ensure non-trivial slide sequences. "
    "BOX objects at hard+ also slide when pushed.",
    "ICE (slides across), EMPTY (stops), WALL (segments), GOAL (on EMPTY), "
    "BOX (pushable, slides on ice at hard+).",
    "Slide across ice to reach the GOAL by planning trajectories through stopping points.",
    "Move in 4 directions only (actions 1-4). On ICE terrain, choosing a direction makes "
    "you slide continuously in that direction until you hit a WALL, an EMPTY cell, or a BOX "
    "— you cannot stop mid-slide. On EMPTY terrain, movement is normal (one cell per action). "
    "Push BOX objects (hard+) by sliding into them — they also slide on ice. Walk onto the "
    "GOAL cell to win. No INTERACT action needed — only movement.",
    [
        ("easy", "9x9, fewer stops, wall segments. 3-4 slides to goal."),
        ("medium", "11x11 grid, fewer stops, L/T-shaped walls. 5-6 slides."),
        ("hard", "13x13 grid, sparse stopping points, 1 pushable box. 7-9 slides."),
        ("expert", "15x15 grid, minimal stopping points, 2 pushable boxes. 10-14 slides."),
    ],
    tags=["motor_control", "planning"],
)

_td(
    "Herding-v0",
    "multi_agent",
    "Herd NPC SHEEP into a pen zone (TARGET cells) using movement pressure. Sheep "
    "flee from the agent when within distance 2. Predators scatter sheep at hard+.",
    "SHEEP NPCs (flee from agent), pen zone (TARGET cells), leader sheep (hard+, "
    "influence herd), ENEMY predators (scatter sheep at hard+), WALL (obstacles).",
    "Move all SHEEP into the pen zone (TARGET cells).",
    "Move in 4 directions only (actions 1-4). Sheep flee from you when you are within "
    "distance 2 — approach from the opposite side to push them toward the pen (TARGET cells). "
    "You cannot walk through NPCs. All sheep must be on TARGET cells simultaneously to win. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "9x9 grid, 2 sheep, 2x2 pen. No obstacles or predators. Simple herding."),
        ("medium", "12x12 grid, 3 sheep, 3x3 pen, 2 obstacles, random pen corner."),
        ("hard", "15x15 grid, 4 sheep, 3x3 pen, 4 obstacles, 1 predator, 1 leader sheep."),
        ("expert", "18x18 grid, 5 sheep, 3x3 pen, 6 obstacles, 2 predators, 2 leader sheep."),
    ],
    tags=["multi_objective_control"],
)

_td(
    "ChaseEvade-v0",
    "multi_agent",
    "Survive against a coordinated ENEMY pack with 4 behavior types inspired by Pacman "
    "ghosts. Chaser (BFS pursuit every step), Ambusher (targets 4 tiles ahead of agent "
    "to cut off escape), Flanker (pincers from opposite side of agent vs nearest ally), "
    "Trapper (moves to block agent's best escape routes). "
    "SWITCH power-ups freeze all enemies for 5 steps. Success = surviving all required steps.",
    "ENEMY demons (4 behavior types distinguished by metadata), SWITCH objects (freeze "
    "all enemies for 5 steps, one-time use), WALL (obstacles).",
    "Survive the required steps without enemy collision.",
    "Move in 4 directions only (actions 1-4). Step on a SWITCH cell to freeze ALL enemies "
    "for 5 steps (one-time use, auto-activates on step). Avoid stepping on ENEMY cells — "
    "collision ends the episode immediately. Survive the required number of steps to win. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7 grid, 1 chaser, survive 30 steps, 1 freeze switch. Open layout."),
        ("medium", "10x10, chaser + ambusher, survive 50 steps, 1 freeze switch, 3 obstacles."),
        ("hard", "13x13, chaser + ambusher + flanker, 80 steps, 1 switch, 5 obstacles."),
        ("expert", "15x15, 4 enemies (chaser+ambusher+flanker+trapper), 100 steps, 2 switches, 8 obstacles."),
    ],
    tags=["reactive_control", "prediction"],
)

_td(
    "TimingChallenge-v0",
    "navigation",
    "Cross a gap blocked by a moving BLOCKER patrol. Time your movement to pass "
    "through when the BLOCKER is not in the way.",
    "BLOCKER (moving patrol obstacle), gap zone (patrol area), safe refuge spots "
    "(hard+), GOAL (beyond patrol zone), WALL (boundaries).",
    "Cross the patrol zone without collision, then reach GOAL. Collision ends episode.",
    "Move in 4 directions only (actions 1-4). Wait (use NOOP action 0) for the BLOCKER "
    "patrol to move away, then cross the gap quickly. Colliding with a BLOCKER ends the "
    "episode immediately. Walk onto the GOAL cell after crossing to win. "
    "No INTERACT action needed — only movement and waiting.",
    [
        ("easy", "7x7 grid, 1 slow blocker, 1 gap. No safe spots. Simple timing."),
        ("medium", "9x9 grid, 2 blockers, 1 gap. Random gap position."),
        ("hard", "11x11 grid, 2 blockers with speed variation, 2 gaps, 1 safe spot."),
        ("expert", "13x13 grid, 3 blockers with variable speed, 2 gaps, 2 safe spots."),
    ],
    tags=["motor_control", "temporal_reasoning"],
)

# ── PLANNING (cont.) / REASONING (cont.) ──────────────────────────────────

_td(
    "PackingPuzzle-v0",
    "planning",
    "Push typed pieces (BOX, GEM, ORB, SCROLL) onto matching TARGET slots. "
    "Each target requires a specific piece type. Distractor pieces add noise.",
    "Piece objects (BOX, GEM, ORB, SCROLL, typed), TARGET slots (with required type "
    "metadata), distractor pieces (unused types, medium+).",
    "Push each piece onto its matching-type target slot.",
    "Move in 4 directions only (actions 1-4). Walk into a piece to push it one cell in "
    "your movement direction. Match piece type to target type — push each piece onto "
    "its matching-type TARGET slot. Cannot push pieces into walls or other pieces. "
    "No INTERACT action needed — pushing is automatic on movement.",
    [
        ("easy", "7x7 grid, 2 pieces of 2 types, 2 targets. No distractors, open layout."),
        ("medium", "9x9 grid, 3 pieces of 2 types, 3 targets, 1 distractor."),
        ("hard", "11x11 maze, 4 pieces of 3 types, 4 targets, 1 distractor."),
        ("expert", "13x13 grid, 5 pieces of 4 types, 5 targets, 2 distractors. Complex maze."),
    ],
    tags=["spatial_reasoning", "planning"],
)

_td(
    "TileSorting-v0",
    "planning",
    "Sliding tile puzzle: arrange numbered tiles into correct order by pushing them "
    "into the empty slot. Classic 15-puzzle mechanic with varying sizes.",
    "Numbered tiles (1..N-1 with unique numbers/colors), empty slot (agent position), "
    "target positions marked on floor showing goal arrangement.",
    "Arrange tiles to goal configuration (1,2,3...N-1 in row-major order).",
    "Move in 4 directions only (actions 1-4). Walk into an adjacent numbered tile to "
    "swap positions with it (the tile slides into the empty slot you just occupied). "
    "Arrange tiles in row-major order (1,2,3...N-1). "
    "No INTERACT action needed — tile swapping is automatic on movement.",
    [
        ("easy", "7x7 grid, 2x2 puzzle (3 tiles), 5 scrambles. Few moves to solve."),
        ("medium", "9x9 grid, 3x3 puzzle (8 tiles), 15 scrambles."),
        ("hard", "11x11 grid, 3x3 puzzle (8 tiles), 30 scrambles. Heavy scrambling."),
        ("expert", "13x13 grid, 4x4 puzzle (15 tiles), 60 scrambles. Maximum depth."),
    ],
    tags=["combinatorial_logic", "planning"],
)

_td(
    "LightsOut-v0",
    "reasoning",
    "Toggle SWITCH objects (lights) to turn them all OFF. Stepping on any cell "
    "toggles it (if a switch) AND its 4 adjacent neighbor switches — classic Lights Out.",
    "SWITCH objects (lights, ON/OFF state visible via color), decoy switches (medium+, "
    "outside puzzle grid).",
    "Turn all lights OFF by toggling switches.",
    "Move in 4 directions (actions 1-4). Switches toggle AUTOMATICALLY when you step on or "
    "near them — no INTERACT needed. Stepping on any cell toggles it (if it's a switch) AND "
    "all adjacent switches. Walking through the switch grid causes cascading "
    "toggles, so plan your path carefully. Goal: turn ALL switches OFF.",
    [
        ("easy", "7x7 grid, 3x3 puzzle, adjacent toggle (self + 4 neighbors)."),
        ("medium", "9x9 grid, 4x4 puzzle, adjacent toggle, 1 decoy."),
        ("hard", "11x11 grid, 4x4 puzzle, adjacent toggle, 2 decoys."),
        ("expert", "13x13 grid, 5x5 puzzle, adjacent toggle, 3 decoys."),
    ],
    tags=["combinatorial_logic"],
)

_td(
    "GraphColoring-v0",
    "reasoning",
    "Color graph nodes via INTERACT action to cycle through colors. Nodes are SWITCH "
    "objects placed in clusters with connected adjacency ensuring non-trivial constraints. "
    "No adjacent nodes may share color.",
    "SWITCH objects (graph nodes, cluster-based placement, color state in metadata), "
    "WALL (obstacles).",
    "Color all nodes so no two adjacent nodes share the same color.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Nodes are solid "
    "SWITCH objects — move toward a node to stand adjacent and face it (orientation "
    "updates even when blocked), then INTERACT (action 5) to cycle its color. Each "
    "INTERACT cycles to the next color. "
    "Only actions 1-4 (movement) and 5 (INTERACT) are useful in this task.",
    [
        ("easy", "9x9 grid, 4 nodes, 2 colors, linear graph. Simple constraint."),
        ("medium", "11x11 grid, 6 nodes, 3 colors, cycle graph with cluster-based adjacency."),
        ("hard", "13x13 grid, 8 nodes, 3 colors, connected clusters with 4 obstacles."),
        ("expert", "15x15 grid, 10 nodes, 4 colors, dense connected clusters with 6 obstacles."),
    ],
    tags=["combinatorial_logic", "constraint_satisfaction"],
)

# ── MULTI-AGENT ───────────────────────────────────────────────────────────

_td(
    "CooperativeTransport-v0",
    "multi_agent",
    "Push heavy boxes into holes with NPC cooperation. Boxes are too heavy to push alone; "
    "the NPC must be adjacent to the same box for a push to succeed. The NPC wanders randomly "
    "but pathfinds to the opposite side of the box when the agent is adjacent.",
    "BOX (heavy, requires NPC cooperation to push), HOLE terrain (destination for boxes), "
    "NPC helper (must be adjacent for push), WALL (obstacles).",
    "Push all heavy boxes into holes with NPC cooperation.",
    "Move in 4 directions only (actions 1-4). Walk into a BOX cell to push it one cell "
    "in your movement direction. Push only succeeds when the NPC helper is also adjacent "
    "to the same box (the NPC pathfinds to help when you're adjacent). A box pushed onto "
    "a HOLE cell removes both the box and the hole. No INTERACT action "
    "needed — pushing is automatic on movement.",
    [
        ("easy", "9x9 grid, 1 box, 1 hole, no obstacles."),
        ("medium", "12x12 grid, 2 boxes, 2 holes, 3 obstacles."),
        ("hard", "15x15 grid, 3 boxes, 3 holes, 6 obstacles."),
        ("expert", "18x18 grid, 4 boxes, 4 holes, 9 obstacles."),
    ],
    tags=["multi_agent", "cooperation"],
)

_td(
    "TagHunt-v0",
    "multi_agent",
    "Tag all fleeing NPCs before time runs out. NPCs evade the agent. "
    "SWITCH objects freeze all NPCs for 5 steps when activated (one-time use).",
    "ENEMY (fleeing NPCs), SWITCH (freeze power-up, one-time use), WALL (obstacles).",
    "Tag all NPCs by stepping onto them. Use freeze switches strategically.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Walk onto an ENEMY "
    "cell to tag and remove it. Freeze switches are solid — move toward a SWITCH to stand "
    "adjacent and face it (orientation updates even when blocked), then INTERACT (action 5) "
    "to freeze all NPCs for 5 steps (one-time use). Tag all NPCs before time runs out "
    "to win.",
    [
        ("easy", "7x7 grid, 1 NPC (50% evade), 1 freeze switch, no obstacles."),
        ("medium", "10x10 grid, 2 NPCs (65% evade), 1 freeze switch, 3 obstacles."),
        ("hard", "13x13 grid, 3 NPCs (80% evade), 1 freeze switch, 5 obstacles."),
        ("expert", "15x15 grid, 4 NPCs (90% evade), 1 freeze switch, 7 obstacles."),
    ],
    tags=["multi_agent", "competition"],
)

# ── NAVIGATION (cont.) / REASONING (cont.) ────────────────────────────────

_td(
    "InstructionFollowing-v0",
    "navigation",
    "Navigate to the one target object (GEM, SCROLL, ORB, or COIN) while avoiding "
    "distractor objects of different types. Touching a distractor ends the episode "
    "with penalty. Hard+ adds key-and-door rooms gating the target.",
    "Target object (one of GEM/SCROLL/ORB/COIN), distractor objects (other types), "
    "KEY and DOOR (hard+, color-coded).",
    "Reach the unique target object without touching any distractor. "
    "Hard+ requires collecting a key to unlock the door.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5) at hard+. Walk onto "
    "the target object cell to complete the task. Stepping on a distractor object ends "
    "the episode with penalty. At hard+: walk onto a KEY cell to auto-collect it. DOORs "
    "are solid — move toward a DOOR to face it (orientation updates even when blocked), "
    "then INTERACT (action 5) to unlock with the matching key. Walk through the opened "
    "doorway to reach the target.",
    [
        ("easy", "7x7 grid, 1 target, 3 distractors, no doors."),
        ("medium", "10x10 grid, 1 target, 6 distractors, no doors."),
        ("hard", "13x13 grid, 1 target, 8 distractors, 1 key-door room."),
        ("expert", "15x15 grid, 1 target, 12 distractors, 2 key-door rooms."),
    ],
    tags=["language", "grounding", "instruction"],
)

_td(
    "ProgramSynthesis-v0",
    "reasoning",
    "Replicate a reference SCROLL pattern by pushing scattered GEM objects (Sokoban-style) "
    "into the same relative shape. The reference pattern (line, L, T, or cross) is shown as "
    "immovable SCROLL objects. Matching is translation-invariant: GEMs must form the same "
    "normalized offsets as SCROLLs, anywhere on the map.",
    "SCROLL objects (immovable reference pattern), GEM objects (pushable, Sokoban-style), "
    "WALL (obstacles). No TARGET markers — GEMs just need to match the SCROLL shape.",
    "Push all GEM objects so they form the same relative pattern as the reference SCROLLs.",
    "Move in 4 directions only (actions 1-4). Walk into a GEM to push it one cell in your "
    "movement direction (Sokoban-style). Cannot push gems into walls, other gems, or SCROLL "
    "blocks. GEMs must form the same relative pattern as SCROLLs (translation-invariant). "
    "No INTERACT action needed — pushing is automatic on movement.",
    [
        ("easy", "9x9 grid, line pattern, 3 gems to push."),
        ("medium", "11x11 grid, L-shape pattern, 4 gems."),
        ("hard", "13x13 grid, T-shape pattern, 5 gems."),
        ("expert", "15x15 grid, cross pattern, 6 gems."),
    ],
    tags=["reasoning", "planning", "abstraction"],
)

_td(
    "RecursiveRooms-v0",
    "navigation",
    "Navigate recursively subdivided nested rooms. Grid is divided into quadrants "
    "with walls; each quadrant subdivides further. GOAL is in the deepest nested room.",
    "Recursive room structure (walls and doorways), doorways (EMPTY gaps in walls, "
    "3 of 4 sides open), GOAL (deepest nested room), sealed doors (1 side per level).",
    "Navigate through nested rooms to reach GOAL in the deepest room.",
    "Move in 4 directions only (actions 1-4). Navigate through doorway gaps in walls "
    "to move between rooms. Walk onto the GOAL cell in the deepest nested room to win. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "13x13 grid, depth 2 (4 rooms), 3 doorways per level."),
        ("medium", "19x19 grid, depth 3 (16 rooms), 3 doorways per level."),
        ("hard", "25x25 grid, depth 4 (64 rooms), 3 doorways, goal in random deepest room."),
        ("expert", "31x31 grid, depth 4 (64 rooms), sealed doorways, deep random goal."),
    ],
    tags=["hierarchical", "planning", "composition"],
)

# ── GENERALIZATION / REASONING (cont.) ────────────────────────────────────

_td(
    "NoisyObservation-v0",
    "generalization",
    "Find the true GOAL hidden among heavy visual noise: decoy TARGETs, ghost objects "
    "that appear/disappear each step, moving decoys, and patrolling guards.",
    "True GOAL (single), decoy TARGETs (look similar), ghost objects (SCROLL/ORB/LEVER "
    "appearing/disappearing), moving decoys (hard+), Guard NPCs (medium+).",
    "Locate and reach the true GOAL amid visual noise.",
    "Move in 4 directions only (actions 1-4). Walk onto the true GOAL cell to complete "
    "the task. The true GOAL is a GOAL object; decoys are TARGET objects that look similar. "
    "Ghost objects appear and disappear randomly each step — ignore them. "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7 grid, 3 decoys, 2 ghosts per step. No guards. Low noise level."),
        ("medium", "9x9 grid, 6 decoys, 4 ghosts, 1 guard. Moderate noise."),
        ("hard", "11x11 maze, 10 decoys, 6 ghosts, 2 guards, 4 moving decoys."),
        ("expert", "13x13 grid, 14 decoys, 10 ghosts, 3 guards, 6 moving decoys. Extreme noise."),
    ],
    tags=["robustness", "navigation", "noise"],
)

_td(
    "DistributionShift-v0",
    "generalization",
    "Multi-task sequential episode. Each phase is a DIFFERENT mini-task type: "
    "goal navigation, key+door, lever+barrier, or gem collection. After completing "
    "each phase, the maze regenerates with a new layout and completely different "
    "mechanics. Hard+ adds action remapping (UP<->DOWN, LEFT<->RIGHT).",
    "GOAL (per phase), KEY/DOOR (key_door phases), LEVER/WALL barriers "
    "(lever_barrier phases), GEM objects (collection phases), "
    "action remap (hard+, UP<->DOWN LEFT<->RIGHT).",
    "Complete all sequential mini-tasks across shifting maze phases.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5). Each phase has "
    "different mechanics: Navigate (walk to goal), Key+Door (collect key, INTERACT on "
    "door), Lever (INTERACT on lever to open barrier), Collect (walk over all gems). "
    "Phase type shown in HUD. Adapt to new mechanics each phase.",
    [
        ("easy", "9x9, 3 phases from 4 task types. max_steps=200."),
        ("medium", "11x11, 4 phases. max_steps=350."),
        ("hard", "13x13, 5 phases, action remap after phase 3. max_steps=500."),
        ("expert", "17x17, 6 phases, action remap after phase 2. max_steps=700."),
    ],
    tags=["generalization", "ood", "robustness"],
)

_td(
    "DeceptiveReward-v0",
    "reasoning",
    "Misleading reward gradient. Decoy paths branch from hub with COIN chains leading "
    "to TARGET objects (touching = failure). True path has no visible rewards, takes a "
    "longer winding route to GOAL. Collecting ANY coin closes the gate to the true path. "
    "Dense reward intentionally points toward decoy coins. Hard+ adds key+door on true "
    "path. Expert places coins near key locations to tempt during key collection.",
    "COIN objects (decoy reward chains along branching paths), TARGET (decoy endpoint, "
    "touching = failure), GOAL (true objective, on reward-free path), KEY and DOOR "
    "(hard+, gate the true path).",
    "Resist the coin reward gradient. Navigate the true (reward-free) path to GOAL "
    "without collecting any coins.",
    "Move in 4 directions (actions 1-4) and use INTERACT (action 5) at hard+. COINs are "
    "auto-collected when you walk onto them — collecting ANY coin permanently closes the "
    "true path gate. Stepping on a TARGET cell = failure (episode ends). At hard+: walk "
    "onto a KEY cell to auto-collect it. DOORs are solid — move toward a DOOR to face it "
    "(orientation updates even when blocked), then INTERACT (action 5) to unlock with the "
    "matching key. Walk onto the GOAL cell via the reward-free path to win.",
    [
        ("easy", "9x9 grid, Y-fork, 1 decoy path, 3 coins. Direct true path, no keys."),
        ("medium", "12x12 grid, 2 decoy paths (right + L-turn), 5 coins each. "
         "Winding true path through mid-grid to bottom-right goal."),
        ("hard", "14x14 grid, 3 decoy paths at varied depths, 6 coins each. "
         "True path behind key+door. Decoy 3 branches off past door."),
        ("expert", "16x16 labyrinth, 4 decoy paths, 7 coins each. 2 keys+doors. "
         "Decoy paths near key alcoves tempt during key collection."),
    ],
    tags=["robustness", "reward_hacking", "exploration"],
)

# ── GENERALIZATION (cont.) / REASONING (cont.) ───────────────────────────

_td(
    "FewShotAdaptation-v0",
    "generalization",
    "K auto-advancing demo trials followed by 1 test trial. Each trial places N "
    "candidate objects (GEM, SCROLL, ORB, COIN). A hidden rule determines the correct "
    "candidate: goto_type, nearest_corner, furthest_start, or most_open (hard+). Demo "
    "trials briefly highlight the correct candidate as GOAL, then revert. Test trial: "
    "agent must navigate to the correct candidate. Wrong choice = failure.",
    "Candidate objects (GEM, SCROLL, ORB, COIN), GOAL highlight (shown briefly during "
    "demos, reverts after reveal_steps), WALL (obstacles at medium+).",
    "Watch demo trials to infer the hidden rule, then navigate to the correct "
    "candidate object in the test trial. Stepping on wrong candidate = failure.",
    "Move in 4 directions only (actions 1-4). Demo trials auto-advance with no agent "
    "interaction — just observe. In the test trial, walk onto a candidate object cell "
    "to select it. Stepping on the wrong candidate = failure (episode ends). "
    "No INTERACT action needed — only movement.",
    [
        ("easy", "7x7 grid, 2 candidates, 2 demos, 12-step reveal. "
         "Rules: goto_type, nearest_corner."),
        ("medium", "9x9 grid, 3 candidates, 2 demos, 8-step reveal, 4 obstacles."),
        ("hard", "11x11 grid, 3 candidates, 2 demos, 5-step reveal, "
         "6 obstacles. Adds most_open rule."),
        ("expert", "13x13 grid, 4 candidates, 3 demos, 3-step reveal, 8 obstacles. All 4 rules."),
    ],
    tags=["meta_learning", "adaptation", "few_shot"],
)

_td(
    "TaskInterference-v0",
    "reasoning",
    "Resource Tug-of-War: two meters (GEM and ORB) start at 0.0 and must both reach a "
    "threshold. GEM items raise GEM meter +0.25 but cross-drain ORB; ORB items raise "
    "ORB meter +0.25 but cross-drain GEM. Cross-drain is 0.10 at easy/medium, 0.15 at "
    "hard+. At hard+: consecutive same-type penalty (extra -0.05 drain). "
    "At medium+: items flee from agent every 3 steps. At hard+: both meters decay 0.005/step.",
    "GEM objects (metadata=1), ORB objects (metadata=2). Items scattered on "
    "open floor. At medium+ items move away from agent every 3 steps.",
    "Raise both GEM and ORB meters to >= threshold simultaneously. Alternate collection "
    "to avoid cross-drain and consecutive penalty.",
    "Move in 4 directions only (actions 1-4). Walk onto a GEM or ORB cell to auto-collect "
    "it. Collecting a GEM raises the GEM meter +0.25 but drains the ORB meter; collecting "
    "an ORB raises the ORB meter +0.25 but drains the GEM meter. Alternate collection to "
    "avoid excessive cross-drain. Both meters must reach the threshold simultaneously to win. "
    "No INTERACT action needed — collection is automatic on step.",
    [
        ("easy", "9x9, 4 gem + 4 orb, threshold=0.6, 0.10 cross-drain, items stationary."),
        ("medium", "11x11, 5+5, threshold=0.7, 0.10 cross-drain, items flee every 3 steps."),
        ("hard", "13x13, 8+8, threshold=0.8, 0.15 drain, penalties, 0.005 decay/step."),
        ("expert", "15x15, 10+10, threshold=0.8, 0.15 drain, penalties, 0.005 decay/step."),
    ],
    tags=["meta_learning", "multi_objective"],
)


# ---------------------------------------------------------------------------
# Internal helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _get_category(task_class: type) -> str:
    """Derive category from module path (e.g. tasks.navigation.go_to_goal -> navigation)."""
    module = task_class.__module__ or ""
    parts = module.split(".")
    try:
        idx = parts.index("tasks")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return "unknown"


def _get_detailed_description(task_class: type) -> str:
    """Extract detailed description from class docstring."""
    doc = inspect.getdoc(task_class)
    if doc:
        return textwrap.dedent(doc).strip()
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_task_descriptions() -> dict[str, TaskDescription]:
    """Return structured descriptions for all registered tasks.

    Uses the curated description store first, falling back to dynamic
    extraction from the registry for any unregistered tasks.

    Returns:
        Mapping of task name -> TaskDescription.
    """
    from agentick.tasks.registry import _TASK_REGISTRY

    descriptions: dict[str, TaskDescription] = {}

    for name, task_class in sorted(_TASK_REGISTRY.items()):
        if name in _TASK_DESCRIPTIONS:
            desc = _TASK_DESCRIPTIONS[name]
            # Merge capability_tags from the class if not already set
            if not desc.capability_tags:
                desc.capability_tags = list(
                    getattr(task_class, "capability_tags", [])
                )
            descriptions[name] = desc
        else:
            # Fallback: dynamic extraction
            descriptions[name] = TaskDescription(
                name=name,
                category=_get_category(task_class),
                summary=getattr(task_class, "description", ""),
                objects="",
                goal="",
                actions="",
                capability_tags=list(
                    getattr(task_class, "capability_tags", [])
                ),
                difficulties=list(
                    getattr(task_class, "difficulty_configs", {}).keys()
                ),
            )

    return descriptions


def get_task_description(task_name: str) -> str:
    """Get a rich natural-language description for *task_name*.

    Returns the curated prompt text (summary + objects + goal + actions +
    difficulty levels) when available, otherwise falls back to the class
    docstring or short description attribute.

    Args:
        task_name: Registered task name (e.g. ``"LightsOut-v0"``).

    Returns:
        Human-readable task description suitable for LLM system prompts.
    """
    # Try curated store first
    if task_name in _TASK_DESCRIPTIONS:
        return _TASK_DESCRIPTIONS[task_name].to_prompt_text()

    # Fallback: dynamic extraction from the registry
    from agentick.tasks.registry import _TASK_REGISTRY

    task_class = _TASK_REGISTRY.get(task_name)
    if task_class is None:
        return "Complete the task objective."

    detailed = _get_detailed_description(task_class)
    if detailed:
        return detailed

    short = getattr(task_class, "description", "")
    if short:
        return short

    return "Complete the task objective."


def get_task_description_structured(task_name: str) -> TaskDescription | None:
    """Get the full structured TaskDescription for a task.

    Args:
        task_name: Registered task name.

    Returns:
        TaskDescription or None if not in curated store.
    """
    return _TASK_DESCRIPTIONS.get(task_name)
