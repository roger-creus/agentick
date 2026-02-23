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
    "Move in 4 directions (up/down/left/right). No interact action needed.",
    [
        ("easy", "5x5 open grid, no walls, guards, or hazards. Straightforward path to goal."),
        ("medium", "10x10 grid with partial walls and 1 guard. No decoys. Requires obstacle avoidance."),
        ("hard", "15x15 dense maze with 2 guards, 2 decoys, and 4 scattered hazards. Complex routing."),
        ("expert", "20x20 complex maze with 3 guards, 4 decoys, and 8 hazards. Maximum challenge."),
    ],
    tags=["basic_navigation", "navigation"],
)

_td(
    "MazeNavigation-v0",
    "navigation",
    "Solve procedurally generated mazes (binary tree or recursive backtracker) "
    "to reach the exit GOAL. Maze complexity and obstacle density scale with difficulty.",
    "GOAL (maze exit), WALL (maze structure), HAZARD (scattered at medium+), "
    "Guard NPCs (patrolling at medium+).",
    "Navigate the maze to reach the GOAL exit. Avoid hazards and guards.",
    "Move in 4 directions. Pure pathfinding through maze corridors.",
    [
        ("easy", "7x7 binary-tree maze with wide corridors. Simple pathfinding exercise."),
        ("medium", "11x11 recursive-backtracker maze, 1 guard, 2 hazards. Moderate complexity."),
        ("hard", "15x15 recursive-backtracker maze with 2 guards, 4 hazards, and dead-end traps."),
        ("expert", "21x21 dense recursive maze with 3 guards, 6 hazards, and maximum branching."),
    ],
    tags=["planning", "spatial_reasoning", "navigation"],
)

_td(
    "MultiGoalRoute-v0",
    "navigation",
    "Visit multiple GOAL objects scattered across the grid in any order. "
    "Tests multi-objective pathfinding similar to the traveling salesman problem.",
    "GOAL objects (multiple, consumed on visit), WALL (obstacles), "
    "Decoy GOALs (medium+, false targets).",
    "Visit all real GOAL objects. Each goal is consumed when reached.",
    "Move in 4 directions. Step on GOAL to collect it automatically.",
    [
        ("easy", "7x7 open grid with 2 goals and no obstacles. Simple multi-stop navigation."),
        ("medium", "10x10 grid with 3 goals, 3 obstacles, and 1 decoy. Route planning required."),
        ("hard", "13x13 grid with 4 goals, 5 obstacles, and 2 decoys. Complex routing needed."),
        ("expert", "15x15 maze-like grid with 5 goals, 8 obstacles, and 3 decoys."),
    ],
    tags=["planning", "optimization", "navigation"],
)

_td(
    "DynamicObstacles-v0",
    "navigation",
    "Navigate toward a GOAL while avoiding moving NPC obstacles that patrol "
    "the grid. NPCs move probabilistically each step; expert mode adds pursuit behavior.",
    "GOAL (static destination), NPC (moving obstacles, fatal on collision), "
    "WALL (static barriers).",
    "Reach GOAL without colliding with any NPC. Collision ends episode.",
    "Move in 4 directions. Timing and prediction of BLOCKER movement is key.",
    [
        ("easy", "7x7 grid with 2 BLOCKERs (50% move chance). Open layout, easy avoidance."),
        ("medium", "10x10 grid with 3 BLOCKERs (75% move chance) and 3 scattered walls."),
        ("hard", "13x13 grid with 5 BLOCKERs (100% move) and 6 wall obstacles."),
        ("expert", "15x15 grid with 7 BLOCKERs (90% move, 15% pursuit), 9 walls. Pursuing behavior."),
    ],
    tags=["reactive_planning", "navigation"],
)

_td(
    "FogOfWarExploration-v0",
    "exploration",
    "Explore a grid with limited visibility (fog of war). The agent can only see "
    "within a small radius and must remember explored areas to find the hidden GOAL.",
    "GOAL (hidden behind fog), WALL (revealed in view), HAZARD (dangerous), "
    "Guard NPCs (hard+), Decoy GOALs (medium+), FOG (unrevealed cells).",
    "Find and reach the GOAL despite incomplete map information.",
    "Move in 4 directions. Fog reveals cells within visibility radius as you move.",
    [
        ("easy", "7x7 grid, visibility radius 2. No hazards, guards, or decoys. Systematic exploration."),
        ("medium", "10x10 grid, visibility 2, 2 decoy goals. Memory of explored areas needed."),
        ("hard", "13x13 grid, visibility 1 (tight fog), 1 guard, 3 decoys. Dense layout."),
        ("expert", "15x15 grid, visibility 1, 2 guards, 4 decoys. Extreme memory demand."),
    ],
    tags=["exploration", "memory", "navigation"],
)

_td(
    "TreasureHunt-v0",
    "exploration",
    "Find hidden treasure GEM objects using proximity clues on the floor. "
    "TARGET markers near treasures indicate distance (closer = warmer). "
    "Collecting a treasure refreshes clues for remaining ones.",
    "GEM objects (hidden treasures), TARGET markers (proximity clues, metadata "
    "encodes distance), WALL (obstacles).",
    "Collect all hidden GEM treasures by following proximity clues.",
    "Move in 4 directions. Step on GEM to collect it. Use TARGET clue markers "
    "to guide exploration — lower metadata values mean closer to treasure.",
    [
        ("easy", "9x9 grid, 2 treasures, clue radius 3. Low walls. Clues clearly guide the way."),
        ("medium", "11x11 grid, 3 treasures, clue radius 3. Moderate walls. Multi-target navigation."),
        ("hard", "14x14 grid, 4 treasures, clue radius 2. Dense walls. Smaller clue range forces exploration."),
        ("expert", "16x16 grid, 5 treasures, clue radius 1. Dense walls. Minimal clues, maximum exploration."),
    ],
    tags=["exploration", "memory", "reasoning"],
)

_td(
    "CuriosityMaze-v0",
    "exploration",
    "Explore a procedural maze to discover and visit all hidden landmark objects "
    "(SWITCH, SCROLL, ORB, COIN). No clues given — pure exploration and systematic "
    "coverage is required. GOAL appears after all landmarks visited.",
    "Landmark objects (SWITCH, SCROLL, ORB, COIN scattered throughout maze), "
    "WALL (maze structure), GOAL (appears after all visited).",
    "Visit every landmark in the maze. Explore systematically to find all of them.",
    "Move in 4 directions. Step on a landmark to visit it. Landmarks disappear "
    "when visited. Explore systematically to find all of them.",
    [
        ("easy", "9x9 maze, 3 landmarks. Low wall density. Small area to cover."),
        ("medium", "13x13 maze, 5 landmarks. Moderate walls. Efficient exploration needed."),
        ("hard", "17x17 maze, 7 landmarks. Dense walls. Large search space."),
        ("expert", "21x21 maze, 10 landmarks. Dense complex maze. Extensive systematic exploration."),
    ],
    tags=["exploration", "memory", "navigation"],
)

# ── MEMORY ────────────────────────────────────────────────────────────────

_td(
    "KeyDoorPuzzle-v0",
    "memory",
    "Collect color-coded keys and unlock matching doors to progress through locked "
    "rooms to the GOAL. Hard+ features chained locks in a hub-and-spoke layout where "
    "later keys are behind earlier doors, forcing back-and-forth traversal.",
    "KEY objects (color-coded: gold/red/blue, carry in inventory), DOOR (opens with "
    "matching-color key), GOAL (in final room), Guard NPCs (hard+), WALL (room separators).",
    "Reach GOAL after unlocking all necessary doors in sequence with matching keys.",
    "Move in 4 directions. Keys are picked up automatically when you step on them. "
    "Doors unlock automatically when you approach with the matching-color key.",
    [
        ("easy", "7x7 grid with 1 color-coded key-door pair. No guards. Simple linear rooms."),
        ("medium", "10x10 grid with 2 color-coded key-door pairs in linear rooms. No guards."),
        ("hard", "13x13 hub-and-spoke layout with 3 chained color-coded locks, 1 guard. "
         "Keys behind earlier doors force backtracking through hub."),
        ("expert", "15x15 hub-and-spoke with 4 chained color-coded locks, 2 guards. "
         "Complex non-linear traversal required."),
    ],
    tags=["memory", "sequential_reasoning"],
)

_td(
    "DelayedGratification-v0",
    "memory",
    "Resist nearby decoy KEY objects (traps) and navigate to a distant true GOAL. "
    "Touching any decoy KEY ends the episode in failure. Tests impulse control.",
    "GOAL (true objective, distant), KEY objects (decoy traps, closer to start), "
    "WALL (optional obstacles), HAZARD (hard+, blocks paths).",
    "Reach the distant true GOAL without touching any decoy KEY. Decoy touch ends episode.",
    "Move in 4 directions. Avoid stepping on KEY objects which act as traps.",
    [
        ("easy", "7x7 grid with 1 decoy KEY. Open layout, clear path to goal."),
        ("medium", "10x10 grid with 2 decoy KEYs, 3 walls. Must navigate around traps."),
        ("hard", "13x13 maze with 3 decoys, 6 walls, 2 hazards. Hazards block shortcuts."),
        ("expert", "15x15 complex maze with 4 decoys, 9 walls, 4 hazards. Careful long-range planning."),
    ],
    tags=["credit_assignment", "long_horizon"],
)

_td(
    "BacktrackPuzzle-v0",
    "memory",
    "GOAL is blocked by a wall gate. Navigate past it to reach a distant SWITCH, "
    "which opens the gate, then backtrack to collect the now-accessible GOAL.",
    "GOAL (initially blocked), SWITCH objects (trigger gate opening), "
    "WALL (gate barrier, opens after switch), dead-end paths.",
    "Activate the correct SWITCH to open the gate, then backtrack to reach GOAL.",
    "Move in 4 directions. Switches activate automatically when you step on them.",
    [
        ("easy", "9x9 L-shaped layout with 1 switch. No dead ends. Clear backtrack path."),
        ("medium", "11x11 T-shaped layout with 2 switches and 1 dead end."),
        ("hard", "13x13 zigzag layout with 3 switches and 2 dead ends."),
        ("expert", "15x15 complex layout with 4 switches and 3 dead ends. Multiple gates."),
    ],
    tags=["memory", "planning"],
)

_td(
    "BreadcrumbTrail-v0",
    "memory",
    "Collect breadcrumbs in strict size order (1, 2, 3...) before the GOAL "
    "becomes accessible. Crumbs fade after a timer. False breadcrumbs act as distractors.",
    "CRUMB objects (numbered, must collect in order), GOAL (appears after all crumbs "
    "collected), false breadcrumbs (wrong order, cause penalties), WALL (optional).",
    "Collect all breadcrumbs in ascending order, then reach GOAL.",
    "Move in 4 directions. Crumbs are collected automatically when you step on them.",
    [
        ("easy", "9x9 grid with 3 crumbs, 20-step fade timer, no false crumbs. Relaxed timing."),
        ("medium", "11x11 grid with 4 crumbs, 15-step fade, 1 false crumb, 3 obstacles."),
        ("hard", "13x13 maze with 5 crumbs, 10-step fade, 2 false crumbs, 5 obstacles. Tight timing."),
        ("expert", "15x15 complex layout with 6 crumbs, 6-step fade, 3 false crumbs, 8 obstacles."),
    ],
    tags=["long_horizon", "memory"],
)

_td(
    "SequenceMemory-v0",
    "memory",
    "Two-phase task: show phase displays target positions as GEM objects one by one. "
    "Reproduce phase requires visiting those positions in exact memorized order.",
    "GEM objects (shown during show phase), TARGET (goal positions during reproduce), "
    "SWITCH (decoy distractors at medium+).",
    "Memorize shown positions, then visit them in exact order during reproduce phase.",
    "Move in 4 directions. Step on TARGET positions to register visits.",
    [
        ("easy", "7x7 grid, 3 targets shown for 4 steps each. No distractors. Short sequence."),
        ("medium", "10x10 grid, 4 targets shown for 3 steps each, 2 distractors, 4 obstacles."),
        ("hard", "13x13 grid, 5 targets shown for 2 steps each, 3 distractors, shuffled order."),
        ("expert", "15x15 grid, 6 targets shown for 1 step each, 4 distractors, shuffled order."),
    ],
    tags=["memory", "pattern_recognition"],
)

# ── REASONING ─────────────────────────────────────────────────────────────

_td(
    "SokobanPush-v0",
    "reasoning",
    "Classic Sokoban: push BOX objects onto TARGET positions by walking into them. "
    "Boxes move in the direction the agent pushes. Cannot push into walls.",
    "BOX objects (pushable), TARGET positions (where boxes must go), "
    "WALL (immovable obstacles), GOAL (appears when all boxes placed).",
    "Push all BOX objects onto matching TARGET positions.",
    "Move in 4 directions. Walk into a BOX to push it in your movement direction. "
    "Cannot push boxes into walls or other boxes.",
    [
        ("easy", "7x7 grid with 1 box and 1 target. Open layout, no dead-end traps."),
        ("medium", "10x10 grid with 2 boxes, 2 targets, 3 obstacles creating push constraints."),
        ("hard", "13x13 grid with 3 boxes, 3 targets, 5 obstacles, 3 hazards."),
        ("expert", "15x15 grid with 4 boxes, 4 targets, 8 obstacles, 5 hazards. Complex mazes."),
    ],
    tags=["reasoning", "planning"],
)

_td(
    "CausalChain-v0",
    "reasoning",
    "Activate switches in sequence to remove barrier walls between zones. "
    "Each switch opens a specific barrier. Decoy levers look identical but do nothing.",
    "SWITCH objects (activate to remove barriers), LEVER objects (decoy, no effect), "
    "WALL barriers (between zones, removed by correct switches), GOAL (in final zone).",
    "Activate switches in causal order to open all barriers, then reach GOAL.",
    "Move in 4 directions. Switches activate automatically when you step on them. "
    "Distinguish real switches from decoy levers.",
    [
        ("easy", "9x9 grid with 2 switches in linear chain. No decoys. Clear zone progression."),
        ("medium", "11x11 grid with 3 switches, 1 decoy lever, maze zones."),
        ("hard", "13x13 grid with 4 switches, 2 decoys. Complex layout."),
        ("expert", "15x15 grid with 5 switches, 3 decoys, nonlinear layout."),
    ],
    tags=["reasoning", "causal_reasoning"],
)

_td(
    "SwitchCircuit-v0",
    "reasoning",
    "Complementary color-coded toggle switches control wall barriers. Toggling switch i "
    "opens barrier i but closes barrier (i+1)%N. Agent must toggle switches in the right "
    "order and navigate through opened barriers before they close.",
    "SWITCH objects (color-coded via metadata), colored WALL barriers (horizontal rows, "
    "metadata matches switch color), GOAL (in final zone beyond all barriers).",
    "Toggle switches in correct order and navigate through opened barriers to reach GOAL.",
    "Move in 4 directions. Step on a SWITCH to toggle it (opens its barrier, closes the "
    "complementary barrier). Navigate through gaps before they close.",
    [
        ("easy", "7x7 grid, 2 switches/barriers. Simple sequential toggle-and-walk."),
        ("medium", "9x9 grid, 3 switches/barriers. Longer sequence, tighter routing."),
        ("hard", "11x11 grid, 4 switches/barriers. Complex multi-zone navigation."),
        ("expert", "13x13 grid, 5 switches/barriers. Maximum complementary constraints."),
    ],
    tags=["combinatorial_logic", "reasoning"],
)

_td(
    "RuleInduction-v0",
    "reasoning",
    "Activate a SWITCH to learn a hidden rule (e.g. 'closest corner', 'furthest from "
    "center'), then navigate to the position the rule predicts. Multi-phase at hard+.",
    "SWITCH (reveals rule on activation), rule-dependent target positions, "
    "WALL (affects some rules), extra switches (decoys and phase transitions at hard+).",
    "Activate reveal SWITCH, identify the rule, then reach the predicted target.",
    "Move in 4 directions. Step on SWITCH to activate and reveal the rule.",
    [
        ("easy", "7x7 grid, 1 simple rule, 1 phase. No decoys."),
        ("medium", "10x10 grid, 1 compound rule, 1 decoy switch, 3 obstacles."),
        ("hard", "13x13 grid, 2 rules across 2 phases, 2 decoys, 5 obstacles."),
        ("expert", "15x15 grid, 3 rules across 3 phases, 3 decoys, 8 obstacles."),
    ],
    tags=["reasoning", "memory"],
)

_td(
    "SymbolMatching-v0",
    "reasoning",
    "Match symbol items (GEM, POTION, SCROLL, COIN, ORB, LEVER) to typed TARGET "
    "positions. Each target requires a specific symbol type. Visually distinct colored "
    "pairs; match by visiting pairs consecutively.",
    "Symbol items (GEM, POTION, SCROLL, COIN, ORB, LEVER), TARGET positions (with "
    "required type), decoy items (noise), GOAL (appears after all matches).",
    "Deliver each symbol item to its matching TARGET position.",
    "Move in 4 directions. Items are picked up and delivered automatically when you "
    "step on them and then step on the matching target.",
    [
        ("easy", "7x7 grid with 2 symbol-target pairs. No decoys. Clear pairing."),
        ("medium", "10x10 grid with 3 pairs, 1 decoy item, 3 obstacles."),
        ("hard", "13x13 maze with 4 pairs, 2 decoy items, 5 obstacles."),
        ("expert", "15x15 grid with 5 pairs, 3 decoy items, 8 obstacles."),
    ],
    tags=["reasoning", "pattern_recognition"],
)

# ── SKILL ─────────────────────────────────────────────────────────────────

_td(
    "RecipeAssembly-v0",
    "skill",
    "Follow a recipe: collect ingredients in exact order and deliver each to the "
    "crafting station. Recipe is displayed visually in corner zone using TARGET markers.",
    "Recipe zone (TARGET objects with ingredient type metadata), ingredient items "
    "(GEM, SCROLL, ORB, COIN), crafting station (GOAL object), decoy ingredients, WALL.",
    "Collect and deliver all ingredients in recipe order to the crafting station.",
    "Move in 4 directions. Ingredients are picked up automatically. Step on crafting "
    "station to deliver the held ingredient.",
    [
        ("easy", "9x9 grid, 2-ingredient recipe. No decoys, open layout. Simple collect-deliver."),
        ("medium", "11x11 grid, 3-ingredient recipe, 2 decoys, 3 obstacles. Plan route around walls."),
        ("hard", "14x14 maze, 4-ingredient recipe, 3 decoys, 5 obstacles."),
        ("expert", "16x16 dense maze, 5-ingredient recipe, 4 decoys, 7 obstacles."),
    ],
    tags=["compositional_logic", "planning"],
)

_td(
    "ToolUse-v0",
    "skill",
    "Navigate a long zigzag corridor from start to goal. Tools create SHORTCUTS that bypass "
    "sections of the long path: Bridge (KEY) crosses WATER, Hammer (TOOL) breaks WALL, "
    "Torch (GEM) passes through HAZARD. Tools have limited durability.",
    "Tools: GEM=torch, KEY=bridge, TOOL=hammer. Shortcut barriers: WATER (1-2 cells), "
    "WALL (cracked, 1-2 cells), HAZARD (lava, 1-2 cells). GOAL (at end of corridor).",
    "Reach GOAL at the end of the corridor. Collect tools and use shortcuts to save steps. "
    "Brute-force long path works but yields worse score.",
    "Move in 4 directions. Tools are picked up automatically. Walk into a shortcut barrier "
    "with the matching tool to pass through. Without tools, take the long way around.",
    [
        ("easy", "9x9 grid, 1 shortcut (water+bridge), durability 3. Short winding path."),
        ("medium", "11x11 grid, 2 shortcuts (water+hazard), durability 2. Longer corridor."),
        ("hard", "14x14 grid, 3 shortcuts (all types), durability 2. Extended winding path."),
        ("expert", "16x16 grid, 3 shortcuts, durability 1 (single use). Longest zigzag path."),
    ],
    tags=["compositional_logic", "planning"],
)

_td(
    "ResourceManagement-v0",
    "skill",
    "Keep energy stations alive by recharging them before they drain to zero. "
    "Stations drain energy over time. Any station dying ends the episode in failure. "
    "Success = surviving all max_steps.",
    "RESOURCE stations (energy 0-100, drain over time), WALL (obstacles between stations).",
    "Keep ALL stations above 0 energy for the entire episode (survive max_steps). "
    "Any station reaching 0 = failure.",
    "Move in 4 directions. Step on a RESOURCE station to recharge it to full (100).",
    [
        ("easy", "9x9 grid, 2 stations, slow drain (every 10 steps). No obstacles."),
        ("medium", "12x12 grid, 3 stations, moderate drain (every 7 steps), 3 obstacles."),
        ("hard", "15x15 grid, 4 stations, fast drain (every 5 steps), 5 obstacles."),
        ("expert", "18x18 grid, 5 stations, very fast drain (every 4 steps), 7 obstacles, far apart."),
    ],
    tags=["planning", "resource_allocation"],
)

_td(
    "MultiRoomEscape-v0",
    "skill",
    "Navigate through N interconnected rooms separated by wall barriers with doorways. "
    "Guards patrol intermediate rooms. Reach the GOAL in the final room.",
    "GOAL (in final room), room boundaries (WALL with doorways), doorways (EMPTY gaps), "
    "Guard NPCs (patrol rooms, collision penalty), HAZARD (scattered in rooms).",
    "Traverse all intermediate rooms through doorways to reach GOAL in final room.",
    "Move in 4 directions through doorways. Avoid guards and hazards.",
    [
        ("easy", "11x11 grid, 2 rooms, no guards. Wide doorways. Simple room traversal."),
        ("medium", "15x15 grid, 3 rooms, 1 guard, 2 hazards, single doorways."),
        ("hard", "19x19 grid, 4 rooms, 2 guards, 4 hazards, narrow doorways."),
        ("expert", "23x23 grid, 5 rooms, 3 guards, 6 hazards, minimal doorways."),
    ],
    tags=["skill_composition", "long_horizon"],
)

_td(
    "EmergentStrategy-v0",
    "skill",
    "SHEEP NPCs block gaps in a wall barrier. The agent must approach sheep to scare them "
    "away (sheep flee when agent is adjacent), then collect KEYs for bonus and "
    "navigate through cleared gaps to reach the GOAL beyond the barrier.",
    "SHEEP NPCs (block barrier gaps, flee from agent), WALL barrier (horizontal row with gaps), "
    "KEY objects (bonus), GOAL (beyond barrier).",
    "Scare sheep out of barrier gaps by approaching them, collect nearby keys, then "
    "navigate through clear gaps to reach GOAL.",
    "Move in 4 directions. Sheep flee when agent is adjacent. Keys are auto-collected. "
    "Position yourself above a gap to push sheep downward and clear the path.",
    [
        ("easy", "9x9 grid, 1 sheep blocking 1 gap, 1 key."),
        ("medium", "12x12 grid, 1 sheep, 1 gap, 2 keys. Barrier closes after 50 steps."),
        ("hard", "15x15 grid, 2 sheep, 2 gaps, 3 keys. Barrier warning at 5 steps."),
        ("expert", "18x18 grid, 3 sheep, 3 gaps, 4 keys. Barrier warning at 8 steps."),
    ],
    tags=["skill_composition", "long_horizon"],
)

# ── CONTROL ───────────────────────────────────────────────────────────────

_td(
    "PreciseNavigation-v0",
    "control",
    "Navigate narrow corridors bordered by fatal HAZARD terrain. Collect waypoints "
    "(GEM objects) in order before reaching the final GOAL. Moving waypoints at hard+.",
    "HAZARD borders (fatal if touched), narrow corridors (2-cell wide), "
    "GEM waypoints (collected in order), GOAL (final destination), "
    "moving waypoints (drift at hard+).",
    "Collect all waypoints in order then reach GOAL without touching HAZARD. "
    "Touching HAZARD ends the episode.",
    "Move in 4 directions. Precise movement is critical; one wrong step is fatal.",
    [
        ("easy", "9x9 grid, 2 stationary waypoints, 2-cell corridors. Safe margin for error."),
        ("medium", "12x12 grid, 3 stationary waypoints, corridors, maze layout."),
        ("hard", "15x15 grid, 4 waypoints (1 moving), tight corridors."),
        ("expert", "18x18 grid, 5 waypoints (2 moving), very narrow corridors."),
    ],
    tags=["motor_control", "planning"],
)

_td(
    "Herding-v0",
    "control",
    "Herd NPC SHEEP into a pen zone (TARGET cells) using movement pressure. Sheep "
    "flee from the agent when within distance 2. Predators scatter sheep at hard+.",
    "SHEEP NPCs (flee from agent), pen zone (TARGET cells), leader sheep (hard+, "
    "influence herd), ENEMY predators (scatter sheep at hard+), WALL (obstacles).",
    "Move all SHEEP into the pen zone (TARGET cells).",
    "Move in 4 directions. Your position influences sheep movement; approach from "
    "the opposite side to push them toward the pen.",
    [
        ("easy", "9x9 grid, 2 sheep, 2x2 pen. No obstacles or predators. Simple herding."),
        ("medium", "12x12 grid, 3 sheep, 2x2 pen, 2 obstacles, random pen corner."),
        ("hard", "15x15 grid, 4 sheep, 3x3 pen, 4 obstacles, 1 predator, 1 leader sheep."),
        ("expert", "18x18 grid, 5 sheep, 3x3 pen, 6 obstacles, 2 predators, 2 leader sheep."),
    ],
    tags=["multi_objective_control"],
)

_td(
    "ChaseEvade-v0",
    "control",
    "Survive against pursuing ENEMY NPCs for a set number of steps. Enemies chase "
    "with configurable probability. POTION power-ups freeze enemies temporarily.",
    "ENEMY NPCs (pursuing agents, chase probability scales), POTION (freeze enemies "
    "for 5 steps), GOAL (appears after survival period), WALL (obstacles).",
    "Survive the required steps without enemy collision, then reach GOAL.",
    "Move in 4 directions. Evade enemies using obstacles and power-ups strategically.",
    [
        ("easy", "7x7 grid, 1 enemy (60% chase), survive 30 steps, 1 power-up. Open layout."),
        ("medium", "10x10 grid, 2 enemies (75% chase), survive 60 steps, 1 power-up, 3 obstacles."),
        ("hard", "13x13 grid, 3 enemies (90% chase), survive 100 steps, 2 power-ups, 5 obstacles."),
        ("expert", "15x15 grid, 5 enemies (100% chase), survive 160 steps, 2 power-ups, 7 obstacles."),
    ],
    tags=["reactive_control", "prediction"],
)

_td(
    "TimingChallenge-v0",
    "control",
    "Cross a gap blocked by a moving BLOCKER patrol. Time your movement to pass "
    "through when the BLOCKER is not in the way.",
    "BLOCKER (moving patrol obstacle), gap zone (patrol area), safe refuge spots "
    "(hard+), GOAL (beyond patrol zone), WALL (boundaries).",
    "Cross the patrol zone without collision, then reach GOAL. Collision ends episode.",
    "Move in 4 directions. Wait for BLOCKER to move away, then cross quickly.",
    [
        ("easy", "7x7 grid, 1 slow blocker, 1 gap. No safe spots. Simple timing."),
        ("medium", "9x9 grid, 2 blockers, 1 gap. Random gap position."),
        ("hard", "11x11 grid, 2 blockers with speed variation, 2 gaps, 1 safe spot."),
        ("expert", "13x13 grid, 3 blockers with variable speed, 2 gaps, 2 safe spots."),
    ],
    tags=["motor_control", "temporal_reasoning"],
)

# ── COMBINATORIAL ─────────────────────────────────────────────────────────

_td(
    "PackingPuzzle-v0",
    "combinatorial",
    "Push typed pieces (BOX, GEM, ORB, SCROLL) onto matching TARGET slots. "
    "Each target requires a specific piece type. Distractor pieces add noise.",
    "Piece objects (BOX, GEM, ORB, SCROLL, typed), TARGET slots (with required type "
    "metadata), distractor pieces (unused types), GOAL (appears after all matches).",
    "Push each piece onto its matching-type target slot.",
    "Move in 4 directions. Walk into a piece to push it. Match piece type to target type.",
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
    "combinatorial",
    "Sliding tile puzzle: arrange numbered tiles into correct order by pushing them "
    "into the empty slot. Classic 15-puzzle mechanic with varying sizes.",
    "Numbered tiles (1..N-1 with unique numbers/colors), empty slot (agent position), "
    "target positions marked on floor showing goal arrangement.",
    "Arrange tiles to goal configuration (1,2,3...N-1 in row-major order).",
    "Move in 4 directions into a tile to swap positions with it (push into empty slot).",
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
    "combinatorial",
    "Toggle SWITCH objects (lights) to turn them all OFF. Easy mode: each switch "
    "toggles only itself. Medium+: toggling a light also toggles its 4 adjacent neighbors.",
    "SWITCH objects (lights, ON/OFF state visible via color), decoy switches (medium+, "
    "outside puzzle grid), GOAL (appears when all lights off).",
    "Turn all lights OFF by toggling switches.",
    "Move in 4 directions. Step on a SWITCH to toggle it (and adjacent neighbors at medium+).",
    [
        ("easy", "7x7 grid, 3x3 puzzle, self-only toggle. Each switch is independent."),
        ("medium", "9x9 grid, 4x4 puzzle, adjacent toggle (self + 4 neighbors), 1 decoy."),
        ("hard", "11x11 grid, 4x4 puzzle, adjacent toggle, 2 decoys."),
        ("expert", "13x13 grid, 5x5 puzzle, adjacent toggle, 3 decoys, random initial state."),
    ],
    tags=["combinatorial_logic"],
)

_td(
    "GraphColoring-v0",
    "combinatorial",
    "Color graph nodes via INTERACT action to cycle through colors. Nodes are RESOURCE "
    "objects on the grid with adjacency determined by proximity. No adjacent nodes may share color.",
    "RESOURCE objects (graph nodes, color state in metadata), WALL (affects adjacency), "
    "GOAL (appears when coloring is valid).",
    "Color all nodes so no two adjacent nodes share the same color.",
    "Move in 4 directions. Use INTERACT (action 5) on a node to cycle its color.",
    [
        ("easy", "9x9 grid, 3 nodes, 2 colors, linear graph. Simple constraint."),
        ("medium", "11x11 grid, 5 nodes, 3 colors, cycle graph with path-checking adjacency."),
        ("hard", "13x13 grid, 7 nodes, 3 colors, complex graph with 4 obstacles."),
        ("expert", "15x15 grid, 9 nodes, 4 colors, dense graph with 6 obstacles."),
    ],
    tags=["combinatorial_logic", "constraint_satisfaction"],
)

# ── MULTI-AGENT ───────────────────────────────────────────────────────────

_td(
    "CooperativeTransport-v0",
    "multi_agent",
    "Push a BOX to a TARGET zone with help from an NPC partner. The NPC only cooperates "
    "when the agent is correctly positioned nearby. Tests coordination and planning.",
    "BOX (requires coordinated push), TARGET zone (goal for box), NPC helper "
    "(cooperates conditionally), GOAL (appears when box on target), WALL (obstacles).",
    "Push BOX onto TARGET zone using coordinated effort with NPC.",
    "Move in 4 directions into BOX to push it. NPC auto-coordinates if you are "
    "positioned correctly on the opposite side.",
    [
        ("easy", "7x7 open grid, no obstacles. NPC responsive."),
        ("medium", "10x10 grid, 2 obstacles."),
        ("hard", "13x13 maze, 4 obstacles. Strict NPC positioning required."),
        ("expert", "15x15 dense maze, 6 obstacles. Maximum coordination challenge."),
    ],
    tags=["multi_agent", "cooperation"],
)

_td(
    "CompetitiveTag-v0",
    "multi_agent",
    "Tag all NPCs while avoiding being tagged in return. Bidirectional tag game "
    "with safe zones (ICE terrain) where no tagging can occur. Tag cooldown after contact.",
    "NPC agents (chase/evade behavior), safe zones (ICE terrain, no tagging), "
    "GOAL (appears after all tagged), WALL (obstacles), tag cooldown state.",
    "Tag all NPCs by reaching their positions. Avoid being tagged by them.",
    "Move in 4 directions. Collision with NPC triggers tagging (bidirectional).",
    [
        ("easy", "7x7 grid, 1 NPC (40% evade, 10% chase), 1 safe zone, cooldown 3."),
        ("medium", "10x10 grid, 2 NPCs (50% evade, 20% chase), 2 safe zones, cooldown 2, 2 obstacles."),
        ("hard", "13x13 grid, 3 NPCs (60% evade, 30% chase), 2 safe zones, cooldown 2, 4 obstacles."),
        ("expert", "15x15 grid, 4 NPCs (55% evade, 40% chase), 3 safe zones, cooldown 1, 6 obstacles."),
    ],
    tags=["multi_agent", "competition"],
)

# ── COMPOSITIONAL ─────────────────────────────────────────────────────────

_td(
    "InstructionFollowing-v0",
    "compositional",
    "Follow an encoded instruction to reach the correct zone among multiple goal "
    "zones. Wrong zone selection incurs penalty. Hard+ adds waypoint and switch prerequisites.",
    "Goal zones (TARGET objects, multiple), instruction indicator (metadata), "
    "waypoints (GEM, hard+), conditional switches (SWITCH, hard+), Guard NPCs (hard+).",
    "Reach the zone indicated by the instruction. Hard+ requires visiting waypoints "
    "and activating switches first.",
    "Move in 4 directions to the designated zone.",
    [
        ("easy", "7x7 grid, 2 zones, simple binary instruction. No guards or prerequisites."),
        ("medium", "10x10 grid, 3 zones, multi-step instruction, 1 distractor. No guards."),
        ("hard", "13x13 maze, 4 zones, conditional instruction, 2 distractors, 1 guard."),
        ("expert", "15x15 maze, 5 zones, multi-conditional, 3 distractors, 2 guards."),
    ],
    tags=["language", "grounding", "instruction"],
)

_td(
    "ProgramSynthesis-v0",
    "compositional",
    "Complete a visible pattern by pushing GEM objects to TARGET positions on the floor. "
    "Fixed SCROLL blocks show the existing partial pattern. GEMs are pushed Sokoban-style "
    "(walk into a gem to slide it one cell forward).",
    "SCROLL objects (fixed pattern blocks, immovable), GEM objects (movable, push to targets), "
    "TARGET positions (marked on floor, where gems must go), WALL (obstacles).",
    "Push all GEM objects onto TARGET positions to complete the pattern.",
    "Move in 4 directions. Walk into a GEM to push it in your movement direction. "
    "Cannot push gems into walls, other gems, or SCROLL blocks.",
    [
        ("easy", "7x7 grid, horizontal line pattern, 3 fixed blocks, 1 gem to push."),
        ("medium", "9x9 grid, L-shape pattern, 4 fixed blocks, 2 gems."),
        ("hard", "11x11 grid, T-shape pattern, 5 fixed blocks, 3 gems."),
        ("expert", "13x13 grid, cross/plus pattern, 7 fixed blocks, 4 gems."),
    ],
    tags=["reasoning", "planning", "abstraction"],
)

_td(
    "RecursiveRooms-v0",
    "compositional",
    "Navigate recursively subdivided nested rooms. Grid is divided into quadrants "
    "with walls; each quadrant subdivides further. GOAL is in the deepest nested room.",
    "Recursive room structure (walls and doorways), doorways (EMPTY gaps in walls, "
    "3 of 4 sides open), GOAL (deepest nested room), sealed doors (1 side per level).",
    "Navigate through nested rooms to reach GOAL in the deepest room.",
    "Move in 4 directions through doorways between rooms.",
    [
        ("easy", "15x15 grid, depth 2 (4 rooms), 3 doorways per level."),
        ("medium", "21x21 grid, depth 3 (16 rooms), 3 doorways per level."),
        ("hard", "27x27 grid, depth 4 (64 rooms), 3 doorways, goal in random deepest room."),
        ("expert", "33x33 grid, depth 5 (256 rooms), sealed doorways, deep random goal."),
    ],
    tags=["hierarchical", "planning", "composition"],
)

# ── ADVERSARIAL ───────────────────────────────────────────────────────────

_td(
    "NoisyObservation-v0",
    "adversarial",
    "Find the true GOAL hidden among heavy visual noise: decoy TARGETs, ghost objects "
    "that appear/disappear each step, moving decoys, and patrolling guards.",
    "True GOAL (single), decoy TARGETs (look similar), ghost objects (SCROLL/ORB/LEVER "
    "appearing/disappearing), moving decoys (hard+), Guard NPCs (medium+).",
    "Locate and reach the true GOAL amid visual noise.",
    "Move in 4 directions. Identify the true GOAL among decoys and noise.",
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
    "adversarial",
    "Multi-phase task: collect COINs, then adapt to terrain shifts (EMPTY<->HAZARD, "
    "walls toggle) and reach the GOAL. Medium+ adds action remaps.",
    "COIN objects (phase 1), GOAL (appears after shift), "
    "WALL (toggles at shift), HAZARD (swaps with EMPTY at shift), action remap (medium+).",
    "Complete phase 1 (collect COINs), adapt to shift, reach GOAL.",
    "Move in 4 directions. After shift, terrain changes and actions may remap (medium+).",
    [
        ("easy", "7x7 grid, 1 shift, 3 coins. No wall toggle or action remap."),
        ("medium", "9x9 grid, 2 shifts, 4 coins, wall toggle, left/right action remap."),
        ("hard", "11x11 grid, 3 shifts, 5 coins, wall toggle + hazard swap, full action remap."),
        ("expert", "13x13 grid, 4 shifts, 6 coins, complex toggles, full action remap."),
    ],
    tags=["generalization", "ood", "robustness"],
)

_td(
    "DeceptiveReward-v0",
    "adversarial",
    "Navigate through HAZARD trap fields to reach the true GOAL. Phase 1 has no visible "
    "goal; phase 2 goal appears. Trap layouts are biased toward the goal to lure the agent.",
    "HAZARD traps (various patterns: scattered, barrier, ring, mines), true GOAL "
    "(appears phase 2), trap gradient (hard+, denser near goal), moving traps (expert).",
    "Survive phase 1, then reach true GOAL in phase 2 after it appears. "
    "Hazard touch is fatal.",
    "Move in 4 directions. Avoid hazard traps at all costs; they end the episode.",
    [
        ("easy", "7x7 grid, 10% trap density, mixed layout. No gradient or moving traps."),
        ("medium", "10x10 grid, 18% trap density, mixed layout."),
        ("hard", "13x13 grid, 28% trap density, trap gradient toward goal."),
        ("expert", "15x15 grid, 38% trap density, trap gradient, moving traps. Maximum danger."),
    ],
    tags=["robustness", "reward_hacking", "exploration"],
)

# ── META ──────────────────────────────────────────────────────────────────

_td(
    "FewShotAdaptation-v0",
    "meta",
    "Observe example demonstrations showing a hidden rule, then apply that rule to "
    "new test cases. Rules are compositional (e.g. color+shape matching).",
    "Example zone (shows correct rule application visually), test zone (agent applies rule), "
    "variant objects (colors, types, sizes vary), GOAL (appears after correct application).",
    "Observe examples, identify the hidden rule pattern, complete test task.",
    "Move in 4 directions. Interact with test objects according to the inferred rule.",
    [
        ("easy", "7x7 grid, 2 candidates, 2 demos shown for 12 steps each. Clear pattern."),
        ("medium", "9x9 grid, 3 candidates, 2 demos shown for 8 steps, 4 obstacles."),
        ("hard", "11x11 grid, 3 candidates, 3 demos shown for 5 steps, 6 obstacles, 1 guard."),
        ("expert", "13x13 grid, 4 candidates, 3 demos shown for 3 steps, 8 obstacles, 2 guards."),
    ],
    tags=["meta_learning", "adaptation", "few_shot"],
)

_td(
    "TaskInterference-v0",
    "meta",
    "Complete two competing objectives: deliver COINs to GOAL-1 and GEMs to GOAL-2. "
    "Interference: picking up a COIN destroys the nearest uncollected GEM (and vice versa). "
    "Physical interference walls appear at medium+.",
    "COIN objects (deliver to GOAL-1), GEM objects (deliver to GOAL-2), "
    "GOAL-1 and GOAL-2 (typed delivery zones), interference walls (medium+, block paths "
    "after pickup), distractor items (expert).",
    "Deliver ALL COINs to GOAL-1 AND ALL GEMs to GOAL-2. Minimize interference.",
    "Move in 4 directions. Items are picked up and delivered automatically at goal zones.",
    [
        ("easy", "9x9 grid, 2 coins + 2 gems. No interference mechanic, open layout."),
        ("medium", "11x11 grid, 3 coins + 3 gems, interference ON, walls appear on coin pickup."),
        ("hard", "13x13 grid, 4+4 items, interference ON, walls on both coin and gem pickup."),
        ("expert", "15x15 grid, 5+5 items, interference walls on both, maximum complexity."),
    ],
    tags=["multi_task", "interference", "meta_learning"],
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
