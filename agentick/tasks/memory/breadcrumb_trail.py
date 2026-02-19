"""BreadcrumbTrail - Follow size-ordered breadcrumbs to unlock the goal.

MECHANICS:
  - Open arena (no walls except borders) — agent must navigate freely
  - N breadcrumbs placed at random positions in OPEN SPACE (not in corridors)
  - Breadcrumbs are SIZE-ORDERED: crumb 1 (KEY=tiny) → crumb 2 (BREADCRUMB=small)
    → crumb 3 (TARGET=medium) → crumb 4 (SWITCH=large) → etc.
  - Agent MUST collect them IN ORDER (1→2→3...) — wrong-order crumbs are ignored
  - GOAL is NOT rewarding or episode-terminating until ALL crumbs collected
  - After fade_steps, crumbs disappear — agent must remember the sequence
  - Tests sequence memory and spatial recall in open environments

DIFFICULTY AXES:
  - More breadcrumbs (harder sequence)
  - Faster fade (less time to memorize)
  - Larger arena (harder to navigate)
  - Solvability: always guaranteed (open space, no walls to block)
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


# Visual encoding: use different ObjectTypes to show breadcrumb "size"
# The agent sees visually distinct markers for each position in sequence
CRUMB_TYPES = [
    ObjectType.KEY,        # crumb 1 = smallest (KEY icon)
    ObjectType.BREADCRUMB, # crumb 2
    ObjectType.TARGET,     # crumb 3
    ObjectType.SWITCH,     # crumb 4
    ObjectType.BLOCKER,    # crumb 5 (largest)
]


@register_task("BreadcrumbTrail-v0", tags=["long_horizon", "memory"])
class BreadcrumbTrailTask(TaskSpec):
    """Collect size-ordered breadcrumbs in sequence to unlock the goal."""

    name = "BreadcrumbTrail-v0"
    description = "Collect size-ordered crumbs in sequence, then reach goal"
    capability_tags = ["long_horizon", "memory"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=9,  max_steps=100, params={"n_crumbs": 3, "fade_steps": 20}),
        "medium": DifficultyConfig(name="medium",  grid_size=11, max_steps=150, params={"n_crumbs": 4, "fade_steps": 15}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=200, params={"n_crumbs": 5, "fade_steps": 10}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=300, params={"n_crumbs": 6, "fade_steps": 6}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_crumbs = self.difficulty_config.params.get("n_crumbs", 3)
        fade = self.difficulty_config.params.get("fade_steps", 20)

        # OPEN ARENA — no internal walls, only border walls
        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent at random position in the arena
        interior = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)]
        rng.shuffle(interior)

        agent_pos = interior[0]
        # Goal in a different quadrant from agent
        goal_candidates = [
            (x, y) for x, y in interior
            if abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > size // 2
        ]
        if not goal_candidates:
            goal_candidates = interior[1:]
        goal_pos = goal_candidates[rng.integers(len(goal_candidates))]

        # Place breadcrumbs at diverse positions spread across arena
        # Avoid agent, goal, and each other (min distance 2)
        used = {agent_pos, goal_pos}
        crumb_positions = []
        candidates = [p for p in interior if p not in used]
        rng.shuffle(candidates)
        for pos in candidates:
            if len(crumb_positions) >= n_crumbs:
                break
            if all(abs(pos[0]-cp[0]) + abs(pos[1]-cp[1]) >= 2 for cp in crumb_positions):
                crumb_positions.append(pos)
                used.add(pos)

        # If not enough crumbs, just take remaining candidates
        remaining = [p for p in candidates if p not in used]
        for pos in remaining:
            if len(crumb_positions) >= n_crumbs:
                break
            crumb_positions.append(pos)

        n_crumbs = len(crumb_positions)  # actual crumbs placed

        # Place crumb objects with SIZE-ORDERED visual types
        for i, (cx, cy) in enumerate(crumb_positions):
            obj_type = CRUMB_TYPES[min(i, len(CRUMB_TYPES) - 1)]
            grid.objects[cy, cx] = obj_type

        # Goal placed but INACTIVE — only active after all crumbs collected
        # We DON'T place GOAL object yet — we'll place it in on_env_reset
        # (goal cell is empty until all crumbs collected)

        return grid, {
            "agent_start":     agent_pos,
            "goal_positions":  [goal_pos],
            "crumb_positions": crumb_positions,  # ordered list
            "n_crumbs":        n_crumbs,
            "fade_steps":      fade,
            "max_steps":       self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        """Initialize per-episode tracking. Goal starts LOCKED."""
        config["_next_crumb"] = 0        # index of next crumb to collect
        config["_all_collected"] = False
        config["_faded"] = False
        self._config = config
        # Redraw crumb objects (in case prior episode cleared them)
        crumbs = config.get("crumb_positions", [])
        for i, (cx, cy) in enumerate(crumbs):
            obj_type = CRUMB_TYPES[min(i, len(CRUMB_TYPES) - 1)]
            grid.objects[cy, cx] = obj_type
        # GOAL is NOT active yet
        gx, gy = config["goal_positions"][0]
        if grid.objects[gy, gx] == ObjectType.GOAL:
            grid.objects[gy, gx] = ObjectType.NONE

    def on_agent_moved(self, pos, agent, grid):
        """Collect next crumb if agent steps on it (in order only)."""
        config = getattr(self, "_config", {})
        if config.get("_all_collected", False):
            return
        ax, ay = pos
        next_idx = config.get("_next_crumb", 0)
        crumbs = config.get("crumb_positions", [])
        if next_idx >= len(crumbs):
            return
        cx, cy = crumbs[next_idx]
        if (ax, ay) == (cx, cy):
            # Collect the correct crumb
            grid.objects[cy, cx] = ObjectType.NONE
            config["_next_crumb"] = next_idx + 1
            # Check if all collected
            if config["_next_crumb"] >= config.get("n_crumbs", len(crumbs)):
                config["_all_collected"] = True
                # Activate goal!
                gx, gy = config["goal_positions"][0]
                grid.objects[gy, gx] = ObjectType.GOAL

    def on_env_step(self, agent, grid, config, step_count):
        """Fade crumbs after fade_steps."""
        fade = config.get("fade_steps", 20)
        if step_count == fade and not config.get("_faded", False):
            config["_faded"] = True
            # Remove all remaining crumbs (agent must remember from memory)
            crumbs = config.get("crumb_positions", [])
            for i, (cx, cy) in enumerate(crumbs):
                obj_type = CRUMB_TYPES[min(i, len(CRUMB_TYPES) - 1)]
                if grid.objects[cy, cx] == obj_type:
                    grid.objects[cy, cx] = ObjectType.NONE

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        next_idx = config.get("_next_crumb", 0)
        all_done = config.get("_all_collected", False)

        # Milestone reward per crumb collected
        old_idx = old_state.get("config", {}).get("_next_crumb", 0) if "config" in old_state else 0
        if next_idx > old_idx:
            reward += 0.3 * (next_idx - old_idx)

        if not all_done:
            # Guide toward next crumb (if not faded) or last known position
            crumbs = config.get("crumb_positions", [])
            if next_idx < len(crumbs) and "agent" in new_state:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                tgt = crumbs[next_idx]
                d_new = abs(ax - tgt[0]) + abs(ay - tgt[1])
                d_old = abs(ox - tgt[0]) + abs(oy - tgt[1])
                reward += 0.05 * (d_old - d_new)
        else:
            # All crumbs done — guide to goal
            goal = config.get("goal_positions", [None])[0]
            if goal and "agent" in new_state:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                d_new = abs(ax - goal[0]) + abs(ay - goal[1])
                d_old = abs(ox - goal[0]) + abs(oy - goal[1])
                reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Success ONLY if ALL crumbs collected AND agent at goal."""
        config = state.get("config", {})
        if not config.get("_all_collected", False):
            return False  # goal is locked until all crumbs collected
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        """Open arena — always reachable."""
        return True

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
