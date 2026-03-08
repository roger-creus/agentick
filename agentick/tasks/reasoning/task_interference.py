"""TaskInterference - Resource Tug-of-War.

MECHANICS:
  - Two resource meters: GEM and ORB (0.0 to 1.0, start at 0.0)
  - GEM items (GEM with metadata=1) and ORB items (ORB with metadata=2) on grid
  - Collecting GEM item: GEM meter += 0.25, ORB meter -= cross_drain
  - Collecting ORB item: ORB meter += 0.25, GEM meter -= cross_drain
  - cross_drain: easy=0.05, medium=0.08, hard=0.10, expert=0.12
  - No decay — meters only change on item collection
  - Meters clamped to [0.0, 1.0]
  - SUCCESS = both meters >= 0.5 AND all items collected
  - Episode terminates when all items are collected (or at max_steps)
  - At medium+: items flee from agent (1 cell per 3 steps)
  - Difficulties: easy=5+5, medium=6+6 items flee,
    hard=8+8 items flee, expert=10+10 items flee
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TaskInterference-v0", tags=["meta_learning", "multi_objective"])
class TaskInterferenceTask(TaskSpec):
    """Balance two competing resource meters by collecting gem and orb items."""

    name = "TaskInterference-v0"
    description = "Balance competing gem and orb meters in a tug-of-war"
    capability_tags = ["multi_objective", "planning", "meta_learning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={
                "n_red": 5,
                "n_blue": 5,
                "threshold": 0.5,
                "cross_drain": 0.05,
                "decay": 0.0,
                "items_move": False,
                "hold_steps": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=180,
            params={
                "n_red": 6,
                "n_blue": 6,
                "threshold": 0.5,
                "cross_drain": 0.08,
                "decay": 0.0,
                "items_move": True,
                "hold_steps": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={
                "n_red": 8,
                "n_blue": 8,
                "threshold": 0.5,
                "cross_drain": 0.10,
                "decay": 0.0,
                "items_move": True,
                "hold_steps": 0,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=450,
            params={
                "n_red": 10,
                "n_blue": 10,
                "threshold": 0.5,
                "cross_drain": 0.12,
                "decay": 0.0,
                "items_move": True,
                "hold_steps": 0,
            },
        ),
    }

    def generate(self, seed: int):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        n_red = params["n_red"]
        n_blue = params["n_blue"]

        grid = Grid(size, size)
        # Border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Random quadrant start
        half = size // 2
        qx = int(rng.integers(1, half) if rng.random() < 0.5
                 else rng.integers(half, size - 1))
        qy = int(rng.integers(1, half) if rng.random() < 0.5
                 else rng.integers(half, size - 1))
        agent_pos = (qx, qy)

        free = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
        ]
        rng.shuffle(free)

        used = {agent_pos}
        idx = 0

        # Place RED items (GEM with metadata=1)
        red_positions = []
        for _ in range(n_red):
            while idx < len(free) and free[idx] in used:
                idx += 1
            if idx >= len(free):
                break
            pos = free[idx]
            idx += 1
            px, py = pos
            grid.objects[py, px] = ObjectType.GEM
            grid.metadata[py, px] = 1
            red_positions.append(pos)
            used.add(pos)

        # Place BLUE items (ORB with metadata=2)
        blue_positions = []
        for _ in range(n_blue):
            while idx < len(free) and free[idx] in used:
                idx += 1
            if idx >= len(free):
                break
            pos = free[idx]
            idx += 1
            px, py = pos
            grid.objects[py, px] = ObjectType.ORB
            grid.metadata[py, px] = 2
            blue_positions.append(pos)
            used.add(pos)

        # Determine which items require holding (expert only)
        hold_steps = params.get("hold_steps", 0)
        hold_items: set[tuple[int, int]] = set()
        if hold_steps >= 2:
            # Mark roughly half the items as requiring holding
            all_items = red_positions + blue_positions
            n_hold = max(1, len(all_items) // 2)
            hold_indices = rng.choice(len(all_items), size=n_hold, replace=False)
            hold_items = {all_items[i] for i in hold_indices}

        total_items = len(red_positions) + len(blue_positions)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": red_positions + blue_positions,
            "red_positions": red_positions,
            "blue_positions": blue_positions,
            "threshold": params["threshold"],
            "cross_drain": params.get("cross_drain", 0.10),
            "decay": params["decay"],
            "items_move": params["items_move"],
            "hold_steps": hold_steps,
            "hold_items": list(hold_items),
            "_total_items": total_items,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_red_meter"] = 0.0
        config["_blue_meter"] = 0.0
        config["_live_red"] = list(config.get("red_positions", []))
        config["_live_blue"] = list(config.get("blue_positions", []))
        config["_hold_items"] = set(
            tuple(p) for p in config.get("hold_items", [])
        )
        config["_hold_counter"] = {}  # pos -> steps standing on it
        config["_items_collected"] = 0
        config["_collected_count"] = 0
        config["_last_collected_color"] = None  # "red" or "blue"
        self._flee_rng = np.random.default_rng(
            config.get("_rng_seed", 0)
        )
        self._prev_red = 0.0
        self._prev_blue = 0.0
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos
        obj = int(grid.objects[y, x])
        meta = int(grid.metadata[y, x])
        hold_steps = config.get("hold_steps", 0)
        hold_items = config.get("_hold_items", set())

        # Check if we are on a collectible item
        is_red = obj == ObjectType.GEM and meta == 1
        is_blue = obj == ObjectType.ORB and meta == 2

        if not (is_red or is_blue):
            return

        # Hold items: require standing on them for hold_steps consecutive
        # steps. Track via _hold_pos and _hold_count in on_env_step.
        # Here we just start the hold by recording the position.
        if hold_steps >= 2 and (x, y) in hold_items:
            config["_hold_pos"] = (x, y)
            config["_hold_count"] = 1  # first step = arriving
            if 1 < hold_steps:
                return  # wait for on_env_step to tick more
            # hold_steps == 1 means instant collect (shouldn't happen but handle it)

        self._collect_item(x, y, is_red, grid, config)

    def _collect_item(self, x, y, is_red, grid, config):
        """Collect the item at (x, y) and update meters."""
        hold_items = config.get("_hold_items", set())
        hold_items.discard((x, y))

        grid.objects[y, x] = ObjectType.NONE
        grid.metadata[y, x] = 0
        config["_items_collected"] = config.get("_items_collected", 0) + 1
        config["_collected_count"] = config.get("_collected_count", 0) + 1

        gain = 0.25
        cross_drain = config.get("cross_drain", 0.10)

        if is_red:
            config["_red_meter"] = min(
                1.0, config.get("_red_meter", 0.0) + gain
            )
            config["_blue_meter"] = max(
                0.0, config.get("_blue_meter", 0.0) - cross_drain
            )
            live = config.get("_live_red", [])
            if (x, y) in live:
                live.remove((x, y))
        else:
            config["_blue_meter"] = min(
                1.0, config.get("_blue_meter", 0.0) + gain
            )
            config["_red_meter"] = max(
                0.0, config.get("_red_meter", 0.0) - cross_drain
            )
            live = config.get("_live_blue", [])
            if (x, y) in live:
                live.remove((x, y))

        config["_last_collected_color"] = "red" if is_red else "blue"

        # Clear hold tracking
        config.pop("_hold_pos", None)
        config.pop("_hold_count", None)

    def on_env_step(self, agent, grid, config, step_count):
        """Tick hold counters and move items (flee)."""
        # --- Hold-item tick ---
        hold_steps = config.get("hold_steps", 0)
        hold_pos = config.get("_hold_pos")
        if hold_pos is not None and hold_steps >= 2:
            ax, ay = agent.position
            hx, hy = hold_pos
            if (ax, ay) == (hx, hy):
                # Agent still standing on the hold item
                count = config.get("_hold_count", 0) + 1
                config["_hold_count"] = count
                if count >= hold_steps:
                    obj = int(grid.objects[hy, hx])
                    meta = int(grid.metadata[hy, hx])
                    is_red = obj == ObjectType.GEM and meta == 1
                    is_blue = obj == ObjectType.ORB and meta == 2
                    if is_red or is_blue:
                        self._collect_item(hx, hy, is_red, grid, config)
            else:
                # Agent moved off — cancel hold
                config.pop("_hold_pos", None)
                config.pop("_hold_count", None)

        # --- Item movement (flee from agent every 3 steps) ---
        if not config.get("items_move", False):
            return
        if step_count % 3 != 0:
            return

        ax, ay = agent.position
        rng = getattr(self, "_flee_rng", np.random.default_rng(0))

        # Collect all occupied positions (items + agent)
        occupied = set()
        occupied.add((ax, ay))
        for pos in config.get("_live_red", []):
            occupied.add(tuple(pos))
        for pos in config.get("_live_blue", []):
            occupied.add(tuple(pos))

        hold_items = config.get("_hold_items", set())

        # Move items one at a time to avoid two items fleeing to the
        # same cell.  Update the grid and occupied set immediately.
        for color_key in ("_live_red", "_live_blue"):
            live = config.get(color_key, [])
            new_live = []
            for old in live:
                new = self._flee_item(old, (ax, ay), grid, occupied, rng)
                if old != new:
                    ox, oy = old
                    nx, ny = new
                    obj_type = grid.objects[oy, ox]
                    meta_val = grid.metadata[oy, ox]
                    grid.objects[oy, ox] = ObjectType.NONE
                    grid.metadata[oy, ox] = 0
                    grid.objects[ny, nx] = obj_type
                    grid.metadata[ny, nx] = meta_val
                    occupied.discard(old)
                    occupied.add(new)
                    if old in hold_items:
                        hold_items.discard(old)
                        hold_items.add(new)
                new_live.append(new)
            config[color_key] = new_live

    @staticmethod
    def _flee_item(
        item_pos: tuple[int, int],
        agent_pos: tuple[int, int],
        grid: Grid,
        occupied: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Move item 1 cell away from agent. Returns new position."""
        ix, iy = item_pos
        ax, ay = agent_pos
        dx = ix - ax
        dy = iy - ay

        # Build list of candidate flee directions, prioritising
        # the direction directly away from the agent
        candidates = []
        if dx > 0:
            candidates.append((1, 0))
        elif dx < 0:
            candidates.append((-1, 0))
        if dy > 0:
            candidates.append((0, 1))
        elif dy < 0:
            candidates.append((0, -1))

        # Add perpendicular directions as fallbacks
        all_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for d in all_dirs:
            if d not in candidates:
                candidates.append(d)

        # Shuffle the fallbacks (but keep primary flee dirs first)
        n_primary = len([d for d in candidates[:2]])
        fallbacks = candidates[n_primary:]
        rng.shuffle(fallbacks)
        candidates = candidates[:n_primary] + list(fallbacks)

        for ddx, ddy in candidates:
            nx, ny = ix + ddx, iy + ddy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and grid.terrain[ny, nx] == CellType.EMPTY
                and (nx, ny) not in occupied
            ):
                return (nx, ny)

        # No valid flee cell — stay put
        return item_pos

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # Step penalty
        config = new_state.get("config", {})
        red = config.get("_red_meter", 0.0)
        blue = config.get("_blue_meter", 0.0)

        # Reward for collecting items
        old_config = old_state.get("config", {})
        old_collected = old_config.get("_items_collected", 0)
        new_collected = config.get("_items_collected", 0)
        if new_collected > old_collected:
            reward += 0.3 * (new_collected - old_collected)

        # Penalty when a meter drops below its previous level
        # (from cross-interference or decay)
        if red < self._prev_red:
            reward -= 0.05
        if blue < self._prev_blue:
            reward -= 0.05

        # Approach shaping: reward getting closer to nearest uncollected item
        agent_info = new_state.get("agent", None)
        if agent_info is not None:
            ax, ay = agent_info.position
            all_live = (
                config.get("_live_red", []) + config.get("_live_blue", [])
            )
            if all_live:
                min_dist = min(
                    abs(ax - px) + abs(ay - py) for px, py in all_live
                )
                old_agent = old_state.get("agent", None)
                if old_agent is not None:
                    oax, oay = old_agent.position
                    old_min = min(
                        abs(oax - px) + abs(oay - py) for px, py in all_live
                    )
                    if min_dist < old_min:
                        reward += 0.02

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        self._prev_red = red
        self._prev_blue = blue
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        threshold = config.get("threshold", 0.5)
        red = config.get("_red_meter", 0.0)
        blue = config.get("_blue_meter", 0.0)
        total = config.get("_total_items", 0)
        collected = config.get("_collected_count", 0)
        return red >= threshold and blue >= threshold and collected >= total

    def check_done(self, state):
        """Episode ends when all items are collected (or on success)."""
        config = state.get("config", {})
        total = config.get("_total_items", 0)
        collected = config.get("_collected_count", 0)
        if total > 0 and collected >= total:
            return True
        return False

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
