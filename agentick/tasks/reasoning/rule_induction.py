"""RuleInduction - Infer hidden rules and navigate to the correct goal.

MECHANICS:
  - Several candidate targets (TARGET) on the grid
  - A SWITCH that reveals which target is the correct goal
  - Agent must first visit the switch to learn the rule, then navigate to the target
  - At medium+: compound rules (spatial relationship, not just "target #N")
  - At hard+: multiple rule phases — after finding first goal, new switch + new rule
  - At expert: mid-episode rule changes where the correct target shifts
  - Compound rules: "closest to corner", "furthest from switch", "adjacent to wall"
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


# Compound rule functions: given targets and grid, return the correct index
def _rule_closest_corner(targets, grid):
    """Correct target is the one closest to any corner."""
    corners = [(1, 1), (grid.width - 2, 1), (1, grid.height - 2), (grid.width - 2, grid.height - 2)]
    best_idx, best_dist = 0, float("inf")
    for i, (tx, ty) in enumerate(targets):
        d = min(abs(tx - cx) + abs(ty - cy) for cx, cy in corners)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _rule_furthest_from_center(targets, grid):
    """Correct target is the one furthest from grid center."""
    cx, cy = grid.width // 2, grid.height // 2
    best_idx, best_dist = 0, -1
    for i, (tx, ty) in enumerate(targets):
        d = abs(tx - cx) + abs(ty - cy)
        if d > best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _rule_most_adjacent_walls(targets, grid):
    """Correct target is the one with most adjacent wall cells."""
    best_idx, best_count = 0, -1
    for i, (tx, ty) in enumerate(targets):
        count = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                if grid.terrain[ny, nx] == CellType.WALL:
                    count += 1
        if count > best_count:
            best_count = count
            best_idx = i
    return best_idx


_COMPOUND_RULES = [
    ("closest_corner", _rule_closest_corner),
    ("furthest_center", _rule_furthest_from_center),
    ("most_walls", _rule_most_adjacent_walls),
]


@register_task("RuleInduction-v0", tags=["reasoning", "memory"])
class RuleInductionTask(TaskSpec):
    """Learn hidden rules from switches, navigate to correct targets."""

    name = "RuleInduction-v0"
    description = "Induce rules from switches to find correct targets"
    capability_tags = ["reasoning", "memory"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=80,
            params={
                "n_targets": 2,
                "n_decoys": 0,
                "n_obstacles": 0,
                "n_phases": 1,
                "compound_rules": False,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=150,
            params={
                "n_targets": 3,
                "n_decoys": 1,
                "n_obstacles": 3,
                "n_phases": 1,
                "compound_rules": True,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={
                "n_targets": 4,
                "n_decoys": 2,
                "n_obstacles": 5,
                "n_phases": 2,
                "compound_rules": True,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=450,
            params={
                "n_targets": 5,
                "n_decoys": 3,
                "n_obstacles": 8,
                "n_phases": 3,
                "compound_rules": True,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_targets", 2)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)
        n_phases = self.difficulty_config.params.get("n_phases", 1)
        compound = self.difficulty_config.params.get("compound_rules", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        switch_pos = (size // 2, size // 2)
        grid.objects[switch_pos[1], switch_pos[0]] = ObjectType.SWITCH

        outer = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
            and (x, y) != switch_pos
            and (abs(x - size // 2) > 1 or abs(y - size // 2) > 1)
        ]
        rng.shuffle(outer)
        target_positions = outer[:n]
        used = {agent_pos, switch_pos} | set(target_positions)

        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.TARGET

        # Determine true goal
        if compound and len(target_positions) >= 2:
            rule_idx = int(rng.integers(0, len(_COMPOUND_RULES)))
            rule_name, rule_fn = _COMPOUND_RULES[rule_idx]
            true_goal_idx = rule_fn(target_positions, grid)
        else:
            rule_name = "index"
            true_goal_idx = int(rng.integers(0, n))

        # Decoy zones
        decoy_positions = []
        for p in outer[n:]:
            if len(decoy_positions) >= n_decoys:
                break
            if p not in used:
                dx2, dy2 = p
                grid.objects[dy2, dx2] = ObjectType.BOX
                decoy_positions.append(p)
                used.add(p)

        # Obstacle walls
        wall_positions = []
        all_cells = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) not in used
        ]
        rng.shuffle(all_cells)
        critical = [agent_pos, switch_pos] + target_positions
        for p in all_cells:
            if len(wall_positions) >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(q in reachable for q in critical):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        # Extra switch positions for additional phases
        extra_switch_positions = []
        if n_phases > 1:
            sw_candidates = [p for p in outer[n + n_decoys :] if p not in used]
            rng.shuffle(sw_candidates)
            for p in sw_candidates[: n_phases - 1]:
                extra_switch_positions.append(p)
                used.add(p)

        # Pre-compute rules for each phase
        phase_rules = [{"rule_name": rule_name, "true_idx": true_goal_idx}]
        for phase_i in range(1, n_phases):
            if compound:
                ri = int(rng.integers(0, len(_COMPOUND_RULES)))
                rn, rf = _COMPOUND_RULES[ri]
                ti = rf(target_positions, grid)
                # Ensure different from previous phase
                if ti == phase_rules[-1]["true_idx"] and n > 1:
                    ti = (ti + 1) % len(target_positions)
            else:
                ti = int(rng.integers(0, n))
            phase_rules.append({"rule_name": rn if compound else "index", "true_idx": ti})

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [target_positions[true_goal_idx]],
            "target_positions": target_positions,
            "switch_pos": switch_pos,
            "true_goal_idx": true_goal_idx,
            "decoy_positions": decoy_positions,
            "n_phases": n_phases,
            "phase_rules": phase_rules,
            "_extra_switch_positions": extra_switch_positions,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_rule_revealed"] = False
        config["_current_phase"] = 0
        config["_phases_completed"] = 0
        self._last_rule_revealed = False
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        """Reveal rule when agent steps on switch."""
        config = getattr(self, "_config", {})
        x, y = pos

        # Check if agent is on the switch
        if grid.objects[y, x] == ObjectType.SWITCH and not config.get("_rule_revealed", False):
            grid.metadata[y, x] = 100  # mark as pressed/on
            phase = config.get("_current_phase", 0)
            rules = config.get("phase_rules", [])
            if phase < len(rules):
                true_idx = rules[phase]["true_idx"]
                targets = config.get("target_positions", [])
                if true_idx < len(targets):
                    gx, gy = targets[true_idx]
                    grid.objects[gy, gx] = ObjectType.GOAL
            config["_rule_revealed"] = True

        # Check if agent reached the GOAL (correct target after reveal)
        if grid.objects[y, x] == ObjectType.GOAL and config.get("_rule_revealed", False):
            n_phases = config.get("n_phases", 1)
            completed = config.get("_phases_completed", 0) + 1
            config["_phases_completed"] = completed

            if completed < n_phases:
                # More phases to go: clear current GOAL, place next switch
                grid.objects[y, x] = ObjectType.NONE

                # Restore all targets as TARGET
                for tx, ty in config.get("target_positions", []):
                    if grid.objects[ty, tx] == ObjectType.NONE:
                        grid.objects[ty, tx] = ObjectType.TARGET

                # Place next switch
                extra = config.get("_extra_switch_positions", [])
                next_phase = completed
                if next_phase - 1 < len(extra):
                    sx, sy = extra[next_phase - 1]
                    grid.objects[sy, sx] = ObjectType.SWITCH

                config["_rule_revealed"] = False
                config["_current_phase"] = next_phase

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_revealed = config.get("_rule_revealed", False)
        if new_revealed and not self._last_rule_revealed:
            reward += 0.2
        self._last_rule_revealed = new_revealed

        # Approach shaping
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            if not new_revealed:
                # Guide toward switch (find visible switch on grid)
                g = new_state.get("grid")
                if g is not None:
                    for sy in range(g.height):
                        for sx in range(g.width):
                            if g.objects[sy, sx] == ObjectType.SWITCH:
                                d_new = abs(ax - sx) + abs(ay - sy)
                                d_old = abs(ox - sx) + abs(oy - sy)
                                reward += 0.05 * (d_old - d_new)
                                break
                        else:
                            continue
                        break
            else:
                # Guide toward true goal
                g = new_state.get("grid")
                if g is not None:
                    for gy2 in range(g.height):
                        for gx2 in range(g.width):
                            if g.objects[gy2, gx2] == ObjectType.GOAL:
                                d_new = abs(ax - gx2) + abs(ay - gy2)
                                d_old = abs(ox - gx2) + abs(oy - gy2)
                                reward += 0.05 * (d_old - d_new)
                                break
                        else:
                            continue
                        break

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All phases completed."""
        config = state.get("config", {})
        n_phases = config.get("n_phases", 1)
        completed = config.get("_phases_completed", 0)
        return completed >= n_phases

    def check_done(self, state):
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
