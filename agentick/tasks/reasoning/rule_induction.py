"""RuleInduction - Infer the hidden rule and navigate to the correct goal.

MECHANICS:
  - Several colored zones (TARGET) on the grid
  - One SWITCH that reveals which zone is the "correct" goal
  - Agent must first visit the switch to learn the rule,
    then navigate to the correct target
  - Without visiting the switch, agent must guess which target is correct
  - Correct target → success; wrong target → penalty + episode ends
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RuleInduction-v0", tags=["reasoning", "memory"])
class RuleInductionTask(TaskSpec):
    """Learn the hidden rule from the switch, then go to the correct target."""

    name = "RuleInduction-v0"
    description = "Learn rule from switch, navigate to correct target"
    capability_tags = ["reasoning", "memory"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=80,  params={"n_targets": 2, "n_decoys": 0, "n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=150, params={"n_targets": 3, "n_decoys": 1, "n_obstacles": 3}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=250, params={"n_targets": 4, "n_decoys": 2, "n_obstacles": 5}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=400, params={"n_targets": 5, "n_decoys": 3, "n_obstacles": 8}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size        = self.difficulty_config.grid_size
        n           = self.difficulty_config.params.get("n_targets", 2)
        n_decoys    = self.difficulty_config.params.get("n_decoys", 0)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        switch_pos = (size//2, size//2)
        grid.objects[switch_pos[1], switch_pos[0]] = ObjectType.SWITCH

        outer = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                 if (x, y) != agent_pos and (x, y) != switch_pos
                 and (abs(x - size//2) > 1 or abs(y - size//2) > 1)]
        rng.shuffle(outer)
        target_positions = outer[:n]
        used = {agent_pos, switch_pos} | set(target_positions)

        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.TARGET

        true_goal_idx = int(rng.integers(0, n))

        # Decoy zones: extra TARGET cells not in target_positions (visual distractors)
        decoy_positions = []
        for p in outer[n:]:
            if len(decoy_positions) >= n_decoys:
                break
            if p not in used:
                dx2, dy2 = p
                grid.objects[dy2, dx2] = ObjectType.BOX  # BOX = visually distinct decoy
                decoy_positions.append(p)
                used.add(p)

        # Obstacle walls — flood-fill check
        wall_positions = []
        all_cells = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                     if (x, y) not in used]
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

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [target_positions[true_goal_idx]],
            "target_positions": target_positions,
            "switch_pos": switch_pos,
            "true_goal_idx": true_goal_idx,
            "decoy_positions": decoy_positions,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_rule_revealed"] = False
        self._last_rule_revealed = False
        self._config = config  # cache for on_agent_moved

    def on_agent_moved(self, pos, agent, grid):
        """Reveal rule when agent steps on switch — fires BEFORE reward/success."""
        config = getattr(self, "_config", {})
        sw = config.get("switch_pos")
        ax, ay = pos
        # Use grid.objects check (robust — avoids any x,y tuple ordering mismatch)
        if sw and grid.objects[ay, ax] == ObjectType.SWITCH and not config.get("_rule_revealed", False):
            # Reveal rule: mark true goal as GOAL (upgrade TARGET → GOAL)
            grid.objects[sw[1], sw[0]] = ObjectType.NONE
            true_idx = config.get("true_goal_idx", 0)
            targets = config.get("target_positions", [])
            if true_idx < len(targets):
                gx, gy = targets[true_idx]
                grid.objects[gy, gx] = ObjectType.GOAL
            config["_rule_revealed"] = True

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_revealed = config.get("_rule_revealed", False)
        if new_revealed and not self._last_rule_revealed:
            reward += 0.2  # reward for discovering the rule
        self._last_rule_revealed = new_revealed
        # Approach shaping: toward switch (before revealed) or toward true goal (after)
        if "agent_position" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            if not new_revealed:
                # Guide toward switch
                sw = config.get("switch_pos")
                if sw:
                    d_new = abs(ax - sw[0]) + abs(ay - sw[1])
                    d_old = abs(ox - sw[0]) + abs(oy - sw[1])
                    reward += 0.05 * (d_old - d_new)
            else:
                # Guide toward true goal (visible as GOAL on grid)
                if "grid" in new_state:
                    from agentick.core.types import ObjectType as OT
                    g = new_state["grid"]
                    goals = [(x, y) for y in range(g.height) for x in range(g.width)
                             if g.objects[y, x] == OT.GOAL]
                    if goals:
                        tgt = goals[0]
                        d_new = abs(ax - tgt[0]) + abs(ay - tgt[1])
                        d_old = abs(ox - tgt[0]) + abs(oy - tgt[1])
                        reward += 0.05 * (d_old - d_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Must visit switch first (to learn the rule), then correct target."""
        config = state.get("config", {})
        if not config.get("_rule_revealed", False):
            return False  # must visit switch before any target counts
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        # After rule revealed, the true goal is marked as GOAL on grid
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
