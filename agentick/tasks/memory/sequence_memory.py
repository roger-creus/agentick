"""SequenceMemory - Visit targets in the correct order.

BUG FIXED: on_agent_moved used (ax,ay)==next_target position comparison.
Now uses grid.objects[ay,ax] == GOAL check (robust to any coordinate ordering).

CREATIVE DIFFICULTY AXES:
  - easy:   3 targets, open map, all targets always visible (GOAL marks next)
  - medium: 4 targets, random obstacles, distractors (wrong TARGET cells that reset)
  - hard:   5 targets, targets SHUFFLE positions after each correct visit (memory test)
  - expert: 6 targets, targets hidden (only current GOAL visible), patrols, shuffling
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SequenceMemory-v0", tags=["memory", "pattern_recognition"])
class SequenceMemoryTask(TaskSpec):
    """Visit marked targets in the correct sequence order."""

    name = "SequenceMemory-v0"
    description = "Visit targets in sequence order"
    capability_tags = ["memory", "pattern_recognition"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=80,  params={"n_targets": 3, "n_distractors": 0, "n_obstacles": 0, "shuffle": False, "hide_future": False}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=160, params={"n_targets": 4, "n_distractors": 2, "n_obstacles": 4, "shuffle": False, "hide_future": False}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=280, params={"n_targets": 5, "n_distractors": 3, "n_obstacles": 6, "shuffle": True,  "hide_future": False}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=420, params={"n_targets": 6, "n_distractors": 4, "n_obstacles": 8, "shuffle": True,  "hide_future": True}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n           = p.get("n_targets", 3)
        n_dist      = p.get("n_distractors", 0)
        n_obs       = p.get("n_obstacles", 0)
        shuffle     = p.get("shuffle", False)
        hide_future = p.get("hide_future", False)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL; grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL; grid.terrain[:, -1] = CellType.WALL

        # Randomize agent start corner
        corners = [(1,1),(size-2,1),(1,size-2),(size-2,size-2)]
        rng.shuffle(corners)
        agent_pos = corners[0]

        # Add obstacles
        interior = [(x,y) for x in range(1,size-1) for y in range(1,size-1) if (x,y)!=agent_pos]
        rng.shuffle(interior)
        placed = 0
        for wx, wy in interior[:n_obs*3]:
            grid.terrain[wy, wx] = CellType.WALL
            if len(grid.flood_fill(agent_pos)) < n + n_dist + 2:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                placed += 1
                if placed >= n_obs: break

        reachable = list(grid.flood_fill(agent_pos) - {agent_pos})
        rng.shuffle(reachable)
        targets     = reachable[:n]
        distractors = reachable[n:n+n_dist]

        # Mark first target as GOAL, rest as TARGET (or hide if hide_future)
        for i, (tx, ty) in enumerate(targets):
            if i == 0:
                grid.objects[ty, tx] = ObjectType.GOAL
            elif not hide_future:
                grid.objects[ty, tx] = ObjectType.TARGET

        for dx, dy in distractors:
            grid.objects[dy, dx] = ObjectType.SWITCH  # distractor visual

        return grid, {
            "agent_start":   agent_pos,
            "goal_positions": targets,
            "sequence":      targets,
            "distractors":   distractors,
            "shuffle":       shuffle,
            "hide_future":   hide_future,
            "_rng_seed":     int(rng.integers(0, 2**31)),
            "max_steps":     self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_seq_progress"] = 0
        config["_wrong_visit"]  = False
        config["_shuffle_rng"]  = np.random.default_rng(config.get("_rng_seed", 0))
        self._config = config
        self._last_seq_progress = 0

    def on_agent_moved(self, pos, agent, grid):
        """Use grid.objects check — robust to any x,y ordering."""
        config = getattr(self, "_config", {})
        x, y = pos
        obj = grid.objects[y, x]
        progress  = config.get("_seq_progress", 0)
        sequence  = config.get("sequence", [])
        hide_fut  = config.get("hide_future", False)
        shuffle   = config.get("shuffle", False)

        if progress >= len(sequence):
            return

        if obj == ObjectType.GOAL:
            # Correct next target
            grid.objects[y, x] = ObjectType.NONE
            config["_seq_progress"] = progress + 1
            new_progress = progress + 1

            if shuffle and new_progress < len(sequence):
                # Shuffle remaining target positions
                rng = config.get("_shuffle_rng")
                remaining = [(tx, ty) for tx, ty in sequence[new_progress:]
                             if (tx, ty) != (x, y)]
                if rng and remaining:
                    idxs = list(range(len(remaining)))
                    rng.shuffle(idxs)
                    new_order = [remaining[i] for i in idxs]
                    # Update sequence with shuffled remainder
                    new_seq = list(sequence[:new_progress]) + new_order
                    config["sequence"] = new_seq
                    sequence = new_seq

            # Show next target
            if new_progress < len(sequence):
                nx, ny = sequence[new_progress]
                grid.objects[ny, nx] = ObjectType.GOAL
                if hide_fut:
                    # Hide all future ones
                    for fx, fy in sequence[new_progress+1:]:
                        if grid.objects[fy, fx] == ObjectType.TARGET:
                            grid.objects[fy, fx] = ObjectType.NONE
                else:
                    for fx, fy in sequence[new_progress+1:]:
                        if grid.objects[fy, fx] == ObjectType.NONE:
                            grid.objects[fy, fx] = ObjectType.TARGET

        elif obj == ObjectType.SWITCH:
            # Distractor — penalty flag
            config["_wrong_visit"] = True
            grid.objects[y, x] = ObjectType.NONE  # consume distractor

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        if config.get("_wrong_visit", False) and not old_state.get("config", {}).get("_wrong_visit", False):
            reward -= 0.3
        new_progress = config.get("_seq_progress", 0)
        if new_progress > self._last_seq_progress:
            reward += 0.3 * (new_progress - self._last_seq_progress)
        self._last_seq_progress = new_progress
        if "agent" in new_state and "grid" in new_state:
            g = new_state["grid"]
            goals = [(x, y) for y in range(g.height) for x in range(g.width)
                     if g.objects[y, x] == ObjectType.GOAL]
            if goals:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                tgt = goals[0]
                reward += 0.05 * ((abs(ox-tgt[0])+abs(oy-tgt[1])) - (abs(ax-tgt[0])+abs(ay-tgt[1])))
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        progress = config.get("_seq_progress", 0)
        sequence = config.get("sequence", [])
        return len(sequence) > 0 and progress >= len(sequence)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
