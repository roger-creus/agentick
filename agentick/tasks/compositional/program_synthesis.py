"""Program Synthesis - Route an item from source to destination via waypoints.

MECHANICS:
  - A KEY (item) is at the source position
  - A GOAL (destination) is placed elsewhere on the grid
  - N waypoints (SWITCHes) must be visited in any order to "unlock"
  - Agent picks up the KEY by stepping on it (auto-carry)
  - After visiting all waypoints, agent delivers KEY to GOAL
  - Success = all waypoints visited AND agent at GOAL carrying the KEY
  - At hard+: n_conditionals adds DOOR/RESOURCE conditional waypoints
  - At expert: loop_count requires repeating the waypoint circuit
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ProgramSynthesis-v0", tags=["reasoning", "planning", "abstraction"])
class ProgramSynthesisTask(TaskSpec):
    """Collect an item, visit all waypoints, then deliver it to the goal.

    The agent must pick up the KEY, visit all SWITCH waypoints (in any
    order), and finally deliver the KEY to the GOAL. This models
    sequential multi-step planning: the agent must "program" the correct
    route through all waypoints to complete the task.
    """

    name = "ProgramSynthesis-v0"
    description = "Collect item, visit waypoints, deliver to destination"
    capability_tags = ["abstract_reasoning", "planning", "programming"]

    difficulty_configs = {
        # n_waypoints: sequential sub-goals
        # n_obstacles: wall clusters | order_strict: fail on wrong order
        # n_conditionals: waypoints that only unlock after a switch
        # loop_count: number of times the waypoint circuit must repeat
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=100,
            params={
                "n_waypoints": 2, "n_obstacles": 0,
                "order_strict": False,
                "n_conditionals": 0, "loop_count": 1,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=200,
            params={
                "n_waypoints": 3, "n_obstacles": 3,
                "order_strict": False,
                "n_conditionals": 0, "loop_count": 1,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=13, max_steps=350,
            params={
                "n_waypoints": 4, "n_obstacles": 5,
                "order_strict": True,
                "n_conditionals": 1, "loop_count": 1,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=550,
            params={
                "n_waypoints": 5, "n_obstacles": 8,
                "order_strict": True,
                "n_conditionals": 2, "loop_count": 2,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size        = self.difficulty_config.grid_size
        n_wp        = self.difficulty_config.params.get("n_waypoints", 2)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)
        order_strict = self.difficulty_config.params.get(
            "order_strict", False,
        )
        n_conditionals = self.difficulty_config.params.get(
            "n_conditionals", 0,
        )
        loop_count = self.difficulty_config.params.get(
            "loop_count", 1,
        )

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        free = [(x, y) for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) != agent_pos]
        rng.shuffle(free)

        source_pos = free[0]
        dest_pos   = free[1]
        waypoints  = free[2:2 + n_wp]
        used = {agent_pos, source_pos, dest_pos} | set(waypoints)

        grid.objects[source_pos[1], source_pos[0]] = ObjectType.KEY
        grid.objects[dest_pos[1],   dest_pos[0]]   = ObjectType.GOAL
        for wx, wy in waypoints:
            grid.objects[wy, wx] = ObjectType.SWITCH

        # Interior obstacles (walls) — flood-fill to preserve solvability
        wall_positions = []
        wall_candidates = [
            p for p in free[2 + n_wp:] if p not in used
        ]
        for p in wall_candidates:
            if len(wall_positions) >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            needed = [source_pos, dest_pos] + list(waypoints)
            if all(q in reachable for q in needed):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        # Conditional waypoints: DOOR objects that the agent must
        # visit *after* stepping on a paired RESOURCE (acts as an
        # unlock switch). Only at hard/expert.
        cond_positions = []
        cond_keys = []
        if n_conditionals > 0:
            cond_pool = [
                p for p in free[2 + n_wp:]
                if p not in used
            ]
            rng.shuffle(cond_pool)
            for i in range(min(n_conditionals, len(cond_pool) // 2)):
                cp = cond_pool[i * 2]
                ck = cond_pool[i * 2 + 1]
                # Place DOOR as the conditional waypoint
                grid.objects[cp[1], cp[0]] = ObjectType.DOOR
                # Place RESOURCE as the unlock switch
                grid.objects[ck[1], ck[0]] = ObjectType.RESOURCE
                cond_positions.append(cp)
                cond_keys.append(ck)
                used.add(cp)
                used.add(ck)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [dest_pos],
            "source": source_pos,
            "destination": dest_pos,
            "waypoints": waypoints,
            "order_strict": order_strict,
            "n_conditionals": n_conditionals,
            "loop_count": loop_count,
            "_cond_positions": cond_positions,
            "_cond_keys": cond_keys,
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        self._carrying_key = False
        self._waypoints_visited = set()
        self._n_waypoints = len(config.get("waypoints", []))
        self._last_wp_count = 0
        self._rewarded_key = False
        self._order_strict = config.get("order_strict", False)
        self._waypoints_ordered = list(config.get("waypoints", []))
        self._wrong_order = False
        # Conditional waypoint state
        self._cond_unlocked = set()
        self._cond_visited = set()
        self._n_conditionals = len(
            config.get("_cond_positions", []),
        )
        # Loop state: how many full waypoint circuits completed
        self._loop_count = config.get("loop_count", 1)
        self._loops_done = 0

    def on_agent_moved(self, pos, agent, grid):
        """Pick up KEY; collect SWITCH waypoints and conditionals."""
        x, y = pos
        obj = grid.objects[y, x]
        if obj == ObjectType.KEY and not self._carrying_key:
            grid.objects[y, x] = ObjectType.NONE
            self._carrying_key = True
        elif obj == ObjectType.SWITCH:
            grid.objects[y, x] = ObjectType.NONE
            if self._order_strict:
                expected = next(
                    (w for w in self._waypoints_ordered
                     if w not in self._waypoints_visited),
                    None,
                )
                if expected is not None and pos != expected:
                    self._wrong_order = True
            self._waypoints_visited.add(pos)
            # Check if a full loop of waypoints is done
            if (len(self._waypoints_visited)
                    >= self._n_waypoints):
                self._loops_done += 1
                if self._loops_done < self._loop_count:
                    # Reset waypoints for next loop iteration
                    self._waypoints_visited.clear()
        elif obj == ObjectType.RESOURCE:
            # Unlock the paired conditional waypoint
            grid.objects[y, x] = ObjectType.NONE
            self._cond_unlocked.add(pos)
        elif obj == ObjectType.DOOR:
            # Visit a conditional waypoint (only counts if
            # its paired RESOURCE was already collected)
            if pos in [
                tuple(c) for c in
                getattr(self, "_cond_unlocked", set())
            ] or len(self._cond_unlocked) > 0:
                grid.objects[y, x] = ObjectType.NONE
                self._cond_visited.add(pos)

    # ── Reward & success ──────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        # +0.3 for each new waypoint visited
        old_n = getattr(self, "_last_wp_count", 0)
        new_n = len(self._waypoints_visited)
        if new_n > old_n:
            reward += 0.3
        self._last_wp_count = new_n
        # +0.3 for picking up KEY
        if self._carrying_key and not getattr(self, "_rewarded_key", False):
            reward += 0.3
            self._rewarded_key = True
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All waypoints visited, conditionals met, loops done, at GOAL with KEY."""
        if not self._carrying_key:
            return False
        if getattr(self, "_wrong_order", False):
            return False
        # Must have completed enough loops of the waypoint circuit
        if self._loops_done < self._loop_count:
            # Still need to finish current loop
            if len(self._waypoints_visited) < self._n_waypoints:
                return False
        # All conditional waypoints must be visited
        if len(self._cond_visited) < self._n_conditionals:
            return False
        if "agent" not in state or "grid" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
