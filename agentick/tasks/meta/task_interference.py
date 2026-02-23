"""TaskInterference - Complete competing/contradictory objectives.

MECHANICS:
  - Two competing resource pools: COIN vs GEM
  - Collecting a COIN destroys the nearest uncollected GEM (and vice versa)
  - Agent must collect ALL of one type and deliver to a matching GOAL
  - Then switch and collect ALL of the other type to its GOAL
  - The destruction mechanic means order matters: collecting wrong type first
    can make the task unsolvable
  - Interference walls appear when first goal is completed
  - Physical interference: at medium+, collecting a COIN places a WALL where
    the destroyed GEM was; at hard+, collecting a GEM also places a WALL where
    the destroyed COIN was. Walls are only placed if remaining items stay
    reachable from the agent position.
  - Tests planning under competing objectives and interference resistance
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TaskInterference-v0", tags=["multi_task", "interference", "meta_learning"])
class TaskInterferenceTask(TaskSpec):
    """Complete competing objectives where progress on one harms the other."""

    name = "TaskInterference-v0"
    description = "Complete competing objectives with destructive interference"
    capability_tags = ["multi_task", "interference_resistance", "attention"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=150,
            params={
                "n_coins": 2,
                "n_gems": 2,
                "interference": False,
                "wall_on_coin_pickup": False,
                "wall_on_gem_pickup": False,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=250,
            params={
                "n_coins": 3,
                "n_gems": 3,
                "interference": True,
                "wall_on_coin_pickup": True,
                "wall_on_gem_pickup": False,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=400,
            params={
                "n_coins": 4,
                "n_gems": 4,
                "interference": True,
                "wall_on_coin_pickup": True,
                "wall_on_gem_pickup": True,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=600,
            params={
                "n_coins": 5,
                "n_gems": 5,
                "interference": True,
                "wall_on_coin_pickup": True,
                "wall_on_gem_pickup": True,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_coins = self.difficulty_config.params.get("n_coins", 2)
        n_gems = self.difficulty_config.params.get("n_gems", 2)
        interference = self.difficulty_config.params.get("interference", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (size // 2, size // 2)

        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) != agent_pos
        ]
        rng.shuffle(free)
        used = {agent_pos}

        # Place coin delivery goal (left side)
        coin_goal = (1, size // 2)
        grid.objects[coin_goal[1], coin_goal[0]] = ObjectType.GOAL
        grid.metadata[coin_goal[1], coin_goal[0]] = int(ObjectType.COIN)
        used.add(coin_goal)

        # Place gem delivery goal (right side)
        gem_goal = (size - 2, size // 2)
        grid.objects[gem_goal[1], gem_goal[0]] = ObjectType.GOAL
        grid.metadata[gem_goal[1], gem_goal[0]] = int(ObjectType.GEM)
        used.add(gem_goal)

        # With interference, each pickup destroys an item of the other type,
        # so we must place extra items to keep the task solvable.
        place_coins = n_coins + n_gems if interference else n_coins
        place_gems = n_gems + n_coins if interference else n_gems

        # Place coins scattered
        coin_positions = []
        coin_candidates = [p for p in free if p not in used]
        rng.shuffle(coin_candidates)
        for p in coin_candidates[:place_coins]:
            cx, cy = p
            grid.objects[cy, cx] = ObjectType.COIN
            coin_positions.append(p)
            used.add(p)

        # Place gems scattered
        gem_positions = []
        gem_candidates = [p for p in free if p not in used]
        rng.shuffle(gem_candidates)
        for p in gem_candidates[:place_gems]:
            gx, gy = p
            grid.objects[gy, gx] = ObjectType.GEM
            gem_positions.append(p)
            used.add(p)

        # Interference wall positions (activated when first objective is completed)
        interference_walls = []
        if interference:
            # Wall positions between the two goals
            mid = size // 2
            for y in range(2, size - 2):
                if y != mid:
                    wx = mid
                    if (wx, y) not in used and grid.terrain[y, wx] == CellType.EMPTY:
                        interference_walls.append((wx, y))

        wall_on_coin = self.difficulty_config.params.get("wall_on_coin_pickup", False)
        wall_on_gem = self.difficulty_config.params.get("wall_on_gem_pickup", False)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [coin_goal, gem_goal],
            "coin_goal": coin_goal,
            "gem_goal": gem_goal,
            "coin_positions": coin_positions,
            "gem_positions": gem_positions,
            "n_coins": n_coins,
            "n_gems": n_gems,
            "interference": interference,
            "wall_on_coin_pickup": wall_on_coin,
            "wall_on_gem_pickup": wall_on_gem,
            "interference_walls": interference_walls,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_coins_held"] = 0
        config["_gems_held"] = 0
        config["_coins_delivered"] = 0
        config["_gems_delivered"] = 0
        config["_first_objective_done"] = False
        config["_live_coins"] = list(config.get("coin_positions", []))
        config["_live_gems"] = list(config.get("gem_positions", []))
        self._last_delivered = 0
        self._config = config

    def _check_all_reachable(self, grid, agent_pos, positions):
        """Check that all positions in the list are reachable from agent_pos."""
        if not positions:
            return True
        reachable = grid.flood_fill(agent_pos)
        return all(p in reachable for p in positions)

    def _try_place_interference_wall(self, grid, wall_pos, agent_pos, config):
        """Place a wall at wall_pos if remaining items stay reachable from agent.

        Returns True if the wall was placed, False if skipped.
        """
        wx, wy = wall_pos
        # Only place on empty terrain that has no object
        if grid.terrain[wy, wx] != CellType.EMPTY:
            return False
        if grid.objects[wy, wx] != ObjectType.NONE:
            return False

        # Tentatively place the wall
        grid.terrain[wy, wx] = CellType.WALL

        # Collect all positions that must remain reachable: live coins, live gems,
        # coin goal, gem goal
        must_reach = []
        must_reach.extend(config.get("_live_coins", []))
        must_reach.extend(config.get("_live_gems", []))
        coin_goal = config.get("coin_goal")
        gem_goal = config.get("gem_goal")
        if coin_goal:
            must_reach.append(coin_goal)
        if gem_goal:
            must_reach.append(gem_goal)

        if self._check_all_reachable(grid, agent_pos, must_reach):
            return True

        # Revert — wall would block a needed path
        grid.terrain[wy, wx] = CellType.EMPTY
        return False

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos
        obj = grid.objects[y, x]
        interference = config.get("interference", False)
        wall_on_coin = config.get("wall_on_coin_pickup", False)
        wall_on_gem = config.get("wall_on_gem_pickup", False)

        # Collect coin
        if obj == ObjectType.COIN:
            grid.objects[y, x] = ObjectType.NONE
            config["_coins_held"] = config.get("_coins_held", 0) + 1
            if (x, y) in config.get("_live_coins", []):
                config["_live_coins"].remove((x, y))

            # Interference: collecting a coin destroys nearest uncollected gem
            if interference:
                live_gems = config.get("_live_gems", [])
                if live_gems:
                    nearest = min(live_gems, key=lambda g: abs(g[0] - x) + abs(g[1] - y))
                    gx, gy = nearest
                    if grid.objects[gy, gx] == ObjectType.GEM:
                        grid.objects[gy, gx] = ObjectType.NONE
                    live_gems.remove(nearest)
                    config["_live_gems"] = live_gems

                    # Physical interference: place wall where the gem was destroyed
                    if wall_on_coin:
                        self._try_place_interference_wall(
                            grid, (gx, gy), pos, config
                        )

        # Collect gem
        elif obj == ObjectType.GEM:
            grid.objects[y, x] = ObjectType.NONE
            config["_gems_held"] = config.get("_gems_held", 0) + 1
            if (x, y) in config.get("_live_gems", []):
                config["_live_gems"].remove((x, y))

            # Interference: collecting a gem destroys nearest uncollected coin
            if interference:
                live_coins = config.get("_live_coins", [])
                if live_coins:
                    nearest = min(live_coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
                    cx, cy = nearest
                    if grid.objects[cy, cx] == ObjectType.COIN:
                        grid.objects[cy, cx] = ObjectType.NONE
                    live_coins.remove(nearest)
                    config["_live_coins"] = live_coins

                    # Physical interference: place wall where the coin was destroyed
                    if wall_on_gem:
                        self._try_place_interference_wall(
                            grid, (cx, cy), pos, config
                        )

        # Deliver at coin goal
        elif obj == ObjectType.GOAL:
            meta = int(grid.metadata[y, x])
            if meta == int(ObjectType.COIN) and config.get("_coins_held", 0) > 0:
                delivered = config.get("_coins_held", 0)
                config["_coins_delivered"] = config.get("_coins_delivered", 0) + delivered
                config["_coins_held"] = 0
                # Check if first objective done
                if config.get("_coins_delivered", 0) >= config.get("n_coins", 2) and not config.get(
                    "_first_objective_done", False
                ):
                    config["_first_objective_done"] = True
                    self._activate_interference_walls(grid, config)

            elif meta == int(ObjectType.GEM) and config.get("_gems_held", 0) > 0:
                delivered = config.get("_gems_held", 0)
                config["_gems_delivered"] = config.get("_gems_delivered", 0) + delivered
                config["_gems_held"] = 0
                if config.get("_gems_delivered", 0) >= config.get("n_gems", 2) and not config.get(
                    "_first_objective_done", False
                ):
                    config["_first_objective_done"] = True
                    self._activate_interference_walls(grid, config)

    def _activate_interference_walls(self, grid, config):
        """Place interference walls when first objective is completed."""
        walls = config.get("interference_walls", [])
        for wx, wy in walls:
            if grid.terrain[wy, wx] == CellType.EMPTY and grid.objects[wy, wx] == ObjectType.NONE:
                grid.terrain[wy, wx] = CellType.WALL

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        total_delivered = config.get("_coins_delivered", 0) + config.get("_gems_delivered", 0)
        if total_delivered > self._last_delivered:
            reward += 0.3 * (total_delivered - self._last_delivered)
        self._last_delivered = total_delivered

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        n_coins = config.get("n_coins", 2)
        n_gems = config.get("n_gems", 2)
        coins_ok = config.get("_coins_delivered", 0) >= n_coins
        gems_ok = config.get("_gems_delivered", 0) >= n_gems
        return coins_ok and gems_ok

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
