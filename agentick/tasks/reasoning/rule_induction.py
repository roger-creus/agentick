"""RuleInduction - XLand-style Combination Discovery.

MECHANICS:
  - Grid has various objects (GEM, POTION, SCROLL, COIN, ORB) scattered on walkable cells.
  - Hidden combination rules map pairs of objects to result objects.
  - Agent walks onto an object to pick it up (one item at a time, stored in inventory).
  - While carrying an item, walking onto another object attempts a combination:
      - If (carried_type, stepped_on_type) is in the rule table:
          both consumed, RESULT object placed at that grid position.
      - If NOT in rule table: both consumed (destructive), nothing placed.
      - Inventory cleared either way.
  - If the result placed is the target object, agent walks onto it again to collect it.
  - Multi-trial system: episode has N trials, each with a step sub-budget. After the
    sub-budget expires, grid objects reset to original positions, inventory clears, trial
    counter increments. Agent retains knowledge across trials but not inventory.

DIFFICULTY:
  - easy:   9x9,  4 objects,  2 valid combos, 5 trials, max_steps=150
  - medium: 11x11, 5 objects, 3 valid combos, 5 trials, max_steps=250
  - hard:   13x13, 5 objects, 4 valid combos, 5 trials, max_steps=400
  - expert: 15x15, 5 objects, 5 valid combos, 5 trials, max_steps=550
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.entity import Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# All object types available for combination rules
_COMBO_TYPES = [
    ObjectType.GEM,
    ObjectType.POTION,
    ObjectType.SCROLL,
    ObjectType.COIN,
    ObjectType.ORB,
]

# Integer values for quick lookup
_COMBO_TYPE_INTS = [int(t) for t in _COMBO_TYPES]


@register_task("RuleInduction-v0", tags=["reasoning", "rule_learning"])
class RuleInductionTask(TaskSpec):
    """Discover hidden combination rules through experimentation and craft a target object."""

    name = "RuleInduction-v0"
    description = (
        "Discover hidden object combination rules through experimentation "
        "and craft the target object"
    )
    capability_tags = ["reasoning", "rule_learning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=150,
            params={
                "n_objects": 4,
                "n_valid_combos": 2,
                "n_trials": 5,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=250,
            params={
                "n_objects": 5,
                "n_valid_combos": 3,
                "n_trials": 5,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=400,
            params={
                "n_objects": 5,
                "n_valid_combos": 4,
                "n_trials": 5,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=550,
            params={
                "n_objects": 5,
                "n_valid_combos": 5,
                "n_trials": 5,
            },
        ),
    }

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(self, seed: int):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        n_objects: int = params["n_objects"]
        n_valid_combos: int = params["n_valid_combos"]
        n_trials: int = params["n_trials"]
        max_steps = self.get_max_steps()
        trial_budget = max_steps // n_trials

        grid = Grid(size, size)

        # Border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Sparse interior walls (at most 10% of interior cells)
        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
        ]
        n_walls = max(0, int(len(interior) * 0.08))
        rng.shuffle(interior)
        wall_positions: set[tuple[int, int]] = set()
        for pos in interior[:n_walls]:
            wall_positions.add(pos)
            wx, wy = pos
            grid.terrain[wy, wx] = CellType.WALL

        # Agent start in a random corner (avoid walls)
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)
        agent_pos = corners[0]
        ax, ay = agent_pos
        # Ensure agent start is not a wall
        grid.terrain[ay, ax] = CellType.EMPTY

        # ---- Rule table generation ----------------------------------------
        # Pick target type randomly from the 5 combo types
        type_indices = list(range(len(_COMBO_TYPES)))
        rng.shuffle(type_indices)
        target_type = _COMBO_TYPES[type_indices[0]]

        # Build a chain of rules that produces the target:
        #   (A, B) -> C, (C, D) -> E, ..., last result = target_type
        # Then add additional valid combos (not necessarily chained).
        rule_table: dict[tuple[int, int], int] = {}

        # Build at least one chain that ends at target_type
        chain_length = max(1, n_valid_combos // 2)
        chain_types = self._build_chain(rng, target_type, chain_length)
        for a_type, b_type, result_type in chain_types:
            key = (int(a_type), int(b_type))
            rule_table[key] = int(result_type)

        # Add more valid combos up to n_valid_combos
        remaining = list(_COMBO_TYPES)
        rng.shuffle(remaining)
        attempts = 0
        while len(rule_table) < n_valid_combos and attempts < 100:
            attempts += 1
            a = _COMBO_TYPES[int(rng.integers(len(_COMBO_TYPES)))]
            b = _COMBO_TYPES[int(rng.integers(len(_COMBO_TYPES)))]
            c = _COMBO_TYPES[int(rng.integers(len(_COMBO_TYPES)))]
            key = (int(a), int(b))
            if key not in rule_table:
                rule_table[key] = int(c)

        # ---- Object placement --------------------------------------------
        # Collect walkable cells (not agent_pos, not walls)
        reachable = self._bfs_reachable(grid, agent_pos)
        candidates = sorted(reachable - {agent_pos})
        rng.shuffle(candidates)

        # Determine which object types to place.
        # For every step in the chain, ensure the non-intermediate inputs are
        # present so that the agent can execute the combo.
        # Step 0: inputs (A, B) → intermediate C (place A and B)
        # Step 1: (C, D) → TARGET (place D; C comes from step 0 result)
        # In general: place every input that is NOT the result of a prior step.
        placed_objects: list[tuple[int, int, ObjectType]] = []

        prior_results: set[int] = set()
        must_place: list[ObjectType] = []
        for a_type, b_type, result_type in chain_types:
            for t in (a_type, b_type):
                if int(t) not in prior_results:
                    must_place.append(t)
            prior_results.add(int(result_type))
        # Deduplicate while preserving order; exclude target_type
        seen: set[int] = set()
        deduped: list[ObjectType] = []
        for t in must_place:
            if int(t) not in seen and t != target_type:
                seen.add(int(t))
                deduped.append(t)
        must_place = deduped

        used_cells: set[tuple[int, int]] = {agent_pos}
        obj_idx = 0

        for required_type in must_place:
            if obj_idx >= len(candidates):
                break
            pos = candidates[obj_idx]
            while pos in used_cells and obj_idx < len(candidates) - 1:
                obj_idx += 1
                pos = candidates[obj_idx]
            if pos not in used_cells:
                px, py = pos
                grid.objects[py, px] = required_type
                placed_objects.append((px, py, required_type))
                used_cells.add(pos)
                obj_idx += 1

        # Track which types are already on the grid (enforce 1 instance per type)
        placed_types = {int(ot) for _, _, ot in placed_objects}

        # Fill remaining object slots with random types (each type used at most once)
        while len(placed_objects) < n_objects and obj_idx < len(candidates):
            remaining_types = [
                t for t in _COMBO_TYPES
                if t != target_type and int(t) not in placed_types
            ]
            if not remaining_types:
                break
            pos = candidates[obj_idx]
            obj_idx += 1
            if pos in used_cells:
                continue
            px, py = pos
            chosen_type = remaining_types[int(rng.integers(len(remaining_types)))]
            placed_types.add(int(chosen_type))
            grid.objects[py, px] = chosen_type
            placed_objects.append((px, py, chosen_type))
            used_cells.add(pos)

        # Serialize rule table with string keys for JSON-safe storage
        rule_table_list = [
            [int(a), int(b), int(c)] for (a, b), c in rule_table.items()
        ]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "max_steps": max_steps,
            "_target_type": int(target_type),
            "_rule_table_list": rule_table_list,
            "_original_objects": [[px, py, int(ot)] for px, py, ot in placed_objects],
            "_n_trials": n_trials,
            "_trial_step_budget": trial_budget,
            # Runtime state — initialised in on_env_reset
            "_current_trial": 0,
            "_trial_steps_used": 0,
            "_target_crafted": False,
            "_valid_combos_made": 0,
            "_destructive_combos_made": 0,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_chain(
        rng: np.random.Generator,
        target_type: ObjectType,
        chain_length: int,
    ) -> list[tuple[ObjectType, ObjectType, ObjectType]]:
        """Build a combo chain whose last step produces target_type.

        Returns a list of (A, B, result) triples.
        Each step's inputs are chosen so they haven't been used as inputs
        in prior steps, ensuring no duplicate object types on the grid.
        """
        rules: list[tuple[ObjectType, ObjectType, ObjectType]] = []
        result = target_type
        used_inputs: set[int] = set()  # types already used as inputs in prior steps

        for step in range(chain_length, 0, -1):
            # Pick two inputs that produce result, excluding the target type
            # (it must NEVER appear as an initial object) and preferring types
            # not yet used as inputs in other chain steps
            available = [
                t for t in _COMBO_TYPES
                if t != result and t != target_type and int(t) not in used_inputs
            ]
            if len(available) < 2:
                # Fall back: still exclude target_type but allow reused inputs
                available = [
                    t for t in _COMBO_TYPES if t != result and t != target_type
                ]
            if len(available) < 2:
                available = list(_COMBO_TYPES)
            rng.shuffle(available)
            a = available[0]
            b = available[1] if len(available) > 1 else available[0]
            used_inputs.add(int(a))
            used_inputs.add(int(b))
            rules.insert(0, (a, b, result))
            # For the next iteration, result becomes the output of the
            # previous step — use a fresh type as the new intermediate
            if step > 1:
                others = [
                    t for t in _COMBO_TYPES
                    if t not in (a, b, result, target_type)
                ]
                if others:
                    rng.shuffle(others)
                    result = others[0]
                else:
                    result = _COMBO_TYPES[int(rng.integers(len(_COMBO_TYPES)))]

        return rules

    @staticmethod
    def _bfs_reachable(grid: Grid, start: tuple[int, int]) -> set[tuple[int, int]]:
        """Return the set of walkable cells reachable from start (ignores objects)."""
        visited: set[tuple[int, int]] = set()
        sx, sy = start
        if not grid.in_bounds(start):
            return visited
        if grid.terrain[sy, sx] == CellType.WALL:
            return visited
        visited.add(start)
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nb = (x + dx, y + dy)
                nx, ny = nb
                if nb in visited or not grid.in_bounds(nb):
                    continue
                if grid.terrain[ny, nx] == CellType.WALL:
                    continue
                visited.add(nb)
                queue.append(nb)
        return visited

    @staticmethod
    def _rebuild_rule_table(config: dict) -> dict[tuple[int, int], int]:
        """Reconstruct the rule table dict from the serialisable list in config."""
        return {
            (int(row[0]), int(row[1])): int(row[2])
            for row in config.get("_rule_table_list", [])
        }

    @staticmethod
    def _restore_objects(grid: Grid, config: dict) -> None:
        """Clear all objects from the grid and restore originals."""
        # Erase all objects first
        grid.objects[:, :] = ObjectType.NONE
        # Re-place originals
        for entry in config.get("_original_objects", []):
            px, py, ot = int(entry[0]), int(entry[1]), int(entry[2])
            grid.objects[py, px] = ot

    # ------------------------------------------------------------------ #
    # Runtime hooks
    # ------------------------------------------------------------------ #

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        config["_current_trial"] = 0
        config["_trial_steps_used"] = 0
        config["_target_crafted"] = False
        config["_valid_combos_made"] = 0
        config["_destructive_combos_made"] = 0
        self._config = config
        self._prev_valid_combos = 0
        self._prev_destructive_combos = 0
        # Restore original objects on the grid (in case of re-reset)
        self._restore_objects(grid, config)

    def on_agent_moved(self, pos, agent, grid):
        """Handle object pickup and combination when agent steps on an object."""
        config = getattr(self, "_config", {})
        x, y = pos
        obj_val = int(grid.objects[y, x])

        if obj_val == int(ObjectType.NONE):
            return

        target_type = config.get("_target_type", -1)
        target_crafted = config.get("_target_crafted", False)

        # ---- Collect a crafted target that was placed on the grid --------
        if target_crafted and obj_val == target_type:
            # Agent steps on a previously crafted target — collect it
            grid.objects[y, x] = ObjectType.NONE
            # Mark success via a distinct flag
            config["_target_collected"] = True
            agent.inventory.clear()
            return

        # ---- No item in inventory: pick up --------------------------------
        if len(agent.inventory) == 0:
            if obj_val in _COMBO_TYPE_INTS:
                grid.objects[y, x] = ObjectType.NONE
                item = Entity(
                    id=f"item_{x}_{y}",
                    entity_type="item",
                    position=pos,
                    properties={"object_type": obj_val},
                )
                agent.inventory.append(item)
            return

        # ---- Item in inventory: attempt combination -----------------------
        carried_type = agent.inventory[0].properties.get("object_type", -1)
        ground_type = obj_val
        agent.inventory.clear()

        # Remove the ground object regardless of outcome
        grid.objects[y, x] = ObjectType.NONE

        rule_table = self._rebuild_rule_table(config)
        result_type = rule_table.get((carried_type, ground_type), None)

        if result_type is not None:
            # Valid combination: place result at a random empty walkable cell
            config["_valid_combos_made"] = config.get("_valid_combos_made", 0) + 1
            # Find all empty walkable cells to place the result
            empty_cells = []
            for cy in range(grid.height):
                for cx in range(grid.width):
                    if (
                        int(grid.terrain[cy, cx]) == int(CellType.EMPTY)
                        and int(grid.objects[cy, cx]) == int(ObjectType.NONE)
                        and (cx, cy) != (x, y)
                    ):
                        empty_cells.append((cx, cy))
            if empty_cells:
                # Use a seeded pick based on step count for determinism
                rng_idx = (x * 31 + y * 17 + config.get("_trial_steps_used", 0)) % len(empty_cells)
                rx, ry = empty_cells[rng_idx]
                grid.objects[ry, rx] = result_type
            else:
                # Fallback: place at combination position
                grid.objects[y, x] = result_type
            # Check if the result is the target
            if result_type == target_type:
                config["_target_crafted"] = True
        else:
            # Destructive: both consumed, nothing placed
            config["_destructive_combos_made"] = (
                config.get("_destructive_combos_made", 0) + 1
            )

    def on_env_step(self, agent, grid, config, step_count):
        """Tick trial step budget; reset grid on trial timeout."""
        self._config = config

        # Update trial step counter
        config["_trial_steps_used"] = config.get("_trial_steps_used", 0) + 1

        budget = config.get("_trial_step_budget", 1)
        current_trial = config.get("_current_trial", 0)

        # Success check — do not advance trial if already done
        if config.get("_target_collected", False):
            return

        if config["_trial_steps_used"] >= budget:
            # Trial expired: reset grid objects and inventory, advance trial
            self._restore_objects(grid, config)
            agent.inventory.clear()
            config["_current_trial"] = current_trial + 1
            config["_trial_steps_used"] = 0
            config["_target_crafted"] = False

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # step penalty
        config = new_state.get("config", {})
        old_config = old_state.get("config", {})

        # +0.2 per new valid combination
        new_valid = config.get("_valid_combos_made", 0)
        old_valid = old_config.get("_valid_combos_made", 0)
        delta_valid = new_valid - old_valid
        if delta_valid > 0:
            reward += 0.2 * delta_valid

        # -0.1 per destructive combination
        new_dest = config.get("_destructive_combos_made", 0)
        old_dest = old_config.get("_destructive_combos_made", 0)
        delta_dest = new_dest - old_dest
        if delta_dest > 0:
            reward -= 0.1 * delta_dest

        # +1.0 for crafting and collecting the target
        if self.check_success(new_state):
            reward += 1.0

        return reward

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def check_success(self, state):
        config = state.get("config", {})
        return bool(config.get("_target_collected", False))

    def check_done(self, state):
        if self.check_success(state):
            return True
        config = state.get("config", {})
        current_trial = config.get("_current_trial", 0)
        n_trials = config.get("_n_trials", 1)
        return current_trial >= n_trials

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
