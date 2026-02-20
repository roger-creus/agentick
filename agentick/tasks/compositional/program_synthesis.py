"""ProgramSynthesis - Discover and replicate a hidden pattern/function.

MECHANICS:
  - Grid has "example" zones: input positions (SCROLL) and output positions (GEM)
  - The pattern maps inputs to outputs via a spatial transformation
    (e.g., "move 2 right", "mirror across center", "rotate 90 degrees")
  - A "test" zone has input positions (SCROLL) but missing outputs
  - Agent must place ORB objects at the correct output positions (predicted by the pattern)
  - ORB items are scattered on the grid; agent carries one at a time
  - Success = all test outputs correctly placed
  - Tests pattern recognition, abstraction, and generalization
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ProgramSynthesis-v0", tags=["reasoning", "planning", "abstraction"])
class ProgramSynthesisTask(TaskSpec):
    """Discover hidden spatial pattern from examples and replicate it."""

    name = "ProgramSynthesis-v0"
    description = "Discover hidden pattern from examples and replicate it"
    capability_tags = ["abstract_reasoning", "planning", "programming"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=9, max_steps=120,
            params={"n_examples": 2, "n_tests": 1, "pattern": "translate"},
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=11, max_steps=200,
            params={"n_examples": 2, "n_tests": 2, "pattern": "translate"},
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=350,
            params={"n_examples": 3, "n_tests": 2, "pattern": "mirror"},
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=550,
            params={"n_examples": 3, "n_tests": 3, "pattern": "rotate"},
        ),
    }

    def _apply_pattern(self, pos, pattern_type, dx, dy, center_x, center_y):
        """Apply a spatial transformation pattern to a position."""
        x, y = pos
        if pattern_type == "translate":
            return (x + dx, y + dy)
        elif pattern_type == "mirror":
            # Mirror across center_x vertical line
            return (2 * center_x - x, y)
        elif pattern_type == "rotate":
            # Rotate 90 degrees clockwise around center
            rx, ry = x - center_x, y - center_y
            return (center_x + ry, center_y - rx)
        return pos

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_examples = self.difficulty_config.params.get("n_examples", 2)
        n_tests = self.difficulty_config.params.get("n_tests", 1)
        pattern_type = self.difficulty_config.params.get("pattern", "translate")

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        center_x = size // 2
        center_y = size // 2

        # Pattern parameters
        if pattern_type == "translate":
            dx = int(rng.integers(1, 4))
            dy = int(rng.integers(-2, 3))
        else:
            dx, dy = 0, 0

        # Generate example and test input positions
        # Use left half for inputs, ensure outputs fit in grid
        input_positions = []
        used = {agent_pos}

        for attempt in range(50):
            candidate_inputs = []
            valid = True
            for _ in range(n_examples + n_tests):
                ix = int(rng.integers(2, size // 2))
                iy = int(rng.integers(2, size - 2))
                inp = (ix, iy)
                out = self._apply_pattern(inp, pattern_type, dx, dy, center_x, center_y)
                ox, oy = out
                if (1 <= ox < size - 1 and 1 <= oy < size - 1
                        and inp not in used and out not in used
                        and inp != out):
                    candidate_inputs.append((inp, out))
                    used.add(inp)
                    used.add(out)
                else:
                    valid = False
                    break
            if valid and len(candidate_inputs) >= n_examples + n_tests:
                input_positions = candidate_inputs
                break
            used = {agent_pos}

        if not input_positions:
            # Fallback: simple right-shift by 3
            pattern_type = "translate"
            dx, dy = 3, 0
            input_positions = []
            used = {agent_pos}
            for i in range(n_examples + n_tests):
                inp = (2, 2 + i * 2)
                out = (5, 2 + i * 2)
                if (1 <= out[0] < size - 1 and 1 <= out[1] < size - 1):
                    input_positions.append((inp, out))
                    used.add(inp)
                    used.add(out)

        # Split into examples and tests
        examples = input_positions[:n_examples]
        tests = input_positions[n_examples:n_examples + n_tests]

        # Place examples: SCROLL for input, GEM for output (visible answer)
        example_info = []
        for inp, out in examples:
            ix, iy = inp
            ox, oy = out
            grid.objects[iy, ix] = ObjectType.SCROLL
            grid.objects[oy, ox] = ObjectType.GEM
            example_info.append({"input": list(inp), "output": list(out)})

        # Place tests: SCROLL for input, TARGET for expected output position
        test_info = []
        test_targets = []
        for inp, out in tests:
            ix, iy = inp
            ox, oy = out
            grid.objects[iy, ix] = ObjectType.SCROLL
            grid.objects[oy, ox] = ObjectType.TARGET  # where ORB should go
            test_info.append({"input": list(inp), "output": list(out)})
            test_targets.append(out)

        # Place ORB items for the agent to pick up and deliver
        orb_positions = []
        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and grid.objects[y, x] == ObjectType.NONE
            and (x, y) != agent_pos
        ]
        rng.shuffle(free)
        for i in range(n_tests):
            if i < len(free):
                ox, oy = free[i]
                grid.objects[oy, ox] = ObjectType.ORB
                orb_positions.append(free[i])

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": test_targets,
            "examples": example_info,
            "tests": test_info,
            "test_targets": test_targets,
            "orb_positions": orb_positions,
            "pattern_type": pattern_type,
            "pattern_dx": dx,
            "pattern_dy": dy,
            "n_tests": len(tests),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        self._carrying_orb = False
        self._targets_filled = 0
        self._last_filled = 0
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        obj = grid.objects[y, x]

        if not self._carrying_orb and obj == ObjectType.ORB:
            # Pick up orb
            grid.objects[y, x] = ObjectType.NONE
            self._carrying_orb = True
        elif self._carrying_orb and obj == ObjectType.TARGET:
            # Place orb on target
            grid.objects[y, x] = ObjectType.GOAL  # correct placement visual
            self._carrying_orb = False
            self._targets_filled += 1

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if self._targets_filled > self._last_filled:
            reward += 0.5
        self._last_filled = self._targets_filled

        # Approach shaping
        if "agent" in new_state and "grid" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            if not self._carrying_orb:
                orbs = [
                    (x, y) for y in range(g.height) for x in range(g.width)
                    if g.objects[y, x] == ObjectType.ORB
                ]
                targets = orbs
            else:
                targets = [
                    (x, y) for y in range(g.height) for x in range(g.width)
                    if g.objects[y, x] == ObjectType.TARGET
                ]
            if targets:
                d_new = min(abs(ax - tx) + abs(ay - ty) for tx, ty in targets)
                d_old = min(abs(ox - tx) + abs(oy - ty) for tx, ty in targets)
                reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        n_tests = config.get("n_tests", 1)
        return self._targets_filled >= n_tests

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
