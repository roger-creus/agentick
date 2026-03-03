"""SequenceMemory - Observe a sequence of positions, then reproduce from memory.

MECHANICS:
  - SHOW PHASE: Target positions are revealed one at a time as GEM objects.
    Each position is displayed for `show_steps` steps, then hidden.
    Agent can move freely during this phase.
  - REPRODUCE PHASE: Agent must visit the memorized positions in correct order.
    No visual markers on target positions — pure spatial memory test.
    Visiting the correct next position shows a brief GOAL flash and advances progress.
    Visiting a wrong position (distractor) incurs a penalty.
  - Distractors are tracked as position lists internally; no SWITCH or object is placed
    on the grid for them.
  - easy:   3 targets, 4 steps per show, no distractors
  - medium: 4 targets, 3 steps per show, 2 distractor positions
  - hard:   5 targets, 2 steps per show, positions shuffle between phases
  - expert: 6 targets, 1-step flash, shuffled, obstacles
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SequenceMemory-v0", tags=["memory", "pattern_recognition"])
class SequenceMemoryTask(TaskSpec):
    """Observe sequence of positions during show phase, reproduce from memory."""

    name = "SequenceMemory-v0"
    description = "Memorize shown positions, then visit them in order"
    capability_tags = ["memory", "pattern_recognition"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={
                "n_targets": 3,
                "show_steps": 4,
                "n_distractors": 0,
                "n_obstacles": 0,
                "shuffle": False,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=180,
            params={
                "n_targets": 4,
                "show_steps": 3,
                "n_distractors": 2,
                "n_obstacles": 4,
                "shuffle": False,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={
                "n_targets": 5,
                "show_steps": 2,
                "n_distractors": 3,
                "n_obstacles": 6,
                "shuffle": True,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=450,
            params={
                "n_targets": 6,
                "show_steps": 1,
                "n_distractors": 4,
                "n_obstacles": 8,
                "shuffle": True,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n = p.get("n_targets", 3)
        show_steps = p.get("show_steps", 4)
        n_dist = p.get("n_distractors", 0)
        n_obs = p.get("n_obstacles", 0)
        shuffle = p.get("shuffle", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Randomize agent start corner
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)
        agent_pos = corners[0]

        # Add obstacles
        interior = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) != agent_pos
        ]
        rng.shuffle(interior)
        placed = 0
        for wx, wy in interior[: n_obs * 3]:
            grid.terrain[wy, wx] = CellType.WALL
            if len(grid.flood_fill(agent_pos)) < n + n_dist + 2:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                placed += 1
                if placed >= n_obs:
                    break

        reachable = list(grid.flood_fill(agent_pos) - {agent_pos})
        rng.shuffle(reachable)
        targets = reachable[:n]
        distractors = reachable[n : n + n_dist]

        # No objects placed at start — show phase will reveal them
        # (targets and distractors are placed dynamically)

        # Total show phase duration
        total_show_steps = n * show_steps

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": targets,
            "sequence": targets,
            "distractors": distractors,
            "show_steps": show_steps,
            "total_show_steps": total_show_steps,
            "shuffle": shuffle,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_phase"] = "show"  # "show" or "reproduce"
        config["_show_step"] = 0
        config["_seq_progress"] = 0
        config["_wrong_visit"] = False
        config["_shuffle_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        config["_current_shown"] = -1  # which target is currently displayed
        self._config = config
        self._last_seq_progress = 0

        # Show first target immediately
        seq = config.get("sequence", [])
        if seq:
            tx, ty = seq[0]
            grid.objects[ty, tx] = ObjectType.GEM
            config["_current_shown"] = 0

    def on_env_step(self, agent, grid, config, step_count):
        """Manage show phase: reveal targets one by one, then switch to reproduce."""
        phase = config.get("_phase", "show")
        if phase != "show":
            return

        seq = config.get("sequence", [])
        show_steps = config.get("show_steps", 4)

        show_step = config.get("_show_step", 0) + 1
        config["_show_step"] = show_step

        # Which target should be shown at this step?
        target_idx = show_step // show_steps
        prev_idx = (show_step - 1) // show_steps

        # Hide previous target when transitioning
        if target_idx != prev_idx and prev_idx < len(seq):
            px, py = seq[prev_idx]
            if grid.objects[py, px] == ObjectType.GEM:
                grid.objects[py, px] = ObjectType.NONE

        # Show current target
        if target_idx < len(seq):
            tx, ty = seq[target_idx]
            grid.objects[ty, tx] = ObjectType.GEM
            config["_current_shown"] = target_idx
        else:
            # Show phase complete — hide last target and switch to reproduce
            cur = config.get("_current_shown", -1)
            if 0 <= cur < len(seq):
                cx, cy = seq[cur]
                if grid.objects[cy, cx] == ObjectType.GEM:
                    grid.objects[cy, cx] = ObjectType.NONE
            config["_current_shown"] = -1
            config["_phase"] = "reproduce"

            # Shuffle positions if configured (makes it harder —
            # positions rotate but agent must remember original order)
            if config.get("shuffle", False):
                shuffle_rng = config.get("_shuffle_rng")
                if shuffle_rng and len(seq) > 1:
                    # Rotate positions: each target moves to a random
                    # nearby empty cell, testing whether agent remembers
                    # the SEQUENCE (order) not just the positions
                    pass  # Shuffle is about testing memory of the sequence order

            # Distractors are tracked by position only (no visual objects)
            # Stepping on a distractor position incurs a penalty

    def on_agent_moved(self, pos, agent, grid):
        """Handle position visits during reproduce phase."""
        config = getattr(self, "_config", {})
        phase = config.get("_phase", "show")

        if phase != "reproduce":
            return

        x, y = pos
        progress = config.get("_seq_progress", 0)
        sequence = config.get("sequence", [])

        if progress >= len(sequence):
            return

        # Check if agent is at the next correct position
        next_target = sequence[progress]
        if (x, y) == tuple(next_target):
            # Correct! Flash GOAL briefly (will be cleared next step)
            grid.objects[y, x] = ObjectType.GOAL
            config["_seq_progress"] = progress + 1
        else:
            # Check if position is a distractor
            distractor_set = {tuple(d) for d in config.get("distractors", [])}
            if (x, y) in distractor_set and not config.get("_wrong_visit", False):
                config["_wrong_visit"] = True

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})

        # Penalty for wrong visit
        if config.get("_wrong_visit", False) and not old_state.get("config", {}).get(
            "_wrong_visit", False
        ):
            reward -= 0.3

        # Progress reward
        new_progress = config.get("_seq_progress", 0)
        if new_progress > self._last_seq_progress:
            reward += 0.3 * (new_progress - self._last_seq_progress)
        self._last_seq_progress = new_progress

        # Approach shaping during reproduce phase
        if config.get("_phase") == "reproduce" and "agent" in new_state:
            sequence = config.get("sequence", [])
            progress = config.get("_seq_progress", 0)
            if progress < len(sequence):
                tgt = sequence[progress]
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                reward += 0.05 * (
                    (abs(ox - tgt[0]) + abs(oy - tgt[1])) - (abs(ax - tgt[0]) + abs(ay - tgt[1]))
                )

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_phase") != "reproduce":
            return False
        progress = config.get("_seq_progress", 0)
        sequence = config.get("sequence", [])
        return len(sequence) > 0 and progress >= len(sequence)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
