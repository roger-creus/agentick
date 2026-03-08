"""PackingPuzzle - Push typed pieces into matching target slots.

MECHANICS:
  - Multiple piece types (BOX, GEM, ORB, SCROLL) must go to matching TARGET slots
  - Each TARGET has a metadata value encoding which piece type it accepts
  - Pieces are pushed Sokoban-style (walk into piece → push it forward)
  - A piece only "fits" a target if the types match (stored in metadata)
  - Success = all target slots filled with matching piece types
  - Differentiated from SokobanPush: type-matching constraint, not just position
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Piece types used in packing puzzle
_PIECE_TYPES = [ObjectType.BOX, ObjectType.GEM, ObjectType.ORB, ObjectType.SCROLL]


@register_task("PackingPuzzle-v0", tags=["spatial_reasoning", "planning"])
class PackingPuzzleTask(TaskSpec):
    """Push typed pieces into matching target slots."""

    name = "PackingPuzzle-v0"
    description = "Push pieces into matching target slots by type"
    capability_tags = ["spatial_reasoning", "planning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_pieces": 2, "n_types": 2, "n_distractors": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=180,
            params={"n_pieces": 3, "n_types": 2, "n_distractors": 1},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=300,
            params={"n_pieces": 4, "n_types": 3, "n_distractors": 1},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=500,
            params={"n_pieces": 5, "n_types": 4, "n_distractors": 2},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_pieces", 2)
        n_types = min(self.difficulty_config.params.get("n_types", 2), len(_PIECE_TYPES))
        n_distractors = self.difficulty_config.params.get("n_distractors", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Randomize agent side and target side
        if rng.random() < 0.5:
            agent_pos = (size - 2, int(rng.integers(1, size - 1)))
            target_col = 1
        else:
            agent_pos = (1, int(rng.integers(1, size - 1)))
            target_col = size - 2

        # Choose piece types
        available_types = list(_PIECE_TYPES[:n_types])

        # Assign a type to each piece
        piece_types = []
        for i in range(n):
            piece_types.append(available_types[i % len(available_types)])

        # Place targets with metadata encoding expected type
        target_positions = []
        piece_positions = []
        used = {agent_pos}

        # Interior cells for pieces (not on border, not on target column)
        interior = [
            (x, y)
            for x in range(2, size - 2)
            for y in range(2, size - 2)
            if (x, y) != agent_pos
        ]
        rng.shuffle(interior)

        for i in range(n):
            # Target position along left column, vertically centred
            ty = 1 + i + (size - 2 - n) // 2  # centre targets
            ty = max(1, min(size - 2, ty))
            tx = target_col
            target_positions.append((tx, ty))

            # Encode expected piece type in metadata
            grid.objects[ty, tx] = ObjectType.TARGET
            grid.metadata[ty, tx] = int(piece_types[i])

            # Piece position in interior
            if i < len(interior):
                px, py = interior[i]
            else:
                px, py = (1 + i, 2)
            piece_positions.append((px, py))
            grid.objects[py, px] = piece_types[i]
            used.add((px, py))
            used.add((tx, ty))

        # Distractor pieces (wrong type, no matching target)
        distractor_positions = []
        dist_candidates = [p for p in interior[n:] if p not in used]
        for i in range(min(n_distractors, len(dist_candidates))):
            dx, dy = dist_candidates[i]
            # Use a type not in current piece_types if possible
            unused = [t for t in _PIECE_TYPES if t not in piece_types]
            dt = unused[0] if unused else _PIECE_TYPES[int(rng.integers(len(_PIECE_TYPES)))]
            grid.objects[dy, dx] = dt
            distractor_positions.append(dist_candidates[i])
            used.add(dist_candidates[i])

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": target_positions,
            "piece_positions": piece_positions,
            "piece_types": [int(t) for t in piece_types],
            "target_positions": target_positions,
            "distractor_positions": distractor_positions,
            "n_pieces": n,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        self._config = config
        self._last_matched = 0
        # Remember original target slots so they survive being overwritten
        self._target_slots: dict[tuple[int, int], int] = {}
        for tx, ty in config.get("target_positions", []):
            self._target_slots[(tx, ty)] = int(grid.metadata[ty, tx])

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Sokoban push: agent walks into piece -> push it forward one cell."""
        x, y = pos
        obj = grid.objects[y, x]
        if obj not in _PIECE_TYPES:
            return True  # not a piece, always passable

        # Compute push direction
        ax, ay = agent.position
        dx, dy = x - ax, y - ay
        nbx, nby = x + dx, y + dy

        # Target cell must be in bounds, non-wall
        if not (0 < nbx < grid.width - 1 and 0 < nby < grid.height - 1):
            return False
        if grid.terrain[nby, nbx] == CellType.WALL:
            return False
        dest_obj = grid.objects[nby, nbx]
        if dest_obj == ObjectType.TARGET:
            # Can push onto target (will check type match in placement)
            pass
        elif dest_obj not in (ObjectType.NONE,):
            return False  # blocked by another piece

        # Push the piece
        piece_type = obj

        # Restore target underneath the source cell if it was a target slot
        if (x, y) in self._target_slots:
            grid.objects[y, x] = ObjectType.TARGET
            grid.metadata[y, x] = self._target_slots[(x, y)]
        else:
            grid.objects[y, x] = ObjectType.NONE

        if dest_obj == ObjectType.TARGET:
            expected = int(grid.metadata[nby, nbx])
            if int(piece_type) == expected:
                # Correct match: piece fits in target
                grid.objects[nby, nbx] = ObjectType.GOAL  # visual: matched
                grid.metadata[nby, nbx] = 0
            else:
                # Wrong type: piece sits on top but doesn't match
                grid.objects[nby, nbx] = piece_type
        elif (nbx, nby) in self._target_slots:
            # Landing on a previously-overwritten target slot
            expected = self._target_slots[(nbx, nby)]
            if int(piece_type) == expected:
                grid.objects[nby, nbx] = ObjectType.GOAL
                grid.metadata[nby, nbx] = 0
            else:
                grid.objects[nby, nbx] = piece_type
        else:
            grid.objects[nby, nbx] = piece_type

        return True  # agent enters the vacated cell

    def _count_matched(self, grid, config):
        """Count targets correctly filled."""
        targets = config.get("target_positions", [])
        count = 0
        for tx, ty in targets:
            if grid.objects[ty, tx] == ObjectType.GOAL:
                count += 1
        return count

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "grid" not in new_state or "config" not in new_state:
            return reward
        matched = self._count_matched(new_state["grid"], new_state["config"])
        if matched > self._last_matched:
            reward += 0.4 * (matched - self._last_matched)
        self._last_matched = matched
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All target slots filled with matching piece types."""
        if "grid" not in state or "config" not in state:
            return False
        config = state["config"]
        targets = config.get("target_positions", [])
        if not targets:
            return False
        n = config.get("n_pieces", 1)
        matched = self._count_matched(state["grid"], config)
        return matched >= n

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
