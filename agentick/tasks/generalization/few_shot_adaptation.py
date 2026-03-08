"""FewShotAdaptation - Learn a hidden rule from K demonstrations, apply to test trial.

MECHANICS:
  - Episode: K auto-advancing demonstration trials, then 1 test trial
  - Each trial places N candidates of distinct types (GEM, SCROLL, ORB, COIN)
  - A hidden RULE determines which candidate is correct each trial:
    - goto_type: always the candidate of a specific ObjectType
    - nearest_corner: the candidate nearest to any grid corner
    - furthest_start: the candidate furthest from agent's start position
    - most_open (hard+): the candidate with most adjacent empty cells
  - Demo trials: correct candidate shown as GOAL for reveal_steps,
    then reverts to its ObjectType; trial auto-advances after demo_duration
  - Test trial: no GOAL highlight — agent must infer and navigate to correct one
  - Stepping on wrong candidate in test = failure (-1.0)
  - Stepping on correct candidate in test = success (+1.0)
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_CANDIDATE_TYPES = [ObjectType.GEM, ObjectType.SCROLL, ObjectType.ORB, ObjectType.COIN]


# ---------------------------------------------------------------------------
# Hidden rules — each returns the index of the correct candidate
# ---------------------------------------------------------------------------
def _rule_goto_type(positions, types, grid, agent_start, target_type=None, **kw):
    """Correct = the candidate whose ObjectType matches target_type."""
    for i, t in enumerate(types):
        if t == target_type:
            return i
    return 0


def _rule_nearest_corner(positions, types, grid, agent_start, **kw):
    """Correct = the candidate nearest (Manhattan) to any grid corner."""
    corners = [
        (1, 1),
        (grid.width - 2, 1),
        (1, grid.height - 2),
        (grid.width - 2, grid.height - 2),
    ]
    best_i, best_d = 0, float("inf")
    for i, (x, y) in enumerate(positions):
        d = min(abs(x - cx) + abs(y - cy) for cx, cy in corners)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _rule_furthest_start(positions, types, grid, agent_start, **kw):
    """Correct = the candidate furthest (Manhattan) from agent start."""
    sx, sy = agent_start
    best_i, best_d = 0, -1
    for i, (x, y) in enumerate(positions):
        d = abs(x - sx) + abs(y - sy)
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def _rule_most_open(positions, types, grid, agent_start, **kw):
    """Correct = the candidate with the most adjacent EMPTY cells."""
    best_i, best_c = 0, -1
    for i, (x, y) in enumerate(positions):
        c = sum(
            1
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
            if 0 <= x + dx < grid.width
            and 0 <= y + dy < grid.height
            and grid.terrain[y + dy, x + dx] == CellType.EMPTY
        )
        if c > best_c:
            best_c = c
            best_i = i
    return best_i


_RULES = {
    "goto_type": _rule_goto_type,
    "nearest_corner": _rule_nearest_corner,
    "furthest_start": _rule_furthest_start,
    "most_open": _rule_most_open,
}


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------
@register_task("FewShotAdaptation-v0", tags=["meta_learning", "adaptation", "few_shot"])
class FewShotAdaptationTask(TaskSpec):
    """Learn a hidden rule from K demonstrations, apply to test trial."""

    name = "FewShotAdaptation-v0"
    description = "Learn hidden rule from demo trials, apply to new targets"
    capability_tags = ["meta_learning", "adaptation", "few_shot"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={
                "k_demos": 3,
                "n_candidates": 2,
                "reveal_steps": 12,
                "demo_duration": 18,
                "n_obstacles": 0,
                "rules": ["goto_type", "nearest_corner"],
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=170,
            params={
                "k_demos": 3,
                "n_candidates": 3,
                "reveal_steps": 8,
                "demo_duration": 15,
                "n_obstacles": 4,
                "rules": ["goto_type", "nearest_corner", "furthest_start"],
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=260,
            params={
                "k_demos": 3,
                "n_candidates": 3,
                "reveal_steps": 5,
                "demo_duration": 12,
                "n_obstacles": 6,
                "rules": [
                    "goto_type",
                    "nearest_corner",
                    "furthest_start",
                    "most_open",
                ],
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=400,
            params={
                "k_demos": 4,
                "n_candidates": 4,
                "reveal_steps": 3,
                "demo_duration": 10,
                "n_obstacles": 8,
                "rules": [
                    "goto_type",
                    "nearest_corner",
                    "furthest_start",
                    "most_open",
                ],
            },
        ),
    }

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        k_demos = p.get("k_demos", 2)
        n_cand = p.get("n_candidates", 2)
        reveal = p.get("reveal_steps", 5)
        demo_dur = p.get("demo_duration", 12)
        n_obs = p.get("n_obstacles", 0)
        rule_pool = p.get("rules", ["goto_type", "nearest_corner"])

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])

        # Place interior wall obstacles
        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
        ]
        rng.shuffle(interior)
        placed = 0
        for wx, wy in interior:
            if placed >= n_obs:
                break
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            min_cells = n_cand * (k_demos + 1) + 5
            if len(reachable) < min_cells:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                placed += 1

        # Pick hidden rule for this episode
        rule_name = rule_pool[int(rng.integers(0, len(rule_pool)))]
        rule_fn = _RULES[rule_name]

        # For goto_type: choose which ObjectType is "correct"
        target_type = _CANDIDATE_TYPES[
            int(rng.integers(0, min(n_cand, len(_CANDIDATE_TYPES))))
        ]

        # All reachable EMPTY cells for candidate placement
        reachable = [
            p2
            for p2 in grid.flood_fill(agent_pos)
            if p2 != agent_pos and grid.terrain[p2[1], p2[0]] == CellType.EMPTY
        ]

        # Rule kwargs
        rule_kw = {"target_type": target_type}

        # Generate K demo + 1 test trial layouts
        n_trials = k_demos + 1
        trials = []
        used = set()

        for trial_i in range(n_trials):
            avail = [p2 for p2 in reachable if p2 not in used]
            if len(avail) < n_cand:
                used.clear()
                avail = [p2 for p2 in reachable if p2 not in used]
            rng.shuffle(avail)
            trial_positions = avail[:n_cand]
            for tp in trial_positions:
                used.add(tp)

            # Assign distinct ObjectTypes (shuffled each trial)
            trial_types = list(_CANDIDATE_TYPES[:n_cand])
            rng.shuffle(trial_types)

            # Ensure target_type is present for goto_type rule
            if rule_name == "goto_type" and target_type not in trial_types:
                trial_types[0] = target_type

            correct_idx = rule_fn(
                trial_positions, trial_types, grid, agent_pos, **rule_kw,
            )

            trials.append({
                "positions": [list(p) for p in trial_positions],
                "types": [int(t) for t in trial_types],
                "correct_idx": correct_idx,
                "is_test": trial_i == n_trials - 1,
            })

        # True goal for the episode
        test_trial = trials[-1]
        true_goal = tuple(test_trial["positions"][test_trial["correct_idx"]])

        # Place first demo trial objects on the grid
        first = trials[0]
        for i, (px, py) in enumerate(first["positions"]):
            if i == first["correct_idx"]:
                grid.objects[py, px] = ObjectType.GOAL  # highlighted
            else:
                grid.objects[py, px] = ObjectType(first["types"][i])

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [true_goal],
            "true_goal": true_goal,
            "trials": trials,
            "k_demos": k_demos,
            "reveal_steps": reveal,
            "demo_duration": demo_dur,
            "rule_name": rule_name,
            "target_type": int(target_type),
            "max_steps": self.get_max_steps(),
        }

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def on_env_reset(self, agent, grid, config):
        config["_current_trial"] = 0
        config["_trial_step"] = 0
        config["_goal_reached"] = False
        config["_failed"] = False
        self._config = config

    # -----------------------------------------------------------------------
    # Per-step (demo auto-advance)
    # -----------------------------------------------------------------------
    def on_env_step(self, agent, grid, config, step_count):
        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        if current >= len(trials):
            return

        trial = trials[current]
        if trial.get("is_test", False):
            return  # no auto-advance in test trial

        config["_trial_step"] = config.get("_trial_step", 0) + 1
        reveal = config.get("reveal_steps", 5)
        demo_dur = config.get("demo_duration", 12)

        # After reveal_steps: hide the GOAL, revert to ObjectType
        if config["_trial_step"] == reveal:
            ci = trial["correct_idx"]
            px, py = trial["positions"][ci]
            if grid.objects[py, px] == ObjectType.GOAL:
                grid.objects[py, px] = ObjectType(trial["types"][ci])

        # After demo_duration: clear trial, advance
        if config["_trial_step"] >= demo_dur:
            self._clear_trial(grid, trial)
            config["_current_trial"] = current + 1
            config["_trial_step"] = 0
            nxt = current + 1
            if nxt < len(trials):
                self._place_trial(grid, trials[nxt])

    # -----------------------------------------------------------------------
    # Agent movement (test trial interaction only)
    # -----------------------------------------------------------------------
    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos
        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        if current >= len(trials):
            return

        trial = trials[current]
        if not trial.get("is_test", False):
            return  # demos auto-advance, no interaction

        # Check if agent stepped on a candidate
        for i, (px, py) in enumerate(trial["positions"]):
            if (x, y) == (px, py) and grid.objects[py, px] != ObjectType.NONE:
                if i == trial["correct_idx"]:
                    config["_goal_reached"] = True
                else:
                    config["_failed"] = True
                break

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _place_trial(self, grid, trial):
        """Place candidate objects for a trial on the grid."""
        for i, (px, py) in enumerate(trial["positions"]):
            if not trial.get("is_test", False) and i == trial["correct_idx"]:
                grid.objects[py, px] = ObjectType.GOAL
            else:
                grid.objects[py, px] = ObjectType(trial["types"][i])

    def _clear_trial(self, grid, trial):
        """Remove all candidate objects from the grid for a trial."""
        removable = {ObjectType.GOAL} | {ObjectType(t) for t in trial["types"]}
        for px, py in trial["positions"]:
            if grid.objects[py, px] in removable:
                grid.objects[py, px] = ObjectType.NONE

    # -----------------------------------------------------------------------
    # Reward / done / success
    # -----------------------------------------------------------------------
    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_failed", False):
            return -1.0
        reward = -0.01
        old_config = old_state.get("config", {})
        if config.get("_goal_reached", False) and not old_config.get(
            "_goal_reached", False
        ):
            reward += 1.0
        elif "agent" in new_state:
            # Shape toward test trial candidates
            trials = config.get("trials", [])
            current = config.get("_current_trial", 0)
            if current < len(trials) and trials[current].get("is_test", False):
                trial = trials[current]
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                positions = trial.get("positions", [])
                if positions:
                    d_new = min(
                        abs(ax - px) + abs(ay - py) for px, py in positions
                    )
                    d_old = min(
                        abs(ox - px) + abs(oy - py) for px, py in positions
                    )
                    reward += 0.03 * (d_old - d_new)
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_failed", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_failed", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_failed", False):
            return False
        return bool(config.get("_goal_reached", False))

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
