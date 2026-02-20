"""FewShotAdaptation - Learn a hidden rule from K demonstration trials, apply to test trial.

MECHANICS:
  - Multiple "trials" within one episode, each with K candidate targets
  - In demonstration trials: agent observes which target is correct (briefly marked GOAL)
  - A hidden RULE determines the correct target (e.g., "closest to corner",
    "furthest from agent start", "most adjacent walls")
  - After K demonstrations, the test trial has new target positions —
    agent must apply the learned rule to pick the correct target
  - Tests rapid few-shot learning and adaptation, not just spatial memory
  - Guards patrol at hard+, episode ends on collision
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


# Hidden rules: given target positions, return the index of the correct target
def _rule_closest_corner(targets, grid, agent_start):
    corners = [(1, 1), (grid.width - 2, 1), (1, grid.height - 2),
               (grid.width - 2, grid.height - 2)]
    best_i, best_d = 0, float("inf")
    for i, (tx, ty) in enumerate(targets):
        d = min(abs(tx - cx) + abs(ty - cy) for cx, cy in corners)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _rule_furthest_from_start(targets, grid, agent_start):
    sx, sy = agent_start
    best_i, best_d = 0, -1
    for i, (tx, ty) in enumerate(targets):
        d = abs(tx - sx) + abs(ty - sy)
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def _rule_most_adjacent_empty(targets, grid, agent_start):
    best_i, best_c = 0, -1
    for i, (tx, ty) in enumerate(targets):
        c = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                if grid.terrain[ny, nx] == CellType.EMPTY:
                    c += 1
        if c > best_c:
            best_c = c
            best_i = i
    return best_i


_HIDDEN_RULES = [
    ("closest_corner", _rule_closest_corner),
    ("furthest_start", _rule_furthest_from_start),
    ("most_empty", _rule_most_adjacent_empty),
]


@register_task("FewShotAdaptation-v0", tags=["meta_learning", "adaptation", "few_shot"])
class FewShotAdaptationTask(TaskSpec):
    """Learn a hidden rule from K demonstrations, apply to test trial."""

    name = "FewShotAdaptation-v0"
    description = "Learn rule from demonstrations, apply to new targets"
    capability_tags = ["meta_learning", "adaptation", "few_shot"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=80,
            params={
                "k_demos": 2, "n_candidates": 2,
                "reveal_steps": 5, "n_obstacles": 0, "n_guards": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=9, max_steps=150,
            params={
                "k_demos": 2, "n_candidates": 3,
                "reveal_steps": 3, "n_obstacles": 4, "n_guards": 0,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=11, max_steps=250,
            params={
                "k_demos": 3, "n_candidates": 3,
                "reveal_steps": 2, "n_obstacles": 6, "n_guards": 1,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=13, max_steps=350,
            params={
                "k_demos": 3, "n_candidates": 4,
                "reveal_steps": 1, "n_obstacles": 8, "n_guards": 2,
            },
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        k_demos     = p.get("k_demos", 2)
        n_cand      = p.get("n_candidates", 2)
        reveal      = p.get("reveal_steps", 5)
        n_obs       = p.get("n_obstacles", 0)
        n_guards    = p.get("n_guards", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)
        agent_pos = corners[0]

        # Add obstacles
        interior = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)
                    if (x, y) != agent_pos]
        rng.shuffle(interior)
        placed = 0
        for wx, wy in interior[:n_obs * 3]:
            grid.terrain[wy, wx] = CellType.WALL
            if len(grid.flood_fill(agent_pos)) < n_cand * (k_demos + 1) + 3:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                placed += 1
                if placed >= n_obs:
                    break

        # Select hidden rule
        rule_idx = int(rng.integers(0, len(_HIDDEN_RULES)))
        rule_name, rule_fn = _HIDDEN_RULES[rule_idx]

        # Generate trial layouts: k_demos demonstrations + 1 test
        reachable = list(grid.flood_fill(agent_pos) - {agent_pos})
        rng.shuffle(reachable)

        n_total_trials = k_demos + 1
        trials = []
        used_positions = set()
        pos_pool = list(reachable)

        for trial_i in range(n_total_trials):
            # Pick n_cand positions for this trial
            available = [p2 for p2 in pos_pool if p2 not in used_positions]
            if len(available) < n_cand:
                available = list(pos_pool)
                rng.shuffle(available)
            trial_targets = available[:n_cand]
            for tp in trial_targets:
                used_positions.add(tp)

            # Determine correct target by the hidden rule
            correct_idx = rule_fn(trial_targets, grid, agent_pos)
            trials.append({
                "targets": trial_targets,
                "correct_idx": correct_idx,
                "is_test": trial_i == n_total_trials - 1,
            })

        # The final goal (test trial correct target)
        test_trial = trials[-1]
        true_goal = test_trial["targets"][test_trial["correct_idx"]]

        # Guards
        guard_pool = [c for c in reachable if c not in used_positions]
        rng.shuffle(guard_pool)
        guard_positions = guard_pool[:n_guards]

        # Place first demo trial targets
        first_trial = trials[0]
        for i, (tx, ty) in enumerate(first_trial["targets"]):
            if i == first_trial["correct_idx"]:
                grid.objects[ty, tx] = ObjectType.GOAL
            else:
                grid.objects[ty, tx] = ObjectType.TARGET

        return grid, {
            "agent_start":   agent_pos,
            "goal_positions": [true_goal],
            "true_goal":     tuple(true_goal),
            "trials":        trials,
            "k_demos":       k_demos,
            "reveal_steps":  reveal,
            "rule_name":     rule_name,
            "n_guards":      n_guards,
            "_guard_positions": guard_positions,
            "_guard_dirs":   [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed":   int(rng.integers(0, 2**31)),
            "max_steps":     self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_current_trial"] = 0
        config["_trial_step"] = 0
        config["_goal_reached"] = False
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config

        # Draw guards
        for gx, gy in config.get("_guard_positions", []):
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos

        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True
            return

        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        if current >= len(trials):
            return

        trial = trials[current]
        is_test = trial.get("is_test", False)

        if grid.objects[y, x] == ObjectType.GOAL:
            if is_test:
                config["_goal_reached"] = True
            else:
                # Demo trial: agent reached the correct target (observing the rule)
                # Clear this trial's objects and advance
                self._clear_trial_objects(grid, trial)
                config["_current_trial"] = current + 1
                config["_trial_step"] = 0

                # Set up next trial
                next_trial_idx = current + 1
                if next_trial_idx < len(trials):
                    next_trial = trials[next_trial_idx]
                    for i, (tx, ty) in enumerate(next_trial["targets"]):
                        if next_trial.get("is_test", False):
                            # Test trial: all shown as TARGET (no GOAL hint)
                            grid.objects[ty, tx] = ObjectType.TARGET
                        elif i == next_trial["correct_idx"]:
                            grid.objects[ty, tx] = ObjectType.GOAL
                        else:
                            grid.objects[ty, tx] = ObjectType.TARGET

        elif grid.objects[y, x] == ObjectType.TARGET and is_test:
            # Test trial: agent picked wrong target (failure but continue)
            grid.objects[y, x] = ObjectType.NONE

    def _clear_trial_objects(self, grid, trial):
        for tx, ty in trial.get("targets", []):
            obj = grid.objects[ty, tx]
            if obj in (ObjectType.GOAL, ObjectType.TARGET):
                grid.objects[ty, tx] = ObjectType.NONE

    def on_env_step(self, agent, grid, config, step_count):
        # Demo trial: hide GOAL after reveal_steps
        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        if current < len(trials):
            trial = trials[current]
            if not trial.get("is_test", False):
                config["_trial_step"] = config.get("_trial_step", 0) + 1
                reveal = config.get("reveal_steps", 5)
                if config["_trial_step"] == reveal:
                    # Hide the GOAL marker, make it look like TARGET
                    correct_idx = trial["correct_idx"]
                    tx, ty = trial["targets"][correct_idx]
                    if grid.objects[ty, tx] == ObjectType.GOAL:
                        grid.objects[ty, tx] = ObjectType.TARGET

        # Move guards
        guards = config.get("_guard_positions", [])
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        ax, ay = agent.position
        if guards and rng is not None:
            for gx, gy in guards:
                if grid.objects[gy, gx] == ObjectType.NPC:
                    grid.objects[gy, gx] = ObjectType.NONE
            new_guards, new_dirs = [], []
            for i, (gx, gy) in enumerate(guards):
                d = dirs[i]
                dx, dy = self._DIRS[d]
                nx, ny = gx + dx, gy + dy
                if (0 < nx < grid.width - 1 and 0 < ny < grid.height - 1
                        and grid.terrain[ny, nx] == CellType.EMPTY
                        and grid.objects[ny, nx] == ObjectType.NONE):
                    new_guards.append((nx, ny))
                else:
                    d = int(rng.integers(0, 4))
                    new_guards.append((gx, gy))
                new_dirs.append(d)
                if new_guards[-1] == (ax, ay):
                    config["_guard_collision"] = True
            config["_guard_positions"] = new_guards
            config["_guard_dirs"] = new_dirs
            for gx, gy in new_guards:
                if grid.terrain[gy, gx] == CellType.EMPTY:
                    grid.objects[gy, gx] = ObjectType.NPC

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_guard_collision", False):
            return -1.0
        reward = -0.01

        if config.get("_goal_reached", False) and not old_state.get(
                "config", {}).get("_goal_reached", False):
            reward += 1.0
        elif "agent" in new_state:
            # Shape toward current trial's targets
            trials = config.get("trials", [])
            current = config.get("_current_trial", 0)
            if current < len(trials):
                trial = trials[current]
                targets = trial.get("targets", [])
                if targets:
                    ax, ay = new_state["agent"].position
                    ox, oy = old_state.get("agent_position", (ax, ay))
                    # Guide toward nearest target
                    d_new = min(abs(ax - tx) + abs(ay - ty) for tx, ty in targets)
                    d_old = min(abs(ox - tx) + abs(oy - ty) for tx, ty in targets)
                    reward += 0.03 * (d_old - d_new)
        return reward

    def check_done(self, state):
        if state.get("config", {}).get("_guard_collision", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        if state.get("config", {}).get("_guard_collision", False):
            return False
        return bool(state.get("config", {}).get("_goal_reached", False))

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
