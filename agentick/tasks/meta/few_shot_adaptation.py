"""FewShotAdaptation - Learn a hidden rule from K demonstration trials, apply to test trial.

MECHANICS:
  - Multiple "trials" within one episode, each with K candidate targets
  - In demonstration trials: agent observes which target is correct (briefly marked GOAL)
  - A GEM is placed on the correct target during the reveal window for extra visual salience,
    then removed when the GOAL converts to TARGET
  - A hidden RULE determines the correct target (e.g., "closest to corner",
    "furthest from agent start", "most adjacent walls")
  - Compositional rules at hard+: "color_match", "avoid_type", "nearest_to_marker"
    use demo marker objects (different ObjectTypes) and reference SWITCH objects
  - At expert difficulty, rules become compound (must satisfy two conditions)
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


# ---------------------------------------------------------------------------
# Object type sets used by compositional rules.  Each trial picks a "demo
# marker type" from _MARKER_TYPES.  The targets are rendered with that type
# or a contrasting type so the agent can learn the visual pattern.
# ---------------------------------------------------------------------------
_MARKER_TYPES = [ObjectType.KEY, ObjectType.TOOL, ObjectType.POTION, ObjectType.SCROLL]


# ---------------------------------------------------------------------------
# Hidden rules: given target positions, return the index of the correct target.
# Spatial rules (original) ------------------------------------------------
# ---------------------------------------------------------------------------
def _rule_closest_corner(targets, grid, agent_start, **kw):
    corners = [
        (1, 1),
        (grid.width - 2, 1),
        (1, grid.height - 2),
        (grid.width - 2, grid.height - 2),
    ]
    best_i, best_d = 0, float("inf")
    for i, (tx, ty) in enumerate(targets):
        d = min(abs(tx - cx) + abs(ty - cy) for cx, cy in corners)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _rule_furthest_from_start(targets, grid, agent_start, **kw):
    sx, sy = agent_start
    best_i, best_d = 0, -1
    for i, (tx, ty) in enumerate(targets):
        d = abs(tx - sx) + abs(ty - sy)
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def _rule_most_adjacent_empty(targets, grid, agent_start, **kw):
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


# ---------------------------------------------------------------------------
# Compositional rules (new) -----------------------------------------------
# These rely on extra kwargs populated during generation:
#   marker_type  – the ObjectType used as the demo marker
#   target_types – list[ObjectType] parallel to *targets*, one per candidate
#   switch_pos   – position of the reference SWITCH object on the grid
# ---------------------------------------------------------------------------
def _rule_color_match(targets, grid, agent_start, **kw):
    """Go to the target whose type matches the demo marker type."""
    marker = kw.get("marker_type")
    target_types = kw.get("target_types", [])
    # Find the target whose type == marker_type
    for i, tt in enumerate(target_types):
        if tt == marker:
            return i
    # Fallback: first target (should not happen with correct generation)
    return 0


def _rule_avoid_type(targets, grid, agent_start, **kw):
    """Go to the target whose type is NOT the same as the demo marker."""
    marker = kw.get("marker_type")
    target_types = kw.get("target_types", [])
    for i, tt in enumerate(target_types):
        if tt != marker:
            return i
    return 0


def _rule_nearest_to_marker(targets, grid, agent_start, **kw):
    """Go to the target closest (Manhattan) to the reference SWITCH object."""
    sx, sy = kw.get("switch_pos", agent_start)
    best_i, best_d = 0, float("inf")
    for i, (tx, ty) in enumerate(targets):
        d = abs(tx - sx) + abs(ty - sy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Rule registry ------------------------------------------------------------
# ---------------------------------------------------------------------------
# Original spatial rules (used at all difficulties)
_SPATIAL_RULES = [
    ("closest_corner", _rule_closest_corner),
    ("furthest_start", _rule_furthest_from_start),
    ("most_empty", _rule_most_adjacent_empty),
]

# Compositional rules (added at hard+ difficulties)
_COMPOSITIONAL_RULES = [
    ("color_match", _rule_color_match),
    ("avoid_type", _rule_avoid_type),
    ("nearest_to_marker", _rule_nearest_to_marker),
]

# Combined list — easy/medium use only _SPATIAL_RULES; hard+ draws from full set
_HIDDEN_RULES = _SPATIAL_RULES + _COMPOSITIONAL_RULES

# Compound rule pairs for expert difficulty — agent must satisfy BOTH conditions
_COMPOUND_RULES = [
    ("nearest_to_marker+avoid_type", _rule_nearest_to_marker, _rule_avoid_type),
    ("nearest_to_marker+color_match", _rule_nearest_to_marker, _rule_color_match),
    ("closest_corner+avoid_type", _rule_closest_corner, _rule_avoid_type),
    ("furthest_start+color_match", _rule_furthest_from_start, _rule_color_match),
]


def _apply_compound_rule(targets, grid, agent_start, rule_a, rule_b, **kw):
    """Return the target index that satisfies both rule_a and rule_b.

    Strategy: score each target by the sum of inverse-ranks from both rules,
    then pick the one that scores best.  If a target is the winner of both
    individual rules it will always win; otherwise the closest compromise wins.
    """
    n = len(targets)
    # Rank each target for each sub-rule (0 = best)
    score_a = [0.0] * n
    score_b = [0.0] * n

    # For each rule, compute a per-target "desirability" value by checking
    # what the rule would pick if we iterated over subsets.  Simpler approach:
    # score = negative manhattan distance to the ideal pick of that rule.
    idx_a = rule_a(targets, grid, agent_start, **kw)
    idx_b = rule_b(targets, grid, agent_start, **kw)

    # Give each target a score: 2 if picked by both, 1 if picked by one, 0 otherwise
    for i in range(n):
        if i == idx_a:
            score_a[i] = 1.0
        if i == idx_b:
            score_b[i] = 1.0

    combined = [score_a[i] + score_b[i] for i in range(n)]
    best = max(range(n), key=lambda i: combined[i])
    return best


# ---------------------------------------------------------------------------
# Task class ---------------------------------------------------------------
# ---------------------------------------------------------------------------
@register_task("FewShotAdaptation-v0", tags=["meta_learning", "adaptation", "few_shot"])
class FewShotAdaptationTask(TaskSpec):
    """Learn a hidden rule from K demonstrations, apply to test trial."""

    name = "FewShotAdaptation-v0"
    description = "Learn rule from demonstrations, apply to new targets"
    capability_tags = ["meta_learning", "adaptation", "few_shot"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={
                "k_demos": 2,
                "n_candidates": 2,
                "reveal_steps": 12,  # longer reveal so demo goal is clearly visible
                "n_obstacles": 0,
                "n_guards": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=180,
            params={
                "k_demos": 2,
                "n_candidates": 3,
                "reveal_steps": 8,  # agent has 8 steps to observe demo goal
                "n_obstacles": 4,
                "n_guards": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=280,
            params={
                "k_demos": 3,
                "n_candidates": 3,
                "reveal_steps": 5,
                "n_obstacles": 6,
                "n_guards": 1,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=400,
            params={
                "k_demos": 3,
                "n_candidates": 4,
                "reveal_steps": 3,
                "n_obstacles": 8,
                "n_guards": 2,
            },
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    def generate(self, seed):
        rng = np.random.default_rng(seed)
        diff_name = self.difficulty_config.name
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        k_demos = p.get("k_demos", 2)
        n_cand = p.get("n_candidates", 2)
        reveal = p.get("reveal_steps", 5)
        n_obs = p.get("n_obstacles", 0)
        n_guards = p.get("n_guards", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)
        agent_pos = corners[0]

        # Add obstacles
        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
        ]
        rng.shuffle(interior)
        placed = 0
        for wx, wy in interior[: n_obs * 3]:
            grid.terrain[wy, wx] = CellType.WALL
            if len(grid.flood_fill(agent_pos)) < n_cand * (k_demos + 1) + 3:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                placed += 1
                if placed >= n_obs:
                    break

        # ---- Determine rule pool based on difficulty ----------------------
        is_compound = False
        compound_fns = None

        if diff_name == "expert":
            # 50 % chance of a compound rule at expert
            if rng.random() < 0.5:
                is_compound = True
                cidx = int(rng.integers(0, len(_COMPOUND_RULES)))
                rule_name, fn_a, fn_b = _COMPOUND_RULES[cidx]
                compound_fns = (fn_a, fn_b)
            else:
                # Pick from the full set
                rule_idx = int(rng.integers(0, len(_HIDDEN_RULES)))
                rule_name, _rule_fn = _HIDDEN_RULES[rule_idx]
        elif diff_name == "hard":
            # hard uses full single-rule set (spatial + compositional)
            rule_idx = int(rng.integers(0, len(_HIDDEN_RULES)))
            rule_name, _rule_fn = _HIDDEN_RULES[rule_idx]
        else:
            # easy / medium use only spatial rules
            rule_idx = int(rng.integers(0, len(_SPATIAL_RULES)))
            rule_name, _rule_fn = _SPATIAL_RULES[rule_idx]

        # ---- Compositional rule extras ------------------------------------
        # Pick a demo marker type used by color_match / avoid_type rules
        marker_type = _MARKER_TYPES[int(rng.integers(0, len(_MARKER_TYPES)))]
        # Pick a second contrasting type (never the same as marker_type)
        other_types = [t for t in _MARKER_TYPES if t != marker_type]
        contrast_type = other_types[int(rng.integers(0, len(other_types)))]

        # ---- Place reference SWITCH for nearest_to_marker -----------------
        reachable_set = grid.flood_fill(agent_pos) - {agent_pos}
        reachable = list(reachable_set)
        rng.shuffle(reachable)

        # Reserve a position for the reference SWITCH
        switch_pos = reachable[0] if reachable else (1, 1)
        used_positions = {switch_pos}

        needs_switch = "nearest_to_marker" in rule_name
        needs_types = any(
            k in rule_name for k in ("color_match", "avoid_type")
        )

        if needs_switch:
            sx, sy = switch_pos
            grid.objects[sy, sx] = ObjectType.SWITCH

        # ---- Build per-trial target type assignments ----------------------
        def _assign_target_types(n, correct_idx):
            """Return a list of ObjectTypes for the n candidate targets.

            For color_match / avoid_type rules the correct target gets the
            marker_type and the others get the contrast_type (or vice-versa
            for avoid_type).  For spatial-only rules all targets just get
            TARGET rendered normally (types list is informational only).
            """
            types = [contrast_type] * n
            if "color_match" in rule_name:
                # Correct target matches the marker
                types[correct_idx] = marker_type
            elif "avoid_type" in rule_name:
                # All targets are marker_type EXCEPT the correct one
                types = [marker_type] * n
                types[correct_idx] = contrast_type
            elif is_compound:
                # For compound rules containing color_match or avoid_type
                if "color_match" in rule_name:
                    types[correct_idx] = marker_type
                elif "avoid_type" in rule_name:
                    types = [marker_type] * n
                    types[correct_idx] = contrast_type
                else:
                    # Neither sub-rule is type-based — use default TARGET
                    types = [ObjectType.TARGET] * n
            else:
                types = [ObjectType.TARGET] * n
            return types

        # ---- Rule kwargs shared across trials -----------------------------
        rule_kw = {
            "marker_type": marker_type,
            "switch_pos": switch_pos,
        }

        # ---- Generate trial layouts: k_demos demonstrations + 1 test -----
        n_total_trials = k_demos + 1
        trials = []
        pos_pool = [p2 for p2 in reachable if p2 not in used_positions]

        for trial_i in range(n_total_trials):
            # Pick n_cand positions for this trial
            available = [p2 for p2 in pos_pool if p2 not in used_positions]
            if len(available) < n_cand:
                available = list(pos_pool)
                rng.shuffle(available)
            trial_targets = available[:n_cand]
            for tp in trial_targets:
                used_positions.add(tp)

            # Build target_types for compositional rules
            # We need a preliminary correct_idx to assign types, then re-eval
            # For non-type rules the types don't affect correctness so order is fine.
            # For type-dependent rules we assign types then evaluate.
            tmp_types = [ObjectType.TARGET] * n_cand

            # Preliminary correct_idx (for spatial / non-type rules)
            if is_compound:
                prelim_idx = _apply_compound_rule(
                    trial_targets, grid, agent_pos,
                    compound_fns[0], compound_fns[1],
                    target_types=tmp_types, **rule_kw,
                )
            else:
                prelim_idx = _rule_fn(
                    trial_targets, grid, agent_pos,
                    target_types=tmp_types, **rule_kw,
                )

            # Assign visual types based on preliminary pick
            trial_types = _assign_target_types(n_cand, prelim_idx)

            # Re-evaluate with correct types (matters for color_match/avoid_type)
            kw_full = {**rule_kw, "target_types": trial_types}
            if is_compound:
                correct_idx = _apply_compound_rule(
                    trial_targets, grid, agent_pos,
                    compound_fns[0], compound_fns[1], **kw_full,
                )
            else:
                correct_idx = _rule_fn(
                    trial_targets, grid, agent_pos, **kw_full,
                )

            # If the re-evaluation shifted the pick, reassign types to stay consistent
            if correct_idx != prelim_idx:
                trial_types = _assign_target_types(n_cand, correct_idx)

            trials.append(
                {
                    "targets": trial_targets,
                    "target_types": trial_types,
                    "correct_idx": correct_idx,
                    "is_test": trial_i == n_total_trials - 1,
                }
            )

        # The final goal (test trial correct target)
        test_trial = trials[-1]
        true_goal = test_trial["targets"][test_trial["correct_idx"]]

        # Guards
        guard_pool = [c for c in reachable if c not in used_positions]
        rng.shuffle(guard_pool)
        guard_positions = guard_pool[:n_guards]

        # Place first demo trial targets on the grid
        first_trial = trials[0]
        for i, (tx, ty) in enumerate(first_trial["targets"]):
            ttype = first_trial["target_types"][i]
            if i == first_trial["correct_idx"]:
                # Correct target: show as GOAL during reveal, plus GEM overlay
                grid.objects[ty, tx] = ObjectType.GOAL
            else:
                # Use the visual type for compositional rules, else TARGET
                if needs_types and ttype != ObjectType.TARGET:
                    grid.objects[ty, tx] = ttype
                else:
                    grid.objects[ty, tx] = ObjectType.TARGET

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [true_goal],
            "true_goal": tuple(true_goal),
            "trials": trials,
            "k_demos": k_demos,
            "reveal_steps": reveal,
            "rule_name": rule_name,
            "is_compound": is_compound,
            "marker_type": int(marker_type),
            "contrast_type": int(contrast_type),
            "switch_pos": tuple(switch_pos) if needs_switch else None,
            "needs_types": needs_types,
            "n_guards": n_guards,
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def on_env_reset(self, agent, grid, config):
        config["_current_trial"] = 0
        config["_trial_step"] = 0
        config["_goal_reached"] = False
        config["_guard_collision"] = False
        config["_gem_placed"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config

        # Draw guards
        for gx, gy in config.get("_guard_positions", []):
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

        # Place GEM highlight on first demo trial's correct target
        trials = config.get("trials", [])
        if trials and not trials[0].get("is_test", False):
            trial = trials[0]
            cx, cy = trial["targets"][trial["correct_idx"]]
            # Store the GEM position — we place it on the metadata layer so it
            # coexists visually with the GOAL object.  Since the grid object layer
            # already has GOAL there, we place the GEM on an adjacent empty cell
            # pointing toward the correct target (visual arrow), OR we briefly
            # swap the GOAL to GEM then back.  Simplest: place GEM at the correct
            # target position using the metadata layer as a flag and handle
            # rendering via a secondary object.  For maximum compatibility we
            # place GEM on a neighbouring cell.
            gem_pos = self._find_gem_neighbor(cx, cy, grid)
            if gem_pos is not None:
                gx, gy = gem_pos
                grid.objects[gy, gx] = ObjectType.GEM
                config["_gem_pos"] = gem_pos
                config["_gem_placed"] = True

    def _find_gem_neighbor(self, cx, cy, grid):
        """Find an adjacent empty cell to place the GEM highlight near (cx, cy)."""
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and grid.terrain[ny, nx] == CellType.EMPTY
                and grid.objects[ny, nx] == ObjectType.NONE
            ):
                return (nx, ny)
        return None

    # -----------------------------------------------------------------------
    # Agent movement
    # -----------------------------------------------------------------------
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
        needs_types = config.get("needs_types", False)

        # Check if agent is at the correct target position for this trial
        correct_idx = trial.get("correct_idx", 0)
        trial_targets = trial.get("targets", [])
        at_correct = correct_idx < len(trial_targets) and (x, y) == tuple(
            trial_targets[correct_idx]
        )
        obj_here = grid.objects[y, x]

        # For compositional rules, targets may be rendered as KEY/TOOL/POTION/SCROLL
        # instead of TARGET — treat those as "target-like" for stepping logic
        target_like = obj_here in (
            ObjectType.TARGET, ObjectType.KEY, ObjectType.TOOL,
            ObjectType.POTION, ObjectType.SCROLL,
        )

        if obj_here == ObjectType.GOAL or (at_correct and not is_test):
            if is_test:
                config["_goal_reached"] = True
            else:
                # Demo trial: agent reached the correct target
                self._clear_trial_objects(grid, trial, needs_types)
                self._remove_gem(grid, config)
                config["_current_trial"] = current + 1
                config["_trial_step"] = 0

                # Set up next trial
                next_trial_idx = current + 1
                if next_trial_idx < len(trials):
                    self._place_trial(grid, trials[next_trial_idx], config)

        elif target_like:
            if is_test:
                if at_correct:
                    config["_goal_reached"] = True
                else:
                    grid.objects[y, x] = ObjectType.NONE

    def _place_trial(self, grid, trial, config):
        """Place objects for a trial on the grid, with GEM highlight for demos."""
        needs_types = config.get("needs_types", False)
        is_test = trial.get("is_test", False)
        for i, (tx, ty) in enumerate(trial["targets"]):
            ttype = trial.get("target_types", [ObjectType.TARGET] * len(trial["targets"]))[i]
            if is_test:
                # Test trial: all targets look the same (TARGET or typed)
                if needs_types and ttype != ObjectType.TARGET:
                    grid.objects[ty, tx] = ttype
                else:
                    grid.objects[ty, tx] = ObjectType.TARGET
            elif i == trial["correct_idx"]:
                grid.objects[ty, tx] = ObjectType.GOAL
            else:
                if needs_types and ttype != ObjectType.TARGET:
                    grid.objects[ty, tx] = ttype
                else:
                    grid.objects[ty, tx] = ObjectType.TARGET

        # Place GEM highlight for demo trials
        if not is_test:
            cx, cy = trial["targets"][trial["correct_idx"]]
            gem_pos = self._find_gem_neighbor(cx, cy, grid)
            if gem_pos is not None:
                gx, gy = gem_pos
                grid.objects[gy, gx] = ObjectType.GEM
                config["_gem_pos"] = gem_pos
                config["_gem_placed"] = True
            else:
                config["_gem_placed"] = False

    def _clear_trial_objects(self, grid, trial, needs_types=False):
        target_objs = {ObjectType.GOAL, ObjectType.TARGET}
        if needs_types:
            target_objs |= set(_MARKER_TYPES)
        for tx, ty in trial.get("targets", []):
            obj = grid.objects[ty, tx]
            if obj in target_objs:
                grid.objects[ty, tx] = ObjectType.NONE

    def _remove_gem(self, grid, config):
        """Remove the GEM highlight if it is currently placed."""
        if config.get("_gem_placed", False):
            gem_pos = config.get("_gem_pos")
            if gem_pos is not None:
                gx, gy = gem_pos
                if grid.objects[gy, gx] == ObjectType.GEM:
                    grid.objects[gy, gx] = ObjectType.NONE
            config["_gem_placed"] = False

    # -----------------------------------------------------------------------
    # Per-step logic
    # -----------------------------------------------------------------------
    def on_env_step(self, agent, grid, config, step_count):
        # Demo trial: hide GOAL (and GEM) after reveal_steps
        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        needs_types = config.get("needs_types", False)
        if current < len(trials):
            trial = trials[current]
            if not trial.get("is_test", False):
                config["_trial_step"] = config.get("_trial_step", 0) + 1
                reveal = config.get("reveal_steps", 5)
                if config["_trial_step"] == reveal:
                    # Hide the GOAL marker, make it look like TARGET (or typed)
                    correct_idx = trial["correct_idx"]
                    tx, ty = trial["targets"][correct_idx]
                    if grid.objects[ty, tx] == ObjectType.GOAL:
                        ttype = trial.get(
                            "target_types",
                            [ObjectType.TARGET] * len(trial["targets"]),
                        )[correct_idx]
                        if needs_types and ttype != ObjectType.TARGET:
                            grid.objects[ty, tx] = ttype
                        else:
                            grid.objects[ty, tx] = ObjectType.TARGET
                    # Remove GEM highlight
                    self._remove_gem(grid, config)

        # Move guards
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])
        rng = config.get("_guard_rng")
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
                if (
                    0 < nx < grid.width - 1
                    and 0 < ny < grid.height - 1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] == ObjectType.NONE
                ):
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

    # -----------------------------------------------------------------------
    # Reward / done
    # -----------------------------------------------------------------------
    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_guard_collision", False):
            return -1.0
        reward = -0.01

        if config.get("_goal_reached", False) and not old_state.get("config", {}).get(
            "_goal_reached", False
        ):
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

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
