"""Oracle bots for meta-learning tasks."""

from __future__ import annotations

from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("FewShotAdaptation-v0")
class FewShotAdaptationOracle(OracleAgent):
    """Oracle knows the hidden rule and trial structure from config.

    Uses privileged config access to know which target is correct in each
    trial. Navigates directly to the correct target position.
    Guards at hard+ require step-by-step re-planning with avoidance.
    """

    def plan(self):
        config = self.api.task_config
        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        ax, ay = self.api.agent_position

        # Build avoid set for guards
        avoid = set()
        for e in self.api.get_entities():
            if e.entity_type in ("npc", "enemy"):
                avoid.add(e.position)
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    avoid.add((e.position[0] + dx, e.position[1] + dy))

        if current < len(trials):
            trial = trials[current]
            correct_idx = trial.get("correct_idx", 0)
            trial_targets = trial.get("targets", [])
            if correct_idx < len(trial_targets):
                tx, ty = trial_targets[correct_idx]
                if avoid:
                    path = self.api.bfs_path_positions(
                        (ax, ay),
                        (tx, ty),
                        avoid=avoid,
                    )
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
                    # Try without adjacent avoidance
                    avoid_exact = set()
                    for e in self.api.get_entities():
                        if e.entity_type in ("npc", "enemy"):
                            avoid_exact.add(e.position)
                    path = self.api.bfs_path_positions(
                        (ax, ay),
                        (tx, ty),
                        avoid=avoid_exact,
                    )
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
                self.action_queue = self.api.move_toward(tx, ty)
                return

        # All trials done or test trial - use true_goal
        true_goal = config.get("true_goal")
        if true_goal:
            tgx, tgy = true_goal
            if avoid:
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    (tgx, tgy),
                    avoid=avoid,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
            self.action_queue = self.api.move_toward(tgx, tgy)
            return

        self.action_queue = [0]


@register_oracle("TaskInterference-v0")
class TaskInterferenceOracle(OracleAgent):
    """Complete competing objectives with sequential collect-deliver.

    Strategy: collect ALL coins first (avoiding gems), deliver at coin goal,
    then collect ALL gems and deliver at gem goal. The task places enough
    extra items so that even with interference destruction this works.
    """

    def _nav_to(self, target, avoid):
        ax, ay = self.api.agent_position
        if avoid:
            path = self.api.bfs_path_positions((ax, ay), target, avoid=avoid)
            if path:
                acts = self.api.positions_to_actions(path)
                if acts:
                    self.action_queue = [acts[0]]
                    return True
            # Fallback: no avoidance
        path = self.api.bfs_path_positions((ax, ay), target)
        if path:
            acts = self.api.positions_to_actions(path)
            if acts:
                self.action_queue = [acts[0]]
                return True
        self.action_queue = self.api.move_toward(*target)
        return True

    def plan(self):
        config = self.api.task_config
        coins_delivered = config.get("_coins_delivered", 0)
        gems_delivered = config.get("_gems_delivered", 0)
        coins_held = config.get("_coins_held", 0)
        gems_held = config.get("_gems_held", 0)
        n_coins = config.get("n_coins", 2)
        n_gems = config.get("n_gems", 2)
        interference = config.get("interference", False)

        # Build avoidance sets
        gem_positions = {e.position for e in self.api.get_entities_of_type("gem")}
        coin_positions = {e.position for e in self.api.get_entities_of_type("coin")}

        # Phase 1: deliver any held coins
        if coins_held > 0:
            coin_goal = config.get("coin_goal")
            if coin_goal:
                avoid = gem_positions if interference else set()
                self._nav_to(tuple(coin_goal), avoid)
                return

        # Phase 2: deliver any held gems
        if gems_held > 0:
            gem_goal = config.get("gem_goal")
            if gem_goal:
                avoid = coin_positions if interference else set()
                self._nav_to(tuple(gem_goal), avoid)
                return

        # Phase 3: collect coins first if still needed
        if coins_delivered < n_coins:
            coins = self.api.get_entities_of_type("coin")
            if coins:
                nearest = min(coins, key=lambda c: c.distance)
                avoid = gem_positions if interference else set()
                self._nav_to(nearest.position, avoid)
                return

        # Phase 4: then collect gems
        if gems_delivered < n_gems:
            gems = self.api.get_entities_of_type("gem")
            if gems:
                nearest = min(gems, key=lambda g: g.distance)
                avoid = coin_positions if interference else set()
                self._nav_to(nearest.position, avoid)
                return

        self.action_queue = [0]
