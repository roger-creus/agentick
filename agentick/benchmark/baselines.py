"""Baseline agents."""

import numpy as np


class RandomAgent:
    """Uniform random action selection."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, valid_actions):
        """Select random action from valid actions."""
        if valid_actions is None or len(valid_actions) == 0:
            return 0
        return self.rng.choice(valid_actions)

    def __call__(self, obs, info):
        """Select random action from valid actions."""
        valid_actions = info.get("valid_actions", [])
        if not valid_actions:
            return 0
        # Map action name to index
        return self.rng.integers(0, len(valid_actions))
