"""Wrapper to convert state_dict observations to flat feature vectors.

For RL training with state-based observations.
"""

import gymnasium as gym
import numpy as np

from agentick.core.feature_extractor import extract_state_features, get_state_feature_space


class StateFeaturesWrapper(gym.ObservationWrapper):
    """Convert state_dict observations to flat feature vectors."""

    def __init__(self, env: gym.Env, grid_size: tuple[int, int] = (20, 20)):
        """Initialize wrapper.

        Args:
            env: Environment with state_dict render_mode
            grid_size: Maximum grid size for feature extraction
        """
        super().__init__(env)
        self.grid_size = grid_size

        # Override observation space
        self.observation_space = get_state_feature_space(grid_size)

    def observation(self, observation: dict) -> np.ndarray:
        """Convert state_dict to flat features.

        Args:
            observation: state_dict from environment

        Returns:
            Flat numpy array of features
        """
        return extract_state_features(observation, self.grid_size)
