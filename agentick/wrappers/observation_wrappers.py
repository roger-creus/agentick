"""Observation wrappers for different agent types."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TextObservationWrapper(gym.ObservationWrapper):
    """Wrapper that returns text observations."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Text(max_length=100000)

    def observation(self, obs):
        """Convert observation to text."""
        if isinstance(obs, str):
            return obs
        return self.env.get_text_observation()


class PixelObservationWrapper(gym.ObservationWrapper):
    """Wrapper that returns RGB pixel observations."""

    def __init__(self, env):
        super().__init__(env)
        # Get pixel dimensions from env
        pixel_obs = env.get_pixel_observation()
        self.observation_space = spaces.Box(low=0, high=255, shape=pixel_obs.shape, dtype=np.uint8)

    def observation(self, obs):
        """Convert observation to pixels."""
        return self.env.get_pixel_observation()


class DictObservationWrapper(gym.ObservationWrapper):
    """Wrapper that returns structured dictionary observations."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({})

    def observation(self, obs):
        """Convert observation to dict."""
        return self.env.get_state_dict()


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Wrapper that flattens grid to 1D vector."""

    def __init__(self, env):
        super().__init__(env)
        grid = env.grid
        flat_size = grid.height * grid.width * 4  # 4 layers
        self.observation_space = spaces.Box(low=0, high=255, shape=(flat_size,), dtype=np.float32)

    def observation(self, obs):
        """Flatten grid layers to vector."""
        grid = self.env.grid
        terrain = grid.terrain.flatten()
        objects = grid.objects.flatten()
        agents = grid.agents.flatten()
        metadata = grid.metadata.flatten()
        return np.concatenate([terrain, objects, agents, metadata]).astype(np.float32)


class LanguageActionWrapper(gym.ActionWrapper):
    """Wrapper that accepts natural language action strings."""

    def __init__(self, env):
        super().__init__(env)
        self.action_names = env.action_space_obj.get_all_action_names()

    def action(self, action):
        """Convert language action to discrete action."""
        if isinstance(action, str):
            return self.env.action_space_obj.parse_action_name(action)
        return action
