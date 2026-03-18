"""Standard Atari-style preprocessing for pixel observations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ResizeObservation(gym.ObservationWrapper):
    """Resize pixel observations to target size."""

    def __init__(self, env: gym.Env, size: tuple[int, int] = (84, 84)):
        """
        Initialize ResizeObservation wrapper.

        Args:
            env: Environment to wrap
            size: Target size as (height, width)
        """
        super().__init__(env)
        self.size = size
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size[0], size[1], old_space.shape[2]),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation to target size."""
        from PIL import Image

        img = Image.fromarray(obs)
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)


class GrayscaleObservation(gym.ObservationWrapper):
    """Convert RGB to grayscale."""

    def __init__(self, env: gym.Env):
        """
        Initialize GrayscaleObservation wrapper.

        Args:
            env: Environment to wrap
        """
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(old_space.shape[0], old_space.shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert RGB observation to grayscale."""
        gray = np.mean(obs, axis=2, keepdims=True).astype(np.uint8)
        return gray


class FrameStack(gym.Wrapper):
    """Stack last N frames as channels."""

    def __init__(self, env: gym.Env, n_frames: int = 4):
        """
        Initialize FrameStack wrapper.

        Args:
            env: Environment to wrap
            n_frames: Number of frames to stack
        """
        super().__init__(env)
        self.n_frames = n_frames
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                old_space.shape[0],
                old_space.shape[1],
                old_space.shape[2] * n_frames,
            ),
            dtype=np.uint8,
        )
        self.frames = np.zeros(self.observation_space.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        # Fill frame stack with initial observation
        for i in range(self.n_frames):
            self.frames[:, :, i * obs.shape[2] : (i + 1) * obs.shape[2]] = obs
        return self.frames.copy(), info

    def step(self, action):
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Roll frames and add new frame
        self.frames = np.roll(self.frames, -obs.shape[2], axis=2)
        self.frames[:, :, -obs.shape[2] :] = obs
        return self.frames.copy(), reward, terminated, truncated, info


def make_atari_env(task_name: str, seed: int = 0, **kwargs) -> gym.Env:
    """
    Create an Agentick env with standard Atari preprocessing.

    Applies: pixels → resize 84x84 → grayscale → frame stack 4.

    Args:
        task_name: Agentick task name (e.g., "GoToGoal-v0")
        seed: Random seed
        **kwargs: Additional args passed to agentick.make(), including render_mode
            (e.g. "rgb_array"). Defaults to "rgb_array"
            if not specified.

    Returns:
        Wrapped gymnasium environment with (84, 84, 4) uint8 observations.
    """
    import agentick

    kwargs.setdefault("render_mode", "rgb_array")
    env = agentick.make(task_name, **kwargs)
    env = ResizeObservation(env, size=(84, 84))
    env = GrayscaleObservation(env)
    env = FrameStack(env, n_frames=4)
    return env
