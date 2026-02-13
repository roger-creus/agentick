"""Reward shaping wrappers."""

import gymnasium as gym


class DenseRewardWrapper(gym.RewardWrapper):
    """Override to use dense rewards."""

    def __init__(self, env):
        super().__init__(env)
        env.reward_mode = "dense"

    def reward(self, reward):
        """Pass through the reward unchanged.

        Args:
            reward: Original reward from environment.

        Returns:
            The same reward value.
        """
        return reward


class SparseRewardWrapper(gym.RewardWrapper):
    """Override to use sparse rewards."""

    def __init__(self, env):
        super().__init__(env)
        env.reward_mode = "sparse"

    def reward(self, reward):
        """Pass through the reward unchanged.

        Args:
            reward: Original reward from environment.

        Returns:
            The same reward value.
        """
        return reward


class RewardScaleWrapper(gym.RewardWrapper):
    """Scale rewards to target range."""

    def __init__(self, env, scale=1.0, shift=0.0):
        super().__init__(env)
        self.scale = scale
        self.shift = shift

    def reward(self, reward):
        """Scale and shift the reward.

        Args:
            reward: Original reward from environment.

        Returns:
            Transformed reward: reward * scale + shift.
        """
        return reward * self.scale + self.shift


class CurriculumWrapper(gym.Wrapper):
    """Auto-advance difficulty on success."""

    def __init__(self, env, success_threshold=0.8, window_size=10):
        super().__init__(env)
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.recent_successes = []

    def step(self, action):
        """Execute action and track success rate for curriculum progression.

        Args:
            action: Action to execute.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            success = info.get("success", False)
            self.recent_successes.append(success)
            if len(self.recent_successes) > self.window_size:
                self.recent_successes.pop(0)

            # Check if should advance difficulty
            if len(self.recent_successes) >= self.window_size:
                success_rate = sum(self.recent_successes) / len(self.recent_successes)
                if success_rate >= self.success_threshold:
                    # Advance difficulty if environment supports it
                    if hasattr(self.env.unwrapped, "set_difficulty"):
                        current = getattr(self.env.unwrapped, "difficulty", "easy")
                        next_difficulty = {"easy": "medium", "medium": "hard"}.get(current)
                        if next_difficulty:
                            self.env.unwrapped.set_difficulty(next_difficulty)
                            self.recent_successes.clear()  # Reset tracking

        return obs, reward, terminated, truncated, info
