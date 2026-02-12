"""Manual curriculum with predefined difficulty progression."""

from __future__ import annotations

from typing import Literal

DifficultyLevel = Literal["easy", "medium", "hard", "expert"]


class ManualCurriculum:
    """
    Manual curriculum with predefined difficulty sequence.

    Example:
        >>> curriculum = ManualCurriculum(
        ...     difficulties=["easy", "medium", "hard"],
        ...     episodes_per_level=[100, 200, 300],
        ... )
        >>> difficulty = curriculum.get_difficulty(episode=0)  # "easy"
        >>> difficulty = curriculum.get_difficulty(episode=150)  # "medium"
    """

    def __init__(
        self,
        difficulties: list[DifficultyLevel],
        episodes_per_level: list[int] | None = None,
    ):
        """
        Initialize manual curriculum.

        Args:
            difficulties: Ordered list of difficulties
            episodes_per_level: Episodes to spend at each level (if None, equal split)
        """
        self.difficulties = difficulties
        self.num_levels = len(difficulties)

        if episodes_per_level is None:
            # Default: 100 episodes per level
            self.episodes_per_level = [100] * self.num_levels
        else:
            assert len(episodes_per_level) == self.num_levels
            self.episodes_per_level = episodes_per_level

        # Compute cumulative episodes for level transitions
        self.cumulative_episodes = []
        total = 0
        for count in self.episodes_per_level:
            total += count
            self.cumulative_episodes.append(total)

    def get_difficulty(self, episode: int) -> DifficultyLevel:
        """
        Get difficulty for given episode number.

        Args:
            episode: Episode number (0-indexed)

        Returns:
            Difficulty level
        """
        for i, threshold in enumerate(self.cumulative_episodes):
            if episode < threshold:
                return self.difficulties[i]

        # After all levels, stay at hardest
        return self.difficulties[-1]

    def get_level(self, episode: int) -> int:
        """
        Get curriculum level (0-indexed).

        Args:
            episode: Episode number

        Returns:
            Level index
        """
        for i, threshold in enumerate(self.cumulative_episodes):
            if episode < threshold:
                return i
        return self.num_levels - 1

    def get_progress(self, episode: int) -> float:
        """
        Get progress through curriculum (0-1).

        Args:
            episode: Episode number

        Returns:
            Progress ratio
        """
        if episode >= self.cumulative_episodes[-1]:
            return 1.0

        level = self.get_level(episode)
        if level == 0:
            level_start = 0
        else:
            level_start = self.cumulative_episodes[level - 1]

        level_end = self.cumulative_episodes[level]
        level_progress = (episode - level_start) / (level_end - level_start)

        # Overall progress
        return (level + level_progress) / self.num_levels
