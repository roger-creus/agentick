"""Adaptive curriculum that adjusts difficulty based on performance."""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

DifficultyLevel = Literal["easy", "medium", "hard", "expert"]


class AdaptiveCurriculum:
    """
    Adaptive curriculum that advances/regresses based on success rate.

    Tracks agent performance over a sliding window and automatically
    adjusts difficulty when thresholds are crossed.

    Example:
        >>> curriculum = AdaptiveCurriculum(
        ...     initial_difficulty="easy",
        ...     advance_threshold=0.8,
        ...     regress_threshold=0.2,
        ...     window_size=50,
        ... )
        >>>
        >>> # After each episode:
        >>> success = episode_return > 0.0
        >>> curriculum.update(success)
        >>> difficulty = curriculum.get_difficulty()
    """

    DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]

    def __init__(
        self,
        initial_difficulty: DifficultyLevel = "easy",
        advance_threshold: float = 0.8,
        regress_threshold: float = 0.2,
        window_size: int = 50,
        min_episodes_per_level: int = 20,
    ):
        """
        Initialize adaptive curriculum.

        Args:
            initial_difficulty: Starting difficulty
            advance_threshold: Success rate to advance (e.g., 0.8 = 80%)
            regress_threshold: Success rate to regress (e.g., 0.2 = 20%)
            window_size: Number of recent episodes to track
            min_episodes_per_level: Minimum episodes before allowing change
        """
        self.current_difficulty = initial_difficulty
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold
        self.window_size = window_size
        self.min_episodes_per_level = min_episodes_per_level

        # Track recent performance
        self.recent_successes = deque(maxlen=window_size)
        self.episodes_at_current_level = 0

        # Statistics
        self.total_episodes = 0
        self.difficulty_history = []
        self.success_rate_history = []

    def get_difficulty(self) -> DifficultyLevel:
        """Get current difficulty level."""
        return self.current_difficulty

    def get_difficulty_index(self) -> int:
        """Get index of current difficulty (0=easy, 3=expert)."""
        return self.DIFFICULTY_ORDER.index(self.current_difficulty)

    def update(self, success: bool) -> bool:
        """
        Update curriculum with episode result.

        Args:
            success: Whether episode was successful

        Returns:
            True if difficulty changed, False otherwise
        """
        self.recent_successes.append(1 if success else 0)
        self.episodes_at_current_level += 1
        self.total_episodes += 1

        # Track history
        self.difficulty_history.append(self.current_difficulty)
        success_rate = self.get_success_rate()
        self.success_rate_history.append(success_rate)

        # Check if enough episodes to consider changing difficulty
        if self.episodes_at_current_level < self.min_episodes_per_level:
            return False

        # Check if window is full
        if len(self.recent_successes) < self.window_size:
            return False

        changed = False

        # Check for advance
        if success_rate >= self.advance_threshold:
            if self._advance_difficulty():
                changed = True

        # Check for regress
        elif success_rate <= self.regress_threshold:
            if self._regress_difficulty():
                changed = True

        return changed

    def get_success_rate(self) -> float:
        """Get recent success rate (0-1)."""
        if len(self.recent_successes) == 0:
            return 0.0
        return np.mean(self.recent_successes)

    def _advance_difficulty(self) -> bool:
        """Advance to next difficulty level if possible."""
        current_idx = self.get_difficulty_index()
        if current_idx >= len(self.DIFFICULTY_ORDER) - 1:
            return False  # Already at max difficulty

        self.current_difficulty = self.DIFFICULTY_ORDER[current_idx + 1]
        self.episodes_at_current_level = 0
        self.recent_successes.clear()
        return True

    def _regress_difficulty(self) -> bool:
        """Regress to previous difficulty level if possible."""
        current_idx = self.get_difficulty_index()
        if current_idx <= 0:
            return False  # Already at min difficulty

        self.current_difficulty = self.DIFFICULTY_ORDER[current_idx - 1]
        self.episodes_at_current_level = 0
        self.recent_successes.clear()
        return True

    def get_stats(self) -> dict:
        """Get curriculum statistics."""
        return {
            "current_difficulty": self.current_difficulty,
            "success_rate": self.get_success_rate(),
            "episodes_at_level": self.episodes_at_current_level,
            "total_episodes": self.total_episodes,
            "difficulty_index": self.get_difficulty_index(),
        }

    def reset(self):
        """Reset curriculum to initial state."""
        self.current_difficulty = self.DIFFICULTY_ORDER[0]
        self.recent_successes.clear()
        self.episodes_at_current_level = 0
        self.total_episodes = 0
        self.difficulty_history.clear()
        self.success_rate_history.clear()
