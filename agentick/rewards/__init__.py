"""Reward engineering utilities for training agents.

Provides reward shaping, intrinsic motivation, and composite rewards.
"""

from agentick.rewards.composite import CompositeReward
from agentick.rewards.intrinsic import CuriosityReward, ExplorationBonus
from agentick.rewards.potential import PotentialBasedReward

__all__ = [
    "PotentialBasedReward",
    "ExplorationBonus",
    "CuriosityReward",
    "CompositeReward",
]
