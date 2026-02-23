"""Exploration tasks — tasks requiring systematic exploration and discovery."""

from agentick.tasks.exploration.curiosity_maze import CuriosityMazeTask
from agentick.tasks.exploration.fog_of_war import FogOfWarExplorationTask
from agentick.tasks.exploration.treasure_hunt import TreasureHuntTask

__all__ = [
    "FogOfWarExplorationTask",
    "TreasureHuntTask",
    "CuriosityMazeTask",
]
