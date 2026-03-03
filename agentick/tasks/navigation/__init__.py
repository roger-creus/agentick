"""Navigation tasks."""

from agentick.tasks.navigation.curiosity_maze import CuriosityMazeTask
from agentick.tasks.navigation.dynamic_obstacles import DynamicObstaclesTask
from agentick.tasks.navigation.go_to_goal import GoToGoalTask
from agentick.tasks.navigation.instruction_following import InstructionFollowingTask
from agentick.tasks.navigation.maze_navigation import MazeNavigationTask
from agentick.tasks.navigation.multi_goal_route import ShortestPathTask
from agentick.tasks.navigation.recursive_rooms import RecursiveRoomsTask
from agentick.tasks.navigation.timing_challenge import TimingChallengeTask

__all__ = [
    "GoToGoalTask",
    "MazeNavigationTask",
    "ShortestPathTask",
    "DynamicObstaclesTask",
    "CuriosityMazeTask",
    "RecursiveRoomsTask",
    "TimingChallengeTask",
    "InstructionFollowingTask",
]
