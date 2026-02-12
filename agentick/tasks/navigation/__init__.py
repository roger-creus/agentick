"""Navigation tasks."""

from agentick.tasks.navigation.dynamic_obstacles import DynamicObstaclesTask
from agentick.tasks.navigation.fog_of_war import FogOfWarExplorationTask
from agentick.tasks.navigation.go_to_goal import GoToGoalTask
from agentick.tasks.navigation.maze_navigation import MazeNavigationTask
from agentick.tasks.navigation.multi_goal_route import MultiGoalRouteTask

__all__ = [
    "GoToGoalTask",
    "MazeNavigationTask",
    "MultiGoalRouteTask",
    "DynamicObstaclesTask",
    "FogOfWarExplorationTask",
]
