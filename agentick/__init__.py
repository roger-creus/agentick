"""Agentick: Universal benchmark for evaluating generally capable AI agents."""

__version__ = "0.1.0"

# Import registry functions
from agentick.core.actions import ActionSpace, ActionType
from agentick.core.entity import Agent, Entity

# Export core classes for custom task development
from agentick.core.env import AgentickEnv
from agentick.core.grid import Grid
from agentick.tasks.base import TaskSpec
from agentick.tasks.registry import list_tasks, make, make_suite, register_task

__all__ = [
    "__version__",
    "make",
    "list_tasks",
    "register_task",
    "make_suite",
    "AgentickEnv",
    "Grid",
    "Entity",
    "Agent",
    "ActionSpace",
    "ActionType",
    "TaskSpec",
]
