"""Skill discovery tasks."""

from agentick.tasks.skill.emergent_strategy import EmergentStrategyTask
from agentick.tasks.skill.multi_room_escape import MultiRoomEscapeTask
from agentick.tasks.skill.recipe_assembly import RecipeAssemblyTask
from agentick.tasks.skill.resource_management import ResourceManagementTask
from agentick.tasks.skill.tool_use import ToolUseTask

__all__ = [
    "ToolUseTask",
    "RecipeAssemblyTask",
    "MultiRoomEscapeTask",
    "ResourceManagementTask",
    "EmergentStrategyTask",
]
