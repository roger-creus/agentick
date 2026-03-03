"""Tasks package - imports all task modules to trigger registration."""

from agentick.tasks import (
    generalization,
    memory,
    multi_agent,
    navigation,
    planning,
    reasoning,
)

__all__ = [
    "navigation",
    "planning",
    "reasoning",
    "memory",
    "generalization",
    "multi_agent",
]
