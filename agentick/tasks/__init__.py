"""Tasks package - imports all task modules to trigger registration."""

from agentick.tasks import (
    adversarial,
    combinatorial,
    compositional,
    control,
    memory,
    meta,
    multi_agent,
    navigation,
    reasoning,
    skill,
    worldmodel,
)

__all__ = [
    "navigation",
    "memory",
    "reasoning",
    "skill",
    "control",
    "combinatorial",
    "multi_agent",
    "worldmodel",
    "compositional",
    "adversarial",
    "meta",
]
