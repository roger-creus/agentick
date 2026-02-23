"""Tasks package - imports all task modules to trigger registration."""

from agentick.tasks import (
    adversarial,
    combinatorial,
    compositional,
    control,
    exploration,
    memory,
    meta,
    multi_agent,
    navigation,
    reasoning,
    skill,
)

__all__ = [
    "navigation",
    "exploration",
    "memory",
    "reasoning",
    "skill",
    "control",
    "combinatorial",
    "multi_agent",
    "compositional",
    "adversarial",
    "meta",
]
