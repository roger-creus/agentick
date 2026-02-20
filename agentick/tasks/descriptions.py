"""Dynamic task description extraction from the registry."""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskDescription:
    """Structured description of an Agentick task."""

    name: str
    description: str
    detailed_description: str
    category: str
    capability_tags: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_category(task_class: type) -> str:
    """Derive category from module path (e.g. tasks.navigation.go_to_goal -> navigation)."""
    module = task_class.__module__ or ""
    parts = module.split(".")
    # Pattern: agentick.tasks.<category>.<task_module>
    try:
        idx = parts.index("tasks")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return "unknown"


def _get_detailed_description(task_class: type) -> str:
    """Extract detailed description from class docstring."""
    doc = inspect.getdoc(task_class)
    if doc:
        return textwrap.dedent(doc).strip()
    return ""


def get_all_task_descriptions() -> dict[str, TaskDescription]:
    """Return structured descriptions for all registered tasks.

    Returns:
        Mapping of task name -> TaskDescription.
    """
    from agentick.tasks.registry import _TASK_REGISTRY

    descriptions: dict[str, TaskDescription] = {}

    for name, task_class in sorted(_TASK_REGISTRY.items()):
        descriptions[name] = TaskDescription(
            name=name,
            description=getattr(task_class, "description", ""),
            detailed_description=_get_detailed_description(task_class),
            category=_get_category(task_class),
            capability_tags=list(getattr(task_class, "capability_tags", [])),
            difficulties=list(getattr(task_class, "difficulty_configs", {}).keys()),
        )

    return descriptions


def get_task_description(task_name: str) -> str:
    """Get a natural-language description for *task_name*.

    Looks up the task class in the registry and returns its docstring (detailed)
    or its short ``description`` attribute.  Falls back to a generic string only
    when the task is not registered at all.

    Args:
        task_name: Registered task name (e.g. ``"LightsOut-v0"``).

    Returns:
        Human-readable task description.
    """
    from agentick.tasks.registry import _TASK_REGISTRY

    task_class = _TASK_REGISTRY.get(task_name)
    if task_class is None:
        return "Complete the task objective."

    # Prefer the full docstring; fall back to the short description attribute.
    detailed = _get_detailed_description(task_class)
    if detailed:
        return detailed

    short = getattr(task_class, "description", "")
    if short:
        return short

    return "Complete the task objective."
