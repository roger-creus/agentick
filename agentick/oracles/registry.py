"""Oracle registry — maps task names to oracle classes."""

from __future__ import annotations

from typing import Any

from agentick.oracles.base import OracleAgent

# Populated by imports below
_ORACLE_REGISTRY: dict[str, type[OracleAgent]] = {}


def register_oracle(task_name: str):
    """Decorator to register an oracle class for a task."""

    def decorator(cls: type[OracleAgent]) -> type[OracleAgent]:
        _ORACLE_REGISTRY[task_name] = cls
        return cls

    return decorator


def get_oracle(task_name: str, env: Any) -> OracleAgent:
    """Create an oracle instance for the given task.

    Args:
        task_name: Registered task name (e.g. ``"GoToGoal-v0"``).
        env: The Agentick environment instance.

    Returns:
        An :class:`OracleAgent` subclass instance.

    Raises:
        ValueError: If no oracle is registered for the task.
    """
    # Ensure all oracle modules are imported
    _import_all_oracles()

    if task_name not in _ORACLE_REGISTRY:
        available = ", ".join(sorted(_ORACLE_REGISTRY.keys()))
        raise ValueError(f"No oracle for '{task_name}'. Available: {available}")
    return _ORACLE_REGISTRY[task_name](env)


def list_oracles() -> list[str]:
    """Return sorted list of task names that have oracles."""
    _import_all_oracles()
    return sorted(_ORACLE_REGISTRY.keys())


_imported = False


def _import_all_oracles() -> None:
    global _imported
    if _imported:
        return
    _imported = True
    # Import all oracle modules to trigger registration
    import agentick.oracles.generalization_oracles  # noqa: F401
    import agentick.oracles.memory_oracles  # noqa: F401
    import agentick.oracles.multi_agent_oracles  # noqa: F401
    import agentick.oracles.navigation_oracles  # noqa: F401
    import agentick.oracles.planning_oracles  # noqa: F401
    import agentick.oracles.reasoning_oracles  # noqa: F401
