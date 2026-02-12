"""Universal agent protocol that all adapters implement."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Universal agent interface that all adapters must implement.

    This protocol allows ANY agent (API, local model, custom code, etc.)
    to be evaluated on Agentick benchmarks.
    """

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select an action given observation and info dict.

        Args:
            observation: Environment observation (format depends on obs_mode)
            info: Info dict with metadata:
                - valid_actions: List of valid action indices
                - task_name: Name of current task
                - difficulty: Difficulty level
                - step: Current step number

        Returns:
            Action index (integer)
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each episode.

        Agents can use this to reset any internal state (memory, hidden states, etc.).
        """
        ...

    @property
    def name(self) -> str:
        """Return agent name for identification."""
        ...


class Agent:
    """
    Base class for implementing agents (alternative to Protocol).

    Users can inherit from this class instead of implementing the Protocol directly.
    """

    def __init__(self, name: str = "BaseAgent"):
        self._name = name

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action given observation.

        Override this method in your agent.
        """
        raise NotImplementedError("Subclasses must implement act()")

    def reset(self) -> None:
        """Reset agent state between episodes."""
        pass  # Default: no state to reset

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name


def validate_agent(agent: Any) -> bool:
    """
    Validate that an object implements AgentProtocol.

    Args:
        agent: Object to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(agent, AgentProtocol):
        return False

    # Check required methods exist and are callable
    if not callable(getattr(agent, "act", None)):
        return False

    if not callable(getattr(agent, "reset", None)):
        return False

    if not hasattr(agent, "name"):
        return False

    return True
