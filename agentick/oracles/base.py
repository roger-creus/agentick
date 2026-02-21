"""Base class for oracle agents."""

from __future__ import annotations

from typing import Any

from agentick.coding_api import AgentickAPI


class OracleAgent:
    """Base oracle agent that uses the Coding API to solve tasks.

    Subclasses override :meth:`plan` to populate ``self.action_queue``
    with a sequence of actions to execute.
    """

    def __init__(self, env: Any) -> None:
        self.api = AgentickAPI(env)
        self.action_queue: list[int] = []

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Called after ``env.reset()``."""
        self.api.update(obs, info)
        self.action_queue = []
        self.plan()

    def update(self, obs: Any, info: dict[str, Any]) -> None:
        """Called after ``env.step()``."""
        self.api.update(obs, info)

    def act(self, obs: Any, info: dict[str, Any]) -> int:
        """Return the next action integer."""
        self.api.update(obs, info)
        if not self.action_queue:
            self.plan()
        if self.action_queue:
            return self.action_queue.pop(0)
        return 0  # noop fallback

    def plan(self) -> None:
        """Populate ``self.action_queue``. Override in subclasses."""
        raise NotImplementedError
