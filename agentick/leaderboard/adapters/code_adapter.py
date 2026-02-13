"""Code-based agent adapter for loading user Python files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from agentick.leaderboard.adapters.protocol import AgentProtocol, validate_agent


class CodeAgent:
    """
    Adapter for loading agents from Python code files.

    The user provides a Python file with a class implementing AgentProtocol.
    """

    def __init__(
        self,
        script_path: str,
        class_name: str = "Agent",
        **init_kwargs,
    ):
        """
        Initialize code agent adapter.

        Args:
            script_path: Path to Python file
            class_name: Name of the agent class to instantiate
            **init_kwargs: Arguments to pass to agent constructor
        """
        self.script_path = Path(script_path)
        self.class_name = class_name
        self.init_kwargs = init_kwargs

        # Load the agent
        self.agent = self._load_agent()

        # Validate it implements the protocol
        if not validate_agent(self.agent):
            raise ValueError(
                f"Agent class '{class_name}' does not implement AgentProtocol. "
                "Must have act(), reset(), and name property."
            )

    def _load_agent(self) -> Any:
        """
        Load agent from Python file.

        Returns:
            Agent instance
        """
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")

        # Import the module
        spec = importlib.util.spec_from_file_location("user_agent", self.script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {self.script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["user_agent"] = module
        spec.loader.exec_module(module)

        # Get the agent class
        if not hasattr(module, self.class_name):
            available = [name for name in dir(module) if not name.startswith("_")]
            raise AttributeError(
                f"Class '{self.class_name}' not found in {self.script_path}. Available: {available}"
            )

        agent_class = getattr(module, self.class_name)

        # Instantiate
        return agent_class(**self.init_kwargs)

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """Delegate to loaded agent."""
        return self.agent.act(observation, info)

    def reset(self) -> None:
        """Delegate to loaded agent."""
        self.agent.reset()

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent.name


def load_agent_from_code(script_path: str, class_name: str = "Agent", **kwargs) -> AgentProtocol:
    """
    Convenience function to load agent from code file.

    Args:
        script_path: Path to Python file
        class_name: Name of agent class
        **kwargs: Arguments for agent constructor

    Returns:
        Agent instance implementing AgentProtocol
    """
    return CodeAgent(script_path, class_name, **kwargs)
