"""Git repository adapter for loading agents from remote repos."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentick.leaderboard.adapters.code_adapter import CodeAgent


class GitRepoAgent:
    """
    Adapter for loading agents from Git repositories.

    Clones the repo, runs setup, imports the agent, then cleans up.
    """

    def __init__(
        self,
        url: str,
        branch: str = "main",
        setup_cmd: str | None = None,
        script_path: str = "agent.py",
        class_name: str = "Agent",
        **agent_kwargs,
    ):
        """
        Initialize Git repo agent adapter.

        Args:
            url: Git repository URL
            branch: Branch to clone
            setup_cmd: Setup command to run (e.g., "pip install -e .")
            script_path: Path to agent script within repo
            class_name: Name of agent class
            **agent_kwargs: Arguments for agent constructor
        """
        self.url = url
        self.branch = branch
        self.setup_cmd = setup_cmd
        self.script_path = script_path
        self.class_name = class_name
        self.agent_kwargs = agent_kwargs

        # Clone and setup
        self.temp_dir = tempfile.mkdtemp(prefix="agentick_git_")
        self.repo_dir = Path(self.temp_dir) / "repo"

        self._clone_repo()
        self._run_setup()
        self._load_agent()

    def _clone_repo(self):
        """Clone Git repository."""
        subprocess.run(
            ["git", "clone", "-b", self.branch, self.url, str(self.repo_dir)],
            check=True,
            capture_output=True,
        )

    def _run_setup(self):
        """Run setup command if specified."""
        if self.setup_cmd:
            subprocess.run(
                self.setup_cmd,
                shell=True,
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
            )

    def _load_agent(self):
        """Load agent from repository."""
        agent_path = self.repo_dir / self.script_path
        self.agent = CodeAgent(
            script_path=str(agent_path),
            class_name=self.class_name,
            **self.agent_kwargs,
        )

    def reset(self) -> None:
        """Reset agent state."""
        self.agent.reset()

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent.name

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """Delegate to loaded agent."""
        return self.agent.act(observation, info)

    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
