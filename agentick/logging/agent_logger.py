"""Agent-side logging protocol."""

from __future__ import annotations

from typing import Any, Protocol


class LoggableAgent(Protocol):
    """Protocol for agents that support logging."""

    def get_log_data(self) -> dict[str, Any]:
        """
        Return agent internals for logging.

        Returns:
            Dict with agent-specific data to log
        """
        ...


class LLMAgentLogger:
    """Logger for LLM agents."""

    def __init__(self):
        self.prompts: list[str] = []
        self.responses: list[str] = []
        self.parsed_actions: list[Any] = []
        self.token_counts: list[dict[str, int]] = []
        self.latencies: list[float] = []
        self.costs: list[float] = []

    def log_call(
        self,
        prompt: str,
        response: str,
        parsed_action: Any,
        token_count: dict[str, int],
        latency: float,
        cost: float,
    ) -> None:
        """Log LLM API call."""
        self.prompts.append(prompt)
        self.responses.append(response)
        self.parsed_actions.append(parsed_action)
        self.token_counts.append(token_count)
        self.latencies.append(latency)
        self.costs.append(cost)

    def export_transcript(self, output_path: str) -> None:
        """Export as readable conversation transcript."""
        with open(output_path, "w") as f:
            for i, (prompt, response, action) in enumerate(
                zip(self.prompts, self.responses, self.parsed_actions)
            ):
                f.write(f"=== Step {i + 1} ===\n")
                f.write(f"[PROMPT]\n{prompt}\n\n")
                f.write(f"[RESPONSE]\n{response}\n\n")
                f.write(f"[PARSED ACTION]\n{action}\n\n")
                f.write("---\n\n")


class RLAgentLogger:
    """Logger for RL agents."""

    def __init__(self):
        self.action_probs: list[Any] = []
        self.value_estimates: list[float] = []
        self.entropies: list[float] = []

    def log_step(self, action_probs: Any, value: float, entropy: float) -> None:
        """Log RL agent step."""
        self.action_probs.append(action_probs)
        self.value_estimates.append(value)
        self.entropies.append(entropy)


class SearchAgentLogger:
    """Logger for search-based agents (A*, MCTS, etc.)."""

    def __init__(self):
        self.nodes_expanded: list[int] = []
        self.search_depths: list[int] = []
        self.search_times: list[float] = []
        self.paths: list[list[Any]] = []
        self.evaluations: list[dict[str, float]] = []

    def log_search(
        self,
        nodes_expanded: int,
        max_depth: int,
        search_time: float,
        path: list[Any] | None = None,
        evaluations: dict[str, float] | None = None,
    ) -> None:
        """
        Log search execution details.

        Args:
            nodes_expanded: Number of nodes expanded during search
            max_depth: Maximum search depth reached
            search_time: Time spent searching (seconds)
            path: Found path/plan (if any)
            evaluations: Node evaluation scores (e.g., heuristic values)
        """
        self.nodes_expanded.append(nodes_expanded)
        self.search_depths.append(max_depth)
        self.search_times.append(search_time)
        self.paths.append(path if path is not None else [])
        self.evaluations.append(evaluations if evaluations is not None else {})

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_searches": len(self.nodes_expanded),
            "total_nodes_expanded": sum(self.nodes_expanded),
            "mean_nodes_per_search": (
                sum(self.nodes_expanded) / len(self.nodes_expanded) if self.nodes_expanded else 0
            ),
            "max_depth": max(self.search_depths) if self.search_depths else 0,
            "total_search_time": sum(self.search_times),
            "mean_search_time": (
                sum(self.search_times) / len(self.search_times) if self.search_times else 0
            ),
        }
