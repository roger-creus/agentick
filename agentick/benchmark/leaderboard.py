"""Leaderboard functionality."""

import json
from typing import Any


class Leaderboard:
    """Save/load results, print formatted tables."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results = {}

    def add_result(self, agent_name: str, results: dict[str, Any]):
        """Add agent results to leaderboard."""
        self.results[agent_name] = results

    def save(self, filename: str):
        """Save results to JSON."""
        with open(f"{self.results_dir}/{filename}", "w") as f:
            json.dump(self.results, f, indent=2)

    def load(self, filename: str):
        """Load results from JSON."""
        with open(f"{self.results_dir}/{filename}") as f:
            self.results = json.load(f)

    def print_table(self):
        """Print formatted leaderboard table."""
        print("\n=== Agentick Leaderboard ===\n")
        print(f"{'Agent':<20} {'Mean Return':>12} {'Success Rate':>13}")
        print("-" * 47)

        for agent_name, results in self.results.items():
            mean_return = results.get("mean_return", 0.0)
            success_rate = results.get("success_rate", 0.0)
            print(f"{agent_name:<20} {mean_return:>12.3f} {success_rate:>12.1%}")
