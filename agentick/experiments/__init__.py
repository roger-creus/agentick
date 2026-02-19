"""Experiment management system for reproducible research."""

from agentick.experiments.config import AgentConfig, ExperimentConfig, TrainingConfig
from agentick.experiments.registry import ExperimentRegistry
from agentick.experiments.reproduce import diff_experiments, reproduce_experiment
from agentick.experiments.runner import ExperimentResults, ExperimentRunner
from agentick.experiments.training_runner import TrainingBenchmarkRunner

__all__ = [
    "ExperimentConfig",
    "AgentConfig",
    "TrainingConfig",
    "ExperimentRunner",
    "ExperimentResults",
    "ExperimentRegistry",
    "TrainingBenchmarkRunner",
    "reproduce_experiment",
    "diff_experiments",
]


def __getattr__(name: str):
    """Lazy imports for agent classes to avoid circular imports."""
    if name in ("BaseAgent", "create_agent"):
        from agentick.agents import BaseAgent, create_agent

        _map = {"BaseAgent": BaseAgent, "create_agent": create_agent}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
