"""Reasoning tasks."""

from agentick.tasks.reasoning.deceptive_reward import DeceptiveRewardTask
from agentick.tasks.reasoning.graph_coloring import GraphColoringTask
from agentick.tasks.reasoning.lights_out import LightsOutTask
from agentick.tasks.reasoning.program_synthesis import ProgramSynthesisTask
from agentick.tasks.reasoning.rule_induction import RuleInductionTask
from agentick.tasks.reasoning.switch_circuit import SwitchCircuitTask
from agentick.tasks.reasoning.symbol_matching import SymbolMatchingTask
from agentick.tasks.reasoning.task_interference import TaskInterferenceTask

__all__ = [
    "SwitchCircuitTask",
    "RuleInductionTask",
    "SymbolMatchingTask",
    "LightsOutTask",
    "GraphColoringTask",
    "ProgramSynthesisTask",
    "TaskInterferenceTask",
    "DeceptiveRewardTask",
]
