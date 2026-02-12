"""Reasoning tasks."""

from agentick.tasks.reasoning.causal_chain import CausalChainTask
from agentick.tasks.reasoning.rule_induction import RuleInductionTask
from agentick.tasks.reasoning.sokoban_push import SokobanPushTask
from agentick.tasks.reasoning.switch_circuit import SwitchCircuitTask
from agentick.tasks.reasoning.symbol_matching import SymbolMatchingTask

__all__ = [
    "SokobanPushTask",
    "SwitchCircuitTask",
    "SymbolMatchingTask",
    "CausalChainTask",
    "RuleInductionTask",
]
