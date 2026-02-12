"""World model evaluation module.

This module provides AutumnBench-style world model evaluation as an additional
evaluation dimension for agents. Tests include:
- State prediction (masked frame prediction)
- Transfer learning (adaptation to modified mechanics)
- Change detection (detecting mid-episode dynamics changes)
- Counterfactual reasoning (alternate outcome prediction)
"""

from agentick.worldmodel.change_detection import ChangeDetectionEvaluator
from agentick.worldmodel.counterfactual import CounterfactualEvaluator
from agentick.worldmodel.evaluator import WorldModelEvaluator
from agentick.worldmodel.prediction import StatePredictionEvaluator
from agentick.worldmodel.transfer import TransferEvaluator

__all__ = [
    "StatePredictionEvaluator",
    "TransferEvaluator",
    "ChangeDetectionEvaluator",
    "CounterfactualEvaluator",
    "WorldModelEvaluator",
]
