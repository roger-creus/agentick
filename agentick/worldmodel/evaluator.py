"""Main world model evaluator that combines all test types.

Provides unified interface for comprehensive world model evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentick.worldmodel.change_detection import ChangeDetectionEvaluator
from agentick.worldmodel.counterfactual import CounterfactualEvaluator
from agentick.worldmodel.prediction import StatePredictionEvaluator
from agentick.worldmodel.transfer import TransferEvaluator


@dataclass
class WorldModelScore:
    """Comprehensive world model evaluation score."""

    prediction_accuracy: float
    transfer_score: float
    change_detection_score: float
    counterfactual_accuracy: float
    overall_score: float


class WorldModelEvaluator:
    """Unified world model evaluator."""

    def __init__(
        self,
        env_factory: Callable,
        modified_env_factory: Callable | None = None,
        change_env_factory: Callable | None = None,
    ):
        """
        Initialize world model evaluator.

        Args:
            env_factory: Factory for base environment
            modified_env_factory: Factory for modified environment (for transfer tests)
            change_env_factory: Factory for changeable environment (for change detection)
        """
        self.env_factory = env_factory
        self.modified_env_factory = modified_env_factory or env_factory
        self.change_env_factory = change_env_factory or env_factory

        # Initialize sub-evaluators
        self.prediction_eval = StatePredictionEvaluator(env_factory, n_tests=10)
        self.transfer_eval = TransferEvaluator(env_factory, self.modified_env_factory)
        self.change_detection_eval = ChangeDetectionEvaluator(env_factory, self.change_env_factory)
        self.counterfactual_eval = CounterfactualEvaluator(env_factory, n_tests=10)

    def evaluate_full(
        self,
        agent: Any,
        seed: int | None = None,
    ) -> WorldModelScore:
        """
        Run full world model evaluation suite.

        Args:
            agent: Agent to evaluate
            seed: Random seed

        Returns:
            WorldModelScore with all metrics
        """
        # State prediction
        pred_result = self.prediction_eval.evaluate_masked_prediction(agent, seed=seed)
        prediction_accuracy = pred_result.accuracy

        # Transfer learning
        transfer_result = self.transfer_eval.evaluate_transfer(agent, seed=seed)
        # Score: 1.0 if perfect transfer, 0.0 if no transfer
        transfer_score = 1.0 - min(
            abs(transfer_result.final_gap) / max(abs(transfer_result.baseline_performance), 1e-6),
            1.0,
        )

        # Change detection
        change_result = self.change_detection_eval.evaluate_change_detection(agent, seed=seed)
        # Combined score from accuracy and false positive rate
        change_detection_score = change_result.detection_accuracy * (
            1.0 - change_result.false_positive_rate
        )

        # Counterfactual reasoning
        counterfactual_result = self.counterfactual_eval.evaluate_counterfactual_prediction(
            agent, seed=seed
        )
        counterfactual_accuracy = counterfactual_result.accuracy

        # Overall score (weighted average)
        overall_score = (
            0.3 * prediction_accuracy
            + 0.3 * transfer_score
            + 0.2 * change_detection_score
            + 0.2 * counterfactual_accuracy
        )

        return WorldModelScore(
            prediction_accuracy=prediction_accuracy,
            transfer_score=transfer_score,
            change_detection_score=change_detection_score,
            counterfactual_accuracy=counterfactual_accuracy,
            overall_score=overall_score,
        )

    def evaluate_prediction_only(self, agent: Any, seed: int | None = None) -> dict[str, float]:
        """Run only prediction tests."""
        masked_result = self.prediction_eval.evaluate_masked_prediction(agent, seed=seed)
        free_result = self.prediction_eval.evaluate_free_form_prediction(agent, seed=seed)

        return {
            "masked_prediction_accuracy": masked_result.accuracy,
            "free_form_prediction_accuracy": free_result.accuracy,
            "combined_prediction_accuracy": (masked_result.accuracy + free_result.accuracy) / 2,
        }

    def evaluate_transfer_only(self, agent: Any, seed: int | None = None) -> dict[str, float]:
        """Run only transfer tests."""
        result = self.transfer_eval.evaluate_transfer(agent, seed=seed)

        return {
            "baseline_performance": result.baseline_performance,
            "transfer_performance": result.transfer_performance,
            "adaptation_speed": result.adaptation_speed,
            "final_gap": result.final_gap,
            "transfer_score": 1.0
            - min(abs(result.final_gap) / max(abs(result.baseline_performance), 1e-6), 1.0),
        }

    def evaluate_change_detection_only(
        self, agent: Any, seed: int | None = None
    ) -> dict[str, float]:
        """Run only change detection tests."""
        result = self.change_detection_eval.evaluate_change_detection(agent, seed=seed)

        return {
            "detection_accuracy": result.detection_accuracy,
            "mean_detection_latency": result.mean_detection_latency,
            "false_positive_rate": result.false_positive_rate,
            "combined_score": result.detection_accuracy * (1.0 - result.false_positive_rate),
        }

    def evaluate_counterfactual_only(self, agent: Any, seed: int | None = None) -> dict[str, float]:
        """Run only counterfactual tests."""
        result = self.counterfactual_eval.evaluate_counterfactual_prediction(agent, seed=seed)

        return {
            "counterfactual_accuracy": result.accuracy,
            "counterfactual_mean_error": result.mean_error,
        }

    def generate_worldmodel_report(
        self,
        agent: Any,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive world model evaluation report.

        Args:
            agent: Agent to evaluate
            seed: Random seed

        Returns:
            Dict with full evaluation results and interpretation
        """
        full_score = self.evaluate_full(agent, seed=seed)

        # Generate interpretation
        interpretation = []
        if full_score.prediction_accuracy > 0.8:
            interpretation.append("Strong state prediction ability")
        elif full_score.prediction_accuracy > 0.5:
            interpretation.append("Moderate state prediction ability")
        else:
            interpretation.append("Weak state prediction ability")

        if full_score.transfer_score > 0.7:
            interpretation.append("Good transfer to modified environments")
        elif full_score.transfer_score > 0.4:
            interpretation.append("Partial transfer capability")
        else:
            interpretation.append("Poor transfer/generalization")

        if full_score.change_detection_score > 0.7:
            interpretation.append("Reliable change detection")
        else:
            interpretation.append("Inconsistent change detection")

        if full_score.counterfactual_accuracy > 0.6:
            interpretation.append("Sound counterfactual reasoning")
        else:
            interpretation.append("Limited counterfactual reasoning")

        return {
            "scores": {
                "prediction_accuracy": full_score.prediction_accuracy,
                "transfer_score": full_score.transfer_score,
                "change_detection_score": full_score.change_detection_score,
                "counterfactual_accuracy": full_score.counterfactual_accuracy,
                "overall_score": full_score.overall_score,
            },
            "interpretation": interpretation,
            "grade": self._assign_grade(full_score.overall_score),
        }

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade to world model score."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        else:
            return "D" if score >= 0.4 else "F"
