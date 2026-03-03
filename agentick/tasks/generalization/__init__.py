"""Generalization tasks."""

from agentick.tasks.generalization.distribution_shift import DistributionShiftTask
from agentick.tasks.generalization.few_shot_adaptation import FewShotAdaptationTask
from agentick.tasks.generalization.noisy_observation import NoisyObservationTask

__all__ = [
    "FewShotAdaptationTask",
    "DistributionShiftTask",
    "NoisyObservationTask",
]
