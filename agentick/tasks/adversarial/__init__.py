"""Adversarial tasks - Robustness and distribution shift testing.

Tasks that test agent robustness to noise, deception, and distribution shifts.
"""

from agentick.tasks.adversarial.deceptive_reward import DeceptiveRewardTask
from agentick.tasks.adversarial.distribution_shift import DistributionShiftTask
from agentick.tasks.adversarial.noisy_observation import NoisyObservationTask

__all__ = [
    "NoisyObservationTask",
    "DeceptiveRewardTask",
    "DistributionShiftTask",
]
