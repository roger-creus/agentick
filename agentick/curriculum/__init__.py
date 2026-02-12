"""Curriculum learning for progressive difficulty training."""

from agentick.curriculum.adaptive import AdaptiveCurriculum
from agentick.curriculum.manual import ManualCurriculum

__all__ = [
    "ManualCurriculum",
    "AdaptiveCurriculum",
]
