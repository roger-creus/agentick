"""Control tasks."""

from agentick.tasks.control.chase_evade import ChaseEvadeTask
from agentick.tasks.control.herding import HerdingTask
from agentick.tasks.control.precise_navigation import PreciseNavigationTask
from agentick.tasks.control.timing_challenge import TimingChallengeTask

__all__ = [
    "PreciseNavigationTask",
    "TimingChallengeTask",
    "ChaseEvadeTask",
    "HerdingTask",
]
