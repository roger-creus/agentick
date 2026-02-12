"""Meta-learning tasks - Few-shot learning and multi-task scenarios.

Tasks that test meta-learning and adaptation capabilities.
"""

from agentick.tasks.meta.few_shot_adaptation import FewShotAdaptationTask
from agentick.tasks.meta.task_interference import TaskInterferenceTask

__all__ = [
    "FewShotAdaptationTask",
    "TaskInterferenceTask",
]
