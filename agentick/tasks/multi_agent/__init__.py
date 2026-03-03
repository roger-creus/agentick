"""Multi-agent tasks."""

from agentick.tasks.multi_agent.chase_evade import ChaseEvadeTask
from agentick.tasks.multi_agent.cooperative_transport import CooperativeTransportTask
from agentick.tasks.multi_agent.emergent_strategy import EmergentStrategyTask
from agentick.tasks.multi_agent.herding import HerdingTask
from agentick.tasks.multi_agent.tag_hunt import TagHuntTask

__all__ = [
    "CooperativeTransportTask",
    "TagHuntTask",
    "ChaseEvadeTask",
    "HerdingTask",
    "EmergentStrategyTask",
]
