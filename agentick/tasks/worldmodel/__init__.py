"""World model tasks - Hidden mechanics and rule discovery.

Tasks that test agent's ability to discover and adapt to hidden rules.
"""

from agentick.tasks.worldmodel.environment_shift import EnvironmentShiftTask
from agentick.tasks.worldmodel.physics_discovery import PhysicsDiscoveryTask
from agentick.tasks.worldmodel.rule_discovery_navigation import RuleDiscoveryNavigationTask

__all__ = [
    "RuleDiscoveryNavigationTask",
    "PhysicsDiscoveryTask",
    "EnvironmentShiftTask",
]
