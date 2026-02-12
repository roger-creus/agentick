"""Compositional tasks - Hierarchical and compositional reasoning.

Tasks that test composability and hierarchical planning.
"""

from agentick.tasks.compositional.instruction_following import InstructionFollowingTask
from agentick.tasks.compositional.program_synthesis import ProgramSynthesisTask
from agentick.tasks.compositional.recursive_rooms import RecursiveRoomsTask

__all__ = [
    "RecursiveRoomsTask",
    "ProgramSynthesisTask",
    "InstructionFollowingTask",
]
