"""Training utilities for agent training.

Framework-agnostic helpers for training, logging, and callbacks.
SFT and BC trainers are imported lazily to avoid heavy dependencies.
"""

from agentick.training.callbacks import CheckpointCallback, CurriculumCallback, EvalCallback
from agentick.training.logger import MultiBackendLogger

__all__ = [
    "EvalCallback",
    "CurriculumCallback",
    "CheckpointCallback",
    "MultiBackendLogger",
]


def __getattr__(name: str):
    if name == "AgentickSFTTrainer":
        from agentick.training.trl.sft import AgentickSFTTrainer

        return AgentickSFTTrainer
    if name == "SFTAgent":
        from agentick.training.trl.sft import SFTAgent

        return SFTAgent
    if name == "BehaviorCloningTrainer":
        from agentick.training.behavior_cloning import BehaviorCloningTrainer

        return BehaviorCloningTrainer
    raise AttributeError(f"module 'agentick.training' has no attribute {name!r}")
