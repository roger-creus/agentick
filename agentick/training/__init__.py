"""Training utilities for agent training.

Framework-agnostic helpers for training, logging, and callbacks.
SFT trainer is imported lazily to avoid heavy dependencies.
"""

from agentick.training.callbacks import CheckpointCallback, EvalCallback

__all__ = [
    "EvalCallback",
    "CheckpointCallback",
]


def __getattr__(name: str):
    if name == "AgentickSFTTrainer":
        from agentick.training.trl.sft import AgentickSFTTrainer

        return AgentickSFTTrainer
    if name == "SFTAgent":
        from agentick.training.trl.sft import SFTAgent

        return SFTAgent
    raise AttributeError(f"module 'agentick.training' has no attribute {name!r}")
