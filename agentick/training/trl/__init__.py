"""TRL-based SFT training for agentick."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "AgentickSFTTrainer":
        from agentick.training.trl.sft import AgentickSFTTrainer

        return AgentickSFTTrainer
    if name == "SFTAgent":
        from agentick.training.trl.sft import SFTAgent

        return SFTAgent
    raise AttributeError(f"module 'agentick.training.trl' has no attribute {name!r}")
