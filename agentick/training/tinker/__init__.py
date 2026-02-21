"""Tinker-based SFT and RL training for agentick."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "TinkerSFTTrainer":
        from agentick.training.tinker.sft import TinkerSFTTrainer

        return TinkerSFTTrainer
    if name == "TinkerSFTAgent":
        from agentick.training.tinker.sft import TinkerSFTAgent

        return TinkerSFTAgent
    if name == "TinkerRLTrainer":
        from agentick.training.tinker.rl import TinkerRLTrainer

        return TinkerRLTrainer
    raise AttributeError(f"module 'agentick.training.tinker' has no attribute {name!r}")
