"""Harness presets controlling how prompts are constructed for LLM/VLM agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from agentick.agents.observation import format_text_observation, numpy_to_base64
from agentick.agents.prompt_templates import (
    SYSTEM_PROMPT,
    get_task_description,
)


class HarnessPreset(ABC):
    """Abstract base for prompt harness presets."""

    @abstractmethod
    def build_messages(
        self,
        obs: Any,
        info: dict[str, Any],
        obs_modes: list[str],
    ) -> list[dict[str, Any]]:
        """Build the message list to send to the model backend.

        Args:
            obs: Current environment observation.
            info: Info dict from env.step / env.reset.
            obs_modes: Observation modes the agent is configured for.

        Returns:
            List of message dicts (role + content).
        """
        ...

    @abstractmethod
    def record_step(
        self,
        obs: Any,
        info: dict[str, Any],
        action: int,
        response: str,
        reward: float,
    ) -> None:
        """Record a step for history-based presets."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state between episodes."""
        ...


def _make_system_message(task_name: str) -> dict[str, Any]:
    """Build system message from task name."""
    task_description = get_task_description(task_name)
    return {
        "role": "system",
        "content": SYSTEM_PROMPT.format(task_description=task_description),
    }


def _make_user_content(
    obs: Any,
    info: dict[str, Any],
    obs_modes: list[str],
) -> str | list[dict[str, Any]]:
    """Build user message content, possibly multimodal."""
    parts: list[dict[str, Any]] = []

    for mode in obs_modes:
        if mode in ("language", "ascii", "language_structured", "state_dict"):
            text = format_text_observation(obs, info, mode)
            parts.append({"type": "text", "text": text})
        elif mode == "rgb_array":
            if isinstance(obs, np.ndarray):
                b64 = numpy_to_base64(obs)
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    }
                )
                # Add minimal text context alongside the image
                task_name = info.get("task_name", "unknown")
                step = info.get("step", 0)
                parts.append(
                    {
                        "type": "text",
                        "text": (
                            f"Task: {task_name}\nStep: {step}\n"
                            "Select the best action (respond with just the action number):"
                        ),
                    }
                )

    # If only one text part, return plain string for simpler API compatibility
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]

    return parts


class MarkovianZeroShot(HarnessPreset):
    """Memoryless zero-shot harness: system prompt + current obs only."""

    def build_messages(
        self,
        obs: Any,
        info: dict[str, Any],
        obs_modes: list[str],
    ) -> list[dict[str, Any]]:
        task_name = info.get("task_name", "unknown")
        return [
            _make_system_message(task_name),
            {"role": "user", "content": _make_user_content(obs, info, obs_modes)},
        ]

    def record_step(self, obs, info, action, response, reward):
        pass  # No history

    def reset(self):
        pass  # No state


COT_SYSTEM_SUFFIX = """

IMPORTANT: Before choosing an action, reason step-by-step but be CONCISE (2-4 sentences max):
1. What do you observe? What is your goal?
2. Which action best advances you toward the goal?
3. Output your final answer on the LAST line as: ACTION: <number>"""


class MarkovianReasoner(HarnessPreset):
    """Memoryless chain-of-thought harness: instructs step-by-step reasoning."""

    def __init__(self, max_tokens: int | None = None):
        self.max_tokens = max_tokens

    def build_messages(
        self,
        obs: Any,
        info: dict[str, Any],
        obs_modes: list[str],
    ) -> list[dict[str, Any]]:
        task_name = info.get("task_name", "unknown")
        task_description = get_task_description(task_name)
        system_text = SYSTEM_PROMPT.format(task_description=task_description)
        # Replace the final instruction with CoT instruction
        system_text = system_text.replace(
            "Respond with ONLY the action number, nothing else.",
            "",
        )
        system_text = system_text.rstrip() + COT_SYSTEM_SUFFIX
        if self.max_tokens:
            system_text += f"\nYou have a budget of {self.max_tokens} tokens for your response."

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": _make_user_content(obs, info, obs_modes)},
        ]

    def record_step(self, obs, info, action, response, reward):
        pass  # No history

    def reset(self):
        pass  # No state


HARNESS_REGISTRY: dict[str, type[HarnessPreset]] = {
    "markovian_zero_shot": MarkovianZeroShot,
    "markovian_reasoner": MarkovianReasoner,
}
