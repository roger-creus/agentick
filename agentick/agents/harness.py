"""Harness presets controlling how prompts are constructed for LLM/VLM agents."""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from agentick.agents.observation import format_text_observation, numpy_to_base64
from agentick.leaderboard.adapters.prompt_templates import (
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


class NonMarkovianZeroShot(HarnessPreset):
    """History-aware harness: maintains full conversation history.

    Args:
        max_history: Maximum number of steps to keep in history (default: 50).
            None means unlimited.
        diff_mode: If True, history entries store a compact diff of the
            observation against the previous step instead of the full text.
            The current (latest) observation is always sent in full.
    """

    def __init__(self, max_history: int | None = 50, diff_mode: bool = True):
        self.max_history = max_history
        self.diff_mode = diff_mode
        self._history: list[dict[str, Any]] = []
        self._prev_obs_text: str | None = None

    def build_messages(
        self,
        obs: Any,
        info: dict[str, Any],
        obs_modes: list[str],
    ) -> list[dict[str, Any]]:
        task_name = info.get("task_name", "unknown")
        messages = [_make_system_message(task_name)]
        messages.extend(self._history)
        messages.append({"role": "user", "content": _make_user_content(obs, info, obs_modes)})
        return messages

    def record_step(self, obs, info, action, response, reward):
        # Record the user turn and the action actually taken (not raw response,
        # since the parser may have extracted a different action than what the
        # raw text seems to say — the env state reflects the parsed action).
        obs_modes = info.get("_obs_modes", ["language"])
        user_content = _make_user_content(obs, info, obs_modes)

        # In diff mode, store a compact diff for history entries
        if self.diff_mode:
            current_text = _content_to_text(user_content)
            if self._prev_obs_text is not None:
                diff_text = _compute_diff(self._prev_obs_text, current_text)
                user_content = diff_text
            self._prev_obs_text = current_text

        self._history.append({"role": "user", "content": user_content})
        self._history.append({"role": "assistant", "content": f"ACTION: {action}"})

        # Truncate if needed
        if self.max_history is not None:
            # Each step = 2 messages (user + assistant)
            max_msgs = self.max_history * 2
            if len(self._history) > max_msgs:
                self._history = self._history[-max_msgs:]

    def reset(self):
        self._history.clear()
        self._prev_obs_text = None


def _content_to_text(content: str | list[dict[str, Any]]) -> str:
    """Extract plain text from user content (string or multimodal blocks)."""
    if isinstance(content, str):
        return content
    parts = [block["text"] for block in content if block.get("type") == "text"]
    return "\n".join(parts)


def _compute_diff(old_text: str, new_text: str) -> str:
    """Compact diff for LLM consumption."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, n=1)
    diff_text = "".join(diff)
    if not diff_text:
        return "[No changes from previous observation]"
    return f"[Changes from previous step]\n{diff_text}"


COT_SYSTEM_SUFFIX = """

IMPORTANT: Before choosing an action, reason step-by-step:
1. Describe what you see in the current observation
2. Identify the goal and your current progress toward it
3. Consider which action moves you closest to the goal
4. Output your final answer on the LAST line as: ACTION: <number>"""


class MarkovianReasoner(HarnessPreset):
    """Memoryless chain-of-thought harness: instructs step-by-step reasoning."""

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
    "non_markovian_zero_shot": NonMarkovianZeroShot,
    "markovian_reasoner": MarkovianReasoner,
}
