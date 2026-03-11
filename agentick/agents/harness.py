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
    """History-aware harness: maintains conversation history with token-based sliding window.

    Uses a sliding window that discards oldest steps when approaching the context
    limit. When diff_mode is enabled, diffs are recomputed at build time so the
    first message in the window always shows a full observation.

    Args:
        max_context_tokens: Maximum context budget in tokens (default: 32768).
            History is trimmed oldest-first to stay within this budget.
        diff_mode: If True, history observations are sent as compact diffs
            against the previous step. The first observation in the window
            and the current (latest) observation are always sent in full.
        context_safety_margin: Fraction of max_context_tokens to actually use
            (default: 0.80). Reserves headroom for chat-template overhead,
            token-estimation error, and response generation so that the
            prompt never exceeds the backend's max_model_len.
    """

    def __init__(
        self,
        max_context_tokens: int = 32768,
        diff_mode: bool = True,
        context_safety_margin: float = 0.80,
    ):
        self.max_context_tokens = max_context_tokens
        self.diff_mode = diff_mode
        self.context_safety_margin = context_safety_margin
        # Store raw (obs_content, assistant_reply) per step for recomputing diffs
        self._steps: list[tuple[str | list[dict[str, Any]], str]] = []

    def build_messages(
        self,
        obs: Any,
        info: dict[str, Any],
        obs_modes: list[str],
    ) -> list[dict[str, Any]]:
        task_name = info.get("task_name", "unknown")
        system_msg = _make_system_message(task_name)
        current_content = _make_user_content(obs, info, obs_modes)
        current_msg = {"role": "user", "content": current_content}

        # Token budget: apply safety margin, then subtract system + current obs.
        # The safety margin accounts for chat-template tokens (BOS/EOS/role
        # markers), token-estimation error (chars/4 can undercount), and
        # response tokens that the backend needs to generate.
        effective_limit = int(self.max_context_tokens * self.context_safety_margin)
        system_tokens = _estimate_tokens_msg(system_msg)
        current_tokens = _estimate_tokens_msg(current_msg)
        budget = max(effective_limit - system_tokens - current_tokens, 0)

        # Build history within budget, trimming oldest steps first
        history = _build_sliding_history(self._steps, budget, self.diff_mode)

        messages = [system_msg]
        messages.extend(history)
        messages.append(current_msg)
        return messages

    def record_step(self, obs, info, action, response, reward):
        obs_modes = info.get("_obs_modes", ["language"])
        user_content = _make_user_content(obs, info, obs_modes)
        self._steps.append((user_content, f"ACTION: {action}"))

    def reset(self):
        self._steps.clear()


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


def _estimate_tokens_content(content: str | list[dict[str, Any]]) -> int:
    """Estimate token count for message content (chars / 3 heuristic).

    Uses chars / 3 (rather than the common chars / 4 rule-of-thumb) to be
    conservative — structured text with special characters, newlines, and
    short tokens tends to tokenize at a higher ratio than plain English.
    Images are estimated at 1000 tokens each.
    """
    if isinstance(content, str):
        return len(content) // 3 + 1
    total = 0
    for block in content:
        if block.get("type") == "text":
            total += len(block.get("text", "")) // 3 + 1
        elif block.get("type") == "image":
            total += 1000  # conservative estimate for a small gridworld image
    return max(total, 1)


def _estimate_tokens_msg(msg: dict[str, Any]) -> int:
    """Estimate token count for a single message (content + role overhead)."""
    return _estimate_tokens_content(msg["content"]) + 4  # role/formatting overhead


def _build_sliding_history(
    steps: list[tuple[str | list[dict[str, Any]], str]],
    budget: int,
    diff_mode: bool,
) -> list[dict[str, Any]]:
    """Build history messages from stored steps, fitting within a token budget.

    Trims oldest steps first. When diff_mode is True, recomputes diffs so the
    first observation in the window is always full (not a dangling diff).

    Args:
        steps: List of (user_content, assistant_reply) tuples.
        budget: Maximum tokens allowed for the history portion.
        diff_mode: Whether to convert history observations to diffs.

    Returns:
        List of message dicts ready to insert between system and current user msg.
    """
    if not steps:
        return []

    # Try including all steps, trim from the front until we fit
    start = 0
    while start < len(steps):
        messages = _render_history_slice(steps, start, diff_mode)
        total = sum(_estimate_tokens_msg(m) for m in messages)
        if total <= budget:
            return messages
        start += 1

    # Nothing fits — return empty history
    return []


def _render_history_slice(
    steps: list[tuple[str | list[dict[str, Any]], str]],
    start: int,
    diff_mode: bool,
) -> list[dict[str, Any]]:
    """Render a slice of steps into message dicts, recomputing diffs from start."""
    messages: list[dict[str, Any]] = []
    prev_text: str | None = None

    for i in range(start, len(steps)):
        user_content, assistant_reply = steps[i]

        if diff_mode and i > start:
            # Compute diff against previous step's full observation
            current_text = _content_to_text(user_content)
            if prev_text is not None:
                display_content: str | list[dict[str, Any]] = _compute_diff(
                    prev_text, current_text
                )
            else:
                display_content = user_content
            prev_text = current_text
        else:
            # First entry in window or diff_mode off: send full observation
            display_content = user_content
            if diff_mode:
                prev_text = _content_to_text(user_content)

        messages.append({"role": "user", "content": display_content})
        messages.append({"role": "assistant", "content": assistant_reply})

    return messages


COT_SYSTEM_SUFFIX = """

IMPORTANT: Before choosing an action, reason step-by-step but be CONCISE (2-4 sentences max):
1. What do you observe? What is your goal?
2. Which action best advances you toward the goal?
3. Output your final answer on the LAST line as: ACTION: <number>"""

# Even more concise for nonmarkovian where history accumulates
COT_SYSTEM_SUFFIX_COMPACT = """

IMPORTANT: Think briefly then act. Keep reasoning to 1-2 sentences.
- State your key observation and chosen action rationale in minimal words.
- Output your final answer on the LAST line as: ACTION: <number>"""


def _compact_cot_response(response: str, max_chars: int = 400) -> str:
    """Compact a CoT response for history storage.

    Keeps the final ACTION line and truncates long reasoning to save context.
    """
    lines = response.strip().splitlines()

    # Find the ACTION line (usually last)
    action_line = ""
    reasoning_lines = []
    for line in lines:
        if line.strip().upper().startswith("ACTION:"):
            action_line = line.strip()
        else:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    if len(reasoning) > max_chars:
        reasoning = reasoning[:max_chars].rsplit(" ", 1)[0] + "..."

    if action_line:
        return f"{reasoning}\n{action_line}" if reasoning else action_line
    return reasoning if reasoning else response[:max_chars]


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


class NonMarkovianReasoner(HarnessPreset):
    """History-aware chain-of-thought harness: CoT reasoning with conversation history.

    Combines concise step-by-step reasoning with the token-based sliding window
    from NonMarkovianZeroShot. CoT responses are compacted before storing in
    history to prevent long reasoning traces from flooding the context.

    Args:
        max_context_tokens: Maximum context budget in tokens (default: 32768).
            History is trimmed oldest-first to stay within this budget.
        diff_mode: If True, history observations are sent as compact diffs
            against the previous step. The first observation in the window
            and the current (latest) observation are always sent in full.
        max_response_chars: Maximum characters to keep per CoT response in
            history (default: 400). Longer responses are truncated, keeping
            the ACTION line.
        context_safety_margin: Fraction of max_context_tokens to actually use
            (default: 0.80). Reserves headroom for chat-template overhead,
            token-estimation error, and response generation.
    """

    def __init__(
        self,
        max_context_tokens: int = 32768,
        diff_mode: bool = True,
        max_response_chars: int = 400,
        context_safety_margin: float = 0.80,
        max_tokens: int | None = None,
    ):
        self.max_context_tokens = max_context_tokens
        self.diff_mode = diff_mode
        self.max_response_chars = max_response_chars
        self.context_safety_margin = context_safety_margin
        self.max_tokens = max_tokens
        self._steps: list[tuple[str | list[dict[str, Any]], str]] = []

    def build_messages(
        self,
        obs: Any,
        info: dict[str, Any],
        obs_modes: list[str],
    ) -> list[dict[str, Any]]:
        task_name = info.get("task_name", "unknown")
        task_description = get_task_description(task_name)
        system_text = SYSTEM_PROMPT.format(task_description=task_description)
        system_text = system_text.replace(
            "Respond with ONLY the action number, nothing else.",
            "",
        )
        system_text = system_text.rstrip() + COT_SYSTEM_SUFFIX_COMPACT
        if self.max_tokens:
            system_text += f"\nYou have a budget of {self.max_tokens} tokens for your response."
        system_msg = {"role": "system", "content": system_text}

        current_content = _make_user_content(obs, info, obs_modes)
        current_msg = {"role": "user", "content": current_content}

        effective_limit = int(self.max_context_tokens * self.context_safety_margin)
        system_tokens = _estimate_tokens_msg(system_msg)
        current_tokens = _estimate_tokens_msg(current_msg)
        budget = max(effective_limit - system_tokens - current_tokens, 0)

        history = _build_sliding_history(self._steps, budget, self.diff_mode)

        messages = [system_msg]
        messages.extend(history)
        messages.append(current_msg)
        return messages

    def record_step(self, obs, info, action, response, reward):
        obs_modes = info.get("_obs_modes", ["language"])
        user_content = _make_user_content(obs, info, obs_modes)
        # Compact the CoT response to prevent long traces from flooding context
        compacted = _compact_cot_response(response, self.max_response_chars)
        self._steps.append((user_content, compacted))

    def reset(self):
        self._steps.clear()


HARNESS_REGISTRY: dict[str, type[HarnessPreset]] = {
    "markovian_zero_shot": MarkovianZeroShot,
    "non_markovian_zero_shot": NonMarkovianZeroShot,
    "markovian_reasoner": MarkovianReasoner,
    "non_markovian_reasoner": NonMarkovianReasoner,
}
