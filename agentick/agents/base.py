"""BaseAgent composing a ModelBackend with a HarnessPreset."""

from __future__ import annotations

import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend
from agentick.agents.harness import HarnessPreset
from agentick.agents.prompt_templates import parse_action_from_text
from agentick.core.types import ActionType


class BaseAgent:
    """First-class LLM/VLM agent for Agentick experiments.

    Composes a ``ModelBackend`` (how to call the model) with a
    ``HarnessPreset`` (how to build prompts / manage context).
    Conforms to ``AgentProtocol``.
    """

    def __init__(
        self,
        backend: ModelBackend,
        harness: HarnessPreset,
        observation_modes: list[str] | None = None,
        agent_name: str | None = None,
    ):
        self.backend = backend
        self.harness = harness
        self.observation_modes = observation_modes or ["language"]
        self._name = agent_name or f"{backend.name}_{harness.__class__.__name__}"

        # Tracking
        self.call_log: list[dict[str, Any]] = []
        self._system_prompt: str = ""
        self.total_tokens: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.total_latency: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        """Reset per-episode state (harness history, call log)."""
        self.harness.reset()
        self.call_log = []

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """Select an action given observation and info dict.

        Delegates to the harness for message construction, the backend
        for generation, and ``parse_action_from_text`` for response parsing.
        """
        # Stash obs_modes in info so NonMarkovian harness can reuse them
        info_with_modes = {**info, "_obs_modes": self.observation_modes}

        # Build messages
        messages = self.harness.build_messages(observation, info_with_modes, self.observation_modes)

        # Extract observation text and image flag from last user message
        user_content = messages[-1]["content"]
        if isinstance(user_content, list):
            obs_text = "\n".join(b["text"] for b in user_content if b.get("type") == "text")
            has_image = any(b.get("type") == "image" for b in user_content)
        else:
            obs_text = user_content
            has_image = False

        # Store system prompt once on first call
        if not self.call_log:
            self._system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""

        # Generate
        start = time.time()
        response: BackendResponse = self.backend.generate(messages)
        latency = time.time() - start

        # Parse action — use integer action IDs (0-8) matching ActionType enum.
        action = parse_action_from_text(response.text, list(range(6)))

        # Extract reasoning for CoT harness
        reasoning = None
        if "ACTION:" in response.text:
            parts = response.text.rsplit("ACTION:", 1)
            if parts[0].strip():
                reasoning = parts[0].strip()

        # Action name from ActionType enum
        try:
            action_name = ActionType(action).name
        except ValueError:
            action_name = f"ACTION_{action}"

        # Record
        self.total_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_tokens += response.input_tokens + response.output_tokens
        self.total_latency += latency

        self.call_log.append(
            {
                "step": info.get("step", 0),
                "observation": obs_text,
                "has_image": has_image,
                "response": response.text,
                "reasoning": reasoning,
                "parsed_action": action,
                "action_name": action_name,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency": latency,
            }
        )

        # Let the harness record the step (for history-based presets)
        reward = info.get("reward", 0.0)
        self.harness.record_step(observation, info_with_modes, action, response.text, reward)

        return action

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate agent statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency": self.total_latency,
            "mean_latency": (
                self.total_latency / self.total_calls if self.total_calls > 0 else 0.0
            ),
        }
