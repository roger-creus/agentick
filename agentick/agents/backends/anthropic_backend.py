"""Anthropic API backend."""

from __future__ import annotations

import os
import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic Claude API."""

    supports_vision = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 100,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.name = f"anthropic/{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found. Set the {api_key_env} environment variable.")

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: uv sync --extra llm")

        self._client = Anthropic(api_key=api_key)

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Call the Anthropic API with exponential backoff retry."""
        # Extract system message (Anthropic uses a separate system param)
        system_text = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"] if isinstance(msg["content"], str) else ""
            else:
                user_messages.append(self._convert_message(msg))

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": user_messages,
                }
                if system_text:
                    kwargs["system"] = system_text

                response = self._client.messages.create(**kwargs)
                latency = time.time() - start

                text = response.content[0].text if response.content else ""
                return BackendResponse(
                    text=text.strip(),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_seconds=latency,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue

        raise RuntimeError(
            f"Anthropic API call failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_batch(
        self, messages_list: list[list[dict[str, Any]]]
    ) -> list[BackendResponse]:
        """Batch generate responses using concurrent threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = len(messages_list)
        if n <= 1:
            return [self.generate(msgs) for msgs in messages_list]

        results: list[BackendResponse | None] = [None] * n
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(self.generate, msgs): i
                for i, msgs in enumerate(messages_list)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results  # type: ignore[return-value]

    def _convert_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Convert internal message format to Anthropic API format."""
        content = msg["content"]
        if isinstance(content, str):
            return {"role": msg["role"], "content": content}
        elif isinstance(content, list):
            # Already in Anthropic's multimodal format (image + text blocks)
            return {"role": msg["role"], "content": content}
        else:
            return {"role": msg["role"], "content": str(content)}
