"""Anthropic API backend (supports Azure via AnthropicFoundry)."""

from __future__ import annotations

import os
import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic models via direct API or Azure (AnthropicFoundry)."""

    supports_vision = True

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key_env: str = "CLAUDE_API_KEY",
        base_url_env: str = "CLAUDE_ENDPOINT",
        max_tokens: int = 4096,
        temperature: float = 1.0,
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

        base_url = os.environ.get(base_url_env, "")

        try:
            from anthropic import AnthropicFoundry
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AnthropicFoundry(**client_kwargs)

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Call the Anthropic API with exponential backoff retry."""
        anthropic_messages = self._convert_messages(messages)

        # Extract system message if present
        system_text = None
        if anthropic_messages and anthropic_messages[0]["role"] == "system":
            system_text = anthropic_messages[0]["content"]
            anthropic_messages = anthropic_messages[1:]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if system_text:
                    kwargs["system"] = system_text

                response = self._client.messages.create(**kwargs)
                latency = time.time() - start

                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

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

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to Anthropic API format."""
        anthropic_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                anthropic_messages.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if block.get("type") == "text":
                        anthropic_content.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        source = block["source"]
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": source["media_type"],
                                    "data": source["data"],
                                },
                            }
                        )
                anthropic_messages.append({"role": msg["role"], "content": anthropic_content})
            else:
                anthropic_messages.append({"role": msg["role"], "content": str(content)})
        return anthropic_messages
