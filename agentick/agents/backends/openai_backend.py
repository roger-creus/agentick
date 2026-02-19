"""OpenAI API backend."""

from __future__ import annotations

import os
import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class OpenAIBackend(ModelBackend):
    """Backend for OpenAI-compatible APIs (GPT-4o, etc.)."""

    supports_vision = True

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 100,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.name = f"openai/{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found. Set the {api_key_env} environment variable.")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: uv sync --extra llm")

        self._client = OpenAI(api_key=api_key)

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Call the OpenAI API with exponential backoff retry."""
        # Convert our message format to OpenAI format
        oai_messages = self._convert_messages(messages)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=oai_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                latency = time.time() - start

                text = response.choices[0].message.content or ""
                usage = response.usage
                return BackendResponse(
                    text=text.strip(),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_seconds=latency,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue

        raise RuntimeError(
            f"OpenAI API call failed after {self.max_retries} attempts: {last_error}"
        )

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to OpenAI API format."""
        oai_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                oai_messages.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # Multimodal: convert our image blocks to OpenAI format
                oai_content = []
                for block in content:
                    if block.get("type") == "text":
                        oai_content.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        source = block["source"]
                        oai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{source['media_type']};base64,{source['data']}",
                                },
                            }
                        )
                oai_messages.append({"role": msg["role"], "content": oai_content})
            else:
                oai_messages.append({"role": msg["role"], "content": str(content)})
        return oai_messages
