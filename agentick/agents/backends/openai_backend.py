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
        # Some models (o1, o3, gpt-5-mini) reject temperature — auto-detected on first call
        self._supports_temperature = True

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Call the OpenAI API with exponential backoff retry."""
        oai_messages = self._convert_messages(messages)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": oai_messages,
                    "max_completion_tokens": self.max_tokens,
                }
                if self._supports_temperature:
                    kwargs["temperature"] = self.temperature

                response = self._client.chat.completions.create(**kwargs)
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
                # Auto-drop temperature for models that don't support it
                if self._supports_temperature and "temperature" in str(e):
                    self._supports_temperature = False
                    print(f"  Note: {self.model} does not support temperature, dropping it")
                    continue
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue

        raise RuntimeError(
            f"OpenAI API call failed after {self.max_retries} attempts: {last_error}"
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
