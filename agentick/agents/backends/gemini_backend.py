"""Google Gemini API backend using the google-genai SDK."""

from __future__ import annotations

import base64
import os
import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class GeminiBackend(ModelBackend):
    """Backend for Google Gemini API."""

    supports_vision = True

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        max_tokens: int = 100,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.name = f"gemini/{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found. Set the {api_key_env} environment variable.")

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. Run: uv add google-genai"
            )

        self._client = genai.Client(api_key=api_key)
        self._types = genai.types

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Call the Gemini API with exponential backoff retry."""
        # Extract system instruction and user/assistant messages
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"] if isinstance(msg["content"], str) else ""
            else:
                chat_messages.append(msg)

        # Build config
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_text:
            config_kwargs["system_instruction"] = system_text

        config = self._types.GenerateContentConfig(**config_kwargs)

        # Convert to Gemini content format
        contents = self._convert_messages(chat_messages)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                latency = time.time() - start

                text = response.text if response.text else ""

                # Extract token usage
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ) or 0
                    output_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ) or 0

                return BackendResponse(
                    text=text.strip(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue

        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries} attempts: {last_error}"
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
        """Convert internal message format to Gemini content format."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]

            if isinstance(content, str):
                contents.append({"role": role, "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image":
                        source = block["source"]
                        image_bytes = base64.b64decode(source["data"])
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": source.get("media_type", "image/png"),
                                    "data": image_bytes,
                                }
                            }
                        )
                contents.append({"role": role, "parts": parts})
            else:
                contents.append({"role": role, "parts": [{"text": str(content)}]})

        return contents
