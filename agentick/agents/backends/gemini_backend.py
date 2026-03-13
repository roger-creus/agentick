"""Google Gemini API backend using the google-genai SDK."""

from __future__ import annotations

import base64
import collections
import os
import random
import threading
import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class _RateLimiter:
    """Thread-safe sliding-window rate limiter for RPM and TPM."""

    def __init__(self, rpm: int = 0, tpm: int = 0):
        self.rpm = rpm  # 0 = unlimited
        self.tpm = tpm  # 0 = unlimited
        self._request_times: collections.deque[float] = collections.deque()
        self._token_log: collections.deque[tuple[float, int]] = collections.deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rpm > 0 or self.tpm > 0

    def acquire(self) -> None:
        """Block until a request is allowed under RPM/TPM limits."""
        if not self.enabled:
            return

        while True:
            with self._lock:
                now = time.time()
                cutoff = now - 60.0

                # Prune old entries
                while self._request_times and self._request_times[0] < cutoff:
                    self._request_times.popleft()
                while self._token_log and self._token_log[0][0] < cutoff:
                    self._token_log.popleft()

                # Check RPM
                rpm_ok = self.rpm <= 0 or len(self._request_times) < self.rpm

                # Check TPM
                tpm_ok = True
                if self.tpm > 0:
                    current_tpm = sum(n for _, n in self._token_log)
                    tpm_ok = current_tpm < self.tpm

                if rpm_ok and tpm_ok:
                    self._request_times.append(now)
                    return

                # Calculate how long to sleep
                if not rpm_ok and self._request_times:
                    sleep_time = self._request_times[0] - cutoff + 0.05
                elif not tpm_ok and self._token_log:
                    sleep_time = self._token_log[0][0] - cutoff + 0.05
                else:
                    sleep_time = 1.0

            time.sleep(max(sleep_time, 0.05))

    def record_tokens(self, tokens: int) -> None:
        """Record token usage after a successful request."""
        if self.tpm <= 0:
            return
        with self._lock:
            self._token_log.append((time.time(), tokens))


class GeminiBackend(ModelBackend):
    """Backend for Google Gemini API."""

    supports_vision = True

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        max_tokens: int = 100,
        temperature: float = 0.0,
        max_retries: int = 5,
        rpm_limit: int = 0,
        tpm_limit: int = 0,
        max_concurrent_requests: int = 10,
    ):
        self.name = f"gemini/{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests

        self._rate_limiter = _RateLimiter(rpm=rpm_limit, tpm=tpm_limit)
        self._semaphore = threading.Semaphore(max_concurrent_requests)

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

        if self._rate_limiter.enabled:
            limits = []
            if rpm_limit > 0:
                limits.append(f"RPM={rpm_limit}")
            if tpm_limit > 0:
                limits.append(f"TPM={tpm_limit}")
            print(
                f"[GeminiBackend] Rate limiting: {', '.join(limits)}, "
                f"max_concurrent={max_concurrent_requests}"
            )

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Call the Gemini API with rate limiting and exponential backoff retry."""
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
            # Wait for rate limiter before each attempt
            self._rate_limiter.acquire()

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

                # Record tokens for TPM tracking
                self._rate_limiter.record_tokens(input_tokens + output_tokens)

                return BackendResponse(
                    text=text.strip(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Longer backoff for rate limit errors (429)
                    err_str = str(e).lower()
                    if "429" in err_str or "rate" in err_str or "quota" in err_str:
                        backoff = (2 ** (attempt + 1)) + random.uniform(0, 2)
                        print(
                            f"[GeminiBackend] Rate limited (attempt {attempt + 1}/"
                            f"{self.max_retries}), sleeping {backoff:.1f}s"
                        )
                    else:
                        backoff = 2**attempt + random.uniform(0, 1)
                    time.sleep(backoff)
                    continue

        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_batch(
        self, messages_list: list[list[dict[str, Any]]]
    ) -> list[BackendResponse]:
        """Batch generate responses using concurrent threads with rate limiting."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = len(messages_list)
        if n <= 1:
            return [self.generate(msgs) for msgs in messages_list]

        def _rate_limited_generate(msgs: list[dict[str, Any]]) -> BackendResponse:
            with self._semaphore:
                return self.generate(msgs)

        workers = min(n, self.max_concurrent_requests)
        results: list[BackendResponse | None] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_rate_limited_generate, msgs): i
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
