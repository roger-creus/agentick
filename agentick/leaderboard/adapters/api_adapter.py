"""API-based agent adapter supporting OpenAI, Anthropic, Gemini, and custom endpoints."""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import requests


class APIAgent:
    """
    Universal API adapter for LLM/VLM agents.

    Supports:
    - OpenAI (GPT-4o, GPT-4, etc.)
    - Anthropic (Claude Sonnet, Opus, etc.)
    - Google Gemini
    - Custom HTTP endpoints
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic", "gemini", "custom"],
        model: str,
        observation_mode: str = "language",
        api_key_env: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        log_calls: bool = True,
        **kwargs,
    ):
        """
        Initialize API agent.

        Args:
            provider: API provider
            model: Model name
            observation_mode: Observation format
            api_key_env: Environment variable containing API key
            api_key: Direct API key (if not using env var)
            endpoint: Custom endpoint URL (for custom provider)
            max_tokens: Max tokens in response
            temperature: Sampling temperature
            max_retries: Max retry attempts on failure
            timeout: Request timeout in seconds
            log_calls: Whether to log all API calls
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider
        self.model = model
        self.observation_mode = observation_mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.log_calls = log_calls
        self.kwargs = kwargs

        # Get API key
        if api_key:
            self.api_key = api_key
        elif api_key_env:
            self.api_key = os.getenv(api_key_env)
            if not self.api_key:
                raise ValueError(f"API key not found in environment variable: {api_key_env}")
        else:
            self.api_key = None

        # Set endpoint
        if provider == "custom":
            if not endpoint:
                raise ValueError("Custom provider requires 'endpoint' parameter")
            self.endpoint = endpoint
        else:
            self.endpoint = self._get_default_endpoint()

        # Call logs
        self.call_log: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.total_calls = 0

        # Episode state
        self.conversation_history: list[dict[str, Any]] = []

    def _get_default_endpoint(self) -> str:
        """Get default API endpoint for provider."""
        if self.provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        elif self.provider == "gemini":
            return (
                f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def reset(self) -> None:
        """Reset conversation history for new episode."""
        self.conversation_history = []

    @property
    def name(self) -> str:
        """Get agent name."""
        return f"{self.provider}_{self.model}"

    def _format_observation(self, observation: Any, info: dict[str, Any]) -> str:
        """
        Format observation into text prompt.

        Args:
            observation: Raw observation
            info: Info dict

        Returns:
            Formatted text prompt
        """
        from agentick.leaderboard.adapters.prompt_templates import format_observation_to_text

        return format_observation_to_text(observation, info, self.observation_mode)

    def _parse_action(self, response_text: str, valid_actions: list[int]) -> int:
        """
        Parse action from LLM response text.

        Args:
            response_text: LLM response
            valid_actions: List of valid action indices

        Returns:
            Action index
        """
        from agentick.leaderboard.adapters.prompt_templates import parse_action_from_text

        return parse_action_from_text(response_text, valid_actions)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]

        # Log tokens
        if "usage" in data:
            self.total_tokens += data["usage"].get("total_tokens", 0)

        return text

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        text = data["content"][0]["text"]

        # Log tokens
        if "usage" in data:
            self.total_tokens += data["usage"].get("input_tokens", 0) + data["usage"].get(
                "output_tokens", 0
            )

        return text

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        url = f"{self.endpoint}?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Gemini doesn't provide token counts in response, approximate
        self.total_tokens += len(prompt.split()) + len(text.split())

        return text

    def _call_custom(self, prompt: str) -> str:
        """Call custom HTTP endpoint."""
        headers = self.kwargs.get("headers", {})

        # Replace ${ENV_VAR} in headers
        for key, value in headers.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                headers[key] = os.getenv(env_var, "")

        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()

        # Try common response formats
        if "text" in data:
            return data["text"]
        elif "response" in data:
            return data["response"]
        elif "output" in data:
            return data["output"]
        else:
            raise ValueError(f"Unknown response format: {data}")

    def _call_api_with_retry(self, prompt: str) -> str:
        """
        Call API with exponential backoff retry.

        Args:
            prompt: Text prompt

        Returns:
            API response text
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Call provider-specific method
                if self.provider == "openai":
                    return self._call_openai(prompt)
                elif self.provider == "anthropic":
                    return self._call_anthropic(prompt)
                elif self.provider == "gemini":
                    return self._call_gemini(prompt)
                elif self.provider == "custom":
                    return self._call_custom(prompt)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

            except requests.exceptions.HTTPError as e:
                last_error = e

                # Check if rate limit (429) or server error (5xx)
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    # Exponential backoff
                    wait_time = 2**attempt
                    if self.log_calls:
                        print(f"API error {e.response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other HTTP error, don't retry
                    raise

            except Exception as e:
                last_error = e
                # Other error, retry
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    break

        # All retries failed
        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action by calling LLM API.

        Args:
            observation: Environment observation
            info: Info dict with valid_actions, task_name, etc.

        Returns:
            Action index
        """
        # Format observation into prompt
        prompt = self._format_observation(observation, info)

        # Call API
        start_time = time.time()
        response_text = self._call_api_with_retry(prompt)
        latency = time.time() - start_time

        # Parse action from response
        valid_actions = info.get("valid_actions", [])
        action = self._parse_action(response_text, valid_actions)

        # Log this call
        self.total_calls += 1
        if self.log_calls:
            self.call_log.append(
                {
                    "step": info.get("step", 0),
                    "prompt": prompt,
                    "response": response_text,
                    "action": action,
                    "latency": latency,
                }
            )

        return action

    def get_statistics(self) -> dict[str, Any]:
        """Get API call statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "call_log": self.call_log if self.log_calls else None,
        }
