"""Base classes for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendResponse:
    """Response from a model backend."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0


class ModelBackend(ABC):
    """Abstract base for model backends (API or local)."""

    name: str
    supports_vision: bool

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Generate a response from the model.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                Content can be a string or a list of content blocks for
                multimodal messages (text + image).

        Returns:
            BackendResponse with generated text and token usage.
        """
        ...

    def generate_batch(
        self, messages_list: list[list[dict[str, Any]]]
    ) -> list[BackendResponse]:
        """Batch generate responses. Default: sequential fallback."""
        return [self.generate(msgs) for msgs in messages_list]
