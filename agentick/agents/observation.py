"""Shared observation formatting utilities for agent harnesses."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np

from agentick.agents.prompt_templates import format_observation_to_text


def format_text_observation(obs: Any, info: dict[str, Any], mode: str) -> str:
    """Format an observation as text for LLM consumption.

    Uses pre-rendered secondary observation from info if available
    (multimodal case where primary obs is a different mode, e.g. rgb_array).
    Otherwise delegates to the existing prompt_templates utility.
    """
    actual_obs = info.get(f"obs_{mode}", obs)
    return format_observation_to_text(actual_obs, info, mode)


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert a numpy RGB array to a base64-encoded PNG string."""
    from PIL import Image

    image = Image.fromarray(image_array.astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def numpy_to_pil(image_array: np.ndarray) -> Any:
    """Convert a numpy RGB array to a PIL Image (for local VLM backends)."""
    from PIL import Image

    return Image.fromarray(image_array.astype(np.uint8))
