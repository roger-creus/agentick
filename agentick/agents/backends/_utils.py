"""Shared utilities for model backends."""

from __future__ import annotations

from typing import Any


def flatten_to_text(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert multimodal messages to text-only."""
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            parts = [block["text"] for block in content if block.get("type") == "text"]
            out.append({"role": msg["role"], "content": "\n".join(parts)})
        else:
            out.append({"role": msg["role"], "content": str(content)})
    return out


def manual_chat_format(messages: list[dict[str, str]]) -> str:
    """Simple fallback chat format when no chat template is available."""
    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"<|{role}|>\n{msg['content']}")
    parts.append("<|ASSISTANT|>\n")
    return "\n".join(parts)
