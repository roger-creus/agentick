"""HuggingFace local LLM backend using transformers."""

from __future__ import annotations

import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class HuggingFaceLLMBackend(ModelBackend):
    """Backend for local HuggingFace causal language models."""

    supports_vision = False

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
        quantization: str | None = None,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ):
        self.name = f"hf/{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazily load the model and tokenizer on first use."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers package not installed. Run: uv sync --extra llm")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs: dict[str, Any] = {}
        if self.quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        elif self.quantization == "8bit":
            model_kwargs["load_in_8bit"] = True

        if self.dtype:
            import torch

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(self.dtype, torch.bfloat16)

        if self.device == "auto":
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        if self.device != "auto":
            self._model = self._model.to(self.device)

        self._model.eval()

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Generate a response using the local HF model."""
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        # Flatten multimodal content blocks to text-only
        text_messages = _flatten_to_text(messages)

        # Use chat template if available, else manual formatting
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = _manual_chat_format(text_messages)

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        input_len = input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        start = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)
        latency = time.time() - start

        new_ids = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return BackendResponse(
            text=text,
            input_tokens=input_len,
            output_tokens=len(new_ids),
            latency_seconds=latency,
        )


def _flatten_to_text(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
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


def _manual_chat_format(messages: list[dict[str, str]]) -> str:
    """Simple fallback chat format when no chat template is available."""
    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"<|{role}|>\n{msg['content']}")
    parts.append("<|ASSISTANT|>\n")
    return "\n".join(parts)
