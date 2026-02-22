"""HuggingFace local LLM backend using transformers."""

from __future__ import annotations

import time
from typing import Any

from agentick.agents.backends._utils import flatten_to_text, manual_chat_format
from agentick.agents.backends.base import BackendResponse, ModelBackend


class HuggingFaceLLMBackend(ModelBackend):
    """Backend for local HuggingFace causal language models."""

    supports_vision = False

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "auto",
        dtype: str = "bfloat16",
        quantization: str | None = None,
        max_new_tokens: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        enable_thinking: bool = False,
    ):
        self.name = f"hf/{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.enable_thinking = enable_thinking

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
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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
        text_messages = flatten_to_text(messages)

        # Use chat template if available, else manual formatting.
        # Pass enable_thinking=False for Qwen3 models to suppress <think>...</think> output,
        # which would exhaust max_new_tokens before producing a valid action.
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    text_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Model's chat template doesn't support enable_thinking (non-Qwen3)
                prompt = self._tokenizer.apply_chat_template(
                    text_messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt = manual_chat_format(text_messages)

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs["attention_mask"].to(self._model.device)
        input_len = input_ids.shape[1]

        from transformers import GenerationConfig

        gen_config_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_config_kwargs["do_sample"] = True
            gen_config_kwargs["temperature"] = self.temperature
            gen_config_kwargs["top_p"] = self.top_p
            gen_config_kwargs["top_k"] = self.top_k
            if self.min_p > 0:
                gen_config_kwargs["min_p"] = self.min_p
        else:
            gen_config_kwargs["do_sample"] = False

        # Use explicit GenerationConfig to avoid the model's stored
        # generation_config overriding our sampling parameters.
        gen_config = GenerationConfig(**gen_config_kwargs)

        start = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        latency = time.time() - start

        new_ids = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return BackendResponse(
            text=text,
            input_tokens=input_len,
            output_tokens=len(new_ids),
            latency_seconds=latency,
        )



# Keep module-level aliases for backward compatibility
_flatten_to_text = flatten_to_text
_manual_chat_format = manual_chat_format
