"""vLLM VLM backend for fast vision-language model inference with batching."""

from __future__ import annotations

import os
import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class VLLMVLMBackend(ModelBackend):
    """Backend for vision-language models served via vLLM.

    Supports multimodal inputs (text + images) with PagedAttention,
    prefix caching, and continuous batching.

    Tested with Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3.5-4B, and other
    chat-template-compatible VLMs.
    """

    supports_vision = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        dtype: str = "bfloat16",
        max_new_tokens: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        enable_thinking: bool = False,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        limit_mm_per_prompt: dict[str, int] | None = None,
    ):
        self.name = f"vllm-vlm/{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.enable_thinking = enable_thinking
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 4}

        self._llm = None
        self._sampling_params = None

    def _ensure_loaded(self) -> None:
        """Lazily load the vLLM engine on first use."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm package not installed. Run: uv sync --extra vllm"
            )

        engine_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "tensor_parallel_size": self.tensor_parallel_size,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
            "max_model_len": self.max_model_len,
            # Use Triton attention to avoid FlashInfer JIT compilation
            # failures on clusters where libcuda.so is not in linker path.
            "attention_backend": "TRITON_ATTN",
        }

        # Per-process compile cache to prevent corruption when multiple
        # SLURM jobs share the same home directory.
        compile_cache = os.environ.get(
            "VLLM_COMPILE_CACHE_DIR",
            f"/tmp/vllm_compile_cache_{os.getpid()}",
        )
        engine_kwargs["compilation_config"] = {"cache_dir": compile_cache}

        self._llm = LLM(**engine_kwargs)

        sampling_kwargs: dict[str, Any] = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.min_p > 0:
            sampling_kwargs["min_p"] = self.min_p

        self._sampling_params = SamplingParams(**sampling_kwargs)

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert internal message format to vLLM-compatible multimodal format.

        Converts Anthropic-style base64 image blocks to OpenAI-style data-URL
        ``image_url`` blocks that vLLM's chat API expects.
        """
        converted = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                converted.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": content}],
                    }
                )
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if block.get("type") == "text":
                        new_blocks.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            data_url = (
                                f"data:{source.get('media_type', 'image/png')};"
                                f"base64,{source['data']}"
                            )
                            new_blocks.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )
                        else:
                            new_blocks.append({"type": "image_url", "image_url": {}})
                converted.append({"role": msg["role"], "content": new_blocks})
            else:
                converted.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": str(content)}],
                    }
                )
        return converted

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Generate a response using vLLM with multimodal inputs."""
        self._ensure_loaded()
        assert self._llm is not None and self._sampling_params is not None

        converted = self._convert_messages(messages)

        chat_kwargs: dict[str, Any] = {}
        chat_kwargs["chat_template_kwargs"] = {
            "enable_thinking": self.enable_thinking,
        }

        start = time.time()
        outputs = self._llm.chat(
            messages=converted,
            sampling_params=self._sampling_params,
            **chat_kwargs,
        )
        latency = time.time() - start

        output = outputs[0]
        text = output.outputs[0].text.strip()
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        return BackendResponse(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            latency_seconds=latency,
        )

    def generate_batch(
        self, messages_list: list[list[dict[str, Any]]]
    ) -> list[BackendResponse]:
        """Batch generate multimodal responses — single vLLM call."""
        self._ensure_loaded()
        assert self._llm is not None and self._sampling_params is not None

        converted_list = [self._convert_messages(msgs) for msgs in messages_list]

        chat_kwargs: dict[str, Any] = {}
        chat_kwargs["chat_template_kwargs"] = {
            "enable_thinking": self.enable_thinking,
        }

        start = time.time()
        outputs = self._llm.chat(
            messages=converted_list,
            sampling_params=self._sampling_params,
            **chat_kwargs,
        )
        latency = time.time() - start

        per_item_latency = latency / max(len(outputs), 1)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            results.append(
                BackendResponse(
                    text=text,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    latency_seconds=per_item_latency,
                )
            )

        return results
