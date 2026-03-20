"""vLLM LLM backend for fast local inference with batching and prefix caching."""

from __future__ import annotations

import os
import time
from typing import Any

from agentick.agents.backends._utils import flatten_to_text
from agentick.agents.backends.base import BackendResponse, ModelBackend


class VLLMLLMBackend(ModelBackend):
    """Backend for text-only LLMs served via vLLM.

    Uses PagedAttention, prefix caching, and continuous batching for
    significantly faster inference than eager HuggingFace transformers.
    """

    supports_vision = False

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
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
        quantization: str | None = None,
        enforce_eager: bool = False,
    ):
        self.name = f"vllm/{model_id.split('/')[-1]}"
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
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization

        self._llm = None
        self._sampling_params = None

    def _ensure_model_downloaded(self) -> None:
        """Pre-download model files under a filelock to prevent concurrent corruption.

        When multiple SLURM jobs share the same HuggingFace cache directory,
        concurrent downloads can corrupt JSON files. This acquires an exclusive
        lock so only one process downloads at a time.
        """
        from huggingface_hub import snapshot_download

        # Let snapshot_download use HF_HOME env var (adds /hub/ prefix automatically)
        lock_dir = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(
            lock_dir, self.model_id.replace("/", "--") + ".lock"
        )

        import filelock

        with filelock.FileLock(lock_path, timeout=1800):
            snapshot_download(self.model_id, local_files_only=True)

    def _ensure_loaded(self) -> None:
        """Lazily load the vLLM engine on first use."""
        if self._llm is not None:
            return

        self._ensure_model_downloaded()

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
            "max_model_len": self.max_model_len,
        }
        if getattr(self, "enforce_eager", False):
            engine_kwargs["enforce_eager"] = True
        if self.quantization is not None:
            engine_kwargs["quantization"] = self.quantization

        # Per-process compile cache to prevent corruption when multiple
        # SLURM jobs share the same home directory.  VLLM_COMPILE_CACHE_DIR
        # is set per-job in job_template.sh; fall back to PID-based dir.
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

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Generate a response using vLLM."""
        self._ensure_loaded()
        assert self._llm is not None and self._sampling_params is not None

        text_messages = flatten_to_text(messages)

        chat_kwargs: dict[str, Any] = {}
        chat_kwargs["chat_template_kwargs"] = {
            "enable_thinking": self.enable_thinking,
        }

        start = time.time()
        outputs = self._llm.chat(
            messages=text_messages,
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
        """Batch generate responses — single vLLM call for all requests."""
        self._ensure_loaded()
        assert self._llm is not None and self._sampling_params is not None

        text_messages_list = [flatten_to_text(msgs) for msgs in messages_list]

        chat_kwargs: dict[str, Any] = {}
        chat_kwargs["chat_template_kwargs"] = {
            "enable_thinking": self.enable_thinking,
        }

        start = time.time()
        outputs = self._llm.chat(
            messages=text_messages_list,
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
