"""Factory for creating agents from experiment configs."""

from __future__ import annotations

from typing import Any

from agentick.agents.backends import get_backend_class
from agentick.agents.base import BaseAgent
from agentick.agents.harness import HARNESS_REGISTRY
from agentick.experiments.config import AgentConfig


def create_agent(agent_config: AgentConfig) -> BaseAgent | None:
    """Create an agent from an AgentConfig.

    Returns None for agent types that don't use the harness system
    (e.g. "random", "ppo").

    Args:
        agent_config: Agent configuration from experiment YAML.

    Returns:
        BaseAgent instance, or None for non-LLM/VLM agents.
    """
    if agent_config.type not in ("llm", "vlm"):
        return None

    hp = agent_config.hyperparameters

    # --- Backend ---
    backend_name = hp.get("backend", "openai")

    # Auto-resolve "vllm" shorthand to vllm_llm or vllm_vlm
    if backend_name == "vllm":
        backend_name = "vllm_vlm" if agent_config.type == "vlm" else "vllm_llm"

    # Auto-upgrade HuggingFace backends to vLLM when vllm is installed
    if backend_name in ("huggingface_llm", "huggingface_vlm"):
        try:
            import vllm as _vllm  # noqa: F401

            _upgrade = {"huggingface_llm": "vllm_llm", "huggingface_vlm": "vllm_vlm"}
            backend_name = _upgrade[backend_name]
        except ImportError:
            pass

    backend_cls = get_backend_class(backend_name)

    # Collect backend-specific kwargs
    backend_kwargs: dict[str, Any] = {}
    if "model" in hp:
        backend_kwargs[
            "model" if backend_name in ("openai", "gemini", "anthropic") else "model_id"
        ] = hp["model"]
    for key in (
        "api_key_env",
        "base_url_env",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_retries",
        "device",
        "dtype",
        "quantization",
        "max_new_tokens",
        "enable_thinking",
        # Rate limiting (Gemini)
        "rpm_limit",
        "tpm_limit",
        "max_concurrent_requests",
        # vLLM-specific kwargs
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "max_model_len",
        "tensor_parallel_size",
        "limit_mm_per_prompt",
        "enforce_eager",
    ):
        if key in hp:
            backend_kwargs[key] = hp[key]

    backend = backend_cls(**backend_kwargs)

    # --- Harness ---
    harness_name = hp.get("harness", "markovian_zero_shot")
    if harness_name not in HARNESS_REGISTRY:
        raise ValueError(
            f"Unknown harness: {harness_name}. Available: {list(HARNESS_REGISTRY.keys())}"
        )
    harness_cls = HARNESS_REGISTRY[harness_name]
    harness_kwargs: dict[str, Any] = {}
    for key in ("max_context_tokens", "diff_mode", "max_response_chars"):
        if key in hp:
            harness_kwargs[key] = hp[key]
    # Pass max_tokens budget to reasoner harnesses
    if "reasoner" in harness_name and "max_tokens" in hp:
        harness_kwargs["max_tokens"] = hp["max_tokens"]
    harness = harness_cls(**harness_kwargs)

    # --- Observation modes ---
    obs_modes = hp.get("observation_modes", ["language"])

    return BaseAgent(
        backend=backend,
        harness=harness,
        observation_modes=obs_modes,
        agent_name=hp.get("agent_name"),
    )
