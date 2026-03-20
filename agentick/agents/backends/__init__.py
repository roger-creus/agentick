"""Model backends for LLM/VLM agents."""

from agentick.agents.backends.base import BackendResponse, ModelBackend

__all__ = [
    "BackendResponse",
    "ModelBackend",
]

# Lazy imports to avoid requiring all SDKs at import time
_BACKEND_REGISTRY: dict[str, str] = {
    "openai": "agentick.agents.backends.openai_backend.OpenAIBackend",
    "huggingface_llm": "agentick.agents.backends.huggingface_llm.HuggingFaceLLMBackend",
    "huggingface_vlm": "agentick.agents.backends.huggingface_vlm.HuggingFaceVLMBackend",
    "vllm_llm": "agentick.agents.backends.vllm_llm.VLLMLLMBackend",
    "vllm_vlm": "agentick.agents.backends.vllm_vlm.VLLMVLMBackend",
    "gemini": "agentick.agents.backends.gemini_backend.GeminiBackend",
    "anthropic": "agentick.agents.backends.anthropic_backend.AnthropicBackend",
}


def get_backend_class(name: str) -> type[ModelBackend]:
    """Get a backend class by name, importing lazily."""
    if name not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKEND_REGISTRY.keys())}")
    module_path, class_name = _BACKEND_REGISTRY[name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
