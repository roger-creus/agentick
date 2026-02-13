"""HuggingFace model adapter for causal LMs and VLMs."""

from __future__ import annotations

from typing import Any


class HuggingFaceAgent:
    """
    Adapter for HuggingFace models (causal LMs and VLMs).

    Auto-downloads models from HuggingFace Hub.
    """

    def __init__(
        self,
        model_id: str,
        observation_mode: str = "language",
        device: str = "auto",
        dtype: str | None = "float16",
        quantization: str | None = None,  # "4bit", "8bit", or None
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        **kwargs,
    ):
        """
        Initialize HuggingFace agent.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3-8B")
            observation_mode: Observation format
            device: Device to run on ("auto", "cuda", "cpu")
            dtype: Data type ("float16", "bfloat16", "float32")
            quantization: Quantization mode ("4bit", "8bit", None)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model/tokenizer kwargs
        """
        self.model_id = model_id
        self.observation_mode = observation_mode
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.kwargs = kwargs

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace Hub."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace transformers not installed. "
                "Install with: uv sync --extra llm"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Load model with quantization if specified
        model_kwargs = {}
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
            model_kwargs["torch_dtype"] = dtype_map.get(self.dtype, torch.float16)

        if self.device == "auto":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
            **self.kwargs,
        )

        # Move to device if not auto
        if self.device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()

    def reset(self) -> None:
        """Reset (no state to reset for stateless models)."""
        pass

    @property
    def name(self) -> str:
        """Get agent name."""
        return f"hf_{self.model_id.replace('/', '_')}"

    def _format_prompt(self, observation: Any, info: dict[str, Any]) -> str:
        """Format observation into prompt."""
        from agentick.leaderboard.adapters.prompt_templates import format_observation_to_text

        return format_observation_to_text(observation, info, self.observation_mode)

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action using HuggingFace model.

        Args:
            observation: Environment observation
            info: Info dict

        Returns:
            Action index
        """
        # Format prompt
        prompt = self._format_prompt(observation, info)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Parse action
        from agentick.leaderboard.adapters.prompt_templates import parse_action_from_text

        valid_actions = info.get("valid_actions", [])
        action = parse_action_from_text(response, valid_actions)

        return action


class VisionLanguageAgent:
    """
    Adapter for Vision-Language Models (VLMs).

    For models that can process both images and text.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str | None = "float16",
        max_new_tokens: int = 50,
        **kwargs,
    ):
        """
        Initialize VLM agent.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen2-VL-7B")
            device: Device to run on
            dtype: Data type
            max_new_tokens: Max tokens to generate
            **kwargs: Additional kwargs
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs

        self._load_model()

    def _load_model(self):
        """Load VLM model and processor."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError("transformers not installed")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            device_map=self.device if self.device == "auto" else None,
            **self.kwargs,
        )

        if self.device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()

    def reset(self) -> None:
        """Reset state."""
        pass

    @property
    def name(self) -> str:
        """Get agent name."""
        return f"vlm_{self.model_id.replace('/', '_')}"

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action using VLM.

        Expects observation to be an image (numpy array or PIL Image).
        """
        # Format text prompt
        task_name = info.get("task_name", "task")
        valid_actions = info.get("valid_actions", [])
        text_prompt = (
            f"Task: {task_name}. Valid actions: {valid_actions}. Select best action (number only):"
        )

        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=observation,
            return_tensors="pt",
        )

        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse action
        from agentick.leaderboard.adapters.prompt_templates import parse_action_from_text

        action = parse_action_from_text(response, valid_actions)

        return action
