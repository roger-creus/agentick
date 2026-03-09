"""HuggingFace local VLM backend using transformers."""

from __future__ import annotations

import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class HuggingFaceVLMBackend(ModelBackend):
    """Backend for local HuggingFace vision-language models.

    Supported models include Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3.5-4B,
    and other AutoModelForImageTextToText-compatible VLMs.
    """

    supports_vision = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        enable_thinking: bool = False,
    ):
        self.name = f"hf-vlm/{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.enable_thinking = enable_thinking

        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """Lazily load the model and processor on first use."""
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as VLMModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as VLMModelClass
        except ImportError:
            raise ImportError("transformers package not installed. Run: uv sync --extra llm")

        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"

        self._model = VLMModelClass.from_pretrained(
            self.model_id, trust_remote_code=True, **model_kwargs
        )

        if self.device != "auto":
            self._model = self._model.to(self.device)

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Generate a response using the local VLM."""
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        import torch

        # Convert messages to Qwen VL format
        qwen_messages = self._convert_messages(messages)

        # Use processor to build inputs.
        # Pass enable_thinking=False for Qwen3-VL to suppress <think>...</think> output.
        try:
            text_prompt = self._processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text_prompt = self._processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )

        # Collect images from messages
        images = self._extract_images(messages)

        if images:
            inputs = self._processor(
                text=[text_prompt], images=images, return_tensors="pt", truncation=True
            )
        else:
            inputs = self._processor(text=[text_prompt], return_tensors="pt", truncation=True)

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]

        from transformers import GenerationConfig

        gen_config_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
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
                **inputs, generation_config=gen_config
            )
        latency = time.time() - start

        new_ids = output_ids[0][input_len:]
        text = self._processor.decode(new_ids, skip_special_tokens=True).strip()

        return BackendResponse(
            text=text,
            input_tokens=int(input_len),
            output_tokens=len(new_ids),
            latency_seconds=latency,
        )

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert our internal format to Qwen VL message format."""
        qwen_msgs = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                qwen_msgs.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": content}],
                    }
                )
            elif isinstance(content, list):
                qwen_content = []
                for block in content:
                    if block.get("type") == "text":
                        qwen_content.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        # Qwen VL expects {"type": "image", "image": <PIL>}
                        qwen_content.append({"type": "image"})
                qwen_msgs.append({"role": msg["role"], "content": qwen_content})
            else:
                qwen_msgs.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": str(content)}],
                    }
                )
        return qwen_msgs

    def _extract_images(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract PIL images from message content blocks."""
        import base64
        from io import BytesIO

        from PIL import Image

        images = []
        for msg in messages:
            content = msg["content"]
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        img_bytes = base64.b64decode(source["data"])
                        img = Image.open(BytesIO(img_bytes))
                        images.append(img)
        return images
