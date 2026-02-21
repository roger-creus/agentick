"""HuggingFace local VLM backend using transformers."""

from __future__ import annotations

import time
from typing import Any

from agentick.agents.backends.base import BackendResponse, ModelBackend


class HuggingFaceVLMBackend(ModelBackend):
    """Backend for local HuggingFace vision-language models (e.g. Qwen3-VL)."""

    supports_vision = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ):
        self.name = f"hf-vlm/{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

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
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def generate(self, messages: list[dict[str, Any]]) -> BackendResponse:
        """Generate a response using the local VLM."""
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        import torch

        # Convert messages to Qwen VL format
        qwen_messages = self._convert_messages(messages)

        # Use processor to build inputs
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
            inputs = self._processor(
                text=[text_prompt], return_tensors="pt", truncation=True
            )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        start = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
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
