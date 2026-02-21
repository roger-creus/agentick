"""Tinker SFT — Supervised fine-tuning of LLMs via the Tinker API.

Uses Tinker's remote training infrastructure for LoRA fine-tuning on
agentick oracle demonstrations.  Data comes from
:class:`agentick.data.CollectedDataset` (conversation format).

.. note::
    Tinker must be installed separately (``pip install tinker``) and a
    Tinker API key must be set (``TINKER_API_KEY``).

Example::

    from agentick.training.tinker.sft import TinkerSFTTrainer

    trainer = TinkerSFTTrainer(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="trajectories/hf_conv/",
        rank=32,
    )
    trainer.train(num_steps=100, learning_rate=1e-4)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TINKER_AVAILABLE = False
try:
    import tinker
    import tinker.types as tinker_types

    _TINKER_AVAILABLE = True
except ImportError:
    tinker = None  # type: ignore[assignment]
    tinker_types = None  # type: ignore[assignment]


def _require_tinker() -> None:
    if not _TINKER_AVAILABLE:
        raise ImportError(
            "Tinker is not installed. Install with: pip install tinker\n"
            "See https://tinker-docs.thinkingmachines.ai/install"
        )


class TinkerSFTTrainer:
    """SFT trainer using Tinker's remote LoRA training API.

    Args:
        base_model: HuggingFace model name (must be supported by Tinker).
        dataset_path: Path to HuggingFace Datasets directory with
            conversation-format data (from ``CollectedDataset.export_to_huggingface``).
        rank: LoRA rank for fine-tuning.
        output_dir: Directory for logs and metadata.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        dataset_path: str | Path | None = None,
        collected_dataset: Any | None = None,
        rank: int = 32,
        output_dir: str | Path = "models/tinker_sft",
    ) -> None:
        _require_tinker()

        self.base_model = base_model
        self.rank = rank
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load conversation data
        self._messages_list: list[list[dict[str, str]]] = []
        if dataset_path is not None:
            self._load_from_hf(Path(dataset_path))
        elif collected_dataset is not None:
            self._load_from_collected(collected_dataset)

        self._training_client = None
        self._sampling_client = None
        self._tokenizer = None

    def _load_from_hf(self, path: Path) -> None:
        """Load conversation-format HF dataset."""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package required: pip install datasets")

        ds = Dataset.load_from_disk(str(path))
        for row in ds:
            messages = row.get("messages", [])
            if messages:
                self._messages_list.append(messages)

    def _load_from_collected(self, dataset: Any) -> None:
        """Load from an in-memory CollectedDataset."""
        rows = dataset._to_conversation_rows(
            dataset.modalities[0] if dataset.modalities else "language",
        )
        for row in rows:
            self._messages_list.append(row["messages"])

    def _prepare_data(self) -> list[Any]:
        """Convert messages to Tinker Datum objects."""
        tokenizer = self._tokenizer

        data = []
        for messages in self._messages_list:
            # Build prompt/completion token sequences
            prompt_text = ""
            completion_text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    completion_text += content + " "
                else:
                    prompt_text += f"{role}: {content}\n"

            prompt_tokens = tokenizer.encode(prompt_text)
            completion_tokens = tokenizer.encode(completion_text)

            input_tokens = prompt_tokens + completion_tokens
            target_tokens = [-100] * len(prompt_tokens) + completion_tokens

            # Weights: 0 for prompt, 1 for completion
            weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

            # Shift: target is input[1:], weights are weights[1:]
            target_tokens = target_tokens[1:]
            weights = weights[1:]

            datum = tinker_types.Datum(
                model_input=tinker_types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker_types.TensorData.from_list(
                        target_tokens,
                        dtype="int64",
                    ),
                    "weights": tinker_types.TensorData.from_list(
                        weights,
                        dtype="float32",
                    ),
                },
            )
            data.append(datum)

        return data

    def train(
        self,
        num_steps: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
    ) -> dict[str, Any]:
        """Run SFT training on Tinker.

        Args:
            num_steps: Number of optimization steps.
            learning_rate: Adam learning rate.
            batch_size: Examples per forward-backward pass.

        Returns:
            Training metrics dict.
        """
        _require_tinker()

        # Create training client
        service_client = tinker.ServiceClient()
        self._training_client = service_client.create_lora_training_client(
            base_model=self.base_model,
            rank=self.rank,
        )
        self._tokenizer = self._training_client.get_tokenizer()

        # Prepare training data
        data = self._prepare_data()
        if not data:
            raise ValueError("No training data available")

        losses = []
        for step in range(num_steps):
            # Sample a batch
            batch_indices = np.random.choice(len(data), size=min(batch_size, len(data)))
            batch = [data[i] for i in batch_indices]

            # Forward-backward
            fwdbwd_future = self._training_client.forward_backward(
                batch,
                loss_fn="cross_entropy",
            )
            optim_future = self._training_client.optim_step(
                tinker_types.AdamParams(learning_rate=learning_rate),
            )

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            # Compute loss
            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs]
            )
            weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
            loss = -np.dot(logprobs, weights) / max(weights.sum(), 1e-8)
            losses.append(float(loss))

            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{num_steps}: loss={loss:.4f}")

        # Save sampling client for inference
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="agentick_sft",
        )

        # Save metadata
        meta = {
            "base_model": self.base_model,
            "rank": self.rank,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "final_loss": losses[-1] if losses else None,
            "num_examples": len(data),
        }
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"losses": losses, "final_loss": losses[-1] if losses else None}

    def as_agent(self) -> TinkerSFTAgent:
        """Wrap the fine-tuned model as an agentick agent.

        Requires :meth:`train` to have been called first.
        """
        if self._sampling_client is None:
            raise RuntimeError("Must call train() first")
        return TinkerSFTAgent(
            sampling_client=self._sampling_client,
            tokenizer=self._tokenizer,
        )


class TinkerSFTAgent:
    """Agent using a Tinker-trained model for inference.

    Implements ``act(obs, info) -> int``.
    """

    def __init__(self, sampling_client: Any, tokenizer: Any) -> None:
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.last_reasoning: str | None = None

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        self.last_reasoning = None

    def act(self, obs: Any, info: dict[str, Any]) -> int:
        """Generate action from observation via Tinker sampling."""
        import re

        obs_text = obs if isinstance(obs, str) else json.dumps(obs)
        prompt = f"Observation: {obs_text}\nAction: "
        tokens = self.tokenizer.encode(prompt)

        model_input = tinker_types.ModelInput.from_ints(tokens=tokens)
        result = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker_types.SamplingParams(max_tokens=32),
        )

        text = self.tokenizer.decode(result.sequences[0].tokens).strip()
        self.last_reasoning = text

        # Parse action integer
        match = re.search(r"\b(\d+)\b", text)
        return int(match.group(1)) if match else 0
