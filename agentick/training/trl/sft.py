"""Supervised Fine-Tuning (SFT) with HuggingFace TRL.

Plug-and-play SFT on data collected by :class:`agentick.data.DataCollector`.
Supports LoRA (default) and full fine-tuning of any HuggingFace causal LM.

Example::

    from agentick.training.trl.sft import AgentickSFTTrainer

    trainer = AgentickSFTTrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        dataset_path="trajectories/hf_conv/",
        output_dir="models/sft/",
        use_lora=True,
    )
    trainer.train()
    agent = trainer.as_agent()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class AgentickSFTTrainer:
    """High-level wrapper for TRL SFTTrainer on agentick data.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-0.5B"``).
        dataset_path: Path to HuggingFace Datasets directory (from
            :meth:`CollectedDataset.export_to_huggingface`).
        output_dir: Directory to save checkpoints and final model.
        observation_mode: Which observation modality was used (for agent wrapping).
        use_lora: Enable LoRA (default ``True``).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_target_modules: LoRA target modules (default ``"all-linear"``).
        learning_rate: Optimizer learning rate.
        num_train_epochs: Training epochs.
        per_device_train_batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_length: Maximum sequence length.
        packing: Pack multiple examples per sequence.
        logging_steps: Log every N steps.
        save_strategy: When to save checkpoints.
        report_to: Experiment tracker (e.g. ``"wandb"``).
        wandb_project: Weights & Biases project name.
        model_kwargs: Extra kwargs for ``AutoModelForCausalLM.from_pretrained``.
        training_kwargs: Extra kwargs for ``SFTConfig``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        dataset_path: str | Path | None = None,
        dataset: Any | None = None,
        output_dir: str | Path = "models/sft",
        observation_mode: str = "language",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: str | list[str] = "all-linear",
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 1024,
        packing: bool = True,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        report_to: str = "none",
        wandb_project: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        training_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from trl import SFTConfig, SFTTrainer  # noqa: F401
        except ImportError:
            raise ImportError(
                "TRL is required for SFT training. Install with: pip install trl[peft]"
            )

        self.model_name = model_name
        self.output_dir = str(output_dir)
        self.observation_mode = observation_mode
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.packing = packing
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.report_to = report_to
        self.wandb_project = wandb_project
        self.model_kwargs = model_kwargs or {}
        self.training_kwargs = training_kwargs or {}

        # Load dataset
        if dataset is not None:
            self._dataset = dataset
        elif dataset_path is not None:
            from datasets import Dataset as HFDataset

            self._dataset = HFDataset.load_from_disk(str(dataset_path))
        else:
            raise ValueError("Provide either dataset_path or dataset")

        self._trainer = None
        self._trained = False

    def _build_trainer(self) -> Any:
        """Build the TRL SFTTrainer."""
        from trl import SFTConfig, SFTTrainer

        # Training config
        sft_kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "packing": self.packing,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "report_to": self.report_to,
        }

        if self.wandb_project and self.report_to == "wandb":
            import os

            os.environ.setdefault("WANDB_PROJECT", self.wandb_project)

        sft_kwargs.update(self.training_kwargs)
        training_args = SFTConfig(**sft_kwargs)

        # LoRA config
        peft_config = None
        if self.use_lora:
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
            )

        trainer = SFTTrainer(
            model=self.model_name,
            args=training_args,
            train_dataset=self._dataset,
            peft_config=peft_config,
        )
        return trainer

    def train(self) -> dict[str, Any]:
        """Run SFT training.

        Returns:
            Training metrics dict.
        """
        self._trainer = self._build_trainer()
        result = self._trainer.train()
        self._trained = True

        # Save final model
        self._trainer.save_model(self.output_dir)

        return result.metrics if hasattr(result, "metrics") else {}

    def push_to_hub(self, repo_id: str) -> str:
        """Push trained model to HuggingFace Hub.

        Args:
            repo_id: Repository identifier (e.g. ``"user/my-model"``).

        Returns:
            URL of the uploaded model.
        """
        if self._trainer is None:
            raise RuntimeError("Must call train() before push_to_hub()")
        self._trainer.push_to_hub(repo_id)
        return f"https://huggingface.co/{repo_id}"

    def as_agent(self, device: str = "auto") -> SFTAgent:
        """Wrap the trained model as an agentick agent.

        Args:
            device: Device for inference (``"auto"``, ``"cpu"``, ``"cuda"``).

        Returns:
            An :class:`SFTAgent` that implements ``act(obs, info) -> int``.
        """
        model_path = self.output_dir
        return SFTAgent(
            model_path=model_path,
            base_model_name=self.model_name,
            use_lora=self.use_lora,
            device=device,
        )

    @classmethod
    def from_collector(
        cls,
        dataset: Any,
        observation_mode: str = "language",
        **kwargs: Any,
    ) -> AgentickSFTTrainer:
        """Create trainer from a :class:`CollectedDataset`.

        Args:
            dataset: A ``CollectedDataset`` from :meth:`DataCollector.collect`.
            observation_mode: Observation modality for text data.
            **kwargs: Additional keyword arguments for ``AgentickSFTTrainer``.

        Returns:
            Configured ``AgentickSFTTrainer``.
        """
        import tempfile

        # Export to HF conversation format
        tmp = tempfile.mkdtemp(prefix="agentick_sft_")
        hf_path = Path(tmp) / "hf_data"
        dataset.export_to_huggingface(
            hf_path,
            format="conversation",
            observation_mode=observation_mode,
        )

        from datasets import Dataset as HFDataset

        hf_dataset = HFDataset.load_from_disk(str(hf_path))

        return cls(
            dataset=hf_dataset,
            observation_mode=observation_mode,
            **kwargs,
        )


class SFTAgent:
    """Agent wrapper for a fine-tuned causal LM.

    Implements the ``act(obs, info) -> int`` interface expected by agentick.
    Parses the model output for an action integer.
    """

    def __init__(
        self,
        model_path: str,
        base_model_name: str | None = None,
        use_lora: bool = True,
        device: str = "auto",
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.last_reasoning: str | None = None

        # Load tokenizer
        tokenizer_path = base_model_name or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if use_lora:
            try:
                from peft import AutoPeftModelForCausalLM

                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
            )
        self.model.eval()

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Reset agent state (no-op for SFT agent)."""
        self.last_reasoning = None

    def act(self, obs: Any, info: dict[str, Any]) -> int:
        """Generate an action from the observation.

        The model is prompted with the observation text and expected to
        output an action integer.
        """
        import torch

        if isinstance(obs, str):
            obs_text = obs
        elif isinstance(obs, dict):
            obs_text = json.dumps(obs)
        else:
            obs_text = str(obs)

        messages = [
            {"role": "user", "content": obs_text},
        ]

        # Apply chat template
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = f"Observation: {obs_text}\nAction: "

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        from transformers import GenerationConfig

        gen_config_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
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

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, generation_config=gen_config
            )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        self.last_reasoning = text

        # Parse action integer from output
        return self._parse_action(text)

    @staticmethod
    def _parse_action(text: str) -> int:
        """Extract an action integer from model output text."""
        # Try "Action: N" pattern
        match = re.search(r"Action:\s*(\d+)", text)
        if match:
            return int(match.group(1))
        # Try bare integer
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return int(match.group(1))
        return 0  # noop fallback
