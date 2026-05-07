"""SFT on oracle trajectory dataset from HuggingFace.

Loads the flat per-step dataset produced by collect_oracle_trajectories.py,
converts it to chat format matching the eval harness prompts, and fine-tunes
a causal LM with LoRA using TRL.

Saves LoRA adapter weights to --output-dir. Use merge_and_push.py afterwards
to merge adapters into the base model and upload to HuggingFace Hub.

Usage:
    # ASCII modality (default), 8 GPUs
    torchrun --standalone --nnodes=1 --nproc_per_node 8 \
        examples/data_and_finetuning/sft_with_trl.py \
        --dataset rogercc/agentick-oracle-trajectories-120k \
        --model Qwen/Qwen3.5-4B

    # Language modality
    torchrun --standalone --nnodes=1 --nproc_per_node 8 \
        examples/data_and_finetuning/sft_with_trl.py \
        --dataset rogercc/agentick-oracle-trajectories-120k \
        --modality language \
        --model Qwen/Qwen3.5-4B

    # Single GPU, wandb logging
    python examples/data_and_finetuning/sft_with_trl.py \
        --dataset rogercc/agentick-oracle-trajectories-120k \
        --model Qwen/Qwen2.5-0.5B \
        --report-to wandb --wandb-project agentick-sft
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentick.agents.prompt_templates import (
    SYSTEM_PROMPT,
    format_observation_to_text,
    patch_chat_template_for_assistant_mask,
)


def build_chat_dataset(dataset, modality: str, max_steps_per_episode: int | None):
    """Convert flat per-step rows into chat-format messages for SFT.

    Each row becomes one independent (system, user, assistant) conversation
    matching the MarkovianZeroShot harness format used at eval time.
    """
    from datasets import Dataset

    from agentick.tasks.descriptions import get_task_description

    # Cache task descriptions
    desc_cache: dict[str, str] = {}

    rows = []
    for example in dataset:
        task_name = example["task"]

        if max_steps_per_episode is not None and example["step"] >= max_steps_per_episode:
            continue

        # Get task description (cached)
        if task_name not in desc_cache:
            desc_cache[task_name] = get_task_description(task_name)
        task_desc = desc_cache[task_name]

        # System message (matches eval harness exactly)
        system_content = SYSTEM_PROMPT.format(task_description=task_desc)

        # Pick the raw render from the dataset row for the requested modality.
        if modality == "ascii":
            raw_obs = str(example["ascii_render"])
        elif modality == "language":
            raw_obs = str(example["language_render"])
        else:
            raise ValueError(f"Unsupported modality for text SFT: {modality}")

        # Delegate prompt assembly + ANSI stripping to the eval source of truth
        # so training and inference are guaranteed to use identical formatting.
        info = {"task_name": task_name, "step": example["step"]}
        user_content = format_observation_to_text(raw_obs, info, modality)

        # Assistant = just the action number (matches what parse_action_from_text expects)
        assistant_content = str(example["action_int"])

        rows.append({
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description="SFT on oracle trajectories with LoRA + multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--dataset", required=True,
        help="HuggingFace dataset ID or local path (from collect_oracle_trajectories.py)",
    )
    parser.add_argument(
        "--modality", default="ascii", choices=["ascii", "language"],
        help="Which observation modality to train on",
    )
    parser.add_argument(
        "--max-steps-per-episode", type=int, default=None,
        help="Max steps per episode to include (None = all)",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Cap train rows before chat conversion (for smoke tests)",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=None,
        help="Cap eval rows before chat conversion (for smoke tests)",
    )

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Base model name")
    parser.add_argument(
        "--attn-implementation", default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: auto)",
    )

    # LoRA
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune (no LoRA)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules", default="all-linear",
        help="LoRA target modules (comma-sep or 'all-linear')",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument(
        "--max-steps", type=int, default=-1,
        help="Max training steps (-1 = use epochs)",
    )
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 (default)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--lr-scheduler", default="cosine", help="LR scheduler type")
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Transformers optimizer name",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=None,
        help="Number of processes for TRL dataset preprocessing",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Training dataloader worker processes",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=True,
        help="Gradient checkpointing to save memory (default: on)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
    )

    # Eval
    parser.add_argument(
        "--skip-eval", action="store_true", default=True,
        help=(
            "Skip per-epoch eval during training "
            "(default: skip; full test split eval is slow)"
        ),
    )
    parser.add_argument(
        "--no-skip-eval", dest="skip_eval", action="store_false",
        help="Run eval during training (SLOW for large test splits)",
    )

    # Logging
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--report-to", default="none", choices=["none", "wandb", "tensorboard"],
        help="Experiment tracker",
    )
    parser.add_argument("--wandb-project", default="agentick-sft", help="W&B project name")
    parser.add_argument("--run-name", default=None, help="Run name (auto-generated if None)")

    # Saving / upload
    parser.add_argument("--output-dir", default="models/sft", help="Output directory")
    parser.add_argument("--save-strategy", default="epoch", help="Checkpoint save strategy")
    parser.add_argument("--save-steps", type=int, default=2000, help="Checkpoint save interval")
    parser.add_argument("--save-total-limit", type=int, default=4, help="Max checkpoints to keep")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path to resume from, or 'auto' to use the latest output-dir checkpoint",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sft_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer

    # Load dataset. Three shapes are supported:
    # 1. HF Hub repo ID (e.g. "rogercc/agentick-oracle-trajectories-120k")
    # 2. Local path in `save_to_disk` layout (has dataset_info.json/dataset_dict.json)
    # 3. Local path in HF Hub snapshot layout (has data/*.parquet but no info JSON) —
    #    typical when pointing at a pre-resolved snapshot dir to avoid Hub calls.
    print(f"Loading dataset: {args.dataset}")
    ds_path = Path(args.dataset)
    if ds_path.exists():
        has_info = (ds_path / "dataset_info.json").exists()
        has_dict = (ds_path / "dataset_dict.json").exists()
        if has_info or has_dict:
            raw_dataset = load_from_disk(args.dataset)
        else:
            # Snapshot layout: use load_dataset with the local directory so
            # it builds from parquet files directly (no Hub call).
            raw_dataset = load_dataset(args.dataset)
    else:
        raw_dataset = load_dataset(args.dataset)

    # Support both DatasetDict (train/test splits) and plain Dataset
    from datasets import DatasetDict
    if isinstance(raw_dataset, DatasetDict):
        raw_train = raw_dataset["train"]
        raw_test = raw_dataset.get("test")
        print(f"Raw dataset: train={len(raw_train)} rows"
              + (f", test={len(raw_test)} rows" if raw_test else ""))
    else:
        raw_train = raw_dataset
        raw_test = None
        print(f"Raw dataset: {len(raw_train)} rows (no test split)")

    if args.max_train_examples is not None and len(raw_train) > args.max_train_examples:
        raw_train = raw_train.select(range(args.max_train_examples))
        print(f"Capped train dataset to {len(raw_train)} rows")
    if (
        raw_test is not None
        and args.max_eval_examples is not None
        and len(raw_test) > args.max_eval_examples
    ):
        raw_test = raw_test.select(range(args.max_eval_examples))
        print(f"Capped test dataset to {len(raw_test)} rows")

    # Convert to chat format
    print(f"Building chat dataset (modality={args.modality})...")
    chat_dataset = build_chat_dataset(raw_train, args.modality, args.max_steps_per_episode)
    print(f"Train chat dataset: {len(chat_dataset)} examples")
    chat_dataset = chat_dataset.shuffle(seed=42)

    eval_dataset = None
    if raw_test is not None and not args.skip_eval:
        eval_dataset = build_chat_dataset(raw_test, args.modality, args.max_steps_per_episode)
        print(f"Test chat dataset: {len(eval_dataset)} examples")
    elif raw_test is not None:
        print("Skipping test chat dataset construction (--skip-eval)")

    # -------------------------------------------------------------------------
    # Model & tokenizer
    # -------------------------------------------------------------------------
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Patch chat template so TRL's `assistant_only_loss=True` can locate
    # assistant tokens via the `{% generation %}` Jinja keyword. Qwen3.5's
    # stock template lacks these markers; without the patch SFTTrainer raises
    # "at least one example has no assistant tokens".
    if not patch_chat_template_for_assistant_mask(tokenizer):
        raise RuntimeError(
            f"Could not patch the chat template of '{args.model}' to emit "
            f"`{{% generation %}}` markers required by `assistant_only_loss=True`. "
            f"This likely means the tokenizer's template shape has drifted from "
            f"the Qwen3.5 family supported by "
            f"agentick.agents.prompt_templates.patch_chat_template_for_assistant_mask. "
            f"Update that helper with the new emission-site string before retrying."
        )

    # -------------------------------------------------------------------------
    # LoRA config
    # -------------------------------------------------------------------------
    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig

        target_modules = args.lora_target_modules
        if target_modules != "all-linear":
            target_modules = [m.strip() for m in target_modules.split(",")]

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, "
              f"dropout={args.lora_dropout}, targets={target_modules}")

    # -------------------------------------------------------------------------
    # Training config
    # -------------------------------------------------------------------------
    from trl import SFTConfig, SFTTrainer

    run_name = args.run_name
    if run_name is None:
        model_short = args.model.split("/")[-1]
        run_name = f"sft-{model_short}-{args.modality}-r{args.lora_r}"

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    model_kwargs = {"trust_remote_code": True}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    training_args = SFTConfig(
        output_dir=args.output_dir,
        logging_dir=str(output_dir / "logs"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        packing=args.packing,
        assistant_only_loss=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        optim=args.optim,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_num_proc=args.dataset_num_proc,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=run_name,
        # DDP — LoRA freezes most params, so DDP needs to find unused parameters
        ddp_find_unused_parameters=not args.no_lora,
        # Model loading kwargs
        model_init_kwargs=model_kwargs,
        # Eval config: only enabled if a test split is present AND the user
        # asked for it. Running eval on the full 120k+ test split takes hours
        # even with sdpa+bf16, so default to skip. See --no-skip-eval.
        **({
            "eval_strategy": args.save_strategy,
            "per_device_eval_batch_size": args.batch_size,
        } if (eval_dataset is not None and not args.skip_eval) else {}),
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=chat_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            from transformers.trainer_utils import get_last_checkpoint

            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint:
                print(f"Resuming from latest checkpoint: {resume_from_checkpoint}")
            else:
                print("No existing checkpoint found; starting from scratch.")
        else:
            resume_from_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    print("Starting training...")
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Log final metrics
    metrics = result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Save adapter / full model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")
    print("Run merge_and_push.py to merge LoRA adapters and upload to HuggingFace Hub.")


if __name__ == "__main__":
    main()
