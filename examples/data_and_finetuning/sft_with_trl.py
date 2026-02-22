"""Complete Supervised Fine-Tuning (SFT) pipeline using AgentickSFTTrainer.

Full pipeline: collect multi-task oracle data -> train with TRL + LoRA -> generate
eval config YAML -> optionally evaluate inline -> optionally push to HuggingFace Hub.

Requires: uv sync --extra finetune (transformers, torch, trl, peft)

Usage:
    # Quick single-task test
    uv run python examples/data_and_finetuning/sft_with_trl.py \
        --tasks GoToGoal-v0 --difficulties easy --n-episodes 5

    # Full multi-task training (all tasks, all difficulties)
    uv run python examples/data_and_finetuning/sft_with_trl.py

    # Train + evaluate + push to Hub
    uv run python examples/data_and_finetuning/sft_with_trl.py \
        --eval --push-to-hub user/agentick-sft-model

    # Re-run eval later with generated config
    python -m agentick.experiments.run --config models/sft/eval_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import (
    add_collection_args,
    add_eval_args,
    add_hub_args,
    add_task_args,
    build_eval_config,
    collect_multi_task_data,
    resolve_tasks,
    run_eval,
    save_eval_config,
)


def check_requirements() -> bool:
    """Check if required packages are installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from trl import SFTTrainer  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Required packages not installed: {e}")
        print("Install with: uv sync --extra finetune")
        return False

    import torch

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be very slow.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SFT pipeline: collect oracle data -> train -> eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task / data args
    add_task_args(parser)
    add_collection_args(parser)

    # Model / LoRA args
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace model name")
    model_group.add_argument(
        "--use-lora", action="store_true", default=True, help="Enable LoRA (default: True)"
    )
    model_group.add_argument(
        "--no-lora", dest="use_lora", action="store_false", help="Disable LoRA"
    )
    model_group.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    model_group.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    # Training args
    train_group = parser.add_argument_group("training")
    train_group.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    train_group.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    train_group.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps"
    )
    train_group.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    train_group.add_argument("--report-to", default="none", help="Experiment tracker (none, wandb)")

    # Eval + Hub args
    add_eval_args(parser)
    add_hub_args(parser)

    args = parser.parse_args()

    if not check_requirements():
        return

    tasks = resolve_tasks(args.tasks)
    output_dir = args.output_dir or "models/sft"
    data_dir = Path(output_dir) / "data"

    print("SFT Pipeline with AgentickSFTTrainer")
    print("=" * 80)
    print(f"Tasks: {len(tasks)} ({', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''})")
    print(f"Difficulties: {args.difficulties}")
    print(f"Model: {args.model}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}" if args.use_lora else "LoRA: disabled")
    print("=" * 80)

    # Step 1: Collect oracle data
    print("\n" + "=" * 80)
    print("STEP 1: Collecting Oracle Demonstrations")
    print("=" * 80)

    combined_path = collect_multi_task_data(
        tasks=tasks,
        difficulties=args.difficulties,
        n_episodes=args.n_episodes,
        render_mode=args.render_mode,
        output_dir=str(data_dir),
        seed_offset=args.seed_offset,
    )

    # Step 2: Train
    print("\n" + "=" * 80)
    print("STEP 2: Fine-tuning with AgentickSFTTrainer")
    print("=" * 80)

    from agentick.training.trl.sft import AgentickSFTTrainer

    trainer = AgentickSFTTrainer(
        model_name=args.model,
        dataset_path=combined_path,
        output_dir=output_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        report_to=args.report_to,
    )

    print("\nTraining model...")
    metrics = trainer.train()
    print(f"\nTraining complete. Metrics: {metrics}")

    # Step 3: Optional push to Hub
    model_id = output_dir
    if args.push_to_hub:
        print("\n" + "=" * 80)
        print("STEP 3: Pushing to HuggingFace Hub")
        print("=" * 80)
        url = trainer.push_to_hub(args.push_to_hub)
        print(f"Model pushed to {url}")
        model_id = args.push_to_hub

    # Step 4: Generate eval config YAML
    print("\n" + "=" * 80)
    print("STEP 4: Generating Eval Config")
    print("=" * 80)

    eval_config = build_eval_config(
        name=f"sft-eval-{Path(output_dir).name}",
        model_path=model_id,
        tasks=tasks,
        difficulties=args.difficulties,
        harness=args.harness,
        render_mode=args.render_mode,
        n_episodes=args.eval_episodes,
        n_seeds=args.eval_seeds,
        output_dir=str(Path(output_dir) / "eval_results"),
    )
    config_path = save_eval_config(eval_config, Path(output_dir) / "eval_config.yaml")

    # Step 5: Optional inline eval
    if args.eval and not args.no_eval:
        print("\n" + "=" * 80)
        print("STEP 5: Running Evaluation")
        print("=" * 80)
        results = run_eval(eval_config)
        print(f"\nEval results saved to {results.output_dir}")
        print(f"Summary: {results.summary}")

    print("\n" + "=" * 80)
    print("SFT PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nModel: {output_dir}")
    print(f"Eval config: {config_path}")
    print("\nRe-run eval with:")
    print(f"  python -m agentick.experiments.run --config {config_path}")
    if args.push_to_hub:
        print(f"\nHF Hub: https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
