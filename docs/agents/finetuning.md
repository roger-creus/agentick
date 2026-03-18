# Fine-Tuning

Fine-tune language models on expert trajectories from Agentick oracles.

## Pre-built Datasets

Oracle trajectory datasets are available on HuggingFace:

| Dataset | Size | Link |
|---------|------|------|
| `rogercc/agentick-oracle-trajectories-100k` | ~100k steps | [HuggingFace](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-100k) |
| `rogercc/agentick-oracle-trajectories-50k` | ~50k steps | [HuggingFace](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-50k) |
| `rogercc/agentick-oracle-trajectories-200k` | ~200k steps | [HuggingFace](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-200k) |
| `rogercc/agentick-oracle-trajectories-400k` | ~400k steps | [HuggingFace](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-400k) |

Each dataset contains per-step records with `ascii_render`, `language_render`, `action_int`, `action_name`, `task`, `difficulty`, `reward`, and `done` columns.

## Pipeline Overview

1. Collect oracle trajectories (or use pre-built datasets) → 2. SFT with TRL → 3. Evaluate

## Step 1: Collect Trajectories (optional)

Skip this if using the pre-built datasets above.

```bash
# ~100k steps: all 38 tasks, all difficulties, 10 episodes each
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
    --n-episodes 10 --push-to-hub rogercc/agentick-oracle-trajectories-100k

# ~50k steps: fewer episodes
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
    --n-episodes 5 --push-to-hub rogercc/agentick-oracle-trajectories-50k

# ~200k steps
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
    --n-episodes 20 --push-to-hub rogercc/agentick-oracle-trajectories-200k

# ~400k steps
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
    --n-episodes 40 --push-to-hub rogercc/agentick-oracle-trajectories-400k
```

## Step 2: Fine-Tune with TRL

Use TRL's `SFTTrainer` directly with LoRA. The script in `examples/data_and_finetuning/sft_with_trl.py` handles everything: loading the dataset, converting to chat format matching the eval harness prompts, and multi-GPU training.

```bash
# Single GPU
uv run python examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-100k \
    --model Qwen/Qwen2.5-0.5B

# Multi-GPU with accelerate
accelerate launch --num_processes 8 \
    examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-100k \
    --model Qwen/Qwen3.5-4B

# Language modality instead of ASCII
uv run python examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-100k \
    --modality language \
    --model Qwen/Qwen3.5-4B
```

Key options:
- `--modality ascii|language` — which observation text to train on (default: ascii)
- `--lora-r 16` — LoRA rank (default: 16)
- `--epochs 3` — training epochs
- `--report-to wandb` — enable wandb logging

## Step 3: Evaluate

After training, merge LoRA adapters and evaluate:

```bash
# Merge adapters into base model
uv run python examples/data_and_finetuning/merge_and_push.py \
    --adapter-path models/sft \
    --base-model Qwen/Qwen2.5-0.5B \
    --push-to-hub rogercc/agentick-qwen-sft

# Evaluate with experiment runner
uv run python -m agentick.experiments.run --config examples/experiments/configs/qwen35_4b_sft_ascii_markov.yaml
```

## Complete Examples

- `examples/data_and_finetuning/collect_oracle_trajectories.py` — collect trajectories from all oracles
- `examples/data_and_finetuning/sft_with_trl.py` — full SFT training script (TRL + LoRA + multi-GPU)
- `examples/data_and_finetuning/merge_and_push.py` — merge LoRA adapters and push to Hub
