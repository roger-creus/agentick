# Data Collection And Fine-Tuning

Scripts for collecting oracle trajectories, fine-tuning language models with
TRL + LoRA, and merging adapters for evaluation.

## Prerequisites

```bash
uv sync --extra finetune
```

## Collect Oracle Trajectories

`collect_oracle_trajectories.py` runs oracle agents on task-difficulty pairs
and writes per-step rows with `task`, `episode_id`, `difficulty`, `step`,
`ascii_render`, `language_render`, `action_name`, `action_int`, `reward`, and
`done`.

```bash
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
    --n-episodes 10 \
    --output-dir trajectories/oracle

uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
    --tasks GoToGoal-v0 KeyDoorPuzzle-v0 \
    --difficulties easy medium \
    --n-episodes 5 \
    --push-to-hub your-org/agentick-oracle-trajectories
```

## Fine-Tune With TRL

The SFT script loads a local dataset or HuggingFace dataset, converts each row
to the same chat format used by the evaluation harness, and saves a LoRA adapter
to `--output-dir`.

```bash
uv run python examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-120k \
    --model Qwen/Qwen3.5-4B \
    --modality ascii \
    --output-dir models/qwen35-4b-sft-agentick

torchrun --standalone --nnodes=1 --nproc_per_node 8 \
    examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-120k \
    --model Qwen/Qwen3.5-4B \
    --modality ascii \
    --output-dir models/qwen35-4b-sft-agentick
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | HuggingFace dataset ID or local path |
| `--modality` | `ascii` | Observation modality: `ascii` or `language` |
| `--model` | `Qwen/Qwen3.5-4B` | Base model name or local path |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--batch-size` | `4` | Per-device batch size |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--max-seq-length` | `2048` | Maximum sequence length |
| `--lora-r` | `16` | LoRA rank |
| `--packing` | off | Enable sequence packing |
| `--report-to` | `none` | Reporting backend, for example `wandb` or `tensorboard` |

## Merge And Push

```bash
uv run python examples/data_and_finetuning/merge_and_push.py \
    --base-model Qwen/Qwen3.5-4B \
    --adapter-dir models/qwen35-4b-sft-agentick \
    --push-to-hub your-org/agentick-qwen35-4b-sft
```

Use `--skip-upload` to save the merged model locally without pushing to the Hub.

## Config-Driven Runs

The training configs under `examples/experiments/configs/qwen35_4b_sft_train_*`
can be run through:

```bash
examples/data_and_finetuning/run_sft_from_config.sh \
    examples/experiments/configs/qwen35_4b_sft_train_ascii_pilot.yaml \
    results/sft-pilot
```
