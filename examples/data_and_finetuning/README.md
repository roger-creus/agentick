# Data Collection and Fine-tuning

Scripts for collecting multi-task oracle trajectory datasets, pushing to HuggingFace,
and fine-tuning LLMs with SFT (LoRA) on the collected data.

## Prerequisites

```bash
uv sync                    # base install for data collection
uv sync --extra finetune   # sft_with_trl.py (transformers, torch, trl, peft)
```

## Pipeline

### Step 1: Collect oracle trajectories

`collect_oracle_trajectories.py` runs oracle agents on all task-difficulty pairs
and produces a flat per-step HuggingFace dataset with columns: `task`, `episode_id`,
`difficulty`, `step`, `ascii_render`, `language_render`, `image` (isometric),
`action_name`, `action_int`, `reward`, `done`.

```bash
# All tasks, all difficulties, 10 episodes each -> push to HF
python collect_oracle_trajectories.py \
    --n-episodes 10 --push-to-hub user/agentick-oracle-trajectories

# Specific tasks
python collect_oracle_trajectories.py \
    --tasks GoToGoal-v0 KeyDoorPuzzle-v0 --difficulties easy medium \
    --n-episodes 5 --push-to-hub user/agentick-oracle-trajectories
```

### Step 2: SFT with LoRA

`sft_with_trl.py` loads the HF dataset, converts rows to chat format matching the
eval harness prompts (system + observation + action), and trains with TRL + LoRA.
After training it merges LoRA into the base model and uploads the merged checkpoint.

The default modality is **ascii**. Use `--modality language` for language observations.

```bash
accelerate launch --num_processes 8 \
    examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories \
    --model Qwen/Qwen3.5-4B \
    --modality ascii \
    --report-to wandb \
    --wandb-project agentick-sft \
    --output-dir $HF_HOME \
    --push-to-hub rogercc/agentick-qwen35-4b-sft-ascii 
```

### Step 3: Evaluate

The merged model works directly with the existing eval configs — just change the
`model:` field:

```yaml
# e.g. configs/qwen35_4b_ascii_markov.yaml
agent:
  hyperparameters:
    model: user/agentick-qwen35-4b-sft-ascii  # <- your merged model
```

Then run:

```bash
python -m agentick.experiments.run --config path/to/config.yaml
```

## SFT Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | HF dataset ID or local path |
| `--modality` | ascii | Observation modality (`ascii` or `language`) |
| `--model` | Qwen/Qwen3.5-4B | Base model name |
| `--epochs` | 3 | Training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--batch-size` | 4 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--max-seq-length` | 2048 | Max sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--no-lora` | off | Full fine-tune instead of LoRA |
| `--packing` | off | Sequence packing |
| `--gradient-checkpointing` | on | Gradient checkpointing |
| `--report-to` | none | `wandb` or `tensorboard` |
| `--push-to-hub` | none | Merge LoRA + push to HF Hub |

## Other Training Scripts

- **behavior_cloning_training.py** -- Train Nature CNN from pixel observations.
- **tinker_sft_training.py** -- SFT via Tinker's remote LoRA infrastructure.
- **tinker_rl_training.py** -- RL (PPO/REINFORCE) via Tinker on live env.
