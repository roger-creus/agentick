# Data Collection and Fine-tuning

Scripts for collecting trajectory datasets, exporting them to HuggingFace
format, and training models with supervised fine-tuning (SFT), behavior
cloning, and reinforcement learning.

## Prerequisites

```bash
uv sync                    # base install for data collection scripts
uv sync --extra finetune   # sft_with_trl.py (transformers, torch, trl, peft)
uv sync --extra viz        # record_videos.py (Pillow)
pip install tinker          # tinker_sft/rl scripts (requires TINKER_API_KEY)
```

## Scripts

### Data Collection (CPU only)

- **collect_oracle_trajectories.py** -- Collect optimal demonstrations using the
  oracle agent and the high-level `DataCollector` API. Saves trajectories and
  exports them to HuggingFace conversation format for SFT.
- **collect_random_trajectories.py** -- Collect random-policy trajectories and
  save them as per-episode JSON files with summary statistics. Accepts CLI
  arguments for task, episodes, difficulty, and render mode.
- **collect_trajectories_finetune.py** -- Run the oracle on multiple tasks and
  export both raw JSONL trajectories and chat-style conversation JSONL for LLM
  fine-tuning.
- **export_to_huggingface.py** -- Load previously collected trajectory JSON
  files and convert them to JSONL conversation format suitable for
  `datasets.load_dataset`. Run a collection script first.
- **record_videos.py** -- Record oracle episodes as animated GIF files using
  pixel rendering. Requires Pillow (`uv sync --extra viz`).

### Training (GPU recommended)

- **sft_with_trl.py** -- End-to-end SFT pipeline: collect oracle data, train
  with `AgentickSFTTrainer` (TRL + LoRA on Qwen2.5-0.5B), and evaluate the
  fine-tuned model. Requires GPU and `uv sync --extra finetune`.
- **behavior_cloning_training.py** -- Train a Nature CNN policy from pixel
  observations using `BehaviorCloningTrainer`. Pure PyTorch, no HuggingFace
  dependencies. Requires `torch`.
- **tinker_sft_training.py** -- Fine-tune an LLM using Tinker's remote LoRA
  infrastructure. Requires `pip install tinker` and `TINKER_API_KEY`.
- **tinker_rl_training.py** -- Train an LLM with PPO/REINFORCE on live
  environment interactions via Tinker. Can warm-start from an SFT checkpoint.
  Requires `pip install tinker` and `TINKER_API_KEY`.

## Running

```bash
# Data collection
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py
uv run python examples/data_and_finetuning/collect_random_trajectories.py --num-episodes 50
uv run python examples/data_and_finetuning/collect_trajectories_finetune.py --tasks GoToGoal-v0
uv run python examples/data_and_finetuning/export_to_huggingface.py
uv run python examples/data_and_finetuning/record_videos.py

# Training (GPU recommended)
uv run python examples/data_and_finetuning/sft_with_trl.py
uv run python examples/data_and_finetuning/behavior_cloning_training.py
uv run python examples/data_and_finetuning/tinker_sft_training.py
uv run python examples/data_and_finetuning/tinker_rl_training.py
```

## Typical Workflow

1. Collect oracle trajectories with `collect_oracle_trajectories.py`.
2. Export to HuggingFace format with `export_to_huggingface.py`.
3. Fine-tune with `sft_with_trl.py` or `behavior_cloning_training.py`.
4. Optionally continue with RL via `tinker_rl_training.py`.
