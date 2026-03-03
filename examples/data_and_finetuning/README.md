# Data Collection and Fine-tuning

Scripts for collecting multi-task trajectory datasets, exporting to HuggingFace
format, and training models with SFT, behavior cloning, and reinforcement learning.

All scripts support `--tasks` and `--difficulties` for multi-task training.
By default, they train on **all oracle tasks across all four difficulties**.

## Prerequisites

```bash
uv sync                    # base install for data collection scripts
uv sync --extra finetune   # sft_with_trl.py (transformers, torch, trl, peft)
pip install tinker          # tinker_sft/rl scripts (requires TINKER_API_KEY)
```

## Quick Start

```bash
# Single-task quick test
python sft_with_trl.py --tasks GoToGoal-v0 --difficulties easy --n-episodes 5

# Full multi-task training (all tasks x all difficulties)
python sft_with_trl.py

# Train + evaluate + push to HuggingFace Hub
python sft_with_trl.py --eval --push-to-hub user/agentick-sft-model

# Re-run eval later with generated config
python -m agentick.experiments.run --config models/sft/eval_config.yaml
```

## Scripts

### Data Collection

- **collect_oracle_trajectories.py** -- Multi-task oracle data collector.
  Collects optimal demonstrations using `DataCollector` + `get_oracle()` across
  all tasks and difficulties. Exports to HuggingFace format.

  ```bash
  # All tasks, all difficulties
  python collect_oracle_trajectories.py

  # Specific tasks
  python collect_oracle_trajectories.py \
      --tasks GoToGoal-v0 KeyDoorPuzzle-v0 --difficulties easy medium

  # Push dataset to Hub
  python collect_oracle_trajectories.py --push-to-hub user/oracle-data
  ```

- **collect_random_trajectories.py** -- Random-policy trajectories for baselines.
  Supports both single-task (`--env`) and multi-task (`--tasks`) modes.

  ```bash
  # Single task (backward-compatible)
  python collect_random_trajectories.py --env GoToGoal-v0 --num-episodes 50

  # Multi-task
  python collect_random_trajectories.py \
      --tasks GoToGoal-v0 MazeNavigation-v0 --difficulties easy medium

  # All oracle tasks
  python collect_random_trajectories.py --tasks all --num-episodes 20
  ```

- **export_to_huggingface.py** -- Convert collected trajectories to HuggingFace
  format with optional Hub push.

  ```bash
  python export_to_huggingface.py --input-dir trajectories/oracle \
      --format conversation --push-to-hub user/agentick-data
  ```

### Training

- **sft_with_trl.py** -- Full SFT pipeline with `AgentickSFTTrainer`:
  collect data -> train (TRL + LoRA) -> generate eval config YAML -> optionally
  evaluate inline -> optionally push to Hub.

  ```bash
  # Quick test
  python sft_with_trl.py --tasks GoToGoal-v0 --difficulties easy --n-episodes 5

  # Full pipeline with eval and Hub push
  python sft_with_trl.py --eval --push-to-hub user/model \
      --model Qwen/Qwen2.5-7B --lora-r 32 --num-epochs 5

  # Re-run eval with generated config
  python -m agentick.experiments.run --config models/sft/eval_config.yaml
  ```

- **behavior_cloning_training.py** -- Train Nature CNN from pixel observations.
  Pure PyTorch, self-contained evaluation (no ExperimentRunner).

  ```bash
  python behavior_cloning_training.py --tasks GoToGoal-v0 --difficulties easy \
      --n-episodes 20 --eval
  ```

- **tinker_sft_training.py** -- SFT via Tinker's remote LoRA infrastructure.
  Evaluates with an inline `TinkerNonMarkovianReasoner` harness wrapper.

  ```bash
  python tinker_sft_training.py --tasks GoToGoal-v0 --difficulties easy \
      --num-steps 100 --eval
  ```

- **tinker_rl_training.py** -- RL (PPO/REINFORCE) via Tinker on live env.
  Loops over task x difficulty combos.

  ```bash
  python tinker_rl_training.py --tasks GoToGoal-v0 --difficulties easy \
      --num-episodes 50 --loss-fn ppo --eval
  ```

## Shared CLI Arguments

All scripts share common argument groups via `_utils.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks` | all oracle tasks | Task names (nargs="+") |
| `--difficulties` | easy medium hard expert | Difficulty levels |
| `--n-episodes` | 10 | Episodes per task/difficulty |
| `--render-mode` | language | Observation mode |
| `--output-dir` | varies | Output directory |
| `--eval` | off | Run evaluation after training |
| `--eval-episodes` | 5 | Episodes per eval combo |
| `--eval-seeds` | 3 | Number of eval seeds |
| `--harness` | markovian_zero_shot | Harness preset for eval |
| `--push-to-hub` | none | HF Hub repo ID |

## Typical Workflows

### 1. Quick experiment on one task

```bash
python collect_oracle_trajectories.py --tasks GoToGoal-v0 --difficulties easy --n-episodes 10
python sft_with_trl.py --tasks GoToGoal-v0 --difficulties easy --eval
```

### 2. Full multi-task training and eval

```bash
# Train on all tasks (defaults)
python sft_with_trl.py --eval --push-to-hub user/agentick-sft

# Re-run eval with the benchmark runner
python -m agentick.experiments.run --config models/sft/eval_config.yaml
```

### 3. HuggingFace Hub workflow

```bash
# Collect + push dataset
python collect_oracle_trajectories.py --push-to-hub user/oracle-data

# Train + push model
python sft_with_trl.py --push-to-hub user/sft-model --eval

# Others can eval with the same config
python -m agentick.experiments.run --config models/sft/eval_config.yaml
```

### 4. Tinker remote training

```bash
# SFT first, then RL fine-tune
python tinker_sft_training.py --tasks GoToGoal-v0 --eval
python tinker_rl_training.py --tasks GoToGoal-v0 --eval
```
