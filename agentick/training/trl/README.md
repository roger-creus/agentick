# TRL SFT Training for Agentick

Train language models to play agentick tasks using Supervised Fine-Tuning (SFT) with HuggingFace TRL.

## Prerequisites

```bash
pip install trl peft datasets transformers
# Or with agentick extras:
uv sync --extra finetune
```

## Quick Start

### 1. Collect Oracle Demonstrations

```python
from agentick.oracles import get_oracle
from agentick.data import DataCollector
import agentick

env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")
oracle = get_oracle("GoToGoal-v0", env)

collector = DataCollector(env, oracle, record_modalities=["language"])
dataset = collector.collect(num_episodes=100, seeds=range(100))
dataset.export_to_huggingface("data/gotogoal_conv/", format="conversation")
env.close()
```

### 2. Train with SFT

```python
from agentick.training.trl.sft import AgentickSFTTrainer

trainer = AgentickSFTTrainer(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_path="data/gotogoal_conv/",
    output_dir="models/gotogoal_sft/",
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    max_length=1024,
)
trainer.train()
```

### 3. Evaluate

```python
agent = trainer.as_agent()

env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = agent.act(obs, info)
    obs, reward, done, trunc, info = env.step(action)
    done = done or trunc

print(f"Success: {info.get('success', False)}")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"Qwen/Qwen2.5-0.5B"` | Any HuggingFace causal LM |
| `use_lora` | `True` | Use LoRA adapters (recommended) |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA alpha |
| `learning_rate` | `2e-5` | Optimizer learning rate |
| `num_train_epochs` | `3` | Training epochs |
| `max_length` | `1024` | Max sequence length |
| `packing` | `True` | Pack multiple examples per sequence |
| `report_to` | `"none"` | Tracker (`"wandb"`, `"tensorboard"`) |

## Training from CollectedDataset

Skip the export step and train directly from a `CollectedDataset`:

```python
trainer = AgentickSFTTrainer.from_collector(
    dataset,
    model_name="Qwen/Qwen2.5-0.5B",
    output_dir="models/sft/",
)
trainer.train()
```

## Multi-Task Training

Collect data from multiple tasks and combine:

```python
from datasets import concatenate_datasets, Dataset

tasks = ["GoToGoal-v0", "MazeNavigation-v0", "KeyDoorPuzzle-v0"]
all_datasets = []

for task_id in tasks:
    env = agentick.make(task_id, difficulty="medium", render_mode="language")
    oracle = get_oracle(task_id, env)
    collector = DataCollector(env, oracle, record_modalities=["language"])
    ds = collector.collect(num_episodes=50, seeds=range(50))
    ds.export_to_huggingface(f"data/{task_id}/", format="conversation")
    all_datasets.append(Dataset.load_from_disk(f"data/{task_id}/"))
    env.close()

combined = concatenate_datasets(all_datasets)
trainer = AgentickSFTTrainer(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset=combined,
    output_dir="models/multi_task_sft/",
)
trainer.train()
```

## Push to Hub

```python
trainer.push_to_hub("your-username/agentick-sft-model")
```
