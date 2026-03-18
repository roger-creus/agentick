# Fine-Tuning

Fine-tune language models on expert trajectories from Agentick oracles.

## Pipeline Overview

1. Collect oracle trajectories → 2. Export to HuggingFace format → 3. SFT → 4. Evaluate

## Step 1: Collect Trajectories

```python
import agentick
from agentick.data.collector import DataCollector
from agentick.oracles import get_oracle

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
oracle = get_oracle("GoToGoal-v0", env)
collector = DataCollector(env, oracle, record_modalities=["language"])

dataset = collector.collect(num_episodes=100, seeds=range(100))
dataset.save("data/oracle_trajectories/")
```

## Step 2: Export

```python
dataset.export_to_huggingface("data/hf_dataset/", format="conversation")
# Formats: "conversation" (chat), "sft" (text/label), "instruction" (instruction tuning)
```

## Step 3: Fine-Tune

### AgentickSFTTrainer (recommended)

```python
from agentick.training.trl.sft import AgentickSFTTrainer

trainer = AgentickSFTTrainer(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_path="data/hf_dataset/",
    output_dir="models/sft/",
    use_lora=True,
    lora_r=16,
    num_train_epochs=3,
    learning_rate=2e-5,
)
trainer.train()
agent = trainer.as_agent()
```

### Direct TRL

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
dataset = load_dataset("data/hf_dataset")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./sft_output", num_train_epochs=3, max_seq_length=512),
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()
```

## Step 4: Evaluate

Use the finetuned model as an agent — see `examples/llm/huggingface_local_agent.py` for the pattern.

## Complete Examples

- `examples/data_and_finetuning/` — end-to-end collect → train → evaluate scripts
- `agentick/training/trl/sft.py` — AgentickSFTTrainer source
