# Fine-Tuning Agents

Fine-tune language models on Agentick using Supervised Fine-Tuning (SFT).

## Overview

Pipeline:
1. Collect trajectories (oracle/human)
2. Export to training format
3. Supervised fine-tuning with TRL/HuggingFace
4. Evaluate fine-tuned model

## Step 1: Collect Trajectories

```python
from agentick.data import TrajectoryCollector
from agentick.benchmark.baselines import OracleAgent

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
collector = TrajectoryCollector(env)
oracle = OracleAgent(env)

trajectories = collector.collect_trajectories(agent=oracle, n_episodes=100, seed=42)
collector.save_trajectories(trajectories, "data/oracle_trajectories.json")
```

**Examples**: `examples/data/collect_oracle_trajectories.py`, `collect_random_trajectories.py`

## Step 2: Export Dataset

```python
from agentick.data import export_to_format
from datasets import load_dataset

export_to_format(
    trajectories,
    output_path="data/hf_dataset",
    format_type="hf_dataset",
)

dataset = load_dataset("data/hf_dataset")
# Dataset({train: ..., test: ...})
```

Dataset format: `{"text": "<observation>", "label": "<action>"}`

**Example**: `examples/data/export_to_huggingface.py`

## Step 3: Supervised Fine-Tuning

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("data/hf_dataset")

training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    max_seq_length=512
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text"
)

trainer.train()
trainer.save_model("./sft_model")
```

**Example**: `examples/data/sft_with_trl.py`

## Step 4: Evaluate

For a complete example of loading and evaluating a finetuned model, see `examples/llm/huggingface_local_agent.py`.

To use your finetuned model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
# Then follow the pattern in examples/llm/huggingface_local_agent.py
```

### Compare Before/After

```python
def evaluate_model(model, tokenizer, env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_return, done = 0, False

        while not done:
            prompt = llm_interface.format_prompt(obs, task_description="Navigate")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20)
            action_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            action = llm_interface.parse_action(action_text)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

    return sum(returns) / len(returns)

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
finetuned_model = AutoModelForCausalLM.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")

base_score = evaluate_model(base_model, tokenizer, env)
finetuned_score = evaluate_model(finetuned_model, tokenizer, env)

print(f"Base: {base_score:.2f}")
print(f"Fine-tuned: {finetuned_score:.2f}")
print(f"Improvement: {finetuned_score - base_score:.2f}")
```

## Complete Pipeline

Full pipeline: `examples/data/sft_with_trl.py`

Runs:
1. Collect oracle trajectories
2. Export to HuggingFace format
3. Fine-tune model with TRL
4. Evaluate fine-tuned model
5. Compare before/after

Run with:
```bash
uv run examples/data/sft_with_trl.py
```

## Dataset Formats

```python
# SFT (default)
export_to_huggingface(trajectories, output_path, format="sft")
# {"text": "<obs>", "label": "<action>"}

# Chat
export_to_huggingface(trajectories, output_path, format="chat")
# {"messages": [{"role": "user", "content": "..."}, ...]}

# Instruction
export_to_huggingface(trajectories, output_path, format="instruction")
# {"instruction": "<task>", "input": "<obs>", "output": "<action>"}
```

See `examples/data/` for complete examples.
