# Fine-tuning Models on Agentick Data

Complete guide to collecting trajectories, preparing datasets, and fine-tuning language and vision models on Agentick task demonstrations.

## Overview

Fine-tuning is ideal when:
- You have collected expert demonstrations
- You want to adapt models to your specific task
- You need better task-specific performance
- You want smaller, faster models for production
- You're combining RL and SFT approaches

**Common Approaches:**
- Supervised Fine-Tuning (SFT) on expert trajectories
- Direct Preference Optimization (DPO) on preference pairs
- LoRA for parameter-efficient tuning
- Multi-task fine-tuning for generalization
- Reinforcement Learning from Human Feedback (RLHF)

## Step 1: Trajectory Collection

### Collection Setup

```python
"""Collect expert trajectories for fine-tuning."""

from agentick.data import TrajectoryCollector
import agentick
import json
from pathlib import Path


class ExpertTrajectoryCollector:
    """Collect and manage expert demonstrations."""

    def __init__(self, output_dir="trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.collector = TrajectoryCollector(buffer_size=10000)

    def collect_from_policy(self, policy_fn, task_name, difficulty, n_episodes):
        """Collect trajectories from a policy."""
        env = agentick.make(task_name, difficulty=difficulty, render_mode="language")

        episode_data = []

        for episode_idx in range(n_episodes):
            obs, info = env.reset()
            self.collector.start_episode(metadata={
                "task": task_name,
                "difficulty": difficulty,
                "episode": episode_idx,
            })

            done = False
            step_idx = 0

            while not done:
                # Get action from policy
                action = policy_fn(obs, info)

                # Execute
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Collect step
                self.collector.add_step(obs, action, reward, done, info)
                step_idx += 1

            # End episode
            self.collector.end_episode()

            # Stats
            success = info.get("success", False)
            total_reward = self.collector.trajectories[-1].total_reward
            episode_data.append({
                "task": task_name,
                "difficulty": difficulty,
                "episode": episode_idx,
                "success": success,
                "reward": total_reward,
                "steps": step_idx,
            })

            if (episode_idx + 1) % 10 == 0:
                print(f"Collected {episode_idx + 1}/{n_episodes} episodes")

        env.close()

        return episode_data

    def filter_trajectories(self, min_reward=None, max_length=None, success_only=False):
        """Filter trajectories by criteria."""
        trajectories = self.collector.get_trajectories(
            min_reward=min_reward,
            max_length=max_length,
        )

        if success_only:
            trajectories = [
                t for t in trajectories
                if t.metadata.get("success", False)
            ]

        return trajectories

    def save_trajectories(self):
        """Save collected trajectories."""
        self.collector.save(str(self.output_dir / "trajectories.npz"))

        # Save stats
        stats = self.collector.get_stats()
        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Saved trajectories to {self.output_dir}")


# Example: Collect from oracle agent
def oracle_policy(obs, info):
    """Simple oracle that moves toward goal."""
    import numpy as np

    # Parse observation (assumes language format)
    if "right" in obs.lower() or "east" in obs.lower():
        return 3  # MOVE_RIGHT
    elif "down" in obs.lower() or "south" in obs.lower():
        return 1  # MOVE_DOWN
    elif "left" in obs.lower() or "west" in obs.lower():
        return 2  # MOVE_LEFT
    elif "up" in obs.lower() or "north" in obs.lower():
        return 0  # MOVE_UP
    else:
        return np.random.randint(0, 4)


if __name__ == "__main__":
    collector = ExpertTrajectoryCollector("./expert_trajectories")

    # Collect from oracle
    episode_data = collector.collect_from_policy(
        oracle_policy,
        task_name="GoToGoal-v0",
        difficulty="easy",
        n_episodes=100,
    )

    # Filter good trajectories
    good_trajectories = collector.filter_trajectories(
        min_reward=0.5,
        success_only=True,
    )

    print(f"Total collected: {len(collector.collector.trajectories)}")
    print(f"Good trajectories: {len(good_trajectories)}")

    # Save
    collector.save_trajectories()
```

## Step 2: Export Formats

### JSONL Format

```python
"""Export to JSONL for easy processing."""

from agentick.data import export_to_format
import json

# Export trajectories to JSONL
trajectories = collector.get_trajectories()
export_to_format(
    trajectories,
    output_path="data/trajectories.jsonl",
    format_type="jsonl",
)

# JSONL format - one episode per line
# {"actions": [0, 1, 2, ...], "rewards": [0.1, 0.2, ...], "total_reward": 1.5, ...}

# Read JSONL
with open("data/trajectories.jsonl") as f:
    for line in f:
        episode = json.loads(line)
        print(episode)
```

### HuggingFace Dataset Format

```python
"""Export to HuggingFace Datasets for easy training."""

from agentick.data import export_to_format
from datasets import load_from_disk

# Export
export_to_format(
    trajectories,
    output_path="data/agentick_dataset",
    format_type="hf_dataset",
)

# Load for training
dataset = load_from_disk("data/agentick_dataset")
print(f"Dataset size: {len(dataset)}")

# Use in training
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
```

### Conversation Format (for Chat Models)

```python
"""Export to conversation format for LLM fine-tuning."""

from agentick.data import export_to_format

export_to_format(
    trajectories,
    output_path="data/conversations.jsonl",
    format_type="conversation",
    system_prompt="You are an expert agent solving gridworld tasks efficiently.",
)

# Format:
# {
#   "messages": [
#     {"role": "system", "content": "You are an expert agent..."},
#     {"role": "user", "content": "Observation: ..."},
#     {"role": "assistant", "content": "Action: MOVE_RIGHT"},
#     ...
#   ]
# }
```

## Step 3: Supervised Fine-Tuning (SFT)

### OpenAI Fine-tuning

```python
"""Fine-tune on OpenAI API."""

from openai import OpenAI
import json
from pathlib import Path

client = OpenAI(api_key="your-api-key")


def prepare_openai_format(trajectories):
    """Prepare data in OpenAI chat format."""
    training_data = []

    for traj in trajectories:
        for obs, action in zip(traj.observations, traj.actions):
            training_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert AI agent. "
                                  "Given the current state, choose the best action."
                    },
                    {
                        "role": "user",
                        "content": f"State:\n{obs}"
                    },
                    {
                        "role": "assistant",
                        "content": f"Action: {action}"
                    }
                ]
            })

    return training_data


def finetune_openai(trajectories, model="gpt-3.5-turbo"):
    """Run fine-tuning job."""
    # Prepare data
    training_data = prepare_openai_format(trajectories)

    # Save to file
    with open("training_data.jsonl", "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    # Upload file
    with open("training_data.jsonl", "rb") as f:
        file_response = client.files.create(
            file=f,
            purpose="fine-tune"
        )

    file_id = file_response.id
    print(f"Uploaded training file: {file_id}")

    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 16,
            "learning_rate_multiplier": 0.1,
        }
    )

    print(f"Fine-tuning job created: {job.id}")
    print(f"Status: {job.status}")

    # Monitor job
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"Job {job.id} - Status: {job.status}")

        if job.status in ["succeeded", "failed"]:
            break

        import time
        time.sleep(30)

    if job.status == "succeeded":
        print(f"Fine-tuning succeeded!")
        print(f"Fine-tuned model: ft:gpt-3.5-turbo:{job.id}")
        return job.fine_tuned_model
    else:
        print(f"Fine-tuning failed: {job.error}")
        return None


# Usage
if __name__ == "__main__":
    trajectories = collector.get_trajectories(success_only=True)
    model_id = finetune_openai(trajectories)
```

### HuggingFace Transformers SFT

```python
"""SFT using HuggingFace TRL library."""

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def sft_train_huggingface(dataset_path, model_name="gpt2"):
    """Train using HuggingFace SFT."""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Prepare training config
    training_args = SFTConfig(
        output_dir="./sft_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        formatting_func=format_fn,
        max_seq_length=512,
    )

    trainer.train()

    # Save
    trainer.save_model("./sft_model_final")
    print("Model saved to ./sft_model_final")


def format_fn(examples):
    """Format examples for SFT."""
    return {
        "text": [
            f"State: {obs}\nAction: {action}"
            for obs, action in zip(examples["observations"], examples["actions"])
        ]
    }
```

### LoRA for Efficient Fine-tuning

```python
"""Parameter-efficient fine-tuning with LoRA."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk


def sft_with_lora(dataset_path, base_model="meta-llama/Llama-2-7b-hf"):
    """Fine-tune with LoRA for efficiency."""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Setup LoRA
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling
        target_modules=["q_proj", "v_proj"],  # Which modules to adapt
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained("./lora_adapter")
    print("LoRA adapter saved")

    # To use later:
    # from peft import AutoPeftModelForCausalLM
    # model = AutoPeftModelForCausalLM.from_pretrained("./lora_adapter")
```

## Step 4: Direct Preference Optimization (DPO)

Fine-tune using preference pairs instead of supervised labels.

```python
"""DPO: Learn from preferred vs rejected trajectories."""

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import json


def collect_preference_pairs(trajectories):
    """Create preference pairs from trajectories."""
    successful = [t for t in trajectories if t.metadata.get("success")]
    failed = [t for t in trajectories if not t.metadata.get("success")]

    pairs = []

    for success_traj in successful[:100]:
        for fail_traj in failed[:100]:
            pairs.append({
                "prompt": f"Navigate a gridworld",
                "chosen": _trajectory_to_text(success_traj),
                "rejected": _trajectory_to_text(fail_traj),
            })

    return pairs


def _trajectory_to_text(trajectory):
    """Convert trajectory to text."""
    text = ""
    for obs, action in zip(trajectory.observations, trajectory.actions):
        text += f"State: {obs}\nAction: {action}\n"
    return text


def dpo_train(trajectory_file, base_model="meta-llama/Llama-2-7b-hf"):
    """Train using DPO."""
    # Load trajectories and create preference pairs
    with open(trajectory_file) as f:
        trajectories = [json.loads(line) for line in f]

    pairs = collect_preference_pairs(trajectories)

    # Create dataset
    dataset = Dataset.from_list(pairs)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # DPO config
    training_args = DPOConfig(
        output_dir="./dpo_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        beta=0.1,  # DPO temperature
        save_steps=100,
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./dpo_model_final")
    print("DPO training complete")
```

## Step 5: Evaluation After Fine-tuning

```python
"""Evaluate fine-tuned models."""

import agentick
from agentick.interfaces import LLMAgentInterface
import numpy as np


def evaluate_finetuned_model(model_or_model_id, tasks, n_episodes=10):
    """Evaluate fine-tuned model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    if isinstance(model_or_model_id, str):
        model = AutoModelForCausalLM.from_pretrained(model_or_model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_or_model_id)
    else:
        model = model_or_model_id
        tokenizer = None

    results_by_task = {}

    for task_name in tasks:
        env = agentick.make(task_name, render_mode="language")
        llm_interface = LLMAgentInterface(env)

        task_results = {
            "successes": [],
            "rewards": [],
            "steps": [],
        }

        for episode in range(n_episodes):
            obs, info = env.reset()

            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < 100:
                # Get model prediction
                prompt = llm_interface.format_prompt(obs)

                # Generate action (implementation depends on model type)
                if tokenizer:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=20)
                    action_text = tokenizer.decode(outputs[0])
                else:
                    action_text = model(prompt)  # Placeholder

                # Parse and execute
                action = llm_interface.parse_action(action_text)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            task_results["successes"].append(info.get("success", False))
            task_results["rewards"].append(episode_reward)
            task_results["steps"].append(steps)

        env.close()

        results_by_task[task_name] = {
            "success_rate": np.mean(task_results["successes"]),
            "mean_reward": np.mean(task_results["rewards"]),
            "mean_steps": np.mean(task_results["steps"]),
        }

    return results_by_task


# Evaluation script
if __name__ == "__main__":
    # Compare base and fine-tuned
    base_results = evaluate_finetuned_model(
        "gpt2",
        ["GoToGoal-v0", "KeyDoorPuzzle-v0"],
        n_episodes=10,
    )

    ft_results = evaluate_finetuned_model(
        "./sft_model_final",
        ["GoToGoal-v0", "KeyDoorPuzzle-v0"],
        n_episodes=10,
    )

    print("\nBase model:")
    for task, results in base_results.items():
        print(f"  {task}: success={results['success_rate']:.1%}, "
              f"reward={results['mean_reward']:.2f}")

    print("\nFine-tuned model:")
    for task, results in ft_results.items():
        print(f"  {task}: success={results['success_rate']:.1%}, "
              f"reward={results['mean_reward']:.2f}")

    # Improvement
    print("\nImprovement:")
    for task in base_results:
        improvement = (ft_results[task]["success_rate"] -
                      base_results[task]["success_rate"])
        print(f"  {task}: {improvement:+.1%}")
```

## Complete End-to-End Pipeline

```python
"""Full fine-tuning pipeline."""

import json
from pathlib import Path
from agentick.data import TrajectoryCollector, export_to_format


class FineTuningPipeline:
    """Complete pipeline for fine-tuning."""

    def __init__(self, output_dir="./ft_pipeline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def step1_collect_trajectories(self, policy_fn, tasks, n_episodes=100):
        """Collect expert trajectories."""
        print("\n=== STEP 1: Collecting Trajectories ===")

        collector = TrajectoryCollector()

        for task in tasks:
            print(f"\nCollecting from {task}...")
            import agentick

            env = agentick.make(task, render_mode="language")

            for ep in range(n_episodes):
                obs, info = env.reset()
                collector.start_episode({"task": task})

                done = False
                while not done:
                    action = policy_fn(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    collector.add_step(obs, action, reward, done, info)

                collector.end_episode()

                if (ep + 1) % 25 == 0:
                    print(f"  {ep + 1}/{n_episodes} episodes")

            env.close()

        # Save
        collector.save(str(self.output_dir / "trajectories.npz"))
        print(f"\nSaved {collector.total_episodes} episodes")

        return collector

    def step2_export_datasets(self, collector):
        """Export to different formats."""
        print("\n=== STEP 2: Exporting Datasets ===")

        trajectories = collector.get_trajectories(success_only=True)

        # JSONL
        export_to_format(
            trajectories,
            self.output_dir / "conversations.jsonl",
            format_type="conversation",
        )
        print("Exported conversation format")

        # HuggingFace
        export_to_format(
            trajectories,
            self.output_dir / "hf_dataset",
            format_type="hf_dataset",
        )
        print("Exported HuggingFace format")

    def step3_train_sft(self, model_name="gpt2"):
        """Train SFT model."""
        print("\n=== STEP 3: Training SFT ===")

        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer, SFTConfig

        # Load
        dataset = load_from_disk(str(self.output_dir / "hf_dataset"))
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Train
        training_args = SFTConfig(
            output_dir=str(self.output_dir / "sft_model"),
            num_train_epochs=2,
            per_device_train_batch_size=8,
            learning_rate=2e-4,
            save_steps=100,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            tokenizer=tokenizer,
        )

        trainer.train()

        trainer.save_model(str(self.output_dir / "sft_model_final"))
        print("SFT training complete")

    def step4_evaluate(self, model_path, tasks):
        """Evaluate fine-tuned model."""
        print("\n=== STEP 4: Evaluating ===")

        results = evaluate_finetuned_model(model_path, tasks, n_episodes=5)

        # Save results
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            # Convert numpy to Python types
            results_serializable = {}
            for task, metrics in results.items():
                results_serializable[task] = {
                    k: float(v) for k, v in metrics.items()
                }
            json.dump(results_serializable, f, indent=2)

        print("\nEvaluation results:")
        for task, metrics in results.items():
            print(f"  {task}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")

        return results


# Run pipeline
if __name__ == "__main__":
    pipeline = FineTuningPipeline("./fine_tuning_output")

    # Define simple policy
    def simple_policy(obs, info):
        if "right" in obs.lower():
            return 3
        elif "down" in obs.lower():
            return 1
        else:
            return 0

    # Execute pipeline
    collector = pipeline.step1_collect_trajectories(
        simple_policy,
        ["GoToGoal-v0"],
        n_episodes=50,
    )

    pipeline.step2_export_datasets(collector)
    pipeline.step3_train_sft("gpt2")
    pipeline.step4_evaluate(
        "./fine_tuning_output/sft_model_final",
        ["GoToGoal-v0"],
    )

    print("\n=== Pipeline Complete ===")
```

## Best Practices

1. **Start with diverse data** - Collect from multiple seeds, difficulties
2. **Filter trajectories** - Keep only successful/high-quality examples
3. **Use LoRA first** - Parameter-efficient, fast iteration
4. **Monitor validation** - Watch for overfitting on held-out tasks
5. **Data augmentation** - Rotate, mirror observations
6. **Multi-task training** - Mix data from multiple tasks
7. **Save checkpoints** - Preserve best model during training
8. **Test generalization** - Evaluate on unseen seeds
9. **Track metrics** - Log success rate, reward, efficiency
10. **Compare baselines** - A/B test against base models

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Overfitting on training tasks | Increase data diversity, use multi-task training |
| Poor generalization | Collect more diverse trajectories |
| Training instability | Lower learning rate, use gradient clipping |
| Mode collapse | Add noise, increase batch size |
| High cost with large models | Use LoRA, quantization, or smaller base models |
| Slow training | Use mixed precision, distributed training |

## Resources

- [HuggingFace Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [TRL (Training Reinforcement Learning)](https://github.com/huggingface/trl)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [OpenAI Fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning)
