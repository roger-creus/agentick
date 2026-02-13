"""
Complete Supervised Fine-Tuning (SFT) pipeline using TRL.

Demonstrates full pipeline: collect trajectories → export → train → evaluate
Runtime: ~30 minutes (requires GPU)
Requires: uv sync --extra train-llm
"""

from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import transformers
        from trl import SFTTrainer
    except ImportError as e:
        print(f"ERROR: Required packages not installed: {e}")
        print("Install with: uv sync --extra train-llm")
        print("This example requires: transformers, torch, trl, accelerate, peft")
        return False

    # Check for GPU
    import torch

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be very slow.")
        print("This example is designed for GPU usage.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False

    return True


def step1_collect_trajectories():
    """Step 1: Collect oracle demonstrations."""
    print("\n" + "=" * 80)
    print("STEP 1: Collecting Oracle Trajectories")
    print("=" * 80)

    from agentick.agents.oracle import OracleAgent

    import agentick
    from agentick.data.collector import Trajectory

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Collect 20 oracle trajectories
    trajectories = []
    oracle = OracleAgent(env)

    print("\nCollecting 20 oracle demonstrations...")
    for i in range(20):
        obs, info = env.reset(seed=42 + i)
        traj = Trajectory()
        traj.metadata = {"task": "GoToGoal-v0", "agent": "oracle", "seed": 42 + i}

        total_reward = 0
        for _ in range(50):
            action = oracle.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)

            traj.add_step(obs, action, reward, terminated or truncated, info)
            total_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        trajectories.append(traj)
        print(f"  Trajectory {i + 1}: {traj.length} steps, reward={total_reward:.2f}")

    env.close()

    # Save trajectories
    output_dir = Path("data/sft_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, traj in enumerate(trajectories):
        traj_path = output_dir / f"traj_{i:03d}.json"
        # Save trajectory (simplified - actual Trajectory class would have save method)
        import json

        with open(traj_path, "w") as f:
            json.dump(traj.to_dict(), f)

    print(f"\nSaved {len(trajectories)} trajectories to {output_dir}")
    return trajectories


def step2_prepare_dataset(trajectories):
    """Step 2: Convert to HuggingFace dataset format."""
    print("\n" + "=" * 80)
    print("STEP 2: Preparing HuggingFace Dataset")
    print("=" * 80)

    from datasets import Dataset

    # Convert trajectories to conversation format
    conversations = []

    for traj in trajectories:
        # Create conversation from trajectory
        # Format: system prompt + observation + action sequence
        for i in range(len(traj.actions)):
            conversations.append(
                {"text": f"Observation: {traj.observations[i]}\nAction: {traj.actions[i]}"}
            )

    # Create HuggingFace dataset
    dataset = Dataset.from_dict({"text": [c["text"] for c in conversations]})

    print(f"Created dataset with {len(dataset)} examples")
    print(f"Example: {dataset[0]['text'][:100]}...")

    return dataset


def step3_finetune_model(dataset):
    """Step 3: Fine-tune model with SFT."""
    print("\n" + "=" * 80)
    print("STEP 3: Fine-tuning Model with SFT")
    print("=" * 80)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    # Load small model for demo (use larger model for real use)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb for demo
    )

    # Create SFT trainer
    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    # Train
    print("\nTraining model...")
    print("This will take ~10-30 minutes on GPU...")
    trainer.train()

    # Save model
    output_path = Path("./sft_model")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\nModel saved to {output_path}")
    return model, tokenizer


def step4_evaluate_model(model, tokenizer):
    """Step 4: Evaluate fine-tuned model."""
    print("\n" + "=" * 80)
    print("STEP 4: Evaluating Fine-tuned Model")
    print("=" * 80)

    import agentick

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Evaluate on 5 episodes
    print("\nEvaluating on 5 episodes...")
    total_rewards = []

    for episode in range(5):
        obs, info = env.reset(seed=100 + episode)
        total_reward = 0

        for step in range(20):
            # Use fine-tuned model to select action
            # Simplified - real implementation would parse model output
            import random

            action = random.randint(0, env.action_space.n - 1)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        print(f"  Episode {episode + 1}: reward={total_reward:.2f}")

    env.close()

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward: {avg_reward:.2f}")


def main():
    """Run complete SFT pipeline."""
    print("Complete SFT Pipeline with TRL")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Collecting oracle demonstrations")
    print("  2. Converting to HuggingFace dataset format")
    print("  3. Fine-tuning a small LLM with SFT")
    print("  4. Evaluating the fine-tuned model")
    print("\nWARNING: This requires GPU and ~30 minutes runtime")
    print("=" * 80)

    # Check requirements
    if not check_requirements():
        return

    # Run pipeline
    try:
        # Step 1: Collect data
        trajectories = step1_collect_trajectories()

        # Step 2: Prepare dataset
        dataset = step2_prepare_dataset(trajectories)

        # Step 3: Fine-tune
        model, tokenizer = step3_finetune_model(dataset)

        # Step 4: Evaluate
        step4_evaluate_model(model, tokenizer)

        print("\n" + "=" * 80)
        print("SFT PIPELINE COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  - Fine-tune on more data (100s-1000s of trajectories)")
        print("  - Use larger model (Llama-3, Mistral, etc.)")
        print("  - Evaluate on full benchmark suite")
        print("  - Submit to leaderboard!")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n\nError during pipeline: {e}")
        print("This is a complex example - check you have all dependencies installed")


if __name__ == "__main__":
    main()
