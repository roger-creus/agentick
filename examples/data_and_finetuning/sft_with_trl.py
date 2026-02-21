"""
Complete Supervised Fine-Tuning (SFT) pipeline using AgentickSFTTrainer.

Demonstrates full pipeline: collect oracle data -> train with TRL -> evaluate
Runtime: ~30 minutes (requires GPU)
Requires: uv sync --extra finetune

Usage:
    uv run python examples/data_and_finetuning/sft_with_trl.py
"""

from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from trl import SFTTrainer  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Required packages not installed: {e}")
        print("Install with: uv sync --extra finetune")
        print("This example requires: transformers, torch, trl, peft")
        return False

    import torch

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False

    return True


def step1_collect_data():
    """Step 1: Collect oracle demonstrations."""
    print("\n" + "=" * 80)
    print("STEP 1: Collecting Oracle Demonstrations")
    print("=" * 80)

    import agentick
    from agentick.data import DataCollector
    from agentick.oracles import get_oracle

    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
    oracle = get_oracle("GoToGoal-v0", env)

    collector = DataCollector(env, oracle, record_modalities=["language"])
    print("\nCollecting 20 oracle demonstrations...")
    dataset = collector.collect(num_episodes=20, seeds=range(20))

    for i, traj in enumerate(dataset.trajectories):
        success = traj.infos[-1].get("success", False) if traj.infos else False
        print(f"  Episode {i + 1}: {traj.length} steps, reward={traj.total_reward:.2f}, success={success}")

    env.close()

    # Export to HuggingFace conversation format
    hf_path = Path("trajectories/sft_conv/")
    dataset.export_to_huggingface(hf_path, format="conversation")
    print(f"\nExported HF dataset to {hf_path}")

    return hf_path


def step2_train(hf_path: Path):
    """Step 2: Fine-tune with AgentickSFTTrainer."""
    print("\n" + "=" * 80)
    print("STEP 2: Fine-tuning with AgentickSFTTrainer")
    print("=" * 80)

    from agentick.training.trl.sft import AgentickSFTTrainer

    trainer = AgentickSFTTrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        dataset_path=hf_path,
        output_dir="models/sft_gotogoal/",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        max_length=1024,
        report_to="none",
    )

    print("\nTraining model (this may take 10-30 minutes on GPU)...")
    metrics = trainer.train()
    print(f"\nTraining complete. Metrics: {metrics}")

    return trainer


def step3_evaluate(trainer):
    """Step 3: Evaluate the fine-tuned model."""
    print("\n" + "=" * 80)
    print("STEP 3: Evaluating Fine-tuned Model")
    print("=" * 80)

    import agentick

    agent = trainer.as_agent()

    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    print("\nEvaluating on 5 episodes...")
    successes = 0
    for episode in range(5):
        obs, info = env.reset(seed=100 + episode)
        agent.reset(obs, info)

        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        success = info.get("success", False)
        successes += success
        print(f"  Episode {episode + 1}: reward={total_reward:.2f}, success={success}")

    env.close()
    print(f"\nSuccess rate: {successes}/5 = {successes / 5:.0%}")


def main():
    """Run complete SFT pipeline."""
    print("Complete SFT Pipeline with AgentickSFTTrainer")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Collecting oracle demonstrations with DataCollector")
    print("  2. Training with AgentickSFTTrainer (TRL + LoRA)")
    print("  3. Evaluating the fine-tuned model as an agentick agent")
    print("=" * 80)

    if not check_requirements():
        return

    try:
        hf_path = step1_collect_data()
        trainer = step2_train(hf_path)
        step3_evaluate(trainer)

        print("\n" + "=" * 80)
        print("SFT PIPELINE COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  - Collect more data (100s-1000s of trajectories)")
        print("  - Use larger model (Qwen2.5-7B, Llama-3, etc.)")
        print("  - Evaluate on full benchmark suite")
        print("  - Push to Hub: trainer.push_to_hub('user/model-name')")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n\nError during pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
