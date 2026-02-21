"""
Tinker SFT training pipeline.

Fine-tunes an LLM on oracle demonstrations using Tinker's remote LoRA
training infrastructure. Requires a Tinker account and API key.

Runtime: ~5 minutes (remote GPU)
Requires: pip install tinker; export TINKER_API_KEY="..."

Usage:
    uv run python examples/data_and_finetuning/tinker_sft_training.py
"""

from pathlib import Path


def main():
    print("Tinker SFT Training Pipeline")
    print("=" * 80)

    try:
        from agentick.training.tinker.sft import TinkerSFTTrainer
    except ImportError:
        print("ERROR: Tinker is not installed.")
        print("Install with: pip install tinker")
        print("Then set: export TINKER_API_KEY='your-api-key'")
        return

    import agentick
    from agentick.data import DataCollector
    from agentick.oracles import get_oracle

    # Step 1: Collect oracle demonstrations
    print("\nStep 1: Collecting oracle demonstrations...")
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
    oracle = get_oracle("GoToGoal-v0", env)

    collector = DataCollector(env, oracle, record_modalities=["language"])
    dataset = collector.collect(num_episodes=50, seeds=range(50))

    hf_path = Path("trajectories/tinker_sft_conv/")
    dataset.export_to_huggingface(hf_path, format="conversation")
    print(f"Exported {len(dataset.trajectories)} trajectories to {hf_path}")
    env.close()

    # Step 2: Train with Tinker
    print("\nStep 2: Training with Tinker SFT...")
    trainer = TinkerSFTTrainer(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_path=hf_path,
        rank=32,
        output_dir="models/tinker_sft/",
    )
    metrics = trainer.train(num_steps=100, learning_rate=1e-4)
    print(f"\nTraining complete. Final loss: {metrics.get('final_loss', 'N/A')}")

    # Step 3: Evaluate
    print("\nStep 3: Evaluating Tinker-trained agent...")
    agent = trainer.as_agent()

    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
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

    print("\n" + "=" * 80)
    print("TINKER SFT PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
