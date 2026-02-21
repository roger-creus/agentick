"""
Behavior Cloning training pipeline.

Trains a CNN (Nature CNN architecture) from pixel observations to actions
using oracle demonstrations. Pure PyTorch, no HuggingFace dependencies.

Runtime: ~10 minutes (GPU recommended)
Requires: torch, Pillow

Usage:
    uv run python examples/data_and_finetuning/behavior_cloning_training.py
"""

from pathlib import Path


def main():
    print("Behavior Cloning Training Pipeline")
    print("=" * 80)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        return

    import agentick
    from agentick.data import DataCollector
    from agentick.oracles import get_oracle
    from agentick.training.behavior_cloning import BehaviorCloningTrainer

    # Step 1: Collect oracle demonstrations with pixel observations
    print("\nStep 1: Collecting oracle pixel demonstrations...")
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
    oracle = get_oracle("GoToGoal-v0", env)

    collector = DataCollector(env, oracle, record_modalities=["rgb_array"])
    dataset = collector.collect(num_episodes=20, seeds=range(20))

    for i, traj in enumerate(dataset.trajectories):
        success = traj.infos[-1].get("success", False) if traj.infos else False
        print(f"  Episode {i + 1}: {traj.length} steps, success={success}")

    # Save dataset
    data_path = Path("trajectories/oracle_pixels/")
    dataset.save(data_path)
    print(f"\nSaved pixel data to {data_path}")
    env.close()

    # Step 2: Train behavior cloning model
    print("\nStep 2: Training BC model...")
    trainer = BehaviorCloningTrainer(
        dataset_path=data_path,
        output_dir="models/bc_gotogoal/",
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-3,
    )
    metrics = trainer.train()
    print(f"\nTraining complete. Final loss: {metrics.get('final_loss', 'N/A')}")

    # Step 3: Evaluate
    print("\nStep 3: Evaluating BC agent...")
    agent = trainer.as_agent()

    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
    successes = 0
    for episode in range(5):
        obs, info = env.reset(seed=100 + episode)

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
    print("BC PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
