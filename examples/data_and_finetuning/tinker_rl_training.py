"""
Tinker RL training pipeline.

Trains an LLM with PPO/REINFORCE on live agentick environment interactions
using Tinker's remote LoRA training infrastructure.

Can warm-start from a Tinker SFT checkpoint (run tinker_sft_training.py first).

Runtime: ~15 minutes (remote GPU)
Requires: pip install tinker; export TINKER_API_KEY="..."

Usage:
    uv run python examples/data_and_finetuning/tinker_rl_training.py
"""


def main():
    print("Tinker RL Training Pipeline")
    print("=" * 80)

    try:
        from agentick.training.tinker.rl import TinkerRLTrainer
    except ImportError:
        print("ERROR: Tinker is not installed.")
        print("Install with: pip install tinker")
        print("Then set: export TINKER_API_KEY='your-api-key'")
        return

    # Train with PPO on GoToGoal
    print("\nTraining with PPO on GoToGoal-v0...")
    trainer = TinkerRLTrainer(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        task_id="GoToGoal-v0",
        difficulty="easy",
        rank=32,
        loss_fn="ppo",
        output_dir="models/tinker_rl/",
        render_mode="language",
    )

    metrics = trainer.train(
        num_episodes=50,
        learning_rate=1e-5,
    )

    # Print results
    import numpy as np

    final_rewards = metrics["episode_rewards"][-10:]
    final_successes = metrics["episode_successes"][-10:]
    print(f"\nFinal 10 episodes:")
    print(f"  Avg reward: {np.mean(final_rewards):.3f}")
    print(f"  Success rate: {np.mean(final_successes):.0%}")

    # Evaluate trained agent
    print("\nEvaluating RL-trained agent...")
    import agentick

    agent = trainer.as_agent()

    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
    successes = 0
    for episode in range(5):
        obs, info = env.reset(seed=200 + episode)
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
    print("TINKER RL PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nTip: For better results, warm-start from SFT:")
    print("  1. Run tinker_sft_training.py first")
    print("  2. Then run RL training on the SFT checkpoint")


if __name__ == "__main__":
    main()
