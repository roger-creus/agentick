"""RL training example (stub loop)."""

import agentick
from agentick.benchmark.baselines import RandomAgent


def main():
    """Simple RL training loop stub."""
    # Create environment
    env = agentick.make(
        "GoToGoal-v0",
        difficulty="easy",
        render_mode="rgb_array",  # Pixel observations for RL
        reward_mode="dense",  # Dense rewards for faster learning
    )

    # Create agent (random baseline for demo)
    agent = RandomAgent(seed=42)

    # Training loop
    num_episodes = 10
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        episode_reward = 0

        while not done:
            # Select action (in real RL, use policy network)
            valid_actions = env.get_valid_actions()
            valid_indices = [i for i, v in enumerate(valid_actions) if v]
            action = agent.act(obs, valid_indices)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # In real RL, you would:
            # - Store transition in replay buffer
            # - Update policy network
            # - Log metrics to wandb

            if terminated or truncated:
                done = True

        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")

    env.close()
    print("\nTraining complete! (This is just a stub - replace with actual RL algorithm)")


if __name__ == "__main__":
    main()
