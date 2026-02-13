"""
Simplest possible Agentick example: make, reset, step, render.

Shows the basic environment interaction loop.
Runtime: <5 seconds
"""

import agentick


def main():
    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")

    # Reset environment
    obs, info = env.reset(seed=42)
    print("Initial observation:")
    print(obs)
    print()

    # Run episode
    total_reward = 0
    for step in range(20):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Render
        print(f"\nStep {step + 1}:")
        print(f"Action: {action}, Reward: {reward}")
        print(obs)

        if terminated or truncated:
            print(f"\nEpisode finished after {step + 1} steps!")
            print(f"Total reward: {total_reward}")
            print(f"Success: {info.get('success', False)}")
            break

    env.close()


if __name__ == "__main__":
    main()
