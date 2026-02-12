"""Human play interface - play tasks with keyboard."""

import agentick


def main():
    # Create environment with human render mode
    env = agentick.make(
        "GoToGoal-v0",
        difficulty="easy",
        render_mode="ascii",
    )

    print("Starting human play mode...")
    print("Note: Human interactive mode requires a display.")
    print("In headless environments, this will run in ASCII mode only.")
    print()

    # Run episode
    obs, info = env.reset(seed=42)
    print("Initial state:")
    print(obs)
    print(f"Valid actions: {info['valid_actions']}")
    print()

    done = False
    total_reward = 0

    while not done:
        # In a real human play interface, you'd capture keyboard input
        # For this demo, we'll just show how to interact
        print("Enter action (0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=PICKUP, q=quit): ")

        try:
            action_input = input()
            if action_input.lower() == "q":
                break
            action = int(action_input)
        except (ValueError, EOFError, KeyboardInterrupt):
            print("Invalid input, using NOOP")
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print("\nState after action:")
        print(obs)
        print(f"Reward: {reward}, Total: {total_reward}")
        print(f"Valid actions: {info['valid_actions']}")

        if terminated or truncated:
            done = True
            print(f"\nEpisode finished! Success: {info.get('success', False)}")

    env.close()


if __name__ == "__main__":
    main()
