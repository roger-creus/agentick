"""Programmatic bot example using BotInterface."""

import agentick
from agentick.interfaces.bot_interface import BotInterface


def main():
    # Create environment
    env = agentick.make(
        "GoToGoal-v0",
        difficulty="easy",
        render_mode="ascii",
    )

    # Create bot interface
    bot = BotInterface(env)

    # Run episode
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0

    print("Initial state:")
    print(obs)
    print()

    while not done:
        # Use bot interface to get information
        agent_pos = bot.get_agent_position()
        goals = bot.get_goal_positions()

        print(f"Agent at: {agent_pos}")
        print(f"Goals at: {goals}")

        # Compute shortest path to first goal
        if goals:
            path = bot.get_shortest_path(goals[0])
            if path and len(path) > 1:
                next_pos = path[1]
                print(f"Path to goal: {len(path)} steps")
                print(f"Next position: {next_pos}")

                # Determine action to move to next position
                dx = next_pos[0] - agent_pos[0]
                dy = next_pos[1] - agent_pos[1]

                if dy < 0:
                    action = 1  # MOVE_UP
                elif dy > 0:
                    action = 2  # MOVE_DOWN
                elif dx < 0:
                    action = 3  # MOVE_LEFT
                elif dx > 0:
                    action = 4  # MOVE_RIGHT
                else:
                    action = 0  # NOOP
            else:
                print("No path found!")
                action = 0
        else:
            print("No goals found!")
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Took action, reward: {reward}")
        print()

        if terminated or truncated:
            done = True

    print(f"Episode finished! Total reward: {total_reward}")
    print(f"Success: {info.get('success', False)}")
    env.close()


if __name__ == "__main__":
    main()
