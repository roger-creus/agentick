"""
Demonstrate difficulty scaling on the same task.

Shows how task complexity increases with difficulty.
Runtime: <15 seconds
"""

import agentick


def run_episode(task, difficulty, seed=42):
    """Run one episode and return stats."""
    env = agentick.make(task, difficulty=difficulty, render_mode="ascii")
    obs, info = env.reset(seed=seed)

    total_reward = 0
    steps = 0

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    env.close()

    return {
        "steps": steps,
        "reward": total_reward,
        "success": info.get("success", False),
        "max_steps": info.get("max_steps", "?"),
        "grid_size": info.get("grid_size", "?"),
    }


def main():
    task = "GoToGoal-v0"
    difficulties = ["easy", "medium", "hard", "expert"]

    print(f"Task: {task}")
    print("=" * 80)
    print(f"{'Difficulty':<12} {'Grid':<10} {'Max Steps':<12} {'Steps':<10} {'Success':<10}")
    print("-" * 80)

    for difficulty in difficulties:
        stats = run_episode(task, difficulty)
        print(
            f"{difficulty:<12} {str(stats['grid_size']):<10} {str(stats['max_steps']):<12} "
            f"{stats['steps']:<10} {str(stats['success']):<10}"
        )

    print()
    print("Note: Higher difficulties have:")
    print("  - Larger grids")
    print("  - More obstacles")
    print("  - Longer max episode length")


if __name__ == "__main__":
    main()
