"""
Compare dense vs sparse reward modes.

Shows how reward shaping affects training.
Runtime: <10 seconds
"""

import agentick


def run_episode(task, difficulty, reward_mode, seed=42):
    """Run one episode and return total reward."""
    env = agentick.make(task, difficulty=difficulty, reward_mode=reward_mode)
    obs, info = env.reset(seed=seed)

    total_reward = 0
    rewards = []

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)

        if terminated or truncated:
            break

    env.close()
    return total_reward, rewards


def main():
    task = "GoToGoal-v0"
    difficulty = "easy"

    print(f"Task: {task}, Difficulty: {difficulty}\n")

    # Sparse rewards
    print("=" * 80)
    print("SPARSE REWARDS (reward only at goal)")
    print("=" * 80)
    total, rewards = run_episode(task, difficulty, "sparse")
    print(f"Total reward: {total}")
    print(f"Reward per step: {rewards[:10]}...")  # Show first 10
    print(f"Non-zero rewards: {sum(1 for r in rewards if r != 0)}/{len(rewards)}")
    print()

    # Dense rewards
    print("=" * 80)
    print("DENSE REWARDS (reward shaping for progress)")
    print("=" * 80)
    total, rewards = run_episode(task, difficulty, "dense")
    print(f"Total reward: {total}")
    print(f"Reward per step: {rewards[:10]}...")  # Show first 10
    print(f"Non-zero rewards: {sum(1 for r in rewards if r != 0)}/{len(rewards)}")
    print()

    print("Note: Dense rewards provide learning signal at every step.")
    print("      Sparse rewards only signal success/failure at the end.")


if __name__ == "__main__":
    main()
