"""
Custom reward shaping for Agentick tasks.

Demonstrates:
- Implementing custom reward functions
- Reward shaping techniques
- Wrapper-based reward customization
- Intrinsic motivation rewards
"""


import gymnasium as gym
import numpy as np

import agentick


class CustomRewardWrapper(gym.Wrapper):
    """
    Wrapper that applies custom reward shaping.

    This wrapper allows you to modify the reward signal
    without changing the underlying task.
    """

    def __init__(self, env, reward_fn=None, **reward_kwargs):
        """
        Initialize custom reward wrapper.

        Args:
            env: Environment to wrap
            reward_fn: Custom reward function (obs, reward, info) -> new_reward
            reward_kwargs: Additional kwargs for reward function
        """
        super().__init__(env)
        self.reward_fn = reward_fn or self._default_reward
        self.reward_kwargs = reward_kwargs

        # Track additional state for reward computation
        self.prev_obs = None
        self.prev_agent_pos = None
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.prev_agent_pos = info.get("agent_pos", None)
        self.step_count = 0
        return obs, info

    def step(self, action):
        """Step with custom reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply custom reward function
        custom_reward = self.reward_fn(
            obs=obs,
            original_reward=reward,
            info=info,
            prev_obs=self.prev_obs,
            prev_agent_pos=self.prev_agent_pos,
            step_count=self.step_count,
            **self.reward_kwargs,
        )

        # Update state
        self.prev_obs = obs
        self.prev_agent_pos = info.get("agent_pos", None)
        self.step_count += 1

        # Store original reward in info
        info["original_reward"] = reward
        info["shaped_reward"] = custom_reward

        return obs, custom_reward, terminated, truncated, info

    def _default_reward(self, **kwargs):
        """Default: return original reward unchanged."""
        return kwargs["original_reward"]


# === Reward Shaping Functions ===


def exploration_bonus_reward(obs, original_reward, info, prev_agent_pos, **kwargs) -> float:
    """
    Add exploration bonus for visiting new cells.

    Encourages the agent to explore the environment.
    """
    reward = original_reward

    # Get current position
    agent_pos = info.get("agent_pos", None)

    if agent_pos and prev_agent_pos:
        # Check if agent moved to new position
        if agent_pos != prev_agent_pos:
            # Small exploration bonus
            reward += 0.01

    return reward


def distance_shaping_reward(
    obs, original_reward, info, goal_pos=(5, 5), discount=0.1, **kwargs
) -> float:
    """
    Shape reward based on distance to goal.

    Gives denser feedback by rewarding progress toward goal.
    """
    reward = original_reward

    agent_pos = info.get("agent_pos", None)

    if agent_pos:
        # Compute Manhattan distance to goal
        distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

        # Negative reward proportional to distance
        # (encourages getting closer to goal)
        distance_penalty = -distance * discount
        reward += distance_penalty

    return reward


def efficiency_reward(obs, original_reward, step_count, max_steps=100, **kwargs) -> float:
    """
    Penalize inefficient solutions.

    Encourages agents to solve tasks quickly.
    """
    reward = original_reward

    # Efficiency bonus if task completed
    if original_reward > 0:  # Assuming positive reward = success
        # Bonus for completing quickly
        efficiency_bonus = 1.0 - (step_count / max_steps)
        reward += efficiency_bonus * 0.5

    return reward


def curiosity_reward(obs, prev_obs, original_reward, **kwargs) -> float:
    """
    Intrinsic curiosity reward based on observation novelty.

    Rewards the agent for experiencing novel states.
    """
    reward = original_reward

    if obs is not None and prev_obs is not None:
        # Simple novelty measure: check if observation changed
        if isinstance(obs, str) and isinstance(prev_obs, str):
            # For text observations
            if obs != prev_obs:
                reward += 0.02  # Small curiosity bonus
        elif isinstance(obs, np.ndarray) and isinstance(prev_obs, np.ndarray):
            # For array observations
            diff = np.sum(np.abs(obs - prev_obs))
            if diff > 0:
                curiosity_bonus = min(0.05, diff * 0.001)
                reward += curiosity_bonus

    return reward


def sparse_to_dense_reward(
    obs,
    original_reward,
    info,
    prev_agent_pos,
    goal_reached_reward=1.0,
    step_penalty=-0.01,
    **kwargs,
) -> float:
    """
    Convert sparse rewards to dense rewards.

    Adds step penalties and progress-based rewards.
    """
    # Base reward is step penalty
    reward = step_penalty

    # Check if goal reached (assuming original_reward > 0 means success)
    if original_reward > 0:
        reward = goal_reached_reward
    elif original_reward < 0:
        # Preserve negative rewards (failures)
        reward = original_reward

    return reward


# === Example Usage ===


def demo_exploration_bonus():
    """Demo: Exploration bonus reward."""
    print("\n" + "=" * 60)
    print("Demo: Exploration Bonus Reward")
    print("=" * 60)

    # Create base environment
    base_env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")

    # Wrap with exploration bonus
    env = CustomRewardWrapper(base_env, reward_fn=exploration_bonus_reward)

    # Run episode
    obs, info = env.reset(seed=42)
    total_reward = 0
    total_shaped_reward = 0

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += info["original_reward"]
        total_shaped_reward += reward

        if terminated or truncated:
            break

    print(f"Original reward: {total_reward:.3f}")
    print(f"Shaped reward: {total_shaped_reward:.3f}")
    print(f"Exploration bonus: {total_shaped_reward - total_reward:.3f}")

    env.close()


def demo_distance_shaping():
    """Demo: Distance-based reward shaping."""
    print("\n" + "=" * 60)
    print("Demo: Distance-Based Reward Shaping")
    print("=" * 60)

    base_env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")

    # Wrap with distance shaping
    env = CustomRewardWrapper(
        base_env, reward_fn=distance_shaping_reward, goal_pos=(5, 1), discount=0.05
    )

    obs, info = env.reset(seed=42)
    total_reward = 0
    total_shaped_reward = 0

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += info["original_reward"]
        total_shaped_reward += reward

        if terminated or truncated:
            break

    print(f"Original reward: {total_reward:.3f}")
    print(f"Distance-shaped reward: {total_shaped_reward:.3f}")
    print(f"Steps taken: {step + 1}")

    env.close()


def demo_efficiency_reward():
    """Demo: Efficiency-based reward."""
    print("\n" + "=" * 60)
    print("Demo: Efficiency Reward")
    print("=" * 60)

    base_env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")

    env = CustomRewardWrapper(base_env, reward_fn=efficiency_reward, max_steps=100)

    obs, info = env.reset(seed=42)
    total_reward = 0

    # Use random actions for demonstration
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(f"Total reward with efficiency bonus: {total_reward:.3f}")
    print(f"Steps taken: {step + 1}")
    print(f"Success: {info.get('success', False)}")

    env.close()


def demo_multi_reward():
    """Demo: Combining multiple reward components."""
    print("\n" + "=" * 60)
    print("Demo: Multi-Component Reward")
    print("=" * 60)

    def combined_reward(obs, original_reward, info, prev_obs, prev_agent_pos, step_count, **kwargs):
        """Combine multiple reward components."""
        reward = original_reward

        # Component 1: Exploration bonus
        agent_pos = info.get("agent_pos", None)
        if agent_pos and prev_agent_pos and agent_pos != prev_agent_pos:
            reward += 0.01

        # Component 2: Distance shaping
        goal_pos = (5, 1)
        if agent_pos:
            distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            reward -= distance * 0.02

        # Component 3: Efficiency bonus
        if original_reward > 0:
            efficiency_bonus = 1.0 - (step_count / 100)
            reward += efficiency_bonus * 0.3

        return reward

    base_env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
    env = CustomRewardWrapper(base_env, reward_fn=combined_reward)

    obs, info = env.reset(seed=42)
    total_reward = 0

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(f"Combined reward: {total_reward:.3f}")
    print(f"Original reward: {sum([info.get('original_reward', 0)]):.3f}")
    print(f"Steps: {step + 1}")

    env.close()


def main():
    """Run all custom reward demonstrations."""
    print("\n" + "=" * 70)
    print("CUSTOM REWARD SHAPING EXAMPLES")
    print("=" * 70)

    demo_exploration_bonus()
    demo_distance_shaping()
    demo_efficiency_reward()
    demo_multi_reward()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. Use CustomRewardWrapper to modify rewards without changing tasks")
    print("2. Reward shaping can make sparse tasks learnable")
    print("3. Combine multiple reward components for complex behaviors")
    print("4. Always track original_reward to evaluate true performance")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
