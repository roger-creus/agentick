"""
OpenAI GPT-4o agent with text observations.

Demonstrates using GPT-4o API for zero-shot task solving via the agent harness.
For benchmarking at scale, use the experiment runner with a YAML config instead.

Runtime: ~30 seconds per episode (API calls)
Cost: ~$0.01-0.05 per episode
"""

import os

from dotenv import load_dotenv

import agentick
from agentick.agents import BaseAgent
from agentick.agents.backends.openai_backend import OpenAIBackend
from agentick.agents.harness import MarkovianZeroShot


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it in .env file or export OPENAI_API_KEY='your-key-here'")
        return

    print("OpenAI GPT-4o-mini Text Agent")
    print("=" * 80)

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Create agent using the harness system
    backend = OpenAIBackend(model="gpt-4o-mini", temperature=0.0, max_tokens=100)
    harness = MarkovianZeroShot()
    agent = BaseAgent(backend=backend, harness=harness, observation_modes=["language"])

    # Run episodes
    num_episodes = 5
    print(f"\nRunning {num_episodes} episodes...")

    total_rewards = []
    total_successes = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        agent.reset()

        total_reward = 0.0
        step = 0

        while True:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                break

        success = bool(info.get("success", False))
        total_rewards.append(total_reward)
        total_successes.append(success)

        print(f"  Episode {episode + 1}: reward={total_reward:.2f}, "
              f"steps={step}, success={success}")

    env.close()

    # Summary
    import numpy as np

    print(f"\n{'=' * 80}")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Success Rate: {np.mean(total_successes):.1%}")

    stats = agent.get_stats()
    print(f"API Calls: {stats['total_calls']}, Tokens: {stats['total_tokens']}")


if __name__ == "__main__":
    main()
