"""
OpenAI GPT-4o agent with Chain-of-Thought reasoning.

Demonstrates using GPT-4o with explicit reasoning steps before action selection.
This often improves performance on complex tasks by encouraging step-by-step thinking.

Runtime: ~30-45 seconds per episode (API calls)
Cost: ~$0.02-0.08 per episode (higher due to longer responses)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

import agentick
from agentick.leaderboard.adapters.api_adapter import APIAgent


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it in .env file or export OPENAI_API_KEY='your-key-here'")
        return

    print("OpenAI GPT-4o Chain-of-Thought Agent")
    print("=" * 80)

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Create GPT-4o agent with Chain-of-Thought prompt
    # We use a higher max_tokens to allow for reasoning + action
    agent = APIAgent(
        provider="openai",
        model="gpt-4o-mini",
        observation_mode="language",
        api_key_env="OPENAI_API_KEY",
        max_tokens=150,  # Higher to allow reasoning
        temperature=0.0,
        log_calls=True,
    )

    # Override the default prompt to use Chain-of-Thought
    # This is a simple override - in production you'd use a proper prompt template
    original_format = agent._format_observation

    def cot_format_observation(observation, info):
        """Format observation with CoT prompt."""
        base_prompt = original_format(observation, info)

        # Add CoT instructions
        cot_prompt = f"""{base_prompt}

Think step-by-step:
1. What do I observe?
2. What is my current goal?
3. What action brings me closer to the goal?

After your reasoning, output your action on a new line starting with "ACTION: " followed by the action number.

Example:
1. I observe: Agent at (2,3), Goal at (4,3)
2. My goal: Reach (4,3)
3. Best action: Move right to get closer
ACTION: 4"""

        return cot_prompt

    agent._format_observation = cot_format_observation

    # Run episodes
    num_episodes = 5
    print(f"\nRunning {num_episodes} episodes with Chain-of-Thought reasoning...")
    print()

    results = {
        "agent": "OpenAI GPT-4o-mini CoT",
        "model": "gpt-4o-mini",
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "episodes": [],
    }

    total_rewards = []
    total_successes = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)

        episode_data = {
            "episode": episode + 1,
            "steps": [],
            "total_reward": 0,
            "success": False,
        }

        print(f"Episode {episode + 1}:")

        total_reward = 0
        step = 0
        max_steps = 50  # Use task default

        while step < max_steps:
            # Get action from GPT-4o with CoT
            action = agent.act(obs, info)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            episode_data["steps"].append({
                "step": step,
                "action": int(action),
                "reward": float(reward),
            })

            if terminated or truncated:
                episode_data["success"] = bool(info.get("success", False))
                break

        episode_data["total_reward"] = float(total_reward)
        episode_data["num_steps"] = step

        results["episodes"].append(episode_data)
        total_rewards.append(total_reward)
        total_successes.append(episode_data["success"])

        print(f"  Reward: {total_reward:.2f}, Steps: {step}, Success: {episode_data['success']}")
        print()

    env.close()

    # Summary statistics
    import numpy as np
    results["summary"] = {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "success_rate": float(np.mean(total_successes)),
        "total_api_calls": agent.total_calls,
        "total_tokens": agent.total_tokens,
    }

    print("=" * 80)
    print("Summary:")
    print(f"  Average Reward: {results['summary']['mean_reward']:.2f} ± {results['summary']['std_reward']:.2f}")
    print(f"  Success Rate: {results['summary']['success_rate']:.1%}")
    print(f"  Total API Calls: {results['summary']['total_api_calls']}")
    print(f"  Total Tokens: {results['summary']['total_tokens']}")

    # Save results to JSON
    output_dir = Path("results/llm")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "openai_cot_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print()
    print("💡 Chain-of-Thought prompting encourages the model to reason explicitly")
    print("   before selecting an action, which can improve performance on complex tasks.")


if __name__ == "__main__":
    main()
