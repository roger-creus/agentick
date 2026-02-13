"""
OpenAI GPT-4o agent with text observations.

Demonstrates using GPT-4o API for zero-shot task solving.
Runtime: ~30 seconds per episode (API calls)
Cost: ~$0.01-0.05 per episode
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

import agentick
from agentick.leaderboard.adapters.api_adapter import APIAgent
from agentick.utils.video import record_episode


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it in .env file or export OPENAI_API_KEY='your-key-here'")
        return

    print("OpenAI GPT-4o-mini Text Agent")
    print("=" * 80)

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Create GPT-4o agent
    agent = APIAgent(
        provider="openai",
        model="gpt-4o-mini",
        observation_mode="language",
        api_key_env="OPENAI_API_KEY",
        max_tokens=100,
        temperature=0.0,
        log_calls=True,
    )

    # Run episodes
    num_episodes = 5
    print(f"\nRunning {num_episodes} episodes...")
    print()

    results = {
        "agent": "OpenAI GPT-4o-mini",
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

        while True:
            # Get action from GPT-4o
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

    env.close()

    # Record a video of one episode (use rgb_array env)
    print("\nRecording video of sample episode...")
    env_video = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")

    def agent_fn(obs, info):
        # For video, we need to handle rgb_array observation
        # Reset agent for clean state
        agent.reset()
        return agent.act(obs, info)

    video_dir = Path("results/llm/videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = record_episode(
            env_video,
            agent_fn,
            output_path=video_dir / "openai_text_sample.mp4",
            max_steps=50,
            fps=5,
        )
        print(f"  Video saved to: {video_path}")
    except Exception as e:
        print(f"  Warning: Could not record video: {e}")
    finally:
        env_video.close()

    # Summary statistics
    import numpy as np
    results["summary"] = {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "success_rate": float(np.mean(total_successes)),
        "total_api_calls": agent.total_calls,
        "total_tokens": agent.total_tokens,
    }

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Average Reward: {results['summary']['mean_reward']:.2f} ± {results['summary']['std_reward']:.2f}")
    print(f"  Success Rate: {results['summary']['success_rate']:.1%}")
    print(f"  Total API Calls: {results['summary']['total_api_calls']}")
    print(f"  Total Tokens: {results['summary']['total_tokens']}")

    # Save results to JSON
    output_dir = Path("results/llm")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "openai_text_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
