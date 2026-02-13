"""
OpenAI GPT-4o Vision agent for Agentick tasks.

This example demonstrates:
- Using GPT-4o with vision capabilities
- Processing visual observations from environments
- Converting actions from text to environment actions

Requirements:
    - OpenAI API key: export OPENAI_API_KEY=your-key
    - uv sync --extra llm

Usage:
    export OPENAI_API_KEY=your-key
    uv run python examples/llm/openai_vision_agent.py
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv
from PIL import Image

import agentick

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install with: uv sync --extra llm")


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded image."""
    # Convert to PIL Image
    image = Image.fromarray(image_array.astype(np.uint8))

    # Save to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    # Encode to base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_base64}"


class OpenAIVisionAgent:
    """GPT-4o vision agent."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: uv sync --extra llm")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def get_action(
        self,
        observation: np.ndarray,
        task_description: str,
        action_space_description: str,
    ) -> str:
        """Get action from GPT-4o vision model."""
        # Convert observation to base64
        image_base64 = numpy_to_base64(observation)

        # Create prompt
        prompt = f"""You are an AI agent playing a game.

Task: {task_description}

Available actions: {action_space_description}

Look at the current game state in the image and decide what action to take.
Respond with ONLY the action name, nothing else. For example: "move_up" or "pick_object"
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_base64}},
                        ],
                    }
                ],
                temperature=self.temperature,
                max_tokens=50,
            )

            action_text = response.choices[0].message.content.strip()
            return action_text

        except Exception as e:
            print(f"Error getting action from GPT-4o: {e}")
            return "noop"


def parse_action(action_text: str, action_list: list[str]) -> int:
    """Parse action text to action index."""
    action_text = action_text.lower().strip()

    # Try exact match
    for i, action in enumerate(action_list):
        if action.lower() == action_text:
            return i

    # Try partial match
    for i, action in enumerate(action_list):
        if action_text in action.lower() or action.lower() in action_text:
            return i

    # Default to first action (usually noop)
    print(f"Could not parse action '{action_text}', using default")
    return 0


def main():
    """Run GPT-4o vision agent."""
    # Load environment variables
    load_dotenv()

    print("OpenAI GPT-4o Vision Agent")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=your-key")
        return

    if not OPENAI_AVAILABLE:
        return

    # Create environment (use rgb_array for vision)
    env_id = "GoToGoal-v0"
    print(f"\nEnvironment: {env_id}")

    env = agentick.make(env_id, difficulty="easy", render_mode="rgb_array")

    # Setup video recording
    video_folder = "videos/openai_vision"
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="openai_vision",
    )
    print(f"✓ Video recording enabled: {video_folder}")

    # Get action descriptions
    action_list = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

    # Create agent
    print("Creating GPT-4o vision agent...")
    agent = OpenAIVisionAgent(api_key=api_key, model="gpt-4o", temperature=0.3)

    # Run episodes
    num_episodes = 3
    print(f"\nRunning {num_episodes} episodes...")
    print()

    results = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 100  # Limit steps for cost

        print(f"Episode {episode + 1}:")

        while not done and steps < max_steps:
            # Get action from GPT-4o
            action_text = agent.get_action(
                observation=obs,
                task_description="Navigate to the goal position in a grid world",
                action_space_description=", ".join(action_list),
            )

            # Parse action
            action = parse_action(action_text, action_list)

            print(f"  Step {steps + 1}: Action = {action_text} ({action_list[action]})")

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        success = info.get("success", False)
        print(f"  Result: reward={total_reward:.2f}, steps={steps}, success={success}")
        print()

        results.append({
            "episode": episode + 1,
            "reward": total_reward,
            "steps": steps,
            "success": success,
        })

    env.close()

    # Save results
    results_path = Path(video_folder) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Episodes: {len(results)}")
    print(f"  Mean reward: {sum(r['reward'] for r in results) / len(results):.2f}")
    print(f"  Mean steps: {sum(r['steps'] for r in results) / len(results):.1f}")
    print(f"  Success rate: {sum(r['success'] for r in results) / len(results):.0%}")
    print("=" * 80)
    print("Done!")
    print(f"\n✓ Videos saved: {video_folder}")
    print("💡 Note: This uses GPT-4o API calls which cost money.")
    print("Each step requires one API call with vision.")


if __name__ == "__main__":
    main()
