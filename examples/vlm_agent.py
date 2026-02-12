"""VLM (Vision-Language Model) agent example (template)."""

import numpy as np

import agentick


def mock_vlm_call(image: np.ndarray, prompt: str) -> str:
    """Mock VLM call - replace with actual VLM API."""
    # In a real implementation, call GPT-4V, Claude 3, Gemini Vision, etc.
    # For now, return a simple heuristic based on image statistics
    mean_color = image.mean()
    if mean_color > 150:
        return "MOVE_RIGHT"
    elif mean_color > 100:
        return "MOVE_DOWN"
    else:
        return "MOVE_UP"


def main():
    # Create environment with pixel observations
    env = agentick.make(
        "GoToGoal-v0",
        difficulty="easy",
        render_mode="rgb_array",  # Pixel observations for VLM
    )

    # Run episode
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0

    while not done:
        # obs is now a numpy array of shape (H, W, 3)
        prompt = f"""You are viewing a gridworld game as an RGB image.

The agent is shown as a blue circle.
The goal is shown in green.
Walls are gray/black.

Valid actions: {", ".join(info["valid_actions"])}

Choose the best action to reach the goal.
Action: """

        # Get VLM response (mock)
        action_name = mock_vlm_call(obs, prompt)

        # Map action name to index
        action_mapping = {
            name: i
            for i, name in enumerate(["NOOP", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"])
        }
        action = action_mapping.get(action_name, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            done = True

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
