"""LLM agent example (template/mock)."""

import agentick


def mock_llm_call(prompt: str) -> str:
    """Mock LLM call - replace with actual LLM API."""
    # In a real implementation, call OpenAI, Anthropic, etc.
    # For now, return a simple heuristic
    if "goal" in prompt.lower():
        if "right" in prompt or "east" in prompt:
            return "MOVE_RIGHT"
        elif "down" in prompt or "south" in prompt:
            return "MOVE_DOWN"
    return "MOVE_UP"


def main():
    # Create environment
    env = agentick.make(
        "GoToGoal-v0",
        difficulty="easy",
        render_mode="language",  # Natural language observations
    )

    # Run episode
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0

    while not done:
        # Format prompt for LLM
        prompt = f"""You are playing a gridworld game.

Observation:
{obs}

Valid actions: {", ".join(info["valid_actions"])}

Choose one action to take. Respond with just the action name.
Action: """

        # Get LLM response (mock)
        action_name = mock_llm_call(prompt)

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
