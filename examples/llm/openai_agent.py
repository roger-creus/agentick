"""OpenAI GPT agent for Agentick.

Example of using OpenAI API (GPT-4 or GPT-3.5) to play Agentick tasks.
"""

import os
import sys

import agentick

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.base_llm_agent import BaseLLMAgent


class OpenAIAgent(BaseLLMAgent):
    """Agent using OpenAI API."""

    def __init__(self, model: str = "gpt-4", **kwargs):
        """
        Initialize OpenAI agent.

        Args:
            model: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
            **kwargs: Additional arguments for BaseLLMAgent
        """
        super().__init__(**kwargs)
        self.model = model
        self.client = None

    def _init_client(self):
        """Initialize OpenAI client (lazy initialization)."""
        if self.client is None:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize OpenAI client. Set OPENAI_API_KEY environment variable. Error: {e}"
                )

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API."""
        self._init_client()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI agent that plays gridworld games. Always respond with only the action name.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=50,
        )

        return response.choices[0].message.content.strip()


def main():
    """Run OpenAI agent on Agentick tasks."""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print("Running OpenAI agent on Agentick...")
    print("=" * 60)

    # Create agent
    agent = OpenAIAgent(model="gpt-4", temperature=0.0)

    # Test on simple task
    task_name = "GoToGoal-v0"
    difficulty = "easy"
    n_episodes = 3

    env = agentick.make(task_name, difficulty=difficulty, render_mode="language")

    total_success = 0
    total_reward = 0.0

    for episode in range(n_episodes):
        obs, info = env.reset()
        agent.reset()

        episode_reward = 0.0
        done = False
        step = 0

        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        while not done and step < 50:
            # Get valid actions
            valid_actions = info.get(
                "valid_actions", ["move_up", "move_down", "move_left", "move_right"]
            )

            # Agent chooses action
            action = agent.act(obs, valid_actions)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1

            # Record for history
            agent.record_step(obs, action, reward)

            if step % 10 == 0:
                print(f"  Step {step}: reward = {episode_reward:.2f}")

        success = info.get("success", False)
        total_success += int(success)
        total_reward += episode_reward

        print(f"  Final: steps = {step}, reward = {episode_reward:.2f}, success = {success}")

    print("\n" + "=" * 60)
    print(f"Results over {n_episodes} episodes:")
    print(f"  Success rate: {total_success}/{n_episodes} ({100 * total_success / n_episodes:.1f}%)")
    print(f"  Mean reward: {total_reward / n_episodes:.2f}")


if __name__ == "__main__":
    main()
