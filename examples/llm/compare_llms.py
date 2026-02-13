"""
Compare multiple LLM agents on Agentick tasks.

This example demonstrates:
- Running multiple LLM agents on the same task
- Collecting and comparing performance metrics
- Generating comparison reports

Requirements:
    - API keys for providers you want to test
    - uv sync --extra llm

Usage:
    export OPENAI_API_KEY=your-key
    export ANTHROPIC_API_KEY=your-key
    uv run python examples/llm/compare_llms.py
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

from dotenv import load_dotenv

import agentick

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class AgentResult:
    """Results for a single agent."""

    name: str
    total_reward: float
    steps: int
    success: bool
    cost_estimate: float  # Estimated cost in USD


def create_openai_agent(model: str = "gpt-4o-mini") -> Callable:
    """Create OpenAI agent function."""
    if not OPENAI_AVAILABLE:
        return None

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_action(observation: str, task_description: str, actions: list[str]) -> str:
        prompt = f"""You are playing a game.

Task: {task_description}

Current state:
{observation}

Available actions: {", ".join(actions)}

Choose the best action. Respond with ONLY the action name.
Action:"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {e}")
            return actions[0]

    return get_action


def create_anthropic_agent(model: str = "claude-sonnet-4-20250514") -> Callable:
    """Create Anthropic agent function."""
    if not ANTHROPIC_AVAILABLE:
        return None

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_action(observation: str, task_description: str, actions: list[str]) -> str:
        prompt = f"""You are playing a game.

Task: {task_description}

Current state:
{observation}

Available actions: {", ".join(actions)}

Choose the best action. Respond with ONLY the action name.
Action:"""

        try:
            message = client.messages.create(
                model=model,
                max_tokens=20,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic error: {e}")
            return actions[0]

    return get_action


def parse_action(action_text: str, action_list: list[str]) -> int:
    """Parse action text to action index."""
    action_text = action_text.lower().strip()

    for i, action in enumerate(action_list):
        if action.lower() in action_text or action_text in action.lower():
            return i

    return 0


def run_agent(
    agent_fn: Callable,
    env_id: str,
    num_episodes: int = 3,
    max_steps: int = 20,
) -> tuple[float, float, float]:
    """Run agent and return average metrics."""
    env = agentick.make(env_id, difficulty="easy", render_mode="language")
    action_list = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

    total_rewards = []
    total_steps = []
    total_successes = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action_text = agent_fn(
                observation=obs,
                task_description="Navigate to the goal in a grid world",
                actions=action_list,
            )

            action = parse_action(action_text, action_list)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)
        total_successes.append(float(info.get("success", False)))

    env.close()

    import numpy as np

    return (
        float(np.mean(total_rewards)),
        float(np.mean(total_steps)),
        float(np.mean(total_successes)),
    )


def estimate_cost(model: str, num_calls: int) -> float:
    """Estimate API cost in USD."""
    # Rough estimates (as of 2025)
    costs = {
        "gpt-4o": 0.005 * num_calls,  # $5 per 1M input tokens
        "gpt-4o-mini": 0.0002 * num_calls,  # $0.15 per 1M
        "claude-sonnet-4-20250514": 0.003 * num_calls,  # $3 per 1M
    }
    return costs.get(model, 0.001 * num_calls)


def main():
    """Compare LLM agents."""
    # Load environment variables
    load_dotenv()

    print("LLM Agent Comparison")
    print("=" * 80)

    env_id = "GoToGoal-v0"
    num_episodes = 5
    max_steps = 20

    # Define agents to test
    agents = []

    # OpenAI agents
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        agents.append(("GPT-4o Mini", "gpt-4o-mini", create_openai_agent("gpt-4o-mini")))
        # Uncomment to test GPT-4o (more expensive):
        # agents.append(("GPT-4o", "gpt-4o", create_openai_agent("gpt-4o")))

    # Anthropic agents
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        agents.append(
            (
                "Claude Sonnet 4",
                "claude-sonnet-4-20250514",
                create_anthropic_agent("claude-sonnet-4-20250514"),
            )
        )

    if not agents:
        print("\n❌ No agents available. Set API keys:")
        print("  export OPENAI_API_KEY=your-key")
        print("  export ANTHROPIC_API_KEY=your-key")
        return

    print(f"\nTesting {len(agents)} agents on {env_id}")
    print(f"Episodes per agent: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print()

    # Run comparison
    results = []

    for name, model, agent_fn in agents:
        if agent_fn is None:
            continue

        print(f"Running {name}...")
        avg_reward, avg_steps, success_rate = run_agent(agent_fn, env_id, num_episodes, max_steps)

        cost = estimate_cost(model, num_episodes * max_steps)

        results.append(
            AgentResult(
                name=name,
                total_reward=avg_reward,
                steps=avg_steps,
                success=success_rate > 0,
                cost_estimate=cost,
            )
        )

        print(
            f"  Reward: {avg_reward:.2f}, Steps: {avg_steps:.1f}, Success: {success_rate:.1%}, Cost: ${cost:.4f}"
        )
        print()

    # Print comparison table
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(
        f"{'Agent':<20} {'Avg Reward':>12} {'Avg Steps':>12} {'Success Rate':>14} {'Est. Cost':>12}"
    )
    print("-" * 80)

    for result in sorted(results, key=lambda x: x.total_reward, reverse=True):
        success_pct = "100%" if result.success else "0%"
        print(
            f"{result.name:<20} {result.total_reward:>12.2f} {result.steps:>12.1f} {success_pct:>14} ${result.cost_estimate:>11.4f}"
        )

    print("=" * 80)

    # Find best agent
    best = max(results, key=lambda x: x.total_reward)
    print(f"\n🏆 Best performer: {best.name} (avg reward: {best.total_reward:.2f})")

    # Find most cost-effective
    cost_effective = min(results, key=lambda x: x.cost_estimate / max(x.total_reward, 0.1))
    print(
        f"💰 Most cost-effective: {cost_effective.name} (${cost_effective.cost_estimate:.4f} for {cost_effective.total_reward:.2f} reward)"
    )


if __name__ == "__main__":
    main()
