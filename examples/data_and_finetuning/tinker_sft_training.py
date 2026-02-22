"""Tinker SFT training pipeline.

Fine-tunes an LLM on multi-task oracle demonstrations using Tinker's remote LoRA
training infrastructure. Evaluates using an inline NonMarkovianReasoner harness
wrapper that demonstrates how to integrate Tinker agents with the harness concept.

Requires: pip install tinker; export TINKER_API_KEY="..."

Usage:
    # Quick single-task
    uv run python examples/data_and_finetuning/tinker_sft_training.py \
        --tasks GoToGoal-v0 --difficulties easy --n-episodes 5

    # Full multi-task training
    uv run python examples/data_and_finetuning/tinker_sft_training.py

    # Push dataset to Hub
    uv run python examples/data_and_finetuning/tinker_sft_training.py \
        --push-to-hub user/agentick-tinker-data --eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import (
    add_collection_args,
    add_eval_args,
    add_hub_args,
    add_task_args,
    collect_multi_task_data,
    resolve_tasks,
)


class TinkerNonMarkovianReasoner:
    """Inline harness wrapper for Tinker agents with history tracking and CoT.

    Maintains a sliding window of (observation, action, reward) history and builds
    prompts with task description + history + current observation. Demonstrates
    how to integrate Tinker finetuned models with the harness concept.

    Args:
        sampling_client: Tinker sampling client from trainer.as_agent().
        tokenizer: Tokenizer from the training client.
        task_description: Natural language task description.
        window_size: Number of recent steps to include in context.
    """

    def __init__(
        self,
        sampling_client: Any,
        tokenizer: Any,
        task_description: str = "",
        window_size: int = 10,
    ) -> None:
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.task_description = task_description
        self.window_size = window_size
        self.history: deque[dict[str, Any]] = deque(maxlen=window_size)
        self.last_reasoning: str | None = None

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Reset history for a new episode."""
        self.history.clear()
        self.last_reasoning = None
        if "task_description" in info:
            self.task_description = info["task_description"]

    def act(self, obs: Any, info: dict[str, Any]) -> int:
        """Generate action with history-aware CoT reasoning."""
        import tinker.types as tinker_types

        obs_text = obs if isinstance(obs, str) else json.dumps(obs)

        # Build prompt with task description + history + current observation
        parts = []
        if self.task_description:
            parts.append(f"Task: {self.task_description}")

        if self.history:
            parts.append("\nRecent history:")
            for i, h in enumerate(self.history):
                parts.append(
                    f"  Step {i + 1}: obs={h['obs'][:100]}... "
                    f"action={h['action']} reward={h['reward']}"
                )

        parts.append(f"\nCurrent observation:\n{obs_text}")
        parts.append(
            "\nThink step by step about the best action, then respond with Action: <number>"
        )

        prompt = "\n".join(parts)
        tokens = self.tokenizer.encode(prompt)

        model_input = tinker_types.ModelInput.from_ints(tokens=tokens)
        result = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker_types.SamplingParams(max_tokens=64),
        )

        text = self.tokenizer.decode(result.sequences[0].tokens).strip()
        self.last_reasoning = text

        # Parse action
        match = re.search(r"Action:\s*(\d+)", text)
        if not match:
            match = re.search(r"\b(\d+)\b", text)
        action = int(match.group(1)) if match else 0

        return action

    def update(self, obs: Any, info: dict[str, Any], action: int, reward: float) -> None:
        """Record step in history."""
        obs_text = obs if isinstance(obs, str) else str(obs)
        self.history.append({"obs": obs_text, "action": action, "reward": reward})


def evaluate_tinker_agent(
    agent: Any,
    tasks: list[str],
    difficulties: list[str],
    n_episodes: int = 5,
    render_mode: str = "language",
) -> dict[str, dict[str, float]]:
    """Evaluate a Tinker agent across tasks and difficulties.

    Args:
        agent: Agent with act(obs, info) -> int method.
        tasks: Task names.
        difficulties: Difficulty levels.
        n_episodes: Episodes per task/difficulty.
        render_mode: Observation render mode.

    Returns:
        Nested dict of {task: {difficulty: success_rate}}.
    """
    import agentick

    results: dict[str, dict[str, float]] = {}

    for task_name in tasks:
        results[task_name] = {}
        for difficulty in difficulties:
            try:
                env = agentick.make(task_name, difficulty=difficulty, render_mode=render_mode)
            except Exception as e:
                print(f"  Skipping {task_name} @ {difficulty}: {e}")
                continue

            successes = 0
            for ep in range(n_episodes):
                obs, info = env.reset(seed=1000 + ep)
                agent.reset(obs, info)

                done = False
                total_reward = 0.0
                prev_obs = obs
                while not done:
                    action = agent.act(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                    if hasattr(agent, "update"):
                        agent.update(prev_obs, info, action, reward)
                    prev_obs = obs

                if info.get("success", False):
                    successes += 1

            success_rate = successes / n_episodes
            results[task_name][difficulty] = success_rate
            print(f"  {task_name} @ {difficulty}: {success_rate:.0%} ({successes}/{n_episodes})")
            env.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tinker SFT pipeline: collect oracle data -> remote LoRA training -> eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task / data args
    add_task_args(parser)
    add_collection_args(parser)

    # Tinker-specific args
    tinker_group = parser.add_argument_group("tinker")
    tinker_group.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model for Tinker"
    )
    tinker_group.add_argument("--rank", type=int, default=32, help="LoRA rank")
    tinker_group.add_argument("--num-steps", type=int, default=100, help="Training steps")
    tinker_group.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    tinker_group.add_argument("--batch-size", type=int, default=4, help="Batch size")

    # Eval + Hub args
    add_eval_args(parser)
    add_hub_args(parser)

    args = parser.parse_args()

    # Check Tinker
    try:
        from agentick.training.tinker.sft import TinkerSFTTrainer
    except ImportError:
        print("ERROR: Tinker is not installed.")
        print("Install with: pip install tinker")
        print("Then set: export TINKER_API_KEY='your-api-key'")
        return

    tasks = resolve_tasks(args.tasks)
    output_dir = args.output_dir or "models/tinker_sft"
    data_dir = Path(output_dir) / "data"

    print("Tinker SFT Training Pipeline")
    print("=" * 80)
    print(f"Tasks: {len(tasks)}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Model: {args.model}, rank={args.rank}")
    print("=" * 80)

    # Step 1: Collect oracle data
    print("\nStep 1: Collecting oracle demonstrations...")
    combined_path = collect_multi_task_data(
        tasks=tasks,
        difficulties=args.difficulties,
        n_episodes=args.n_episodes,
        render_mode=args.render_mode,
        output_dir=str(data_dir),
        seed_offset=args.seed_offset,
        push_to_hub=args.push_to_hub,
    )

    # Step 2: Train with Tinker
    print("\nStep 2: Training with Tinker SFT...")
    trainer = TinkerSFTTrainer(
        base_model=args.model,
        dataset_path=combined_path,
        rank=args.rank,
        output_dir=output_dir,
    )
    metrics = trainer.train(
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    print(f"\nTraining complete. Final loss: {metrics.get('final_loss', 'N/A')}")

    # Step 3: Evaluate with NonMarkovianReasoner harness
    if args.eval and not args.no_eval:
        print("\nStep 3: Evaluating with NonMarkovianReasoner harness...")
        tinker_agent = trainer.as_agent()

        # Wrap in NonMarkovianReasoner for history-aware CoT evaluation
        reasoner = TinkerNonMarkovianReasoner(
            sampling_client=tinker_agent.sampling_client,
            tokenizer=tinker_agent.tokenizer,
        )

        results = evaluate_tinker_agent(
            agent=reasoner,
            tasks=tasks,
            difficulties=args.difficulties,
            n_episodes=args.eval_episodes,
            render_mode=args.render_mode,
        )

        # Print summary
        print("\nEval Summary:")
        for task_name, diff_results in results.items():
            for diff, sr in diff_results.items():
                print(f"  {task_name} @ {diff}: {sr:.0%}")

    print("\n" + "=" * 80)
    print("TINKER SFT PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
