"""Collect expert trajectories for fine-tuning.

Runs oracle agent on tasks and exports trajectories in various formats.
"""

import argparse
import json
from pathlib import Path

import agentick
from agentick.agents import OracleAgent


def collect_trajectories(
    task_names: list[str],
    difficulty: str = "easy",
    n_episodes_per_task: int = 10,
    output_dir: str = "trajectories",
):
    """
    Collect expert trajectories.

    Args:
        task_names: List of task names to collect from
        difficulty: Difficulty level
        n_episodes_per_task: Number of episodes per task
        output_dir: Directory to save trajectories
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_trajectories = []

    for task_name in task_names:
        print(f"\nCollecting trajectories for {task_name} ({difficulty})...")

        env = agentick.make(task_name, difficulty=difficulty, render_mode="language")

        for episode in range(n_episodes_per_task):
            obs, info = env.reset()
            trajectory = {
                "task": task_name,
                "difficulty": difficulty,
                "episode": episode,
                "steps": [],
            }

            # Use oracle agent
            oracle = OracleAgent(env)

            done = False
            step = 0
            episode_reward = 0.0

            while not done and step < 100:
                action = oracle.act(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                trajectory["steps"].append(
                    {
                        "step": step,
                        "observation": str(obs)[:500],  # Truncate for size
                        "action": int(action),
                        "action_name": oracle.get_action_name(action),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )

                obs = next_obs
                done = terminated or truncated
                step += 1

            trajectory["total_reward"] = episode_reward
            trajectory["success"] = info.get("success", False)
            trajectory["n_steps"] = step

            all_trajectories.append(trajectory)

            if episode % 5 == 0:
                print(
                    f"  Episode {episode + 1}/{n_episodes_per_task}: {step} steps, success = {info.get('success', False)}"
                )

    # Export to JSONL (one episode per line)
    jsonl_path = output_path / "trajectories.jsonl"
    with open(jsonl_path, "w") as f:
        for traj in all_trajectories:
            json.dump(traj, f)
            f.write("\n")

    print(f"\n✓ Saved {len(all_trajectories)} trajectories to {jsonl_path}")

    # Export to conversation format for chat fine-tuning
    conversations = []
    for traj in all_trajectories:
        messages = []
        messages.append(
            {
                "role": "system",
                "content": f"You are an expert AI agent playing {traj['task']}. Your goal is to complete the task efficiently.",
            }
        )

        for step_data in traj["steps"]:
            # User message: observation
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation:\n{step_data['observation']}\n\nWhat action should I take?",
                }
            )

            # Assistant message: action
            messages.append({"role": "assistant", "content": step_data["action_name"]})

        conversations.append({"messages": messages})

    conversation_path = output_path / "conversations.jsonl"
    with open(conversation_path, "w") as f:
        for conv in conversations:
            json.dump(conv, f)
            f.write("\n")

    print(f"✓ Saved conversation format to {conversation_path}")

    # Summary statistics
    successful = sum(1 for t in all_trajectories if t["success"])
    mean_steps = sum(t["n_steps"] for t in all_trajectories) / len(all_trajectories)
    mean_reward = sum(t["total_reward"] for t in all_trajectories) / len(all_trajectories)

    print("\nSummary:")
    print(f"  Total trajectories: {len(all_trajectories)}")
    print(
        f"  Success rate: {successful}/{len(all_trajectories)} ({100 * successful / len(all_trajectories):.1f}%)"
    )
    print(f"  Mean steps: {mean_steps:.1f}")
    print(f"  Mean reward: {mean_reward:.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect expert trajectories")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["GoToGoal-v0", "KeyDoorPuzzle-v0", "MazeNavigation-v0"],
        help="Task names to collect from",
    )
    parser.add_argument("--difficulty", default="easy", help="Difficulty level")
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--output-dir", default="trajectories", help="Output directory")

    args = parser.parse_args()

    collect_trajectories(
        task_names=args.tasks,
        difficulty=args.difficulty,
        n_episodes_per_task=args.n_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
