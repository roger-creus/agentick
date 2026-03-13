"""Collect oracle trajectories across all task-difficulty pairs and upload to HuggingFace.

Produces a flat per-step dataset with columns:
    task, episode_id, difficulty, step, ascii_render, language_render,
    action_name, action_int, reward, done
    [optional] image (uint8 rgb array, stored natively in parquet)

Uses deterministic training seeds from the benchmark suite system.

Usage:
    # All tasks, all difficulties, 10 episodes each, no images (fast)
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
        --push-to-hub user/agentick-oracle-trajectories

    # With images (uint8 rgb arrays, for VLM/BC training)
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
        --include-images --push-to-hub user/agentick-oracle-trajectories

    # Specific tasks, fewer episodes
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
        --tasks GoToGoal-v0 KeyDoorPuzzle-v0 --difficulties easy medium \
        --n-episodes 5 --push-to-hub user/agentick-oracle-trajectories

    # Save locally only
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
        --output-dir trajectories/oracle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def _collect_split(
    tasks: list[str],
    difficulties: list[str],
    n_episodes: int,
    split: str,
    include_images: bool = False,
    episode_offset: int = 0,
) -> list[dict]:
    """Collect oracle trajectories for a single split (train or test).

    Args:
        split: Seed split to use — "train" or "eval".
        episode_offset: Starting episode ID (so train/test IDs don't collide).

    Returns:
        List of per-step row dicts.
    """
    import agentick
    from agentick.leaderboard.seeds import generate_task_seeds
    from agentick.oracles import get_oracle

    rows: list[dict] = []
    episode_counter = episode_offset

    total_combos = len(tasks) * len(difficulties)
    combo_idx = 0

    for task_name in tasks:
        for difficulty in difficulties:
            combo_idx += 1
            print(
                f"  [{combo_idx}/{total_combos}] {task_name} @ {difficulty} "
                f"({n_episodes} episodes)..."
            )

            try:
                env = agentick.make(
                    task_name, difficulty=difficulty, render_mode="ascii"
                )
            except Exception as e:
                print(f"    SKIP (env creation failed): {e}")
                continue

            try:
                oracle = get_oracle(task_name, env)
            except Exception as e:
                print(f"    SKIP (no oracle): {e}")
                env.close()
                continue

            action_space_obj = env.unwrapped.action_space_obj

            seeds = generate_task_seeds(
                task_name, difficulty, split, max(n_episodes, 1)
            )
            seeds = seeds[:n_episodes]

            successes = 0
            for seed in seeds:
                ep_id = episode_counter
                episode_counter += 1

                obs, info = env.reset(seed=seed)
                oracle.reset(obs, info)

                step_idx = 0
                done = False
                truncated = False
                while not (done or truncated):
                    ascii_render = env.unwrapped.render_in_mode("ascii")
                    lang_render = env.unwrapped.render_in_mode("language")

                    row = {
                        "task": task_name,
                        "episode_id": ep_id,
                        "difficulty": difficulty,
                        "step": step_idx,
                        "ascii_render": str(ascii_render),
                        "language_render": str(lang_render),
                    }

                    if include_images:
                        rgb = env.unwrapped.render_in_mode("rgb_array")
                        row["image"] = np.asarray(rgb, dtype=np.uint8)

                    action = oracle.act(obs, info)
                    obs, reward, done, truncated, info = env.step(action)
                    oracle.update(obs, info)

                    row["action_name"] = action_space_obj.get_action_name(int(action))
                    row["action_int"] = int(action)
                    row["reward"] = float(reward)
                    row["done"] = bool(done or truncated)

                    rows.append(row)
                    step_idx += 1

                if info.get("success", False):
                    successes += 1

            print(f"    {successes}/{len(seeds)} success, {step_idx} steps (last ep)")
            env.close()

    return rows


def _rows_to_dataset(rows: list[dict], include_images: bool):
    """Convert row dicts to a HuggingFace Dataset."""
    from datasets import Dataset, Features, Value

    feature_dict = {
        "task": Value("string"),
        "episode_id": Value("int32"),
        "difficulty": Value("string"),
        "step": Value("int32"),
        "ascii_render": Value("string"),
        "language_render": Value("string"),
    }
    if include_images:
        from datasets import Array3D
        feature_dict["image"] = Array3D(shape=(512, 512, 3), dtype="uint8")
    feature_dict.update({
        "action_name": Value("string"),
        "action_int": Value("int32"),
        "reward": Value("float32"),
        "done": Value("bool"),
    })

    features = Features(feature_dict)
    return Dataset.from_list(rows, features=features)


def collect_all(
    tasks: list[str],
    difficulties: list[str],
    n_episodes: int,
    push_to_hub: str | None,
    output_dir: str | None,
    include_images: bool = False,
    n_test_episodes: int | None = None,
) -> None:
    """Collect oracle trajectories and build a HuggingFace dataset.

    When n_test_episodes is set, produces a DatasetDict with train/test splits
    using different deterministic seeds. Otherwise produces a single Dataset.
    """
    from datasets import DatasetDict

    # --- Train split ---
    print(f"\n=== Collecting TRAIN split ({n_episodes} episodes/combo) ===")
    train_rows = _collect_split(
        tasks, difficulties, n_episodes, split="train",
        include_images=include_images, episode_offset=0,
    )

    if not train_rows:
        print("ERROR: No data collected. Check task names / oracles.")
        return

    # --- Test split (optional) ---
    test_rows = []
    if n_test_episodes and n_test_episodes > 0:
        print(f"\n=== Collecting TEST split ({n_test_episodes} episodes/combo) ===")
        test_rows = _collect_split(
            tasks, difficulties, n_test_episodes, split="eval",
            include_images=include_images, episode_offset=len(set(r["episode_id"] for r in train_rows)),
        )

    # --- Build dataset(s) ---
    if test_rows:
        train_ds = _rows_to_dataset(train_rows, include_images)
        test_ds = _rows_to_dataset(test_rows, include_images)
        ds = DatasetDict({"train": train_ds, "test": test_ds})
        print(f"\nDatasetDict: train={len(train_ds)} rows, test={len(test_ds)} rows")
    else:
        ds = _rows_to_dataset(train_rows, include_images)
        print(f"\nDataset: {len(ds)} rows")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_path))
        print(f"Saved to disk: {out_path}")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {push_to_hub} ...")
        ds.push_to_hub(push_to_hub)
        print(f"Done: https://huggingface.co/datasets/{push_to_hub}")

    if not output_dir and not push_to_hub:
        default_path = Path("trajectories/oracle/hf_dataset")
        default_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(default_path))
        print(f"Saved to disk: {default_path}")

    # Summary
    all_rows = train_rows + test_rows
    unique_tasks = len(set(r["task"] for r in all_rows))
    unique_eps = len(set(r["episode_id"] for r in all_rows))
    print(f"\nSummary: {len(all_rows)} steps, {unique_eps} episodes, {unique_tasks} tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Collect oracle trajectories and upload to HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names (default: all tasks with oracles)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=ALL_DIFFICULTIES,
        choices=ALL_DIFFICULTIES,
        help="Difficulty levels",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes per task-difficulty pair",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="Push dataset to HuggingFace Hub (e.g. user/agentick-oracle-data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Save dataset to local directory",
    )
    parser.add_argument(
        "--n-test-episodes",
        type=int,
        default=None,
        help="Number of test episodes per combo (uses eval seeds). Omit for train-only.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        default=False,
        help="Include images as uint8 rgb arrays (increases dataset size and upload time)",
    )

    args = parser.parse_args()

    # Resolve tasks
    if args.tasks is None:
        from agentick.oracles import list_oracles
        tasks = list_oracles()
    else:
        tasks = args.tasks

    print("Oracle Trajectory Collection")
    print("=" * 80)
    print(f"Tasks: {len(tasks)} ({', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''})")
    print(f"Difficulties: {args.difficulties}")
    print(f"Train episodes per combo: {args.n_episodes}")
    if args.n_test_episodes:
        print(f"Test episodes per combo: {args.n_test_episodes}")
    print(f"Images: {'uint8 rgb arrays' if args.include_images else 'disabled'}")
    if args.push_to_hub:
        print(f"Push to Hub: {args.push_to_hub}")
    if args.output_dir:
        print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    collect_all(
        tasks=tasks,
        difficulties=args.difficulties,
        n_episodes=args.n_episodes,
        push_to_hub=args.push_to_hub,
        output_dir=args.output_dir,
        include_images=args.include_images,
        n_test_episodes=args.n_test_episodes,
    )


if __name__ == "__main__":
    main()
