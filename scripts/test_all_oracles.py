#!/usr/bin/env python3
"""Test all oracle bots across all tasks and difficulties using eval seeds."""

import argparse
import sys

import agentick
from agentick.leaderboard.seeds import generate_task_seeds
from agentick.oracles import get_oracle, list_oracles

DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def test_oracle(task_name, difficulty, seed, max_steps=None):
    """Run one oracle episode. Returns (success, steps, error_msg)."""
    try:
        env = agentick.make(task_name, difficulty=difficulty)
        oracle = get_oracle(task_name, env)
        obs, info = env.reset(seed=seed)
        oracle.reset(obs, info)
        ms = max_steps or env.max_steps or 500
        for step in range(ms):
            action = oracle.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        success = info.get("success", False)
        env.close()
        return success, step + 1, None
    except Exception as e:
        return False, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Test all oracles on eval seeds")
    parser.add_argument("tasks", nargs="*", help="Specific tasks (default: all)")
    parser.add_argument(
        "--n-seeds", type=int, default=25, help="Number of seeds per (task, difficulty)"
    )
    parser.add_argument(
        "--split", default="eval", choices=["eval", "train"], help="Seed split"
    )
    args = parser.parse_args()

    available = list_oracles()
    if args.tasks:
        available = [t for t in available if t in args.tasks]

    total_success = 0
    total_episodes = 0
    results = {}

    for task_name in sorted(available):
        task_successes = 0
        task_total = 0
        for diff in DIFFICULTIES:
            seeds = generate_task_seeds(task_name, diff, args.split, args.n_seeds)
            for seed in seeds:
                success, steps, err = test_oracle(task_name, diff, seed)
                task_successes += int(success)
                task_total += 1
                total_success += int(success)
                total_episodes += 1
                if err:
                    print(f"  ERROR {task_name}/{diff}/s{seed}: {err}")
        rate = task_successes / task_total if task_total > 0 else 0
        results[task_name] = (task_successes, task_total, rate)
        status = "OK" if rate >= 0.9 else "FAIL"
        print(f"  {status} {task_name}: {task_successes}/{task_total} = {rate:.0%}")

    print(f"\n{'='*60}")
    overall = total_success / total_episodes if total_episodes > 0 else 0
    print(f"OVERALL: {total_success}/{total_episodes} = {overall:.1%}")

    failing = {k: v for k, v in results.items() if v[2] < 0.9}
    if failing:
        print(f"\nFailing tasks ({len(failing)}):")
        for name, (s, t, r) in sorted(failing.items(), key=lambda x: x[1][2]):
            print(f"  {name}: {s}/{t} = {r:.0%}")


if __name__ == "__main__":
    main()
