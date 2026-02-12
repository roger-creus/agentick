"""Performance profiling for AgentTick tasks.

This module provides utilities for profiling task performance, particularly
the step() method which is on the critical path for training and evaluation.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from agentick.benchmark.baselines import RandomAgent
from agentick.tasks.registry import list_tasks, make


@dataclass
class ProfilingResult:
    """Results from profiling a task."""

    task_name: str
    num_steps: int
    total_time: float
    steps_per_sec: float
    avg_step_time_us: float
    render_mode: str

    # Breakdown
    step_times: list[float]
    min_step_time_us: float
    max_step_time_us: float
    p50_step_time_us: float
    p95_step_time_us: float
    p99_step_time_us: float

    def __str__(self) -> str:
        """Format profiling result as string."""
        return (
            f"{self.task_name} ({self.render_mode}):\n"
            f"  Steps/sec: {self.steps_per_sec:,.0f}\n"
            f"  Avg: {self.avg_step_time_us:.1f}μs\n"
            f"  P50: {self.p50_step_time_us:.1f}μs\n"
            f"  P95: {self.p95_step_time_us:.1f}μs\n"
            f"  P99: {self.p99_step_time_us:.1f}μs\n"
        )


def profile_task_step(
    task_name: str,
    num_steps: int = 10000,
    render_mode: str = "state_dict",
    seed: int = 42,
    warmup_steps: int = 100,
    fast_mode: bool = True,
) -> ProfilingResult:
    """Profile the step() method of a task.

    Args:
        task_name: Name of the task to profile
        num_steps: Number of steps to profile
        render_mode: Render mode ('state_dict', 'rgb_array', 'ansi')
        seed: Random seed
        warmup_steps: Number of warmup steps before profiling
        fast_mode: Enable fast mode for state_dict rendering

    Returns:
        ProfilingResult with timing statistics
    """
    # Create environment
    env = make(task_name, render_mode=render_mode, seed=seed, fast_mode=fast_mode)
    agent = RandomAgent(seed=seed)

    # Warmup
    obs, info = env.reset()
    for _ in range(warmup_steps):
        action = agent(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    # Profile
    step_times = []
    obs, info = env.reset()

    start_time = time.perf_counter()
    for _ in range(num_steps):
        action = agent(obs, info)

        step_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_end = time.perf_counter()

        step_times.append(step_end - step_start)

        if terminated or truncated:
            obs, info = env.reset()

    total_time = time.perf_counter() - start_time

    # Compute statistics
    step_times_us = np.array(step_times) * 1e6  # Convert to microseconds

    return ProfilingResult(
        task_name=task_name,
        num_steps=num_steps,
        total_time=total_time,
        steps_per_sec=num_steps / total_time,
        avg_step_time_us=np.mean(step_times_us),
        render_mode=render_mode,
        step_times=step_times,
        min_step_time_us=np.min(step_times_us),
        max_step_time_us=np.max(step_times_us),
        p50_step_time_us=np.percentile(step_times_us, 50),
        p95_step_time_us=np.percentile(step_times_us, 95),
        p99_step_time_us=np.percentile(step_times_us, 99),
    )


def profile_all_tasks(
    num_steps: int = 10000, render_mode: str = "state_dict", seed: int = 42
) -> dict[str, ProfilingResult]:
    """Profile all tasks.

    Args:
        num_steps: Number of steps to profile per task
        render_mode: Render mode
        seed: Random seed

    Returns:
        Dictionary mapping task names to profiling results
    """
    tasks = list_tasks()
    results = {}

    print(f"Profiling {len(tasks)} tasks ({num_steps} steps each)...")
    for i, task_name in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task_name}...", end=" ", flush=True)
        try:
            result = profile_task_step(
                task_name, num_steps=num_steps, render_mode=render_mode, seed=seed
            )
            results[task_name] = result
            print(f"{result.steps_per_sec:,.0f} steps/sec")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def generate_profiling_report(
    results: dict[str, ProfilingResult], output_path: str | None = None
) -> str:
    """Generate a profiling report.

    Args:
        results: Profiling results from profile_all_tasks
        output_path: Optional path to save report (markdown format)

    Returns:
        Report as string
    """
    # Sort by steps per second (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1].steps_per_sec, reverse=True)

    lines = [
        "# AgentTick Performance Profile",
        "",
        f"Profiled {len(results)} tasks",
        f"Render mode: {sorted_results[0][1].render_mode if sorted_results else 'N/A'}",
        "",
        "## Summary Statistics",
        "",
    ]

    # Overall statistics
    all_steps_per_sec = [r.steps_per_sec for r in results.values()]
    all_avg_times = [r.avg_step_time_us for r in results.values()]

    lines.extend(
        [
            f"- **Mean throughput**: {np.mean(all_steps_per_sec):,.0f} steps/sec",
            f"- **Median throughput**: {np.median(all_steps_per_sec):,.0f} steps/sec",
            f"- **Min throughput**: {np.min(all_steps_per_sec):,.0f} steps/sec",
            f"- **Max throughput**: {np.max(all_steps_per_sec):,.0f} steps/sec",
            f"- **Mean step time**: {np.mean(all_avg_times):.1f}μs",
            "",
            "## Per-Task Performance",
            "",
            "| Task | Steps/sec | Avg (μs) | P50 (μs) | P95 (μs) | P99 (μs) |",
            "|------|-----------|----------|----------|----------|----------|",
        ]
    )

    for task_name, result in sorted_results:
        lines.append(
            f"| {task_name} | {result.steps_per_sec:,.0f} | "
            f"{result.avg_step_time_us:.1f} | {result.p50_step_time_us:.1f} | "
            f"{result.p95_step_time_us:.1f} | {result.p99_step_time_us:.1f} |"
        )

    lines.extend(["", "## Fastest Tasks (Top 10)", ""])

    for i, (task_name, result) in enumerate(sorted_results[:10], 1):
        lines.append(f"{i}. **{task_name}**: {result.steps_per_sec:,.0f} steps/sec")

    lines.extend(["", "## Slowest Tasks (Bottom 10)", ""])

    for i, (task_name, result) in enumerate(sorted_results[-10:][::-1], 1):
        lines.append(f"{i}. **{task_name}**: {result.steps_per_sec:,.0f} steps/sec")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


def profile_render_modes(
    task_name: str = "GoToGoal-v0", num_steps: int = 10000
) -> dict[str, ProfilingResult]:
    """Profile different render modes for a task.

    Args:
        task_name: Task to profile
        num_steps: Number of steps per mode

    Returns:
        Dictionary mapping render mode to results
    """
    modes = ["state_dict", "rgb_array"]  # Most common modes
    results = {}

    for mode in modes:
        print(f"Profiling {task_name} with {mode}...", end=" ", flush=True)
        try:
            result = profile_task_step(task_name, num_steps=num_steps, render_mode=mode)
            results[mode] = result
            print(f"{result.steps_per_sec:,.0f} steps/sec")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def identify_bottlenecks(
    results: dict[str, ProfilingResult], threshold_steps_per_sec: float = 50000
) -> list[str]:
    """Identify tasks with performance below threshold.

    Args:
        results: Profiling results
        threshold_steps_per_sec: Minimum acceptable performance

    Returns:
        List of task names with performance issues
    """
    bottlenecks = []
    for task_name, result in results.items():
        if result.steps_per_sec < threshold_steps_per_sec:
            bottlenecks.append(task_name)

    return bottlenecks


def compare_implementations(task_name: str, num_steps: int = 10000) -> dict[str, Any]:
    """Compare performance of different implementation approaches.

    This is useful for A/B testing optimizations.

    Args:
        task_name: Task to profile
        num_steps: Number of steps

    Returns:
        Comparison metrics
    """
    # Profile original
    original = profile_task_step(task_name, num_steps=num_steps)

    return {
        "task": task_name,
        "original_steps_per_sec": original.steps_per_sec,
        "original_avg_time_us": original.avg_step_time_us,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile AgentTick performance")
    parser.add_argument("--task", type=str, help="Profile specific task (default: all tasks)")
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of steps to profile (default: 10000)"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="state_dict",
        choices=["state_dict", "rgb_array", "ansi"],
        help="Render mode (default: state_dict)",
    )
    parser.add_argument("--output", type=str, help="Output path for report (markdown)")
    parser.add_argument(
        "--compare-modes", action="store_true", help="Compare all render modes for a task"
    )

    args = parser.parse_args()

    if args.compare_modes:
        if not args.task:
            args.task = "GoToGoal-v0"
        print(f"Comparing render modes for {args.task}...")
        results = profile_render_modes(args.task, num_steps=args.steps)
        for mode, result in results.items():
            print(f"\n{result}")

    elif args.task:
        # Profile single task
        result = profile_task_step(args.task, num_steps=args.steps, render_mode=args.render_mode)
        print(result)

    else:
        # Profile all tasks
        results = profile_all_tasks(num_steps=args.steps, render_mode=args.render_mode)
        report = generate_profiling_report(results, output_path=args.output)
        if not args.output:
            print("\n" + report)
