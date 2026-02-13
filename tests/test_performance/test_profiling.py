"""Performance profiling tests to identify bottlenecks."""

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from agentick import make
from agentick.vector import SyncVectorAgentickEnv


def profile_single_env_step():
    """Profile a single environment step to identify bottlenecks."""
    env = make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
    env.reset(seed=42)

    # Profile 1000 steps
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Top 20 functions

    print("\n=== Single Environment Step Profiling ===")
    print(s.getvalue())

    env.close()


def profile_vectorized_env_step():
    """Profile vectorized environment steps."""
    n_envs = 8
    vec_env = SyncVectorAgentickEnv(
        num_envs=n_envs,
        task_id="GoToGoal-v0",
        difficulty="easy",
        render_mode="ascii",
    )
    vec_env.reset()

    # Profile 1000 steps
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print("\n=== Vectorized Environment Step Profiling ===")
    print(s.getvalue())

    vec_env.close()


def benchmark_component_overhead():
    """Benchmark overhead of different environment components."""
    env = make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
    env.reset(seed=42)
    n_iterations = 1000

    # Baseline: just calling step with noop
    start = time.perf_counter()
    for _ in range(n_iterations):
        env.step(0)  # noop
    baseline_time = time.perf_counter() - start

    # Reset for fair comparison
    env.reset(seed=42)

    # With rendering (calling render separately)
    start = time.perf_counter()
    for _ in range(n_iterations):
        env.step(0)
        _ = env.render()
    with_render_time = time.perf_counter() - start

    print("\n=== Component Overhead Analysis ===")
    print(f"Baseline step time: {baseline_time / n_iterations * 1000:.3f} ms/step")
    print(f"With render: {with_render_time / n_iterations * 1000:.3f} ms/step")
    print(
        f"Render overhead: {(with_render_time - baseline_time) / n_iterations * 1000:.3f} ms/step"
    )

    env.close()


if __name__ == "__main__":
    print("Running performance profiling...")
    benchmark_component_overhead()
    profile_single_env_step()
    profile_vectorized_env_step()
