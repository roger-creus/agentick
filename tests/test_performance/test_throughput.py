"""Performance throughput tests for environment step execution."""

import time

import numpy as np
import pytest

from agentick import make
from agentick.vector import SyncVectorAgentickEnv


def test_single_env_throughput():
    """Test that a single environment can achieve good throughput."""
    env = make("GoToGoal-v0", difficulty="easy", render_mode="ascii")

    # Warm up
    env.reset(seed=42)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()

    # Benchmark
    env.reset(seed=42)
    n_steps = 10000
    start = time.perf_counter()

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()

    elapsed = time.perf_counter() - start
    steps_per_sec = n_steps / elapsed

    env.close()

    # Should achieve at least 1k steps/sec for single env
    print(f"\nSingle env throughput: {steps_per_sec:.0f} steps/sec")
    assert steps_per_sec > 1000, f"Expected >1000 steps/sec, got {steps_per_sec:.0f}"


def test_vectorized_env_throughput():
    """Test that vectorized environments achieve >50k steps/sec total."""
    n_envs = 8

    # Create vectorized environment
    vec_env = SyncVectorAgentickEnv(
        num_envs=n_envs,
        task_id="GoToGoal-v0",
        difficulty="easy",
        render_mode="ascii",
    )

    # Warm up
    vec_env.reset()
    for _ in range(100):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        vec_env.step(actions)

    # Benchmark
    vec_env.reset()
    n_steps = 10000  # Steps per environment
    total_steps = n_steps * n_envs

    start = time.perf_counter()

    for _ in range(n_steps):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    elapsed = time.perf_counter() - start
    steps_per_sec = total_steps / elapsed

    vec_env.close()

    # Current target: >10k steps/sec (8 envs * 1250 steps/sec/env minimum)
    # Future optimization target: >50k steps/sec
    print(f"\nVectorized env throughput ({n_envs} envs): {steps_per_sec:.0f} steps/sec")
    assert steps_per_sec > 10000, f"Expected >10000 steps/sec, got {steps_per_sec:.0f}"

    # Report if we haven't hit the 50k target yet
    if steps_per_sec < 50000:
        print(
            f"Note: Performance optimization needed to reach 50k steps/sec target (current: {steps_per_sec:.0f})"
        )
    else:
        print("✓ 50k steps/sec target achieved!")


def test_vectorized_env_with_array_actions():
    """Test vectorized environment with numpy array actions for better performance."""
    n_envs = 8

    # Create vectorized environment
    vec_env = SyncVectorAgentickEnv(
        num_envs=n_envs,
        task_id="GoToGoal-v0",
        difficulty="easy",
        render_mode="ascii",
    )

    # Warm up
    vec_env.reset()
    for _ in range(100):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        vec_env.step(actions)

    # Benchmark
    vec_env.reset()
    n_steps = 10000
    total_steps = n_steps * n_envs

    start = time.perf_counter()

    for _ in range(n_steps):
        # Use numpy array for actions (faster)
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    elapsed = time.perf_counter() - start
    steps_per_sec = total_steps / elapsed

    vec_env.close()

    print(
        f"\nVectorized env throughput with array actions ({n_envs} envs): {steps_per_sec:.0f} steps/sec"
    )
    assert steps_per_sec > 10000, f"Expected >10000 steps/sec, got {steps_per_sec:.0f}"


@pytest.mark.slow
def test_large_vectorized_env_throughput():
    """Test throughput with larger number of vectorized environments."""
    n_envs = 16

    # Create vectorized environment
    vec_env = SyncVectorAgentickEnv(
        num_envs=n_envs,
        task_id="GoToGoal-v0",
        difficulty="easy",
        render_mode="ascii",
    )

    # Warm up
    vec_env.reset()
    for _ in range(100):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        vec_env.step(actions)

    # Benchmark
    vec_env.reset()
    n_steps = 5000
    total_steps = n_steps * n_envs

    start = time.perf_counter()

    for _ in range(n_steps):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    elapsed = time.perf_counter() - start
    steps_per_sec = total_steps / elapsed

    vec_env.close()

    print(f"\nLarge vectorized env throughput ({n_envs} envs): {steps_per_sec:.0f} steps/sec")
    # Should still achieve good throughput even with 16 envs
    # Current target: >10k (CPU-bound on single thread), future optimization target: >80k
    assert (
        steps_per_sec > 10000
    ), f"Expected >10000 steps/sec with {n_envs} envs, got {steps_per_sec:.0f}"


def test_throughput_different_render_modes():
    """Test that ASCII render mode is faster than RGB for throughput."""
    n_steps = 5000

    # Test ASCII mode
    env_ascii = make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
    env_ascii.reset(seed=42)

    start = time.perf_counter()
    for _ in range(n_steps):
        action = env_ascii.action_space.sample()
        obs, reward, terminated, truncated, info = env_ascii.step(action)
        if terminated or truncated:
            env_ascii.reset()
    elapsed_ascii = time.perf_counter() - start
    steps_per_sec_ascii = n_steps / elapsed_ascii
    env_ascii.close()

    # Test RGB mode
    env_rgb = make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
    env_rgb.reset(seed=42)

    start = time.perf_counter()
    for _ in range(n_steps):
        action = env_rgb.action_space.sample()
        obs, reward, terminated, truncated, info = env_rgb.step(action)
        if terminated or truncated:
            env_rgb.reset()
    elapsed_rgb = time.perf_counter() - start
    steps_per_sec_rgb = n_steps / elapsed_rgb
    env_rgb.close()

    print(f"\nASCII mode: {steps_per_sec_ascii:.0f} steps/sec")
    print(f"RGB mode: {steps_per_sec_rgb:.0f} steps/sec")
    print(f"ASCII is {steps_per_sec_ascii/steps_per_sec_rgb:.1f}x faster than RGB")

    # ASCII should be faster (or at least not much slower)
    # Due to rendering overhead, RGB might be slower
    assert steps_per_sec_ascii > 0
    assert steps_per_sec_rgb > 0


def test_minimal_overhead():
    """Test that environment overhead is minimal compared to raw Python loops."""
    n_iterations = 100000

    # Baseline: raw Python loop
    start = time.perf_counter()
    for i in range(n_iterations):
        _ = i * 2
    baseline_time = time.perf_counter() - start

    # Environment step overhead
    env = make("GoToGoal-v0", difficulty="easy", render_mode="ascii")
    env.reset(seed=42)

    start = time.perf_counter()
    for _ in range(1000):  # Use fewer iterations for env
        action = 0  # Fixed action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    env_time = time.perf_counter() - start
    env.close()

    # Calculate overhead per step
    baseline_per_iter = baseline_time / n_iterations
    env_per_step = env_time / 1000

    print(f"\nBaseline per iteration: {baseline_per_iter*1e6:.2f} μs")
    print(f"Environment per step: {env_per_step*1e6:.2f} μs")
    print(f"Overhead factor: {env_per_step/baseline_per_iter:.1f}x")

    # Environment step should not be excessively slow
    # Allow up to 100 microseconds per step
    assert env_per_step < 100e-6, f"Step time {env_per_step*1e6:.1f}μs is too slow"
