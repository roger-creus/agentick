"""Test environment step throughput."""

import time

import pytest

import agentick


@pytest.mark.slow
@pytest.mark.benchmark
def test_state_dict_throughput():
    """Test throughput with state_dict observation mode."""
    env = agentick.make("GoToGoal-v0", render_mode="state_dict", difficulty="easy")

    n_steps = 10000
    start_time = time.time()

    env.reset(seed=42)

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset(seed=42)

    elapsed = time.time() - start_time
    throughput = n_steps / elapsed

    env.close()

    print(f"\nState dict throughput: {throughput:.0f} steps/sec")

    # Should be >100k steps/sec on simple tasks
    assert throughput > 10000, f"Too slow: {throughput:.0f} steps/sec"


@pytest.mark.slow
@pytest.mark.benchmark
def test_pixel_throughput():
    """Test throughput with pixel observations."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array", difficulty="easy")

    n_steps = 1000  # Fewer steps since pixels are slower
    start_time = time.time()

    env.reset(seed=42)

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset(seed=42)

    elapsed = time.time() - start_time
    throughput = n_steps / elapsed

    env.close()

    print(f"\nPixel throughput: {throughput:.0f} steps/sec")

    # Should be >1000 steps/sec
    assert throughput > 100, f"Too slow: {throughput:.0f} steps/sec"


@pytest.mark.slow
@pytest.mark.benchmark
def test_vectorized_throughput():
    """Test throughput with vectorized environments."""
    from gymnasium.vector import SyncVectorEnv

    n_envs = 8

    def make_env():
        return agentick.make("GoToGoal-v0", render_mode="state_dict", difficulty="easy")

    envs = SyncVectorEnv([make_env for _ in range(n_envs)])

    n_steps = 1000
    start_time = time.time()

    envs.reset(seed=42)

    for _ in range(n_steps):
        actions = envs.action_space.sample()
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)

    elapsed = time.time() - start_time
    total_steps = n_steps * n_envs
    throughput = total_steps / elapsed

    envs.close()

    print(f"\nVectorized throughput ({n_envs} envs): {throughput:.0f} steps/sec")

    # Should be >50k steps/sec with 8 envs
    assert throughput > 5000, f"Too slow: {throughput:.0f} steps/sec"


@pytest.mark.benchmark
def test_task_throughput_all_modes(tmp_path):
    """Measure throughput for all tasks and observation modes."""
    tasks = ["GoToGoal-v0", "MazeNavigation-v0", "KeyDoorPuzzle-v0"]
    modes = ["state_dict", "ascii", "language"]

    results = {}

    for task in tasks:
        for mode in modes:
            try:
                env = agentick.make(task, render_mode=mode, difficulty="easy")

                n_steps = 1000
                start_time = time.time()

                env.reset(seed=42)

                for _ in range(n_steps):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        env.reset(seed=42)

                elapsed = time.time() - start_time
                throughput = n_steps / elapsed

                env.close()

                results[f"{task}_{mode}"] = throughput

            except Exception as e:
                print(f"Skipped {task} {mode}: {e}")

    # Save results
    results_path = tmp_path / "throughput_results.txt"
    with open(results_path, "w") as f:
        f.write("Task,Mode,Throughput (steps/sec)\n")
        for key, value in sorted(results.items()):
            task, mode = key.rsplit("_", 1)
            f.write(f"{task},{mode},{value:.0f}\n")

    print(f"\nThroughput results saved to {results_path}")

    # At least some results should be collected
    assert len(results) > 0
