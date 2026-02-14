"""Run full benchmark suite."""

from agentick.benchmark import BenchmarkRunner, RandomAgent

# Create random agent
agent = RandomAgent(seed=42)

# Run benchmark on quick suite
runner = BenchmarkRunner(suite="quick", n_episodes=100)
results = runner.evaluate(agent_fn=agent, difficulty="easy")

# Print results
print("\n=== Benchmark Results ===\n")
for task_name, metrics in results.items():
    print(f"{task_name}:")
    print(f"  Mean Return: {metrics['mean_return']:.3f}")
    print(f"  Success Rate: {metrics['success_rate']:.1%}\n")
