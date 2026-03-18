# Results Format

Every experiment run produces a standardized results directory:

```
results/{name}_{timestamp}/
  config.yaml              # Original experiment config
  metadata.json            # Runtime info (git hash, version, platform, agent details)
  summary.json             # Aggregate metrics across all tasks
  per_task/
    {TaskName-v0}/
      metrics.json          # Per-difficulty episodes + aggregate metrics
      episodes/
        diff_{difficulty}_seed_{idx}_ep_{idx}.json
  figures/                  # Auto-generated plots (if visualization extra installed)
```

## metadata.json

```json
{
  "agentick_version": "0.1.0",
  "python_version": "3.11.0",
  "platform": "Linux-5.15.0-x86_64",
  "timestamp": "2026-03-17T12:00:00",
  "git_hash": "abc123",
  "agent_name": "Qwen3-4B-Markov-ASCII",
  "agent_type": "llm",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "backend": "vllm_llm",
  "observation_modes": ["ascii"],
  "harness": "markovian_zero_shot"
}
```

## summary.json

```json
{
  "mean_return": 0.42,
  "success_rate": 0.38,
  "mean_length": 45.2,
  "total_time_seconds": 3600,
  "total_episodes": 3800
}
```

## per_task/{task}/metrics.json

```json
{
  "task_name": "GoToGoal-v0",
  "per_difficulty": {
    "easy": {
      "difficulty": "easy",
      "episodes": [
        {"seed": 12345, "episode_idx": 0, "return": 1.0, "length": 8, "success": true}
      ],
      "metrics": {
        "mean_return": 0.95,
        "success_rate": 0.92,
        "mean_length": 10.3
      }
    }
  },
  "aggregate_metrics": {
    "mean_return": 0.75,
    "success_rate": 0.68,
    "mean_length": 25.1
  }
}
```
