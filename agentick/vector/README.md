# Vector

Vectorized environment support for parallel episode collection, built on Gymnasium's `VectorEnv` interface.

## Classes

- `SyncVectorAgentickEnv(num_envs, task_id, difficulty, render_mode, **kwargs)` -- runs N environments sequentially in a single process. Subclasses `gymnasium.vector.SyncVectorEnv`. Fast for lightweight environments on CPU.
- `AsyncVectorAgentickEnv(num_envs, task_id, difficulty, render_mode, **kwargs)` -- runs N environments across multiple processes for true parallelism. Subclasses `gymnasium.vector.AsyncVectorEnv`. Better when per-step computation is significant.

Both classes handle seed broadcasting: passing a single integer seed offsets it per sub-environment automatically.

## Factory Function

- `make_vec_env(task_id, num_envs=8, difficulty="easy", vec_env_cls=SyncVectorAgentickEnv, render_mode="rgb_array", **kwargs)` -- creates a vectorized environment with sensible defaults

## Usage

```python
from agentick.vector import make_vec_env, AsyncVectorAgentickEnv

# Synchronous (default, single process)
env = make_vec_env("GoToGoal-v0", num_envs=8)
obs, info = env.reset(seed=42)  # obs.shape = (8, H, W, 3)

# Asynchronous (multiprocess)
env = make_vec_env(
    "KeyDoorPuzzle-v0",
    num_envs=16,
    vec_env_cls=AsyncVectorAgentickEnv,
    difficulty="medium",
)
```

Vectorized environments are primarily used for RL training throughput, where an agent collects transitions from many environments in parallel.
