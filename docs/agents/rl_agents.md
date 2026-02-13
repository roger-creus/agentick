# RL Agents

Train RL agents on Agentick using CleanRL and Stable-Baselines3.

## Quick Start with Stable-Baselines3

```python
from stable_baselines3 import PPO
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
model = PPO("CnnPolicy", env, learning_rate=2.5e-4)
model.learn(total_timesteps=100_000)
model.save("ppo_gotogoal")

# Evaluate
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

## Observation Modes

```python
# RGB for CNN-based
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
model = PPO("CnnPolicy", env)

# State dict for MLP-based
env = agentick.make("GoToGoal-v0", render_mode="state_dict")
model = PPO("MlpPolicy", env)
```

## Vectorized Environments

```python
import gymnasium as gym
from stable_baselines3 import PPO

def make_env(task_id, difficulty, seed, idx):
    def thunk():
        env = agentick.make(task_id, difficulty=difficulty, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk

# Create 8 parallel environments
envs = gym.vector.SyncVectorEnv([
    make_env("GoToGoal-v0", "easy", 42, i) for i in range(8)
])

model = PPO("CnnPolicy", envs)
model.learn(total_timesteps=100_000)
```

## Stable-Baselines3 Examples

### PPO

```python
from stable_baselines3 import PPO

env = agentick.make("MazeNavigation-v0", difficulty="medium", render_mode="rgb_array")
model = PPO("CnnPolicy", env, learning_rate=2.5e-4, n_steps=2048, batch_size=64)
model.learn(total_timesteps=500_000)
model.save("ppo_maze")
```

### DQN

```python
from stable_baselines3 import DQN

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
model = DQN("CnnPolicy", env, learning_rate=1e-4, buffer_size=10000)
model.learn(total_timesteps=100_000)
model.save("dqn_gotogoal")
```

## CleanRL Examples

See `examples/rl/` for complete implementations:

- `examples/rl/ppo_cleanrl.py` - PPO with wandb, checkpoints
- `examples/rl/dqn_cleanrl.py` - DQN with replay buffer
- `examples/rl/curriculum_training.py` - Adaptive curriculum

Run CleanRL PPO:
```bash
uv run examples/rl/ppo_cleanrl.py --task GoToGoal-v0 --total-timesteps 100000
```

## Reward Modes

```python
# Sparse: +1.0 on success
env = agentick.make("GoToGoal-v0", reward_mode="sparse")

# Dense: shaped rewards
env = agentick.make("GoToGoal-v0", reward_mode="dense")
```

## Curriculum Learning

```python
from agentick.wrappers import CurriculumWrapper

env = agentick.make("GoToGoal-v0", difficulty="easy")
env = CurriculumWrapper(env, success_threshold=0.8, window_size=10)

model = PPO("CnnPolicy", env)
model.learn(total_timesteps=500_000)
```

## Evaluation

```python
def evaluate_agent(model, env, n_episodes=100):
    returns, successes = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_return, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        successes.append(info.get("success", False))
    return {
        "mean_return": sum(returns) / len(returns),
        "success_rate": sum(successes) / len(successes)
    }

model = PPO.load("ppo_gotogoal")
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
results = evaluate_agent(model, env)
print(f"Success rate: {results['success_rate']:.2%}")
```

## Complete Examples

See `examples/rl/`:
- **Stable-Baselines3**: `sb3_ppo.py`, `sb3_dqn.py`
- **CleanRL**: `ppo_cleanrl.py`, `dqn_cleanrl.py`
- **Curriculum**: `curriculum_training.py`
