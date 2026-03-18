# RL Agents

Train RL agents on Agentick using CleanRL-style scripts (recommended) or Stable-Baselines3. All examples use 8 vectorized environments by default.

## Quick Start (CleanRL)

```python
# See examples/rl/sb3_ppo.py for the full implementation
import gymnasium as gym
from agentick.wrappers import make_atari_env

# make_atari_env: pixels -> resize 84x84 -> grayscale -> frame stack 4
envs = gym.vector.SyncVectorEnv(
    [lambda: make_atari_env("GoToGoal-v0", difficulty="easy",
                            render_mode="rgb_array") for _ in range(8)]
)
# obs shape: (8, 84, 84, 4), uint8
```

## Quick Start (SB3)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from agentick.wrappers import make_atari_env

train_env = DummyVecEnv([lambda: Monitor(make_atari_env("GoToGoal-v0")) for _ in range(8)])
model = PPO("CnnPolicy", train_env, n_steps=128, batch_size=256, learning_rate=2.5e-4)
model.learn(total_timesteps=500_000)
```

## Reward Modes

```python
env = agentick.make("GoToGoal-v0", reward_mode="sparse")  # +1 on success
env = agentick.make("GoToGoal-v0", reward_mode="dense")   # Shaped progress reward
```

## Complete Examples

See `examples/rl/`:

- **CleanRL (recommended)**: `ppo_cleanrl.py`, `dqn_cleanrl.py`, `ppo_pixels.py`, `dqn_pixels.py`
- **SB3**: `sb3_ppo.py`, `sb3_dqn.py`
