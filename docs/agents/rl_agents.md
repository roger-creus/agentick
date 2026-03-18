# RL Agents

Train RL agents on Agentick using CleanRL-style single-file scripts or Stable-Baselines3. All pixel-based examples use the standard Atari preprocessing pipeline: isometric 512x512 -> resize 84x84 -> grayscale -> 4-frame stack.

## Quick Start (CleanRL)

```python
# See examples/rl/ppo_cleanrl.py for the full implementation
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

- **CleanRL**: `ppo_cleanrl.py`, `dqn_cleanrl.py` — single-file, hackable, TensorBoard logging
- **SB3**: `sb3_ppo.py`, `sb3_dqn.py` — higher-level API, wandb integration, checkpointing

```bash
# CleanRL PPO (default: GoToGoal-v0, easy, dense, 500k steps)
uv run python examples/rl/ppo_cleanrl.py

# CleanRL PPO on a harder task
uv run python examples/rl/ppo_cleanrl.py --task-id MazeNavigation-v0 --difficulty medium

# CleanRL DQN
uv run python examples/rl/dqn_cleanrl.py

# SB3 PPO
uv run python examples/rl/sb3_ppo.py
```
