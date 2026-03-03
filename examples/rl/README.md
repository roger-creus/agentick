# Reinforcement Learning

RL training examples covering PPO and DQN in two flavours: CleanRL-style
single-file implementations (recommended) and Stable Baselines3 (SB3) wrappers.
All scripts use vectorized environments (8 parallel envs by default) with CNN
policies on pixel observations via the Agentick Atari-style preprocessing wrapper.

## Prerequisites

```bash
uv sync --extra rl          # stable-baselines3, torch, gymnasium
uv sync --extra rl --extra all  # adds wandb for experiment tracking
```

GPU is strongly recommended. Training runs range from 50k to 500k timesteps.

## Scripts

### CleanRL-style (recommended, single file, no SB3 dependency)

- **ppo_cleanrl.py** -- Full PPO implementation with dataclass config, Nature
  CNN actor-critic, GAE advantage estimation, wandb integration, periodic
  evaluation, video recording, checkpointing, and learning curve plots.
  8 vectorized envs, 128 steps per rollout.
- **dqn_cleanrl.py** -- Full DQN implementation with vectorized data collection,
  replay buffer, target network, epsilon decay, wandb logging, evaluation,
  checkpointing, and learning curve plots. 8 vectorized envs.
- **ppo_pixels.py** -- Minimal CleanRL PPO for pixel observations. Accepts CLI
  arguments for task, difficulty, seed, and all hyperparameters. Good starting
  point for custom experiments. 8 vectorized envs, 128 steps per rollout.
- **dqn_pixels.py** -- Minimal CleanRL DQN for pixel observations. Vectorized
  data collection, replay buffer, target network, epsilon-greedy schedule.
  8 vectorized envs.

### Stable Baselines3

- **sb3_ppo.py** -- PPO with CnnPolicy via SB3. Includes wandb logging,
  periodic evaluation with video recording, model checkpointing, and a final
  evaluation loop. 8 vectorized envs.
- **sb3_dqn.py** -- DQN with CnnPolicy via SB3. Same structure as sb3_ppo
  (wandb, video, checkpoints) but uses an experience replay buffer and
  epsilon-greedy exploration. 8 vectorized envs.

## Running

```bash
# CleanRL-style (recommended)
uv run python examples/rl/ppo_cleanrl.py
uv run python examples/rl/dqn_cleanrl.py

# Minimal single-file scripts with CLI args
uv run python examples/rl/ppo_pixels.py --task GoToGoal-v0 --difficulty easy --total-timesteps 100000
uv run python examples/rl/dqn_pixels.py --task GoToGoal-v0 --difficulty easy --total-timesteps 50000

# Stable Baselines3
uv run python examples/rl/sb3_ppo.py
uv run python examples/rl/sb3_dqn.py
```

## Output Locations

- Checkpoints: `checkpoints/` (configurable)
- Videos: `videos/` (recorded during evaluation)
- TensorBoard logs: `./logs/` (SB3 scripts)
- Learning curve plots: saved alongside checkpoints (CleanRL scripts)
- Wandb dashboards: logged automatically when wandb is installed
