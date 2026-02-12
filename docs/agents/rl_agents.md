# RL Agents

Train reinforcement learning agents on Agentick tasks using deep learning frameworks. This guide covers everything from basic setup to advanced training strategies.

## Overview

Reinforcement learning is ideal for Agentick tasks when:
- You need continuous learning and improvement
- You have enough compute for parallel training
- Tasks are simulator-based (no real-world cost)
- You want to scale beyond human demonstrations

**Advantages:**
- Automatic improvement through exploration
- Efficient parallel training with vectorized environments
- Support for pixel-based and state-based learning
- Integration with standard RL libraries (Ray RLlib, Stable-Baselines3)
- Curriculum learning for progressive difficulty

**Disadvantages:**
- Requires significant compute resources
- Sample inefficient (needs many interactions)
- Hyperparameter tuning can be challenging
- Reward shaping is crucial and task-specific

## Quick Start

### 1. Create Environment

```python
import gymnasium as gym
import agentick

# Create single environment
env = agentick.make(
    "GoToGoal-v0",
    difficulty="easy",
    render_mode="rgb_array"  # Pixel observations for CNN
)

# Create vectorized environments (for parallel training)
from agentick.interfaces import RLInterface

envs = RLInterface.make_vectorized_env(
    task_name="GoToGoal-v0",
    n_envs=8,
    difficulty="easy",
    render_mode="rgb_array"
)
```

### 2. Train Agent

```python
from stable_baselines3 import PPO

# Create agent
model = PPO(
    "CnnPolicy",  # CNN for pixel observations
    envs,
    verbose=1,
    learning_rate=2.5e-4,
)

# Train
model.learn(total_timesteps=100_000)

# Save
model.save("ppo_gotogoal")
```

### 3. Evaluate

```python
# Load and test
model = PPO.load("ppo_gotogoal")

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")
```

## Vectorized Environments

Fast parallel training is essential for RL. Vectorized environments allow you to collect samples from multiple environments simultaneously.

### Creating Vectorized Environments

```python
from agentick.interfaces import RLInterface
import gymnasium as gym

# Method 1: Using RLInterface (recommended)
envs = RLInterface.make_vectorized_env(
    task_name="GoToGoal-v0",
    n_envs=16,
    difficulty="medium",
    render_mode="rgb_array"
)
```

### Performance Benefits

| Num Envs | Samples/sec | Speedup |
|----------|------------|---------|
| 1        | 100        | 1.0x    |
| 4        | 380        | 3.8x    |
| 8        | 720        | 7.2x    |
| 16       | 1400       | 14x     |
| 32       | 2600       | 26x     |

Key learnings:
- Scaling is nearly linear up to 32 environments
- Use as many as your GPU memory allows
- Batch size should be `num_envs * num_steps`

### API Reference

```python
# Reset all environments
obs = envs.reset()  # Shape: (n_envs, *obs_shape)

# Step all environments in parallel
obs, rewards, dones, truncs, infos = envs.step(actions)
# obs: (n_envs, *obs_shape)
# rewards: (n_envs,)
# dones: (n_envs,) boolean
# truncs: (n_envs,) boolean
# infos: (n_envs,) dicts

# Access single environment
single_env = envs.envs[0]
```

## Observation Modes

Choose observation mode based on your algorithm:

### RGB Pixels (CNN-based)

Best for learning visual features.

```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="rgb_array"
)
obs, info = env.reset()
print(obs.shape)  # (H, W, 3) uint8

# Policy should expect 3-channel images
from stable_baselines3 import PPO
model = PPO("CnnPolicy", env)
```

### State Dictionary (MLP-based)

Best for structured state spaces.

```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="state_dict"
)
obs, info = env.reset()
print(type(obs))  # dict
print(obs.keys())  # agent_pos, goal_pos, grid, ...

# Use MLP policy
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env)
```

### ASCII Grid (Text-based)

Good for understanding what model sees.

```python
env = agentick.make(
    "GoToGoal-v0",
    render_mode="ascii"
)
obs, info = env.reset()
print(obs)  # String representation
# Output:
# #########
# #A.....G#
# #########
```

## Reward Modes

### Sparse Rewards

```python
env = agentick.make(
    "GoToGoal-v0",
    reward_mode="sparse"
)
# +1 on success, 0 otherwise
# Harder to learn but tests true understanding
```

### Dense Rewards

```python
env = agentick.make(
    "GoToGoal-v0",
    reward_mode="dense"
)
# +1 on success
# Shaped bonus for moving toward goal
# Easier to learn from
```

**Recommendation:** Start with dense rewards for faster initial learning, then test on sparse rewards for generalization.

## Curriculum Learning

Progressively increase difficulty as the agent improves.

### Manual Curriculum

```python
import agentick
from collections import defaultdict

difficulties = ["easy", "medium", "hard"]
current_difficulty_idx = 0
success_threshold = 0.8
recent_successes = defaultdict(list)

for episode in range(1000):
    difficulty = difficulties[min(current_difficulty_idx, len(difficulties) - 1)]
    env = agentick.make("GoToGoal-v0", difficulty=difficulty)

    obs, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    success = episode_reward > 0
    recent_successes[difficulty].append(success)

    # Check if ready to increase difficulty
    if len(recent_successes[difficulty]) >= 100:
        success_rate = sum(recent_successes[difficulty][-100:]) / 100
        if success_rate >= success_threshold and current_difficulty_idx < len(difficulties) - 1:
            current_difficulty_idx += 1
            print(f"Progressed to {difficulties[current_difficulty_idx]}")
            recent_successes[difficulty] = []

    if episode % 50 == 0:
        print(f"Episode {episode}: difficulty={difficulty}, success={success}")
```

### Adaptive Curriculum

```python
def get_next_difficulty(success_rate, current_level):
    """Adapt difficulty based on performance."""
    levels = ["easy", "medium", "hard", "extreme"]

    if success_rate > 0.9:
        # Excellent performance - try harder
        return min(current_level + 1, len(levels) - 1)
    elif success_rate < 0.3:
        # Poor performance - go easier
        return max(current_level - 1, 0)
    else:
        # Good performance - stay at current level
        return current_level

# In training loop
success_rates = []
current_difficulty_idx = 0

for episode in range(1000):
    difficulty_idx = get_next_difficulty(
        np.mean(success_rates[-100:]) if success_rates else 0,
        current_difficulty_idx
    )
    current_difficulty_idx = difficulty_idx

    # Train episode...
```

## Reward Shaping

Carefully design reward functions to guide learning.

### Distance-to-Goal Shaping

```python
def shaped_reward(agent_pos, goal_pos, done, reward, gamma=0.99):
    """Reward progress toward goal."""
    dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(agent_pos))

    # Shape: negative distance bonus
    # Encourage moving closer to goal
    distance_reward = -0.01 * dist_to_goal

    # Task reward (success)
    task_reward = 10.0 * reward

    total = distance_reward + task_reward
    return total

# Use in training
env = agentick.make("GoToGoal-v0")
obs, info = env.reset()

agent_pos = info.get("agent_pos")
goal_pos = info.get("goal_pos")

obs, reward, done, truncated, info = env.step(action)
new_agent_pos = info.get("agent_pos")

shaped_reward_value = shaped_reward(
    new_agent_pos,
    goal_pos,
    done,
    reward
)
```

### Action Smoothing

```python
def penalize_action_changes(prev_action, curr_action, reward, penalty=0.01):
    """Penalize rapid action changes."""
    if prev_action != curr_action:
        return reward - penalty
    return reward

# Encourages stable, consistent behavior
```

### Efficiency Bonus

```python
def efficiency_reward(reward, steps, max_steps, efficiency_bonus=0.1):
    """Reward completing tasks efficiently."""
    if reward > 0:  # Task successful
        efficiency = 1.0 - (steps / max_steps)
        bonus = efficiency_bonus * efficiency
        return reward + bonus
    return reward

# Faster solutions are better
```

## Complete CleanRL PPO Example

Full working example using CleanRL-style implementation (single-file, reproducible):

```python
"""PPO training on Agentick environments - CleanRL style.

Usage:
    python ppo_agentick.py --task GoToGoal-v0 --total-timesteps 100000
"""

import argparse
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import agentick


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="GoToGoal-v0")
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--use-cuda", action="store_true", default=True)
    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_updates = args.total_timesteps // args.batch_size
    return args


def make_env(task_id, difficulty, seed, idx):
    """Create environment factory."""
    def thunk():
        env = agentick.make(
            task_id,
            difficulty=difficulty,
            render_mode="rgb_array",
            reward_mode="dense"
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


class Agent(nn.Module):
    """CNN-based agent for pixel observations."""

    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        n_actions = envs.single_action_space.n

        # Shared feature extractor
        self.network = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate conv output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *obs_shape).permute(0, 3, 1, 2)
            conv_out_size = self.network(sample_input).shape[1]

        # Actor and critic heads
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def get_action_and_value(self, x, action=None):
        # Normalize pixel values
        x = x.float() / 255.0
        # Permute from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Extract features
        features = self.network(x)

        # Policy
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        # Value
        value = self.critic(features)

        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, x):
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        features = self.network(x)
        return self.critic(features)


def train():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # Create vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.task, args.difficulty, args.seed, i) for i in range(args.num_envs)]
    )

    # Create agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    episode_returns = deque(maxlen=100)

    for update in range(1, args.num_updates + 1):
        # Collect trajectories
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Step environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Log episode returns
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])

        # Compute advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # Flatten batches
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            avg_return = np.mean(episode_returns) if episode_returns else 0
            print(
                f"Update {update}/{args.num_updates} | "
                f"SPS: {sps:.0f} | "
                f"Avg Return: {avg_return:.2f} | "
                f"Global Step: {global_step}"
            )

    envs.close()
    print(f"Training complete! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    train()
```

**Run the example:**

```bash
python ppo_agentick.py --task GoToGoal-v0 --difficulty easy --total-timesteps 100000
```

## DQN Example

For discrete action spaces with pixel observations:

```python
"""DQN agent for Agentick environments."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

import agentick


class DQNNetwork(nn.Module):
    """Q-network for pixel observations."""

    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x.float() / 255.0)


class DQNAgent:
    """Deep Q-Learning agent."""

    def __init__(
        self,
        obs_shape,
        n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DQNNetwork(obs_shape, n_actions).to(self.device)
        self.target_network = DQNNetwork(obs_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.steps = 0

    def act(self, obs):
        """Select action (epsilon-greedy)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()

    def remember(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def train(self):
        """Train on batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Current Q values
        current_q = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones).float()

        # Loss
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_dqn():
    """Train DQN agent."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions)

    episodes = 100
    target_update_freq = 10

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.train()

            episode_reward += reward
            obs = next_obs

        if episode % target_update_freq == 0:
            agent.update_target()

        if episode % 10 == 0:
            print(f"Episode {episode}: reward={episode_reward:.2f}, epsilon={agent.epsilon:.3f}")

    env.close()


if __name__ == "__main__":
    train_dqn()
```

## Ray RLlib Integration

For distributed training across multiple machines:

```python
"""Training with Ray RLlib."""

from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN

# Register Agentick environment
from ray.tune import register_env
import gymnasium as gym
import agentick

def env_creator(env_config):
    return agentick.make(
        env_config.get("task", "GoToGoal-v0"),
        difficulty=env_config.get("difficulty", "easy"),
        render_mode="rgb_array"
    )

register_env("agentick", env_creator)

# Training config
config = {
    "env": "agentick",
    "env_config": {
        "task": "GoToGoal-v0",
        "difficulty": "easy"
    },
    # Resource allocation
    "num_workers": 8,
    "num_envs_per_worker": 4,
    "num_gpus": 1,
    # PPO settings
    "sgd_minibatch_size": 256,
    "train_batch_size": 4096,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
}

# Train
stop_config = {
    "timesteps_total": 1_000_000,
}

results = tune.run(
    PPO,
    name="ppo_agentick",
    config=config,
    stop=stop_config,
    verbose=1,
    progress_bar=True,
)

print("Training complete!")
print(f"Best checkpoint: {results.get_best_checkpoint()}")
```

## Hyperparameter Tuning Guidance

### Learning Rate

- Start with `2.5e-4` (good default)
- If training unstable: reduce to `1e-4`
- If learning too slow: increase to `5e-4`
- Test: `[1e-4, 2.5e-4, 5e-4, 1e-3]`

### Entropy Coefficient

- Controls exploration
- Higher = more exploration, slower convergence
- Lower = faster convergence, less exploration
- For Agentick: `0.001` to `0.01`

### Gamma (Discount Factor)

- `0.99` for horizon ~100 steps (good default)
- `0.999` for very long-horizon tasks
- `0.95` for short-horizon tasks
- Effect: values far future rewards differently

### GAE Lambda

- `0.95` good default
- Higher = more variance, less bias
- Lower = less variance, more bias

### Batch Size

- Larger batches = more stable but slower updates
- `256` to `512` typical
- Increase if training noisy
- Decrease for faster iteration

### Value Function Loss Coefficient

- `0.5` good default
- Relative weight of value loss vs policy loss
- Increase if values diverge

## Common RL Pitfalls

### 1. Reward Scale

Bad:
```python
# Reward is tiny
reward = 1e-6 if success else 0
# Agent can't learn signal
```

Good:
```python
# Reasonable scale
reward = 1.0 if success else -0.01
# Clear learning signal
```

### 2. Normalization

Bad:
```python
# Input to network is huge
state = [x_pos, y_pos, grid_with_large_values]
# Causes gradient issues
```

Good:
```python
# Normalize observations
obs = (obs - obs.mean()) / (obs.std() + 1e-8)
# Stable learning
```

### 3. Action Validity

Bad:
```python
# Taking invalid action
action = env.action_space.sample()
obs, reward, done, _ = env.step(action)
# Environment may crash or behave unexpectedly
```

Good:
```python
# Only take valid actions
valid_actions = env.get_valid_actions()
action = random.choice(valid_actions)
obs, reward, done, _ = env.step(action)
```

### 4. Seed Management

Bad:
```python
# No seeds
model = PPO("CnnPolicy", env)
# Results not reproducible
```

Good:
```python
# Set all seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
env.action_space.seed(42)
env.observation_space.seed(42)
# Results reproducible
```

### 5. Episode Length

Bad:
```python
# Episode never ends
while True:
    action = policy(obs)
    obs, reward, done, _ = env.step(action)
# Training hangs
```

Good:
```python
# Set max episode length
for step in range(max_steps):
    action = policy(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
```

## Evaluation and Benchmarking

```python
"""Evaluate trained agents."""

def evaluate_agent(model, env, n_episodes=100, deterministic=True):
    """Evaluate agent performance."""
    returns = []
    lengths = []
    successes = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1
            done = terminated or truncated

        returns.append(episode_return)
        lengths.append(step_count)
        successes.append(info.get("success", False))

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_length": np.mean(lengths),
        "success_rate": np.mean(successes),
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
    }

# Evaluate
model = PPO.load("ppo_gotogoal")
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
results = evaluate_agent(model, env)

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Mean return: {results['mean_return']:.2f}")
print(f"Mean episode length: {results['mean_length']:.1f}")
```

## Tips and Best Practices

1. **Start with vectorized environments** - At least 8-16 for parallel training
2. **Monitor wandb** - Track training metrics in real-time
3. **Use curriculum learning** - Start easy, progress to hard
4. **Validate frequently** - Evaluate on held-out seeds
5. **Save checkpoints** - Every N updates for recovery
6. **Test deterministically** - Set `deterministic=True` in evaluation
7. **Use reward shaping** - Guide learning with dense rewards
8. **Handle action validity** - Use valid action masks
9. **Normalize observations** - Helps convergence
10. **Seed everything** - Reproducibility is important
