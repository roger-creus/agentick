"""DQN training on Agentick with pixel observations.

CleanRL-style single-file DQN with vectorized environments and replay buffer.

Usage:
    python examples/rl/dqn_pixels.py --task GoToGoal-v0 --difficulty easy
"""

import argparse
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agentick.wrappers import make_atari_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="GoToGoal-v0")
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-network-frequency", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-frequency", type=int, default=4)
    return parser.parse_args()


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, buffer_size, obs_shape, device):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.obs = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.next_obs = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.from_numpy(self.obs[idxs]).to(self.device),
            torch.from_numpy(self.actions[idxs]).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            torch.from_numpy(self.next_obs[idxs]).to(self.device),
            torch.from_numpy(self.dones[idxs]).to(self.device),
        )


class QNetwork(nn.Module):
    """CNN Q-network for pixel observations."""

    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate conv output size
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, obs_shape[0], obs_shape[1])
            conv_out_size = self.network(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # Normalize
        x = x.float() / 255.0
        # Transpose (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        features = self.network(x)
        return self.fc(features)


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_env(task_id, difficulty, seed, idx, render_mode="rgb_array_flat"):
    def thunk():
        env = make_atari_env(task_id, difficulty=difficulty, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create vectorized envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.task, args.difficulty, args.seed, i) for i in range(args.num_envs)]
    )

    # Initialize networks
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    q_network = QNetwork(obs_shape, n_actions).to(device)
    target_network = QNetwork(obs_shape, n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Replay buffer
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

    # Training
    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = deque(maxlen=10)
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy action selection
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step all envs
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Handle auto-reset: use final_observation for terminal transitions
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx in range(args.num_envs):
                if dones[idx] and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]

        # Add all transitions to replay buffer
        for idx in range(args.num_envs):
            rb.add(obs[idx], actions[idx], rewards[idx], real_next_obs[idx], dones[idx])

        obs = next_obs

        # Log episode returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_returns.append(info["episode"]["r"])
                    if global_step % 1000 == 0:
                        print(
                            f"Step {global_step}: return={info['episode']['r']:.2f}, "
                            f"avg={np.mean(episodic_returns):.2f}, epsilon={epsilon:.2f}"
                        )

        # Training
        if rb.size > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_actions, s_rewards, s_next_obs, s_dones = rb.sample(args.batch_size)

            with torch.no_grad():
                target_max = target_network(s_next_obs).max(dim=1)[0]
                td_target = s_rewards + args.gamma * target_max * (1 - s_dones)

            q_values = q_network(s_obs)
            old_val = q_values.gather(1, s_actions.unsqueeze(1)).squeeze()
            loss = F.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    envs.close()
    print(f"Training complete! Total time: {time.time() - start_time:.2f}s")
