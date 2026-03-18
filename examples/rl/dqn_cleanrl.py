"""
CleanRL-style DQN with CNN Q-network for Agentick pixel observations.

Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
Uses the standard Atari preprocessing pipeline: 512x512 isometric -> 84x84 grayscale -> 4-frame stack.

Requirements:
    uv sync --extra rl

Usage:
    uv run python examples/rl/dqn_cleanrl.py
    uv run python examples/rl/dqn_cleanrl.py --task-id MazeNavigation-v0 --difficulty medium
    uv run python examples/rl/dqn_cleanrl.py --total-timesteps 500000 --track
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import agentick
from agentick.wrappers import make_atari_env


@dataclass
class Args:
    exp_name: str = "dqn_cleanrl"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "agentick-rl"
    """the wandb's project name"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Agentick-specific
    task_id: str = "GoToGoal-v0"
    """the Agentick task to train on"""
    difficulty: str = "easy"
    """difficulty level (easy, medium, hard, expert)"""
    reward_mode: str = "dense"
    """reward mode (sparse or dense)"""

    # Algorithm specific arguments
    total_timesteps: int = 500_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of total-timesteps for epsilon decay"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


class ReplayBuffer:
    """Simple numpy replay buffer for DQN."""

    def __init__(self, buffer_size, obs_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.buffer_size if self.full else self.pos
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.tensor(self.observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_observations[idxs], dtype=torch.float32).to(
                self.device
            ),
            torch.tensor(self.actions[idxs], dtype=torch.long).to(self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return self.buffer_size if self.full else self.pos


def make_env(task_id, difficulty, reward_mode, seed, idx, capture_video, run_name):
    def thunk():
        env = make_atari_env(
            task_id,
            seed=seed + idx,
            difficulty=difficulty,
            reward_mode=reward_mode,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


class QNetwork(nn.Module):
    """Nature CNN Q-network (Mnih et al., 2015) for 84x84x4 frame-stacked observations."""

    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def parse_args():
    """Parse CLI args, falling back to Args defaults."""
    import argparse

    defaults = Args()
    parser = argparse.ArgumentParser(description="CleanRL DQN for Agentick")
    for field_name, field_val in vars(defaults).items():
        field_type = type(field_val) if field_val is not None else str
        if field_type is bool:
            parser.add_argument(
                f"--{field_name.replace('_', '-')}",
                action="store_true",
                default=field_val,
            )
        else:
            parser.add_argument(
                f"--{field_name.replace('_', '-')}",
                type=field_type,
                default=field_val,
            )
    parsed = parser.parse_args()
    args = Args(**vars(parsed))
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.task_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # Environment setup (DQN uses single env)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.task_id,
                args.difficulty,
                args.reward_mode,
                args.seed,
                0,
                args.capture_video,
                run_name,
            )
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    print(f"Task: {args.task_id} ({args.difficulty})")
    print(f"Obs space: {envs.single_observation_space.shape}")
    print(f"Act space: {envs.single_action_space.n}")
    print(f"Device: {device}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Buffer size: {args.buffer_size:,}")
    print()

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy action selection
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episodes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_r = info["episode"]["r"]
                    ep_l = info["episode"]["l"]
                    print(
                        f"global_step={global_step}, "
                        f"episodic_return={ep_r:.2f}, "
                        f"length={ep_l}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", ep_r, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", ep_l, global_step
                    )

        # Handle truncation (use final_observation for bootstrap)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Store transition
        rb.add(
            obs[0],
            real_next_obs[0],
            actions[0],
            rewards[0],
            terminations[0],
        )
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb.sample(
                    args.batch_size
                )
                with torch.no_grad():
                    target_max, _ = target_network(s_next_obs).max(dim=1)
                    td_target = s_rewards + args.gamma * target_max * (1 - s_dones)
                old_val = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", sps, global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if global_step % args.target_network_frequency == 0:
                for target_param, q_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * q_param.data
                        + (1.0 - args.tau) * target_param.data
                    )

        if global_step % 50_000 == 0 and global_step > 0:
            sps = int(global_step / (time.time() - start_time))
            print(
                f"Step {global_step:,}/{args.total_timesteps:,} | "
                f"SPS: {sps} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Buffer: {len(rb):,}"
            )

    envs.close()
    writer.close()
    print(f"\nTraining complete! Logs: runs/{run_name}")
