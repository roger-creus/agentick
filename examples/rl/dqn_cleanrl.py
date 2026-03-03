"""
CleanRL DQN training example with wandb logging, evaluation, video recording, and checkpoints.

This example demonstrates:
- DQN training from scratch using CleanRL-style implementation
- Vectorized environments for parallel data collection
- Integration with wandb for experiment tracking
- Periodic evaluation with video recording
- Model checkpointing
- Experience replay buffer

Requirements:
    uv sync --extra rl --extra all

Usage:
    uv run python examples/rl/dqn_cleanrl.py
"""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agentick.wrappers.atari_preprocessing import make_atari_env

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: uv sync --extra all")


@dataclass
class DQNConfig:
    """DQN hyperparameters."""

    # Environment
    env_id: str = "GoToGoal-v0"
    num_envs: int = 8

    # Training
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10_000
    batch_size: int = 128
    gamma: float = 0.99
    target_network_frequency: int = 1000
    tau: float = 1.0
    learning_starts: int = 10_000
    train_frequency: int = 4

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000

    # Evaluation
    eval_freq: int = 50_000
    eval_episodes: int = 10
    record_video: bool = True

    # Logging
    log_freq: int = 100
    use_wandb: bool = True
    wandb_project: str = "agentick-dqn"
    wandb_entity: str | None = None

    # Checkpointing
    checkpoint_freq: int = 100_000
    checkpoint_dir: str = "checkpoints/dqn"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1


class QNetwork(nn.Module):
    """Q-network with Nature CNN architecture for visual observations."""

    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()

        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            h, w, c = obs_space.shape
            in_channels = c

            self.network = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                dummy_input = torch.zeros(1, c, h, w)
                dummy_output = self.network(dummy_input)
                feature_dim = dummy_output.shape[1]

            self.fc = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
            )
        else:
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = int(np.prod(obs_space["text"].shape))
            else:
                obs_dim = int(np.prod(obs_space.shape))

            self.network = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        if self.fc is not None:
            return self.fc(features)
        return features


class ReplayBuffer:
    """Experience replay buffer with efficient uint8 storage."""

    def __init__(self, capacity: int, obs_shape: tuple, device: str):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros((capacity,), dtype=torch.long)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.uint8)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = torch.from_numpy(np.array(obs).astype(np.uint8))
        self.next_observations[self.pos] = torch.from_numpy(np.array(next_obs).astype(np.uint8))
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = int(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs_batch, next_obs_batch, actions, rewards, dones):
        """Add a batch of transitions (one per env)."""
        batch_size = len(actions)
        for i in range(batch_size):
            self.add(obs_batch[i], next_obs_batch[i], actions[i], rewards[i], dones[i])

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices].float().to(self.device) / 255.0,
            self.next_observations[indices].float().to(self.device) / 255.0,
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.dones[indices].float().to(self.device),
        )


def make_env(env_id: str, seed: int, idx: int):
    """Create a single environment with Atari preprocessing."""

    def thunk():
        env = make_atari_env(env_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def _format_obs(obs, device):
    """Convert observation to tensor: (N, H, W, C) -> (N, C, H, W), normalized."""
    if isinstance(obs, dict):
        if "rgb" in obs:
            obs_array = np.transpose(obs["rgb"], (0, 3, 1, 2)) / 255.0
            return torch.FloatTensor(obs_array).to(device)
        return torch.FloatTensor(obs["text"].reshape(obs["text"].shape[0], -1)).to(device)
    if len(obs.shape) == 4:
        obs_array = np.transpose(obs, (0, 3, 1, 2)) / 255.0
        return torch.FloatTensor(obs_array).to(device)
    return torch.FloatTensor(obs.reshape(obs.shape[0], -1)).to(device) / 255.0


def _format_obs_single(obs, device):
    """Convert single observation to tensor for evaluation."""
    if isinstance(obs, dict):
        if "rgb" in obs:
            obs_array = obs["rgb"].transpose(2, 0, 1) / 255.0
            return torch.FloatTensor(obs_array).unsqueeze(0).to(device)
        return torch.FloatTensor(obs["text"].flatten()).unsqueeze(0).to(device)
    if len(obs.shape) == 3:
        obs_array = obs.transpose(2, 0, 1) / 255.0
        return torch.FloatTensor(obs_array).unsqueeze(0).to(device)
    return torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device) / 255.0


def _format_obs_for_buffer(obs):
    """Convert observation to (C, H, W) uint8 for replay buffer storage."""
    if isinstance(obs, dict):
        if "rgb" in obs:
            return obs["rgb"].transpose(2, 0, 1)
        return obs["text"].flatten()
    if len(obs.shape) == 3:
        return obs.transpose(2, 0, 1)
    return obs.flatten()


def evaluate(
    q_network: QNetwork,
    env_id: str,
    num_episodes: int,
    device: str,
    record_video: bool = False,
    video_dir: str | None = None,
) -> dict:
    """Evaluate the agent."""
    eval_env = make_atari_env(env_id, seed=42)

    if record_video and video_dir:
        try:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            eval_env = gym.wrappers.RecordVideo(
                eval_env,
                video_folder=video_dir,
                episode_trigger=lambda x: x < 3,
            )
        except Exception:
            pass

    returns = []
    lengths = []
    successes = []

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        while not done:
            obs_tensor = _format_obs_single(obs, device)
            with torch.no_grad():
                q_values = q_network(obs_tensor)
                action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

        if "success" in info:
            successes.append(float(info["success"]))

    eval_env.close()

    results = {
        "eval/mean_return": np.mean(returns),
        "eval/std_return": np.std(returns),
        "eval/mean_length": np.mean(lengths),
        "eval/std_length": np.std(lengths),
    }

    if successes:
        results["eval/success_rate"] = np.mean(successes)

    return results


def train(config: DQNConfig):
    """Train DQN agent."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    run_name = f"{config.env_id.split('/')[-1]}_{int(time.time())}"
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=vars(config),
            sync_tensorboard=False,
        )

    # Create vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, config.seed, i) for i in range(config.num_envs)]
    )

    # Get observation shape for buffer (C, H, W)
    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        if "rgb" in envs.single_observation_space.spaces:
            obs_shape = envs.single_observation_space["rgb"].shape
            obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        else:
            obs_shape = (int(np.prod(envs.single_observation_space["text"].shape)),)
    elif len(envs.single_observation_space.shape) == 3:
        h, w, c = envs.single_observation_space.shape
        obs_shape = (c, h, w)
    else:
        obs_shape = (int(np.prod(envs.single_observation_space.shape)),)

    # Create networks
    q_network = QNetwork(envs.single_observation_space, envs.single_action_space).to(config.device)
    target_network = QNetwork(envs.single_observation_space, envs.single_action_space).to(
        config.device
    )
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)

    # Create replay buffer
    replay_buffer = ReplayBuffer(config.buffer_size, obs_shape, config.device)

    # Learning curve tracking
    learning_curve = {"steps": [], "returns": [], "lengths": []}

    # Training loop
    obs, _ = envs.reset(seed=config.seed)
    episode_count = 0

    print(f"Training DQN on {config.env_id}")
    print(f"Device: {config.device}")
    print(f"Num envs: {config.num_envs}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Wandb: {config.use_wandb and WANDB_AVAILABLE}")
    print()

    for global_step in range(config.total_timesteps):
        # Epsilon decay
        epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * max(
            0, 1 - global_step / config.epsilon_decay_steps
        )

        # Batched epsilon-greedy action selection
        obs_tensor = _format_obs(obs, config.device)

        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(config.num_envs)]
            )
        else:
            with torch.no_grad():
                q_values = q_network(obs_tensor)
                actions = q_values.argmax(dim=1).cpu().numpy()

        # Step all environments
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Handle auto-reset: use final_observation for terminal transitions
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx in range(config.num_envs):
                if dones[idx] and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]

        # Add all transitions to replay buffer
        for idx in range(config.num_envs):
            obs_fmt = _format_obs_for_buffer(obs[idx])
            next_obs_fmt = _format_obs_for_buffer(real_next_obs[idx])
            replay_buffer.add(obs_fmt, next_obs_fmt, actions[idx], rewards[idx], dones[idx])

        obs = next_obs

        # Log episode returns
        if "final_info" in infos:
            for item in infos["final_info"]:
                if item and "episode" in item:
                    episode_count += 1
                    learning_curve["steps"].append(global_step)
                    learning_curve["returns"].append(item["episode"]["r"])
                    learning_curve["lengths"].append(item["episode"]["l"])

                    if episode_count % config.log_freq == 0:
                        print(
                            f"Step {global_step:,}: episode={episode_count}, "
                            f"return={item['episode']['r']:.2f}, "
                            f"length={item['episode']['l']}, epsilon={epsilon:.3f}"
                        )

                    if config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(
                            {
                                "train/episode_return": item["episode"]["r"],
                                "train/episode_length": item["episode"]["l"],
                                "train/epsilon": epsilon,
                                "train/global_step": global_step,
                            }
                        )

        # Training
        if (
            replay_buffer.size >= config.learning_starts
            and global_step % config.train_frequency == 0
        ):
            obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = (
                replay_buffer.sample(config.batch_size)
            )

            with torch.no_grad():
                target_q_values = target_network(next_obs_batch)
                target_max = target_q_values.max(dim=1)[0]
                target = rewards_batch + config.gamma * target_max * (1 - dones_batch)

            current_q_values = q_network(obs_batch)
            current = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

            loss = F.mse_loss(current, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % (config.log_freq * config.train_frequency) == 0:
                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/q_value": current.mean().item(),
                            "train/global_step": global_step,
                        }
                    )

        # Update target network
        if global_step % config.target_network_frequency == 0:
            if config.tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
            else:
                for target_param, param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )

        # Evaluation
        if global_step > 0 and global_step % config.eval_freq == 0:
            video_dir = None
            if config.record_video:
                video_dir = f"{config.checkpoint_dir}/videos/step_{global_step}"

            eval_results = evaluate(
                q_network,
                config.env_id,
                config.eval_episodes,
                config.device,
                config.record_video,
                video_dir,
            )

            print(f"\nEvaluation at step {global_step:,}:")
            for key, value in eval_results.items():
                print(f"  {key}: {value:.2f}")
            print()

            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({**eval_results, "train/global_step": global_step})

        # Checkpointing
        if global_step > 0 and global_step % config.checkpoint_freq == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{global_step}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "global_step": global_step,
                    "q_network_state_dict": q_network.state_dict(),
                    "target_network_state_dict": target_network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(config),
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    print("\nFinal evaluation...")
    video_dir = None
    if config.record_video:
        video_dir = f"{config.checkpoint_dir}/videos/final"

    final_results = evaluate(
        q_network,
        config.env_id,
        config.eval_episodes * 2,
        config.device,
        config.record_video,
        video_dir,
    )

    print("\nFinal results:")
    for key, value in final_results.items():
        print(f"  {key}: {value:.2f}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.log({**final_results, "train/global_step": global_step})
        wandb.finish()

    # Save final model
    final_model_path = Path(config.checkpoint_dir) / "final_model.pt"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "q_network_state_dict": q_network.state_dict(),
            "target_network_state_dict": target_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
        },
        final_model_path,
    )
    print(f"\nFinal model saved: {final_model_path}")

    # Save and plot learning curves
    if learning_curve["steps"]:
        learning_curve_path = Path(config.checkpoint_dir) / "learning_curve.npz"
        np.savez(
            learning_curve_path,
            steps=np.array(learning_curve["steps"]),
            returns=np.array(learning_curve["returns"]),
            lengths=np.array(learning_curve["lengths"]),
        )
        print(f"Learning curve data saved: {learning_curve_path}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(learning_curve["steps"], learning_curve["returns"], alpha=0.6)
        window = min(50, len(learning_curve["returns"]) // 10)
        if window > 1:
            smoothed_returns = np.convolve(
                learning_curve["returns"], np.ones(window) / window, mode="valid"
            )
            smoothed_steps = learning_curve["steps"][window - 1 :]
            ax1.plot(smoothed_steps, smoothed_returns, linewidth=2, label="Smoothed")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Episode Return")
        ax1.set_title("Training Returns")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(learning_curve["steps"], learning_curve["lengths"], alpha=0.6)
        if window > 1:
            smoothed_lengths = np.convolve(
                learning_curve["lengths"], np.ones(window) / window, mode="valid"
            )
            ax2.plot(smoothed_steps, smoothed_lengths, linewidth=2, label="Smoothed")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Episode Length")
        ax2.set_title("Episode Lengths")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = Path(config.checkpoint_dir) / "learning_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Learning curve plot saved: {plot_path}")
        plt.close()

    envs.close()


if __name__ == "__main__":
    config = DQNConfig()
    train(config)
