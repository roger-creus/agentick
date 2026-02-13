"""
Curriculum learning for RL agents.

This example demonstrates how to train an RL agent with adaptive curriculum:
- Start with easy tasks
- Gradually increase difficulty based on performance
- Track success rates and adapt progression speed

Requirements:
    uv sync --extra rl

Usage:
    uv run python examples/rl/curriculum_training.py
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agentick.wrappers import make_atari_env


@dataclass
class CurriculumConfig:
    """Curriculum training configuration."""

    env_id: str = "GoToGoal-v0"
    difficulties: list[str] = None

    # Curriculum parameters
    min_success_rate: float = 0.7  # Move to next level at 70% success
    eval_window: int = 20  # Evaluate over last 20 episodes
    min_episodes_per_level: int = 50  # Minimum episodes before advancing

    # Training
    total_episodes: int = 500
    max_steps_per_episode: int = 100
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1

    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = ["easy", "medium", "hard", "expert"]


class SimplePolicy(nn.Module):
    """CNN policy network for curriculum learning with visual observations."""

    def __init__(self, obs_shape: tuple, action_dim: int):
        super().__init__()

        # Check if we have image observations
        if len(obs_shape) == 3:
            # Nature CNN for image observations
            # Input is (H, W, C) but we need (C, H, W) for PyTorch
            h, w, c = obs_shape
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

            # Calculate output size with a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, c, h, w)
                dummy_output = self.network(dummy_input)
                feature_dim = dummy_output.shape[1]

            self.fc = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
            )
            feature_size = 512
        else:
            # Fallback for non-image observations
            obs_dim = int(np.prod(obs_shape))
            self.network = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.fc = None
            feature_size = 128

        self.actor = nn.Linear(feature_size, action_dim)
        self.critic = nn.Linear(feature_size, 1)

    def forward(self, obs):
        features = self.network(obs)
        if self.fc is not None:
            features = self.fc(features)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value


class CurriculumTrainer:
    """Adaptive curriculum trainer."""

    def __init__(self, config: CurriculumConfig):
        self.config = config

        # Initialize environment to get dimensions
        env = make_atari_env(config.env_id, difficulty=config.difficulties[0], render_mode="rgb_array")

        # Get observation shape
        obs, _ = env.reset()

        if isinstance(obs, dict):
            if "rgb" in obs:
                obs_shape = obs["rgb"].shape
            else:
                obs_shape = (int(np.prod(obs["text"].shape)),)
        else:
            obs_shape = obs.shape

        action_dim = env.action_space.n
        env.close()

        # Create policy
        self.policy = SimplePolicy(obs_shape, action_dim).to(config.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        # Curriculum state
        self.current_level = 0
        self.episode_rewards = []
        self.episode_successes = []

        # Set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def get_obs_tensor(self, obs):
        """Convert observation to tensor."""
        if isinstance(obs, dict):
            if "rgb" in obs:
                # Image observation: (H, W, C) -> (C, H, W) and normalize
                obs_array = obs["rgb"].transpose(2, 0, 1) / 255.0
                return torch.FloatTensor(obs_array).unsqueeze(0).to(self.config.device)
            else:
                obs_flat = obs["text"].flatten()
        elif len(obs.shape) == 3:
            # Image observation: (H, W, C) -> (C, H, W) and normalize
            obs_array = obs.transpose(2, 0, 1) / 255.0
            return torch.FloatTensor(obs_array).unsqueeze(0).to(self.config.device)
        else:
            obs_flat = obs.flatten()

        return torch.FloatTensor(obs_flat).unsqueeze(0).to(self.config.device) / 255.0

    def select_action(self, obs_tensor):
        """Select action using policy."""
        logits, value = self.policy(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value

    def run_episode(self, env):
        """Run a single episode."""
        obs, _ = env.reset()
        done = False

        episode_data = {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
        }

        steps = 0
        episode_reward = 0

        while not done and steps < self.config.max_steps_per_episode:
            obs_tensor = self.get_obs_tensor(obs)
            action, log_prob, value = self.select_action(obs_tensor)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data["observations"].append(obs_tensor)
            episode_data["actions"].append(action)
            episode_data["log_probs"].append(log_prob)
            episode_data["rewards"].append(reward)
            episode_data["values"].append(value)

            obs = next_obs
            episode_reward += reward
            steps += 1

        success = info.get("success", False)

        return episode_data, episode_reward, success

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.insert(0, G)

        return torch.FloatTensor(returns).to(self.config.device)

    def update_policy(self, episode_data):
        """Update policy using REINFORCE."""
        returns = self.compute_returns(episode_data["rewards"])

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute loss
        policy_loss = []
        value_loss = []

        for log_prob, value, G in zip(episode_data["log_probs"], episode_data["values"], returns):
            advantage = G - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(value.squeeze(), G))

        loss = torch.stack(policy_loss).mean() + torch.stack(value_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def should_advance_level(self):
        """Check if should advance to next difficulty level."""
        # Need minimum episodes
        if len(self.episode_successes) < self.config.min_episodes_per_level:
            return False

        # Check success rate over recent episodes
        recent_successes = self.episode_successes[-self.config.eval_window :]
        success_rate = np.mean(recent_successes)

        return success_rate >= self.config.min_success_rate

    def train(self):
        """Train with curriculum."""
        print(f"Curriculum Training on {self.config.env_id}")
        print(f"Difficulties: {self.config.difficulties}")
        print(f"Device: {self.config.device}")
        print(f"Total episodes: {self.config.total_episodes}")
        print()

        episode_count = 0

        while episode_count < self.config.total_episodes:
            # Get current difficulty
            difficulty = self.config.difficulties[self.current_level]

            # Create environment
            env = make_atari_env(self.config.env_id, difficulty=difficulty, render_mode="rgb_array")

            # Run episode
            episode_data, episode_reward, success = self.run_episode(env)

            # Update policy
            loss = self.update_policy(episode_data)

            # Track statistics
            self.episode_rewards.append(episode_reward)
            self.episode_successes.append(float(success))

            episode_count += 1

            # Log progress
            if episode_count % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_successes = self.episode_successes[-10:]

                print(
                    f"Episode {episode_count} | "
                    f"Level: {difficulty} | "
                    f"Reward: {np.mean(recent_rewards):.2f} | "
                    f"Success: {np.mean(recent_successes):.2%} | "
                    f"Loss: {loss:.4f}"
                )

            # Check if should advance level
            if (
                self.should_advance_level()
                and self.current_level < len(self.config.difficulties) - 1
            ):
                old_level = self.config.difficulties[self.current_level]
                self.current_level += 1
                new_level = self.config.difficulties[self.current_level]

                print(f"\n🎓 Advancing curriculum: {old_level} → {new_level}\n")

                # Reset tracking for new level
                self.episode_rewards = []
                self.episode_successes = []

            env.close()

        # Final statistics
        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Final Level: {self.config.difficulties[self.current_level]}")
        print("=" * 80)


def main():
    """Main training function."""
    config = CurriculumConfig()
    trainer = CurriculumTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
