"""
CleanRL PPO training example with wandb logging, evaluation, video recording, and checkpoints.

This example demonstrates:
- PPO training from scratch using CleanRL-style implementation
- Integration with wandb for experiment tracking
- Periodic evaluation with video recording
- Model checkpointing
- Performance metrics logging

Requirements:
    uv sync --extra rl --extra all

Usage:
    uv run python examples/rl/ppo_cleanrl.py
"""

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agentick.wrappers import make_atari_env

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: uv sync --extra all")


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    # Environment
    env_id: str = "GoToGoal-v0"
    num_envs: int = 4
    num_steps: int = 128

    # Training
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Evaluation
    eval_freq: int = 50_000
    eval_episodes: int = 10
    record_video: bool = True

    # Logging
    log_freq: int = 10
    use_wandb: bool = True
    wandb_project: str = "agentick-ppo"
    wandb_entity: str | None = None

    # Checkpointing
    checkpoint_freq: int = 100_000
    checkpoint_dir: str = "checkpoints/ppo"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1


class ActorCritic(nn.Module):
    """Combined actor-critic network with Nature CNN architecture for visual observations."""

    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()

        # Get action dimension
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

        # Check if we have image observations
        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            # Nature CNN for image observations
            # Input is (H, W, C) but we need (C, H, W) for PyTorch
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

            # Calculate output size with a dummy forward pass
            # Create dummy input as (1, C, H, W)
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
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = int(np.prod(obs_space["text"].shape))
            else:
                obs_dim = int(np.prod(obs_space.shape))

            self.network = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self.fc = None
            feature_size = 256

        # Actor head
        self.actor = nn.Linear(feature_size, action_dim)

        # Critic head
        self.critic = nn.Linear(feature_size, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        features = self.network(obs)
        if self.fc is not None:
            features = self.fc(features)
        return self.critic(features)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        features = self.network(obs)
        if self.fc is not None:
            features = self.fc(features)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


def make_env(env_id: str, seed: int, idx: int):
    """Create a single environment with Atari preprocessing."""

    def thunk():
        env = make_atari_env(env_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def evaluate(
    agent: ActorCritic,
    env_id: str,
    num_episodes: int,
    device: str,
    record_video: bool = False,
    video_dir: str | None = None,
) -> dict:
    """Evaluate the agent."""
    eval_env = make_atari_env(env_id, seed=42)

    if record_video and video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            video_folder=video_dir,
            episode_trigger=lambda x: x < 3,  # Record first 3 episodes
        )

    returns = []
    lengths = []
    successes = []

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        while not done:
            # Get observation - normalize images to [0, 1]
            if isinstance(obs, dict):
                if "rgb" in obs:
                    # Image observation: (H, W, C) -> (C, H, W) and normalize
                    obs_array = obs["rgb"].transpose(2, 0, 1) / 255.0
                    obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
                else:
                    obs_tensor = torch.FloatTensor(obs["text"].flatten()).unsqueeze(0).to(device)
            elif len(obs.shape) == 3:
                # Image observation: (H, W, C) -> (C, H, W) and normalize
                obs_array = obs.transpose(2, 0, 1) / 255.0
                obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
            else:
                obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)

            # Get action
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, info = eval_env.step(action.item())
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

        # Track success if available
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


def train(config: PPOConfig):
    """Train PPO agent."""
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initialize wandb
    run_name = f"{config.env_id.split('/')[-1]}_{int(time.time())}"
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=vars(config),
            sync_tensorboard=False,
        )

    # Create vectorized environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, config.seed, i) for i in range(config.num_envs)]
    )

    # Create agent
    agent = ActorCritic(envs.single_observation_space, envs.single_action_space).to(config.device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Get observation shape for buffer
    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        if "rgb" in envs.single_observation_space.spaces:
            # Use rgb observation
            obs_shape = envs.single_observation_space["rgb"].shape
            obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
        else:
            # Use text observation
            obs_shape = (int(np.prod(envs.single_observation_space["text"].shape)),)
    elif len(envs.single_observation_space.shape) == 3:
        # Image observation (H, W, C) -> store as (C, H, W)
        h, w, c = envs.single_observation_space.shape
        obs_shape = (c, h, w)
    else:
        obs_shape = (int(np.prod(envs.single_observation_space.shape)),)

    # Learning curve tracking
    learning_curve = {"steps": [], "returns": [], "lengths": []}

    # Storage
    obs_buffer = torch.zeros((config.num_steps, config.num_envs, *obs_shape)).to(config.device)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    values_buffer = torch.zeros((config.num_steps, config.num_envs)).to(config.device)

    # Initialize environment
    next_obs, _ = envs.reset(seed=config.seed)
    next_done = torch.zeros(config.num_envs).to(config.device)

    # Training loop
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    global_step = 0

    print(f"Training PPO on {config.env_id}")
    print(f"Device: {config.device}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Updates: {num_updates:,}")
    print(f"Wandb: {config.use_wandb and WANDB_AVAILABLE}")
    print()

    for update in range(1, num_updates + 1):
        # Anneal learning rate
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout
        for step in range(config.num_steps):
            global_step += config.num_envs

            # Convert observation - normalize images to [0, 1]
            if isinstance(next_obs, dict):
                if "rgb" in next_obs:
                    # Image observation: (N, H, W, C) -> (N, C, H, W) and normalize
                    obs_array = np.transpose(next_obs["rgb"], (0, 3, 1, 2)) / 255.0
                    obs_tensor = torch.FloatTensor(obs_array).to(config.device)
                else:
                    obs_tensor = torch.FloatTensor(next_obs["text"].reshape(config.num_envs, -1)).to(
                        config.device
                    )
            elif len(next_obs.shape) == 4:
                # Image observation: (N, H, W, C) -> (N, C, H, W) and normalize
                obs_array = np.transpose(next_obs, (0, 3, 1, 2)) / 255.0
                obs_tensor = torch.FloatTensor(obs_array).to(config.device)
            else:
                obs_tensor = torch.FloatTensor(next_obs.reshape(config.num_envs, -1)).to(
                    config.device
                )

            obs_buffer[step] = obs_tensor
            dones_buffer[step] = next_done

            # Get action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                values_buffer[step] = value.flatten()

            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # Step environment
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buffer[step] = torch.tensor(reward).to(config.device)
            next_done = torch.Tensor(done).to(config.device)

            # Log episode statistics
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        # Track learning curve
                        learning_curve["steps"].append(global_step)
                        learning_curve["returns"].append(item["episode"]["r"])
                        learning_curve["lengths"].append(item["episode"]["l"])

                        if update % config.log_freq == 0:
                            print(
                                f"Step {global_step:,}: episode_return={item['episode']['r']:.2f}, episode_length={item['episode']['l']}"
                            )

                        if config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(
                                {
                                    "train/episode_return": item["episode"]["r"],
                                    "train/episode_length": item["episode"]["l"],
                                    "train/global_step": global_step,
                                }
                            )

        # Bootstrap value
        with torch.no_grad():
            if isinstance(next_obs, dict):
                if "rgb" in next_obs:
                    obs_array = np.transpose(next_obs["rgb"], (0, 3, 1, 2)) / 255.0
                    next_obs_tensor = torch.FloatTensor(obs_array).to(config.device)
                else:
                    next_obs_tensor = torch.FloatTensor(
                        next_obs["text"].reshape(config.num_envs, -1)
                    ).to(config.device)
            elif len(next_obs.shape) == 4:
                obs_array = np.transpose(next_obs, (0, 3, 1, 2)) / 255.0
                next_obs_tensor = torch.FloatTensor(obs_array).to(config.device)
            else:
                next_obs_tensor = torch.FloatTensor(next_obs.reshape(config.num_envs, -1)).to(
                    config.device
                )
            next_value = agent.get_value(next_obs_tensor).reshape(1, -1)

            # GAE
            advantages = torch.zeros_like(rewards_buffer).to(config.device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = (
                    rewards_buffer[t]
                    + config.gamma * nextvalues * nextnonterminal
                    - values_buffer[t]
                )
                advantages[t] = lastgaelam = (
                    delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buffer

        # Flatten batch
        b_obs = obs_buffer.reshape((-1, *obs_shape))
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        values_buffer.reshape(-1)

        # Optimize policy
        b_inds = np.arange(config.num_steps * config.num_envs)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0,
                config.num_steps * config.num_envs,
                config.num_steps * config.num_envs // config.num_minibatches,
            ):
                end = start + config.num_steps * config.num_envs // config.num_minibatches
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        # Log training metrics
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/value_loss": v_loss.item(),
                    "train/policy_loss": pg_loss.item(),
                    "train/entropy": entropy_loss.item(),
                    "train/approx_kl": approx_kl.item(),
                    "train/clipfrac": np.mean(clipfracs),
                    "train/global_step": global_step,
                }
            )

        # Evaluation
        if global_step % config.eval_freq < (config.num_steps * config.num_envs):
            video_dir = None
            if config.record_video:
                video_dir = f"{config.checkpoint_dir}/videos/step_{global_step}"

            eval_results = evaluate(
                agent,
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
        if global_step % config.checkpoint_freq < (config.num_steps * config.num_envs):
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{global_step}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": agent.state_dict(),
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
        agent,
        config.env_id,
        config.eval_episodes * 2,  # More episodes for final eval
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
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
        },
        final_model_path,
    )
    print(f"\nFinal model saved: {final_model_path}")

    # Save and plot learning curves
    if learning_curve["steps"]:
        # Save raw data
        learning_curve_path = Path(config.checkpoint_dir) / "learning_curve.npz"
        np.savez(
            learning_curve_path,
            steps=np.array(learning_curve["steps"]),
            returns=np.array(learning_curve["returns"]),
            lengths=np.array(learning_curve["lengths"]),
        )
        print(f"Learning curve data saved: {learning_curve_path}")

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot returns
        ax1.plot(learning_curve["steps"], learning_curve["returns"], alpha=0.6)
        # Add smoothed curve (moving average)
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

        # Plot episode lengths
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
    config = PPOConfig()
    train(config)
