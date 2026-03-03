"""
Stable Baselines3 DQN training with CNN policy, wandb logging, and video recording.

This example demonstrates:
- DQN training with CnnPolicy for visual observations
- Integration with wandb for experiment tracking
- Periodic evaluation with video recording
- Model checkpointing

Requirements:
    uv sync --extra rl --extra all

Usage:
    uv run python examples/rl/sb3_dqn.py
"""

from pathlib import Path

from agentick.wrappers import make_atari_env

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  stable-baselines3 not installed. Install with: uv sync --extra rl")

try:
    from wandb.integration.sb3 import WandbCallback

    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: uv sync --extra all")


def main():
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 is required. Install with: uv sync --extra rl")
        return

    print("Stable Baselines3 DQN Training with CNN Policy")
    print("=" * 80)

    # Config
    env_id = "GoToGoal-v0"
    total_timesteps = 500_000
    n_eval_episodes = 10
    eval_freq = 50_000
    checkpoint_freq = 100_000
    video_folder = "videos/sb3_dqn"
    checkpoint_dir = "checkpoints/sb3_dqn"

    # Initialize wandb
    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(
            project="agentick-sb3-dqn",
            config={
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                "algorithm": "DQN",
                "policy": "CnnPolicy",
            },
            sync_tensorboard=True,
            monitor_gym=True,
        )
        print(f"✓ Wandb initialized: {run.url}")

    # Create training environment (8 parallel envs)
    n_envs = 8

    def make_env():
        env = make_atari_env(env_id)
        env = Monitor(env)
        return env

    train_env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Create evaluation environment with video recording
    def make_eval_env():
        env = make_atari_env(env_id)
        env = Monitor(env)
        return env

    eval_env = DummyVecEnv([make_eval_env])

    # Wrap with video recorder
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: x % eval_freq == 0,
        video_length=200,
        name_prefix="eval",
    )

    print(f"\nEnvironment: {env_id}")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")

    # Create DQN agent with CnnPolicy
    print("\nCreating DQN agent with CnnPolicy...")
    model = DQN(
        "CnnPolicy",
        train_env,
        verbose=1,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        learning_rate=1e-4,
        gamma=0.99,
        target_update_interval=10_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=1,
        tensorboard_log="./logs/sb3_dqn",
        device="auto",
    )

    print(f"✓ Model created with policy: {model.policy.__class__.__name__}")
    print(f"✓ Device: {model.device}")

    # Setup callbacks
    callbacks = []

    # Eval callback with video recording
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{checkpoint_dir}/best",
        log_path=f"{checkpoint_dir}/logs",
        eval_freq=eval_freq // train_env.num_envs,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // train_env.num_envs,
        save_path=checkpoint_dir,
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Wandb callback
    if WANDB_AVAILABLE and run is not None:
        wandb_callback = WandbCallback(
            model_save_path=f"{checkpoint_dir}/wandb",
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Train
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("=" * 80)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_model_path = Path(checkpoint_dir) / "final_model.zip"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    eval_env_final = make_atari_env(env_id)
    returns = []
    lengths = []

    for episode in range(n_eval_episodes):
        obs, info = eval_env_final.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_final.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        returns.append(total_reward)
        lengths.append(steps)

        success_str = "✓" if info.get("success", False) else "✗"
        print(
            f"Episode {episode + 1:2d}: {success_str} Steps: {steps:3d}, Reward: {total_reward:6.2f}"
        )

    print("\nFinal Results:")
    print(f"  Mean return: {sum(returns) / len(returns):.2f} ± {(max(returns) - min(returns)) / 2:.2f}")
    print(f"  Mean length: {sum(lengths) / len(lengths):.2f}")

    if WANDB_AVAILABLE and run is not None:
        wandb.log(
            {
                "final/mean_return": sum(returns) / len(returns),
                "final/mean_length": sum(lengths) / len(lengths),
            }
        )
        run.finish()

    eval_env_final.close()
    train_env.close()
    eval_env.close()

    print("\n✓ Training complete!")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Videos: {video_folder}")
    print("  TensorBoard logs: ./logs/sb3_dqn")


if __name__ == "__main__":
    main()
