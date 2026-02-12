"""Example of curriculum learning with Agentick.

Demonstrates adaptive curriculum that adjusts difficulty based on agent performance.
"""

import agentick
from agentick.training import CurriculumCallback, MultiBackendLogger


class SimpleRLAgent:
    """Simple RL agent for demonstration (normally would use PPO, DQN, etc.)."""

    def __init__(self, env):
        self.env = env
        self.success_history = []

    def act(self, obs):
        """Choose action (random for this demo)."""
        return self.env.action_space.sample()

    def learn(self, obs, action, reward, next_obs, done):
        """Update policy (placeholder for real RL algorithm)."""
        pass


def main():
    """Run curriculum learning example."""
    print("Curriculum Learning Example")
    print("=" * 60)

    # Setup logger
    logger = MultiBackendLogger(log_dir="logs/curriculum", use_stdout=True, use_json=True)

    # Create curriculum callback
    def env_factory(difficulty="easy"):
        return agentick.make("MazeNavigation-v0", difficulty=difficulty, render_mode="state_dict")

    curriculum_callback = CurriculumCallback(
        curriculum_env_factory=env_factory,
        advance_threshold=0.8,  # Advance when 80% success
        regress_threshold=0.2,  # Regress when below 20% success
        window_size=50,  # Evaluate over 50 episodes
        min_episodes_per_level=20,  # At least 20 episodes per level
        logger=logger,
    )

    # Training loop
    n_episodes = 200

    for episode in range(n_episodes):
        # Get environment at current difficulty
        env = curriculum_callback.get_current_env()
        agent = SimpleRLAgent(env)

        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < 100:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.learn(obs, action, reward, next_obs, terminated or truncated)

            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
            step += 1

        success = info.get("success", False)

        # Update curriculum
        curriculum_info = curriculum_callback.on_episode_end(episode_reward, success, episode)

        # Log metrics
        logger.log("train/reward", episode_reward, episode)
        logger.log("train/steps", step, episode)
        logger.log("train/success", float(success), episode)
        logger.log("curriculum/difficulty", curriculum_info["curriculum/difficulty"], episode)
        logger.log("curriculum/success_rate", curriculum_info["curriculum/success_rate"], episode)

        if episode % 10 == 0:
            print(
                f"Episode {episode:3d}: difficulty={curriculum_info['curriculum/difficulty']:6s}, "
                f"success_rate={curriculum_info['curriculum/success_rate']:.2f}, "
                f"reward={episode_reward:6.2f}"
            )

    # Save final summary
    logger.save_summary()
    logger.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Logs saved to: logs/curriculum/")


if __name__ == "__main__":
    main()
