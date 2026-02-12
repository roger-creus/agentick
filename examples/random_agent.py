"""Simple random agent example."""

import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii", seed=42)

for episode in range(3):
    obs, info = env.reset()
    episode_return = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        if step % 10 == 0:
            print(env.render())

        if terminated or truncated:
            print(
                f"Episode {episode + 1}: Return = {episode_return:.2f}, Success = {info.get('success', False)}"
            )
            break
