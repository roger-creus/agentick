"""Profile step() method using cProfile."""

import cProfile
import pstats
import io

from agentick.tasks.registry import make
from agentick.benchmark.baselines import RandomAgent

def profile_steps():
    """Profile 1000 steps."""
    env = make("GoToGoal-v0", render_mode="state_dict", fast_mode=True, seed=42)
    agent = RandomAgent(seed=42)

    obs, info = env.reset()
    for _ in range(1000):
        action = agent(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    profile_steps()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
