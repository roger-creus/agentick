"""Oracle bots — one programmatic solver per task, powered by the Coding API.

Usage::

    from agentick.oracles import get_oracle
    import agentick

    env = agentick.make("GoToGoal-v0", difficulty="hard")
    oracle = get_oracle("GoToGoal-v0", env)
    obs, info = env.reset(seed=42)
    oracle.reset(obs, info)

    done = False
    while not done:
        action = oracle.act(obs, info)
        obs, reward, done, trunc, info = env.step(action)
        oracle.update(obs, info)
        done = done or trunc
"""

from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import get_oracle, list_oracles

__all__ = ["OracleAgent", "get_oracle", "list_oracles"]
