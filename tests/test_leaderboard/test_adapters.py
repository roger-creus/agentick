"""Tests for agent adapters."""

from agentick.leaderboard.adapters.protocol import Agent, validate_agent


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self):
        super().__init__("MockAgent")
        self.reset_called = False

    def act(self, observation, info):
        return 0

    def reset(self):
        self.reset_called = True


def test_agent_protocol():
    """Test agent protocol."""
    agent = MockAgent()

    assert agent.name == "MockAgent"
    assert validate_agent(agent)

    action = agent.act(None, {})
    assert isinstance(action, int)

    agent.reset()
    assert agent.reset_called


def test_validate_agent():
    """Test agent validation."""
    agent = MockAgent()
    assert validate_agent(agent)

    # Test invalid agent
    class InvalidAgent:
        pass

    assert not validate_agent(InvalidAgent())
