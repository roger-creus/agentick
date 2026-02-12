"""Tests for submission specification."""

import pytest

from agentick.leaderboard.submission import SubmissionSpec


def test_submission_spec_validation():
    """Test submission spec validation."""
    spec = SubmissionSpec(
        agent_name="TestAgent",
        author="Test Author",
        description="Test description",
        agent_type="api",
        observation_mode="language",
        config={"provider": "openai", "model": "gpt-4o"},
        suites=["agentick-quick-v1"],
    )

    assert spec.agent_name == "TestAgent"
    assert spec.agent_type == "api"


def test_submission_spec_invalid_suite():
    """Test that invalid suite raises error."""
    with pytest.raises(Exception):
        SubmissionSpec(
            agent_name="TestAgent",
            author="Test",
            description="Test",
            agent_type="api",
            observation_mode="language",
            config={"provider": "openai", "model": "gpt-4o"},
            suites=["invalid-suite"],
        )


def test_submission_yaml_roundtrip(tmp_path):
    """Test YAML serialization and deserialization."""
    spec = SubmissionSpec(
        agent_name="TestAgent",
        author="Test",
        description="Test",
        agent_type="code",
        observation_mode="language",
        config={"script_path": "test.py"},
        suites=["agentick-quick-v1"],
    )

    yaml_path = tmp_path / "test.yaml"
    spec.to_yaml(yaml_path)

    loaded = SubmissionSpec.from_yaml(yaml_path)
    assert loaded.agent_name == spec.agent_name
