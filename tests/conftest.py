"""Shared test fixtures and configuration."""

import os

import pytest

# Set pygame to headless mode for testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.pop("UV_ENV_FILE", None)


@pytest.fixture
def seed():
    """Default seed for reproducible tests."""
    return 42
