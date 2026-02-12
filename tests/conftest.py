"""Shared test fixtures and configuration."""

import os

import pytest

# Set pygame to headless mode for testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


@pytest.fixture
def seed():
    """Default seed for reproducible tests."""
    return 42


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--update-references",
        action="store_true",
        default=False,
        help="Update reference renders for visual regression tests",
    )
