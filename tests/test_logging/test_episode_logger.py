"""Tests for episode logger."""

import json

import pytest

from agentick.logging.episode_logger import EpisodeLogger


def test_basic_logging(tmp_path):
    """Test basic episode logging."""
    log_path = tmp_path / "episode.jsonl.gz"

    logger = EpisodeLogger(str(log_path), verbosity="standard")

    # Log steps
    for step in range(5):
        logger.log_step(
            step=step,
            observation={"state": [1, 2, 3]},
            action={"index": step % 4, "name": f"action_{step}"},
            reward={"total": 1.0 + step, "components": {}},
            info={"success": False},
        )

    logger.close()

    # Verify file exists
    assert log_path.exists()

    # Load and verify
    steps = EpisodeLogger.load(str(log_path))
    assert len(steps) == 5
    assert steps[0]["step"] == 0
    assert steps[4]["step"] == 4


def test_verbosity_levels(tmp_path):
    """Test different verbosity levels."""
    # Minimal
    minimal_path = tmp_path / "minimal.jsonl.gz"
    logger = EpisodeLogger(str(minimal_path), verbosity="minimal")

    logger.log_step(
        step=0,
        observation={"state": [1, 2, 3]},
        action={"index": 0, "name": "action_0"},
        reward={"total": 1.0},
        info={},
    )
    logger.close()

    steps = EpisodeLogger.load(str(minimal_path))
    # Minimal should log only essential info
    assert "step" in steps[0]

    # Full
    full_path = tmp_path / "full.jsonl.gz"
    logger_full = EpisodeLogger(str(full_path), verbosity="full")

    logger_full.log_step(
        step=0,
        observation={"state": [1, 2, 3], "rgb": [[0, 0, 0]]},
        action={"index": 0, "name": "action_0"},
        reward={"total": 1.0},
        info={},
    )
    logger_full.close()

    steps_full = EpisodeLogger.load(str(full_path))
    # Full should log all observations
    assert "observation" in steps_full[0]


def test_compression(tmp_path):
    """Test gzip compression."""
    log_path = tmp_path / "episode.jsonl.gz"

    logger = EpisodeLogger(str(log_path))

    # Log many steps
    for step in range(100):
        logger.log_step(
            step=step,
            observation={"state": list(range(100))},
            action={"index": 0},
            reward={"total": 1.0},
            info={},
        )

    logger.close()

    # Compressed file should be smaller than uncompressed
    uncompressed_size = sum(
        len(json.dumps(step).encode()) for step in EpisodeLogger.load(str(log_path))
    )
    compressed_size = log_path.stat().st_size

    assert compressed_size < uncompressed_size


def test_streaming(tmp_path):
    """Test streaming writes."""
    log_path = tmp_path / "episode.jsonl.gz"

    logger = EpisodeLogger(str(log_path))

    # Write steps one at a time
    for step in range(10):
        logger.log_step(
            step=step,
            observation={},
            action={"index": 0},
            reward={"total": 1.0},
            info={},
        )

        # Should be able to read partial log
        if step > 0:
            logger.flush()
            steps = EpisodeLogger.load(str(log_path))
            assert len(steps) >= step

    logger.close()


def test_load_invalid_file():
    """Test loading invalid file."""
    with pytest.raises(Exception):
        EpisodeLogger.load("nonexistent_file.jsonl.gz")


def test_append_mode(tmp_path):
    """Test appending to existing log."""
    log_path = tmp_path / "episode.jsonl.gz"

    # Initial write
    logger1 = EpisodeLogger(str(log_path))
    logger1.log_step(step=0, observation={}, action={"index": 0}, reward={"total": 1.0}, info={})
    logger1.close()

    # Append
    logger2 = EpisodeLogger(str(log_path), mode="a")
    logger2.log_step(step=1, observation={}, action={"index": 1}, reward={"total": 2.0}, info={})
    logger2.close()

    # Should have both steps
    steps = EpisodeLogger.load(str(log_path))
    assert len(steps) >= 2
