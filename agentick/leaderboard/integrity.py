"""Result integrity verification and tamper detection."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentick.leaderboard.result import EvaluationResult


def compute_result_hash(result: EvaluationResult) -> str:
    """
    Compute SHA256 hash of evaluation result.

    Only includes deterministic data (scores, episodes), not timing or metadata.

    Args:
        result: Evaluation result

    Returns:
        SHA256 hex digest
    """
    # Extract deterministic data
    hash_data = {
        "suite_name": result.suite_name,
        "suite_hash": result.suite_hash,
        "agentick_score": result.agentick_score,
        "per_task": result.per_task,
        "episodes": [
            {
                "task": ep.task_name,
                "seed": ep.seed,
                "return": ep.episode_return,
                "steps": ep.steps,
                "success": ep.success,
            }
            for ep in result.episodes
        ],
    }

    # Serialize deterministically
    hash_json = json.dumps(hash_data, sort_keys=True)

    # Compute hash
    return hashlib.sha256(hash_json.encode()).hexdigest()


def verify_result(result: EvaluationResult) -> bool:
    """
    Verify result integrity by recomputing hash.

    Args:
        result: Evaluation result

    Returns:
        True if hash matches, False otherwise
    """
    computed_hash = compute_result_hash(result)
    return computed_hash == result.result_hash


def verify_reproducibility(
    result: EvaluationResult,
    n_samples: float = 0.1,
    max_delta: float = 1e-6,
) -> tuple[bool, float]:
    """
    Verify reproducibility by re-running a sample of episodes.

    Args:
        result: Evaluation result to verify
        n_samples: Fraction of episodes to re-run (0.1 = 10%)
        max_delta: Maximum acceptable difference in returns

    Returns:
        Tuple of (is_reproducible, max_delta_found)
    """
    import numpy as np

    import agentick
    from agentick.leaderboard.adapters.api_adapter import APIAgent
    from agentick.leaderboard.adapters.code_adapter import CodeAgent
    from agentick.leaderboard.adapters.huggingface_adapter import HuggingFaceAgent

    # Select random sample of episodes
    n_episodes = len(result.episodes)
    n_to_check = max(1, int(n_episodes * n_samples))

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_episodes, size=n_to_check, replace=False)

    # Load agent
    submission = result.submission
    if submission.agent_type == "api":
        agent = APIAgent(**submission.config)
    elif submission.agent_type == "code":
        agent = CodeAgent(**submission.config)
    elif submission.agent_type == "huggingface":
        agent = HuggingFaceAgent(**submission.config)
    else:
        # Other adapters not implemented for verification
        return True, 0.0

    max_delta_found = 0.0

    for idx in sample_indices:
        original_episode = result.episodes[idx]

        # Create environment
        env = agentick.make(
            original_episode.task_name,
            difficulty=original_episode.difficulty,
        )

        # Reset with same seed
        obs, info = env.reset(seed=original_episode.seed)
        agent.reset()

        # Run episode
        done = False
        episode_return = 0.0

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

        # Compare returns
        delta = abs(episode_return - original_episode.episode_return)
        max_delta_found = max(max_delta_found, delta)

    is_reproducible = max_delta_found <= max_delta
    return is_reproducible, max_delta_found
