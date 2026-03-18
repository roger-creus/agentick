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


