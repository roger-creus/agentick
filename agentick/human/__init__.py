"""Human evaluation and baseline collection module.

Provides tools for collecting human performance data and analyzing baselines.
"""

from agentick.human.analysis import HumanBaselineAnalyzer
from agentick.human.baselines import (
    HUMAN_BASELINES,
    compare_to_human,
    estimate_human_performance,
    get_all_baselines,
    get_baselines_by_difficulty,
    get_human_baseline,
    get_summary_statistics,
)
from agentick.human.player import HumanPlayer
from agentick.human.recorder import HumanDataRecorder

__all__ = [
    "HumanPlayer",
    "HumanDataRecorder",
    "HumanBaselineAnalyzer",
    "get_human_baseline",
    "get_all_baselines",
    "estimate_human_performance",
    "compare_to_human",
    "get_baselines_by_difficulty",
    "get_summary_statistics",
    "HUMAN_BASELINES",
]
