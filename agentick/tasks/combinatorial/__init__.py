"""Combinatorial tasks."""

from agentick.tasks.combinatorial.graph_coloring import GraphColoringTask
from agentick.tasks.combinatorial.lights_out import LightsOutTask
from agentick.tasks.combinatorial.packing_puzzle import PackingPuzzleTask
from agentick.tasks.combinatorial.tile_sorting import TileSortingTask

__all__ = [
    "LightsOutTask",
    "TileSortingTask",
    "GraphColoringTask",
    "PackingPuzzleTask",
]
