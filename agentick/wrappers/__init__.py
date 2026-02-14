"""Gymnasium environment wrappers for Agentick."""

from agentick.wrappers.atari_preprocessing import (
    FrameStack,
    GrayscaleObservation,
    ResizeObservation,
    make_atari_env,
)
from agentick.wrappers.observation_wrappers import (
    DictObservationWrapper,
    FlattenObservationWrapper,
    LanguageActionWrapper,
    PixelObservationWrapper,
    TextObservationWrapper,
)
from agentick.wrappers.recording_wrappers import (
    EpisodeRecorder,
    TrajectoryWrapper,
)
from agentick.wrappers.reward_wrappers import (
    CurriculumWrapper,
    DenseRewardWrapper,
    RewardScaleWrapper,
    SparseRewardWrapper,
)

__all__ = [
    # Atari preprocessing
    "ResizeObservation",
    "GrayscaleObservation",
    "FrameStack",
    "make_atari_env",
    # Observation wrappers
    "DictObservationWrapper",
    "FlattenObservationWrapper",
    "LanguageActionWrapper",
    "PixelObservationWrapper",
    "TextObservationWrapper",
    # Recording wrappers
    "EpisodeRecorder",
    "TrajectoryWrapper",
    # Reward wrappers
    "CurriculumWrapper",
    "DenseRewardWrapper",
    "RewardScaleWrapper",
    "SparseRewardWrapper",
]
