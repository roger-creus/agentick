# Wrappers

Gymnasium-compatible wrappers for transforming observations, rewards, and recording episodes.

## Observation Wrappers (`observation_wrappers.py`)

- `TextObservationWrapper` -- returns text/language observations via `get_text_observation()`
- `PixelObservationWrapper` -- returns RGB pixel arrays via `get_pixel_observation()`
- `DictObservationWrapper` -- returns structured dict observations via `get_state_dict()`
- `FlattenObservationWrapper` -- flattens all four grid layers (terrain, objects, agents, metadata) into a single 1D `float32` vector
- `LanguageActionWrapper` -- `gym.ActionWrapper` that accepts natural language action strings and maps them to discrete action indices

## Atari Preprocessing (`atari_preprocessing.py`)

Standard RL preprocessing pipeline for pixel observations:

- `ResizeObservation(env, size=(84, 84))` -- resizes pixel observations using PIL bilinear interpolation
- `GrayscaleObservation(env)` -- converts RGB to single-channel grayscale
- `FrameStack(env, n_frames=4)` -- stacks the last N frames along the channel axis
- `make_atari_env(task_name, seed=0, **kwargs)` -- convenience function that composes all three: `rgb_array -> resize 84x84 -> grayscale -> frame stack 4`, producing `(84, 84, 4)` uint8 observations

## Recording Wrappers (`recording_wrappers.py`)

- `EpisodeRecorder(env, save_path=None, record_all_modalities=True)` -- records full episodes with text, language, pixel, and state_dict observations simultaneously; stores action names, reward breakdowns, and timestamps; saves to JSON
- `TrajectoryWrapper(env)` -- lightweight trajectory buffer storing `(observations, actions, rewards)` lists for offline RL and imitation learning; call `get_trajectories()` to retrieve

## Reward Wrappers (`reward_wrappers.py`)

- `DenseRewardWrapper(env)` -- switches the environment to dense reward mode
- `SparseRewardWrapper(env)` -- switches the environment to sparse reward mode
- `RewardScaleWrapper(env, scale=1.0, shift=0.0)` -- linear transform: `reward * scale + shift`
- `CurriculumWrapper(env, success_threshold=0.8, window_size=10)` -- tracks recent success rate and auto-advances difficulty (easy -> medium -> hard) when the threshold is met

## State Features Wrapper (`state_features_wrapper.py`)

- `StateFeaturesWrapper(env, grid_size=(20, 20))` -- converts `state_dict` observations into flat numpy feature vectors using `core.feature_extractor`; intended for state-based RL training

## Composing Wrappers

Wrappers are applied inside-out. Order matters -- observation wrappers should go after reward wrappers:

```python
import agentick
from agentick.wrappers import (
    RewardScaleWrapper,
    DenseRewardWrapper,
    ResizeObservation,
    GrayscaleObservation,
    FrameStack,
)

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
env = DenseRewardWrapper(env)
env = RewardScaleWrapper(env, scale=0.1)
env = ResizeObservation(env, size=(84, 84))
env = GrayscaleObservation(env)
env = FrameStack(env, n_frames=4)
```

Or use the convenience function for the standard Atari pipeline:

```python
from agentick.wrappers import make_atari_env
env = make_atari_env("GoToGoal-v0", seed=42, difficulty="medium")
```
