# Data -- Trajectory Collection and Dataset Export

Two-level API for collecting agent trajectories and exporting them as
training datasets.

## API Levels

**Low-level**: `TrajectoryCollector` -- manual step-by-step recording with
`start_episode()`, `add_step()`, `end_episode()`. Stores `Trajectory`
dataclasses in a bounded buffer. Supports save/load to `.npz` and `.json`.

**High-level**: `DataCollector` -- wraps an env + agent, runs episodes via
`collect()`, returns a `CollectedDataset` with multi-modal observations
and one-call export methods.

## Files

### `collector.py`
- **`Trajectory`** -- Dataclass holding observations, actions, rewards,
  dones, infos, and metadata for a single episode. Has `add_step()` and
  `to_dict()`.
- **`TrajectoryCollector`** -- Low-level collector with configurable
  `buffer_size` and optional observation skipping. Methods: `start_episode`,
  `add_step`, `end_episode`, `get_trajectories` (with `min_reward` /
  `max_length` filtering), `save`, `save_json`, `load`, `clear`.
- **`MultiModalStep`** / **`MultiModalTrajectory`** -- Dataclasses for
  steps and episodes with `dict[str, Any]` observations keyed by modality
  (e.g. `"language"`, `"ascii"`, `"rgb_array"`).
- **`CollectedDataset`** -- Returned by `DataCollector.collect()`. Key methods:
  - `save(path)` -- writes `meta.json` + `trajectories.jsonl`
  - `load(path)` -- class method to reload a saved dataset
  - `export_to_huggingface(path, format)` -- exports to HF Datasets in
    `"conversation"` (chat SFT), `"decision"` (obs/action/reward tuples),
    or `"trajectory"` (full episodes) format
  - `push_to_hub(repo_id)` -- uploads directly to HuggingFace Hub
  - `stats()` -- returns success rate, mean reward, mean length
- **`DataCollector`** -- Accepts any env and any `AgentProtocol` agent.
  Configurable `record_modalities`, optional reasoning capture. The
  `collect(num_episodes, seeds, difficulty)` method runs rollouts and
  returns a `CollectedDataset`.

### `demonstrations.py`
Batch oracle data generation:
- `collect_oracle_trajectories(env_id, num_episodes, ...)` -- runs the
  environment's built-in oracle policy (falls back to random).
- `collect_random_trajectories(...)` -- random baseline collection.
- `create_preference_pairs(...)` -- generates oracle/random pairs from
  the same seed for DPO/preference learning.

### `formats.py` -- `export_to_format(trajectories, path, format_type)`
Converts low-level `Trajectory` lists to:
- `"jsonl"` -- JSON Lines (one episode per line)
- `"hf_dataset"` -- HuggingFace Datasets directory
- `"d4rl"` -- HDF5 file (actions, rewards, terminals, observations)
- `"conversation"` -- chat JSONL for LLM fine-tuning

## Quick Start

```python
import agentick
from agentick.data import DataCollector
from agentick.oracles import get_oracle

env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")
oracle = get_oracle("GoToGoal-v0", env)
collector = DataCollector(env, oracle, record_modalities=["language", "ascii"])
dataset = collector.collect(num_episodes=50)
dataset.save("data/oracle/")
dataset.export_to_huggingface("data/hf/", format="conversation")
```
