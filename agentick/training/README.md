# training/ -- Training Backends, Callbacks, and Logging

Framework-agnostic training infrastructure with three backend integrations
for fine-tuning agents on Agentick data.

## Training Backends

All trainers are lazy-imported via `__getattr__` in `__init__.py` to avoid
pulling in heavy dependencies (torch, TRL, tinker) at import time.

### `trl/` -- HuggingFace TRL SFT
- **`AgentickSFTTrainer`** -- High-level wrapper around TRL's `SFTTrainer`.
  Trains any HuggingFace causal LM on conversation-format data from
  `CollectedDataset.export_to_huggingface()`. Supports LoRA (default) and
  full fine-tuning.
- **`SFTAgent`** -- Wraps a fine-tuned model back into an agent for evaluation.
- See `trl/TRL_README.md` for detailed usage.

### `tinker/` -- Tinker Remote Training
- **`TinkerSFTTrainer`** -- LoRA fine-tuning via Tinker's remote
  infrastructure on oracle demonstrations. Requires `TINKER_API_KEY`.
- **`TinkerRLTrainer`** -- PPO/REINFORCE training on live environment
  interactions via the Tinker API. Supports the SFT warmstart -> RL
  fine-tune pipeline.
- **`TinkerSFTAgent`** -- Agent wrapper for Tinker-trained models.
- See `tinker/TINKER_README.md` for detailed usage.

### `behavior_cloning.py` -- Pure PyTorch BC
- **`BehaviorCloningTrainer`** -- Trains a Nature CNN (Mnih et al., 2015)
  from pixel observations (`rgb_array` modality) to action predictions.
  Uses `CollectedDataset` data. Produces a standalone `BCAgent`.

## Shared Infrastructure

### `callbacks.py`
- **`EvalCallback`** -- Periodic evaluation during training. Runs
  `n_eval_episodes` at configurable frequency and logs mean reward,
  success rate, and episode length.
- **`CurriculumCallback`** -- Adaptive difficulty advancement. Monitors
  success rate over a sliding window and advances/regresses through
  easy/medium/hard/expert levels.
- **`CheckpointCallback`** -- Saves periodic and best-model checkpoints.
  Tracks a configurable metric (default `eval/mean_reward`) for
  best-model selection.

### `logger.py`
- **`MultiBackendLogger`** -- Logs scalars and histograms to stdout,
  JSON file, Weights & Biases, and/or TensorBoard simultaneously.
  Methods: `log`, `log_dict`, `log_histogram`, `save_summary`, `close`.
- **`StdoutLogger`** -- Minimal print-only logger.

## Lazy Import Pattern

```python
# These trigger lazy imports -- no torch/trl/tinker loaded until accessed:
from agentick.training import AgentickSFTTrainer
from agentick.training import BehaviorCloningTrainer

# Callbacks and logger are always available:
from agentick.training import EvalCallback, MultiBackendLogger
```

## Subpackage READMEs

- `trl/TRL_README.md` -- TRL SFT setup, LoRA config, and training examples
- `tinker/TINKER_README.md` -- Tinker API setup, SFT and RL workflows
