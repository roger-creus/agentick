# Curriculum

Curriculum learning strategies for progressively adjusting task difficulty during training.

## Modules

### `adaptive.py`
- **`AdaptiveCurriculum`** -- Automatically advances or regresses difficulty based on a sliding-window success rate. Levels follow the order `easy -> medium -> hard -> expert`. Configurable parameters:
  - `advance_threshold` (default 0.8) -- success rate to move up
  - `regress_threshold` (default 0.2) -- success rate to move down
  - `window_size` (default 50) -- number of recent episodes tracked
  - `min_episodes_per_level` (default 20) -- minimum episodes before allowing a transition
- Key methods: `update(success)` returns `True` if difficulty changed, `get_difficulty()` returns current level, `get_stats()` returns a summary dict.

### `manual.py`
- **`ManualCurriculum`** -- User-defined difficulty schedule. Takes an ordered list of difficulty levels and episode counts per level. Computes cumulative thresholds so `get_difficulty(episode)` returns the correct level for any episode number. Also provides `get_level()` (0-indexed) and `get_progress()` (0-1 ratio through the full curriculum).

## Usage

```python
from agentick.curriculum import AdaptiveCurriculum, ManualCurriculum

# Adaptive: auto-adjusts based on performance
curriculum = AdaptiveCurriculum(advance_threshold=0.8, regress_threshold=0.2)
for episode in range(1000):
    difficulty = curriculum.get_difficulty()
    success = run_episode(task, difficulty)
    curriculum.update(success)

# Manual: fixed schedule
curriculum = ManualCurriculum(
    difficulties=["easy", "medium", "hard"],
    episodes_per_level=[200, 300, 500],
)
difficulty = curriculum.get_difficulty(episode=250)  # "medium"
```
