# Rewards

Reward shaping utilities for training agents. These are optional add-ons used alongside or on top of the base task rewards.

## Modules

### composite.py -- `CompositeReward`

Combines multiple reward functions into a weighted sum. Tracks per-component reward history and provides statistics (mean, std, min, max, sum). Supports dynamic weight updates, adding/removing components at runtime, and returning component breakdowns via `get_component_rewards()` and `get_weighted_components()`.

### intrinsic.py -- Exploration and curiosity rewards

- `ExplorationBonus` -- count-based intrinsic reward that decays with state visit frequency. Supports `"inverse"` (1/n) and `"sqrt"` (1/sqrt(n)) decay schedules. Hashes agent position, grid terrain/objects, and inventory into a state key.
- `CuriosityReward` -- prediction-error-based reward. Higher reward when the agent's predicted next state diverges from the actual state. Normalizes by a rolling window of recent errors.
- `InformationGainReward` -- rewards rare state transitions using `-log(p(transition))`. Tracks per-state observation counts and transition frequencies.
- `NoveltyReward` -- distance-based reward comparing current state to a bounded memory of recent states. Novel states (above a similarity threshold) receive a bonus.

### potential.py -- Potential-based reward shaping

- `PotentialBasedReward` -- implements F(s, s') = gamma * phi(s') - phi(s) (Ng et al., 1999). Provably preserves optimal policy while providing denser learning signal. Call `reset()` at episode start and `shape_reward()` after each step.
- `manhattan_distance_potential(goal)` -- negative Manhattan distance to goal.
- `euclidean_distance_potential(goal)` -- negative Euclidean distance to goal.
- `subgoal_potential(subgoals)` -- counts completed ordered subgoals.
- `inventory_potential(target_items)` -- counts collected target items.
- `composite_potential(*fns, weights)` -- weighted sum of multiple potential functions.
