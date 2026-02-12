# World Model Evaluation

Agentick includes comprehensive world model evaluation assessing an agent's understanding of environment dynamics. This guide explains the four test types, scoring methodology, and interpretation.

## What is World Model Evaluation?

A **world model** is an agent's internal representation of how the environment works. It enables:
- **Prediction**: Forecasting future states given actions
- **Planning**: Reasoning about action sequences before execution
- **Adaptation**: Detecting and responding to environmental changes
- **Reasoning**: Understanding counterfactual scenarios ("what if?")

World model evaluation measures these four capabilities independently.

## Four Test Types

### 1. Prediction Tests

Test the agent's ability to predict future states given action sequences.

#### Masked State Prediction

Agent is shown trajectory with some states hidden, must predict from multiple choices.

**Setup:**
- Generate random trajectory: s₀ → a₁ → s₁ → a₂ → s₂ → ... → sₜ
- Hide one state s_mask at position i ∈ [1, T-1]
- Provide n_choices options (correct state + distractors)
- Agent must identify correct state

**Scoring:**
- Accuracy = (# correct predictions) / (# tests)
- Higher accuracy indicates understanding of state dynamics

**Example:**
```python
from agentick.worldmodel import StatePredictionEvaluator

evaluator = StatePredictionEvaluator(
    env_factory=lambda: agentick.make("GoToGoal-v0"),
    n_tests=10,
    prediction_horizon=5
)

result = evaluator.evaluate_masked_prediction(agent, n_choices=4, seed=42)
print(f"Masked Prediction Accuracy: {result.accuracy:.1%}")
# 75% means agent correctly identified masked state 75% of time
```

**Interpretation:**
- **90%+**: Excellent state understanding
- **70-90%**: Good state tracking
- **50-70%**: Moderate understanding
- **<50%**: Weak or no state model

#### Free-Form Prediction

Agent must predict exact next state given history.

**Setup:**
- Provide observation sequence [o₀, o₁, ..., o_t]
- Provide action sequence [a₁, a₂, ..., a_t]
- Agent predicts next observation o_{t+1}
- Compare to ground truth

**Scoring:**
- Mean squared error (MSE) for continuous observations
- Accuracy for discrete observations
- Normalized to [0, 1] range

**Example:**
```python
result = evaluator.evaluate_free_form_prediction(agent, seed=42)
print(f"Free-Form Prediction Accuracy: {result.accuracy:.1%}")
print(f"Mean Error: {result.mean_error:.3f}")
```

**Interpretation:**
- Accuracy 0.9+: Near-perfect prediction
- Accuracy 0.7-0.9: Good prediction
- Accuracy 0.5-0.7: Noisy but usable
- Accuracy <0.5: Poor prediction capability

### 2. Transfer Tests

Test the agent's ability to generalize to modified environment mechanics.

**Setup:**
1. Measure baseline performance on original environment
2. Introduce modifications to environment (e.g., change physics, layout)
3. Measure transfer performance
4. Track adaptation speed

**Modifications tested:**
- Gravity changes (physics discovery)
- Obstacle patterns (spatial reasoning)
- Reward functions (goal understanding)
- State transitions (rule discovery)

**Scoring:**
- Baseline performance: mean return on original environment
- Transfer performance: mean return on modified environment
- Adaptation speed: episodes to reach 90% of baseline
- Transfer score = 1.0 - (performance_gap / baseline_performance), clamped to [0, 1]

**Example:**
```python
from agentick.worldmodel import TransferEvaluator

evaluator = TransferEvaluator(
    base_env_factory=lambda: agentick.make("GoToGoal-v0"),
    modified_env_factory=lambda: agentick.make("GoToGoal-v0", difficulty="hard"),
    n_episodes=10,
    adaptation_episodes=5
)

result = evaluator.evaluate_transfer(agent, seed=42)
print(f"Baseline Performance: {result.baseline_performance:.3f}")
print(f"Transfer Performance: {result.transfer_performance:.3f}")
print(f"Performance Gap: {result.final_gap:.3f}")
print(f"Adaptation Speed: {result.adaptation_speed:.3f}")
```

**Interpretation:**
- **Transfer score 0.9+**: Excellent generalization, minimal degradation
- **Transfer score 0.7-0.9**: Good transfer, minor adaptation needed
- **Transfer score 0.5-0.7**: Moderate transfer, noticeable degradation
- **Transfer score <0.5**: Poor transfer, major degradation or failure

**Few-Shot Transfer:**
Agent observes demonstrations of modified environment, then performs:

```python
result = evaluator.evaluate_few_shot_transfer(
    agent,
    n_demonstrations=3,  # Show 3 demo episodes
    seed=42
)
# Evaluates how quickly agent learns from examples
```

### 3. Change Detection Tests

Test the agent's ability to detect when environment dynamics change mid-episode.

**Setup:**
1. Run episodes with mid-episode dynamics changes
   - Change at random step t ∈ [10, episode_length-10]
   - Agent must detect change in observations
2. Run episodes without changes (for false positive rate)
3. Compute detection metrics

**Scoring:**
- Detection accuracy: (# changes correctly detected) / (# change episodes)
- Detection latency: steps between change and detection (lower is better)
- False positive rate: (# false alarms) / (# no-change episodes)
- Combined score = detection_accuracy × (1.0 - false_positive_rate)

**Example:**
```python
from agentick.worldmodel import ChangeDetectionEvaluator

evaluator = ChangeDetectionEvaluator(
    env_factory=lambda: agentick.make("EnvironmentShift-v0"),
    change_env_factory=lambda: agentick.make("EnvironmentShift-v0"),
    n_tests=20,
    episode_length=100
)

result = evaluator.evaluate_change_detection(agent, seed=42)
print(f"Detection Accuracy: {result.detection_accuracy:.1%}")
print(f"Mean Detection Latency: {result.mean_detection_latency:.1f} steps")
print(f"False Positive Rate: {result.false_positive_rate:.1%}")
print(f"Combined Score: {result.detection_accuracy * (1.0 - result.false_positive_rate):.3f}")
```

**Interpretation:**
- **Detection accuracy 90%+**: Catches almost all changes
- **Detection accuracy 70-90%**: Good change detection
- **Detection accuracy 50-70%**: Inconsistent detection
- **Detection accuracy <50%**: Poor change detection

**Change Localization:**
Agent must identify exactly when change occurred:

```python
localization = evaluator.evaluate_change_localization(agent, seed=42)
print(f"Mean Error: {localization['mean_localization_error']:.1f} steps")
# Lower error = more precise localization
```

### 4. Counterfactual Tests

Test the agent's ability to reason about alternate outcomes ("what if?").

**Setup:**
1. Generate factual trajectory: a₀ → a₁ → ... → a_t
2. Choose intervention point i and alternative action a'_i ≠ a_i
3. Regenerate trajectory with a'_i at step i (ground truth)
4. Agent must predict outcome of counterfactual

**Scoring:**
- Accuracy: (# correct counterfactual predictions) / (# tests)
- Mean error: average prediction error between predicted and true counterfactual

**Example:**
```python
from agentick.worldmodel import CounterfactualEvaluator

evaluator = CounterfactualEvaluator(
    env_factory=lambda: agentick.make("CausalChain-v0"),
    n_tests=10,
    trajectory_length=10
)

result = evaluator.evaluate_counterfactual_prediction(agent, seed=42)
print(f"Counterfactual Accuracy: {result.accuracy:.1%}")
print(f"Mean Error: {result.mean_error:.3f}")
```

**Interpretation:**
- **Accuracy 80%+**: Strong causal understanding
- **Accuracy 60-80%**: Good counterfactual reasoning
- **Accuracy 40-60%**: Partial understanding
- **Accuracy <40%**: Weak causal reasoning

**Causal Attribution:**
Agent identifies which actions causally responsible for outcomes:

```python
attribution = evaluator.evaluate_causal_attribution(agent, seed=42)
print(f"Mean Attribution Error: {attribution['mean_attribution_error']:.3f}")
print(f"Median Attribution Error: {attribution['median_attribution_error']:.3f}")
# Lower error = better causal attribution
```

## Running World Model Evaluation

### Single Test Type

```python
from agentick.worldmodel import WorldModelEvaluator

# Create evaluator with environment factories
evaluator = WorldModelEvaluator(
    env_factory=lambda: agentick.make("GoToGoal-v0"),
    modified_env_factory=lambda: agentick.make("GoToGoal-v0", difficulty="hard"),
    change_env_factory=lambda: agentick.make("EnvironmentShift-v0")
)

# Run only prediction tests
prediction_results = evaluator.evaluate_prediction_only(agent, seed=42)
print(f"Masked Prediction: {prediction_results['masked_prediction_accuracy']:.1%}")
print(f"Free-Form Prediction: {prediction_results['free_form_prediction_accuracy']:.1%}")
print(f"Combined: {prediction_results['combined_prediction_accuracy']:.1%}")
```

### Full Evaluation Suite

```python
# Run all four test types
full_results = evaluator.evaluate_full(agent, seed=42)

print(f"Prediction Accuracy: {full_results.prediction_accuracy:.3f}")
print(f"Transfer Score: {full_results.transfer_score:.3f}")
print(f"Change Detection: {full_results.change_detection_score:.3f}")
print(f"Counterfactual Accuracy: {full_results.counterfactual_accuracy:.3f}")
print(f"Overall Score: {full_results.overall_score:.3f}")
```

### Comprehensive Report

```python
# Generate full interpretation report
report = evaluator.generate_worldmodel_report(agent, seed=42)

print(f"Overall Score: {report['scores']['overall_score']:.3f}")
print(f"Grade: {report['grade']}")
print(f"\nInterpretation:")
for interpretation in report['interpretation']:
    print(f"  - {interpretation}")
```

Example output:
```
Overall Score: 0.732
Grade: B

Interpretation:
  - Strong state prediction ability
  - Good transfer to modified environments
  - Reliable change detection
  - Sound counterfactual reasoning
```

## Score Computation Details

### Prediction Score

$$\text{Score}_{pred} = \frac{\text{Masked Accuracy} + \text{Free-Form Accuracy}}{2}$$

Both components weighted equally.

### Transfer Score

$$\text{Score}_{transfer} = 1.0 - \min\left(\frac{|p_{final} - p_{baseline}|}{p_{baseline}}, 1.0\right)$$

Where p_{baseline} is baseline performance, p_{final} is final transfer performance.

### Change Detection Score

$$\text{Score}_{change} = \text{Detection Accuracy} \times (1.0 - \text{False Positive Rate})$$

Balanced between catching real changes and avoiding false alarms.

### Counterfactual Score

$$\text{Score}_{cf} = 1.0 - \text{Mean Error}$$

Normalized by maximum possible error.

### Overall Score (Weighted Average)

$$\text{Score}_{overall} = 0.3 \times \text{Score}_{pred} + 0.3 \times \text{Score}_{transfer} + 0.2 \times \text{Score}_{change} + 0.2 \times \text{Score}_{cf}$$

Weights reflect importance: prediction and transfer most critical, change detection and counterfactual secondary.

## Example World Model Tasks

### PhysicsDiscovery-v0

Agent discovers physics rules (gravity, friction, etc.) through experimentation.

**Evaluation:**
- Prediction: Can agent predict object trajectories?
- Transfer: Does understanding generalize to new objects?
- Counterfactual: Can agent predict "what if" physics changes?

```python
env = agentick.make("PhysicsDiscovery-v0", difficulty="medium")
evaluator = WorldModelEvaluator(env_factory=lambda: env)

# Agent should achieve high prediction/counterfactual scores
# But lower transfer (different objects/physics)
```

### EnvironmentShift-v0

Environment changes mid-episode; agent must detect and adapt.

**Evaluation:**
- Change Detection: Detects shift in rules?
- Transfer: Adapts to new rules?
- Prediction: Learns new dynamics?

```python
env = agentick.make("EnvironmentShift-v0", difficulty="medium")
evaluator = WorldModelEvaluator(env_factory=lambda: env)

# Agent should achieve high change detection score
# Transfer score reflects adaptation quality
```

### RuleDiscoveryNavigation-v0

Navigation with hidden rules (one-way passages, etc.).

**Evaluation:**
- Prediction: Understands state transitions?
- Change Detection: Detects rule violations?
- Counterfactual: Reasons about rule consequences?

```python
env = agentick.make("RuleDiscoveryNavigation-v0", difficulty="medium")
evaluator = WorldModelEvaluator(env_factory=lambda: env)

# Success requires high prediction accuracy
# Combined with good counterfactual reasoning
```

## Interpreting World Model Scores

### Overall Score Scale

**Grade A+ (0.90+)**: Excellent world understanding
- Strong prediction (>0.85)
- Good transfer (>0.80)
- Reliable change detection (>0.80)
- Sound counterfactual reasoning (>0.85)
- Agent demonstrates sophisticated environment understanding

**Grade A (0.85-0.90)**: Strong world understanding
- Good prediction (0.75-0.85)
- Solid transfer (0.70-0.80)
- Good change detection (0.70-0.80)
- Reliable counterfactual (0.75-0.85)

**Grade B (0.70-0.85)**: Competent world understanding
- Moderate prediction (0.65-0.75)
- Reasonable transfer (0.60-0.70)
- Inconsistent change detection (0.60-0.70)
- Partial counterfactual reasoning (0.60-0.70)

**Grade C (0.55-0.70)**: Basic world understanding
- Limited prediction (0.50-0.65)
- Weak transfer (0.45-0.60)
- Unreliable change detection (0.50-0.65)
- Basic counterfactual (0.45-0.60)

**Grade D-F (<0.55)**: Poor world understanding
- Weak prediction (<0.50)
- Poor transfer (<0.45)
- Unreliable change detection (<0.50)
- Limited counterfactual (<0.45)

### Capability Profiles

Different agent types show characteristic profiles:

**Model-Based RL:**
```
Prediction: 0.82        ★★★★☆
Transfer: 0.76          ★★★★☆
Change Detection: 0.68  ★★★☆☆
Counterfactual: 0.71    ★★★★☆
Overall: 0.74           ★★★★☆
```

**LLM Agent (GPT-4):**
```
Prediction: 0.78        ★★★★☆
Transfer: 0.85          ★★★★☆
Change Detection: 0.62  ★★★☆☆
Counterfactual: 0.88    ★★★★★
Overall: 0.78           ★★★★☆
```

**Random Agent:**
```
Prediction: 0.25        ★☆☆☆☆
Transfer: 0.15          ★☆☆☆☆
Change Detection: 0.05  ☆☆☆☆☆
Counterfactual: 0.10    ★☆☆☆☆
Overall: 0.13           ★☆☆☆☆
```

## Complete Example

```python
import agentick
from agentick.worldmodel import WorldModelEvaluator

# Create agent
agent = MyWorldModelAgent()

# Create evaluator
evaluator = WorldModelEvaluator(
    env_factory=lambda: agentick.make("PhysicsDiscovery-v0", difficulty="medium"),
    modified_env_factory=lambda: agentick.make("PhysicsDiscovery-v0", difficulty="hard"),
    change_env_factory=lambda: agentick.make("EnvironmentShift-v0", difficulty="medium")
)

# Run full evaluation
results = evaluator.evaluate_full(agent, seed=42)

# Generate report
report = evaluator.generate_worldmodel_report(agent, seed=42)

# Print results
print(f"=== World Model Evaluation Results ===")
print(f"Overall Score: {report['scores']['overall_score']:.3f}")
print(f"Grade: {report['grade']}")
print(f"\nComponent Scores:")
print(f"  Prediction: {report['scores']['prediction_accuracy']:.3f}")
print(f"  Transfer: {report['scores']['transfer_score']:.3f}")
print(f"  Change Detection: {report['scores']['change_detection_score']:.3f}")
print(f"  Counterfactual: {report['scores']['counterfactual_accuracy']:.3f}")
print(f"\nInterpretation:")
for line in report['interpretation']:
    print(f"  - {line}")

# Save results
import json
with open("worldmodel_results.json", "w") as f:
    json.dump({
        "scores": report['scores'],
        "grade": report['grade'],
        "interpretation": report['interpretation']
    }, f, indent=2)
```

Output:
```
=== World Model Evaluation Results ===
Overall Score: 0.742
Grade: B

Component Scores:
  Prediction: 0.812
  Transfer: 0.715
  Change Detection: 0.681
  Counterfactual: 0.752

Interpretation:
  - Strong state prediction ability
  - Good transfer to modified environments
  - Inconsistent change detection
  - Sound counterfactual reasoning
```

## Best Practices

**1. Use consistent seeds**
```python
# Ensure reproducibility
results = evaluator.evaluate_full(agent, seed=42)
```

**2. Test with multiple environment variants**
```python
# Don't just test on one difficulty level
for difficulty in ["easy", "medium", "hard", "expert"]:
    env_factory = lambda: agentick.make("GoToGoal-v0", difficulty=difficulty)
    evaluator = WorldModelEvaluator(env_factory=env_factory)
    results = evaluator.evaluate_full(agent)
    print(f"{difficulty}: {results.overall_score:.3f}")
```

**3. Report all component scores**
```python
# Don't just report overall score
# Show component breakdown for insights
print(f"Overall: {overall:.3f}")
print(f"  Prediction: {pred:.3f}")
print(f"  Transfer: {transfer:.3f}")
print(f"  Change Detection: {change:.3f}")
print(f"  Counterfactual: {cf:.3f}")
```

**4. Compare against baselines**
```python
# Compare world model quality to random agent
random_agent = RandomAgent()
random_eval = evaluator.evaluate_full(random_agent, seed=42)
agent_eval = evaluator.evaluate_full(my_agent, seed=42)

print(f"Improvement over random: {(agent_eval.overall_score - random_eval.overall_score):.3f}")
```

See [Rewards](rewards.md) and [Scoring](scoring.md) for related evaluation concepts.
