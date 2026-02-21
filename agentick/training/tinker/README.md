# Tinker Integration for Agentick

Train LLMs/VLMs to play agentick tasks using [Tinker](https://tinker-docs.thinkingmachines.ai/) for remote LoRA fine-tuning with SFT and RL.

## Prerequisites

```bash
# Install Tinker SDK
pip install tinker

# Set API key
export TINKER_API_KEY="your-api-key"

# Install agentick with training extras
uv sync --extra finetune
```

## Pipeline: SFT Warmstart -> RL Fine-Tune

### Step 1: Collect Oracle Demonstrations

```python
from agentick.oracles import get_oracle
from agentick.data import DataCollector
import agentick

env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")
oracle = get_oracle("GoToGoal-v0", env)
collector = DataCollector(env, oracle, record_modalities=["language"])
dataset = collector.collect(num_episodes=100, seeds=range(100))
dataset.export_to_huggingface("data/gotogoal_conv/", format="conversation")
env.close()
```

### Step 2: SFT with Tinker

```python
from agentick.training.tinker.sft import TinkerSFTTrainer

trainer = TinkerSFTTrainer(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="data/gotogoal_conv/",
    rank=32,
)
trainer.train(num_steps=100, learning_rate=1e-4)

# Quick evaluation
agent = trainer.as_agent()
env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")
obs, info = env.reset(seed=0)
done = False
while not done:
    action = agent.act(obs, info)
    obs, reward, done, trunc, info = env.step(action)
    done = done or trunc
print(f"Success: {info.get('success', False)}")
```

### Step 3: RL Fine-Tune with Tinker

```python
from agentick.training.tinker.rl import TinkerRLTrainer

rl_trainer = TinkerRLTrainer(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    task_id="GoToGoal-v0",
    difficulty="medium",
    rank=32,
    loss_fn="ppo",  # or "importance_sampling" for REINFORCE
)
metrics = rl_trainer.train(
    num_episodes=100,
    learning_rate=1e-5,
)
print(f"Final success rate: {metrics['episode_successes'][-10:]}")
```

## Configuration

### TinkerSFTTrainer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `"Qwen/Qwen2.5-7B-Instruct"` | Tinker-supported model |
| `rank` | `32` | LoRA rank |
| `num_steps` | `100` | Training steps |
| `learning_rate` | `1e-4` | Adam LR |

### TinkerRLTrainer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `"Qwen/Qwen2.5-7B-Instruct"` | Tinker-supported model |
| `task_id` | `"GoToGoal-v0"` | Agentick task |
| `difficulty` | `"easy"` | Task difficulty |
| `rank` | `32` | LoRA rank |
| `loss_fn` | `"ppo"` | `"ppo"` or `"importance_sampling"` |
| `num_episodes` | `100` | Training episodes |
| `learning_rate` | `1e-5` | Adam LR |

## Supported Models

Tinker supports the following model families (check `tinker.ServiceClient().get_server_capabilities()` for latest):
- Qwen2.5 series (0.5B - 72B)
- Qwen3-VL series (for multimodal tasks)
- Llama 3.x series

## Without Tinker Installed

If Tinker is not installed, importing the modules will succeed but creating trainer instances will raise `ImportError` with installation instructions:

```python
try:
    from agentick.training.tinker.sft import TinkerSFTTrainer
    trainer = TinkerSFTTrainer(...)  # Raises ImportError
except ImportError as e:
    print(f"Tinker not available: {e}")
```
