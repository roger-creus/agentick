# LLM Agents

Evaluate LLMs on Agentick using OpenAI, Anthropic, and HuggingFace.

## Quick Start

```python
import agentick
from agentick.leaderboard.adapters import APIAdapter

env = agentick.make("GoToGoal-v0", render_mode="language")
llm_interface = LLMAgentInterface(env)

obs, info = env.reset()
prompt = llm_interface.format_prompt(obs, task_description="Navigate to goal")
action = llm_interface.parse_action(llm_response_text)
obs, reward, terminated, truncated, info = env.step(action)
```

## OpenAI

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
env = agentick.make("GoToGoal-v0", render_mode="language")
llm_interface = LLMAgentInterface(env)

obs, info = env.reset()
for step in range(100):
    prompt = llm_interface.format_prompt(obs, task_description="Navigate to goal")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    action_text = response.choices[0].message.content
    action = llm_interface.parse_action(action_text)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/openai_text_agent.py`

## Anthropic

```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-...")
env = agentick.make("GoToGoal-v0", render_mode="language")
llm_interface = LLMAgentInterface(env)

obs, info = env.reset()
for step in range(100):
    prompt = llm_interface.format_prompt(obs, task_description="Navigate maze")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    action_text = response.content[0].text
    action = llm_interface.parse_action(action_text)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/anthropic_text_agent.py`

## HuggingFace

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

env = agentick.make("GoToGoal-v0", render_mode="language")
llm_interface = LLMAgentInterface(env)

obs, info = env.reset()
for step in range(100):
    prompt = llm_interface.format_prompt(obs, task_description="Navigate to goal")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    action_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    action = llm_interface.parse_action(action_text)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/huggingface_local_agent.py`

## Prompt Templates

```python
# Basic
llm_interface.format_prompt(obs, task_description="Your task")

# Custom template
template = "ENVIRONMENT: {observation}\nGOAL: {task_description}\nACTIONS: {valid_actions}"
llm_interface.set_template(template)

# Few-shot examples
examples = [
    {"observation": "At (1,1). Goal at (3,3).", "action": "move_right"},
    {"observation": "At (2,1). Goal at (3,3).", "action": "move_down"}
]
llm_interface.add_few_shot_examples(examples)
```

## Action Parsing

```python
# Parse from LLM text
action = llm_interface.parse_action("move_up")

# Custom parser
def custom_parser(text: str, valid_actions: list[str]) -> int:
    for action_name in valid_actions:
        if action_name.lower() in text.lower():
            return env.action_space.get_action_index(action_name)
    return env.action_space.get_action_index("noop")

llm_interface.set_action_parser(custom_parser)
```

## Complete Examples

See `examples/llm/`:
- `openai_text_agent.py` - OpenAI GPT-4
- `anthropic_text_agent.py` - Anthropic Claude
- `huggingface_local_agent.py` - Local HF models
- `compare_llms.py` - Multi-provider comparison
