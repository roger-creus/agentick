# VLM Agents

Evaluate Vision-Language Models on Agentick using OpenAI, Anthropic, and HuggingFace.

## Quick Start

```python
import agentick
from agentick.leaderboard.adapters import APIAdapter

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
vlm_interface = VLMAgentInterface(env)

obs, info = env.reset()
image_b64 = vlm_interface.encode_observation(obs)
prompt = vlm_interface.format_prompt(image_b64, task_description="Navigate to goal")
action = vlm_interface.parse_action(vlm_response_text)
obs, reward, terminated, truncated, info = env.step(action)
```

## Image Encoding

```python
# PNG (default)
image_b64 = vlm_interface.encode_observation(obs)

# JPEG (smaller)
image_b64 = vlm_interface.encode_observation(obs, format="JPEG", quality=85)

# Resize
image_b64 = vlm_interface.encode_observation(obs, resize=(256, 256))
```

## OpenAI (GPT-4 Vision, GPT-4o)

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
vlm_interface = VLMAgentInterface(env)

obs, info = env.reset()
for step in range(100):
    image_b64 = vlm_interface.encode_observation(obs)
    prompt = vlm_interface.format_prompt(image_b64, task_description="Navigate to goal")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt["text"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prompt['image']}"}}
            ]
        }]
    )

    action_text = response.choices[0].message.content
    action = vlm_interface.parse_action(action_text)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/openai_vision_agent.py`

## Anthropic (Claude 3.5)

```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-...")
env = agentick.make("MazeNavigation-v0", render_mode="rgb_array")
vlm_interface = VLMAgentInterface(env)

obs, info = env.reset()
for step in range(100):
    image_b64 = vlm_interface.encode_observation(obs)
    prompt = vlm_interface.format_prompt(image_b64, task_description="Navigate maze")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt["text"]},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": prompt["image"]}}
            ]
        }]
    )

    action_text = response.content[0].text
    action = vlm_interface.parse_action(action_text)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/anthropic_vision_agent.py`

## HuggingFace (LLaVA, Qwen-VL)

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

model_name = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(model_name, device_map="auto")

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
vlm_interface = VLMAgentInterface(env)

obs, info = env.reset()
for step in range(100):
    image = Image.fromarray(obs)
    text_prompt = f"Task: Navigate to goal. Actions: {info['valid_actions']}. Select one:"

    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    action_text = processor.decode(outputs[0], skip_special_tokens=True)
    action = vlm_interface.parse_action(action_text)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/huggingface_local_agent.py`

## Prompt Engineering

```python
# Basic
prompt = vlm_interface.format_prompt(image_b64, task_description="Navigate to goal")

# Custom template
template = "Analyze the gridworld.\nTASK: {task_description}\nACTIONS: {valid_actions}\nSelect action:"
vlm_interface.set_template(template)

# Chain-of-thought
cot_template = """
IMAGE: <gridworld>
TASK: {task_description}
Think step-by-step:
1. What do I see?
2. Where is the goal?
3. Best action?
ACTIONS: {valid_actions}
"""
vlm_interface.set_template(cot_template)
```

## Complete Examples

See `examples/llm/`:
- `openai_vision_agent.py` - GPT-4o vision
- `anthropic_vision_agent.py` - Claude 3.5 vision
- `huggingface_local_agent.py` - Local vision models
- `compare_llms.py` - Multi-provider comparison
