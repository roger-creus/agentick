# VLM Agents

Evaluate Vision-Language Models on Agentick tasks using pixel observations and visual prompts. This guide covers image encoding, all major VLM providers, and performance optimization.

## Overview

Vision-Language Models are ideal when:
- Tasks have visual components (pixel-based)
- You want models that understand images natively
- You need stronger visual reasoning than text descriptions
- You're testing frontier models (GPT-4V, Claude 3, Gemini)
- You want to study visual understanding in agents

**Advantages:**
- Direct pixel understanding (no conversion to text)
- Strong visual reasoning capabilities
- Frontier models available (GPT-4V, Claude 3)
- Zero training required
- Interpretable via visual prompts

**Disadvantages:**
- More expensive than text LLMs (image costs extra)
- Slower inference (image encoding + processing)
- May struggle with small details in gridworlds
- Fewer open-source options than text LLMs
- Visual understanding can be unpredictable

## Quick Start

### 1. Render Pixel Observations

```python
import agentick

# Create environment with pixel rendering
env = agentick.make(
    "GoToGoal-v0",
    difficulty="easy",
    render_mode="rgb_array"  # Get pixel observations
)

obs, info = env.reset()
print(obs.shape)  # (H, W, 3) uint8
```

### 2. Format VLM Input

```python
from PIL import Image
from agentick.interfaces import VLMAgentInterface

# Convert to PIL Image
image = Image.fromarray(obs)

# Create prompt
vlm_interface = VLMAgentInterface(env)
formatted = vlm_interface.format_observation(
    task_description="Navigate to the green goal"
)
# Returns {"image": PIL.Image, "prompt": str}
```

### 3. Call VLM API

```python
from openai import OpenAI
import base64
from io import BytesIO

client = OpenAI()

# Encode image to base64
buffered = BytesIO()
image.save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode()

# Call GPT-4V
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What action should the agent take?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        }
    ],
    max_tokens=100
)

action_text = response.choices[0].message.content
```

### 4. Parse Action

```python
action = vlm_interface.parse_action(action_text)
obs, reward, terminated, truncated, info = env.step(action)
```

## Image Encoding Strategies

### Base64 Encoding

Best for: API calls to cloud providers

```python
import base64
from io import BytesIO
from PIL import Image

def encode_image_base64(image):
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Usage
image_b64 = encode_image_base64(image)
url = f"data:image/png;base64,{image_b64}"
```

### URL-based Images

Best for: Hosted images, CDN delivery

```python
def upload_image_to_cdn(image, bucket_name):
    """Upload image to cloud storage."""
    import boto3
    from io import BytesIO

    # Convert to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Upload to S3
    s3 = boto3.client("s3")
    key = f"vlm_images/{time.time()}.png"
    s3.put_object(Bucket=bucket_name, Key=key, Body=image_bytes)

    # Return URL
    return f"https://{bucket_name}.s3.amazonaws.com/{key}"
```

### PIL Image Objects

Best for: Local models (HuggingFace, LLaVA)

```python
from PIL import Image
import numpy as np

def numpy_to_pil(obs):
    """Convert numpy array to PIL Image."""
    # obs shape: (H, W, 3) uint8
    return Image.fromarray(obs)

def pil_to_numpy(image):
    """Convert PIL Image back to numpy."""
    return np.array(image)
```

### Tensor Format

Best for: PyTorch-based models

```python
import torch
from torchvision import transforms

def numpy_to_tensor(obs):
    """Convert observation to tensor."""
    # obs shape: (H, W, 3) uint8
    obs_tensor = torch.from_numpy(obs).float() / 255.0
    # Permute to (C, H, W)
    return obs_tensor.permute(2, 0, 1)

def tensor_to_numpy(tensor):
    """Convert tensor to numpy."""
    # tensor shape: (C, H, W) float
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
```

## VLM Providers

### GPT-4 Vision

```python
"""GPT-4V agent implementation."""

from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import agentick
from agentick.interfaces import VLMAgentInterface


class GPT4VisionAgent:
    """GPT-4V agent for visual gridworld navigation."""

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.message_history = []

    def reset(self, task_description=""):
        """Reset for new episode."""
        self.message_history = []
        self.system_prompt = (
            "You are an AI agent in a gridworld environment. "
            "You see the current state as an image. "
            f"{task_description}\n"
            "Respond with a single action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."
        )

    def encode_image(self, image):
        """Encode PIL image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def act(self, observation, task_prompt="What action should you take?"):
        """Get action from GPT-4V."""
        # Ensure observation is PIL Image
        if isinstance(observation, Image.Image):
            image = observation
        else:
            image = Image.fromarray(observation)

        # Encode image
        img_b64 = self.encode_image(image)

        # Add message
        self.message_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": task_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        })

        # Call API
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.message_history
            ],
            max_tokens=100,
        )

        action_text = response.choices[0].message.content

        # Add to history
        self.message_history.append({
            "role": "assistant",
            "content": action_text
        })

        return action_text


def run_episode_gpt4v():
    """Run complete episode with GPT-4V."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
    vlm_interface = VLMAgentInterface(env)
    agent = GPT4VisionAgent()

    obs, info = env.reset()
    agent.reset("Navigate to the green goal square.")

    total_reward = 0

    for step in range(100):
        # Format observation
        image = Image.fromarray(obs)
        prompt = f"Step {step+1}: What action should the agent take?"

        # Get action
        action_text = agent.act(image, prompt)
        print(f"Step {step+1}: {action_text}")

        # Execute
        action = vlm_interface.parse_action(action_text)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"\nSuccess: {info['success']}, Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    run_episode_gpt4v()
```

### Claude 3 Vision

```python
"""Claude 3 Vision agent."""

from anthropic import Anthropic
import base64
from io import BytesIO
from PIL import Image
import agentick


class Claude3VisionAgent:
    """Claude 3 agent with vision capabilities."""

    def __init__(self, api_key=None):
        self.client = Anthropic(api_key=api_key)
        self.messages = []

    def reset(self, task_description=""):
        """Reset for new episode."""
        self.messages = []
        self.task_description = task_description

    def encode_image(self, image):
        """Encode PIL image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def act(self, observation):
        """Get action from Claude 3."""
        # Convert to PIL Image
        if isinstance(observation, Image.Image):
            image = observation
        else:
            image = Image.fromarray(observation)

        img_b64 = self.encode_image(image)

        # Build message with image
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                },
                {
                    "type": "text",
                    "text": "What action should the agent take? Respond with just the action name."
                }
            ]
        }

        self.messages.append(message)

        # Call API
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system=f"You are an AI agent in a gridworld. {self.task_description}",
            messages=self.messages
        )

        action_text = response.content[0].text

        # Add response to history
        self.messages.append({
            "role": "assistant",
            "content": action_text
        })

        return action_text
```

### Gemini Vision

```python
"""Gemini Vision agent."""

import google.generativeai as genai
from PIL import Image
import agentick


class GeminiVisionAgent:
    """Gemini Pro Vision agent."""

    def __init__(self, api_key=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro-vision")

    def reset(self, task_description=""):
        """Reset for new episode."""
        self.task_description = task_description

    def act(self, observation):
        """Get action from Gemini."""
        # Convert to PIL Image
        if isinstance(observation, Image.Image):
            image = observation
        else:
            image = Image.fromarray(observation)

        # Build prompt
        prompt = (
            f"You are an AI agent in a gridworld. {self.task_description}\n\n"
            "Looking at the image, what action should you take? "
            "Respond with just the action name: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."
        )

        # Call API
        response = self.model.generate_content([prompt, image])

        return response.text
```

### LLaVA (Local Open-Source)

```python
"""LLaVA local vision-language model."""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image


class LLaVAAgent:
    """Local LLaVA agent (no API costs)."""

    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def reset(self, task_description=""):
        """Reset for new episode."""
        self.task_description = task_description

    def act(self, observation):
        """Get action from LLaVA."""
        # Convert to PIL Image
        if isinstance(observation, Image.Image):
            image = observation
        else:
            image = Image.fromarray(observation)

        prompt = (
            f"{self.task_description}\n\n"
            "What action should the agent take? "
            "MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT?"
        )

        # Process image and text
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
            )

        action_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return action_text
```

## Prompt Engineering for Vision

### Basic Vision Prompt

```python
def vision_prompt_basic(task_description=""):
    """Basic vision prompt."""
    return f"""You are an AI agent in a gridworld environment viewed from above.

The image shows:
- Blue square: the agent (you)
- Green square: the goal
- Gray squares: walls and obstacles
- White/empty: walkable space

{task_description}

What action should you take to reach the green goal?

Respond with a single action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."""
```

### Detailed Analysis Prompt

```python
def vision_prompt_detailed(task_description=""):
    """Detailed analysis prompt."""
    return f"""Analyze this gridworld image carefully:

1. Locate the blue agent square (your position)
2. Locate the green goal square (target)
3. Identify gray wall obstacles
4. Find the shortest path
5. Decide your next action

{task_description}

What is your next action?
Respond with: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."""
```

### Multi-Shot Learning Prompt

```python
def vision_prompt_few_shot(examples=None):
    """Few-shot learning with example images."""
    if examples is None:
        examples = []

    prompt = "You are an AI agent in a gridworld. Here are examples:\n\n"

    for i, example in enumerate(examples):
        prompt += (
            f"Example {i+1}:\n"
            f"[Image shows agent at {example['agent_pos']}, goal at {example['goal_pos']}]\n"
            f"Correct action: {example['action']}\n\n"
        )

    prompt += "Now for the current state, what action should you take?"
    return prompt
```

## Image Optimization for Cost

### 1. Resize Images

```python
from PIL import Image

def resize_for_api(obs, target_size=(224, 224)):
    """Resize image to reduce token usage."""
    if isinstance(obs, Image.Image):
        image = obs
    else:
        image = Image.fromarray(obs)

    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    return image
```

### 2. Compress Images

```python
from PIL import Image
from io import BytesIO

def compress_image(obs, quality=85):
    """Compress image to reduce size."""
    if isinstance(obs, Image.Image):
        image = obs
    else:
        image = Image.fromarray(obs)

    # Convert to RGB (remove alpha if present)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Compress
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    buffered.seek(0)

    return Image.open(buffered)
```

### 3. Selective Processing

```python
class CachedVLMAgent:
    """Only process key frames."""

    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.last_action = None
        self.frame_count = 0

    def get_action(self, obs):
        """Get action, skipping some frames."""
        self.frame_count += 1

        # Skip intermediate frames
        if self.frame_count % self.check_interval != 0:
            return self.last_action or 0

        # Actually call VLM
        action = self.call_vlm(obs)
        self.last_action = action

        return action
```

## Performance Optimization

### Parallel Processing

```python
"""Process multiple images in parallel."""

import concurrent.futures
from typing import List, Tuple
from PIL import Image


class ParallelVLMAgent:
    """Process multiple observations concurrently."""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def get_actions_batch(self, observations: List) -> List[str]:
        """Get actions for multiple observations in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.get_action_single, obs)
                for obs in observations
            ]
            actions = [future.result() for future in futures]

        return actions

    def get_action_single(self, obs):
        """Get action for single observation."""
        # Implement VLM call
        pass
```

### Batch API Calls

```python
"""Batch multiple images in one API call."""


def batch_get_actions_gpt4v(images: List, prompts: List = None):
    """Get actions for multiple images in one call."""
    from openai import OpenAI
    import base64
    from io import BytesIO

    client = OpenAI()

    if prompts is None:
        prompts = ["What action?"] * len(images)

    # Encode all images
    encoded_images = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # Build batch message
    content = [
        {"type": "text", "text": "For each image, suggest the best action.\n\n"}
    ]

    for i, (img_b64, prompt) in enumerate(zip(encoded_images, prompts)):
        content.append({
            "type": "text",
            "text": f"Image {i+1}: {prompt}"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    # Call API once
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": content}],
        max_tokens=200,
    )

    # Parse responses
    return response.choices[0].message.content
```

## Cost Analysis and Optimization

```python
"""Analyze and optimize VLM costs."""

from dataclasses import dataclass
from typing import Dict
import json


@dataclass
class VLMCost:
    """VLM API pricing."""
    name: str
    input_token_price: float  # per 1000 tokens
    output_token_price: float  # per 1000 tokens
    image_token_cost: float  # per image


# Updated pricing (check official docs for current rates)
VLM_PRICING = {
    "gpt-4-vision": VLMCost(
        name="GPT-4 Vision",
        input_token_price=0.01,
        output_token_price=0.03,
        image_token_cost=0.085,  # per image
    ),
    "claude-3-opus": VLMCost(
        name="Claude 3 Opus",
        input_token_price=0.015,
        output_token_price=0.075,
        image_token_cost=0.001152,  # per image
    ),
}


class VLMCostTracker:
    """Track VLM API costs."""

    def __init__(self):
        self.calls = []

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        num_images: int = 1,
    ):
        """Log API call."""
        if model not in VLM_PRICING:
            print(f"Unknown model: {model}")
            return

        pricing = VLM_PRICING[model]
        cost = (
            (input_tokens / 1000) * pricing.input_token_price +
            (output_tokens / 1000) * pricing.output_token_price +
            num_images * pricing.image_token_cost
        )

        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_images": num_images,
            "cost": cost,
        })

    def total_cost(self) -> float:
        """Get total cost."""
        return sum(c["cost"] for c in self.calls)

    def cost_per_episode(self, max_steps: int = 100) -> float:
        """Estimate cost per episode."""
        if not self.calls:
            return 0

        avg_cost = self.total_cost() / len(self.calls)
        return avg_cost * max_steps

    def report(self):
        """Print cost report."""
        print("\n=== VLM Cost Report ===")
        print(f"Total API calls: {len(self.calls)}")
        print(f"Total cost: ${self.total_cost():.4f}")

        # By model
        by_model = {}
        for call in self.calls:
            model = call["model"]
            if model not in by_model:
                by_model[model] = {"calls": 0, "cost": 0}
            by_model[model]["calls"] += 1
            by_model[model]["cost"] += call["cost"]

        print("\nBy model:")
        for model, stats in by_model.items():
            print(
                f"  {model}: {stats['calls']} calls, "
                f"${stats['cost']:.4f}"
            )

        print(f"\nEstimated cost per 100-step episode: "
              f"${self.cost_per_episode(100):.4f}")


# Usage
tracker = VLMCostTracker()

# Log calls
tracker.log_call(
    model="gpt-4-vision",
    input_tokens=200,
    output_tokens=50,
    num_images=1,
)

tracker.report()
```

## Complete VLM Benchmark Example

```python
"""Comprehensive VLM benchmark."""

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import agentick
from agentick.interfaces import VLMAgentInterface


@dataclass
class VLMBenchmarkResult:
    """Result from VLM benchmark."""
    model: str
    task: str
    difficulty: str
    success: bool
    reward: float
    steps: int
    duration: float
    cost_estimate: float


class VLMBenchmark:
    """Benchmark VLM agents."""

    def __init__(self):
        self.results: List[VLMBenchmarkResult] = []

    def run(
        self,
        tasks: List[str],
        models: Dict[str, callable],
        difficulties: List[str] = ["easy", "medium"],
        n_episodes: int = 3,
    ):
        """Run complete benchmark."""
        for task in tasks:
            for difficulty in difficulties:
                print(f"\nTask: {task}, Difficulty: {difficulty}")

                env = agentick.make(
                    task,
                    difficulty=difficulty,
                    render_mode="rgb_array"
                )
                vlm_interface = VLMAgentInterface(env)

                for model_name, agent_class in models.items():
                    print(f"  Model: {model_name}")

                    agent = agent_class()

                    for episode in range(n_episodes):
                        start = time.time()

                        obs, info = env.reset()
                        agent.reset(f"Complete {task}")

                        total_reward = 0
                        steps = 0

                        for step in range(100):
                            image = Image.fromarray(obs)
                            action_text = agent.act(image)
                            action = vlm_interface.parse_action(action_text)

                            obs, reward, terminated, truncated, info = env.step(action)
                            total_reward += reward
                            steps += 1

                            if terminated or truncated:
                                break

                        duration = time.time() - start

                        result = VLMBenchmarkResult(
                            model=model_name,
                            task=task,
                            difficulty=difficulty,
                            success=info.get("success", False),
                            reward=total_reward,
                            steps=steps,
                            duration=duration,
                            cost_estimate=0.0,  # Update based on model
                        )
                        self.results.append(result)

                        print(f"    Episode {episode+1}: "
                              f"success={result.success}, "
                              f"reward={result.reward:.2f}, "
                              f"time={result.duration:.1f}s")

                env.close()

    def save_results(self, path="vlm_benchmark.json"):
        """Save results."""
        with open(path, "w") as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2
            )

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("VLM BENCHMARK SUMMARY")
        print("="*60)

        # Group by model and task
        by_model_task = {}
        for result in self.results:
            key = (result.model, result.task)
            if key not in by_model_task:
                by_model_task[key] = []
            by_model_task[key].append(result)

        for (model, task), results in by_model_task.items():
            successes = sum(1 for r in results if r.success)
            total = len(results)
            avg_reward = sum(r.reward for r in results) / total
            avg_steps = sum(r.steps for r in results) / total

            print(f"\n{model} - {task}:")
            print(f"  Success Rate: {successes}/{total} ({100*successes/total:.1f}%)")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")


if __name__ == "__main__":
    from PIL import Image

    # Define models
    models = {
        "GPT-4V": GPT4VisionAgent,
        "Claude3": Claude3VisionAgent,
        "Gemini": GeminiVisionAgent,
        "LLaVA": LLaVAAgent,
    }

    # Run benchmark
    benchmark = VLMBenchmark()
    benchmark.run(
        tasks=["GoToGoal-v0"],
        models=models,
        difficulties=["easy", "medium"],
        n_episodes=2,
    )

    benchmark.print_summary()
    benchmark.save_results()
```

## Best Practices

1. **Compress images** - Reduce from 512x512 to 224x224
2. **Cache responses** - For identical frames
3. **Skip frames** - Only process every Nth frame
4. **Monitor costs** - VLM APIs are expensive
5. **Use robust parsing** - Handle formatting variations
6. **Test on samples** - Validate before full benchmarks
7. **Batch when possible** - Multiple images per API call
8. **Choose local models** - For cost-sensitive applications
9. **Use conversation history** - Maintain context
10. **Document assumptions** - About image format and resolution

## Debugging Checklist

- [ ] Image encoding correct (base64, PIL, tensor)
- [ ] Prompt format matches API requirements
- [ ] Action parsing handles all model outputs
- [ ] Image size reasonable (not too large or small)
- [ ] API keys valid and have sufficient credits
- [ ] Proper error handling for API failures
- [ ] Timeout handling for slow responses
- [ ] Cost tracking and alerts configured
