# LLM Agents

Evaluate Large Language Models directly on Agentick tasks using natural language prompts. This guide covers all major providers, prompt engineering strategies, and production best practices.

## Overview

LLM agents are ideal when:
- You want to leverage pre-trained models (no training)
- You need strong zero-shot or few-shot performance
- You want to test frontier models (GPT-4, Claude, etc.)
- You're studying language understanding and reasoning
- You want interpretable decision-making through prompts

**Advantages:**
- Zero training required - use immediately
- Strong zero-shot reasoning capabilities
- Access to state-of-the-art models (GPT-4, Claude 3, Gemini)
- Interpretable decisions through prompts
- Support for multi-shot learning via prompts
- Easy to iterate on prompts

**Disadvantages:**
- API costs accumulate quickly (100+ queries per episode)
- Slower than local models (network latency)
- LLMs may struggle with precise action formats
- Input token costs for observations
- Less reliable than RL for perfect execution

## Quick Start

### 1. Setup LLMAgentInterface

```python
import agentick
from agentick.interfaces import LLMAgentInterface

# Create environment with language observations
env = agentick.make(
    "GoToGoal-v0",
    difficulty="easy",
    render_mode="language"  # Natural language descriptions
)

# Create interface
llm_interface = LLMAgentInterface(env)

# Format prompt
obs, info = env.reset()
prompt = llm_interface.format_prompt(
    obs,
    task_description="Navigate to the goal efficiently"
)
print(prompt)
```

### 2. Call LLM API

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an AI agent in a gridworld."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    max_tokens=100
)

action_text = response.choices[0].message.content
print(f"LLM response: {action_text}")
```

### 3. Parse Action

```python
# Parse LLM response to action
action = llm_interface.parse_action(action_text)
obs, reward, terminated, truncated, info = env.step(action)
```

## Provider Examples

### OpenAI (GPT-3.5, GPT-4)

#### GPT-4 Agent

```python
"""Full OpenAI GPT-4 agent example."""

from openai import OpenAI
import agentick
from agentick.interfaces import LLMAgentInterface

client = OpenAI(api_key="your-api-key")


class GPT4Agent:
    """GPT-4 agent for Agentick tasks."""

    def __init__(self, model="gpt-4", temperature=0, max_tokens=100):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []

    def reset(self, task_description=""):
        """Reset agent state for new episode."""
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI agent in a gridworld environment. "
                    "You need to navigate efficiently to reach your goal. "
                    f"{task_description}"
                ),
            }
        ]

    def act(self, observation):
        """Get action from GPT-4."""
        # Add user message
        self.messages.append({"role": "user", "content": observation})

        # Call API
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        action_text = response.choices[0].message.content

        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": action_text})

        return action_text

    def get_token_usage(self):
        """Track token usage for cost calculation."""
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        return response.usage.prompt_tokens, response.usage.completion_tokens


def run_episode_gpt4():
    """Run complete episode with GPT-4."""
    env = agentick.make(
        "GoToGoal-v0",
        difficulty="easy",
        render_mode="language"
    )
    llm_interface = LLMAgentInterface(env)
    agent = GPT4Agent(model="gpt-4")

    obs, info = env.reset()
    agent.reset("Navigate to the green goal marked as G.")

    total_reward = 0
    total_tokens = 0
    steps = 0

    for step in range(100):
        # Format observation
        prompt = llm_interface.format_prompt(obs)

        # Get action from GPT-4
        action_text = agent.act(prompt)
        print(f"Step {step+1}: {action_text}")

        # Parse and execute
        action = llm_interface.parse_action(action_text)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"Episode finished! Success: {info['success']}")
            break

    print(f"\nResults:")
    print(f"  Total reward: {total_reward}")
    print(f"  Steps: {steps}")
    print(f"  Success: {info['success']}")

    env.close()


if __name__ == "__main__":
    run_episode_gpt4()
```

#### Cost Tracking

```python
"""Track API costs for LLM agents."""

import os
from datetime import datetime

class OpenAICostTracker:
    """Track OpenAI API costs."""

    # Pricing (update these based on current rates)
    PRICING = {
        "gpt-4": {
            "input": 0.03 / 1000,      # $0.03 per 1k tokens
            "output": 0.06 / 1000,     # $0.06 per 1k tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.0005 / 1000,    # $0.0005 per 1k tokens
            "output": 0.0015 / 1000,   # $0.0015 per 1k tokens
        },
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls_by_model = {}

    def log_usage(self, model, input_tokens, output_tokens):
        """Log token usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if model not in self.calls_by_model:
            self.calls_by_model[model] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }

        self.calls_by_model[model]["calls"] += 1
        self.calls_by_model[model]["input_tokens"] += input_tokens
        self.calls_by_model[model]["output_tokens"] += output_tokens

        model_cost = (
            input_tokens * self.PRICING[model]["input"]
            + output_tokens * self.PRICING[model]["output"]
        )
        self.calls_by_model[model]["cost"] += model_cost

    def total_cost(self):
        """Calculate total cost."""
        return sum(m["cost"] for m in self.calls_by_model.values())

    def report(self):
        """Print cost report."""
        print("\n=== OpenAI Cost Report ===")
        print(f"Total API calls: {sum(m['calls'] for m in self.calls_by_model.values())}")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Total cost: ${self.total_cost():.4f}")
        print("\nBy model:")
        for model, stats in self.calls_by_model.items():
            print(
                f"  {model}: {stats['calls']} calls, "
                f"{stats['input_tokens']} input, "
                f"{stats['output_tokens']} output, "
                f"${stats['cost']:.4f}"
            )

# Usage
tracker = OpenAICostTracker()

# After each API call
response = client.chat.completions.create(...)
tracker.log_usage(
    model="gpt-4",
    input_tokens=response.usage.prompt_tokens,
    output_tokens=response.usage.completion_tokens,
)

tracker.report()
```

### Anthropic Claude

```python
"""Claude agent using Anthropic API."""

from anthropic import Anthropic
import agentick
from agentick.interfaces import LLMAgentInterface


class ClaudeAgent:
    """Claude agent for Agentick tasks."""

    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key="your-api-key")
        self.model = model
        self.conversation_history = []

    def reset(self, task_description=""):
        """Reset agent state."""
        self.conversation_history = []
        self.system_prompt = (
            "You are an AI agent in a gridworld environment. "
            "Your goal is to navigate efficiently to reach the target. "
            f"{task_description}\n\n"
            "When you see the current state, respond with a single action: "
            "MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."
        )

    def act(self, observation):
        """Get action from Claude."""
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": observation
        })

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            system=self.system_prompt,
            messages=self.conversation_history,
        )

        action_text = response.content[0].text

        # Add assistant response
        self.conversation_history.append({
            "role": "assistant",
            "content": action_text
        })

        return action_text


def run_episode_claude():
    """Run episode with Claude."""
    env = agentick.make("GoToGoal-v0", render_mode="language")
    llm_interface = LLMAgentInterface(env)
    agent = ClaudeAgent()

    obs, info = env.reset()
    agent.reset("Navigate to the goal marked as G.")

    total_reward = 0

    for step in range(100):
        prompt = llm_interface.format_prompt(obs)
        action_text = agent.act(prompt)

        action = llm_interface.parse_action(action_text)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Success: {info['success']}, Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    run_episode_claude()
```

### Google Gemini

```python
"""Gemini agent using Google API."""

import google.generativeai as genai
import agentick
from agentick.interfaces import LLMAgentInterface


class GeminiAgent:
    """Gemini agent for Agentick tasks."""

    def __init__(self, api_key="your-api-key"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.conversation_history = []

    def reset(self, task_description=""):
        """Reset agent state."""
        self.conversation_history = []
        self.task_description = task_description

    def act(self, observation):
        """Get action from Gemini."""
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "parts": [observation]
        })

        # Create message with full context
        messages = []
        system_prompt = (
            "You are an AI agent in a gridworld. "
            f"{self.task_description} "
            "Respond with a single action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."
        )

        # Call Gemini
        response = self.model.generate_content(
            [system_prompt] + [m["parts"][0] for m in self.conversation_history],
        )

        action_text = response.text

        # Add to history
        self.conversation_history.append({
            "role": "model",
            "parts": [action_text]
        })

        return action_text


def run_episode_gemini():
    """Run episode with Gemini."""
    env = agentick.make("GoToGoal-v0", render_mode="language")
    llm_interface = LLMAgentInterface(env)
    agent = GeminiAgent(api_key="your-api-key")

    obs, info = env.reset()
    agent.reset("Navigate to the goal.")

    total_reward = 0

    for step in range(100):
        prompt = llm_interface.format_prompt(obs)
        action_text = agent.act(prompt)

        action = llm_interface.parse_action(action_text)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Success: {info['success']}")
    env.close()


if __name__ == "__main__":
    run_episode_gemini()
```

### Local Models (HuggingFace, vLLM)

```python
"""Local LLM agent using vLLM or HuggingFace."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalLLMAgent:
    """Local LLM agent using HuggingFace models."""

    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.conversation_history = []

    def reset(self, task_description=""):
        """Reset agent state."""
        self.conversation_history = []
        self.system_prompt = (
            "You are an AI agent in a gridworld environment. "
            f"{task_description}\n"
            "Respond with a single action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT."
        )

    def act(self, observation):
        """Get action from local LLM."""
        # Build prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation},
        ]

        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode
        action_text = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )

        return action_text


# vLLM version (faster inference)
class VLLMAgent:
    """Agent using vLLM for fast inference."""

    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,
        )

    def act(self, observation):
        """Get action from vLLM."""
        prompt = f"Observation: {observation}\nAction:"

        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text
```

## Prompt Templates

### Basic Template

```python
def format_basic_prompt(observation, task_description, valid_actions):
    """Basic prompt template."""
    return f"""You are an AI agent in a gridworld environment.

Task: {task_description}

Current state:
{observation}

Available actions: {", ".join(valid_actions)}

What action should you take? Respond with just the action name."""
```

### Chain-of-Thought Template

```python
def format_cot_prompt(observation, task_description, valid_actions):
    """Chain-of-thought prompt."""
    return f"""You are an AI agent in a gridworld environment.

Task: {task_description}

Current state:
{observation}

Available actions: {", ".join(valid_actions)}

Let's think step by step:
1. Where am I currently?
2. Where is the goal?
3. What obstacles are present?
4. What is the best next action?

What action should you take? Respond with just the action name."""
```

### Few-Shot Template

```python
def format_few_shot_prompt(observation, task_description, examples, valid_actions):
    """Few-shot learning prompt with examples."""
    examples_text = "\n\n".join([
        f"State: {e['state']}\nAction: {e['action']}"
        for e in examples
    ])

    return f"""You are an AI agent in a gridworld environment.

Task: {task_description}

Examples of good actions:
{examples_text}

Current state:
{observation}

Available actions: {", ".join(valid_actions)}

What action should you take? Respond with just the action name."""
```

## Action Parsing Strategies

### Text Parsing

```python
def parse_action_text(response, action_map):
    """Parse action from text response."""
    response = response.strip().upper()

    # Exact match
    if response in action_map:
        return action_map[response]

    # Substring search
    for action_name, action_id in action_map.items():
        if action_name in response:
            return action_id

    # First valid action found
    words = response.split()
    for word in words:
        if word in action_map:
            return action_map[word]

    # Default to first action
    return list(action_map.values())[0]


# Usage
action_map = {
    "MOVE_UP": 0,
    "MOVE_DOWN": 1,
    "MOVE_LEFT": 2,
    "MOVE_RIGHT": 3,
    "UP": 0,  # Aliases
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
}

response = "I should MOVE_RIGHT to get closer to the goal"
action = parse_action_text(response, action_map)
```

### Function Calling (OpenAI)

```python
"""Use OpenAI function calling for structured outputs."""

import json
from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def get_action_with_function_calling(observation):
    """Get structured action using function calling."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"What action should the agent take?\n\n{observation}"
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Take an action in the gridworld",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"],
                                "description": "The action to take"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Why this action is chosen"
                            }
                        },
                        "required": ["action"]
                    }
                }
            }
        ],
        tool_choice="auto"
    )

    # Parse function call
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return args["action"], args.get("reasoning", "")

    return "MOVE_UP", "Default action"


# Usage
action, reasoning = get_action_with_function_calling(observation)
print(f"Action: {action}, Reasoning: {reasoning}")
```

### JSON Mode (Claude)

```python
"""Use Claude's JSON mode for structured outputs."""

from anthropic import Anthropic
import json

client = Anthropic(api_key="your-api-key")


def get_action_with_json(observation):
    """Get structured action using JSON mode."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""Respond with a JSON object in this format:
{{
  "action": "MOVE_UP" or "MOVE_DOWN" or "MOVE_LEFT" or "MOVE_RIGHT",
  "reasoning": "Why this action?"
}}

Observation:
{observation}"""
            }
        ],
    )

    text = response.content[0].text

    # Try to parse JSON
    try:
        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        parsed = json.loads(json_str)
        return parsed["action"], parsed.get("reasoning", "")
    except:
        # Fallback to text parsing
        for action in ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]:
            if action in text:
                return action, text
        return "MOVE_UP", text
```

## Cost Optimization Strategies

### 1. Observation Compression

```python
def compress_observation(obs, max_length=500):
    """Compress observation to reduce token count."""
    if isinstance(obs, str):
        # Truncate text observations
        return obs[:max_length]
    elif isinstance(obs, dict):
        # Keep only essential fields
        return {
            "position": obs.get("agent_pos"),
            "goal": obs.get("goal_pos"),
            "nearby": obs.get("nearby_cells", [])[:5],  # Limit nearby cells
        }
    return obs
```

### 2. Action Caching

```python
from functools import lru_cache


class CachedLLMAgent:
    """Cache LLM responses for identical observations."""

    def __init__(self, llm_func, cache_size=1000):
        self.llm_func = llm_func
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_action(self, observation):
        """Get action with caching."""
        obs_hash = hash(str(observation))

        if obs_hash in self.cache:
            self.hits += 1
            return self.cache[obs_hash]

        self.misses += 1
        action = self.llm_func(observation)
        self.cache[obs_hash] = action

        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        return action

    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": hit_rate,
        }
```

### 3. Batch Processing

```python
"""Process multiple observations in one API call."""


def batch_get_actions(observations, model_name="gpt-4"):
    """Get actions for multiple observations efficiently."""
    from openai import OpenAI

    client = OpenAI()

    # Combine observations
    batch_prompt = "For each observation below, what action should be taken?\n\n"
    for i, obs in enumerate(observations):
        batch_prompt += f"Observation {i+1}:\n{obs}\n\n"

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": batch_prompt}],
    )

    # Parse response to extract actions
    response_text = response.choices[0].message.content
    actions = []
    for action_name in ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]:
        if action_name in response_text:
            actions.append(action_name)

    return actions
```

## Complete Working Example: Multi-Provider Benchmark

```python
"""Benchmark multiple LLM providers on Agentick tasks."""

import json
import time
from typing import Dict, List
from dataclasses import dataclass

import agentick
from agentick.interfaces import LLMAgentInterface


@dataclass
class EpisodeResult:
    """Result from one episode."""
    model: str
    task: str
    success: bool
    reward: float
    steps: int
    duration: float
    tokens_used: int
    cost: float


class LLMBenchmark:
    """Benchmark LLM agents across multiple providers."""

    def __init__(self):
        self.results: List[EpisodeResult] = []
        self.providers = {}

    def register_provider(self, name: str, provider_class):
        """Register an LLM provider."""
        self.providers[name] = provider_class

    def run_benchmark(self, tasks: List[str], providers: List[str], n_episodes=5):
        """Run benchmark across tasks and providers."""
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print('='*60)

            env = agentick.make(task, render_mode="language")
            llm_interface = LLMAgentInterface(env)

            for provider_name in providers:
                if provider_name not in self.providers:
                    print(f"Provider {provider_name} not registered")
                    continue

                print(f"\n{provider_name}:")

                provider_class = self.providers[provider_name]
                agent = provider_class()

                for episode in range(n_episodes):
                    start_time = time.time()

                    obs, info = env.reset()
                    agent.reset(f"Navigate to goal in {task}")

                    total_reward = 0
                    steps = 0
                    tokens_used = 0

                    for step in range(100):
                        prompt = llm_interface.format_prompt(obs)
                        action_text = agent.act(prompt)
                        action = llm_interface.parse_action(action_text)

                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        steps += 1

                        if terminated or truncated:
                            break

                    duration = time.time() - start_time

                    # Create result
                    result = EpisodeResult(
                        model=provider_name,
                        task=task,
                        success=info.get("success", False),
                        reward=total_reward,
                        steps=steps,
                        duration=duration,
                        tokens_used=tokens_used,
                        cost=0.0,  # Update based on provider
                    )
                    self.results.append(result)

                    print(f"  Episode {episode+1}: "
                          f"success={result.success}, "
                          f"reward={result.reward:.2f}, "
                          f"steps={result.steps}")

            env.close()

    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            print("No results to summarize")
            return

        # Group by model
        by_model = {}
        for result in self.results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        for model, results in by_model.items():
            successes = sum(1 for r in results if r.success)
            total = len(results)
            avg_reward = sum(r.reward for r in results) / total
            avg_steps = sum(r.steps for r in results) / total

            print(f"\n{model}:")
            print(f"  Success Rate: {successes}/{total} ({100*successes/total:.1f}%)")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")

    def save_results(self, path="benchmark_results.json"):
        """Save results to file."""
        data = [
            {
                "model": r.model,
                "task": r.task,
                "success": r.success,
                "reward": r.reward,
                "steps": r.steps,
                "duration": r.duration,
                "tokens_used": r.tokens_used,
                "cost": r.cost,
            }
            for r in self.results
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {path}")


# Example usage
if __name__ == "__main__":
    from examples.llm.openai_agent import GPT4Agent
    from examples.llm.anthropic_agent import ClaudeAgent
    from examples.llm.gemini_agent import GeminiAgent

    benchmark = LLMBenchmark()
    benchmark.register_provider("GPT-4", GPT4Agent)
    benchmark.register_provider("Claude", ClaudeAgent)
    benchmark.register_provider("Gemini", GeminiAgent)

    benchmark.run_benchmark(
        tasks=["GoToGoal-v0", "KeyDoorPuzzle-v0"],
        providers=["GPT-4", "Claude", "Gemini"],
        n_episodes=3,
    )

    benchmark.print_summary()
    benchmark.save_results()
```

## Best Practices

1. **Set temperature to 0** - For deterministic, reproducible behavior
2. **Use clear action formats** - Guide LLM with explicit choices
3. **Compress observations** - Reduce token usage and cost
4. **Cache responses** - For identical observations
5. **Handle failures gracefully** - Have sensible defaults if parsing fails
6. **Track token usage** - Monitor API costs
7. **Test prompts locally** - Before running expensive evaluations
8. **Use conversation history** - Maintain context across steps
9. **Batch process when possible** - Reduce API calls
10. **Monitor success rates** - Different models have different capabilities

## Debugging Tips

```python
"""Debugging LLM agent issues."""


def debug_agent(env, agent, task_description, max_steps=10):
    """Debug agent behavior step by step."""
    from agentick.interfaces import LLMAgentInterface

    llm_interface = LLMAgentInterface(env)
    agent.reset(task_description)

    obs, info = env.reset()

    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")
        print(f"Observation:\n{obs}\n")

        prompt = llm_interface.format_prompt(obs)
        print(f"Prompt:\n{prompt}\n")

        action_text = agent.act(prompt)
        print(f"LLM Response:\n{action_text}\n")

        action = llm_interface.parse_action(action_text)
        print(f"Parsed Action: {action}\n")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Info: {info}\n")

        if terminated or truncated:
            print(f"Episode finished: Success={info['success']}")
            break

    env.close()


# Usage
debug_agent(env, agent, "Navigate to the goal")
```
