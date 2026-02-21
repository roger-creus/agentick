# LLM Agents

Examples of using large language models -- both cloud APIs and local models --
as Agentick agents. Covers text and vision observations, chain-of-thought
prompting, and multi-provider comparison.

## Prerequisites

```bash
uv sync --extra llm        # openai, anthropic, transformers, torch, Pillow
pip install python-dotenv   # for .env file loading (used by most scripts)
```

Set API keys as environment variables or in a `.env` file:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Scripts

### Foundation

- **base_llm_agent.py** -- Abstract base class (`BaseLLMAgent`) providing
  prompt construction, action parsing with retry logic, and action history
  tracking. Not meant to be run directly; imported by `openai_agent.py`.

### Text Agents (language observations)

- **openai_agent.py** -- Minimal OpenAI agent using `BaseLLMAgent`. Calls
  GPT-4 to choose actions from text observations. Requires `OPENAI_API_KEY`.
- **openai_text_agent.py** -- OpenAI GPT-4o-mini text agent using the
  `APIAgent` adapter from the leaderboard module. Records a sample video.
  Requires `OPENAI_API_KEY`.
- **openai_cot_agent.py** -- GPT-4o-mini with chain-of-thought prompting that
  asks the model to reason step-by-step before selecting an action. Often
  improves performance on complex tasks. Requires `OPENAI_API_KEY`.
- **anthropic_text_agent.py** -- Claude Sonnet 4 text agent using the
  `APIAgent` adapter. Same structure as the OpenAI text agent but calls the
  Anthropic API. Requires `ANTHROPIC_API_KEY`.

### Vision Agents (pixel observations)

- **openai_vision_agent.py** -- GPT-4o vision agent that sends rendered pixel
  frames as base64 images. Records episodes via `gym.wrappers.RecordVideo`.
  Requires `OPENAI_API_KEY` and Pillow.
- **anthropic_vision_agent.py** -- Claude Sonnet 4 vision agent that sends
  pixel frames via the Anthropic messages API. Records episodes as video.
  Requires `ANTHROPIC_API_KEY` and Pillow.

### Local Model

- **huggingface_local_agent.py** -- Run a local HuggingFace model
  (Qwen2.5-0.5B-Instruct by default) with no API calls. Uses few-shot
  prompting. GPU recommended for reasonable speed.

### Comparison

- **compare_llms.py** -- Run all available LLM providers (OpenAI, Anthropic)
  on the same task and print a comparison table with reward, steps, success
  rate, and estimated cost. Requires at least one API key.

## Running

```bash
# Text agents
uv run python examples/llm/openai_agent.py
uv run python examples/llm/openai_text_agent.py
uv run python examples/llm/openai_cot_agent.py
uv run python examples/llm/anthropic_text_agent.py

# Vision agents
uv run python examples/llm/openai_vision_agent.py
uv run python examples/llm/anthropic_vision_agent.py

# Local model (no API key needed, GPU recommended)
uv run python examples/llm/huggingface_local_agent.py

# Compare providers
uv run python examples/llm/compare_llms.py
```

## Cost Notes

Each API-based agent makes one API call per environment step. Typical costs:
- GPT-4o-mini text: ~$0.01-0.05 per episode
- GPT-4o vision: ~$0.05-0.20 per episode (image tokens are larger)
- Claude Sonnet 4 text: ~$0.01-0.05 per episode
- Local HuggingFace model: $0 (runs on your hardware)
