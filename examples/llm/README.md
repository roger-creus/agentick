# LLM Agents

Runnable examples for language-model agents.

## Prerequisites

```bash
uv sync --extra llm
```

Set API keys as environment variables or in a local `.env` file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Scripts

| Script | Description |
|---|---|
| `openai_text_agent.py` | OpenAI text agent using the Agentick harness and leaderboard adapter path. |
| `huggingface_local_agent.py` | Local HuggingFace text model with no API calls. GPU recommended. |

## Running

```bash
uv run python examples/llm/openai_text_agent.py
uv run python examples/llm/huggingface_local_agent.py
```

For Gemini, Anthropic, vLLM, and vision-language backends, use the config-driven
agent system documented in `docs/agents/llm_agents.md`.
