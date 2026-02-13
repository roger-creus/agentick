# VLM Agents

Evaluate Vision-Language Models on Agentick using OpenAI, Anthropic, and HuggingFace.

VLM agents process visual observations (RGB pixel arrays) from environments. For complete working examples, see `examples/llm/openai_vision_agent.py` and `examples/llm/anthropic_vision_agent.py`.

## OpenAI (GPT-4o)

See `examples/llm/openai_vision_agent.py` for a complete working example using:
- GPT-4o with vision capabilities
- Base64 image encoding from numpy arrays
- Action parsing from text responses

## Anthropic (Claude Sonnet 4)

See `examples/llm/anthropic_vision_agent.py` for a complete working example using:
- Claude Sonnet 4 with vision
- Image encoding and API formatting
- Action extraction from responses

## Key Patterns

All vision agent examples follow a similar pattern:

1. Create environment with `render_mode="rgb_array"`
2. Convert observations (numpy arrays) to base64-encoded images
3. Send image + text prompt to VLM API
4. Parse action text from response
5. Convert action text to environment action index

For custom prompt engineering and action parsing, see the example implementations.
