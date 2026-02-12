# Installation

Complete guide to installing Agentick and its optional dependencies.

## System Requirements

- **Python**: 3.11, 3.12, or 3.13
- **OS**: Linux, macOS, or Windows
- **RAM**: 2GB minimum (4GB recommended for RL training)
- **GPU**: Optional (CUDA 11.8+ for GPU acceleration)

## Quick Installation

### Using uv (Recommended)

The fastest way to install Agentick is with the `uv` package manager:

```bash
uv add agentick
```

Or to get started in a project immediately:

```bash
uv run --with agentick python -c "import agentick; print(agentick.__version__)"
```

### Using pip

If you prefer pip:

```bash
pip install agentick
```

Or for the latest development version:

```bash
pip install --upgrade agentick
```

## Installation from Source

For development or to use the latest features:

```bash
git clone https://github.com/anthropics/agentick.git
cd agentick
uv sync  # Install all dependencies
```

Or with pip:

```bash
git clone https://github.com/anthropics/agentick.git
cd agentick
pip install -e .
```

## Optional Dependencies

Agentick is modular. Install only what you need:

### Reinforcement Learning (RL)

For training agents with PyTorch, PPO, DQN, and vectorized environments:

```bash
uv add agentick[rl]
# or
pip install agentick[rl]
```

**Includes:**
- `torch>=2.0`: Deep learning framework
- `wandb`: Experiment tracking and visualization
- `gymnasium[classic-control]`: Environment utilities

**Use cases:**
- Training agents with PPO or DQN
- Running vectorized parallel environments
- Monitoring training with Weights & Biases

### Large Language Models (LLMs)

For evaluating agents with API-based LLMs:

```bash
uv add agentick[llm]
# or
pip install agentick[llm]
```

**Includes:**
- `openai`: OpenAI API client (ChatGPT, GPT-4)
- `anthropic`: Anthropic API client (Claude)
- `google-genai`: Google AI API client (Gemini)
- `tinker`: LLM framework utilities

**Use cases:**
- Running agents with GPT-4 or Claude
- Evaluating zero-shot LLM performance
- Using LLMs as decision-makers

### Local LLMs

For running open-source language models locally without API costs:

```bash
uv add agentick[local-llm]
# or
pip install agentick[local-llm]
```

**Includes:**
- `transformers`: Hugging Face model loading
- `torch>=2.0`: Deep learning framework
- `vllm`: High-performance LLM inference engine

**Use cases:**
- Running Llama, Mistral, or other open models
- Offline inference (no API calls needed)
- Fine-tuned models on local hardware

### LLM Training

For fine-tuning language models on agent trajectories:

```bash
uv add agentick[train-llm]
# or
pip install agentick[train-llm]
```

**Includes:**
- `trl`: Transformer Reinforcement Learning library
- `accelerate`: Distributed training utilities
- `peft`: Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
- `torch>=2.0`: Deep learning framework
- `tinker`: LLM framework utilities

**Use cases:**
- Supervised fine-tuning (SFT) on demonstrations
- Direct Preference Optimization (DPO)
- Collecting and preparing training data

### Visualization

For plotting results, generating comparison charts, and recording videos:

```bash
uv add agentick[viz]
# or
pip install agentick[viz]
```

**Includes:**
- `matplotlib>=3.10.8`: Static plotting library
- `seaborn>=0.13.2`: Statistical visualization
- `plotly>=6.5.2`: Interactive plots
- `imageio-ffmpeg>=0.6.0`: Video recording and generation

**Use cases:**
- Generating learning curves and comparison plots
- Creating publication-ready figures
- Recording episode video walkthroughs
- Interactive result visualization

### Documentation

For building and modifying this documentation:

```bash
uv add agentick[docs]
# or
pip install agentick[docs]
```

**Includes:**
- `mkdocs>=1.6.1`: Static site generator
- `mkdocs-material>=9.7.1`: Material theme
- `mkdocstrings[python]>=1.0.3`: Python docstring rendering

**Use cases:**
- Building documentation locally
- Contributing to docs
- Generating API references

### Everything

To install all optional dependencies at once:

```bash
uv add agentick[all]
# or
pip install agentick[all]
```

This gives you everything for RL, LLMs, visualization, and documentation.

## Verification

After installation, verify everything works:

```python
# Python REPL
import agentick

# Check version
print(f"Agentick version: {agentick.__version__}")

# List available tasks
tasks = agentick.list_tasks()
print(f"Available tasks: {len(tasks)}")

# Create and test an environment
env = agentick.make("GoToGoal-v0", difficulty="easy")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("✓ Installation successful!")
```

Expected output:
```
Agentick version: 0.1.0
Available tasks: 40
✓ Installation successful!
```

## Troubleshooting

### Import Error: "No module named 'agentick'"

**Solution**: Ensure Agentick is installed:
```bash
pip install --upgrade agentick
```

Or verify installation:
```bash
python -c "import agentick; print(agentick.__version__)"
```

### CUDA/GPU Issues

If you want GPU support for RL training:

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

If False, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then reinstall:
```bash
pip install --upgrade agentick[rl]
```

### Display/Rendering Issues (Linux)

If you get pygame display errors on headless systems:

```bash
# Install pygame without display support
pip install pygame --no-binary pygame

# Or set display environment variable
export SDL_VIDEODRIVER=dummy
```

### Pygame Import Error

Some systems need SDL libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# macOS with homebrew
brew install sdl2

# Then reinstall
pip install --upgrade --force-reinstall pygame
```

### Version Conflicts

If you have dependency conflicts with other packages:

```bash
# Create a fresh environment (recommended)
python -m venv agentick_env
source agentick_env/bin/activate  # On Windows: agentick_env\Scripts\activate
pip install agentick[all]

# Or use uv for automatic conflict resolution
uv venv agentick_env
source agentick_env/bin/activate
uv add agentick[all]
```

### ImportError with Optional Dependencies

If you get import errors for optional features:

```python
# Wrong (will fail if deps not installed)
from agentick.interfaces import RLInterface  # Requires [rl]

# Right (with error handling)
try:
    from agentick.interfaces import RLInterface
except ImportError:
    print("Install RL dependencies: pip install agentick[rl]")
```

Install the missing dependency group:
```bash
# For RL features
pip install agentick[rl]

# For LLM features
pip install agentick[llm]

# For all features
pip install agentick[all]
```

## Platform-Specific Notes

### macOS

May need Xcode command line tools:
```bash
xcode-select --install
```

### Windows

If you have issues with building packages, install Visual C++ build tools:
1. Download Microsoft C++ Build Tools
2. Install with "Desktop development with C++" workload
3. Reinstall: `pip install --upgrade agentick`

### Docker

To run Agentick in a Docker container:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install agentick[all]

WORKDIR /workspace
CMD ["python"]
```

Build and run:
```bash
docker build -t agentick .
docker run -it agentick
```

## Updating Agentick

To upgrade to the latest version:

```bash
# With uv
uv add agentick --upgrade

# With pip
pip install --upgrade agentick

# Upgrade all dependencies
pip install --upgrade agentick[all]
```

## Development Installation

If you're contributing to Agentick:

```bash
git clone https://github.com/anthropics/agentick.git
cd agentick

# Install with all extras + dev dependencies
uv sync --all-extras

# Or with pip
pip install -e ".[all]"
pip install pytest pytest-cov ruff mypy pre-commit
```

Then verify:
```bash
# Run tests
pytest tests/ -v

# Check code quality
ruff check agentick/
ruff format agentick/

# Type checking
mypy agentick/ --ignore-missing-imports
```

## Next Steps

Once installed, check out:

1. **[Quickstart](quickstart.md)** - Run your first environment in 5 minutes
2. **[First Experiment](first_experiment.md)** - Complete guide to running experiments
3. **[Concepts](../concepts/architecture.md)** - Understand the system architecture

## Getting Help

- **Questions?** Check the [FAQ](../faq.md)
- **Found a bug?** Open an issue on [GitHub](https://github.com/anthropics/agentick/issues)
- **Need examples?** See the [examples/](https://github.com/anthropics/agentick/tree/main/examples) directory
