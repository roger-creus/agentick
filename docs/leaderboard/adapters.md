# Adapter Configuration Guide

Complete reference for configuring all six agent adapter types for leaderboard evaluation.

## Overview

Adapters are how the leaderboard loads and runs different types of agents. Each adapter type handles a different deployment pattern.

| Adapter | Best For | Dependencies | Setup Time |
|---------|----------|--------------|-----------|
| **API** | Cloud LLMs (OpenAI, Anthropic, Google) | requests, API key | 5 min |
| **HuggingFace** | Open-source models on HF Hub | transformers, torch | 10 min |
| **LocalWeights** | Custom PyTorch/safetensors models | torch, weights file | 10 min |
| **Code** | Custom Python agents | Custom code | 10 min |
| **Docker** | Containerized agent servers | Docker daemon | 20 min |
| **GitRepo** | Public Git repositories | git, repo URL | 15 min |

## API Adapter

For evaluating agents via cloud API endpoints (OpenAI, Anthropic, Google, custom).

### Supported Providers

- **OpenAI**: GPT-4, GPT-4o, GPT-4 Turbo, GPT-3.5-Turbo
- **Anthropic**: Claude Opus, Claude Sonnet, Claude Haiku
- **Google**: Gemini Pro, Gemini 2.0
- **Custom**: Any HTTP endpoint compatible with specified format

### Configuration

```yaml
agent_type: "api"
observation_mode: "language"  # or "rgb_array" for vision models

config:
  provider: "openai"           # or "anthropic", "gemini", "custom"
  model: "gpt-4o"              # Model identifier
  api_key_env: "OPENAI_API_KEY" # Env var with API key (preferred)
  # OR
  api_key: "sk-..."            # Direct API key (not recommended)

  max_tokens: 100              # Max tokens in response
  temperature: 0.0             # Sampling temperature (0 = deterministic)
  max_retries: 3               # Retries on failure
  timeout: 30.0                # Request timeout in seconds
  log_calls: true              # Log all API calls
```

### Provider-Specific Examples

#### OpenAI (GPT-4o)

```yaml
agent_name: "GPT-4o-Text-v1"
author: "OpenAI"
description: "GPT-4o with text observation mode"
agent_type: "api"
observation_mode: "language"

config:
  provider: "openai"
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"
  max_tokens: 100
  temperature: 0.0

suites:
  - "agentick-quick-v1"

hardware: "API"
estimated_cost: "$0.01-0.05 per task"
```

**Setup**:
```bash
pip install agentick requests

# Set API key
export OPENAI_API_KEY="sk-proj-..."

# Run
agentick evaluate --submission gpt4o.yaml --suite agentick-quick-v1
```

#### OpenAI (GPT-4o with Vision)

```yaml
agent_name: "GPT-4o-Vision-v1"
author: "OpenAI"
description: "GPT-4o with vision capabilities for RGB observations"
agent_type: "api"
observation_mode: "rgb_array"  # <-- Using vision

config:
  provider: "openai"
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"
  max_tokens: 100
  temperature: 0.0
  # Will encode RGB observations as base64 images in request

suites:
  - "agentick-quick-v1"

hardware: "API"
estimated_cost: "$0.02-0.10 per task"  # Vision is more expensive
```

#### Anthropic (Claude Sonnet)

```yaml
agent_name: "Claude-Sonnet-4-Text-v1"
author: "Anthropic"
description: "Claude Sonnet 4 with text observations"
agent_type: "api"
observation_mode: "language"

config:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  api_key_env: "ANTHROPIC_API_KEY"
  max_tokens: 100
  temperature: 0.0

suites:
  - "agentick-quick-v1"
  - "agentick-core-v1"

hardware: "API"
estimated_cost: "$0.01-0.05 per task"
```

**Setup**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
agentick evaluate --submission claude_sonnet.yaml --suite agentick-quick-v1
```

#### Google Gemini

```yaml
agent_name: "Gemini-2.0-Flash-v1"
author: "Google"
description: "Gemini 2.0 Flash with text observations"
agent_type: "api"
observation_mode: "language"

config:
  provider: "gemini"
  model: "gemini-2.0-flash"
  api_key_env: "GOOGLE_API_KEY"
  max_tokens: 100
  temperature: 0.0

suites:
  - "agentick-quick-v1"

hardware: "API"
estimated_cost: "$0.005-0.02 per task"
```

**Setup**:
```bash
export GOOGLE_API_KEY="AIzaSy..."
agentick evaluate --submission gemini.yaml --suite agentick-quick-v1
```

#### Custom HTTP Endpoint

```yaml
agent_name: "CustomLLM-Endpoint-v1"
author: "Your Lab"
description: "Custom LLM accessible via HTTP endpoint"
agent_type: "api"
observation_mode: "language"

config:
  provider: "custom"
  model: "custom-model"
  endpoint: "https://your-server.com/api/predict"
  api_key_env: "CUSTOM_API_KEY"
  max_tokens: 100
  temperature: 0.0

  # Optional: custom headers
  headers:
    "Authorization": "Bearer ${CUSTOM_API_KEY}"
    "X-Custom-Header": "value"

suites:
  - "agentick-quick-v1"

hardware: "Custom server"
estimated_cost: "Varies"
```

**Server Requirements**:
```python
# Your server should handle POST requests like:
# POST /api/predict
# {
#   "prompt": "Task: ... Valid actions: ... Select action:",
#   "max_tokens": 100,
#   "temperature": 0.0
# }
#
# Response format:
# {
#   "text": "The best action is: 2"
# }
# OR
# {
#   "response": "The best action is: 2"
# }
# OR
# {
#   "output": "The best action is: 2"
# }
```

### API-Specific Configuration Options

```yaml
config:
  provider: "openai"
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"

  # Common options
  max_tokens: 100
  temperature: 0.0
  max_retries: 3
  timeout: 30.0
  log_calls: true

  # Provider-specific options (optional)
  top_p: 0.9                    # OpenAI
  frequency_penalty: 0.0        # OpenAI
  presence_penalty: 0.0         # OpenAI
  system_prompt: "You are..."   # Custom endpoint
```

### Cost Management

API costs can add up. To estimate:

```bash
# Quick suite: 5 tasks × 10 seeds × ~50-100 tokens = 2500-5000 tokens
# Rough cost: GPT-4o: $0.05-0.10 | Claude: $0.03-0.05 | Gemini: $0.01-0.02

# Core suite: 27 tasks × 50 seeds × ~50 tokens = 67,500 tokens
# Rough cost: GPT-4o: $0.50-1.00 | Claude: $0.30-0.50 | Gemini: $0.10-0.15

# Full suite: 38 tasks × 50 seeds × ~50 tokens = 95,000 tokens
# Rough cost: GPT-4o: $1.00-2.00 | Claude: $0.50-1.00 | Gemini: $0.15-0.30
```

**Cost Reduction Strategies**:

1. Use cheaper models first (Gemini → Claude Haiku → GPT-4o mini)
2. Test on Quick suite before Full suite
3. Set `max_tokens` lower if possible
4. Use cheaper provider-specific endpoints
5. Batch similar requests

### Troubleshooting

**"API key not found"**
```bash
# Verify env var
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-..."
```

**"Rate limited (429)"**
Adapter auto-retries with exponential backoff. If still failing:
- Increase `timeout` and `max_retries`
- Space out evaluation over time
- Use cheaper models for testing

**"Response parsing error"**
Check the LLM is outputting valid actions:
```python
# Valid: "1", "Action: 2", "Best action is 3"
# Invalid: "zero", "left", "multiple actions"
```

---

## HuggingFace Adapter

For evaluating open-source models from HuggingFace Hub.

### Supported Models

- **LLMs**: Llama 3.1, Mistral, Qwen, Phi, etc.
- **Vision-Language**: LLaVA, Qwen-VL, etc.
- **Specialized**: Any causal LM or VLM on HF Hub

### Configuration

```yaml
agent_type: "huggingface"
observation_mode: "language"  # or "rgb_array" for VLMs

config:
  model_id: "meta-llama/Llama-3.1-8B"  # HF Hub model ID

  # Device and dtype
  device: "auto"               # auto, cuda:0, cpu
  dtype: "float16"             # float32, float16, bfloat16
  quantization: null           # null, "4bit", "8bit"

  # Generation
  max_new_tokens: 50
  temperature: 0.0

  # Optional: additional kwargs for model loading
  trust_remote_code: true
  cache_dir: "/path/to/cache"
```

### Configuration Examples

#### Llama 3.1 8B

```yaml
agent_name: "Llama-3.1-8B-v1"
author: "Meta"
description: "Llama 3.1 8B with text observations"
agent_type: "huggingface"
observation_mode: "language"

config:
  model_id: "meta-llama/Llama-3.1-8B"
  device: "auto"
  dtype: "float16"
  quantization: null
  max_new_tokens: 50
  temperature: 0.0

suites:
  - "agentick-quick-v1"
  - "agentick-core-v1"

hardware: "1x RTX 4090 24GB"
estimated_cost: "free (local)"
training_data: "None (zero-shot)"
```

**Setup**:
```bash
pip install transformers torch

# First run will download ~16GB model
# Ensure sufficient disk space
agentick evaluate --submission llama.yaml --suite agentick-quick-v1
```

#### Mistral 7B with Quantization

```yaml
agent_name: "Mistral-7B-4bit-v1"
author: "Mistral"
description: "Mistral 7B with 4-bit quantization for VRAM efficiency"
agent_type: "huggingface"
observation_mode: "language"

config:
  model_id: "mistralai/Mistral-7B-Instruct-v0.2"
  device: "auto"
  dtype: "float16"
  quantization: "4bit"          # 4-bit quantization
  max_new_tokens: 50
  temperature: 0.0

  # Quantization options
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"

suites:
  - "agentick-quick-v1"
  - "agentick-full-v1"

hardware: "1x RTX 3090 24GB"
estimated_cost: "free (local)"
```

#### Qwen VLM (Vision-Language)

```yaml
agent_name: "Qwen2-VL-7B-v1"
author: "Alibaba"
description: "Qwen 2 Vision-Language Model with RGB observations"
agent_type: "huggingface"
observation_mode: "rgb_array"  # Using vision

config:
  model_id: "Qwen/Qwen2-VL-7B-Instruct"
  device: "auto"
  dtype: "float16"
  quantization: null
  max_new_tokens: 50
  temperature: 0.0

  trust_remote_code: true  # Some models need this

suites:
  - "agentick-quick-v1"
  - "agentick-multimodal-v1"

hardware: "1x RTX 4090 24GB"
estimated_cost: "free (local)"
```

### Memory-Efficient Configurations

**For 8GB VRAM** (e.g., RTX 3080):
```yaml
config:
  model_id: "meta-llama/Llama-3.1-8B"
  device: "auto"
  dtype: "float16"
  quantization: "4bit"
  max_new_tokens: 50
```

**For 4GB VRAM** (e.g., RTX 2060):
```yaml
config:
  model_id: "mistralai/Mistral-7B-v0.1"
  device: "auto"
  dtype: "float16"
  quantization: "8bit"
  max_new_tokens: 25  # Smaller max_new_tokens
```

**For CPU Only**:
```yaml
config:
  model_id: "distilbert-base-uncased"  # Much smaller model
  device: "cpu"
  dtype: "float32"
  quantization: null
  max_new_tokens: 20
```

### Troubleshooting

**"Model not found"**
```bash
# Check model exists
huggingface-cli list-models --filter "meta-llama/Llama-3.1-8B"

# Check you have permission (HF token for gated models)
huggingface-cli login
```

**"Out of memory"**
```yaml
config:
  dtype: "float16"        # Use 16-bit
  quantization: "4bit"    # Use 4-bit quantization
  max_new_tokens: 25      # Reduce output size
```

**"Model download fails"**
```bash
# Pre-download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B')"

# Use custom cache
export HF_HOME=/path/with/space/
```

---

## LocalWeights Adapter

For evaluating custom models loaded from .pt or .safetensors files.

### Configuration

```yaml
agent_type: "local_weights"
observation_mode: "rgb_array"  # or "state_dict", etc.

config:
  weights_path: "./models/agent.pt"     # Path to weights file
  model_class: "agents.AgentCNN"        # Importable class path
  device: "cuda"                        # cuda, cpu

  # Additional kwargs passed to model __init__()
  hidden_dims: [256, 256]
  input_shape: [3, 64, 64]
```

### Configuration Examples

#### CNN Policy for Images

```yaml
agent_name: "CustomCNN-Policy-v1"
author: "Your Lab"
description: "CNN policy trained on Agentick oracle demonstrations"
agent_type: "local_weights"
observation_mode: "rgb_array"

config:
  weights_path: "/home/user/models/agent_cnn.pt"
  model_class: "agents.AgentCNN"
  device: "cuda:0"

  # Passed to AgentCNN.__init__()
  hidden_dims: [256, 256]
  input_channels: 3
  action_space: 4

suites:
  - "agentick-quick-v1"
  - "agentick-full-v1"

hardware: "1x RTX 4090"
estimated_cost: "free (local)"
training_data: "Agentick oracle (1000 episodes)"
training_compute: "8x A100 for 24h"
```

**Model Definition** (`agents.py`):
```python
import torch
import torch.nn as nn

class AgentCNN(nn.Module):
    def __init__(self, hidden_dims=[256, 256], input_channels=3, action_space=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_out = nn.Linear(hidden_dims[1], action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)
```

**Setup**:
```bash
# 1. Create agents.py in import path
cp agents.py ./

# 2. Verify import works
python -c "from agents import AgentCNN; print('OK')"

# 3. Run
agentick evaluate --submission cnn.yaml --suite agentick-quick-v1
```

#### MLP Policy for State Observations

```yaml
agent_name: "CustomMLP-Policy-v1"
author: "Your Lab"
description: "MLP policy for state-based observations"
agent_type: "local_weights"
observation_mode: "state_dict"

config:
  weights_path: "./models/agent_mlp.pt"
  model_class: "agents.AgentMLP"
  device: "cuda"

  input_size: 128
  hidden_sizes: [256, 256]
  output_size: 4

suites:
  - "agentick-quick-v1"

hardware: "CPU + GPU"
estimated_cost: "free"
```

**Model Definition**:
```python
class AgentMLP(nn.Module):
    def __init__(self, input_size=128, hidden_sizes=[256, 256], output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)
```

### Troubleshooting

**"Weights file not found"**
```yaml
# Use absolute path
config:
  weights_path: "/home/user/agentick/models/agent.pt"
```

**"Class not found"**
```bash
# Verify class exists
python -c "from agents import AgentCNN; print(AgentCNN)"

# Check importable path matches YAML
# YAML: agents.AgentCNN
# Python: from agents import AgentCNN
```

**"Incompatible weight shape"**
```python
# Ensure model_class __init__() params match weights_path dimensions

# In YAML:
config:
  model_class: "agents.AgentCNN"
  hidden_dims: [256, 256]

# In agents.py:
def __init__(self, hidden_dims=[256, 256]):
    # hidden_dims must match saved checkpoint
```

---

## Code Adapter

For custom Python agents implementing the `AgentProtocol`.

### Configuration

```yaml
agent_type: "code"
observation_mode: "language"

config:
  script_path: "./my_agent.py"      # Path to Python file
  class_name: "MyAgent"             # Class to instantiate

  # Optional: kwargs passed to class __init__()
  param1: "value1"
```

### Configuration Examples

#### Simple Rule-Based Agent

```yaml
agent_name: "RuleBasedNavigator-v1"
author: "Your Name"
description: "Hand-crafted rule-based navigation agent"
agent_type: "code"
observation_mode: "language"

config:
  script_path: "./rule_agent.py"
  class_name: "RuleBasedAgent"

suites:
  - "agentick-quick-v1"
  - "agentick-navigation-v1"

hardware: "CPU"
estimated_cost: "free"
training_data: "Hand-crafted rules"
training_compute: "N/A"
```

**Agent Implementation** (`rule_agent.py`):
```python
from agentick.leaderboard.adapters.protocol import Agent

class RuleBasedAgent(Agent):
    def __init__(self):
        super().__init__(name="RuleBasedNavigator")

    def act(self, observation, info):
        """
        Simple navigation rules.

        Args:
            observation: Observation string/dict
            info: Dict with 'valid_actions', 'task_name', etc.

        Returns:
            Action index (int)
        """
        valid_actions = info.get("valid_actions", [])
        if not valid_actions:
            return 0

        # Simple heuristic: prefer forward, then turn right
        action_map = {0: "forward", 1: "right", 2: "left", 3: "backward"}

        for action in [0, 1, 2, 3]:
            if action in valid_actions:
                return action

        return valid_actions[0]

    def reset(self):
        """Reset state between episodes."""
        pass
```

#### Learning-Based Agent

```yaml
agent_name: "PPOAgent-Trained-v1"
author: "Your Lab"
description: "PPO agent trained on Agentick oracle demonstrations"
agent_type: "code"
observation_mode: "language"

config:
  script_path: "./ppo_agent.py"
  class_name: "PPOAgent"
  checkpoint_path: "./models/ppo_checkpoint.pkl"
  device: "cuda"

suites:
  - "agentick-quick-v1"
  - "agentick-full-v1"

hardware: "1x RTX 4090"
estimated_cost: "free"
training_data: "Agentick oracle (10K episodes)"
training_compute: "8x A100 for 48h"
```

**Agent Implementation** (`ppo_agent.py`):
```python
from agentick.leaderboard.adapters.protocol import Agent
import torch
import pickle

class PPOAgent(Agent):
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__(name="PPOAgent")
        self.device = device

        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Reconstruct model
        self.policy = checkpoint['policy'].to(device)
        self.policy.eval()

    def act(self, observation, info):
        """Get action from learned policy."""
        valid_actions = info.get("valid_actions", [])

        # Convert observation to tensor
        obs_tensor = self._process_observation(observation)
        obs_tensor = obs_tensor.to(self.device)

        # Get action
        with torch.no_grad():
            action_logits = self.policy(obs_tensor)

        action = action_logits.argmax(dim=-1).item()

        # Validate
        if action not in valid_actions:
            action = valid_actions[0]

        return action

    def _process_observation(self, obs):
        """Convert observation to tensor."""
        # Implementation depends on observation format
        pass

    def reset(self):
        """Reset state."""
        pass
```

### AgentProtocol Requirements

Your agent class must implement:

```python
class MyAgent:
    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action given observation.

        Args:
            observation: Environment observation (format depends on obs_mode)
            info: Dict with:
                - valid_actions: List of valid action indices
                - task_name: Name of current task
                - difficulty: Difficulty level
                - step: Current step number

        Returns:
            Action index (integer)
        """
        ...

    def reset(self) -> None:
        """Reset agent state for new episode."""
        ...

    @property
    def name(self) -> str:
        """Return agent name for identification."""
        ...
```

### Troubleshooting

**"Script file not found"**
```bash
# Use absolute path or run from correct directory
agentick evaluate --submission my_agent.yaml --cwd ./
```

**"Class not found"**
```bash
# Verify class exists in file
python -c "from my_agent import MyAgent; print(MyAgent)"

# Check class_name matches YAML
# YAML: class_name: "MyAgent"
# File: class MyAgent: ...
```

---

## Docker Adapter

For containerized agent servers exposing HTTP APIs.

### Configuration

```yaml
agent_type: "docker"
observation_mode: "language"

config:
  image: "myregistry/myagent:latest"  # Docker image
  port: 8080                           # Port to expose
  endpoint: "/predict"                 # Prediction endpoint

  # Optional: docker run arguments
  environment:
    - "MODEL_PATH=/models/agent.ckpt"
  volumes:
    - "/data:/data"
  devices:
    - "/dev/nvidia.0"                 # GPU access
```

### Configuration Examples

#### Simple Agent Server

```yaml
agent_name: "DockerAgent-v1"
author: "Your Lab"
description: "Agent running in Docker container"
agent_type: "docker"
observation_mode: "language"

config:
  image: "myagent:latest"
  port: 8080
  endpoint: "/predict"

suites:
  - "agentick-quick-v1"

hardware: "Docker with GPU"
estimated_cost: "free (local)"
```

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy agent code
COPY agent.py .
COPY models/ models/

# Expose port
EXPOSE 8080

# Start server
CMD ["python", "server.py"]
```

**Server Implementation** (`server.py`):
```python
from flask import Flask, request, jsonify
from agent import MyAgent

app = Flask(__name__)
agent = MyAgent(checkpoint_path="/app/models/agent.pt")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    observation = data.get('observation')
    info = data.get('info', {})

    # Get action
    action = agent.act(observation, info)

    return jsonify({"action": int(action)}), 200

@app.route('/reset', methods=['POST'])
def reset():
    agent.reset()
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

#### Multi-GPU Agent Server

```yaml
agent_name: "LargeModelServer-v1"
author: "Your Lab"
description: "Large agent running on multiple GPUs"
agent_type: "docker"
observation_mode: "rgb_array"

config:
  image: "large-agent:latest"
  port: 8080
  endpoint: "/predict"

  # GPU access
  devices:
    - "/dev/nvidia.0"
    - "/dev/nvidia.1"

  # Environment
  environment:
    - "CUDA_VISIBLE_DEVICES=0,1"
    - "MODEL_PATH=/models/agent_70b"

suites:
  - "agentick-quick-v1"

hardware: "2x A100 80GB"
estimated_cost: "free (local)"
```

### Container Requirements

Your Docker image must expose:

1. **Health Check Endpoint**
   ```
   GET /health
   Returns: 200 if ready, 5xx if not
   ```

2. **Prediction Endpoint**
   ```
   POST /predict
   Request: {
     "observation": [...],  # Raw observation
     "info": {...}          # Info dict
   }
   Response: {
     "action": 2  # Action index
   }
   ```

3. **Optional: Reset Endpoint**
   ```
   POST /reset
   Response: {"status": "ok"}
   ```

### Building and Testing

```bash
# 1. Build image
docker build -t myagent:latest .

# 2. Test locally
docker run -p 8080:8080 myagent:latest

# 3. Test endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [1,2,3,4],
    "info": {"valid_actions": [0,1,2,3]}
  }'

# 4. Push to registry (if using remote)
docker tag myagent:latest myregistry/myagent:latest
docker push myregistry/myagent:latest
```

### Troubleshooting

**"Docker daemon not running"**
```bash
sudo systemctl start docker
# or on macOS
open -a Docker
```

**"Image not found"**
```bash
# Build locally first
docker build -t myagent:latest .

# Or pull from registry
docker pull myregistry/myagent:latest
```

**"Port already in use"**
```yaml
config:
  port: 8081  # Use different port
```

---

## GitRepo Adapter

For loading agents from public Git repositories.

### Configuration

```yaml
agent_type: "git_repo"
observation_mode: "language"

config:
  url: "https://github.com/yourname/repo.git"  # Repo URL
  branch: "main"                                # Branch
  setup_cmd: "pip install -e ."                 # Setup command
  script_path: "agent.py"                       # Path within repo
  class_name: "Agent"                           # Class in script
```

### Configuration Examples

#### Public Research Repo

```yaml
agent_name: "PublicResearchAgent-v1"
author: "Research Team"
description: "Agent from public repository"
url: "https://github.com/researchteam/agentick-agent"
agent_type: "git_repo"
observation_mode: "language"

config:
  url: "https://github.com/researchteam/agentick-agent.git"
  branch: "main"
  setup_cmd: "pip install -e ."
  script_path: "agent.py"
  class_name: "Agent"

suites:
  - "agentick-quick-v1"
  - "agentick-full-v1"

hardware: "Varies (see repo)"
estimated_cost: "Varies"
training_data: "See repository"
```

#### Specific Branch/Commit

```yaml
agent_name: "AgentFromBranch-v1"
author: "Your Lab"
description: "Agent from specific branch with custom setup"
agent_type: "git_repo"
observation_mode: "state_dict"

config:
  url: "https://github.com/yourlab/agent.git"
  branch: "v2.0-release"
  setup_cmd: |
    pip install torch transformers
    python scripts/download_weights.py
    pip install -e .
  script_path: "src/agents/main_agent.py"
  class_name: "MainAgent"

suites:
  - "agentick-quick-v1"

hardware: "1x A100"
estimated_cost: "free"
```

### Repository Requirements

Your Git repo should have:

1. **setup.py** or **requirements.txt** for dependencies
2. **Agent class file** implementing `AgentProtocol`
3. **setup_cmd** that makes dependencies available
4. **Public URL** (not private)

Example structure:
```
repo/
├── setup.py
├── requirements.txt
├── agent.py              # Contains Agent class
├── models/
│   └── checkpoint.pt
└── README.md
```

### Troubleshooting

**"Failed to clone repository"**
```bash
# Verify URL and access
git clone https://github.com/yourname/repo.git /tmp/test

# If private, provide credentials
git config --global credential.helper store
```

**"Setup command failed"**
```bash
# Test setup manually
git clone https://github.com/yourname/repo.git /tmp/test
cd /tmp/test
pip install -e .  # or your setup_cmd
```

**"Class not found"**
```bash
# Verify file and class exist in repo
cd /tmp/repo
python -c "from agent import Agent; print(Agent)"
```

---

## Security Considerations

### API Keys

- **Never commit** API keys to version control
- **Always use** environment variables (`api_key_env`)
- **Use different keys** for different environments (dev/prod)
- **Rotate keys** periodically

**Good**:
```yaml
config:
  api_key_env: "OPENAI_API_KEY"
```

**Bad**:
```yaml
config:
  api_key: "sk-..."  # DON'T DO THIS
```

### Docker Sandboxing

Docker containers are partially isolated but not fully sandboxed:

```yaml
config:
  image: "agent:latest"

  # Limit resources
  mem_limit: "4g"           # 4GB RAM limit
  cpus: 2                   # 2 CPU limit

  # Don't expose sensitive mounts
  # ❌ volumes: ["/root/.ssh:/root/.ssh"]  # Security risk
  # ✓ volumes: ["/models:/models"]         # Model cache only

  # Don't expose privileged mode
  # ❌ privileged: true                    # Don't do this
```

### Code Execution

When using Code or GitRepo adapters, code is executed in your environment:

- Review code before submission
- Use virtual environments
- Don't use untrusted repositories
- Check dependencies are reputable

### Resource Limits

Set timeouts and resource limits:

```yaml
# API
config:
  timeout: 30.0              # 30 second timeout
  max_retries: 3

# Docker
config:
  mem_limit: "4g"            # RAM limit
  cpus: 4                    # CPU limit

# HuggingFace
config:
  quantization: "4bit"       # Reduce memory usage
```

---

## Comparing Adapters

| Aspect | API | HF | Local | Code | Docker | Git |
|--------|-----|----|----|------|--------|-----|
| **Ease** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Cost** | $$$ | Free | Free | Free | Free | Free |
| **Latency** | High | Low | Low | Depends | Medium | Low |
| **Reproducibility** | Good | Excellent | Excellent | Good | Good | Excellent |
| **Customization** | Low | Low | High | High | High | High |
| **Best For** | Cloud LLMs | Open-source | Custom models | Logic/rules | Complex | Research |

---

## Next Steps

- See [Submission Guide](submitting.md) for step-by-step examples
- Check [Suite Documentation](suites.md) for evaluation details
- Visit [Overview](overview.md) for general information

---

Last updated: 2026-02-12 | Version: 1.0
