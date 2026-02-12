# Leaderboard Submission Guide

## Complete Step-by-Step Submission Process

This guide walks you through submitting an agent to the Agentick Leaderboard, from setup to published results.

## Step 1: Choose Your Agent Type

First, determine how your agent will be deployed. The leaderboard supports six adapter types:

| Type | When to Use | Setup Time | Dependencies |
|------|------------|-----------|--------------|
| **API** | Cloud LLMs (OpenAI, Anthropic, Google) | 5 min | API key |
| **HuggingFace** | Open-source models from HF Hub | 10 min | transformers, torch |
| **Local Weights** | Custom PyTorch/safetensors models | 10 min | torch, weights file |
| **Code** | Custom Python agents | 10 min | Agent implements AgentProtocol |
| **Docker** | Containerized agent servers | 20 min | Docker, HTTP API |
| **Git Repo** | Code from public Git repositories | 15 min | git, setup script |

## Step 2: Create Your Submission YAML

Create a YAML file that defines how to load and run your agent. Use the appropriate template for your agent type:

### API Agent (OpenAI/Anthropic/Google)

```yaml
# my_gpt4_agent.yaml
agent_name: "GPT-4o-Text-v1"
author: "OpenAI"
description: "GPT-4o with natural language observations - zero-shot agent"
url: "https://openai.com/gpt-4"
tags: ["llm", "api", "zero-shot", "multimodal"]
license: "proprietary"
open_weights: false

agent_type: "api"
observation_mode: "language"  # or "rgb_array" if using vision

config:
  provider: "openai"           # or "anthropic", "gemini", "custom"
  model: "gpt-4o"              # Model identifier
  api_key_env: "OPENAI_API_KEY" # Environment variable with API key
  max_tokens: 100
  temperature: 0.0

suites:
  - "agentick-quick-v1"        # Start with quick for testing
  # - "agentick-full-v1"       # Add when ready for full eval

hardware: "API"
estimated_cost: "$10-50 for full suite"
training_data: "None (zero-shot)"
training_compute: "N/A"
```

**Setup**:
```bash
# 1. Install agentick
pip install agentick

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Verify it works (quick test)
agentick evaluate --submission my_gpt4_agent.yaml --suite agentick-quick-v1 --output test_results
```

### HuggingFace Model

```yaml
# my_llama_agent.yaml
agent_name: "Llama-3.1-8B-v1"
author: "Meta"
description: "Llama 3.1 8B causal LM with text observations"
url: "https://huggingface.co/meta-llama/Llama-3.1-8B"
tags: ["llm", "huggingface", "open-weights"]
license: "llama2"
open_weights: true

agent_type: "huggingface"
observation_mode: "language"

config:
  model_id: "meta-llama/Llama-3.1-8B"  # HF Hub model ID
  device: "auto"                        # or "cuda", "cpu"
  dtype: "float16"                      # or "float32", "bfloat16"
  quantization: null                    # or "4bit", "8bit"
  max_new_tokens: 50
  temperature: 0.0

suites:
  - "agentick-quick-v1"
  - "agentick-navigation-v1"

hardware: "1x RTX 4090 24GB"
estimated_cost: "free (local)"
training_data: "None (zero-shot)"
training_compute: "N/A"
```

**Setup**:
```bash
pip install transformers torch

# First run will download the model (check available disk space!)
agentick evaluate --submission my_llama_agent.yaml --suite agentick-quick-v1 --output test_results
```

### Local Weights Model

```yaml
# my_custom_agent.yaml
agent_name: "CustomAgentCNN-v1"
author: "Your Lab"
description: "Custom CNN policy trained on Agentick oracle demonstrations"
url: "https://github.com/yourlab/agentick-agent"
tags: ["custom", "cnn", "trained"]
license: "MIT"
open_weights: true

agent_type: "local_weights"
observation_mode: "rgb_array"  # or "state_dict"

config:
  weights_path: "./models/agent_checkpoint.pt"
  model_class: "agents.AgentCNN"  # Importable path to model class
  device: "cuda"
  hidden_dims: [256, 256]  # Additional model kwargs

suites:
  - "agentick-quick-v1"
  - "agentick-full-v1"

hardware: "1x A100 for training, 1x RTX 4090 for eval"
estimated_cost: "free (local)"
training_data: "Agentick oracle demonstrations (1000 episodes)"
training_compute: "8x A100 for 24 hours"
```

**Setup**:
```bash
# 1. Create agents.py with your model class
cat > agents.py << 'EOF'
import torch
import torch.nn as nn

class AgentCNN(nn.Module):
    def __init__(self, hidden_dims=[256, 256]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_out = nn.Linear(hidden_dims[1], 4)  # 4 actions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)
EOF

# 2. Ensure you can import it
python -c "from agents import AgentCNN; print('OK')"

# 3. Run evaluation
agentick evaluate --submission my_custom_agent.yaml --suite agentick-quick-v1
```

### Custom Python Code

```yaml
# my_custom_logic.yaml
agent_name: "RuleBasedNavigator-v1"
author: "Your Name"
description: "Hand-crafted rule-based navigation agent"
url: "https://github.com/yourname/rule-based-agent"
tags: ["custom", "rule-based", "zero-shot"]
license: "MIT"
open_weights: true

agent_type: "code"
observation_mode: "language"

config:
  script_path: "./my_agent.py"      # Path to Python file
  class_name: "MyAgent"              # Class to instantiate
  # Additional kwargs passed to MyAgent.__init__()

suites:
  - "agentick-quick-v1"

hardware: "CPU only"
estimated_cost: "free"
training_data: "None"
training_compute: "N/A"
```

**Setup**:
```bash
# 1. Create my_agent.py implementing AgentProtocol
cat > my_agent.py << 'EOF'
from agentick.leaderboard.adapters.protocol import Agent

class MyAgent(Agent):
    def __init__(self):
        super().__init__(name="RuleBasedNavigator")

    def act(self, observation, info):
        """Choose action based on observation."""
        valid_actions = info.get("valid_actions", [])
        if not valid_actions:
            return 0
        # Your logic here
        return valid_actions[0]

    def reset(self):
        """Reset internal state for new episode."""
        pass
EOF

# 2. Run evaluation
agentick evaluate --submission my_custom_logic.yaml --suite agentick-quick-v1
```

### Docker Container

```yaml
# my_docker_agent.yaml
agent_name: "ComplexAgent-Docker-v1"
author: "Your Lab"
description: "Complex agent running in Docker container"
url: "https://github.com/yourlab/agent"
tags: ["docker", "custom"]
license: "MIT"
open_weights: false

agent_type: "docker"
observation_mode: "state_dict"

config:
  image: "myregistry/myagent:latest"  # Docker image
  port: 8080                           # Port to expose
  endpoint: "/predict"                 # Prediction endpoint
  # Optional docker run arguments
  environment:
    - "MODEL_PATH=/models/agent.ckpt"
  volumes:
    - "/data:/data"

suites:
  - "agentick-quick-v1"

hardware: "Docker with GPU support"
estimated_cost: "varies by deployment"
training_data: "Custom dataset"
training_compute: "8x GPUs for training"
```

**Setup**:
```bash
# 1. Build and push your Docker image
docker build -t myregistry/myagent:latest .
docker push myregistry/myagent:latest

# 2. Your container should expose HTTP API:
#    POST /predict with {"observation": ..., "info": ...}
#    GET /health for health checks

# 3. Run evaluation (will auto start/stop container)
agentick evaluate --submission my_docker_agent.yaml --suite agentick-quick-v1
```

### Git Repository

```yaml
# my_git_agent.yaml
agent_name: "PublicAgentRepo-v1"
author: "Research Team"
description: "Agent from public GitHub repository"
url: "https://github.com/researchteam/agentick-agent"
tags: ["github", "research"]
license: "apache-2.0"
open_weights: true

agent_type: "git_repo"
observation_mode: "language"

config:
  url: "https://github.com/researchteam/agentick-agent.git"
  branch: "main"                      # Branch to clone
  setup_cmd: "pip install -e ."       # Setup command
  script_path: "agent.py"             # Path within repo
  class_name: "Agent"                 # Class in that file

suites:
  - "agentick-quick-v1"

hardware: "Varies (specified in repo)"
estimated_cost: "Varies"
training_data: "See repository"
training_compute: "See repository"
```

**Setup**:
```bash
# 1. Ensure your repo has:
#    - agent.py with Agent class implementing AgentProtocol
#    - setup.py or requirements.txt for dependencies

# 2. Run evaluation (will clone, setup, and run)
agentick evaluate --submission my_git_agent.yaml --suite agentick-quick-v1
```

## Step 3: Validate Your Submission YAML

Before running full evaluation, validate the YAML structure:

```bash
# Validate YAML syntax and agent config
agentick validate --submission my_agent.yaml

# Expected output if valid:
# ✓ YAML syntax valid
# ✓ Agent config valid
# ✓ Suites exist
# ⚠ Warning: API key environment variable OPENAI_API_KEY not set
```

## Step 4: Run on Quick Suite (Testing)

Always start with the quick suite to verify everything works:

```bash
# Run quick suite (5-10 min, 5 easy tasks)
agentick evaluate \
  --submission my_agent.yaml \
  --suite agentick-quick-v1 \
  --output results_quick \
  --verbose

# Expected output:
# Loading agent...
# Evaluating on agentick-quick-v1 (5 tasks, 10 seeds)
# [████████████████████] GoToGoal-v0
# [████████████████████] MazeNavigation-v0
# [████████████████████] KeyDoorPuzzle-v0
# [████████████████████] SokobanPush-v0
# [████████████████████] PreciseNavigation-v0
#
# ✓ Evaluation complete in 8m 32s
# Results saved to results_quick/evaluation_result.json
```

## Step 5: Verify and Review Results

```bash
# Verify result integrity
agentick verify --result results_quick/evaluation_result.json

# Expected output:
# ✓ Result integrity verified - hash matches
#
# Summary:
# Agent: GPT-4o-Text-v1
# Suite: agentick-quick-v1
# Overall Score: 0.68 (95% CI: 0.65-0.71)
#
# Per-Capability Scores:
#   navigation:    0.72 (95% CI: 0.68-0.76)
#   memory:        0.65 (95% CI: 0.61-0.69)
#   reasoning:     0.68 (95% CI: 0.64-0.72)
#   control:       0.65 (95% CI: 0.61-0.69)
#   skill:         0.69 (95% CI: 0.65-0.73)
```

## Step 6: Run Full Evaluation (When Ready)

Once you've confirmed the quick suite works:

```bash
# Run full benchmark (4-8 hours, 38 tasks, 50 seeds)
agentick evaluate \
  --submission my_agent.yaml \
  --suite agentick-full-v1 \
  --output results_full \
  --save-logs \
  --verify-reproducibility

# With reproducibility check, will run twice to verify results match
# Results are automatically saved with:
# - evaluation_result.json (scores and metadata)
# - run_logs/ (detailed per-task/seed logs)
# - reproducibility_check.json (seed-by-seed comparison)
```

## Step 7: Submit Results to Leaderboard

### Generate Submission Package

```bash
# Package results for submission
agentick package-submission \
  --result results_full/evaluation_result.json \
  --submission my_agent.yaml \
  --metadata results_full/run_logs \
  --output submission_package

# Creates:
# submission_package/
# ├── evaluation_result.json
# ├── submission_spec.yaml
# ├── reproducibility_check.json
# ├── metadata.json
# └── logs.tar.gz
```

### Submit Online (Coming Soon)

Once the leaderboard website launches:

```bash
# Login and submit
agentick submit \
  --email your@email.com \
  --api-key <leaderboard-api-key> \
  submission_package/

# Or submit via web interface at https://agentick-leaderboard.com
```

## Submission Requirements

### 1. Reproducible Code

Your agent must be deterministic (or use fixed random seeds):

```python
# Good: Deterministic agent
class MyAgent(Agent):
    def act(self, observation, info):
        return np.argmax(scores)  # Deterministic

# Bad: Non-deterministic without seed
class MyAgent(Agent):
    def act(self, observation, info):
        return np.random.choice(valid_actions)  # Random each time
```

If your agent uses randomness, set seed at initialization:

```python
class MyAgent(Agent):
    def __init__(self):
        np.random.seed(42)
        self.rng = np.random.default_rng(42)

    def act(self, observation, info):
        return self.rng.choice(valid_actions)
```

### 2. Documented Approach

Clearly explain how your agent works:

```yaml
agent_name: "MyAgent-v1"
description: |
  Vision-Language Model agent using GPT-4o:
  1. Convert observation to natural language description
  2. Create prompt: "Task: {task}. State: {description}. Valid actions: {actions}. Choose best action."
  3. Query GPT-4o with temperature=0.0
  4. Parse action from response (default to first valid action if unparseable)
  5. No training, zero-shot in-context learning
url: "https://arxiv.org/abs/..."  # Link to paper, blog, etc.
```

### 3. Honest Reporting

Be transparent about:

```yaml
hardware: "1x A100 80GB GPU"
estimated_cost: "$50-100 for full suite"
training_data: "1M examples from Anthropic Constitutional AI"
training_compute: "8x H100 for 72 hours"
```

Don't overstate capabilities or hide limitations.

## Submission Template Walkthrough

### Metadata Fields

```yaml
agent_name: "MyAgent-v1"              # Unique identifier (alphanumeric, -, _, .)
author: "Your Organization"           # Author or org name
description: "..."                    # What it does (multiline OK)
url: "https://..."                    # Paper, repo, blog (optional)
tags: ["tag1", "tag2"]               # Searchable tags (optional)
license: "MIT"                        # License type
open_weights: false                   # Are weights publicly available?
```

### Agent Configuration

```yaml
agent_type: "api"                                    # Type of agent
observation_mode: "language"                        # How agent sees environment
config:                                             # Adapter-specific config
  provider: "openai"                               # For API agents
  model: "gpt-4o"
  # ... more config based on agent_type
suites:                                            # Which suites to run
  - "agentick-quick-v1"
  - "agentick-full-v1"
```

### Metadata Fields

```yaml
hardware: "API"                       # Hardware used
estimated_cost: "$10-50"             # Cost estimate
training_data: "None (zero-shot)"    # Training data
training_compute: "N/A"              # Training compute
```

## CLI Commands Reference

```bash
# 1. Validate YAML
agentick validate --submission my_agent.yaml

# 2. Run evaluation
agentick evaluate \
  --submission my_agent.yaml \
  --suite agentick-quick-v1 \
  --output results \
  --verbose \
  --verify-reproducibility

# 3. Verify results
agentick verify --result results/evaluation_result.json

# 4. Package for submission
agentick package-submission \
  --result results/evaluation_result.json \
  --submission my_agent.yaml \
  --output submission_package

# 5. List all suites
agentick list-suites

# 6. Get suite info
agentick suite-info --suite agentick-quick-v1

# 7. Compare results
agentick compare results/evaluation_result.json baseline_result.json
```

## Verification Process

### What Happens After Submission

1. **Automated Checks** (immediate)
   - YAML validation
   - Result integrity hash verification
   - Baseline consistency checks
   - No NaN/invalid values

2. **Reproducibility Verification** (1-2 days)
   - Re-run a subset of tasks to verify results are reproducible
   - Expected variance within confidence intervals
   - Check seed matching across runs

3. **Manual Review** (2-7 days)
   - Verify agent code/setup is sound
   - Check claims in description are accurate
   - Look for any issues or concerns
   - Community can submit feedback

4. **Publication** (1-7 days after review)
   - Results appear on public leaderboard
   - Included in official comparisons
   - Featured in newsletter if top performer
   - Discoverable by tag and capability

### Timeline

```
Day 0:   Submit via web or CLI
Day 0:   Automated checks (pass/fail)
Day 1-2: Reproducibility verification
Day 2-7: Manual review
Day 7:   Publication (if approved)
```

## Common Issues and Troubleshooting

### API Agents

**Issue: "API key not found"**
```bash
# Check environment variable is set
echo $OPENAI_API_KEY

# Set if missing
export OPENAI_API_KEY="sk-..."

# Or add to submission YAML
config:
  api_key: "sk-..."  # Not recommended - use env var instead
```

**Issue: "Rate limited (429)"**
The adapter automatically retries with exponential backoff. If persists:
- Reduce max_tokens or temperature
- Add delays between requests
- Use cheaper models for testing (GPT-4o mini instead of GPT-4o)

**Issue: "Timeout errors"**
```yaml
config:
  timeout: 60.0  # Increase timeout in seconds
  max_retries: 5 # More retry attempts
```

### HuggingFace Models

**Issue: "Model not found"**
```bash
# Verify model exists on HF Hub
huggingface-cli list-models --filter "meta-llama/Llama-3.1-8B"

# Check you have permission (private models need HF token)
huggingface-cli login
```

**Issue: "Out of memory"**
```yaml
config:
  dtype: "float16"        # Use 16-bit instead of 32-bit
  quantization: "4bit"    # Use 4-bit quantization
  device: "cuda:0"        # Specify GPU if multiple available
```

**Issue: "Model download fails"**
```bash
# Set cache directory with more space
export HF_HOME=/path/with/more/space/

# Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B')"
```

### Local Weights

**Issue: "Script file not found: ./models/agent.pt"**
```bash
# Use absolute paths in submission YAML
config:
  weights_path: "/home/user/agentick/models/agent.pt"

# Or run from the directory containing the weights
cd /path/with/weights
agentick evaluate --submission my_agent.yaml
```

**Issue: "Class 'MyAgent' not found"**
```bash
# Test import before submitting
python -c "from agents import MyAgent; print(MyAgent)"

# Check class name matches YAML
# YAML says: class_name: "MyAgent"
# Python file has: class MyAgent(...)
```

### Docker Agents

**Issue: "Docker daemon not running"**
```bash
# Start Docker
sudo systemctl start docker

# Or on macOS
open -a Docker
```

**Issue: "Image not found"**
```bash
# Build and test image first
docker build -t myagent:latest .
docker run -p 8080:8080 myagent:latest

# Test HTTP endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [...], "info": {}}'
```

### Git Repo Agents

**Issue: "Failed to clone repository"**
```bash
# Check URL is correct and accessible
git clone https://github.com/yourname/repo.git

# If private, provide credentials
git config --global credential.helper store
```

**Issue: "Setup command failed"**
```bash
# Test setup manually first
git clone https://github.com/yourname/repo.git /tmp/test
cd /tmp/test
pip install -e .  # Or your setup_cmd

# Check all dependencies install correctly
```

### General Issues

**Issue: "Observation format mismatch"**
Agent expects text but gets images, or vice versa:
```yaml
# Check observation_mode matches your agent
observation_mode: "language"  # Agent expects text descriptions
# Available: ascii, language, language_structured, rgb_array, state_dict
```

**Issue: "Action out of bounds"**
Agent returns invalid action number:
```python
# Always validate actions
valid_actions = info.get("valid_actions", [])
if action not in valid_actions:
    action = valid_actions[0]  # Fallback to first valid action
```

**Issue: "Results don't match SOTA"**
Your agent scores much lower than expected:
1. Verify agent works on quick suite first
2. Check observation format (text vs image)
3. Ensure API/model configuration is correct
4. Test with simpler task like navigation
5. Check for NaN/infinity in scores

## Getting Help

- **Questions**: Check [FAQ](faq.md) (coming soon)
- **Issues**: Report bugs on [GitHub](https://github.com/agentick/leaderboard)
- **Discussions**: Community forum at [agentick-leaderboard.com](https://agentick-leaderboard.com)
- **Direct support**: Email support@agentick-leaderboard.com

---

Last updated: 2026-02-12 | Version: 1.0
