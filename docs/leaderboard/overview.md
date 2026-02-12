# Agentick Leaderboard Overview

## What is the Agentick Leaderboard?

The **Agentick Leaderboard** is an official benchmarking platform for evaluating and comparing agent capabilities across standardized test suites. It provides a transparent, reproducible way to measure agent performance on challenging decision-making tasks spanning multiple domains and difficulty levels.

The leaderboard is designed for:

- **AI researchers** developing new agent architectures and training methods
- **Model providers** (OpenAI, Anthropic, Google, open-source communities) benchmarking their models
- **Engineers** evaluating agent candidates for production deployment
- **Students** learning about agent capabilities and limitations

## How It Works

### The Evaluation Pipeline

1. **Submit Your Agent**: Define your agent using one of our supported adapters (API, HuggingFace, local code, Docker, etc.)

2. **Choose Benchmark Suites**: Select which evaluation suites to run:
   - **Quick Sanity Check** (~5 minutes): Fast validation for development iteration
   - **Category-Specific Suites**: Deep-dive into specific capabilities like navigation, reasoning, or memory
   - **Full Benchmark** (~4-8 hours): Complete evaluation across all 38 official tasks

3. **Run Evaluation**: The leaderboard framework automatically:
   - Instantiates your agent
   - Runs deterministic, locked random seeds for reproducibility
   - Evaluates across multiple difficulty levels
   - Handles different observation modes (text, images, state, etc.)
   - Tracks all API costs and compute metrics

4. **Get Scored**: Results are scored against random and optimal baselines using our standardized methodology:
   - Per-task normalized scores (0-1 range)
   - Per-capability breakdowns (navigation, reasoning, control, etc.)
   - Overall Agentick score with confidence intervals
   - Reproducibility verification

5. **Compare & Publish**: Results appear on the public leaderboard where you can:
   - Compare against SOTA baselines (GPT-4o, Claude, Llama, etc.)
   - View detailed capability breakdowns
   - Share papers, code, and blog posts
   - Track performance trends over time

## Hosted Leaderboard

**Coming soon**: The official leaderboard site will be available at `https://agentick-leaderboard.com`

Currently, you can:
- Run evaluations locally using the CLI
- Compare results programmatically
- Verify submission integrity
- Generate comparison visualizations

## Why Submit Your Agent?

### 1. Compare to SOTA
Benchmark against carefully tuned baselines including:
- GPT-4o, GPT-4 Turbo, GPT-4 (OpenAI)
- Claude Opus, Claude Sonnet, Claude Haiku (Anthropic)
- Gemini Pro, Gemini 2.0 (Google)
- Llama 3.1, Mistral, Qwen models (Open-source)

### 2. Validate Your Research
Use official, reproducible benchmarks for academic papers:
- Immutable task definitions (prevent overfitting)
- Deterministic evaluation seeds
- Integrity verification (hash-based)
- Bootstrap confidence intervals
- Peer-reviewable methodology

### 3. Community Visibility
Gain exposure through:
- Public leaderboard ranking
- Automatic comparison against baselines
- Community discussion and feedback
- Integration with popular platforms (Papers with Code, ArXiv)
- Newsletter features for top submissions

### 4. Production Readiness
Make informed deployment decisions with:
- Detailed capability breakdowns (where does agent excel/struggle?)
- Cost and latency metrics
- Hardware requirements
- Reliability across different observation modes

## Scoring Methodology Overview

### Normalized Scoring

Each task is scored on a **0-1 scale** using:

```
normalized_score = (agent_return - random_baseline) / (optimal_return - random_baseline)
```

Where:
- **agent_return**: Agent's average return on this task
- **random_baseline**: Return of a random action agent
- **optimal_return**: Best possible return (expert demonstration or oracle)

This normalization makes scores comparable across tasks with different reward scales.

### Aggregation

Scores are aggregated hierarchically:

1. **Per-Task**: Mean normalized score across episodes
2. **Per-Capability**: Average of all tasks in that capability (navigation, reasoning, memory, etc.)
3. **Overall Agentick Score**: Equal-weighted average of capability scores

This ensures all capabilities contribute equally to the final score, regardless of task count.

### Confidence Intervals

All scores include **95% bootstrap confidence intervals** to quantify uncertainty:

```
agentick_score: 0.65 (95% CI: 0.62-0.68)
```

This allows meaningful comparison even with small sample sizes.

### Full Documentation

For complete scoring details including:
- Baseline computation methodology
- Normalization alternatives (z-score, min-max)
- Bootstrap resampling procedures
- Per-suite scoring configurations

See [Scoring Details](scoring.md) (coming soon)

## Benchmark Suites at a Glance

| Suite | Tasks | Difficulty | Estimated Time | Use Case |
|-------|-------|-----------|-----------------|----------|
| **agentick-quick-v1** | 5 | Easy | ~5 min | Development iteration |
| **agentick-core-v1** | 27 | Medium | ~2 hours | Standard baseline |
| **agentick-full-v1** | 38 | Medium | ~4-8 hours | Publication/SOTA |
| **agentick-navigation-v1** | 5 | All levels | ~1 hour | Navigation deep-dive |
| **agentick-memory-v1** | 5 | All levels | ~1 hour | Memory deep-dive |
| **agentick-reasoning-v1** | 5 | All levels | ~1 hour | Reasoning deep-dive |
| **agentick-skill-v1** | 5 | All levels | ~1 hour | Skill discovery deep-dive |
| **agentick-control-v1** | 4 | All levels | ~45 min | Control deep-dive |
| **agentick-combinatorial-v1** | 4 | All levels | ~45 min | Combinatorial deep-dive |
| **agentick-worldmodel-v1** | 3 | All levels | ~30 min | World model deep-dive |
| **agentick-adversarial-v1** | 3 | Medium | ~30 min | Robustness evaluation |
| **agentick-meta-v1** | 2 | Medium | ~20 min | Meta-learning evaluation |
| **agentick-multiagent-v1** | 2 | Medium | ~20 min | Multi-agent coordination |
| **agentick-difficulty-v1** | 10 | All levels | ~1.5 hours | Difficulty scaling analysis |
| **agentick-multimodal-v1** | 10 | Medium | ~1.5 hours | Observation mode comparison |

See [Suite Documentation](suites.md) for complete details on each suite including task lists, seeds, and when to use each.

## Submission Types

The leaderboard supports multiple agent implementations:

| Type | Use Case | Example |
|------|----------|---------|
| **API** | Cloud LLMs (OpenAI, Anthropic, Google, custom) | GPT-4o, Claude Sonnet |
| **HuggingFace** | Open-source models from HF Hub | Llama 3.1, Mistral, Qwen |
| **Local Weights** | Pre-trained PyTorch/safetensors models | Fine-tuned checkpoints |
| **Code** | Custom Python agents | Specialized logic, rule-based agents |
| **Docker** | Containerized agent servers | Complex deployment pipelines |
| **Git Repo** | Agent code from public repositories | Public research implementations |

See [Adapter Documentation](adapters.md) for configuration examples for each type.

## Getting Started

### For API-based Agents (fastest)

```bash
# 1. Create a submission YAML
cat > my_agent.yaml << 'EOF'
agent_name: "MyGPT4Agent-v1"
author: "Your Name"
description: "GPT-4o with text observations"
tags: ["llm", "api"]
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
estimated_cost: "$10-50"
training_data: "None (zero-shot)"
EOF

# 2. Run evaluation
agentick evaluate --submission my_agent.yaml --suite agentick-quick-v1 --output results

# 3. Verify results
agentick verify --result results/evaluation_result.json
```

### For Local Models (HuggingFace)

```bash
# 1. Create a submission YAML
cat > my_llama_agent.yaml << 'EOF'
agent_name: "Llama3.1-8B-v1"
author: "Meta"
description: "Llama 3.1 8B with text observations"
agent_type: "huggingface"
observation_mode: "language"

config:
  model_id: "meta-llama/Llama-3.1-8B"
  device: "auto"
  dtype: "float16"
  max_new_tokens: 50

suites:
  - "agentick-quick-v1"

hardware: "1x RTX 4090"
estimated_cost: "free (local)"
training_data: "None (zero-shot)"
EOF

# 2. Run evaluation
agentick evaluate --submission my_llama_agent.yaml --suite agentick-quick-v1 --output results

# 3. Verify results
agentick verify --result results/evaluation_result.json
```

## Next Steps

- **New to the leaderboard?** Start with [Submission Guide](submitting.md)
- **Want to understand benchmark suites?** See [Suite Documentation](suites.md)
- **Need to configure an adapter?** Check [Adapter Documentation](adapters.md)
- **Running into issues?** See [Troubleshooting](submitting.md#troubleshooting)

## Key Principles

1. **Reproducibility**: Deterministic seeds, immutable suite definitions, hash-based integrity verification
2. **Fairness**: Equal capability weighting, standardized evaluation pipeline, transparent baselines
3. **Transparency**: Public results, methodology documentation, peer-reviewed scoring
4. **Accessibility**: Support for all agent types (APIs, local models, code), flexible observation modes
5. **Scientific Rigor**: Confidence intervals, multiple task categories, difficulty levels, SOTA baselines

---

Last updated: 2026-02-12 | Version: 1.0
