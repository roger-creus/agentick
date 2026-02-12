# Agentick

**Universal benchmark for evaluating AI agents across all paradigms**

## Overview

Agentick is a comprehensive, research-grade benchmark for evaluating AI agents across diverse cognitive capabilities. Unlike existing benchmarks, Agentick is designed for **training AND evaluation**, supports **all agent types** (RL, LLM, VLM, bots, humans), and provides **capability-decomposed** diagnostics.

## Key Features

- 🧠 **Comprehensive**: 20+ tasks spanning navigation, memory, reasoning, and skill composition
- 🔄 **Training-First**: Fast vectorized environments, trajectory export, curriculum learning
- 🎯 **Multi-Modal**: Text, pixels, language, structured state—all for the same task
- 📊 **Research-Grade**: Statistical rigor, reproducibility, publication-ready figures
- 🔬 **Capability Profile**: Not just "73% score"—detailed diagnostic breakdown

## Quick Start

```bash
# Install
pip install agentick

# Run a simple experiment
agentick experiment run --config experiments/configs/quick/sanity_check.yaml

# Generate visualizations
python experiments/scripts/generate_all_figures.py
```

## Why Agentick?

| Feature | ARC-AGI-3 | AutumnBench | **Agentick** |
|---------|-----------|-------------|--------------|
| Trainable | ❌ | ❌ | ✅ |
| Multi-modal | ❌ | ❌ | ✅ |
| Capability Breakdown | ❌ | Limited | ✅ |
| All Agent Types | ❌ | Partial | ✅ |
| Infinite Curriculum | ❌ | ❌ | ✅ |

## Citation

```bibtex
@software{agentick2025,
  title={Agentick: A Comprehensive Benchmark for Evaluating AI Agents},
  author={Agentick Team},
  year={2025},
  url={https://github.com/agentick/agentick}
}
```
