# CLI Reference

Complete command-line interface documentation for Agentick.

## Installation

The Agentick CLI is available after installation:

```bash
pip install agentick
```

To verify installation:

```bash
agentick --version
# Output: agentick version 0.1.0
```

## Overview

The Agentick CLI provides commands for:

- Listing available tasks and their properties
- Getting detailed task information
- Running evaluations and benchmarks
- Submitting results to the leaderboard
- Validating submissions
- Viewing leaderboard rankings

## Commands

### agentick list

List all available tasks with optional filtering.

#### Syntax

```bash
agentick list [OPTIONS]
```

#### Options

- `--capability CAPABILITY`: Filter by capability tag (e.g., `navigation`, `memory`, `reasoning`)
- `--difficulty DIFFICULTY`: Filter by difficulty level (e.g., `easy`, `medium`, `hard`, `expert`)
- `--format {text,json}`: Output format (default: `text`)
- `--help`: Show help message

#### Examples

List all available tasks:
```bash
agentick list
```

Filter by capability:
```bash
agentick list --capability navigation
```

Filter by difficulty:
```bash
agentick list --difficulty hard
```

Show as JSON:
```bash
agentick list --format json
```

#### Output

Text format:
```
Available Tasks (40 total):

Navigation (5 tasks)
  - GoToGoal-v0
  - MazeNavigation-v0
  - MultiGoalRoute-v0
  - DynamicObstacles-v0
  - FogOfWar-v0

Memory (5 tasks)
  - KeyDoorPuzzle-v0
  - SequenceMemory-v0
  ...
```

JSON format:
```json
{
  "tasks": [
    {
      "name": "GoToGoal-v0",
      "description": "Navigate to a visible goal in an open grid",
      "capability": "navigation",
      "difficulties": ["easy", "medium", "hard", "expert"]
    }
  ]
}
```

### agentick info

Get detailed information about a specific task.

#### Syntax

```bash
agentick info TASK_NAME [OPTIONS]
```

#### Arguments

- `TASK_NAME`: Task identifier (e.g., `GoToGoal-v0`)

#### Options

- `--difficulty DIFFICULTY`: Show config for specific difficulty (default: all)
- `--format {text,json}`: Output format (default: `text`)
- `--help`: Show help message

#### Examples

Get full task info:
```bash
agentick info GoToGoal-v0
```

Get info for specific difficulty:
```bash
agentick info MazeNavigation-v0 --difficulty hard
```

Get as JSON:
```bash
agentick info KeyDoorPuzzle-v0 --format json
```

#### Output

Text format:
```
Task: GoToGoal-v0
Description: Navigate to a visible goal in an open grid

Capabilities: navigation, basic_navigation

Difficulty Configurations:
  easy:
    - Grid size: 5x5
    - Max steps: 20
    - Obstacles: None

  medium:
    - Grid size: 10x10
    - Max steps: 50
    - Obstacles: 10% density

  hard:
    - Grid size: 15x15
    - Max steps: 100
    - Obstacles: 20% density

  expert:
    - Grid size: 20x20
    - Max steps: 200
    - Obstacles: 25% density

Observation Modes: ascii, language, language_structured, rgb_array, state_dict
```

### agentick evaluate

Run evaluation on a task or suite of tasks.

#### Syntax

```bash
agentick evaluate [OPTIONS]
```

#### Options

- `--task TASK_NAME`: Task to evaluate (required if --suite not provided)
- `--suite SUITE_NAME`: Suite of tasks to evaluate (required if --task not provided)
- `--difficulty DIFFICULTY`: Difficulty level (default: `medium`)
- `--render-mode {ascii,language,rgb_array,state_dict}`: Observation format (default: `ascii`)
- `--n-episodes N`: Number of episodes per task (default: 10)
- `--seed SEED`: Random seed for reproducibility
- `--agent AGENT_TYPE`: Agent type (default: `random`)
- `--output OUTPUT_DIR`: Output directory for results (default: `./results`)
- `--verbose`: Enable verbose output
- `--help`: Show help message

#### Agent Types

- `random`: Random action selection
- `oracle`: Oracle/optimal solution (if available)
- `greedy`: Greedy heuristic-based agent
- `custom:PATH`: Custom agent from Python file

#### Examples

Evaluate random agent on single task:
```bash
agentick evaluate --task GoToGoal-v0 --n-episodes 5
```

Evaluate on full benchmark suite:
```bash
agentick evaluate --suite full --difficulty hard --n-episodes 10
```

Evaluate quick benchmark:
```bash
agentick evaluate --suite quick --difficulty medium
```

Evaluate with custom agent:
```bash
agentick evaluate --task MazeNavigation-v0 --agent custom:./my_agent.py --n-episodes 20
```

Save results to custom location:
```bash
agentick evaluate --suite navigation --output ./my_results --verbose
```

#### Output

```
Evaluating suite: quick (5 tasks)

[1/5] GoToGoal-v0 (medium)
  Episode 1: success=True, reward=1.0, steps=15
  Episode 2: success=True, reward=1.0, steps=18
  Episode 3: success=False, reward=0.0, steps=50
  ...
  Summary: 7/10 success, avg_reward=0.85, avg_steps=21.3

[2/5] MazeNavigation-v0 (medium)
  Episode 1: success=True, reward=1.0, steps=47
  ...

Results saved to: ./results/evaluation_2025-02-12_14-32-45.json
```

### agentick submit

Prepare and submit results to the leaderboard.

#### Syntax

```bash
agentick submit [OPTIONS]
```

#### Options

- `--config CONFIG_FILE`: Path to submission YAML file (required)
- `--results RESULTS_FILE`: Path to evaluation results JSON (required)
- `--verify`: Verify reproducibility before submission
- `--dry-run`: Simulate submission without uploading
- `--verbose`: Enable verbose output
- `--help`: Show help message

#### Submission YAML Format

Create a `submission.yaml` file:

```yaml
# Identity
agent_name: "MyGPT4Agent-v1"
author: "John Doe"
description: "GPT-4 Vision agent with chain-of-thought prompting"
url: "https://github.com/example/my-agent"
tags: ["llm", "vision", "reasoning"]
license: "MIT"
open_weights: false

# Agent Type
agent_type: "api"  # api, huggingface, local_weights, code, docker, git_repo

# Configuration
config:
  provider: "openai"
  model: "gpt-4-vision-preview"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.7

# Observation Mode
observation_mode: "rgb_array"  # ascii, language, rgb_array, language_structured, state_dict

# Evaluation Config
suites:
  - "quick"
  - "full"

# Metadata
hardware: "API (no local compute)"
estimated_cost: "$10"
training_data: "None (zero-shot)"
training_compute: "N/A"
```

#### Examples

Submit with results:
```bash
agentick submit --config submission.yaml --results results.json
```

Verify before submission:
```bash
agentick submit --config submission.yaml --results results.json --verify
```

Dry-run submission:
```bash
agentick submit --config submission.yaml --results results.json --dry-run
```

#### Output

```
Loading submission: submission.yaml
Validating agent configuration...
✓ Agent configuration valid

Loading results: results.json
Verifying reproducibility...
✓ Results verified

Submission Summary:
  Agent: MyGPT4Agent-v1
  Author: John Doe
  Observation mode: rgb_array
  Suites: quick, full
  Tasks: 10
  Average Score: 0.72

Ready to submit. Submit? [y/n]: y
Uploading to leaderboard...
✓ Submission successful
```

### agentick leaderboard

View and interact with the leaderboard.

#### Syntax

```bash
agentick leaderboard [OPTIONS]
```

#### Subcommands

#### agentick leaderboard view

View leaderboard rankings.

```bash
agentick leaderboard view [OPTIONS]
```

##### Options

- `--suite SUITE_NAME`: Show results for specific suite (default: `quick`)
- `--capability CAPABILITY`: Filter by capability (e.g., `navigation`)
- `--sort {score,date,name}`: Sort order (default: `score`)
- `--limit N`: Show top N entries (default: 20)
- `--format {text,json,csv}`: Output format (default: `text`)
- `--help`: Show help message

##### Examples

View top agents on quick suite:
```bash
agentick leaderboard view --suite quick
```

View by capability:
```bash
agentick leaderboard view --suite full --capability navigation --limit 10
```

Export to CSV:
```bash
agentick leaderboard view --format csv > leaderboard.csv
```

##### Output

```
Agentick Leaderboard - Quick Suite (Last 24 hours)

Rank  Agent Name                    Author        Score    Suites   Date
──────────────────────────────────────────────────────────────────────────
1     GPT-4V + Reasoning            OpenAI        0.883    quick    2025-02-12
2     Claude-3-Opus                 Anthropic     0.818    quick    2025-02-11
3     PPO + Curriculum              DeepMind      0.825    quick    2025-02-10
4     Llama-3-70B Instruct          Meta          0.745    quick    2025-02-12
5     GATO Universal Agent          DeepMind      0.712    quick    2025-02-09
```

#### agentick leaderboard compare

Compare multiple agent submissions.

```bash
agentick leaderboard compare AGENT_ID [AGENT_ID ...]
```

##### Examples

Compare two agents:
```bash
agentick leaderboard compare gpt4v claude3 --suite full
```

Compare across capabilities:
```bash
agentick leaderboard compare gpt4v ppo --breakdown
```

##### Output

```
Comparison: GPT-4V vs PPO (Full Suite)

Task                  GPT-4V    PPO       Difference
────────────────────────────────────────────────────
GoToGoal-v0           0.98      0.96      +0.02
MazeNavigation-v0     0.92      0.94      -0.02
KeyDoorPuzzle-v0      0.88      0.72      +0.16
...
```

### agentick validate

Validate a submission or result file.

#### Syntax

```bash
agentick validate [OPTIONS]
```

#### Options

- `--submission FILE`: Path to submission YAML file
- `--result FILE`: Path to result JSON file
- `--check-reproducibility`: Verify reproducibility constraints
- `--verbose`: Show detailed validation output
- `--help`: Show help message

#### Examples

Validate submission file:
```bash
agentick validate --submission submission.yaml
```

Validate results:
```bash
agentick validate --result results.json --check-reproducibility
```

#### Output

```
Validating submission.yaml...

✓ File format valid (YAML)
✓ Required fields present:
  - agent_name
  - author
  - agent_type
  - observation_mode
  - config
✓ Configuration valid for agent_type: api
✓ Suites valid: quick, full

Validation passed!
```

## Global Options

These options are available for all commands:

- `--help`: Show help message for command
- `--version`: Show Agentick version
- `--config FILE`: Load CLI configuration from file
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set logging level
- `--no-color`: Disable colored output

## Environment Variables

### Configuration

- `AGENTICK_HOME`: Directory for Agentick data (default: `~/.agentick`)
- `AGENTICK_CACHE`: Directory for task cache (default: `$AGENTICK_HOME/cache`)
- `AGENTICK_SEED`: Default random seed

### API Keys (for submissions)

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_API_KEY`: Google API key

### Display

- `SDL_VIDEODRIVER`: Set to `dummy` for headless environments
- `SDL_AUDIODRIVER`: Set to `dummy` for headless environments

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Command-line syntax error |
| 3 | Configuration error |
| 4 | Task/benchmark not found |
| 5 | Validation failed |
| 10 | Network error (leaderboard operations) |
| 11 | Authentication failed |

## Configuration Files

### Global Config

Create `~/.agentick/config.yaml`:

```yaml
# Default difficulty level
default_difficulty: medium

# Default render mode
default_render_mode: ascii

# Default output directory
output_dir: ./results

# Logging
log_level: INFO
log_file: ~/.agentick/agentick.log

# Leaderboard
leaderboard_url: https://leaderboard.agentick.ai
timeout: 30
```

### Task Cache

Agentick caches generated task instances for reproducibility:

```bash
~/.agentick/cache/
├── tasks/
│   ├── GoToGoal-v0/
│   │   ├── medium_12345.pkl
│   │   └── hard_67890.pkl
│   └── ...
```

## Examples

### Example 1: Quick Experiment

```bash
# List available tasks
agentick list --capability navigation

# Get task details
agentick info GoToGoal-v0

# Run evaluation
agentick evaluate --task GoToGoal-v0 --n-episodes 5 --verbose
```

### Example 2: Benchmark Submission

```bash
# Run full benchmark
agentick evaluate --suite full --difficulty medium --n-episodes 20 --output my_results/

# Validate submission
agentick validate --result my_results/evaluation.json

# Prepare and submit
agentick submit --config submission.yaml --results my_results/evaluation.json --verify
```

### Example 3: Compare Results

```bash
# View top agents
agentick leaderboard view --suite quick --limit 10

# Compare specific agents
agentick leaderboard compare gpt4v claude3 --suite full
```

## Troubleshooting

### Command Not Found

If `agentick` command is not found, ensure package is installed:

```bash
pip install --upgrade agentick
```

Verify installation:

```bash
python -m agentick --version
```

### Permission Denied (Cache)

If you get permission errors with cache:

```bash
chmod -R u+w ~/.agentick/cache/
```

Or specify custom cache location:

```bash
export AGENTICK_CACHE=/tmp/agentick_cache
```

### Display Issues in Headless Environment

For running in servers without display:

```bash
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
agentick evaluate --task GoToGoal-v0
```

### YAML Parsing Errors

Ensure YAML files are properly formatted:

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('submission.yaml'))"
```

## See Also

- [Getting Started](getting_started/installation.md) - Installation guide
- [Quick Start](getting_started/quickstart.md) - First steps
- [API Reference](api/index.md) - Python API documentation
- [Leaderboard Guide](leaderboard/submitting.md) - Detailed submission guide
