# Human Agents

Collect human baselines, demonstrations, and evaluate human performance on Agentick tasks. This guide covers interactive play, data collection, ethical considerations, and comparative analysis.

## Overview

Human evaluation and baselines are essential because:
- Establish performance ceiling for tasks
- Collect expert demonstrations for training
- Study human strategies and reasoning
- Compare agent performance to humans
- Validate task difficulty and clarity
- Generate labeled training data

**Use Cases:**
- Collecting imitation learning demonstrations
- Running user studies and human evaluations
- Creating difficulty baselines
- Behavioral cloning datasets
- Human-in-the-loop refinement

## Quick Start: Human Play

### Basic Interactive Play

```python
import agentick
from agentick.human import HumanPlayer

# Create environment
env = agentick.make(
    "GoToGoal-v0",
    difficulty="easy",
    render_mode="rgb_array"
)

# Create human player interface
player = HumanPlayer(
    env,
    window_size=(800, 600),
    fps=10,
    show_tutorial=True,
    allow_undo=False,
    practice_rounds=1,
)

# Play session
results = player.play_session(n_rounds=3)

# Print results
for i, result in enumerate(results):
    print(f"Round {i+1}: Reward={result['total_reward']:.1f}, "
          f"Success={result['success']}, Steps={result['step_count']}")
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| Arrow Keys / WASD | Move in direction |
| Space | Pickup/Use item |
| E | Drop item |
| P | Pause |
| U | Undo (if enabled) |
| ESC | Quit |

## Recording Human Trajectories

### Simple Recording

```python
from agentick.data import TrajectoryCollector
import agentick

# Create environment
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")

# Create trajectory collector
collector = TrajectoryCollector(buffer_size=1000)

# Simulate human play (in real scenario, user provides input)
obs, info = env.reset()
collector.start_episode(metadata={"task": "GoToGoal-v0", "difficulty": "easy"})

for step in range(100):
    # In real scenario: action = human_input()
    action = env.action_space.sample()  # Replace with human input

    obs, reward, terminated, truncated, info = env.step(action)
    collector.add_step(obs, action, reward, terminated or truncated, info)

    if terminated or truncated:
        break

collector.end_episode()

# Save trajectories
trajectories = collector.get_trajectories()
print(f"Collected {len(trajectories)} episodes")
```

### Advanced Recording with Metadata

```python
from agentick.data import TrajectoryCollector
import agentick
from datetime import datetime
import json


class HumanDataCollector:
    """Advanced human data collection with rich metadata."""

    def __init__(self, output_dir="human_data"):
        self.output_dir = output_dir
        self.collector = TrajectoryCollector()
        self.participant_metadata = {}

    def start_study(self, participant_id, demographics=None):
        """Start new participant session."""
        self.current_participant = participant_id
        self.participant_metadata[participant_id] = {
            "participant_id": participant_id,
            "start_time": datetime.now().isoformat(),
            "demographics": demographics or {},
            "episodes": [],
        }

    def record_episode(
        self,
        task_name,
        difficulty,
        episode_num,
        results,
    ):
        """Record single episode."""
        episode_data = {
            "task": task_name,
            "difficulty": difficulty,
            "episode": episode_num,
            "success": results["success"],
            "total_reward": results["total_reward"],
            "step_count": results["step_count"],
            "duration": results["duration"],
            "timestamp": datetime.now().isoformat(),
        }
        self.participant_metadata[self.current_participant]["episodes"].append(episode_data)

    def end_study(self):
        """Finalize study for participant."""
        self.participant_metadata[self.current_participant]["end_time"] = \
            datetime.now().isoformat()

    def save_data(self):
        """Save all collected data."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Save participant metadata
        for participant_id, metadata in self.participant_metadata.items():
            path = f"{self.output_dir}/{participant_id}_metadata.json"
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Save trajectories
        trajectories = self.collector.get_trajectories()
        self.collector.save(f"{self.output_dir}/trajectories.npz")

        print(f"Data saved to {self.output_dir}")


# Usage
collector = HumanDataCollector("human_study_data")

for participant_id in ["human_001", "human_002"]:
    collector.start_study(
        participant_id,
        demographics={"age": 25, "experience": "novice"}
    )

    for episode_num in range(3):
        # Run episode...
        results = {"success": True, "total_reward": 10, "step_count": 20, "duration": 5.0}
        collector.record_episode("GoToGoal-v0", "easy", episode_num, results)

    collector.end_study()

collector.save_data()
```

## Running Human Studies

### Study Protocol

```python
"""Complete human study protocol."""

from agentick.human import HumanPlayer
import agentick
import json
from datetime import datetime


class HumanStudy:
    """Manage human study with consent and protocols."""

    CONSENT_FORM = """
    INFORMED CONSENT FORM

    You are being asked to participate in a study evaluating AI agents
    on navigation tasks. The study will take approximately 30 minutes.

    Participation is voluntary and you may withdraw at any time.

    Your data will be kept confidential and used only for research.

    Do you consent to participate? (yes/no)
    """

    def __init__(self, study_name, tasks, n_participants=10):
        self.study_name = study_name
        self.tasks = tasks
        self.n_participants = n_participants
        self.participants = []

    def run_study(self):
        """Run full study with multiple participants."""
        for p_id in range(1, self.n_participants + 1):
            participant_id = f"P{p_id:03d}"

            # Get consent
            if not self.get_consent(participant_id):
                print(f"Participant {participant_id} declined")
                continue

            # Collect demographics
            demographics = self.collect_demographics(participant_id)

            # Run tasks
            results = self.run_participant_session(participant_id, demographics)

            # Debrief
            self.debrief(participant_id, results)

            # Save data
            self.save_participant_data(participant_id, results)

    def get_consent(self, participant_id):
        """Get informed consent."""
        print(f"\n{self.CONSENT_FORM}")
        response = input("Do you consent? (yes/no): ").lower()
        return response == "yes"

    def collect_demographics(self, participant_id):
        """Collect participant demographics."""
        print(f"\nDemographics for {participant_id}")
        age = input("Age: ")
        experience = input("Game experience (novice/intermediate/expert): ")
        return {
            "age": age,
            "experience": experience,
            "timestamp": datetime.now().isoformat(),
        }

    def run_participant_session(self, participant_id, demographics):
        """Run task session for participant."""
        all_results = []

        # Practice task (optional)
        print("\nPractice round - familiarize yourself with controls")
        env = agentick.make(self.tasks[0], difficulty="easy")
        player = HumanPlayer(env, practice_rounds=1)
        practice_results = player.play_session(n_rounds=1)

        # Scored tasks
        for task in self.tasks:
            print(f"\nTask: {task}")
            env = agentick.make(task, difficulty="easy")
            player = HumanPlayer(env)
            task_results = player.play_session(n_rounds=3)
            all_results.extend(task_results)

        return {
            "participant_id": participant_id,
            "demographics": demographics,
            "results": all_results,
        }

    def debrief(self, participant_id, results):
        """Debrief participant after session."""
        print(f"\nDebrief - {participant_id}")
        print("Thank you for participating!")

        successes = sum(1 for r in results["results"] if r.get("success"))
        total = len(results["results"])
        avg_reward = sum(r.get("total_reward", 0) for r in results["results"]) / total

        print(f"Your performance:")
        print(f"  Success rate: {successes}/{total} ({100*successes/total:.0f}%)")
        print(f"  Average reward: {avg_reward:.1f}")

        feedback = input("Any comments about the study? ")
        return feedback

    def save_participant_data(self, participant_id, results):
        """Save participant data."""
        import os
        os.makedirs("study_data", exist_ok=True)

        path = f"study_data/{participant_id}_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def analyze_results(self):
        """Analyze all participant results."""
        import os
        import numpy as np

        results_by_task = {}
        all_results = []

        for filename in os.listdir("study_data"):
            if filename.endswith("_results.json"):
                with open(f"study_data/{filename}") as f:
                    data = json.load(f)
                    all_results.append(data)

        # Calculate statistics
        print("\n" + "="*60)
        print("STUDY RESULTS")
        print("="*60)

        all_successes = []
        all_rewards = []

        for result in all_results:
            for episode in result["results"]:
                all_successes.append(episode.get("success", False))
                all_rewards.append(episode.get("total_reward", 0))

        print(f"Total participants: {len(all_results)}")
        print(f"Total episodes: {len(all_successes)}")
        print(f"Overall success rate: {np.mean(all_successes):.1%}")
        print(f"Mean reward: {np.mean(all_rewards):.2f}")
        print(f"Reward std: {np.std(all_rewards):.2f}")


# Run study
if __name__ == "__main__":
    study = HumanStudy(
        study_name="agent_evaluation_2024",
        tasks=["GoToGoal-v0", "KeyDoorPuzzle-v0"],
        n_participants=5
    )

    study.run_study()
    study.analyze_results()
```

## Data Analysis and Comparison

### Analyzing Human Data

```python
"""Analyze human performance data."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict


class HumanPerformanceAnalyzer:
    """Analyze human performance data."""

    def __init__(self, data_dir="human_data"):
        self.data_dir = Path(data_dir)
        self.data = self._load_data()

    def _load_data(self):
        """Load all participant data."""
        data = {}
        for file in self.data_dir.glob("*_results.json"):
            with open(file) as f:
                participant_id = file.stem.replace("_results", "")
                data[participant_id] = json.load(f)
        return data

    def get_statistics(self):
        """Get overall statistics."""
        all_successes = []
        all_rewards = []
        all_steps = []
        all_durations = []

        for participant, results in self.data.items():
            for episode in results.get("results", []):
                all_successes.append(episode.get("success", False))
                all_rewards.append(episode.get("total_reward", 0))
                all_steps.append(episode.get("step_count", 0))
                all_durations.append(episode.get("duration", 0))

        return {
            "n_participants": len(self.data),
            "n_episodes": len(all_successes),
            "success_rate": np.mean(all_successes),
            "success_std": np.std(all_successes),
            "mean_reward": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "mean_steps": np.mean(all_steps),
            "steps_std": np.std(all_steps),
            "mean_duration": np.mean(all_durations),
            "duration_std": np.std(all_durations),
        }

    def get_per_participant_stats(self):
        """Get statistics per participant."""
        stats = {}
        for participant, results in self.data.items():
            episodes = results.get("results", [])
            successes = [e.get("success", False) for e in episodes]
            rewards = [e.get("total_reward", 0) for e in episodes]

            stats[participant] = {
                "n_episodes": len(episodes),
                "success_rate": np.mean(successes),
                "mean_reward": np.mean(rewards),
            }

        return stats

    def get_per_task_stats(self):
        """Get statistics per task."""
        task_data = {}

        for participant, results in self.data.items():
            for episode in results.get("results", []):
                task = episode.get("task", "unknown")
                if task not in task_data:
                    task_data[task] = []
                task_data[task].append(episode)

        stats = {}
        for task, episodes in task_data.items():
            successes = [e.get("success", False) for e in episodes]
            rewards = [e.get("total_reward", 0) for e in episodes]

            stats[task] = {
                "n_episodes": len(episodes),
                "success_rate": np.mean(successes),
                "mean_reward": np.mean(rewards),
                "reward_std": np.std(rewards),
            }

        return stats

    def print_report(self):
        """Print comprehensive report."""
        print("\n" + "="*70)
        print("HUMAN PERFORMANCE REPORT")
        print("="*70)

        # Overall stats
        overall = self.get_statistics()
        print("\nOverall Performance:")
        print(f"  Participants: {overall['n_participants']}")
        print(f"  Total episodes: {overall['n_episodes']}")
        print(f"  Success rate: {overall['success_rate']:.1%} "
              f"(std: {overall['success_std']:.1%})")
        print(f"  Mean reward: {overall['mean_reward']:.2f} "
              f"(std: {overall['reward_std']:.2f})")
        print(f"  Mean steps: {overall['mean_steps']:.1f} "
              f"(std: {overall['steps_std']:.1f})")

        # Per-task stats
        per_task = self.get_per_task_stats()
        print("\nPer-Task Performance:")
        for task, stats in per_task.items():
            print(f"  {task}:")
            print(f"    Episodes: {stats['n_episodes']}")
            print(f"    Success rate: {stats['success_rate']:.1%}")
            print(f"    Mean reward: {stats['mean_reward']:.2f}")

        # Per-participant stats
        per_participant = self.get_per_participant_stats()
        print("\nPer-Participant Performance:")
        for participant, stats in sorted(per_participant.items()):
            print(f"  {participant}:")
            print(f"    Success rate: {stats['success_rate']:.1%}")
            print(f"    Mean reward: {stats['mean_reward']:.2f}")


# Usage
analyzer = HumanPerformanceAnalyzer("human_study_data")
analyzer.print_report()
```

### Comparing Human vs Agent

```python
"""Compare human and agent performance."""

import numpy as np
from typing import Dict, List


class HumanAgentComparison:
    """Compare human and agent performance."""

    def __init__(self, human_results: List[Dict], agent_results: List[Dict]):
        self.human_results = human_results
        self.agent_results = agent_results

    def compare_metrics(self):
        """Compare key metrics."""
        human_success = np.mean([r.get("success") for r in self.human_results])
        agent_success = np.mean([r.get("success") for r in self.agent_results])

        human_reward = np.mean([r.get("total_reward") for r in self.human_results])
        agent_reward = np.mean([r.get("total_reward") for r in self.agent_results])

        human_steps = np.mean([r.get("step_count") for r in self.human_results])
        agent_steps = np.mean([r.get("step_count") for r in self.agent_results])

        return {
            "success_rate": {
                "human": human_success,
                "agent": agent_success,
                "agent_advantage": agent_success - human_success,
            },
            "reward": {
                "human": human_reward,
                "agent": agent_reward,
                "agent_advantage": agent_reward - human_reward,
            },
            "efficiency": {
                "human": human_steps,
                "agent": agent_steps,
                "efficiency_ratio": agent_steps / human_steps if human_steps > 0 else 1,
            },
        }

    def statistical_test(self):
        """Perform statistical significance test."""
        from scipy import stats

        human_rewards = [r.get("total_reward", 0) for r in self.human_results]
        agent_rewards = [r.get("total_reward", 0) for r in self.agent_results]

        t_stat, p_value = stats.ttest_ind(human_rewards, agent_rewards)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        }

    def print_comparison(self):
        """Print comparison report."""
        metrics = self.compare_metrics()
        stats = self.statistical_test()

        print("\n" + "="*60)
        print("HUMAN vs AGENT COMPARISON")
        print("="*60)

        print("\nSuccess Rate:")
        print(f"  Human: {metrics['success_rate']['human']:.1%}")
        print(f"  Agent: {metrics['success_rate']['agent']:.1%}")
        print(f"  Difference: {metrics['success_rate']['agent_advantage']:+.1%}")

        print("\nMean Reward:")
        print(f"  Human: {metrics['reward']['human']:.2f}")
        print(f"  Agent: {metrics['reward']['agent']:.2f}")
        print(f"  Difference: {metrics['reward']['agent_advantage']:+.2f}")

        print("\nMean Steps to Goal:")
        print(f"  Human: {metrics['efficiency']['human']:.1f}")
        print(f"  Agent: {metrics['efficiency']['agent']:.1f}")
        print(f"  Efficiency ratio: {metrics['efficiency']['efficiency_ratio']:.2f}x")

        print("\nStatistical Significance (t-test):")
        print(f"  t-statistic: {stats['t_statistic']:.3f}")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  Significant: {'Yes' if stats['is_significant'] else 'No'}")

        if metrics['success_rate']['agent_advantage'] > 0:
            print("\nAgent outperforms human")
        elif metrics['success_rate']['agent_advantage'] < 0:
            print("\nHuman outperforms agent")
        else:
            print("\nPerformance is equivalent")


# Usage
human_results = [
    {"success": True, "total_reward": 10, "step_count": 15},
    {"success": True, "total_reward": 10, "step_count": 18},
    {"success": False, "total_reward": 0, "step_count": 100},
]

agent_results = [
    {"success": True, "total_reward": 10, "step_count": 12},
    {"success": True, "total_reward": 10, "step_count": 13},
    {"success": True, "total_reward": 10, "step_count": 14},
]

comparison = HumanAgentComparison(human_results, agent_results)
comparison.print_comparison()
```

## Ethical Considerations

### Informed Consent

```python
"""Proper informed consent procedures."""

CONSENT_TEMPLATE = """
INFORMED CONSENT FORM FOR AI AGENT EVALUATION STUDY

Study: {study_name}
Researcher: {researcher_name}
Institution: {institution}

PURPOSE:
This study evaluates how well AI agents can complete navigation tasks
compared to human players.

PROCEDURE:
- You will complete 3-5 navigation tasks
- Each task takes 5-10 minutes
- Your actions and performance will be recorded
- Total time: approximately 30 minutes

RISKS:
This study poses minimal risk. Tasks are simple navigation exercises.

BENEFITS:
Your participation helps advance research in AI evaluation.

COMPENSATION:
You will receive ${compensation} for your participation.

CONFIDENTIALITY:
Your data will be kept confidential and identified only by a participant ID.
You will not be identified in any published results.

VOLUNTARY PARTICIPATION:
You may withdraw at any time without penalty.

DATA RETENTION:
Data will be retained for 3 years and then securely deleted.

QUESTIONS:
For questions, contact: {contact_info}

I have read and understand the above information and consent to participate.

Participant signature: ____________________  Date: ________

Researcher signature: ____________________  Date: ________
"""


def get_informed_consent(study_name, researcher_name, institution):
    """Get informed consent from participant."""
    form = CONSENT_TEMPLATE.format(
        study_name=study_name,
        researcher_name=researcher_name,
        institution=institution,
        compensation=50,
        contact_info="researcher@institution.edu",
    )

    print(form)
    response = input("\nDo you consent to participate? (yes/no): ")

    return response.lower() == "yes"
```

### Privacy and Data Protection

```python
"""Protect participant privacy."""

from pathlib import Path
import json
import hashlib
from datetime import datetime, timedelta


class PrivacyCompliantDataHandler:
    """Handle data with privacy protection."""

    def __init__(self, data_dir="study_data", retention_days=1095):
        """Initialize privacy-compliant handler."""
        self.data_dir = Path(data_dir)
        self.retention_days = retention_days  # 3 years default

    def anonymize_participant(self, participant_id: str) -> str:
        """Create anonymous ID from participant info."""
        # Hash to prevent reverse engineering
        return hashlib.sha256(participant_id.encode()).hexdigest()[:8]

    def save_participant_data(
        self,
        participant_id: str,
        data: dict,
        include_demographics: bool = False,
    ):
        """Save data with anonymization."""
        # Anonymize ID
        anon_id = self.anonymize_participant(participant_id)

        # Remove PII unless needed
        if not include_demographics:
            data.pop("demographics", None)
            data.pop("name", None)
            data.pop("email", None)

        # Add retention date
        retention_date = (datetime.now() + timedelta(days=self.retention_days)).isoformat()
        data["_retention_date"] = retention_date

        # Save
        self.data_dir.mkdir(exist_ok=True)
        path = self.data_dir / f"{anon_id}_data.json"

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def cleanup_expired_data(self):
        """Delete data past retention date."""
        now = datetime.now()

        for file in self.data_dir.glob("*_data.json"):
            with open(file) as f:
                data = json.load(f)

            retention_date = datetime.fromisoformat(data.get("_retention_date", "2099-01-01"))

            if now > retention_date:
                file.unlink()
                print(f"Deleted expired data: {file}")

    def securely_delete_file(self, path):
        """Securely delete file (overwrite before deleting)."""
        import os

        try:
            # Overwrite with random data
            file_size = os.path.getsize(path)
            with open(path, "wb") as f:
                f.write(os.urandom(file_size))

            # Delete
            os.remove(path)
            print(f"Securely deleted: {path}")
        except Exception as e:
            print(f"Error securely deleting {path}: {e}")
```

## Best Practices

1. **Get explicit consent** - Use formal consent forms
2. **Protect privacy** - Anonymize data, use IDs only
3. **Secure data** - Encrypt files, secure storage
4. **Document protocols** - Follow ethical guidelines
5. **Compensate fairly** - Pay participants appropriately
6. **Debrief thoroughly** - Explain purpose and findings
7. **Respect withdrawal** - Allow participants to quit anytime
8. **Retain data appropriately** - Delete after retention period
9. **Get IRB approval** - For formal studies
10. **Report results** - Share findings with participants

## Complete Study Example

See the example code above in the "Running Human Studies" section for a complete working example of implementing a human study with proper protocols.
