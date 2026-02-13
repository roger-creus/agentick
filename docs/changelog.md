# Changelog

All notable changes to Agentick are documented in this file. The project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-02-12

### Initial Release

**Agentick 0.1.0** is the first public release of the universal benchmark for evaluating AI agents across all paradigms.

#### Core Features

**38 Procedurally Generated Tasks**
- 5 Navigation tasks: GoToGoal, MazeNavigation, MultiGoalRoute, DynamicObstacles, FogOfWar
- 5 Memory tasks: KeyDoorPuzzle, SequenceMemory, BreadcrumbTrail, DelayedGratification, BacktrackPuzzle
- 5 Reasoning tasks: SokobanPush, SwitchCircuit, SymbolMatching, CausalChain, RuleInduction
- 5 Skill Composition tasks: ToolUse, RecipeAssembly, MultiRoomEscape, ResourceManagement, EmergentStrategy
- 4 Control tasks: PreciseNavigation, TimingChallenge, ChaseEvade, Herding
- 4 Combinatorial tasks: LightsOut, TileSorting, GraphColoring, PackingPuzzle
- 3 Compositional tasks: InstructionFollowing, ProgramSynthesis, RecursiveRooms
- 3 Adversarial tasks: NoisyObservation, DistributionShift, DeceptiveReward
- 2 Meta-Learning tasks: FewShotAdaptation, TaskInterference
- 2 Multi-Agent tasks: CooperativeTransport, CompetitiveTag

**Multi-Modal Observations (6 rendering modes)**
- ASCII: Text grid representation
- Language: Natural language descriptions
- Language Structured: JSON/dict with detailed state info
- RGB Array: Pixel observations (32x32, 64x64, configurable)
- State Dict: Full internal game state
- Human: Interactive play with keyboard control

**Universal Agent Support**
- Deep RL agents (PPO, DQN, A3C, etc.)
- Language Model agents (GPT, Claude, Llama)
- Vision-Language Model agents (GPT-4V, Gemini)
- Programmatic bots (rule-based, search-based)
- Human players (interactive mode)

**Fast Vectorized Environments**
- Parallel environment execution (1-256 parallel envs)
- ~10,000 steps/second on modern hardware
- GPU-accelerated rendering pipeline
- Memory-efficient batch processing

**Reproducibility & Seeding**
- Deterministic procedural generation
- Full seed control (environment seed, RNG seed)
- Results reproducible across runs
- Seed verification system

**Benchmark Suites**
- Quick suite (5 representative tasks)
- Category-specific suites (navigation, memory, reasoning, etc.)
- Full suite (all 40+ tasks)
- Custom suite creation

**Scoring & Evaluation**
- Normalized scoring formula [0, 1]
- Random baseline computation
- Optimal performance estimation
- Success rate metrics
- Dense and sparse reward modes
- Capability profile generation

**Task Generation System**
- Procedural maze generation (recursive backtracking, DFS)
- Grid randomization with configurable density
- Instance validation and solvability checking
- Optimal path computation for analytics
- Grid search and pathfinding utilities

**Gymnasium API Compatibility**
- Standard reset(), step(), render() interface
- Compatible with RL libraries (CleanRL, Stable Baselines)
- Proper action and observation space definitions
- Metadata and specification support

**Flexible Configuration**
- 4 difficulty levels per task (easy, medium, hard, expert)
- Configurable grid sizes and episode lengths
- Custom task parameters
- Difficulty-specific scaling

**Community Leaderboard System**
- Agent submission framework
- Multiple submission types (API, HuggingFace, local, code, Docker)
- Result integrity verification (hashing)
- Leaderboard ranking and comparison
- Public and private submission modes

#### Architecture Components

**Core Environment**
- `agentick.make()`: Simple environment creation
- `agentick.make_suite()`: Benchmark suite creation
- `AgentickEnv`: Base environment class
- Task registry system with decorators

**Grid System**
- 2D grid data structure with terrain and objects
- Entity system (agents, items, obstacles)
- Cell type system (empty, wall, water, lava, etc.)
- Object type system (goal, key, door, etc.)

**Rendering Engine**
- Multi-modal renderer factory
- Language generation for natural descriptions
- ASCII art rendering
- Pixel rendering with sprite system
- State dict export

**Agent Interfaces**
- RLInterface: Vectorized environment wrapper
- LLMInterface: Text-based agent interface
- VLMInterface: Vision-language model interface
- BotInterface: Programmatic bot interface
- HumanInterface: Interactive play interface

**Reward System**
- Dense reward shaping
- Sparse reward (terminal only)
- Potential-based reward shaping
- Custom reward functions
- Reward normalization

**Validation System**
- Instance solvability checking
- Reachability analysis
- Configuration validation
- Test instance generation

#### Documentation

**Comprehensive documentation with:**
- Getting started guides (installation, quickstart, first experiment)
- Conceptual documentation (architecture, tasks, observations, rewards, scoring)
- Agent-specific guides (RL, LLM, VLM, custom)
- Extension guides (custom tasks, custom rewards, custom renderers)
- Experiment guides (running, analysis, visualization, reproducibility)
- Full API documentation (auto-generated from code)
- 50+ code examples
- FAQ with 20+ common questions

#### Examples

**Included example scripts:**
- `examples/random_agent.py`: Simple random exploration
- `examples/rl_training.py`: Deep RL training with PyTorch
- `examples/llm_agent.py`: LLM zero-shot evaluation
- `examples/vlm_agent.py`: Vision-language model evaluation
- `examples/programmatic_bot.py`: Rule-based bot
- `examples/human_play.py`: Interactive human play
- `examples/custom_task.py`: Creating custom tasks
- `examples/curriculum_training.py`: Curriculum learning

#### Testing

**Comprehensive test suite:**
- Environment functionality tests
- Task registration and generation tests
- Observation mode tests
- Agent interface tests
- End-to-end integration tests
- ~90% code coverage

#### Package Structure

```
agentick/
├── core/              # Core environment and grid system
├── tasks/             # Task implementations (38 tasks)
├── generation/        # Procedural generation utilities
├── interfaces/        # Agent interface wrappers
├── benchmark/         # Benchmark runner and scoring
├── leaderboard/       # Leaderboard system
├── utils/             # Utility functions
└── ...
```

#### Dependencies

**Core dependencies:**
- gymnasium >= 1.0 (RL environment API)
- numpy >= 1.24 (numerical computing)
- pygame >= 2.5 (rendering)
- Pillow >= 10.0 (image processing)
- pydantic >= 2.0 (validation)
- rich >= 13.0 (CLI output)
- scipy >= 1.17.0 (scientific computing)

**Optional dependencies:**
- RL: torch, wandb
- LLM: openai, anthropic, google-genai
- Local LLM: transformers, vllm
- Fine-tuning: trl, accelerate, peft
- Visualization: matplotlib, plotly, seaborn
- Docs: mkdocs, mkdocs-material

#### Platforms Supported

- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (WSL2 recommended)
- Cloud environments (AWS, GCP, Azure)
- Docker containers

#### Python Versions

- Python 3.11
- Python 3.12
- Python 3.13

#### License

MIT License - Open source for research and commercial use

#### Citation

```bibtex
@software{agentick2025,
  title={Agentick: A Comprehensive Benchmark for Evaluating Generally Capable AI Agents},
  author={Agentick Team},
  year={2025},
  url={https://github.com/anthropics/agentick}
}
```

#### Acknowledgments

Agentick is built on:
- Gymnasium (OpenAI Gym successor)
- NumPy and SciPy
- PyGame and Pillow
- Pydantic
- Rich

#### Known Limitations & Future Work

**v0.1.0 Known Limitations:**
- Leaderboard web interface coming soon
- Some rendering modes optimized for CPU
- Multi-agent tasks single-machine only (no distributed agents)

**Planned for v0.2.0:**
- 3D environment support
- Hierarchical task generation
- Imitation learning support
- Advanced curriculum strategies

**Planned for v0.3.0+:**
- Cross-agent competition modes
- Skill discovery and abstraction
- Generalization benchmarks
- Transfer learning evaluation

---

## Versioning Policy

Agentick follows semantic versioning:

- **MAJOR**: Breaking changes to API or evaluation methodology
- **MINOR**: New tasks, features, or non-breaking enhancements
- **PATCH**: Bug fixes and documentation updates

### Upgrade Guide: Updating from Future Versions

When new versions are released:

```bash
# Check current version
uv pip show agentick

# Upgrade to latest
uv sync

# Upgrade to specific version
# Edit pyproject.toml and run uv sync
```

To pin version in projects:

```bash
# In pyproject.toml
dependencies = ["agentick>=0.1.0,<0.2.0"]
```

---

## Release Timeline

**v0.1.0** (Current) - Initial release with 38 tasks
- Release date: February 12, 2025
- Status: Stable

**v0.2.0** (Planned Q2 2025)
- 60+ tasks
- Web leaderboard interface
- Distributed evaluation
- Advanced curriculum learning

**v0.3.0** (Planned Q4 2025)
- 100+ tasks
- 3D environments
- Cross-agent competitions
- Skill abstraction framework

---

## Contributing & Community

### Report Issues

Found a bug? Please report it:
https://github.com/anthropics/agentick/issues

### Contribute Code

Want to contribute? See [CONTRIBUTING.md](https://github.com/agentick/agentick/blob/main/CONTRIBUTING.md) for guidelines.

### Discussions

Discuss ideas and share feedback:
https://github.com/anthropics/agentick/discussions

---

## See Also

- [README.md](https://github.com/agentick/agentick/blob/main/README.md) - Project overview
- [CONTRIBUTING.md](https://github.com/agentick/agentick/blob/main/CONTRIBUTING.md) - Contribution guidelines
- [LICENSE](../LICENSE) - MIT license
- [docs/](.) - Full documentation

---

**Questions?** Check the [FAQ](faq.md) or open an issue!
