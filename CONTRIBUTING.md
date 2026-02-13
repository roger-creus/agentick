# Contributing to Agentick

Thank you for your interest in contributing to Agentick! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please treat all contributors with respect.

## How to Contribute

### Reporting Issues

- **Search existing issues** before creating a new one
- **Use the issue template** and provide as much detail as possible
- **Include reproducible steps** for bugs
- **Provide context** for feature requests

### Contributing Code

1. **Fork the repository** and create a new branch:
   ```bash
   git clone https://github.com/anthropics/agentick
   cd agentick
   git checkout -b your-feature-branch
   ```

2. **Set up your development environment**:
   ```bash
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   uv sync
   uv sync --extra all
   ```

3. **Make your changes** following our coding standards:
   - Follow PEP 8 style guide
   - Add docstrings to all public classes and methods (Google style)
   - Include type hints on all function signatures
   - Write clear, descriptive commit messages

4. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/ -v

   # Check coverage
   pytest --cov=agentick --cov-report=term-missing

   # Lint your code
   ruff check agentick/ tests/
   ruff format agentick/ tests/

   # Type check
   mypy agentick/ --ignore-missing-imports
   ```

5. **Submit a pull request**:
   - Push your changes to your fork
   - Create a PR against the `main` branch
   - Fill out the PR template
   - Link any related issues

## Development Guidelines

### Adding a New Task

1. Create a new file in `agentick/tasks/<category>/`
2. Subclass `TaskSpec` and implement:
   - `generate(seed)` - procedural instance generation
   - `compute_dense_reward()` - reward shaping (optional)
   - `check_success()` - success criterion
   - `get_optimal_return()` - theoretical optimal
   - `get_random_baseline()` - random agent performance
3. Register with `@register_task()` decorator
4. Add tests in `tests/test_tasks/`
5. Document in `docs/concepts/tasks.md`

See [docs/extending/custom_tasks.md](docs/extending/custom_tasks.md) for details.

### Code Style

- **Python**: PEP 8, enforced with ruff
- **Line length**: 100 characters
- **Imports**: Absolute imports, sorted with ruff
- **Naming**:
  - Functions/methods: `snake_case`
  - Classes: `CamelCase`
  - Constants: `UPPER_CASE`
- **Docstrings**: Google style
- **Type hints**: All public functions

### Testing

- **Unit tests** for individual components
- **Integration tests** for task interactions
- **Regression tests** for bugs
- **Coverage target**: >90%
- Use `pytest` fixtures for common setups
- Use `tmp_path` for temporary files
- Parametrize tests when testing multiple cases

### Documentation

- Update relevant docs in `docs/` for new features
- Add docstrings to all public APIs
- Include code examples in documentation
- Update `README.md` if adding major features
- Keep `CHANGELOG.md` up to date

## Pull Request Process

1. **Before submitting**:
   - Ensure all tests pass
   - Update documentation
   - Add changelog entry
   - Rebase on latest `main`

2. **PR requirements**:
   - Descriptive title and description
   - Linked issues (if applicable)
   - Passing CI checks
   - Code review approval

3. **Review process**:
   - Maintainers will review within 5 business days
   - Address feedback and update PR
   - Once approved, maintainers will merge

## Project Structure

```
agentick/
├── core/           # Core environment, grid, entity, renderer
├── tasks/          # Task implementations by category
├── generation/     # Procedural generation algorithms
├── interfaces/     # Agent interfaces (RL, LLM, VLM, bot)
├── benchmark/      # Benchmarking and metrics
├── leaderboard/    # Leaderboard system
├── rewards/        # Reward shaping utilities
├── wrappers/       # Gymnasium wrappers
├── worldmodel/     # World model evaluation
└── ...

tests/              # Comprehensive test suite
docs/               # Documentation source files
examples/           # Example scripts
scripts/            # Development scripts
```

## Questions?

- **General questions**: Open a GitHub discussion
- **Bug reports**: Create an issue
- **Feature proposals**: Create an issue with [FEATURE] tag
- **Security issues**: Email security@agentick.ai

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes for major contributions
- Project documentation where appropriate

Thank you for contributing to Agentick!
