# Basics

Start here. These five scripts introduce the core Agentick API in order of
increasing complexity. Each one runs in seconds with no extra dependencies
beyond the base install.

## Prerequisites

```bash
uv sync
```

## Scripts

- **01_make_and_step.py** -- Create an environment, reset it, take random
  actions, and print rewards. The minimal interaction loop.
- **02_render_modes.py** -- Show the same task rendered as ASCII, natural
  language, structured language, and state dict so you can see what each
  observation mode looks like.
- **03_reward_modes.py** -- Compare sparse rewards (signal only at the goal)
  with dense rewards (progress signal every step) on the same task.
- **04_list_tasks.py** -- Enumerate all registered tasks and filter them by
  capability tag (navigation, memory, reasoning, etc.).
- **05_difficulty_levels.py** -- Run GoToGoal at easy, medium, hard, and expert
  to see how grid size, obstacle count, and max steps scale with difficulty.

## Learning Progression

1. Start with `01_make_and_step.py` to understand the Gymnasium step loop.
2. Move to `02_render_modes.py` to learn what observations look like.
3. Run `03_reward_modes.py` to understand reward shaping.
4. Use `04_list_tasks.py` to discover the full task catalogue.
5. Finish with `05_difficulty_levels.py` to see how difficulty affects a task.

## Running

```bash
uv run python examples/basics/01_make_and_step.py
uv run python examples/basics/02_render_modes.py
uv run python examples/basics/03_reward_modes.py
uv run python examples/basics/04_list_tasks.py
uv run python examples/basics/05_difficulty_levels.py
```
