"""
Demonstrate all observation modes on the same task.

Shows how different agents can perceive the environment.
Runtime: <10 seconds
"""

import agentick


def main():
    task = "GoToGoal-v0"
    difficulty = "easy"
    seed = 42

    # All supported render modes
    render_modes = [
        "ascii",
        "language",
        "language_structured",
        "state_dict",
        # "rgb_array",  # Would return numpy array
        # "human",  # Would open pygame window
    ]

    print(f"Task: {task}, Difficulty: {difficulty}\n")
    print("=" * 80)

    for mode in render_modes:
        print(f"\nRender Mode: {mode}")
        print("-" * 80)

        env = agentick.make(task, difficulty=difficulty, render_mode=mode)
        obs, info = env.reset(seed=seed)

        # Show observation
        if mode == "state_dict":
            print(f"State dict keys: {obs.keys()}")
            if hasattr(obs.get('grid'), 'shape'):
                print(f"Grid shape: {obs['grid'].shape}")
            else:
                print(f"Grid type: {type(obs.get('grid'))}")
            if 'agent_pos' in obs:
                print(f"Agent position: {obs['agent_pos']}")
        else:
            print(obs)

        env.close()
        print()


if __name__ == "__main__":
    main()
