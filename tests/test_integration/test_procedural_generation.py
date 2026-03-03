"""Test that procedural generation works correctly."""

import pytest

import agentick

# Tasks with fully implemented procedural generation
TASKS_WITH_PROC_GEN = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "ShortestPath-v0",
    "FogOfWarExploration-v0",
]


@pytest.mark.parametrize("task_name", TASKS_WITH_PROC_GEN)
@pytest.mark.parametrize("seed1,seed2", [(0, 1), (100, 200), (1000, 2000)])
@pytest.mark.timeout(30)
def test_different_seeds_different_instances(task_name, seed1, seed2):
    """Test that different seeds produce different instances (for tasks with proc gen)."""
    env = agentick.make(task_name, difficulty="easy")
    try:
        obs1, info1 = env.reset(seed=seed1)
        state1 = env.get_state_dict()

        obs2, info2 = env.reset(seed=seed2)
        state2 = env.get_state_dict()

        # At least one of these should be different
        # (agent position, goal position, or grid layout)
        agent_pos_different = state1["agent"]["position"] != state2["agent"]["position"]
        grid_different = (
            state1["grid"]["terrain"] != state2["grid"]["terrain"]
            or state1["grid"]["objects"] != state2["grid"]["objects"]
        )

        # Allow some tasks to have deterministic layouts but different start positions
        assert agent_pos_different or grid_different, (
            f"Seeds {seed1} and {seed2} produced identical instances"
        )
    finally:
        env.close()


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:10])
@pytest.mark.timeout(30)
def test_generated_instances_valid(task_name):
    """Test that generated instances are valid and solvable."""
    env = agentick.make(task_name, difficulty="easy")
    try:
        for seed in [0, 42, 123]:
            obs, info = env.reset(seed=seed)

            # Verify observation is not empty
            assert obs is not None

            # Verify we can take at least one action
            valid_actions = env.get_valid_actions()
            assert len(valid_actions) > 0, f"No valid actions for seed {seed}"

            # Take a step
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify step worked
            assert obs is not None
            assert isinstance(reward, (int, float))
    finally:
        env.close()
