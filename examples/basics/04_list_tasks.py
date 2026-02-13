"""
List all available tasks and filter by capability.

Shows task discovery and filtering.
Runtime: <2 seconds
"""

from agentick.tasks.registry import list_tasks


def main():
    # List all tasks
    all_tasks = list_tasks()
    print(f"Total Tasks: {len(all_tasks)}")
    print("=" * 80)
    for task in all_tasks:
        print(f"  - {task}")
    print()

    # Filter by capability
    capabilities = ["navigation", "memory", "reasoning", "skill", "control", "combinatorial"]

    for capability in capabilities:
        tasks = list_tasks(capability=capability)
        print(f"\n{capability.upper()} Tasks ({len(tasks)} total):")
        print("-" * 80)
        for task in tasks:
            print(f"  - {task}")


if __name__ == "__main__":
    main()
