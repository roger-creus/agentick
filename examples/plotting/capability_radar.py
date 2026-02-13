"""
Generate capability radar chart from experiment results.

Shows agent performance across different capability dimensions.
Runtime: <5 seconds
Requires: uv sync --extra viz
"""


def main():
    print("Capability Radar Chart")
    print("=" * 80)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed.")
        print("Install with: uv sync --extra viz")
        return

    # Example data: scores per capability (0-1 scale)
    capabilities = ["Navigation", "Memory", "Reasoning", "Skill", "Control", "Combinatorial"]

    agents = {
        "Random": [0.1, 0.05, 0.02, 0.08, 0.15, 0.03],
        "PPO": [0.65, 0.45, 0.35, 0.40, 0.70, 0.25],
        "GPT-4o": [0.80, 0.75, 0.85, 0.70, 0.60, 0.65],
    }

    # Create radar chart
    num_vars = len(capabilities)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    for agent_name, scores in agents.items():
        values = scores + scores[:1]  # Complete the circle
        ax.plot(angles, values, "o-", linewidth=2, label=agent_name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized Score", labelpad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Agent Performance by Capability", pad=20, fontsize=14, fontweight="bold")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("capability_radar.png", dpi=300, bbox_inches="tight")
    print("Saved capability_radar.png")
    plt.show()


if __name__ == "__main__":
    main()
