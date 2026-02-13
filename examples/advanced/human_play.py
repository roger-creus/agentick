"""
Human-playable interface for Agentick tasks.

Allows humans to play tasks using keyboard controls.
Useful for:
- Understanding task difficulty
- Collecting human demonstrations
- Debugging task design
"""

import sys

import agentick


class HumanPlayer:
    """
    Interactive human player for Agentick tasks.

    Controls:
    - Arrow keys or WASD: Move
    - Q: Quit
    - R: Reset episode
    - H: Show help
    """

    def __init__(self, task_id: str, difficulty: str = "easy"):
        """
        Initialize human player.

        Args:
            task_id: Task to play
            difficulty: Difficulty level
        """
        self.task_id = task_id
        self.difficulty = difficulty

        # Create environment with ASCII rendering for terminal play
        self.env = agentick.make(task_id, difficulty=difficulty, render_mode="ascii")

        # Action mapping
        self.action_map = {
            "w": 0,  # up
            "d": 1,  # right
            "s": 2,  # down
            "a": 3,  # left
            "W": 0,
            "D": 1,
            "S": 2,
            "A": 3,
        }

        # Stats
        self.episode_count = 0
        self.total_reward = 0
        self.step_count = 0
        self.successes = 0

    def show_help(self):
        """Display help message."""
        print("\n" + "=" * 60)
        print("CONTROLS")
        print("=" * 60)
        print("  W / Up    - Move up")
        print("  D / Right - Move right")
        print("  S / Down  - Move down")
        print("  A / Left  - Move left")
        print("  R         - Reset episode")
        print("  H         - Show this help")
        print("  Q         - Quit")
        print("=" * 60)

    def play_episode(self, seed: int | None = None):
        """Play a single episode."""
        obs, info = self.env.reset(seed=seed)
        episode_reward = 0
        steps = 0

        # Display initial state
        self.clear_screen()
        print(f"\n=== {self.task_id} - Difficulty: {self.difficulty} ===")
        print(f"Episode {self.episode_count + 1}")
        print()
        print(obs)
        print()
        print(f"Step: {steps} | Reward: {episode_reward:.2f}")
        print("\nPress H for help, Q to quit")

        # Episode loop
        while True:
            # Get user input
            try:
                print("\nAction (WASD): ", end="", flush=True)
                key = self.get_key()

                # Handle special commands
                if key.lower() == "q":
                    return "quit"
                elif key.lower() == "r":
                    return "reset"
                elif key.lower() == "h":
                    self.show_help()
                    input("\nPress Enter to continue...")
                    self.clear_screen()
                    print(obs)
                    continue

                # Convert key to action
                if key not in self.action_map:
                    print(f"Invalid key: {key}")
                    continue

                action = self.action_map[key]

                # Take step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1

                # Display new state
                self.clear_screen()
                print(f"\n=== {self.task_id} - Difficulty: {self.difficulty} ===")
                print(f"Episode {self.episode_count + 1}")
                print()
                print(obs)
                print()
                print(f"Step: {steps} | Reward: {episode_reward:.2f}")

                # Check if episode ended
                if terminated or truncated:
                    success = info.get("success", False)

                    print("\n" + "=" * 60)
                    print("EPISODE COMPLETE")
                    print("=" * 60)
                    print(f"Total reward: {episode_reward:.2f}")
                    print(f"Steps: {steps}")
                    print(f"Success: {'✓' if success else '✗'}")

                    if success:
                        self.successes += 1

                    input("\nPress Enter to continue...")
                    return "done"

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                return "quit"
            except Exception as e:
                print(f"\nError: {e}")
                continue

    def run(self):
        """Run interactive session."""
        self.clear_screen()
        print("\n" + "=" * 60)
        print("AGENTICK HUMAN PLAYER")
        print("=" * 60)
        print(f"Task: {self.task_id}")
        print(f"Difficulty: {self.difficulty}")
        print()
        input("Press Enter to start...")

        while True:
            result = self.play_episode(seed=None)

            if result == "quit":
                break
            elif result == "reset":
                continue
            elif result == "done":
                self.episode_count += 1

                # Ask if player wants to continue
                print("\nPlay again? (y/n): ", end="", flush=True)
                choice = self.get_key().lower()

                if choice != "y":
                    break

        # Show final stats
        self.show_stats()
        self.env.close()

    def show_stats(self):
        """Show session statistics."""
        print("\n" + "=" * 60)
        print("SESSION STATISTICS")
        print("=" * 60)
        print(f"Episodes played: {self.episode_count}")
        print(f"Successes: {self.successes}")
        if self.episode_count > 0:
            success_rate = self.successes / self.episode_count
            print(f"Success rate: {success_rate:.1%}")
        print("=" * 60)

    @staticmethod
    def get_key() -> str:
        """Get single keypress from user."""
        # For Unix/Linux/Mac
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                key = sys.stdin.read(1)

                # Handle arrow keys (escape sequences)
                if key == "\x1b":
                    next1 = sys.stdin.read(1)
                    next2 = sys.stdin.read(1)
                    if next1 == "[":
                        if next2 == "A":
                            return "w"  # Up
                        elif next2 == "B":
                            return "s"  # Down
                        elif next2 == "C":
                            return "d"  # Right
                        elif next2 == "D":
                            return "a"  # Left

                return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        except ImportError:
            # Fallback for Windows or if termios unavailable
            return input().strip()[:1] if input else ""

    @staticmethod
    def clear_screen():
        """Clear terminal screen."""
        import os

        os.system("cls" if os.name == "nt" else "clear")


def main():
    """Run human player interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Play Agentick tasks as a human")
    parser.add_argument("--task", type=str, default="GoToGoal-v0", help="Task ID to play")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "expert"],
        help="Difficulty level",
    )

    args = parser.parse_args()

    # Create and run player
    player = HumanPlayer(task_id=args.task, difficulty=args.difficulty)
    player.run()


if __name__ == "__main__":
    main()
