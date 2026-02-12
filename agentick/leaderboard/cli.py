"""CLI commands for leaderboard evaluation."""

from __future__ import annotations

import argparse

from agentick.leaderboard.evaluator import LeaderboardEvaluator
from agentick.leaderboard.integrity import verify_result
from agentick.leaderboard.result import EvaluationResult
from agentick.leaderboard.submission import SubmissionSpec


def cmd_evaluate(args):
    """Run evaluation command."""
    # Load submission
    submission = SubmissionSpec.from_yaml(args.submission)

    # Validate
    warnings = submission.validate_submission()
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Run evaluation
    evaluator = LeaderboardEvaluator(verbose=True)

    _result = evaluator.evaluate(
        submission=submission,
        suite=args.suite,
        output_dir=args.output,
        verify_reproducibility_flag=args.verify_reproducibility,
    )

    print("\n✓ Evaluation complete!")
    return 0


def cmd_verify(args):
    """Verify evaluation result."""
    # Load result
    result = EvaluationResult.from_json(args.result)

    # Verify hash
    is_valid = verify_result(result)

    if is_valid:
        print("✓ Result integrity verified - hash matches")
    else:
        print("✗ Result integrity check FAILED - hash mismatch")
        return 1

    # Print summary
    print(f"\n{result.get_summary()}")

    return 0


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Agentick Leaderboard CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an agent submission")
    eval_parser.add_argument("--submission", required=True, help="Path to submission YAML")
    eval_parser.add_argument("--suite", required=True, help="Benchmark suite name")
    eval_parser.add_argument("--output", default="results", help="Output directory")
    eval_parser.add_argument(
        "--verify-reproducibility", action="store_true", help="Verify reproducibility"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify evaluation result")
    verify_parser.add_argument("--result", required=True, help="Path to result JSON")
    verify_parser.set_defaults(func=cmd_verify)

    # Parse and run
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    exit(main())
