"""
Validate a leaderboard submission before uploading.

This example demonstrates:
- Loading and validating submission YAML
- Checking required fields
- Verifying adapter code syntax
- Ensuring submission is complete

Requirements:
    uv sync

Usage:
    uv run python examples/leaderboard/validate_submission.py submission.yaml
"""

import argparse
import ast
import sys
from pathlib import Path

import yaml


def validate_yaml_structure(data: dict) -> list[str]:
    """Validate YAML structure."""
    errors = []

    # Required top-level fields
    required_fields = ["name", "agent", "contact"]

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate agent section
    if "agent" in data:
        agent = data["agent"]

        if not isinstance(agent, dict):
            errors.append("'agent' must be a dictionary")
        else:
            # Required agent fields
            agent_required = ["adapter", "model"]

            for field in agent_required:
                if field not in agent:
                    errors.append(f"Missing required agent field: {field}")

            # Validate adapter code
            if "adapter" in agent:
                adapter = agent["adapter"]

                if not isinstance(adapter, str):
                    errors.append("'adapter' must be a string (Python code)")
                elif len(adapter.strip()) == 0:
                    errors.append("'adapter' code is empty")

    # Validate contact section
    if "contact" in data:
        contact = data["contact"]

        if not isinstance(contact, dict):
            errors.append("'contact' must be a dictionary")
        else:
            if "email" not in contact and "github" not in contact:
                errors.append("'contact' must have at least 'email' or 'github'")

    return errors


def validate_python_syntax(code: str) -> list[str]:
    """Validate Python code syntax."""
    errors = []

    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error in adapter code: {e}")

    return errors


def validate_adapter_structure(code: str) -> list[str]:
    """Validate adapter code structure."""
    errors = []

    # Check for required function
    if "def get_action" not in code:
        errors.append("Adapter must define 'get_action' function")

    # Check for common issues
    if "import os" in code:
        # OK - common for API keys
        pass

    if "os.system" in code or "subprocess" in code:
        errors.append("Adapter should not execute shell commands")

    if "__import__" in code or "eval(" in code or "exec(" in code:
        errors.append("Adapter should not use dynamic code execution")

    return errors


def main():
    """Validate submission file."""
    parser = argparse.ArgumentParser(description="Validate leaderboard submission")
    parser.add_argument(
        "submission",
        type=str,
        help="Path to submission YAML file",
    )
    args = parser.parse_args()

    print("Leaderboard Submission Validator")
    print("=" * 80)

    # Load submission file
    submission_path = Path(args.submission)

    if not submission_path.exists():
        print(f"❌ Submission file not found: {args.submission}")
        sys.exit(1)

    print(f"Validating: {submission_path}")
    print()

    try:
        with open(submission_path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load YAML: {e}")
        sys.exit(1)

    # Run validation checks
    all_errors = []

    # 1. YAML structure
    print("Checking YAML structure...")
    errors = validate_yaml_structure(data)
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✓ YAML structure valid")

    # 2. Python syntax
    if "agent" in data and "adapter" in data["agent"]:
        print("\nChecking Python syntax...")
        adapter_code = data["agent"]["adapter"]
        errors = validate_python_syntax(adapter_code)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"  ❌ {error}")
        else:
            print("  ✓ Python syntax valid")

        # 3. Adapter structure
        print("\nChecking adapter structure...")
        errors = validate_adapter_structure(adapter_code)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"  ❌ {error}")
        else:
            print("  ✓ Adapter structure valid")

    # 4. Optional fields
    print("\nChecking optional fields...")
    optional_fields = {
        "description": "Submission description",
        "paper": "Link to paper",
        "code": "Link to code repository",
    }

    for field, description in optional_fields.items():
        if field in data:
            print(f"  ✓ {description}: {data[field][:50]}...")
        else:
            print(f"  ⚠️  {description}: Not provided (optional)")

    # Summary
    print("\n" + "=" * 80)

    if all_errors:
        print("VALIDATION FAILED")
        print("=" * 80)
        print(f"\nFound {len(all_errors)} error(s):")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease fix these errors before submitting.")
        sys.exit(1)
    else:
        print("VALIDATION PASSED")
        print("=" * 80)
        print("\n✓ Submission is valid!")
        print(f"\nSubmission name: {data.get('name', 'Unknown')}")
        print(f"Model: {data.get('agent', {}).get('model', 'Unknown')}")
        print(
            f"Contact: {data.get('contact', {}).get('email', data.get('contact', {}).get('github', 'Unknown'))}"
        )
        print("\n💡 Next steps:")
        print("  1. Test your submission locally with run_evaluation.py")
        print("  2. Submit to the leaderboard (instructions at https://agentick.dev)")


if __name__ == "__main__":
    main()
