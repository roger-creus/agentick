"""Tests for CLI commands."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import yaml

import agentick


def run_cli(*args):
    """Run CLI command and return result."""
    cmd = ["uv", "run", "agentick"] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    return result


class TestCLIBasic:
    """Test basic CLI commands."""

    def test_version(self):
        """Test --version flag."""
        result = run_cli("--version")
        assert result.returncode == 0
        assert agentick.__version__ in result.stdout

    def test_help(self):
        """Test help output."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "agentick" in result.stdout.lower()
        assert "command" in result.stdout.lower()

    def test_no_command(self):
        """Test running without a command shows help."""
        result = run_cli()
        assert result.returncode == 1
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


class TestCLIListCommands:
    """Test list-tasks and list-suites commands."""

    def test_list_tasks(self):
        """Test list-tasks command."""
        result = run_cli("list-tasks")
        assert result.returncode == 0
        assert "GoToGoal-v0" in result.stdout
        assert "37 total" in result.stdout or "tasks" in result.stdout.lower()

    def test_list_suites(self):
        """Test list-suites command."""
        result = run_cli("list-suites")
        assert result.returncode == 0
        assert "agentick-full-v2" in result.stdout or "full" in result.stdout.lower()
        assert "7 total" in result.stdout or "suites" in result.stdout.lower()


class TestCLISubmitCommands:
    """Test submit init and validate commands."""

    def test_submit_init(self):
        """Test submit init command creates template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_submission.yaml"
            result = run_cli("submit", "init", "--output", str(output_file))

            assert result.returncode == 0
            assert output_file.exists()

            # Verify template structure
            with open(output_file) as f:
                data = yaml.safe_load(f)

            assert "agent_name" in data
            assert "author" in data
            assert "agent_type" in data
            assert "observation_mode" in data
            assert "config" in data
            assert "suites" in data

    def test_submit_init_no_overwrite(self):
        """Test submit init does not overwrite without --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_submission.yaml"

            # Create file first
            output_file.write_text("existing content")

            # Try to init without --force
            result = run_cli("submit", "init", "--output", str(output_file))

            assert result.returncode == 1
            assert "exists" in result.stdout.lower() or "exists" in result.stderr.lower()
            assert output_file.read_text() == "existing content"

    def test_submit_init_with_force(self):
        """Test submit init overwrites with --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_submission.yaml"

            # Create file first
            output_file.write_text("existing content")

            # Init with --force
            result = run_cli("submit", "init", "--output", str(output_file), "--force")

            assert result.returncode == 0
            assert output_file.read_text() != "existing content"

    def test_submit_validate_valid(self):
        """Test submit validate with valid submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_file = Path(tmpdir) / "submission.yaml"

            # Create valid submission
            submission_data = {
                "agent_name": "test-agent-v1",
                "author": "Test Author",
                "description": "Test agent for validation",
                "agent_type": "api",
                "observation_mode": "ascii",
                "config": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "suites": ["agentick-full-v2"],
            }

            with open(submission_file, "w") as f:
                yaml.dump(submission_data, f)

            result = run_cli("submit", "validate", str(submission_file))

            assert result.returncode == 0
            assert "valid" in result.stdout.lower()

    def test_submit_validate_missing_file(self):
        """Test submit validate with non-existent file."""
        result = run_cli("submit", "validate", "/nonexistent/file.yaml")

        # Now redirects to scripts/validate_submission.py
        assert result.returncode == 0


class TestCLIExperimentCommands:
    """Test experiment commands."""

    def test_experiment_run_help(self):
        """Test experiment run --help."""
        result = run_cli("experiment", "run", "--help")
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_experiment_run_missing_config(self):
        """Test experiment run without config fails."""
        result = run_cli("experiment", "run", "--config", "/nonexistent/config.yaml")
        # Should fail with error
        assert result.returncode != 0


class TestCLIEvaluateCommands:
    """Test evaluate and verify commands."""

    def test_evaluate_help(self):
        """Test evaluate --help."""
        result = run_cli("evaluate", "--help")
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_evaluate_missing_config(self):
        """Test evaluate with missing config fails."""
        result = run_cli(
            "evaluate",
            "--config",
            "/nonexistent/config.yaml",
        )
        # Should fail
        assert result.returncode != 0

    def test_verify_missing_result(self):
        """Test verify with missing result fails."""
        result = run_cli("verify", "--result", "/nonexistent/result.json")
        # Should fail
        assert result.returncode != 0


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def test_init_validate_workflow(self):
        """Test full init -> validate workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_file = Path(tmpdir) / "submission.yaml"

            # Step 1: Init
            result = run_cli("submit", "init", "--output", str(submission_file))
            assert result.returncode == 0

            # Step 2: Validate
            result = run_cli("submit", "validate", str(submission_file))
            assert result.returncode == 0
