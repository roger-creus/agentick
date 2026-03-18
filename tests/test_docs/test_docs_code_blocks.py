"""Test that Python code blocks in documentation are syntactically valid."""

import ast
import re
from pathlib import Path

import pytest


def extract_python_code_blocks(md_file: Path) -> list[tuple[int, str]]:
    """
    Extract Python code blocks from a markdown file.

    Returns:
        List of (line_number, code) tuples
    """
    content = md_file.read_text()
    blocks = []

    # Find all ```python ... ``` blocks
    pattern = r"```python\n(.*?)```"
    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        # Remove common leading whitespace (for code blocks inside lists)
        lines = code.split("\n")
        if lines:
            # Find minimum indentation (excluding empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Remove that indentation from all lines
                code = "\n".join(line[min_indent:] if len(line) > min_indent else line for line in lines)

        # Find line number of this block
        line_num = content[: match.start()].count("\n") + 1
        blocks.append((line_num, code))

    return blocks


def get_all_doc_files() -> list[Path]:
    """Get all markdown files in docs/."""
    docs_dir = Path(__file__).parent.parent.parent / "docs"
    return list(docs_dir.rglob("*.md"))


@pytest.mark.parametrize("doc_file", get_all_doc_files())
def test_python_code_blocks_syntax(doc_file: Path):
    """Test that all Python code blocks in docs have valid syntax."""
    blocks = extract_python_code_blocks(doc_file)

    if not blocks:
        pytest.skip(f"No Python code blocks in {doc_file.name}")

    errors = []
    for line_num, code in blocks:
        # Skip code blocks that are intentionally incomplete (imports only, etc.)
        # or are just comments
        stripped = code.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Skip blocks that are just import statements (very common in docs)
        lines = [line for line in code.split("\n") if line.strip() and not line.strip().startswith("#")]
        if all(
            line.strip().startswith(("import ", "from "))
            or line.strip().startswith("...")
            or line.strip().startswith("# ")
            for line in lines
        ):
            continue

        # Try to parse the code
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Line {line_num}: {e}")

    if errors:
        error_msg = f"\n{doc_file.relative_to(Path.cwd())}:\n" + "\n".join(errors)
        pytest.fail(error_msg)


def test_no_removed_interface_references():
    """Test that docs don't reference removed interfaces."""
    removed_interfaces = [
        "LLMAgentInterface",
        "VLMAgentInterface",
        "RLInterface",
        "BotInterface",
        "HumanInterface",
        "LLMInterface",
        "VLMInterface",
    ]

    docs_dir = Path(__file__).parent.parent.parent / "docs"
    errors = []

    # Check all markdown files except changelog (historical references OK)
    for doc_file in docs_dir.rglob("*.md"):
        if doc_file.name == "changelog.md":
            continue

        content = doc_file.read_text()
        for interface in removed_interfaces:
            if interface in content:
                errors.append(f"{doc_file.relative_to(Path.cwd())}: contains '{interface}'")

    if errors:
        pytest.fail("Found references to removed interfaces:\n" + "\n".join(errors))


def test_no_pip_install_in_docs():
    """Test that docs don't show pip install (should use uv only)."""
    docs_dir = Path(__file__).parent.parent.parent / "docs"
    errors = []

    for doc_file in docs_dir.rglob("*.md"):
        content = doc_file.read_text()
        if "pip install" in content.lower():
            errors.append(f"{doc_file.relative_to(Path.cwd())}: contains 'pip install'")

    if errors:
        pytest.fail("Found pip install references (should use uv):\n" + "\n".join(errors))


def test_imports_use_correct_paths():
    """Test that code blocks use correct import paths."""
    docs_dir = Path(__file__).parent.parent.parent / "docs"
    errors = []

    # Incorrect patterns that should not appear
    incorrect_patterns = [
        (r"from agentick\.interfaces", "Interfaces module doesn't exist"),
        (r"from agentick\.leaderboard\.adapters", "Adapters module has been removed"),
        (r"from agentick\.leaderboard\.evaluator", "Evaluator module has been removed"),
        (r"from agentick\.leaderboard\.submission", "Submission module has been removed"),
    ]

    for doc_file in docs_dir.rglob("*.md"):
        if doc_file.name == "changelog.md":
            continue

        blocks = extract_python_code_blocks(doc_file)
        for line_num, code in blocks:
            for pattern, msg in incorrect_patterns:
                if re.search(pattern, code):
                    errors.append(f"{doc_file.relative_to(Path.cwd())}:{line_num}: {msg}")

    if errors:
        pytest.fail("Found incorrect import paths:\n" + "\n".join(errors))
