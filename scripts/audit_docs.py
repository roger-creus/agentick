#!/usr/bin/env python3
"""Documentation audit script for Phase 5.1.

Finds all gaps in documentation across the agentick codebase:
- Missing module docstrings
- Missing class docstrings
- Missing method docstrings
- Modules not mentioned in docs/
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple


def has_docstring(node) -> bool:
    """Check if an AST node has a docstring."""
    return (
        isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node) is not None
    )


def is_public(name: str) -> bool:
    """Check if a name is public (doesn't start with _)."""
    return not name.startswith("_")


def audit_file(filepath: Path) -> dict:
    """Audit a single Python file for documentation gaps."""
    gaps = {
        "module_docstring": False,
        "classes_missing_docstring": [],
        "methods_missing_docstring": [],
        "functions_missing_docstring": [],
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, str(filepath))

        # Check module docstring
        gaps["module_docstring"] = has_docstring(tree)

        # Check classes and methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and is_public(node.name):
                if not has_docstring(node):
                    gaps["classes_missing_docstring"].append(node.name)

                # Check methods within the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if is_public(item.name) and not has_docstring(item):
                            gaps["methods_missing_docstring"].append(f"{node.name}.{item.name}")

            # Check module-level functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only check if it's a direct child of the module
                if is_public(node.name):
                    for top_node in tree.body:
                        if top_node is node and not has_docstring(node):
                            gaps["functions_missing_docstring"].append(node.name)
                            break

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return gaps


def is_documented_in_docs(module_path: str, docs_dir: Path) -> bool:
    """Check if a module is mentioned in any documentation file."""
    # Convert module path to importable name
    module_name = module_path.replace("/", ".").replace(".py", "")

    # Also check for just the module base name
    base_name = Path(module_path).stem

    # Search in all markdown files
    for doc_file in docs_dir.rglob("*.md"):
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()
                if module_name in content or base_name in content:
                    return True
        except Exception:
            continue

    return False


def main():
    """Run the full documentation audit."""
    root = Path("/home/roger/Desktop/agentick-prime")
    agentick_dir = root / "agentick"
    docs_dir = root / "docs"

    audit_results = []

    # Find all Python files
    py_files = sorted(agentick_dir.rglob("*.py"))

    print(f"Auditing {len(py_files)} Python files...\n")

    for py_file in py_files:
        rel_path = py_file.relative_to(root)
        gaps = audit_file(py_file)

        # Check if documented in docs/
        in_docs = is_documented_in_docs(str(rel_path), docs_dir)

        # Only report if there are gaps
        has_gaps = (
            not gaps["module_docstring"]
            or gaps["classes_missing_docstring"]
            or gaps["methods_missing_docstring"]
            or gaps["functions_missing_docstring"]
            or not in_docs
        )

        if has_gaps:
            audit_results.append({
                "file": str(rel_path),
                "gaps": gaps,
                "in_docs": in_docs,
            })

    # Write audit report
    report_path = root / "PHASE5_AUDIT.md"
    with open(report_path, "w") as f:
        f.write("# PHASE 5.1: Documentation Audit Report\n\n")
        f.write(f"**Audit Date:** 2026-02-12\n\n")
        f.write(f"**Total Python Files:** {len(py_files)}\n")
        f.write(f"**Files with Documentation Gaps:** {len(audit_results)}\n\n")
        f.write("---\n\n")

        # Summary statistics
        total_missing_module_docs = sum(1 for r in audit_results if not r["gaps"]["module_docstring"])
        total_missing_class_docs = sum(len(r["gaps"]["classes_missing_docstring"]) for r in audit_results)
        total_missing_method_docs = sum(len(r["gaps"]["methods_missing_docstring"]) for r in audit_results)
        total_missing_function_docs = sum(len(r["gaps"]["functions_missing_docstring"]) for r in audit_results)
        total_not_in_docs = sum(1 for r in audit_results if not r["in_docs"])

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Missing Module Docstrings:** {total_missing_module_docs}\n")
        f.write(f"- **Missing Class Docstrings:** {total_missing_class_docs}\n")
        f.write(f"- **Missing Method Docstrings:** {total_missing_method_docs}\n")
        f.write(f"- **Missing Function Docstrings:** {total_missing_function_docs}\n")
        f.write(f"- **Not Mentioned in docs/:** {total_not_in_docs}\n\n")
        f.write("---\n\n")

        # Detailed gaps by file
        f.write("## Detailed Gaps by File\n\n")

        for result in audit_results:
            f.write(f"### `{result['file']}`\n\n")

            gaps = result["gaps"]

            if not gaps["module_docstring"]:
                f.write("- ❌ **Missing module docstring**\n")

            if gaps["classes_missing_docstring"]:
                f.write(f"- ❌ **Missing class docstrings:** {', '.join(gaps['classes_missing_docstring'])}\n")

            if gaps["methods_missing_docstring"]:
                f.write(f"- ❌ **Missing method docstrings:** {', '.join(gaps['methods_missing_docstring'])}\n")

            if gaps["functions_missing_docstring"]:
                f.write(f"- ❌ **Missing function docstrings:** {', '.join(gaps['functions_missing_docstring'])}\n")

            if not result["in_docs"]:
                f.write("- ⚠️  **Not mentioned in docs/**\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. Add module docstrings to all files\n")
        f.write("2. Add class docstrings using Google style\n")
        f.write("3. Add method/function docstrings using Google style\n")
        f.write("4. Update docs/ to mention all modules\n")
        f.write("5. Re-run audit to verify all gaps are fixed\n")

    print(f"Audit complete! Report written to {report_path}")
    print(f"\nSummary:")
    print(f"  - Files with gaps: {len(audit_results)}")
    print(f"  - Missing module docstrings: {total_missing_module_docs}")
    print(f"  - Missing class docstrings: {total_missing_class_docs}")
    print(f"  - Missing method docstrings: {total_missing_method_docs}")
    print(f"  - Missing function docstrings: {total_missing_function_docs}")
    print(f"  - Not in docs/: {total_not_in_docs}")


if __name__ == "__main__":
    main()
