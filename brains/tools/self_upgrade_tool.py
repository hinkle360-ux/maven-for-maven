"""
Self-Upgrade Tool
=================

Enables Maven to write code and commit changes to its own codebase.
This allows the system to evolve and add new capabilities autonomously.

SECURITY: This tool should only be invoked by trusted internal processes.
All writes are logged and version-controlled via git.

Usage:
    from brains.tools.self_upgrade_tool import write_and_commit_module

    result = write_and_commit_module(
        module_name="brains/tools/new_tool.py",
        code="def new_capability(): ...",
        commit_msg="Self-upgrade: add new tool for X"
    )
"""

from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory (maven2_fix)."""
    # Start from this file and navigate up to find project root
    current = Path(__file__).resolve()

    # Navigate up until we find the maven2_fix directory or hit filesystem root
    for _ in range(10):  # Safety limit
        if current.name == "maven2_fix":
            return current
        if current.parent == current:
            break
        current = current.parent

    # Fallback: try common paths
    possible_roots = [
        Path("/home/user/maven/maven2_fix"),
        Path.cwd() / "maven2_fix",
        Path.cwd(),
    ]

    for root in possible_roots:
        if (root / "brains").exists():
            return root

    raise RuntimeError("Could not locate project root")


def _run_git_command(args: list[str], cwd: Path) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except Exception as e:
        return False, f"Git command failed: {str(e)}"


def validate_module_path(module_name: str) -> tuple[bool, str]:
    """
    Validate that the module path is safe to write to.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Normalize path
    normalized = module_name.replace("\\", "/")

    # Must be a Python file
    if not normalized.endswith(".py"):
        return False, "Module must be a .py file"

    # Prevent path traversal
    if ".." in normalized:
        return False, "Path traversal not allowed"

    # Must be within allowed directories
    allowed_prefixes = [
        "brains/",
        "optional/",
        "tools/",
    ]

    if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        return False, f"Module must be in one of: {allowed_prefixes}"

    # Don't allow overwriting critical files
    protected_files = [
        "brains/cognitive/integrator/service/integrator_brain.py",
        "brains/cognitive/teacher/service/teacher_helper.py",
        "__init__.py",
    ]

    if any(normalized.endswith(pf) for pf in protected_files):
        return False, f"Cannot overwrite protected file: {normalized}"

    return True, ""


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate that the code is syntactically correct Python.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        compile(code, "<self_upgrade>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e.msg} at line {e.lineno}"


def write_and_commit_module(
    module_name: str,
    code: str,
    commit_msg: str = "Self-upgrade: add new capability",
    author_name: str = "Maven Self-Upgrade",
    author_email: str = "maven@localhost",
    push: bool = False,
) -> Dict[str, Any]:
    """
    Write a Python module and commit it to git.

    This is the main entry point for self-upgrade operations.

    Args:
        module_name: Relative path to the module (e.g., "brains/tools/new_tool.py")
        code: The Python code to write
        commit_msg: Git commit message
        author_name: Author name for the commit
        author_email: Author email for the commit
        push: Whether to push to remote after commit

    Returns:
        Dict with:
        - success: bool
        - file_path: str (absolute path to written file)
        - commit_hash: str (if commit succeeded)
        - error: str (if failed)
    """
    result = {
        "success": False,
        "file_path": None,
        "commit_hash": None,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    # Step 1: Validate module path
    is_valid, error = validate_module_path(module_name)
    if not is_valid:
        result["error"] = f"Invalid module path: {error}"
        logger.error(result["error"])
        return result

    # Step 2: Validate code syntax
    is_valid, error = validate_code(code)
    if not is_valid:
        result["error"] = f"Invalid code: {error}"
        logger.error(result["error"])
        return result

    # Step 3: Get project root
    try:
        project_root = _get_project_root()
    except RuntimeError as e:
        result["error"] = str(e)
        logger.error(result["error"])
        return result

    # Step 4: Calculate file path
    file_path = project_root / module_name
    result["file_path"] = str(file_path)

    # Step 5: Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 6: Write the file
    try:
        # Add header comment
        header = f'''"""
Auto-generated module by Maven Self-Upgrade
Generated: {datetime.now().isoformat()}

{commit_msg}
"""

'''
        # Only add header if code doesn't already have a docstring
        if not code.strip().startswith('"""') and not code.strip().startswith("'''"):
            full_code = header + code
        else:
            full_code = code

        file_path.write_text(full_code)
        logger.info(f"[SELF_UPGRADE] Wrote module: {file_path}")
    except Exception as e:
        result["error"] = f"Failed to write file: {str(e)}"
        logger.error(result["error"])
        return result

    # Step 7: Git add
    success, output = _run_git_command(["add", str(file_path)], project_root)
    if not success:
        result["error"] = f"Git add failed: {output}"
        logger.error(result["error"])
        return result

    # Step 8: Git commit
    commit_args = [
        "commit",
        "-m", commit_msg,
        "--author", f"{author_name} <{author_email}>",
    ]
    success, output = _run_git_command(commit_args, project_root)
    if not success:
        # Check if it's just "nothing to commit"
        if "nothing to commit" in output.lower():
            result["error"] = "No changes to commit"
            logger.warning(result["error"])
            return result
        result["error"] = f"Git commit failed: {output}"
        logger.error(result["error"])
        return result

    # Step 9: Get commit hash
    success, commit_hash = _run_git_command(["rev-parse", "HEAD"], project_root)
    if success:
        result["commit_hash"] = commit_hash[:8]

    # Step 10: Optionally push
    if push:
        success, output = _run_git_command(["push"], project_root)
        if not success:
            result["error"] = f"Git push failed (commit succeeded): {output}"
            logger.warning(result["error"])
            # Don't return failure - commit succeeded

    result["success"] = True
    logger.info(f"[SELF_UPGRADE] Committed: {result['commit_hash']} - {commit_msg}")

    return result


def create_tool(
    tool_name: str,
    description: str,
    operations: list[str],
    implementation: str,
) -> Dict[str, Any]:
    """
    Convenience function to create a new tool following Maven's conventions.

    Args:
        tool_name: Name of the tool (e.g., "new_tool")
        description: Human-readable description
        operations: List of operation names (e.g., ["PROCESS", "HEALTH"])
        implementation: The Python implementation code

    Returns:
        Result from write_and_commit_module
    """
    ops_str = ", ".join(f'"{op}"' for op in operations)

    template = f'''"""
{tool_name.replace("_", " ").title()} Tool
{"=" * (len(tool_name) + 5)}

{description}

Operations: {", ".join(operations)}
"""

from __future__ import annotations
from typing import Dict, Any

TOOL_NAME = "{tool_name}"
TOOL_DESCRIPTION = "{description}"
TOOL_OPERATIONS = [{ops_str}]


{implementation}


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Service API entry point."""
    op = (msg or {{}}).get("op", "").upper()
    payload = msg.get("payload") or {{}}
    mid = msg.get("mid") or "{tool_name.upper()}"

    if op == "HEALTH":
        return {{
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {{
                "status": "operational",
                "service": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
            }}
        }}

    # Dispatch to implementation
    try:
        result = handle_operation(op, payload)
        return {{
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": result,
        }}
    except Exception as e:
        return {{
            "ok": False,
            "op": op,
            "mid": mid,
            "error": {{"code": "OPERATION_ERROR", "message": str(e)}},
        }}


handle = service_api


def is_available() -> bool:
    """Check if tool is available."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata."""
    return {{
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
    }}
'''

    module_path = f"brains/tools/{tool_name}.py"
    commit_msg = f"Self-upgrade: add {tool_name} tool - {description}"

    return write_and_commit_module(module_path, template, commit_msg)


# =============================================================================
# SERVICE API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for self-upgrade tool.

    Operations:
    - WRITE_MODULE: Write and commit a new module
    - CREATE_TOOL: Create a new tool using template
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "SELF_UPGRADE"

    if op == "WRITE_MODULE":
        module_name = payload.get("module_name")
        code = payload.get("code")
        commit_msg = payload.get("commit_msg", "Self-upgrade: add new capability")
        push = payload.get("push", False)

        if not module_name or not code:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_PARAMS", "message": "module_name and code required"},
            }

        result = write_and_commit_module(module_name, code, commit_msg, push=push)
        return {
            "ok": result["success"],
            "op": op,
            "mid": mid,
            "payload": result if result["success"] else None,
            "error": {"code": "WRITE_FAILED", "message": result["error"]} if not result["success"] else None,
        }

    if op == "CREATE_TOOL":
        tool_name = payload.get("tool_name")
        description = payload.get("description", "")
        operations = payload.get("operations", ["PROCESS", "HEALTH"])
        implementation = payload.get("implementation", "def handle_operation(op, payload):\n    pass")

        if not tool_name:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_PARAMS", "message": "tool_name required"},
            }

        result = create_tool(tool_name, description, operations, implementation)
        return {
            "ok": result["success"],
            "op": op,
            "mid": mid,
            "payload": result if result["success"] else None,
            "error": {"code": "CREATE_FAILED", "message": result["error"]} if not result["success"] else None,
        }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service": "self_upgrade",
                "capability": "code_generation",
                "description": "Write and commit Python modules",
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"},
    }


handle = service_api


# =============================================================================
# TOOL METADATA
# =============================================================================

TOOL_NAME = "self_upgrade"
TOOL_CAPABILITY = "code_generation"
TOOL_DESCRIPTION = "Write and commit Python modules to the codebase"
TOOL_OPERATIONS = ["WRITE_MODULE", "CREATE_TOOL", "HEALTH"]


def is_available() -> bool:
    """Check if the self-upgrade tool is available."""
    try:
        project_root = _get_project_root()
        # Check if git is available
        success, _ = _run_git_command(["status"], project_root)
        return success
    except Exception:
        return False


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": is_available(),
        "module": "brains.tools.self_upgrade_tool",
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing self_upgrade_tool:")
    print("-" * 40)

    # Test HEALTH
    response = service_api({"op": "HEALTH"})
    print(f"HEALTH response: {response}")

    # Test validation
    print(f"\nValidation tests:")
    print(f"  Valid path: {validate_module_path('brains/tools/test.py')}")
    print(f"  Invalid path: {validate_module_path('../../../etc/passwd')}")
    print(f"  Valid code: {validate_code('def foo(): pass')}")
    print(f"  Invalid code: {validate_code('def foo(')}")
