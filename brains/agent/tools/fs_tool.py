"""
Filesystem utility functions for Agent Executor
---------------------------------------------

This module provides simple helpers to read files, generate diffs and
apply new content with atomic backups.  The agent executor delegates
file operations through these functions to ensure consistency across
executions.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple

from brains.maven_paths import get_reports_path

# The agent tools are located under ``brains/agent/tools``.  Use the shared
# reports helper to keep backups confined to the Maven project root regardless
# of file relocation.
BACKUP_ROOT = get_reports_path("agent", "backups")
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)


def read(path: str) -> str:
    """Read and return the contents of the given file.  Returns an empty string on error."""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def diff(path: str, new_text: str) -> str:
    """Compute a unified diff between the current contents of the file and the provided new text."""
    import difflib
    old_text = read(path)
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=f"{path}.new", lineterm="")
    )


def apply(path: str, new_text: str) -> Tuple[str, str]:
    """Apply new_text to the file at the given path.  Returns a tuple (status, backup_path)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    backup_path = ""
    # Create backup if file exists
    if p.exists():
        ts = int(os.times().elapsed * 1000)
        backup = BACKUP_ROOT / f"{p.name}.{ts}.bak"
        try:
            shutil.copy2(p, backup)
            backup_path = str(backup)
        except Exception:
            backup_path = ""
    # Write new content atomically
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    os.replace(tmp_path, p)
    return ("applied", backup_path)


# ============================================================================
# Service API (Standard Tool Interface)
# ============================================================================

from typing import Dict, Any


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for the fs_tool.

    Operations:
    - READ: Read file contents
    - DIFF: Generate diff between current file and new content
    - APPLY: Apply new content to file (with backup)
    - HEALTH: Health check

    Args:
        msg: Dict with "op" and optional "payload"

    Returns:
        Dict with "ok", "payload" or "error"
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "FS_TOOL"

    if op == "READ":
        path = payload.get("path", "")
        if not path:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_PATH", "message": "Path is required"}
            }

        try:
            content = read(path)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "content": content,
                    "path": path,
                    "size": len(content),
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "READ_ERROR", "message": str(e)}
            }

    if op == "DIFF":
        path = payload.get("path", "")
        new_text = payload.get("new_text", "")

        if not path:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_PATH", "message": "Path is required"}
            }

        try:
            diff_result = diff(path, new_text)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "diff": diff_result,
                    "path": path,
                    "has_changes": bool(diff_result.strip()),
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "DIFF_ERROR", "message": str(e)}
            }

    if op == "APPLY":
        path = payload.get("path", "")
        new_text = payload.get("new_text", "")

        if not path:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_PATH", "message": "Path is required"}
            }

        try:
            status, backup_path = apply(path, new_text)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "status": status,
                    "path": path,
                    "backup_path": backup_path,
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "APPLY_ERROR", "message": str(e)}
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service": "fs_tool",
                "capability": "filesystem",
                "description": "File read, diff, and apply operations",
                "backup_root": str(BACKUP_ROOT),
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"}
    }


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "fs_tool"
TOOL_CAPABILITY = "filesystem"
TOOL_DESCRIPTION = "File read, diff, and apply operations"
TOOL_OPERATIONS = ["READ", "DIFF", "APPLY", "HEALTH"]


def is_available() -> bool:
    """Check if the fs tool is available (always True)."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
        "requires_execution": False,
        "module": "brains.agent.tools.fs_tool",
    }