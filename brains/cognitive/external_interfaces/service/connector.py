"""
External Connector Interface
============================

This module provides a lightweight interface for interacting with
external resources such as the web, local files or remote APIs. All
operations are wrapped in safety policies (allowlists, depth limits,
time budgets) so downstream components can request data without
directly touching the operating system.

Functions exported:

* ``list_tools`` – return the names of available external connectors.
* ``execute`` – run a registered tool through the safety wrapper.

The connector now registers safe external tools (file_scan, web_search)
with explicit allowlists, depth limits, time budgets and per-call
request caps. All external operations must flow through this manager so
governance checks can be applied consistently.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from api.utils import CFG
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning external API interaction patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("external_interfaces")
except Exception as e:
    print(f"[EXTERNAL_INTERFACES] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("external_interfaces")


@dataclass
class ExternalTool:
    """Definition for an external tool available to the system."""

    tool_id: str
    description: str
    allowed_operations: List[str]
    safety_policy: Dict[str, Any]


class ToolManager:
    """Registry and executor for external tools with safety enforcement."""

    def __init__(self) -> None:
        self._tools: Dict[str, ExternalTool] = {}

    def register_tool(self, tool: ExternalTool) -> None:
        if not tool or not tool.tool_id:
            return
        self._tools[tool.tool_id] = tool

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def _is_path_allowed(self, root_path: Path, allowlist: List[Path]) -> bool:
        try:
            root_resolved = root_path.resolve()
            for base in allowlist:
                try:
                    base_resolved = base.resolve()
                except Exception:
                    continue
                # Path.is_relative_to is 3.9+; use fallback for compatibility
                root_str = str(root_resolved)
                base_str = str(base_resolved)
                if root_str == base_str or root_str.startswith(base_str + os.sep):
                    return True
        except Exception:
            return False
        return False

    def _safe_file_scan(self, params: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a guarded file scan respecting allowlists and limits."""

        permission_granted = bool(params.get("permission_granted"))
        if not permission_granted:
            return {
                "ok": False,
                "error": {
                    "code": "PERMISSION_REQUIRED",
                    "message": "File scan requires explicit permission and a root folder."
                }
            }

        root_path_raw = params.get("root_path")
        if not root_path_raw:
            return {
                "ok": False,
                "error": {
                    "code": "MISSING_ROOT",
                    "message": "A root_path inside the allowlist is required for scanning."
                }
            }

        allowlist = [Path(p) for p in policy.get("allow_base_paths", []) if p]
        max_depth = int(policy.get("max_depth", 3))
        max_files = int(policy.get("max_files", 200))
        max_bytes = int(policy.get("max_bytes_per_file", 1024 * 1024))
        allowed_exts = set(policy.get("allowed_extensions", []))

        include_extensions = set(params.get("include_extensions") or [])
        exclude_patterns = params.get("exclude_patterns") or []
        requested_depth = params.get("max_depth")
        requested_max_files = params.get("max_files")
        if isinstance(requested_depth, int):
            max_depth = min(max_depth, max(1, requested_depth))
        if isinstance(requested_max_files, int):
            max_files = min(max_files, max(1, requested_max_files))

        root_path = Path(root_path_raw)
        if not self._is_path_allowed(root_path, allowlist):
            return {
                "ok": False,
                "error": {
                    "code": "PATH_NOT_ALLOWED",
                    "message": "Requested root_path is outside the allowed sandbox."
                }
            }

        files: List[Dict[str, Any]] = []
        errors: List[str] = []
        truncated = False

        try:
            for current_path, dirs, filenames in os.walk(root_path):
                rel = Path(current_path).resolve().relative_to(root_path.resolve())
                depth = len(rel.parts)
                if depth >= max_depth:
                    dirs[:] = []  # Do not descend further

                for fname in filenames:
                    if len(files) >= max_files:
                        truncated = True
                        break
                    fpath = Path(current_path) / fname

                    try:
                        rel_path = str(fpath.resolve().relative_to(root_path.resolve()))
                    except Exception:
                        rel_path = str(fpath)

                    # Simple exclude pattern matching
                    try:
                        skip = any(Path(rel_path).match(pat) for pat in exclude_patterns)
                    except Exception:
                        skip = False
                    if skip:
                        continue

                    try:
                        stat = fpath.stat()
                    except Exception as e:
                        errors.append(f"stat_failed:{rel_path}:{str(e)[:80]}")
                        continue

                    record: Dict[str, Any] = {
                        "path": rel_path,
                        "size": int(stat.st_size),
                        "mtime": int(stat.st_mtime),
                    }

                    ext = fpath.suffix.lower()
                    include_check = (not include_extensions) or (ext in include_extensions)
                    allow_check = (not allowed_exts) or (ext in allowed_exts)

                    if include_check and allow_check and stat.st_size <= max_bytes and ext:
                        try:
                            with fpath.open("r", encoding="utf-8", errors="ignore") as fh:
                                snippet = fh.read(min(max_bytes, 2048))
                            record["snippet"] = snippet
                        except Exception:
                            errors.append(f"read_failed:{rel_path}")

                    files.append(record)

                if truncated:
                    break
        except Exception as e:
            errors.append(str(e))

        # Log the call for governance traceability
        try:
            _memory.store(
                content={
                    "op": "file_scan",
                    "root_path": str(root_path),
                    "count": len(files),
                    "truncated": truncated,
                    "errors": errors[:3],
                },
                metadata={"kind": "external_tool", "tool": "file_scan", "confidence": 0.9}
            )
        except Exception:
            pass

        return {
            "ok": True,
            "files": files,
            "truncated": truncated,
            "errors": errors,
        }

    def _budgeted_web_search(self, params: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        """Run web search with an enforced time budget and request limits."""

        try:
            from brains.external_interfaces import web_client
        except Exception as e:
            return {"ok": False, "error": {"code": "WEB_CLIENT_UNAVAILABLE", "message": str(e)}}

        time_budget = int(params.get("time_budget_seconds") or policy.get("time_budget_seconds") or 0)
        time_budget = max(0, time_budget)
        started_at = params.get("start_time") or time.monotonic()
        elapsed = time.monotonic() - float(started_at)
        if time_budget and elapsed > time_budget:
            return {"ok": False, "budget_exhausted": True, "error": {"code": "BUDGET_EXHAUSTED", "message": "Time budget exceeded before search."}}

        max_requests = int(params.get("max_requests") or policy.get("max_requests") or 3)
        requests_made = int(params.get("requests_made") or 0)
        if requests_made >= max_requests:
            return {"ok": False, "budget_exhausted": True, "error": {"code": "REQUEST_LIMIT", "message": "Max web requests reached."}}

        query = str(params.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": {"code": "MISSING_QUERY", "message": "query is required"}}

        search_res = web_client.search(query, max_results=int(params.get("max_results") or 3))

        # Record audit trail
        try:
            _memory.store(
                content={"op": "web_search", "query": query, "elapsed": elapsed},
                metadata={"kind": "external_tool", "tool": "web_search", "confidence": 0.7}
            )
        except Exception:
            pass

        return {"ok": True, **search_res, "budget_exhausted": False, "time_budget_seconds": time_budget}

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(tool_name)
        if not tool:
            return {
                "ok": False,
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": f"Tool '{tool_name}' is not registered"
                }
            }

        policy = tool.safety_policy or {}
        if tool_name == "file_scan":
            return self._safe_file_scan(params or {}, policy)
        if tool_name == "web_search":
            return self._budgeted_web_search(params or {}, policy)

        return {
            "ok": False,
            "error": {
                "code": "UNSUPPORTED_TOOL",
                "message": f"Tool '{tool_name}' has no executor"
            }
        }


_tool_manager = ToolManager()


def _register_default_tools() -> None:
    """Register built-in external tools with safety policies."""

    sandbox_root = Path(__file__).resolve().parents[4] / "sandbox_workspace"
    file_scan_policy = {
        "allow_base_paths": [sandbox_root],
        "max_depth": 3,
        "max_files": 200,
        "max_bytes_per_file": 1_000_000,
        "allowed_extensions": [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"],
    }
    web_search_policy = {
        "time_budget_seconds": int(CFG.get("WEB_RESEARCH_MAX_SECONDS", 1200)),
        "max_requests": int(CFG.get("WEB_RESEARCH_MAX_REQUESTS", 10)),
    }

    _tool_manager.register_tool(
        ExternalTool(
            tool_id="file_scan",
            description="Safely list and sample text files under an approved root path.",
            allowed_operations=["list", "metadata", "read_text_snippet"],
            safety_policy=file_scan_policy,
        )
    )
    _tool_manager.register_tool(
        ExternalTool(
            tool_id="web_search",
            description="Budgeted web search with time and request limits.",
            allowed_operations=["search"],
            safety_policy=web_search_policy,
        )
    )


_register_default_tools()


def get_tool_manager() -> ToolManager:
    return _tool_manager


def list_tools() -> List[str]:
    """Return the names of available external tools."""

    try:
        return get_tool_manager().list_tools()
    except Exception:
        return []


def execute(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an external operation using the specified tool."""
    # Check for learned API patterns first
    learned_pattern = None
    if _teacher_helper and _memory and tool_name:
        try:
            learned_patterns = _memory.retrieve(
                query=f"api pattern: {tool_name}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, dict) and "tool" in content:
                        learned_pattern = content
                        print(f"[EXTERNAL_INTERFACES] Using learned API pattern from Teacher")
                        break
        except Exception:
            pass

    if learned_pattern:
        return learned_pattern

    manager = get_tool_manager()
    result = manager.execute_tool(tool_name, params or {})

    if not result.get("ok") and _teacher_helper and tool_name:
        try:
            print(f"[EXTERNAL_INTERFACES] No learned API pattern for {tool_name}, calling Teacher...")
            teacher_result = _teacher_helper.maybe_call_teacher(
                question=f"What API pattern should I use for tool '{tool_name}' with params: {params}?",
                context={
                    "tool_name": tool_name,
                    "params": params,
                    "current_result": result
                },
                check_memory_first=True
            )

            if teacher_result and teacher_result.get("answer"):
                patterns_stored = teacher_result.get("patterns_stored", 0)
                print(f"[EXTERNAL_INTERFACES] Learned from Teacher: {patterns_stored} API patterns stored")
        except Exception as e:
            print(f"[EXTERNAL_INTERFACES] Teacher call failed: {str(e)[:100]}")

    return result



def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    External Interfaces service API.

    Supported operations:
    - LIST_TOOLS: List available external tools
    - EXECUTE: Execute an external tool with safety controls
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "LIST_TOOLS":
        try:
            tools = list_tools()
            return {
                "ok": True,
                "payload": {
                    "tools": tools,
                    "count": len(tools)
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "LIST_FAILED",
                    "message": str(e)
                }
            }

    if op == "EXECUTE":
        try:
            tool_name = payload.get("tool", "")
            params = payload.get("params", {})

            if not tool_name:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_TOOL",
                        "message": "tool parameter is required"
                    }
                }

            result = execute(tool_name, params)
            return {
                "ok": bool(result.get("ok", True)),
                "payload": result
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "EXECUTE_FAILED",
                    "message": str(e)
                }
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "payload": {
                "status": "operational",
                "note": "External tools are routed through the safety manager"
            }
        }

    return {
        "ok": False,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": op
        }
    }


# Standard service contract: handle is the entry point
service_api = handle
