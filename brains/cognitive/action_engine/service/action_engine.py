"""
Action Engine Service

Concrete implementation of tool-like actions for Maven:

- Filesystem operations (read, write, list, mkdir).
- Git operations (status, diff, commit) via tool interface.
- Python sandbox execution via tool interface.
- Hot module reload.

All actions are:

- Confined to a configured root_dir.
- Gated by the central execution_guard.
- Logged and returned as structured dicts.

IMPORTANT: This module uses tool interfaces for subprocess/git operations.
No direct subprocess calls are made. External I/O is delegated to host tools.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from brains.memory.brain_memory import BrainMemory
from brains.tools.execution_guard import require_execution_allowed
from brains.tools_api import (
    ToolRegistry,
    GitTool,
    PythonSandboxTool,
    ShellTool,
    NullGitTool,
    NullPythonSandboxTool,
    NullShellTool,
)


# Global tool registry - set by host runtime
_tool_registry: Optional[ToolRegistry] = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry (called by host runtime)."""
    global _tool_registry
    _tool_registry = registry


def _get_git_tool() -> GitTool:
    """Get git tool from registry."""
    if _tool_registry and _tool_registry.git:
        return _tool_registry.git
    return NullGitTool()


def _get_sandbox_tool() -> PythonSandboxTool:
    """Get Python sandbox tool from registry."""
    if _tool_registry and _tool_registry.python_sandbox:
        return _tool_registry.python_sandbox
    return NullPythonSandboxTool()


def _get_shell_tool() -> ShellTool:
    """Get shell tool from registry."""
    if _tool_registry and _tool_registry.shell:
        return _tool_registry.shell
    return NullShellTool()


logger = logging.getLogger(__name__)

# Tool-layer helpers (legacy integration)
try:
    from brains.tools import fs_scan_tool, fs_write_tool, git_tool, reload_tool
except Exception as e:
    print(f"[ACTION_ENGINE] Tool layer unavailable: {e}")
    fs_scan_tool = None  # type: ignore
    fs_write_tool = None  # type: ignore
    git_tool = None  # type: ignore
    reload_tool = None  # type: ignore

# Teacher integration for learning action execution and coordination patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("action_engine")
except Exception as e:
    print(f"[ACTION_ENGINE] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Agency tool execution
try:
    from brains.cognitive.integrator.service.integrator_brain import get_agency_tool_info
    from brains.cognitive.integrator.agency_executor import execute_agency_tool, format_agency_response
    _agency_tools_available = True
except Exception as e:
    print(f"[ACTION_ENGINE] Agency tools not available: {e}")
    _agency_tools_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("action_engine")


@dataclass
class ActionResult:
    ok: bool
    action: str
    summary: str
    details: Dict[str, Any]


class ActionEngineService:
    """
    Low-level action executor, confined to a Maven root directory.
    All operations are gated by execution_guard.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir).resolve()

    # ----------------- helpers -----------------

    def _normalize_path(self, rel: str) -> Path:
        """
        Ensure rel is inside root_dir. Raises PermissionError if it escapes.
        """
        p = (self.root_dir / rel).resolve()
        if not str(p).startswith(str(self.root_dir)):
            raise PermissionError(f"Path escapes Maven root: {p}")
        return p

    def _result(self, ok: bool, action: str, summary: str, **details: Any) -> Dict[str, Any]:
        return {
            "ok": ok,
            "action": action,
            "summary": summary,
            "details": details,
        }

    # ----------------- filesystem -----------------

    def list_python_files(self, max_files: int = 500, include_tests: bool = True) -> Dict[str, Any]:
        require_execution_allowed(
            "list_python_files",
            risk_level="LOW",
            intent="introspect Maven source tree",
            write_required=False,
        )

        files: List[str] = []
        for p in self.root_dir.rglob("*.py"):
            name_lower = p.name.lower()
            if not include_tests and ("test" in name_lower or name_lower.startswith("test_")):
                continue
            try:
                rel = str(p.relative_to(self.root_dir))
            except Exception:
                continue
            files.append(rel)
            if len(files) >= max_files:
                break

        return self._result(
            ok=True,
            action="list_python_files",
            summary=f"Found {len(files)} Python files",
            files=files,
            root=str(self.root_dir),
        )

    def read_text_file(self, path: str, max_chars: int = 100_000) -> Dict[str, Any]:
        require_execution_allowed(
            "read_text_file",
            risk_level="LOW",
            intent=f"read file {path}",
            write_required=False,
        )

        p = self._normalize_path(path)
        if not p.exists():
            return self._result(False, "read_text_file", "File not found", path=path)

        text = p.read_text(encoding="utf-8")[:max_chars]
        truncated = p.stat().st_size > len(text.encode("utf-8"))
        return self._result(
            True,
            "read_text_file",
            f"Read {len(text)} chars from {path}",
            path=path,
            content=text,
            truncated=truncated,
        )

    def read_bytes_file(self, path: str, max_bytes: int = 100_000) -> Dict[str, Any]:
        require_execution_allowed(
            "read_bytes_file",
            risk_level="LOW",
            intent=f"read file {path} as bytes",
            write_required=False,
        )

        p = self._normalize_path(path)
        if not p.exists():
            return self._result(False, "read_bytes_file", "File not found", path=path)

        data = p.read_bytes()[:max_bytes]
        truncated = p.stat().st_size > len(data)
        return self._result(
            True,
            "read_bytes_file",
            f"Read {len(data)} bytes from {path}",
            path=path,
            content_bytes=data,
            truncated=truncated,
        )

    def mkdir(self, path: str, exist_ok: bool = True) -> Dict[str, Any]:
        require_execution_allowed(
            "mkdir",
            risk_level="MEDIUM",
            intent=f"create directory {path}",
            write_required=True,
        )

        p = self._normalize_path(path)
        try:
            p.mkdir(parents=True, exist_ok=exist_ok)
        except Exception as e:
            logger.exception("mkdir failed for %s", p)
            return self._result(False, "mkdir", "Failed to create directory", path=path, error=str(e))

        return self._result(True, "mkdir", f"Created directory {path}", path=path)

    def write_text_file(self, path: str, content: str, backup: bool = True) -> Dict[str, Any]:
        require_execution_allowed(
            "write_text_file",
            risk_level="MEDIUM",
            intent=f"overwrite file {path}",
            write_required=True,
        )

        p = self._normalize_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        backup_path: Optional[str] = None
        if backup and p.exists():
            backup_dir = self.root_dir / ".maven" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / (p.name + ".bak")
            backup_file.write_bytes(p.read_bytes())
            backup_path = str(backup_file)

        p.write_text(content, encoding="utf-8")
        return self._result(
            True,
            "write_text_file",
            f"Wrote {len(content)} chars to {path}",
            path=path,
            backup_path=backup_path,
        )

    def append_text_file(self, path: str, content: str, backup: bool = True) -> Dict[str, Any]:
        require_execution_allowed(
            "append_text_file",
            risk_level="MEDIUM",
            intent=f"append to file {path}",
            write_required=True,
        )

        p = self._normalize_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        backup_path: Optional[str] = None
        if backup and p.exists():
            backup_dir = self.root_dir / ".maven" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / (p.name + ".bak.append")
            backup_file.write_bytes(p.read_bytes())
            backup_path = str(backup_file)

        with p.open("a", encoding="utf-8") as f:
            f.write(content)

        return self._result(
            True,
            "append_text_file",
            f"Appended {len(content)} chars to {path}",
            path=path,
            backup_path=backup_path,
        )

    # ----------------- python sandbox -----------------

    def run_python_sandbox(self, code: str, timeout_ms: int = 3_000) -> Dict[str, Any]:
        """
        Execute Python code via the host-provided sandbox tool.
        This is a real execution, so HIGH risk.
        """
        require_execution_allowed(
            "run_python_sandbox",
            risk_level="HIGH",
            intent="execute Python code in sandbox subprocess",
            write_required=True,
        )

        sandbox = _get_sandbox_tool()
        result = sandbox.execute(code, timeout_ms=timeout_ms, cwd=str(self.root_dir))

        if result.timed_out:
            return self._result(
                False,
                "run_python_sandbox",
                "Execution timed out",
                timeout_ms=timeout_ms,
            )

        if result.error and not result.ok:
            logger.error("Sandbox execution failed: %s", result.error)
            return self._result(
                False,
                "run_python_sandbox",
                "Execution failed",
                error=result.error,
            )

        return self._result(
            result.ok,
            "run_python_sandbox",
            "Execution completed" if result.ok else f"Execution failed with code {result.returncode}",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    # ----------------- Shell command -----------------

    def run_shell(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a shell command via the host-provided shell tool.
        This is a real execution, so HIGH/CRITICAL risk.
        """
        require_execution_allowed(
            "run_shell",
            risk_level="CRITICAL",
            intent=f"execute shell command: {command[:50]}...",
            write_required=True,
        )

        shell = _get_shell_tool()
        result = shell.run(command, timeout=timeout, check_policy=True)

        if result.status == "denied":
            return self._result(
                False,
                "run_shell",
                "Command denied by policy",
                command=command,
                error=result.error,
            )

        if result.status == "timeout":
            return self._result(
                False,
                "run_shell",
                "Command timed out",
                command=command,
                timeout=timeout,
            )

        if result.status == "error":
            return self._result(
                False,
                "run_shell",
                "Command execution failed",
                command=command,
                error=result.error,
            )

        return self._result(
            result.exit_code == 0,
            "run_shell",
            "Command completed" if result.exit_code == 0 else f"Command failed with code {result.exit_code}",
            command=command,
            returncode=result.exit_code or 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    # ----------------- Git -----------------

    def git_status(self) -> Dict[str, Any]:
        require_execution_allowed(
            "git_status",
            risk_level="LOW",
            intent="inspect git status",
            write_required=False,
        )

        git = _get_git_tool()
        try:
            output = git.status(short=True)
            return self._result(
                True,
                "git_status",
                "OK",
                stdout=output,
                stderr="",
                returncode=0,
            )
        except Exception as e:
            return self._result(
                False,
                "git_status",
                "git status failed",
                stdout="",
                stderr=str(e),
                returncode=1,
            )

    def git_diff(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        require_execution_allowed(
            "git_diff",
            risk_level="LOW",
            intent="inspect git diff",
            write_required=False,
        )

        git = _get_git_tool()
        try:
            # Get diff for specified path or all
            file_path = paths[0] if paths else None
            output = git.diff(file_path=file_path)
            return self._result(
                True,
                "git_diff",
                "OK",
                paths=paths or [],
                stdout=output,
                stderr="",
                returncode=0,
            )
        except Exception as e:
            return self._result(
                False,
                "git_diff",
                "git diff failed",
                paths=paths or [],
                stdout="",
                stderr=str(e),
                returncode=1,
            )

    def git_commit(self, message: str, add_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        require_execution_allowed(
            "git_commit",
            risk_level="HIGH",
            intent=f"commit changes: {message}",
            write_required=True,
        )

        git = _get_git_tool()
        try:
            # Stage files
            if add_paths:
                add_result = git.add(add_paths)
            else:
                if hasattr(git, "add_all"):
                    add_result = git.add_all()
                else:
                    add_result = git.add(["."])

            if not add_result.ok:
                return self._result(
                    False,
                    "git_commit",
                    "git add failed",
                    stdout=add_result.output,
                    stderr=add_result.error or "",
                    returncode=1,
                )

            # Commit
            commit_result = git.commit(message)
            return self._result(
                commit_result.ok,
                "git_commit",
                "Commit created" if commit_result.ok else "git commit failed",
                message=message,
                stdout=commit_result.output,
                stderr=commit_result.error or "",
                returncode=0 if commit_result.ok else 1,
            )
        except Exception as e:
            return self._result(
                False,
                "git_commit",
                "git commit failed",
                message=message,
                stdout="",
                stderr=str(e),
                returncode=1,
            )

    # ----------------- Hot reload -----------------

    def hot_reload_module(self, module_name: str) -> Dict[str, Any]:
        """
        Reload a Python module in-process. High risk because it mutates runtime.
        """
        require_execution_allowed(
            "hot_reload_module",
            risk_level="HIGH",
            intent=f"reload Python module {module_name}",
            write_required=True,
        )

        try:
            import importlib

            module = importlib.import_module(module_name)
            importlib.reload(module)
        except Exception as e:
            logger.exception("Hot reload failed for %s", module_name)
            return self._result(
                False,
                "hot_reload_module",
                "Hot reload failed",
                module=module_name,
                error=str(e),
            )

        return self._result(
            True,
            "hot_reload_module",
            f"Reloaded module {module_name}",
            module=module_name,
        )


# Module-level service instance (lazy initialization)
_service_instance: Optional[ActionEngineService] = None


def get_service(root_dir: Optional[str] = None) -> ActionEngineService:
    """Get or create the ActionEngineService instance."""
    global _service_instance
    if _service_instance is None:
        if root_dir is None:
            # Default to maven2_fix directory
            root_dir = str(Path(__file__).resolve().parents[4])
        _service_instance = ActionEngineService(root_dir=root_dir)
    return _service_instance


# =============================================================================
# Legacy functions for backward compatibility
# =============================================================================

def schedule_actions(goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Schedule actions for a list of goals.

    Args:
        goals: A list of goal dictionaries.
    Returns:
        A list of scheduled action dictionaries.
    """
    if not goals:
        return []

    scheduled = []
    if _teacher_helper and _memory and len(goals) >= 1:
        try:
            goal_types = [g.get("type", "unknown") for g in goals[:3]]
            goal_pattern = "-".join(sorted(set(goal_types)))

            learned_patterns = _memory.retrieve(
                query=f"action scheduling pattern: {goal_pattern}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, list):
                        scheduled = content
                        print(f"[ACTION_ENGINE] Using learned scheduling pattern from Teacher")
                        break
        except Exception:
            pass

    if _memory and len(goals) > 0:
        try:
            _memory.store(
                content={"goals": goals, "scheduled_count": len(scheduled)},
                metadata={"kind": "schedule_event", "source": "action_engine", "confidence": 0.8}
            )
        except Exception:
            pass

    return scheduled


def execute_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute a list of scheduled actions.

    Routes each action through ActionEngineService or action_engine_brain.
    """
    if not actions:
        return []

    results = []
    svc = get_service()

    # Try action_engine_brain first
    try:
        from brains.action_engine_brain import ActionEngineBrain, ActionRequest
        brain = ActionEngineBrain(root_dir=str(svc.root_dir))
        brain_available = True
    except ImportError:
        brain_available = False
        print("[ACTION_ENGINE] Warning: action_engine_brain not available, using ActionEngineService")

    for action in actions:
        action_type = action.get("type") or action.get("action", "")
        params = action.get("params", {})

        result = None

        # Try ActionEngineBrain first (has comprehensive routing)
        if brain_available:
            try:
                result = brain.handle(ActionRequest(action=action_type, params=params))
            except Exception as e:
                print(f"[ACTION_ENGINE] Brain execution failed: {e}")
                result = None

        # Fallback to direct ActionEngineService methods
        if result is None:
            try:
                if action_type == "list_python_files":
                    result = svc.list_python_files(**params)
                elif action_type == "read_text_file":
                    result = svc.read_text_file(**params)
                elif action_type == "write_text_file":
                    result = svc.write_text_file(**params)
                elif action_type == "append_text_file":
                    result = svc.append_text_file(**params)
                elif action_type == "mkdir":
                    result = svc.mkdir(**params)
                elif action_type == "run_python_sandbox":
                    result = svc.run_python_sandbox(**params)
                elif action_type == "run_shell":
                    result = svc.run_shell(**params)
                elif action_type == "git_status":
                    result = svc.git_status()
                elif action_type == "git_diff":
                    result = svc.git_diff(**params)
                elif action_type == "git_commit":
                    result = svc.git_commit(**params)
                elif action_type == "hot_reload_module":
                    result = svc.hot_reload_module(**params)
                # Legacy tool-layer fallbacks
                elif action_type == "fs_scan" and fs_scan_tool:
                    root = params.get("root")
                    pattern = params.get("pattern", "*.py")
                    files = fs_scan_tool.scan_codebase(root=root, pattern=pattern)
                    result = {"ok": True, "action": action_type, "summary": f"Found {len(files)} files", "details": {"files": files}}
                elif action_type == "fs_write" and fs_write_tool:
                    path = params.get("path")
                    content = params.get("content", "")
                    mode = params.get("mode", "create_or_backup")
                    written = fs_write_tool.write_file(path, content, mode=mode)
                    result = {"ok": True, "action": action_type, "summary": f"Wrote to {written}", "details": {"path": str(written)}}
                elif action_type == "reload" and reload_tool:
                    modules = params.get("modules", [])
                    reload_result = reload_tool.reload_modules(modules)
                    result = {"ok": True, "action": action_type, "summary": "Modules reloaded", "details": {"results": reload_result}}
                else:
                    result = {"ok": False, "action": action_type, "summary": f"Unknown action: {action_type}", "details": {"error": "UNKNOWN_ACTION"}}
            except PermissionError as e:
                result = {"ok": False, "action": action_type, "summary": "Execution not permitted", "details": {"error": str(e)}}
            except Exception as e:
                result = {"ok": False, "action": action_type, "summary": "Action failed", "details": {"error": str(e)}}

        results.append(result)

    # Store execution events in memory
    if _memory:
        try:
            _memory.store(
                content={"actions": actions, "results": results, "executed_count": len(results)},
                metadata={"kind": "execute_event", "source": "action_engine", "confidence": 0.9}
            )
        except Exception:
            pass

    return results


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Action Engine service API.

    Supported operations:
    - SCHEDULE: Schedule actions for goals
    - EXECUTE: Execute scheduled actions
    - EXECUTE_AGENCY_TOOL: Execute agency tools directly (auto-detected)
    - Direct action operations: LIST_PYTHON_FILES, READ_FILE, WRITE_FILE, etc.
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    # AUTO-DETECT: Check if integrator detected an agency tool pattern
    if _agency_tools_available and op in ("EXECUTE", "SCHEDULE"):
        try:
            agency_info = get_agency_tool_info()
            if agency_info and agency_info.get('bypass_teacher', True):
                print(f"[ACTION_ENGINE] Auto-detected agency tool: {agency_info['tool']}, executing directly")
                op = "EXECUTE_AGENCY_TOOL"
        except Exception as e:
            print(f"[ACTION_ENGINE] Failed to check for agency tools: {e}")

    svc = get_service()

    # Direct action operations via ActionEngineService
    if op == "LIST_PYTHON_FILES":
        try:
            result = svc.list_python_files(
                max_files=payload.get("max_files", 500),
                include_tests=payload.get("include_tests", True)
            )
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "LIST_FAILED", "message": str(e)}}

    if op == "READ_FILE":
        try:
            result = svc.read_text_file(
                path=payload.get("path", ""),
                max_chars=payload.get("max_chars", 100_000)
            )
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "READ_FAILED", "message": str(e)}}

    if op == "WRITE_FILE":
        try:
            result = svc.write_text_file(
                path=payload.get("path", ""),
                content=payload.get("content", ""),
                backup=payload.get("backup", True)
            )
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "WRITE_FAILED", "message": str(e)}}

    if op == "RUN_PYTHON":
        try:
            result = svc.run_python_sandbox(
                code=payload.get("code", ""),
                timeout_ms=payload.get("timeout_ms", 3000)
            )
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "EXEC_FAILED", "message": str(e)}}

    if op == "RUN_SHELL":
        try:
            result = svc.run_shell(
                command=payload.get("command", ""),
                timeout=payload.get("timeout", 30)
            )
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "SHELL_FAILED", "message": str(e)}}

    if op == "GIT_STATUS":
        try:
            result = svc.git_status()
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "GIT_FAILED", "message": str(e)}}

    if op == "GIT_DIFF":
        try:
            result = svc.git_diff(paths=payload.get("paths"))
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "GIT_FAILED", "message": str(e)}}

    if op == "GIT_COMMIT":
        try:
            result = svc.git_commit(
                message=payload.get("message", ""),
                add_paths=payload.get("add_paths")
            )
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "GIT_FAILED", "message": str(e)}}

    if op == "HOT_RELOAD":
        try:
            result = svc.hot_reload_module(module_name=payload.get("module_name", ""))
            return {"ok": result["ok"], "payload": result}
        except PermissionError as e:
            return {"ok": False, "error": {"code": "PERMISSION_DENIED", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "RELOAD_FAILED", "message": str(e)}}

    # Legacy operations
    if op == "SCHEDULE":
        goals = payload.get("goals", [])
        try:
            scheduled = schedule_actions(goals)
            return {"ok": True, "payload": {"scheduled": scheduled, "count": len(scheduled)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "SCHEDULE_FAILED", "message": str(e)}}

    if op == "EXECUTE":
        actions = payload.get("actions", [])
        try:
            results = execute_actions(actions)
            all_ok = all(r.get("ok", False) for r in results) if results else True
            return {
                "ok": all_ok,
                "payload": {
                    "executed": True,
                    "count": len(actions),
                    "results": results,
                    "succeeded": sum(1 for r in results if r.get("ok")),
                    "failed": sum(1 for r in results if not r.get("ok"))
                }
            }
        except Exception as e:
            return {"ok": False, "error": {"code": "EXECUTE_FAILED", "message": str(e)}}

    if op == "HEALTH":
        return {
            "ok": True,
            "payload": {
                "status": "operational",
                "service_class": "ActionEngineService",
                "root_dir": str(svc.root_dir),
                "tools_available": {
                    "fs_scan": fs_scan_tool is not None,
                    "fs_write": fs_write_tool is not None,
                    "git": git_tool is not None,
                    "reload": reload_tool is not None,
                    "agency": _agency_tools_available
                }
            }
        }

    # Legacy tool-specific operations
    if op == "EXECUTE_FS_SCAN":
        if not fs_scan_tool:
            return {"ok": False, "error": {"code": "TOOL_UNAVAILABLE", "message": "fs_scan_tool missing"}}
        try:
            files = fs_scan_tool.scan_codebase(root=payload.get("root"), pattern=payload.get("pattern", "*.py"))
            return {"ok": True, "payload": {"files": files, "count": len(files)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "FS_SCAN_FAILED", "message": str(e)}}

    if op == "EXECUTE_FS_WRITE":
        if not fs_write_tool:
            return {"ok": False, "error": {"code": "TOOL_UNAVAILABLE", "message": "fs_write_tool missing"}}
        try:
            written = fs_write_tool.write_file(payload.get("path"), payload.get("content", ""), mode=payload.get("mode", "create_or_backup"))
            return {"ok": True, "payload": {"path": str(written)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "FS_WRITE_FAILED", "message": str(e)}}

    if op == "EXECUTE_GIT":
        if not git_tool:
            return {"ok": False, "error": {"code": "TOOL_UNAVAILABLE", "message": "git_tool missing"}}
        git_op = (payload.get("git_op") or "").lower()
        try:
            if git_op == "status":
                return {"ok": True, "payload": {"status": git_tool.git_status()}}
            if git_op == "add":
                git_tool.git_add(payload.get("paths") or [])
                return {"ok": True, "payload": {"added": payload.get("paths") or []}}
            if git_op == "commit":
                commit_hash = git_tool.git_commit(payload.get("message", ""))
                return {"ok": True, "payload": {"commit": commit_hash}}
            if git_op == "push":
                pushed = git_tool.git_push(remote=payload.get("remote", "origin"), branch=payload.get("branch"))
                return {"ok": True, "payload": {"branch": pushed}}
            return {"ok": False, "error": {"code": "UNSUPPORTED_GIT_OP", "message": git_op or "missing"}}
        except Exception as e:
            return {"ok": False, "error": {"code": "GIT_FAILED", "message": str(e)}}

    if op == "EXECUTE_RELOAD":
        if not reload_tool:
            return {"ok": False, "error": {"code": "TOOL_UNAVAILABLE", "message": "reload_tool missing"}}
        try:
            result = reload_tool.reload_modules(payload.get("modules") or [])
            return {"ok": True, "payload": {"results": result}}
        except Exception as e:
            return {"ok": False, "error": {"code": "RELOAD_FAILED", "message": str(e)}}

    if op == "EXECUTE_AGENCY_TOOL":
        if not _agency_tools_available:
            return {"ok": False, "error": {"code": "AGENCY_TOOLS_UNAVAILABLE", "message": "Agency tools not loaded"}}
        try:
            agency_info = get_agency_tool_info()
            if not agency_info:
                return {"ok": False, "error": {"code": "NO_AGENCY_PATTERN", "message": "No agency tool pattern detected"}}
            print(f"[ACTION_ENGINE] Executing agency tool: {agency_info['tool']}")
            tool_result = execute_agency_tool(
                tool_path=agency_info['tool'],
                method_name=agency_info.get('method'),
                args=agency_info.get('args')
            )
            formatted_response = format_agency_response(payload.get("query", ""), tool_result)
            return {
                "ok": True,
                "payload": {
                    "result": tool_result,
                    "formatted_response": formatted_response,
                    "tool_executed": agency_info['tool'],
                    "bypass_teacher": agency_info.get('bypass_teacher', True)
                }
            }
        except Exception as e:
            import traceback
            return {"ok": False, "error": {"code": "AGENCY_TOOL_FAILED", "message": str(e), "details": traceback.format_exc()}}

    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}


# Standard service contract: handle is the entry point
service_api = handle
