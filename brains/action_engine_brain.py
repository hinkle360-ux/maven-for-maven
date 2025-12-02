"""
Action Engine Brain

High-level coordinator that:

- Receives structured action requests from the cognitive stack.
- Chooses the correct concrete operation (filesystem/git/sandbox/hot reload).
- Delegates to ActionEngineService.
- Returns structured results (NO stubs, NO fake answers).

This brain does not decide "should we act?" – that is governed by:
- execution_guard
- upstream routing/governance

It only decides "HOW to perform the requested action".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Get Maven root
try:
    from brains.maven_paths import get_maven_root
    MAVEN_ROOT = get_maven_root()
except ImportError:
    MAVEN_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ActionRequest:
    """Standard action request structure."""
    action: str
    params: Dict[str, Any]


class ActionEngineBrain:
    """
    Glue between cognition and the action service.
    """

    def __init__(self, root_dir: Optional[str] = None) -> None:
        if root_dir is None:
            root_dir = str(MAVEN_ROOT)

        # Import ActionEngineService from the service module
        try:
            from brains.cognitive.action_engine.service.action_engine import ActionEngineService
            self._svc = ActionEngineService(root_dir=root_dir)
            self._svc_available = True
        except ImportError as e:
            logger.warning("ActionEngineService not available: %s", e)
            self._svc = None
            self._svc_available = False
            self._root_dir = Path(root_dir)

    def handle(self, request: ActionRequest) -> Dict[str, Any]:
        """
        Main entry point. No TODOs. Every known action is fully implemented.
        Unknown actions return ok=False with an explicit error.
        """
        action = request.action
        params = request.params or {}

        # Use ActionEngineService if available
        if self._svc_available and self._svc:
            return self._handle_via_service(action, params)

        # Fallback to direct implementation
        return self._handle_direct(action, params)

    def _handle_via_service(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle action via ActionEngineService."""
        try:
            if action == "list_python_files":
                return self._svc.list_python_files(
                    max_files=int(params.get("max_files", 500)),
                    include_tests=bool(params.get("include_tests", True)),
                )

            if action == "read_text_file":
                return self._svc.read_text_file(
                    path=str(params["path"]),
                    max_chars=int(params.get("max_chars", 100_000)),
                )

            if action == "read_bytes_file":
                return self._svc.read_bytes_file(
                    path=str(params["path"]),
                    max_bytes=int(params.get("max_bytes", 100_000)),
                )

            if action == "mkdir":
                return self._svc.mkdir(
                    path=str(params["path"]),
                    exist_ok=bool(params.get("exist_ok", True)),
                )

            if action == "write_text_file":
                return self._svc.write_text_file(
                    path=str(params["path"]),
                    content=str(params["content"]),
                    backup=bool(params.get("backup", True)),
                )

            if action == "append_text_file":
                return self._svc.append_text_file(
                    path=str(params["path"]),
                    content=str(params["content"]),
                    backup=bool(params.get("backup", True)),
                )

            if action == "run_python_sandbox":
                return self._svc.run_python_sandbox(
                    code=str(params["code"]),
                    timeout_ms=int(params.get("timeout_ms", 3_000)),
                )

            if action == "run_shell":
                return self._svc.run_shell(
                    command=str(params["command"]),
                    timeout=int(params.get("timeout", 30)),
                )

            if action == "git_status":
                return self._svc.git_status()

            if action == "git_diff":
                paths_param = params.get("paths")
                paths = list(paths_param) if isinstance(paths_param, (list, tuple)) else None
                return self._svc.git_diff(paths=paths)

            if action == "git_commit":
                paths_param = params.get("add_paths")
                add_paths = list(paths_param) if isinstance(paths_param, (list, tuple)) else None
                return self._svc.git_commit(
                    message=str(params["message"]),
                    add_paths=add_paths,
                )

            if action == "hot_reload_module":
                return self._svc.hot_reload_module(
                    module_name=str(params["module_name"]),
                )

            # Legacy action name mappings
            if action == "read_file":
                return self._svc.read_text_file(
                    path=str(params["path"]),
                    max_chars=int(params.get("max_bytes", 100_000)),
                )

            if action == "write_file":
                return self._svc.write_text_file(
                    path=str(params["path"]),
                    content=str(params["content"]),
                    backup=bool(params.get("backup", True)),
                )

            if action == "append_file":
                return self._svc.append_text_file(
                    path=str(params["path"]),
                    content=str(params["content"]),
                    backup=bool(params.get("backup", True)),
                )

        except KeyError as e:
            logger.exception("Missing parameter for action %s", action)
            return {
                "ok": False,
                "action": action,
                "summary": f"Missing required parameter: {e}",
                "details": {"missing_param": str(e)},
            }
        except PermissionError as e:
            # Raised by execution_guard
            logger.warning("Action %s denied: %s", action, e)
            return {
                "ok": False,
                "action": action,
                "summary": "Execution not permitted by guard",
                "details": {"error": str(e)},
            }
        except Exception as e:
            logger.exception("Action %s failed", action)
            return {
                "ok": False,
                "action": action,
                "summary": "Action execution failed",
                "details": {"error": str(e)},
            }

        # Unknown action
        logger.warning("Unknown action requested: %s", action)
        return {
            "ok": False,
            "action": action,
            "summary": f"Unknown action: {action}",
            "details": {"error": "UNKNOWN_ACTION"},
        }

    def _handle_direct(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Direct handling without ActionEngineService (fallback)."""
        from brains.tools.execution_guard import require_execution_allowed

        try:
            if action == "list_python_files":
                require_execution_allowed("list_python_files", "LOW", write_required=False)
                max_files = int(params.get("max_files", 500))
                files = []
                for p in self._root_dir.rglob("*.py"):
                    try:
                        files.append(str(p.relative_to(self._root_dir)))
                    except Exception:
                        continue
                    if len(files) >= max_files:
                        break
                return {
                    "ok": True,
                    "action": action,
                    "summary": f"Found {len(files)} Python files",
                    "details": {"files": files, "root": str(self._root_dir)},
                }

            if action in ("read_file", "read_text_file"):
                require_execution_allowed("read_text_file", "LOW", write_required=False)
                path = self._root_dir / params["path"]
                if not path.exists():
                    return {"ok": False, "action": action, "summary": "File not found", "details": {"path": str(path)}}
                content = path.read_text(encoding="utf-8")[:params.get("max_chars", 100_000)]
                return {
                    "ok": True,
                    "action": action,
                    "summary": f"Read {len(content)} chars",
                    "details": {"path": str(path), "content": content},
                }

            if action in ("write_file", "write_text_file"):
                require_execution_allowed("write_text_file", "MEDIUM", write_required=True)
                path = self._root_dir / params["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(params["content"], encoding="utf-8")
                return {
                    "ok": True,
                    "action": action,
                    "summary": f"Wrote {len(params['content'])} chars",
                    "details": {"path": str(path)},
                }

        except PermissionError as e:
            return {"ok": False, "action": action, "summary": "Execution not permitted", "details": {"error": str(e)}}
        except Exception as e:
            return {"ok": False, "action": action, "summary": f"Action failed: {e}", "details": {"error": str(e)}}

        return {"ok": False, "action": action, "summary": f"Unknown action: {action}", "details": {"error": "UNKNOWN_ACTION"}}


# =============================================================================
# Module-level functions for backward compatibility
# =============================================================================

# Default brain instance
_default_brain: Optional[ActionEngineBrain] = None


def _get_brain() -> ActionEngineBrain:
    """Get or create default brain instance."""
    global _default_brain
    if _default_brain is None:
        _default_brain = ActionEngineBrain()
    return _default_brain


def handle_action(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Handle an action request (backward compatible function).

    Args:
        action: Action name
        params: Action parameters

    Returns:
        Standard result schema
    """
    brain = _get_brain()
    return brain.handle(ActionRequest(action=action, params=params or {}))


def execute_action(action_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a structured action object (backward compatible function).

    Args:
        action_obj: Dictionary with 'type' and 'params' keys

    Returns:
        Standard result schema
    """
    action_type = action_obj.get("type", "")
    params = action_obj.get("params", {})
    return handle_action(action_type, params)


def get_available_actions() -> List[Dict[str, Any]]:
    """Get list of available actions with their risk levels."""
    return [
        {"name": "list_python_files", "risk": "LOW", "available": True},
        {"name": "read_text_file", "risk": "LOW", "available": True},
        {"name": "read_bytes_file", "risk": "LOW", "available": True},
        {"name": "mkdir", "risk": "MEDIUM", "available": True},
        {"name": "write_text_file", "risk": "MEDIUM", "available": True},
        {"name": "append_text_file", "risk": "MEDIUM", "available": True},
        {"name": "run_python_sandbox", "risk": "HIGH", "available": True},
        {"name": "git_status", "risk": "LOW", "available": True},
        {"name": "git_diff", "risk": "LOW", "available": True},
        {"name": "git_commit", "risk": "HIGH", "available": True},
        {"name": "hot_reload_module", "risk": "HIGH", "available": True},
    ]


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API entry point for the action engine brain.

    Supports operations:
    - EXECUTE: Execute a single action
    - BATCH_EXECUTE: Execute multiple actions
    - LIST_ACTIONS: List available actions
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid")

    if op == "EXECUTE":
        action_type = payload.get("action") or payload.get("type", "")
        params = payload.get("params", {})
        result = handle_action(action_type, params)
        return {"ok": result["ok"], "op": op, "mid": mid, "payload": result}

    if op == "BATCH_EXECUTE":
        actions = payload.get("actions", [])
        results = []
        for action_obj in actions:
            result = execute_action(action_obj)
            results.append(result)
            if payload.get("stop_on_error") and not result["ok"]:
                break

        all_ok = all(r["ok"] for r in results) if results else True
        return {
            "ok": all_ok,
            "op": op,
            "mid": mid,
            "payload": {
                "results": results,
                "total": len(actions),
                "executed": len(results),
                "succeeded": sum(1 for r in results if r["ok"]),
            }
        }

    if op == "LIST_ACTIONS":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {"actions": get_available_actions()}
        }

    if op == "HEALTH":
        brain = _get_brain()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service_available": brain._svc_available,
                "brain_class": "ActionEngineBrain",
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unsupported operation: {op}"}
    }


# Legacy compatibility
def list_python_files() -> Dict[str, Any]:
    """Legacy function for backwards compatibility."""
    return handle_action("list_python_files", {"root": str(MAVEN_ROOT)})


# Standard service contract
service_api = handle
