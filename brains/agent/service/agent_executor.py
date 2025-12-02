"""
Agent Executor Service
======================

This module provides a simple agent executor which can plan, dry‑run,
execute and rollback tasks against the Maven codebase.  It is placed
outside of the governance brains to keep a clear separation between
cognitive reasoning and operational side effects.

Operations exposed via ``service_api``:

 - ``HEALTH``     : Return a basic health check.
 - ``PLAN``       : Accept a task specification and produce a plan.
 - ``DRY_RUN``    : Generate diffs without modifying files.
 - ``EXECUTE``    : Apply file changes and run shell commands (requires governance allow).
 - ``ROLLBACK``   : Restore files from backups created by EXECUTE.
 - ``REPORT``     : Return a summary of the last executed task.
 - ``CHAT``       : Very simple natural language interface that echoes the user.

This is a minimal skeleton and does not implement a full natural language
understanding pipeline.  It exists to provide a safe entry point for future
extensions.
"""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from brains.governance.policy_engine.service import policy_engine  # type: ignore
from api.policy import validate_taskspec  # type: ignore
from brains.maven_paths import MAVEN_ROOT, get_brains_path, get_reports_path

# For dynamic maintenance operations, import generate_mid lazily to avoid
# circular imports when agent_executor is loaded outside the Maven package.
try:
    from api.utils import generate_mid  # type: ignore
except Exception:
    generate_mid = None  # type: ignore

# Define a root for agent reports and backups using the confined reports helper
AGENT_REPORT_DIR = get_reports_path("agent")
BACKUP_DIR = AGENT_REPORT_DIR / "backups"
AGENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

_last_report: Optional[Dict[str, Any]] = None


def _read_file(path: str) -> str:
    """Read a file and return its contents."""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_file(path: str, content: str) -> None:
    """Write content to a file atomically with backup."""
    p = Path(path)
    # Ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)
    # Create backup
    ts = int(os.times().elapsed * 1000)
    backup_path = BACKUP_DIR / f"{p.name}.{ts}.bak"
    try:
        if p.exists():
            shutil.copy2(p, backup_path)
    except Exception:
        pass
    # Write new content
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp_path, p)


def _generate_diff(original: str, new_text: str) -> str:
    """Generate a unified diff between two strings."""
    import difflib

    original_lines = original.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile="original",
        tofile="new",
        lineterm="",
    )
    return "".join(diff)


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implementation for the agent executor service.

    The `msg` must contain an "op" field specifying the operation and may
    include a "payload" and "mid" (message id).  This function dispatches
    operations to the corresponding handlers below.
    """
    global _last_report
    # Check for global kill switch.  If the environment variable
    # MAVEN_KILL_SWITCH is set to "1" or "true", all operations that would
    # perform side effects are immediately aborted.  This provides a
    # safety valve to halt execution mid‑run.
    kill_switch = os.environ.get("MAVEN_KILL_SWITCH", "0").lower() in {"1", "true", "yes"}
    if kill_switch and op in {"PLAN", "DRY_RUN", "EXECUTE", "ROLLBACK", "MINILOOP"}:
        return {
            "ok": False,
            "mid": mid,
            "error": {
                "code": "KILL_SWITCH_ACTIVE",
                "message": "Execution disabled: global kill switch is active.",
                "hint": "Unset MAVEN_KILL_SWITCH to re‑enable agent operations."
            }
        }
    op = msg.get("op")
    mid = msg.get("mid", "UNKNOWN")
    payload = msg.get("payload") or {}

    if op == "HEALTH":
        return {"ok": True, "status": "operational", "agent_mode": True}

    if op == "PLAN":
        # Validate the taskspec against the IO schema before planning
        spec = payload.get("taskspec") or {}
        ok, err = validate_taskspec(spec)
        if not ok:
            return {
                "ok": False,
                "mid": mid,
                "error": {
                    "code": "INVALID_TASKSPEC",
                    "message": f"Invalid task specification: {err}",
                    "hint": "Ensure 'edits' and 'run' are properly structured."
                }
            }
        # For now the plan is simply to return the edits and commands unmodified.
        steps: List[Dict[str, Any]] = []
        for edit in spec.get("edits", []):
            steps.append({"type": "edit", "path": edit.get("path"), "selector": edit.get("selectors")})
        for run in spec.get("run", []):
            steps.append({"type": "shell", "cmd": run.get("cmd")})
        return {"ok": True, "plan": {"steps": steps}, "mid": mid}

    if op == "DRY_RUN":
        # Perform a dry‑run diff for each edit
        spec = payload.get("taskspec") or {}
        # Validate spec before diffing
        ok, err = validate_taskspec(spec)
        if not ok:
            return {
                "ok": False,
                "mid": mid,
                "error": {
                    "code": "INVALID_TASKSPEC",
                    "message": f"Invalid task specification: {err}",
                    "hint": "Dry runs require valid 'edits' and 'run' entries."
                }
            }
        diffs: List[Dict[str, Any]] = []
        for edit in spec.get("edits", []):
            path = edit.get("path")
            new_text = edit.get("new_text")
            if not path or new_text is None:
                continue
            old = _read_file(path)
            diff = _generate_diff(old, new_text)
            diffs.append({"path": path, "diff": diff})
        report = {"task": spec, "diffs": diffs, "executed": False}
        _last_report = report
        return {"ok": True, "report": report, "mid": mid}

    if op == "EXECUTE":
        # Execute file edits and shell commands with governance checks
        spec = payload.get("taskspec") or {}
        # Validate spec before execution
        ok, err = validate_taskspec(spec)
        if not ok:
            return {
                "ok": False,
                "mid": mid,
                "error": {
                    "code": "INVALID_TASKSPEC",
                    "message": f"Invalid task specification: {err}",
                    "hint": "Execution requires a valid taskspec with 'edits' and/or 'run'."
                }
            }
        results: List[Dict[str, Any]] = []
        # Apply edits
        for edit in spec.get("edits", []):
            path = edit.get("path")
            new_text = edit.get("new_text")
            if not path or new_text is None:
                continue
            # Request governance allow
            allow = policy_engine.service_api(
                {"op": "ENFORCE", "mid": mid, "payload": {"action": "WRITE_FILE", "target": path}}
            )
            if not (allow.get("ok") and allow.get("allowed", True)):
                results.append({"path": path, "status": "denied"})
                continue
            old = _read_file(path)
            diff = _generate_diff(old, new_text)
            _write_file(path, new_text)
            results.append({"path": path, "status": "applied", "diff": diff})
        # Run commands
        for run in spec.get("run", []):
            cmd = run.get("cmd")
            if not cmd:
                continue
            allow = policy_engine.service_api(
                {"op": "ENFORCE", "mid": mid, "payload": {"action": "RUN_SHELL", "target": cmd}}
            )
            if not (allow.get("ok") and allow.get("allowed", True)):
                results.append({"cmd": cmd, "status": "denied"})
                continue

            # Use shell tool from tool registry instead of subprocess
            from brains.tools_api import NullShellTool
            try:
                from brains.agent.tools.shell_tool import get_shell_tool
                shell_tool = get_shell_tool()
            except Exception:
                shell_tool = NullShellTool()

            if isinstance(shell_tool, NullShellTool):
                results.append({"cmd": cmd, "status": "error", "error": "Shell tool not available"})
                continue

            try:
                result = shell_tool.run(cmd, cwd=str(MAVEN_ROOT), timeout=120, check_policy=False)
                results.append(
                    {
                        "cmd": cmd,
                        "status": result.status,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout or "",
                        "stderr": result.stderr or "",
                    }
                )
            except Exception as e:
                results.append({"cmd": cmd, "status": "error", "error": str(e)})
        report = {"task": spec, "results": results, "executed": True}
        _last_report = report
        return {"ok": True, "report": report, "mid": mid}

    if op == "ROLLBACK":
        # Restore from the most recent backup for each path in spec.edits
        spec = payload.get("taskspec") or {}
        restored = []
        for edit in spec.get("edits", []):
            path = edit.get("path")
            if not path:
                continue
            p = Path(path)
            # Find latest backup for this file
            backups = sorted(BACKUP_DIR.glob(f"{p.name}.*.bak"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not backups:
                restored.append({"path": path, "status": "no_backup"})
                continue
            backup = backups[0]
            # Governance allow
            allow = policy_engine.service_api(
                {"op": "ENFORCE", "mid": mid, "payload": {"action": "ROLLBACK", "target": path}}
            )
            if not (allow.get("ok") and allow.get("allowed", True)):
                restored.append({"path": path, "status": "denied"})
                continue
            shutil.copy2(backup, p)
            restored.append({"path": path, "status": "restored", "backup": str(backup)})
        report = {"task": spec, "restored": restored, "executed": False}
        _last_report = report
        return {"ok": True, "report": report, "mid": mid}

    if op == "REPORT":
        # Return the last report captured by DRY_RUN, EXECUTE or ROLLBACK
        return {"ok": True, "report": _last_report, "mid": mid}

    if op == "CHAT":
        # Minimal chat: echoes the user and acknowledges agent mode
        text = payload.get("text", "")
        response = f"Agent received: {text}. Natural language processing not yet implemented."
        return {"ok": True, "response": response, "mid": mid}

    if op == "STATUS":
        return {"ok": True, "status": {"last_report": _last_report is not None}, "mid": mid}

    # -------------------------------------------------------------------------
    # MINILOOP operation: run a maintenance mini‑loop
    # -------------------------------------------------------------------------
    # This custom operation performs a lightweight maintenance cycle.  It tunes
    # memory rotation thresholds for all major cognitive brains via the
    # ``autotune`` helper and triggers a Self‑DMN reflection to produce
    # updated metrics.  The result of the cycle is persisted to a timestamped
    # JSON file under ``reports/agent`` and returned in the response.
    if op == "MINILOOP":
        try:
            window = int(payload.get("window", 10))
        except Exception:
            window = 10
        # Import autotune lazily to avoid import cycles when agent runs
        try:
            from api.memory import autotune  # type: ignore
        except Exception as e:
            autotune = None  # type: ignore
        results: List[Dict[str, Any]] = []
        # Tune each cognitive brain if autotune is available
        if autotune:
            try:
                cog_root = get_brains_path("cognitive")
                brain_names = [
                    "sensorium",
                    "planner",
                    "language",
                    "pattern_recognition",
                    "reasoning",
                    "affect_priority",
                    "personality",
                ]
                for bn in brain_names:
                    broot = cog_root / bn
                    try:
                        autotune(broot)
                        results.append({"brain": bn, "tuned": True})
                    except Exception as e:
                        results.append({"brain": bn, "tuned": False, "error": str(e)})
            except Exception as e:
                # If cognitive root resolution fails, record a single error
                results.append({"error": f"Failed to tune brains: {e}"})
        else:
            results.append({"error": "autotune unavailable"})
        # Invoke Self‑DMN reflection
        try:
            import importlib
            sd_mod = importlib.import_module(
                "brains.cognitive.self_dmn.service.self_dmn_brain"
            )
            sd_payload = {"op": "REFLECT", "mid": generate_mid() if generate_mid else mid, "payload": {"window": window}}
            sd_res = sd_mod.service_api(sd_payload)
            # Extract payload or entire response for inclusion in the result
            if isinstance(sd_res, dict) and sd_res.get("ok"):
                sd_out = sd_res.get("payload") or sd_res
            else:
                sd_out = sd_res
        except Exception as e:
            sd_out = {"error": f"Failed to reflect: {e}"}
        # Compose the mini‑loop result
        loop_result = {"tuned": results, "self_dmn": sd_out}
        # Persist result to reports/agent with a timestamp
        try:
            import time as _time
            ts = int(_time.time())
            AGENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = AGENT_REPORT_DIR / f"mini_loop_{ts}.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(loop_result, fh, indent=2)
        except Exception:
            pass
        return {"ok": True, "payload": loop_result, "mid": mid}

    return {"ok": False, "error": f"Unsupported op '{op}'", "mid": mid}


def handle(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Brain contract entry point.

    Args:
        context: Standard brain context dict

    Returns:
        Result dict from _handle_impl
    """
    return _handle_impl(context)


# Brain contract alias
service_api = handle
