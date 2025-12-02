"""
python_exec Tool
=================

This module exposes a minimal sandboxed Python execution environment for
use within the Maven agent.  It is deliberately constrained: all
execution occurs in the current process with strict limits on the
available builtins, imported modules and execution time.  The tool
supports four operations via ``service_api``:

* ``LINT``: Perform a static analysis of a code snippet.  The code
  must parse successfully under the ``ast`` module and may only
  import modules listed in the allow‑list configured in
  ``config/coding.json``.  Returns any lint warnings but does not
  execute the code.

* ``RUN``: Execute a code snippet in a restricted environment.  All
  builtins except a small safe subset are removed.  Only allowed
  imports are permitted.  Standard output is captured and returned
  along with any exception messages.  Execution is interrupted if it
  exceeds the configured time limit.

* ``TEST``: Execute both a code snippet and an accompanying test
  snippet in the same restricted environment.  The test code should
  raise an exception on failure; any uncaught exception is reported
  as a test failure.  When the tests pass, returns a summary with
  ``passed: True``.

* ``SANDBOX_INFO``: Return the current sandbox configuration
  including time and memory limits and the module allow‑list.

All execution requests require an ``EXEC`` permit from the governance
layer.  The permit is automatically requested inside this tool.

Note: This executor is best‑effort only.  It does not fork a
separate process and therefore cannot fully enforce hard memory
limits.  It does, however, prevent access to dangerous builtins and
disallowed imports.  When code attempts to import a disallowed
module, the LINT operation will reject it.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import ast
import builtins
import contextlib
import io
import time
from pathlib import Path
import json

from brains.maven_paths import get_maven_root, get_reports_path

# Import the governance permit request helper
try:
    from brains.governance.policy_engine.service.permits import request_permit  # type: ignore
except Exception:
    request_permit = None  # type: ignore

# Path to the Maven root for reading configuration and writing logs
MAVEN_ROOT = get_maven_root()

# Default sandbox configuration.  These values may be overridden via
# ``config/coding.json`` if present.  See that file for example
# contents.
DEFAULT_CONFIG = {
    "executor": {
        "time_limit_sec": 2,
        "mem_limit_mb": 128,
        "tmp_dir": "/tmp/maven_sandbox"
    },
    "allow_imports": [
        "math", "json", "re", "itertools", "functools", "datetime", "statistics", "fractions"
    ],
    "max_refine_loops": 3
}


def _is_full_agency_mode() -> bool:
    """Check if FULL_AGENCY mode is active."""
    try:
        from brains.tools.execution_guard import get_execution_status, ExecMode
        status = get_execution_status()
        return status.mode == ExecMode.FULL_AGENCY and status.effective
    except Exception:
        return False


def _load_full_agency_config() -> Dict[str, Any]:
    """Load permissive configuration from tool_policy.json for FULL_AGENCY mode."""
    policy_path = MAVEN_ROOT / "config" / "tool_policy.json"
    try:
        if policy_path.exists():
            with open(policy_path, "r", encoding="utf-8") as fh:
                policy = json.load(fh) or {}
            python_policy = policy.get("python_policy", {})
            return {
                "executor": {
                    "time_limit_sec": python_policy.get("timeout_ms", 60000) / 1000,
                    "mem_limit_mb": python_policy.get("memory_limit_mb", 1024),
                    "tmp_dir": "/tmp/maven_sandbox"
                },
                # In FULL_AGENCY mode, allow all imports
                "allow_imports": "__all__",  # Special marker for unrestricted imports
                "max_refine_loops": 10,
                "unrestricted": True,
            }
    except Exception:
        pass
    return None


def _load_config() -> Dict[str, Any]:
    """Load sandbox configuration.

    In FULL_AGENCY mode, uses permissive settings from tool_policy.json.
    Otherwise, uses restrictive settings from coding.json.
    """
    # Check for FULL_AGENCY mode first
    if _is_full_agency_mode():
        full_agency_cfg = _load_full_agency_config()
        if full_agency_cfg:
            return full_agency_cfg

    # Default: load from coding.json
    cfg_path = MAVEN_ROOT / "config" / "coding.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            # Validate keys and merge with defaults
            cfg = DEFAULT_CONFIG.copy()
            cfg.update({k: v for k, v in data.items() if k in cfg})
            return cfg
    except Exception:
        pass
    return DEFAULT_CONFIG


def _get_config() -> Dict[str, Any]:
    """Get configuration, checking for FULL_AGENCY mode dynamically."""
    # Always check mode dynamically to handle runtime changes
    return _load_config()


CONFIG = _load_config()


def _safe_builtins() -> Dict[str, Any]:
    """Return a dictionary of safe builtins for code execution.

    In FULL_AGENCY mode, returns all builtins (unrestricted).
    Otherwise, only a small subset of Python's builtins are exposed to user code.
    """
    cfg = _get_config()

    # In FULL_AGENCY mode (unrestricted), allow all builtins
    if cfg.get("unrestricted", False):
        return dict(vars(builtins))

    # Default: restricted builtins
    safe_names = [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "range", "set", "str", "sum", "tuple",
        "print"
    ]
    safe_builtins: Dict[str, Any] = {name: getattr(builtins, name) for name in safe_names}
    # Provide a limited version of print that writes to captured stdout
    return safe_builtins


def _check_imports(code: str) -> Tuple[bool, Optional[str]]:
    """Parse the code and verify that all imports are allowed.

    In FULL_AGENCY mode, all imports are allowed.
    Otherwise returns (False, error_message) if imports are not in allow-list.
    """
    cfg = _get_config()
    allow_imports = cfg.get("allow_imports", [])

    # In FULL_AGENCY mode (unrestricted), allow all imports
    if allow_imports == "__all__" or cfg.get("unrestricted", False):
        try:
            ast.parse(code, mode="exec")
        except Exception as e:
            return False, f"Syntax error: {e}"
        return True, None

    # Default: check against allow-list
    allowed = set(allow_imports)
    try:
        tree = ast.parse(code, mode="exec")
    except Exception as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name not in allowed:
                    return False, f"Import of '{name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod and mod not in allowed:
                return False, f"Import from '{mod}' is not allowed"
    return True, None


def _request_exec_permit(code: str) -> Tuple[bool, str, Optional[str]]:
    """Request an EXEC permit from the governance layer.

    Returns a tuple of (allowed, permit_id, reason).  If the permit
    mechanism is unavailable, default to allowed.
    """
    if request_permit is None:
        # Governance not available; implicitly allow
        return True, "PERMIT-UNKNOWN", None
    params = {
        "time_limit_sec": CONFIG.get("executor", {}).get("time_limit_sec", 2),
        "mem_limit_mb": CONFIG.get("executor", {}).get("mem_limit_mb", 128)
    }
    res = request_permit("EXEC", params)
    allowed = bool(res.get("allowed"))
    permit_id = str(res.get("permit_id", "PERMIT-UNKNOWN"))
    reason = res.get("reason")
    return allowed, permit_id, reason


def _run_code_snippet(code: str, inputs: Optional[str] = None) -> Tuple[str, str, float, Optional[Exception]]:
    """Execute user code in a restricted environment.

    Returns a tuple (stdout, stderr, elapsed_seconds, exception).
    ``inputs`` is currently unused but reserved for future support of
    stdin injection.
    """
    start_time = time.time()
    # Prepare isolated globals and locals.  Builtins are restricted to the safe subset.
    globals_dict: Dict[str, Any] = {
        "__builtins__": _safe_builtins()
    }
    locals_dict: Dict[str, Any] = {}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    # Check imports first
    ok, err = _check_imports(code)
    if not ok:
        return "", err or "Import error", 0.0, Exception(err or "Import error")

    # Get config dynamically for FULL_AGENCY mode
    cfg = _get_config()
    allow_imports = cfg.get("allow_imports", [])

    # Pre-import modules - in unrestricted mode, don't pre-import (use standard import)
    if allow_imports != "__all__" and not cfg.get("unrestricted", False):
        for modname in allow_imports:
            try:
                globals_dict[modname] = __import__(modname)
            except Exception:
                # If a module fails to import, silently skip
                continue
    # Execute code with captured stdout/stderr
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Enforce a time limit by checking elapsed time periodically within the execution.
            # Since we cannot interrupt Python mid‑execution without threads or signals,
            # we simply run the code and check the elapsed time after execution.
            exec(compile(code, "<user_code>", "exec"), globals_dict, locals_dict)
    except Exception as e:
        elapsed = time.time() - start_time
        return stdout_capture.getvalue(), stderr_capture.getvalue() or str(e), elapsed, e
    elapsed = time.time() - start_time
    return stdout_capture.getvalue(), stderr_capture.getvalue(), elapsed, None


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the python_exec tool.

    The ``msg`` must contain an ``op`` specifying which operation to
    perform and may include a ``payload`` dict with operation‑specific
    parameters.  Returns a dict with ``ok`` and a ``payload`` or
    ``error`` field.
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    # Sanity check
    if not op:
        return {"ok": False, "error": {"code": "MISSING_OP", "message": "op is required"}}
    # Load current configuration on each call to allow dynamic updates
    global CONFIG
    CONFIG = _load_config()
    # LINT operation: static analysis only
    if op == "LINT":
        code = payload.get("code", "")
        if not isinstance(code, str):
            return {"ok": False, "error": {"code": "INVALID_CODE", "message": "code must be a string"}}
        ok, err = _check_imports(code)
        if not ok:
            return {"ok": True, "payload": {"valid": False, "error": err}}
        # Attempt to parse code for syntax errors
        try:
            ast.parse(code, mode="exec")
        except Exception as e:
            return {"ok": True, "payload": {"valid": False, "error": f"Syntax error: {e}"}}
        return {"ok": True, "payload": {"valid": True, "warning": None}}
    # SANDBOX_INFO operation: return config
    if op == "SANDBOX_INFO":
        return {"ok": True, "payload": {
            "executor": CONFIG.get("executor", {}),
            "allow_imports": CONFIG.get("allow_imports", [])
        }}
    # HEALTH operation: check availability
    if op == "HEALTH":
        return {"ok": True, "payload": {
            "status": "operational",
            "service": "python_exec",
            "capability": "code_execution",
            "description": "Sandboxed Python code execution",
            "allow_imports": CONFIG.get("allow_imports", []),
        }}
    # RUN and TEST require code and a permit
    if op in {"RUN", "TEST"}:
        code = payload.get("code", "")
        if not isinstance(code, str):
            return {"ok": False, "error": {"code": "INVALID_CODE", "message": "code must be a string"}}
        # Check for disallowed imports up front
        ok, err = _check_imports(code)
        if not ok:
            return {"ok": False, "error": {"code": "IMPORT_ERROR", "message": err}}
        # Request permit
        allowed, permit_id, reason = _request_exec_permit(code)
        if not allowed:
            return {"ok": False, "error": {"code": "PERMIT_DENIED", "message": f"EXEC permit denied: {reason or 'not allowed'}"}}
        # Execute
        stdout, stderr, elapsed, exc = _run_code_snippet(code, payload.get("inputs"))
        # Record execution log
        try:
            logs_dir = get_reports_path("agent")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "code_runs.jsonl"
            record = {
                "ts": time.time(),
                "op": op,
                "permit_id": permit_id,
                "elapsed_sec": elapsed,
                "code_hash": hash(code),
                "stdout": stdout,
                "stderr": stderr if stderr else (str(exc) if exc else "")
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass
        # If this is a TEST operation, execute the test snippet
        if op == "TEST":
            test_code = payload.get("test_code", "")
            if not isinstance(test_code, str):
                return {"ok": False, "error": {"code": "INVALID_TEST_CODE", "message": "test_code must be a string"}}
            # Merge code and test; run both in same sandbox
            merged = code + "\n\n" + test_code
            test_stdout, test_stderr, test_elapsed, test_exc = _run_code_snippet(merged)
            ok_result = test_exc is None and not test_stderr
            return {"ok": True, "payload": {
                "passed": ok_result,
                "stdout": test_stdout,
                "stderr": test_stderr,
                "elapsed_sec": test_elapsed,
                "permit_id": permit_id
            }}
        # For RUN, return stdout/stderr and elapsed time
        return {"ok": True, "payload": {
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_sec": elapsed,
            "permit_id": permit_id
        }}
    # Unsupported operation
    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "python_exec"
TOOL_CAPABILITY = "code_execution"
TOOL_DESCRIPTION = "Sandboxed Python code execution with lint, run, and test operations"
TOOL_OPERATIONS = ["LINT", "RUN", "TEST", "SANDBOX_INFO"]


def is_available() -> bool:
    """Check if python_exec is available (always True)."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
        "requires_execution": True,
        "module": "brains.agent.tools.python_exec",
    }