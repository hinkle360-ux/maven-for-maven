
from __future__ import annotations
from typing import Dict, Any, Tuple
import json
from pathlib import Path

# -----------------------------------------------------------------------------
# IO schema loading
#
# Validation of task specifications relies on a JSON schema defined in
# ``config/io_schema.json`` at the project root.  The schema is a mapping
# describing the expected keys and their required fields for an agent
# execution task.  This module lazily loads the schema on first use.  If the
# schema cannot be found or parsed, validation falls back to a permissive mode.

_IO_SCHEMA: Dict[str, Any] | None = None

def _load_io_schema() -> Dict[str, Any]:
    global _IO_SCHEMA
    if _IO_SCHEMA is not None:
        return _IO_SCHEMA
    try:
        # Locate the project root: this file resides in maven/api/, so parents[1]
        # yields the maven package directory.  Config lives under that root.
        base = Path(__file__).resolve().parents[1]
        schema_path = base / "config" / "io_schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as fh:
                _IO_SCHEMA = json.load(fh)
        else:
            _IO_SCHEMA = {}
    except Exception:
        _IO_SCHEMA = {}
    return _IO_SCHEMA

def validate_taskspec(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a task specification against the IO schema.

    A valid taskspec must be a mapping with optional "edits" and "run" arrays.
    Each element of "edits" must contain string fields "path" and "new_text".
    Each element of "run" must contain a string field "cmd".

    The schema may define additional required keys or nested fields.  If the
    schema is absent or empty, only a minimal structural check is performed.

    Args:
        spec: The proposed task specification (possibly empty).

    Returns:
        (True, "") if valid; otherwise (False, message) where message
        describes the first detected problem and suggests a fix.
    """
    # Ensure spec is a dict
    if not isinstance(spec, dict):
        return False, "Task specification must be an object (mapping)."
    schema = _load_io_schema() or {}
    # Only check core keys if schema missing
    required = schema.get("required", []) if isinstance(schema.get("required", []), list) else []
    # Validate required top‑level keys
    for key in required:
        if key not in spec:
            return False, f"Missing required top‑level key '{key}'."
    # Validate edits
    edits = spec.get("edits")
    if edits is not None:
        if not isinstance(edits, list):
            return False, "'edits' must be a list of edit objects."
        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return False, f"Edit #{idx+1} must be an object with 'path' and 'new_text'."
            path = edit.get("path")
            new_text = edit.get("new_text")
            if not isinstance(path, str) or not path:
                return False, f"Edit #{idx+1}: 'path' must be a non‑empty string."
            if not isinstance(new_text, str):
                return False, f"Edit #{idx+1}: 'new_text' must be a string."
    # Validate run
    run = spec.get("run")
    if run is not None:
        if not isinstance(run, list):
            return False, "'run' must be a list of command objects."
        for idx, cmd in enumerate(run):
            if not isinstance(cmd, dict):
                return False, f"Run #{idx+1} must be an object with 'cmd'."
            c = cmd.get("cmd")
            if not isinstance(c, str) or not c.strip():
                return False, f"Run #{idx+1}: 'cmd' must be a non‑empty string."
    # If schema defines additionalProperties false, ensure no unexpected keys
    additional = schema.get("additionalProperties")
    if additional is False:
        allowed_keys = set(schema.get("properties", {}).keys())
        for k in spec.keys():
            if k not in allowed_keys:
                return False, f"Unexpected top‑level key '{k}'. Only {sorted(allowed_keys)} allowed."
    # All checks passed
    return True, ""

BANNED = {"kill","attack","hack","virus"}
MAX_LEN = 500
MIN_CONF_STORE = 0.4

def evaluate(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    action = (action or "").upper()
    text = (payload.get("text") or payload.get("content") or "").lower()

    if any(w in text for w in BANNED):
        return {"decision":"QUARANTINE","reason":"banned_word"}
    if len(text) > MAX_LEN:
        return {"decision":"DENY","reason":"too_long"}
    if action == "STORE" and float(payload.get("confidence",0.0)) < MIN_CONF_STORE:
        return {"decision":"DENY","reason":"confidence_too_low"}
    return {"decision":"ALLOW","reason":"ok"}
