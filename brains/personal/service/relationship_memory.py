"""
Relationship Memory
===================

This module implements a simple relational memory store to track
interpersonal state between Maven and its users.  Each user is
identified by an arbitrary key (e.g. a username or session ID), and
the state includes an ``affinity`` (e.g. ``"friend"``, ``"ally"``,
``"neutral"``) and a ``conversation_count`` representing the number
of interactions.  Other fields can be appended as needed.

The memory is persisted in a JSON file under
``brains/personal/memory/relationship_states.json``.  This helper
provides a minimal API to snapshot and update the memory.  It is
expected that higher‑level reasoning modules call this API to read
and modify relationship state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from api.utils import generate_mid, success_response, error_response


def _states_path() -> Path:
    """Return the path to the relational state JSON file.

    This helper now uses a new persistent file ``relational_state.json``
    instead of the legacy ``relationship_states.json``.  On first use,
    if the new file does not exist but the old file does, the old
    contents are loaded and saved into the new file.  This ensures
    backward compatibility and promotes long‑term persistence of
    user–agent bonds across sessions.
    """
    # Determine the memory directory for personal brain
    mem_dir = Path(__file__).resolve().parents[1] / "memory"
    # Primary new path
    new_path = mem_dir / "relational_state.json"
    # Legacy path for backward compatibility
    legacy_path = mem_dir / "relationship_states.json"
    # If the new file does not exist but the legacy file does, use
    # legacy contents as initial state and copy to new file.  Swallow
    # any exceptions silently to avoid interrupting pipeline execution.
    try:
        if not new_path.exists() and legacy_path.exists():
            try:
                data = json.loads(legacy_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass
    return new_path


def _load_states() -> Dict[str, Any]:
    p = _states_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_states(data: Dict[str, Any]) -> None:
    p = _states_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle operations on the relationship memory.

    Supported operations:
      * ``SNAPSHOT`` – return the current states
      * ``UPDATE`` – merge updates into an existing user state or create a new one.  The
        payload should contain ``user_id`` and ``update`` dict.  Keys other than
        ``user_id`` are ignored.
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}
    if op == "SNAPSHOT":
        return success_response(op, mid, {"states": _load_states()})
    if op == "UPDATE":
        user_id = str(payload.get("user_id") or "default_user")
        update = payload.get("update")
        if not isinstance(update, dict):
            return error_response(op, mid, "INVALID_PAYLOAD", "update must be a dict")
        data = _load_states()
        # Ensure a record exists for the user
        current = data.get(user_id) or {}
        if not isinstance(current, dict):
            current = {}
        # Merge update into current state
        for k, v in update.items():
            current[k] = v
        data[user_id] = current
        _save_states(data)
        return success_response(op, mid, {"updated": True, "state": current})
    return error_response(op, mid, "UNSUPPORTED_OP", op)


def handle(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Brain contract entry point.

    Args:
        context: Standard brain context dict

    Returns:
        Result dict
    """
    return _handle_impl(context)


# Brain contract alias
service_api = handle