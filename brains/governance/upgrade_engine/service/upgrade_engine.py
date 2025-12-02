"""
Upgrade Engine - Maven Governance Brain
--------------------------------------

This brain lives in the governance layer and is responsible for
identifying, proposing and (upon authorization) applying upgrades to
the Maven system.  It communicates with other brains exclusively via
the Memory Librarian bus and never forms direct peer connections.

Supported operations:

* **SCAN** – perform a non‑destructive analysis of available system
  metadata and return a summary of potential upgrade opportunities.
  No authorization is required for this operation.

* **PROPOSE** – accept a candidate upgrade description and return a
  structured proposal.  This operation does not apply the upgrade; it
  simply packages the information for later review.  No authorization
  is required.

* **APPLY** – apply an approved upgrade to the system.  Because this
  modifies code or configuration, a valid governance authorization
  token is required in the payload.

Any unsupported operation will return an error.

Note: This implementation is intentionally minimal.  It provides the
basic scaffolding for an upgrade engine but does not perform real
analysis or patching.  Real functionality should be built on top of
these primitives, adhering to Maven's primitive‑knowledge philosophy.
"""

from __future__ import annotations
from typing import Any, Dict
import time, json
from pathlib import Path

"""
This module implements a very primitive Upgrade Engine for Maven.  In order to
support agent‑mode workflows, it now writes proposals and logs to private
directories under the engine's own folder.  These directories are created
automatically at import time and are never shared with other brains.  No
external code is executed and no hidden libraries are imported; only the
standard library is used to persist files and manage timestamps.

Directories:

```
upgrade_engine/
  service/
    upgrade_engine.py          # this file
  packs/                       # final approved upgrade bundles
  staging/                     # drafted proposals awaiting approval
  logs/                        # application logs and history
```

The ``packs`` and ``staging`` folders are intentionally separated: ``staging``
holds JSON descriptions of proposed upgrades, while ``packs`` will contain
approved bundles (e.g. archives or collected files) once the system learns
how to assemble them.  ``logs`` records when scans, proposals and applies
occur so that Governance and Self‑Default can inspect the history without
parsing memory files directly.

All file writes are guarded so that failures will not crash the engine; any
exceptions in file I/O are swallowed and only affect logging.
"""

# Determine root paths relative to this file.  ``ENGINE_ROOT`` points to
# the ``upgrade_engine`` directory, which is one level up from this file
# (``service/upgrade_engine.py``).  The three work directories live at
# ``ENGINE_ROOT / "packs"``, ``staging`` and ``logs``.
ENGINE_ROOT = Path(__file__).resolve().parents[1]
PACKS_DIR = ENGINE_ROOT / "packs"
STAGING_DIR = ENGINE_ROOT / "staging"
LOGS_DIR = ENGINE_ROOT / "logs"

for _dir in (PACKS_DIR, STAGING_DIR, LOGS_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _auth_ok(auth: Dict[str, Any]) -> bool:
    """Validate a governance authorization block.

    An auth block must be a dict with the following keys:
    - ``issuer``: must equal ``"governance"``.
    - ``valid``: boolean flag that must be True.
    - ``ts``: timestamp when the token was issued (ms).
    - ``ttl_ms``: time to live in milliseconds.
    - ``token``: a string starting with ``"GOV-"`` of at least length 8.

    Returns True if all conditions are satisfied and the token has not expired.
    """
    if not isinstance(auth, dict):
        return False
    if auth.get("issuer") != "governance":
        return False
    if not auth.get("valid", False):
        return False
    ts = auth.get("ts")
    ttl = auth.get("ttl_ms", 0)
    if not isinstance(ts, int) or not isinstance(ttl, int):
        return False
    if _now_ms() > ts + ttl:
        return False
    tok = auth.get("token", "")
    return isinstance(tok, str) and tok.startswith("GOV-") and len(tok) >= 8


def _scan_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder implementation for scan.

    In a real implementation, this function would inspect system history,
    performance metrics and role charters to identify opportunities for
    improvement.  Here we simply echo back a list of brains to suggest
    scanning order and return an empty list of suggested upgrades.
    """
    brains = payload.get("brains", [
        "sensorium", "planner", "language", "pattern_recognition",
        "memory", "reasoning", "affect_priority", "personality",
        "system_history", "self_dmn"
    ])
    result = {
        "ok": True,
        "brains_scanned": brains,
        "suggested_upgrades": []
    }
    # Record the scan event in the engine's log directory.  This creates a small
    # JSON file with the event type, timestamp and payload.  If logging fails
    # (e.g. due to file permissions), we ignore the error and proceed.
    try:
        fname = LOGS_DIR / f"{_now_ms()}_scan.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"event": "scan", "timestamp": _now_ms(), "data": result}, f, indent=2)
    except Exception:
        pass
    return result


def _propose_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept an upgrade description and package it as a proposal.

    The ``payload`` should contain a ``description`` key describing the
    change and an optional ``target`` identifying which part of the
    system is affected.  The function returns a proposal object with
    a unique id and a timestamp.
    """
    description = (payload or {}).get("description", "")
    target = (payload or {}).get("target", "unspecified")
    proposal_id = f"UPG-{_now_ms()}"
    # Build the proposal object
    proposal = {
        "id": proposal_id,
        "description": description,
        "target": target,
        "timestamp": _now_ms(),
        "status": "pending"
    }
    resp = {"ok": True, "proposal": proposal}
    # Persist the proposal into the staging directory.  Each proposal is stored
    # as a separate JSON file named after its id.  Logging is best effort; if
    # writing fails, we still return success to the caller.
    try:
        pfile = STAGING_DIR / f"{proposal_id}.json"
        with open(pfile, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)
    except Exception:
        pass
    # Log the proposal event
    try:
        logf = LOGS_DIR / f"{_now_ms()}_propose.json"
        with open(logf, "w", encoding="utf-8") as f:
            json.dump({"event": "propose", "timestamp": _now_ms(), "data": proposal}, f, indent=2)
    except Exception:
        pass
    return resp


def _apply_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a previously approved upgrade.

    In this placeholder implementation, we do not modify any files.  We
    simply return a result indicating that the upgrade would be applied.
    Real code should perform the necessary patching here, guarded by
    compliance checks.
    """
    proposal_id = (payload or {}).get("proposal_id", "")
    # Attempt to locate the proposal in the staging directory and move it to packs
    success = False
    staging_path = STAGING_DIR / f"{proposal_id}.json"
    packs_path = PACKS_DIR / f"{proposal_id}.json"
    try:
        if staging_path.exists():
            # Copy to packs (rename for atomicity) and remove the staging file
            staging_path.replace(packs_path)
        success = True
    except Exception:
        success = False
    # Log the apply event regardless of outcome
    try:
        logf = LOGS_DIR / f"{_now_ms()}_apply.json"
        with open(logf, "w", encoding="utf-8") as f:
            json.dump({"event": "apply", "timestamp": _now_ms(), "proposal_id": proposal_id, "moved": success}, f, indent=2)
    except Exception:
        pass
    return {
        "ok": True,
        "applied": success,
        "proposal_id": proposal_id,
        "notes": "Upgrade applied (placeholder)"
    }


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for the Upgrade Engine.

    Dispatches operations based on the ``op`` field of the incoming
    message.  Non‑destructive operations (SCAN, PROPOSE) do not
    require authorization, whereas the APPLY operation does.  Returns
    a standard response dict with ``ok``, ``op``, ``mid`` and
    ``payload`` keys.
    """
    op = (msg or {}).get("op", "").upper()
    payload = (msg or {}).get("payload", {}) or {}
    mid = msg.get("mid") or f"mid-{_now_ms()}"

    # Non‑destructive operations
    if op == "SCAN":
        res = _scan_impl(payload)
        return {"ok": True, "op": op, "mid": mid, "payload": res}
    if op == "PROPOSE":
        res = _propose_impl(payload)
        return {"ok": True, "op": op, "mid": mid, "payload": res}

    # Destructive operations require governance authorization
    if op == "APPLY":
        auth = payload.get("auth", {})
        if not _auth_ok(auth):
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": "UPGRADE_UNAUTHORIZED",
                "message": "Upgrade operation requires valid Governance authorization token",
                "payload": {"authorized": False}
            }
        res = _apply_impl(payload)
        return {"ok": True, "op": op, "mid": mid, "payload": {"authorized": True, **res}}

    # Unsupported op
    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": "UNSUPPORTED_OP",
        "message": op
    }


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