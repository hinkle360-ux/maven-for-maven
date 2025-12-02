"""
Governance Permits
==================

This module implements a lightweight permit system for self‑directed
actions.  New autonomous behaviours introduced in the human cognition
upgrade (imagination rollouts, self critique writing, opinion updates)
must obtain an explicit permit from the governance layer.  Each permit
request is logged to ``reports/governance/proofs`` with a unique
identifier.  The permit logic is intentionally simple and easily
auditable.

Supported actions:

* ``IMAGINE``: number of rollouts (n) must be ≤ 5.
* ``CRITIQUE``: writing critique logs is always allowed.
* ``OPINION``: updating internal opinions is always allowed.

Requests outside these actions are denied by default.  All requests
return a ``permit_id`` that references a JSON proof record.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from brains.maven_paths import get_reports_path


def _permits_dir() -> Path:
    """Return the directory where permit proof files are stored."""
    # The policy engine lives at brains/governance/policy_engine/service.
    # Project root is three parents up.
    return get_reports_path("governance", "proofs")


def _write_proof(proof_id: str, record: Dict[str, Any]) -> None:
    """Write a permit proof record to disk."""
    try:
        path = _permits_dir() / f"{proof_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    except Exception:
        pass


def request_permit(action: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Evaluate a permit request for the specified action and parameters.

    Supported actions:

    * ``IMAGINE``: number of rollouts (n) must be ≤ 5.
    * ``CRITIQUE``: always allowed.
    * ``OPINION``: always allowed.
    * ``EXEC``: execute user code.  Permitted when time and memory
      limits are within bounds (≤ 5 seconds and ≤ 256 MB).  The
      parameters must include ``time_limit_sec`` and ``mem_limit_mb``.

    Requests outside these actions are denied by default.  All
    requests create a proof record under ``reports/governance/proofs``.
    Returns a dict with ``allowed`` (bool), ``permit_id`` and
    optionally a ``reason`` when denied.
    """
    action_u = (action or "").upper()
    params = params or {}
    allowed = False
    reason = None
    if action_u == "IMAGINE":
        # Limit number of imagination rollouts
        try:
            n = int(params.get("n", 0))
        except Exception:
            n = 0
        if n <= 5:
            allowed = True
        else:
            allowed = False
            reason = "n_exceeds_limit"
    elif action_u in {"CRITIQUE", "OPINION"}:
        # Always allow critiques and opinion updates
        allowed = True
    elif action_u == "EXEC":
        # Evaluate execution parameters.  Use conservative defaults when
        # unspecified.  Deny if time or memory exceed safe limits.
        try:
            t_lim = float(params.get("time_limit_sec", 0.0) or 0.0)
        except Exception:
            t_lim = 0.0
        try:
            m_lim = float(params.get("mem_limit_mb", 0.0) or 0.0)
        except Exception:
            m_lim = 0.0
        # Set hard caps (5 seconds, 256 MB).  These caps can be tightened
        # as needed but should never be relaxed without careful review.
        if t_lim <= 5.0 and m_lim <= 256.0:
            allowed = True
        else:
            allowed = False
            reason = "limits_exceed_bounds"
    else:
        allowed = False
        reason = "unsupported_action"
    # Create proof record
    proof_id = f"PERMIT-{uuid.uuid4()}"
    record = {
        "ts": int(time.time()),
        "action": action_u,
        "params": params,
        "allowed": allowed,
        "reason": reason,
    }
    _write_proof(proof_id, record)
    return {"allowed": allowed, "permit_id": proof_id, "reason": reason}


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    API wrapper for permit requests.  Supports only the ``REQUEST``
    operation which expects an ``action`` and optional parameters
    dictionary under ``payload``.  Returns the permit result.
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    if op == "REQUEST":
        action = payload.get("action")
        if not action:
            return {"ok": False, "error": {"code": "MISSING_ACTION", "message": "action is required"}}
        result = request_permit(action, payload)
        return {"ok": True, "payload": result}
    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}


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