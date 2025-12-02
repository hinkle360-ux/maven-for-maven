"""
Identity Consolidator
=====================

This module promotes stable self‑beliefs from the personal identity
snapshot into long‑term memory.  During consolidation Maven examines
the current identity snapshot and compares it against the existing
long‑term self factual bank.  Facts that are not yet present in
long‑term memory are added along with a confidence score and
promotion timestamp.  The consolidator does not remove any entries
from long‑term memory; retention and decay policies can be applied in
future iterations.

Invocation of the consolidator is expected to occur at the tail end
of the cognitive pipeline (e.g. Stage 14) or on a scheduled basis.
It exposes a simple API that accepts a ``CONSOLIDATE`` operation to
trigger consolidation and returns the updated long‑term record.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

from api.utils import generate_mid, success_response, error_response
from brains.maven_paths import get_brains_path


def _identity_snapshot_path() -> Path:
    """Return the path to the identity snapshot JSON.

    The identity snapshot resides under ``brains/personal/memory``.  This
    helper ascends from the service module to the ``personal`` package
    directory and then appends the memory location.
    """
    return get_brains_path("personal", "memory", "identity_snapshot.json")


def _ltm_path() -> Path:
    """Return the path to the long‑term self factual bank JSON.

    Long‑term self facts are stored alongside the personal memory in
    ``ltm/self_factual_bank.json``.  This helper ascends one level up
    to ``brains/personal`` and builds the full path from there.
    """
    return get_brains_path("personal", "memory", "ltm", "self_factual_bank.json")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def evaluate_promotions() -> Dict[str, Any]:
    """Promote eligible identity facts into long‑term memory.

    The current implementation promotes any key/value pair present in
    the identity snapshot but absent from the long‑term self factual
    bank.  Each promoted entry stores the original value along with a
    fixed confidence score and a promotion timestamp.
    """
    identity_data = _load_json(_identity_snapshot_path())
    ltm_data = _load_json(_ltm_path()) or {}
    updated = False
    for key, value in identity_data.items():
        # Skip non‑primitive types to avoid serialisation issues
        if key not in ltm_data and not isinstance(value, dict):
            ltm_data[key] = {
                "value": value,
                "confidence": 1.0,
                "promoted_ts": time.time()
            }
            updated = True
    if updated:
        _save_json(_ltm_path(), ltm_data)
    return ltm_data


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle service operations for identity consolidation."""
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    if op == "CONSOLIDATE":
        ltm = evaluate_promotions()
        return success_response(op, mid, {"ltm": ltm})
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