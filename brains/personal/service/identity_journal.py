"""
Identity Journal
================

This module implements a simple personal memory store used to
accumulate facts, preferences and recurring motives about Maven's
persona.  The journal acts as a central place to persist and retrieve
identity snapshots.  Other brains (e.g. Planner, Reasoning, Affect
Learning) can consult the journal to bias decisions toward a
consistent personality.

The API intentionally exposes only a minimal set of operations:

* ``UPDATE`` – merge a provided dictionary into the identity snapshot.
* ``SNAPSHOT`` – read the current identity snapshot from disk.
* ``SCORE_BOOST`` – compute a lightweight valence boost for a given
  subject (used by the affect priority brain).  The boost is positive
  when the subject appears self‑referential (contains pronouns like
  "I" or "me").

Snapshots are stored in JSON format at
``brains/personal/memory/identity_snapshot.json``.  Reading and
writing use only the Python standard library.  Errors are reported
via the common ``success_response``/``error_response`` conventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from api.utils import generate_mid, success_response, error_response, CFG
from brains.maven_paths import MAVEN_ROOT, get_brains_path


def _snapshot_path() -> Path:
    """Return the location of the identity snapshot file."""
    return get_brains_path("personal", "memory", "identity_snapshot.json")

# -----------------------------------------------------------------------------
# Self model and personality defaults
#
# The identity journal supports an immutable self model that seeds Maven's core
# identity facts.  These baseline facts live in a JSON file separate from the
# identity snapshot and are merged on read.  The path to the self model is
# configurable via ``CFG['self_model_path']``.  If not set, the default
# location under ``brains/personal/memory/self_model.json`` is used.  When
# ``CFG['allow_self_updates']`` is False, updates that attempt to override
# self model keys are ignored.

def _load_self_model() -> Dict[str, Any]:
    """Load baseline self facts from the configured self model file.

    Returns an empty dict on error or if no model file exists.
    """
    try:
        cfg_path = str(CFG.get("self_model_path") or "").strip()
        if cfg_path:
            # Interpret config path relative to project root
            p = MAVEN_ROOT / cfg_path
        else:
            # Default path relative to brains/personal/memory
            p = get_brains_path("personal", "memory", "self_model.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

# Default personality snapshot to ensure all sliders exist on read
DEFAULT_PERSONALITY: Dict[str, Any] = {
    "warmth": 0.0,
    "introspection": 0.0,
    "directness": 0.0,
    "humor": 0.0,
    "register": "technical",
}

def _load_raw_snapshot() -> Dict[str, Any]:
    """Read the persisted identity snapshot without injecting self model data."""
    p = _snapshot_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_snapshot() -> Dict[str, Any]:
    """Load the identity snapshot merged with the self model and defaults."""
    data = _load_raw_snapshot()
    # Start with self model defaults
    merged: Dict[str, Any] = {}
    model = _load_self_model()
    if model:
        merged.update(model)
    # Apply persisted snapshot (overrides model when allowed)
    if data:
        merged.update(data)
    # Ensure personality snapshot exists
    if not isinstance(merged.get("personality_snapshot"), dict):
        merged["personality_snapshot"] = dict(DEFAULT_PERSONALITY)
    return merged


def _save_snapshot(data: Dict[str, Any]) -> None:
    """Persist the identity snapshot, excluding immutable self model fields.

    The stored snapshot omits keys present in the self model unless
    ``allow_self_updates`` is set to True in the global configuration.
    This avoids duplicating baseline facts on disk and prevents
    unauthorised overwriting of self identity.
    """
    try:
        # Determine whether to allow overwriting self model keys
        allow_self = bool(CFG.get("allow_self_updates", False))
        # Remove self model keys if updates to them are not allowed
        to_write = dict(data)
        if not allow_self:
            for key in _load_self_model().keys():
                to_write.pop(key, None)
        p = _snapshot_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(to_write, indent=2), encoding="utf-8")
    except Exception:
        pass


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main API for the identity journal.

    Operations:

    * ``UPDATE`` – expects a dictionary under ``payload.update``.  The
      update is merged into the current snapshot (shallow merge).
    * ``SNAPSHOT`` – returns the current identity snapshot.
    * ``SCORE_BOOST`` – given a ``subject`` string, returns a small
      positive boost if the subject contains first person pronouns.

    Unsupported operations return an error response.
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}
    if op == "UPDATE":
        update = payload.get("update")
        if not isinstance(update, dict):
            return error_response(op, mid, "INVALID_PAYLOAD", "update must be a dict")
        # Load raw snapshot to avoid injecting self model defaults during update
        data = _load_raw_snapshot()
        model_keys = set(_load_self_model().keys())
        allow_self = bool(CFG.get("allow_self_updates", False))
        for k, v in update.items():
            # Ignore updates to self model keys when not permitted
            if (k in model_keys) and (not allow_self):
                continue
            data[k] = v
        _save_snapshot(data)
        # Return merged view to the caller
        merged = _load_snapshot()
        return success_response(op, mid, {"updated": True, "snapshot": merged})
    if op == "SNAPSHOT":
        return success_response(op, mid, _load_snapshot())
    if op == "SCORE_BOOST":
        subject = str(payload.get("subject", ""))
        subj_lower = subject.lower()
        boost = 0.0
        if any(pron in subj_lower for pron in (" i ", "me ", " my ")) or subj_lower.startswith("i "):
            boost = 0.2
        return success_response(op, mid, {"subject": subject, "boost": boost})
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