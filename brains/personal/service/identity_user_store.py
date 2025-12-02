"""
identity_user_store.py
=======================

This module provides a minimal, dependency‑free persistent store for
tracking the primary user identity across sessions.  It writes a JSON
document to ``reports/personal/identity.json`` under the project root
and exposes a small API similar to the high‑level plan:

* ``GET()`` – return a dictionary with the current identity fields.
* ``SET(name)`` – record a new primary user name and update the
  timestamp and confidence.  If no record exists, one is created.
* ``ADD_ALIAS(name)`` – append an alias to the identity record
  without changing the primary name.
* ``CONFIRM()`` – bump the confirmation timestamp and confidence.

The data schema persisted to disk matches the suggested format in the
upgrade plan.  All functions silently ignore I/O errors to avoid
breaking the main pipeline.  The store never depends on any external
libraries beyond the Python standard library and does not import
anything from other Maven modules to minimise coupling.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Any, Optional

from brains.maven_paths import get_reports_path

# Path to the persistent identity file.  Stored under reports/personal.
IDENTITY_FILE = get_reports_path("personal", "identity.json")

def _load_identity() -> Dict[str, Any]:
    """Load the identity record from disk.

    Returns an empty dictionary if the file is missing or cannot be
    parsed.  Errors are suppressed to avoid exceptions.
    """
    try:
        if not IDENTITY_FILE.exists():
            return {}
        with IDENTITY_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _save_identity(data: Dict[str, Any]) -> None:
    """Persist the identity record to disk.

    Creates the parent directory if it does not exist.  Any errors
    during writing are silently ignored to avoid disrupting callers.
    """
    try:
        IDENTITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with IDENTITY_FILE.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def GET() -> Dict[str, Any]:
    """Return the current identity record.

    Returns a dictionary with keys ``name``, ``aliases``, ``confidence``
    and ``last_confirmed_ts`` if present.  Missing keys default to
    sensible values.  If no record exists on disk, an empty
    dictionary is returned.
    """
    data = _load_identity()
    primary = data.get("primary_user", {}) if isinstance(data.get("primary_user"), dict) else {}
    return {
        "name": str(primary.get("name") or "").strip(),
        "aliases": list(primary.get("aliases") or []),
        "confidence": float(primary.get("confidence") or 0.0),
        "last_confirmed_ts": float(primary.get("last_confirmed_ts") or 0.0),
    }


def SET(name: str) -> None:
    """Set the primary user name.

    Creates or updates the identity record with the provided name.
    The first time ``SET`` is called, ``first_seen_ts`` is set to
    the current timestamp.  ``last_confirmed_ts`` is always updated to
    now.  ``confidence`` is set to 1.0 on update.  Names consisting
    only of whitespace are ignored.
    """
    try:
        nm = str(name or "").strip()
        if not nm:
            return
        data = _load_identity()
        now = time.time()
        primary = data.get("primary_user") if isinstance(data.get("primary_user"), dict) else {}
        if not primary.get("first_seen_ts"):
            primary["first_seen_ts"] = now
        primary["name"] = nm
        primary["last_confirmed_ts"] = now
        primary["confidence"] = 1.0
        aliases = set([nm] + list(primary.get("aliases") or []))
        primary["aliases"] = sorted(a for a in aliases if a)
        data["primary_user"] = primary
        _save_identity(data)
    except Exception:
        pass


def ADD_ALIAS(alias: str) -> None:
    """Add an alias for the primary user.

    The alias is appended to the list of aliases if it does not
    already exist and is not equal to the primary name.  Empty or
    whitespace aliases are ignored.
    """
    try:
        nm = str(alias or "").strip()
        if not nm:
            return
        data = _load_identity()
        primary = data.get("primary_user") if isinstance(data.get("primary_user"), dict) else {}
        primary_name = str(primary.get("name") or "").strip()
        aliases = set(primary.get("aliases") or [])
        if nm and nm != primary_name:
            aliases.add(nm)
            primary["aliases"] = sorted(a for a in aliases if a)
            data["primary_user"] = primary
            _save_identity(data)
    except Exception:
        pass


def CONFIRM() -> None:
    """Mark the identity as confirmed.

    Updates the ``last_confirmed_ts`` to the current time and
    increments the confidence by a small amount up to a maximum of
    1.0.  If no identity exists, this function does nothing.
    """
    try:
        data = _load_identity()
        primary = data.get("primary_user") if isinstance(data.get("primary_user"), dict) else {}
        if not primary:
            return
        now = time.time()
        primary["last_confirmed_ts"] = now
        try:
            conf = float(primary.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        conf += 0.05
        if conf > 1.0:
            conf = 1.0
        primary["confidence"] = conf
        data["primary_user"] = primary
        _save_identity(data)
    except Exception:
        pass