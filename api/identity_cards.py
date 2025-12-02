from __future__ import annotations

import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

"""
Identity card helpers for Maven.

This module defines a tiny, dependency‑free card system to store
information about Maven itself (the agent), its primary user and
additional users.  Cards are persisted as JSON files in the
``reports`` directory under the project root.  Functions in this
module perform atomic writes and reads to avoid corrupting files on
crash.  A small denylist prevents placeholder names like ``Bob`` from
being returned as real identities.

These helpers are intentionally simple and do not depend on any
external libraries beyond the Python standard library.
"""

from brains.maven_paths import get_reports_path

# Path to the ``reports`` directory inside the project.
BASE = get_reports_path()

# File locations for the various cards.  These files are created on
# demand by :func:`ensure_cards_exist` if they do not already exist.
MAVEN_CARD_PATH = BASE / "maven_card.json"
PRIMARY_USER_CARD_PATH = BASE / "primary_user_card.json"
OTHER_USERS_IDX_PATH = BASE / "other_users_index.json"

# Names in this set will never be treated as valid identities.  This
# prevents fallback to generic test seeds like "Bob" or "Alice".
_PLACEHOLDER_DENYLIST = {"bob", "alice", "test user"}

def _read_json(path: Path) -> Dict[str, Any]:
    """Read JSON from ``path`` and return a dictionary.

    If the file does not exist or cannot be parsed, an empty dict
    is returned.  Errors are suppressed to avoid exceptions at call
    sites.
    """
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically as pretty‑printed JSON.

    This helper writes to a temporary file in the same directory and
    then replaces the destination atomically.  It ensures the parent
    directory exists and cleans up temporary files on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass

def ensure_cards_exist() -> None:
    """Create card files if they do not already exist.

    This function initialises the Maven self card, the primary user
    card and the stub for other users.  It does nothing if the files
    already exist.  The Maven card records the agent's metadata and the
    ID of the primary user.  The primary user card stores the
    display name, ID and aliases of the person currently using the
    system.  The other users index is a stub list for future use.
    """
    if not MAVEN_CARD_PATH.exists():
        now_iso = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(MAVEN_CARD_PATH, {
            "self": {
                "name": "Maven",
                "first_run_iso": now_iso,
                "build": None,
                "notes": [],
                "primary_user_id": None
            }
        })
    if not PRIMARY_USER_CARD_PATH.exists():
        _atomic_write_json(PRIMARY_USER_CARD_PATH, {
            "id": None,
            "display_name": None,
            "aliases": [],
            "last_confirmed_iso": None,
            "consent_identity": None
        })
    if not OTHER_USERS_IDX_PATH.exists():
        _atomic_write_json(OTHER_USERS_IDX_PATH, {"users": []})

def get_maven_card() -> Dict[str, Any]:
    """Return the contents of the Maven self card.  Creates it on demand."""
    ensure_cards_exist()
    return _read_json(MAVEN_CARD_PATH)

def set_maven_primary_user_id(user_id: Optional[str]) -> None:
    """Set the ``primary_user_id`` field in the Maven card to ``user_id``.

    If the Maven card does not yet exist, it will be created.  The
    ``first_run_iso`` timestamp will be set on first creation.
    """
    ensure_cards_exist()
    card = get_maven_card()
    card.setdefault("self", {})
    card["self"]["primary_user_id"] = user_id
    # If no first run timestamp exists, initialise it now
    first_run = card["self"].get("first_run_iso")
    if not first_run:
        card["self"]["first_run_iso"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(MAVEN_CARD_PATH, card)

def add_maven_note(note: str) -> None:
    """Append a timestamped note to Maven's self card."""
    ensure_cards_exist()
    card = get_maven_card()
    notes: List[Dict[str, Any]] = card.get("self", {}).get("notes") or []
    notes.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "note": note
    })
    card.setdefault("self", {})["notes"] = notes
    _atomic_write_json(MAVEN_CARD_PATH, card)

def get_primary_user_card() -> Dict[str, Any]:
    """Return the primary user's card.  Creates it on demand."""
    ensure_cards_exist()
    return _read_json(PRIMARY_USER_CARD_PATH)

def set_primary_user(display_name: str, user_id: Optional[str] = None, consent: Optional[bool] = True) -> None:
    """
    Create or update the primary user card with the given display name.

    If ``user_id`` is not provided, a simple slug is generated from the
    display name.  ``consent`` records whether the user approved
    storage of their identity; if None, the consent field is left unchanged.
    Placeholder names (e.g. 'Bob') are ignored to prevent test seeds
    from being recorded.
    """
    if not display_name:
        return
    name = display_name.strip()
    if not name or name.lower() in _PLACEHOLDER_DENYLIST:
        return
    ensure_cards_exist()
    uid = user_id or f"user:{name.lower().replace(' ', '-')}"
    now = datetime.now(timezone.utc).isoformat()
    card = {
        "id": uid,
        "display_name": name,
        "aliases": [name],
        "last_confirmed_iso": now,
        "consent_identity": bool(consent) if consent is not None else None
    }
    _atomic_write_json(PRIMARY_USER_CARD_PATH, card)
    # Update Maven card to reference the new primary user
    set_maven_primary_user_id(uid)

def add_primary_user_alias(alias: str) -> None:
    """Add an alias to the primary user's card."""
    if not alias:
        return
    ensure_cards_exist()
    card = get_primary_user_card()
    aliases = set(card.get("aliases") or [])
    aliases.add(alias.strip())
    card["aliases"] = sorted(a for a in aliases if a)
    card["last_confirmed_iso"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(PRIMARY_USER_CARD_PATH, card)

def resolve_primary_user_name() -> Optional[str]:
    """Return the primary user's display name, or None if not set.

    Names in the placeholder denylist are ignored.  Leading/trailing
    whitespace is stripped before comparison.  If no valid name is
    present, ``None`` is returned.
    """
    ensure_cards_exist()
    card = get_primary_user_card()
    name = (card.get("display_name") or "") or ""
    if not isinstance(name, str):
        return None
    nm = name.strip()
    if not nm:
        return None
    if nm.lower() in _PLACEHOLDER_DENYLIST:
        return None
    return nm

def resolve_maven_primary_user_id() -> Optional[str]:
    """Return the ID of the primary user recorded in the Maven card."""
    ensure_cards_exist()
    card = get_maven_card()
    try:
        return (card.get("self") or {}).get("primary_user_id") or None
    except Exception:
        return None