"""
Self‑DMN Personal Integration
============================

This module provides higher level reflection and promotion operations
that unify the personal brain (preferences and dynamic memory) with the
identity journal (persistent self model and personality snapshot).

These operations allow the Self‑DMN to introspect the personal brain
for emerging preferences and to update the identity snapshot accordingly.

Available operations:

  - PERSONAL_REFLECT: return top preference candidates from the personal brain.
  - PERSONAL_PROMOTE: promote stable preferences into the personality snapshot.
  - PERSONAL_DEMOTE: stub operation for future demotion support.
  - PERSONAL_CONSOLIDATE: stub operation for consolidating promotions.

The API signature matches other service_api modules, returning
success_response or error_response dictionaries.
"""

from __future__ import annotations

from typing import Dict, Any, List

# Attempt to import the personal brain and identity journal.  When these
# modules are unavailable (e.g. incomplete installation), the module will
# still import, but personal operations will return an error.
try:
    # personal_brain and identity_journal live under brains.personal.service
    from brains.personal.service import personal_brain, identity_journal  # type: ignore
except Exception:
    personal_brain = None  # type: ignore
    identity_journal = None  # type: ignore


def reflect(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import error_response  # type: ignore
    from api.utils import success_response  # type: ignore
    from api.utils import generate_mid  # type: ignore
    """
    Reflect on the personal brain by returning the top preference candidates.

    Args:
        msg: A message dict containing at least an 'op' and optional payload.

    Returns:
        A success response with the top candidates or an error response.
    """
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    # Ensure personal brain is available
    if personal_brain is None:
        return error_response(op, mid, "ERROR", "personal brain unavailable")
    try:
        # Determine limit from payload, default to 10
        payload = msg.get("payload") or {}
        limit = int(payload.get("limit", 10))
        likes: List[dict] = personal_brain._top_likes(limit=limit)  # type: ignore[attr-defined]
        return success_response(op, mid, {"candidates": likes})
    except Exception as exc:
        return error_response(op, mid, "ERROR", f"reflection failed: {exc}")


def promote(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Promote stable preferences into the identity snapshot's personality.

    Candidates are taken from the personal brain's top likes. When a candidate's
    ``score_boost`` meets or exceeds the threshold (default 0.8), its subject
    is appended to the 'style' list in the personality snapshot. If the
    personality snapshot is absent or not a dict, it will be initialised.

    Args:
        msg: A message dict containing an 'op' and optional payload.

    Returns:
        A success response indicating whether any promotions were applied.
    """
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    if personal_brain is None or identity_journal is None:
        return error_response(op, mid, "ERROR", "personal brain or identity journal unavailable")
    try:
        payload = msg.get("payload") or {}
        threshold = float(payload.get("threshold", 0.8))
        limit = int(payload.get("limit", 10))
        # Fetch candidates from personal brain
        candidates: List[dict] = personal_brain._top_likes(limit=limit)  # type: ignore[attr-defined]
        # Load current identity snapshot
        snap_res = identity_journal.service_api({"op": "SNAPSHOT", "mid": mid})  # type: ignore[attr-defined]
        snapshot: Dict[str, Any] = (snap_res.get("payload") or {})  # type: ignore[assignment]
        personality: Dict[str, Any] = snapshot.get("personality_snapshot") or {}
        updated = False
        # Ensure style is a list
        style = personality.get("style", [])
        if not isinstance(style, list):
            style = [style] if style else []
        # Iterate candidates and promote those meeting the threshold
        for cand in candidates:
            subj = cand.get("subject")
            try:
                score = float(cand.get("score_boost", 0))
            except Exception:
                score = 0.0
            if subj and score >= threshold and subj not in style:
                style.append(subj)
                updated = True
        if updated:
            personality["style"] = style
            update_payload = {"update": {"personality_snapshot": personality}}
            identity_journal.service_api({"op": "UPDATE", "mid": mid, "payload": update_payload})  # type: ignore[attr-defined]
        return success_response(op, mid, {"promoted": updated, "personality": personality})
    except Exception as exc:
        return error_response(op, mid, "ERROR", f"promotion failed: {exc}")


def demote(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub demotion operation. Returns success without demoting any preferences.

    Demotion logic can be implemented in the future to remove subjects from
    the personality snapshot when preferences are no longer stable or relevant.
    """
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    return success_response(op, mid, {"demoted": False})


def consolidate(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub consolidation operation. Returns success indicating a no-op.

    Consolidation can be used to aggregate promotion decisions over time or
    prune outdated preferences. For now, it simply returns success.
    """
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    return success_response(op, mid, {"consolidated": True})


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch Personal operations based on the 'op' field of the message.

    Supported operations:

      - PERSONAL_REFLECT: call reflect(...)
      - PERSONAL_PROMOTE: call promote(...)
      - PERSONAL_DEMOTE: call demote(...)
      - PERSONAL_CONSOLIDATE: call consolidate(...)

    Args:
        msg: A dict containing 'op' and optional payload.

    Returns:
        A success or error response dict.
    """
    op = (msg or {}).get("op", " ").upper()
    if op == "PERSONAL_REFLECT":
        return reflect(msg)
    if op == "PERSONAL_PROMOTE":
        return promote(msg)
    if op == "PERSONAL_DEMOTE":
        return demote(msg)
    if op == "PERSONAL_CONSOLIDATE":
        return consolidate(msg)
    mid = msg.get("mid") or generate_mid()
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