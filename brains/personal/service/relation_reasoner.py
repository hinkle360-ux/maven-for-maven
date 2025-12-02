"""
Relation Reasoner
=================

This module performs basic reasoning about interpersonal relationships
between Maven and its users.  It uses the relationship memory bank to
determine the current affinity and may update the state based on
affirmations found in conversation history.  When asked whether the
user and Maven are friends (or similar relational queries), this
reasoner provides a grounded answer (YES, NO, or UNSURE) along with
a friendly reply.

The detection of affirmations is deliberately simple: it searches
recent STM records for phrases like "you are my friend".  Future
extensions can implement more robust NLP matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

from api.utils import generate_mid, success_response, error_response

# Import relationship memory helper
try:
    from brains.personal.service.relationship_memory import service_api as rel_mem_api
except Exception:
    rel_mem_api = None  # type: ignore


def _load_stm_records(limit: int = 200) -> list:
    """Load recent STM records from multiple possible memory files.

    To infer relationships based on past conversation, this helper
    aggregates the latest ``limit`` entries from short‑term memory
    files.  It first attempts to read the personal memory file under
    ``brains/personal/memory/stm/records.jsonl``.  If that file does
    not exist or is empty, it falls back to the language brain's STM
    file at ``brains/cognitive/language/memory/stm/records.jsonl``.
    Additional files can be appended to the ``possible_paths`` list
    if future modules record conversation history elsewhere.
    Errors during file access or JSON parsing are silently ignored.
    """
    records: list = []
    # Construct possible STM paths.  The first path points to the
    # personal memory; the second points up to the project root and
    # into the language brain's STM.  Use ``parents`` to compute
    # relative directories without hard coding the project structure.
    try:
        here = Path(__file__).resolve()
        possible_paths = [
            # Personal STM: brains/personal/memory/stm/records.jsonl
            here.parents[1] / "memory" / "stm" / "records.jsonl",
            # Language STM: ascend to project root and into cognitive/language memory
            here.parents[3] / "cognitive" / "language" / "memory" / "stm" / "records.jsonl",
        ]
    except Exception:
        possible_paths = []
    for stm_path in possible_paths:
        try:
            if stm_path.exists():
                with open(stm_path, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                # Append the most recent entries up to the limit.  Keep
                # reading across paths until the limit is reached.
                for line in lines[-limit:]:
                    if len(records) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        rec = {}
                    if isinstance(rec, dict):
                        records.append(rec)
        except Exception:
            continue
        # If we have collected enough records from the first file,
        # stop; otherwise continue to the next path.
        if len(records) >= limit:
            break
    return records


def _detect_affirmation(records: list) -> bool:
    """Detect if the user has previously affirmed friendship.

    Returns True if any record's input or content includes a phrase
    indicating that the user considers Maven a friend.  The matching is
    case insensitive and simple to avoid false positives.
    """
    patterns = [
        r"\byou\s+are\s+my\s+friend\b",
        r"\bi\s+consider\s+you\s+my\s+friend\b",
        r"\byou\s+are\s+my\s+ally\b",
        r"\bi\s+trust\s+you\b",
        r"\bwe\s+are\s+friends\b",
        r"\bwe're\s+friends\b",
        r"\bwe\s+are\s+allies\b",
        r"\byou're\s+my\s+friend\b",
    ]
    for rec in records:
        text = ""
        try:
            text = str(rec.get("input") or rec.get("content") or rec.get("text") or "").lower()
        except Exception:
            text = ""
        for pat in patterns:
            try:
                if re.search(pat, text):
                    return True
            except Exception:
                continue
    return False


def evaluate_relation(query: str, user_id: str = "default_user") -> Dict[str, Any]:
    """Evaluate a relation query and return an appropriate reply.

    This function looks up the user's affinity in the relationship
    memory.  If the affinity is unknown or neutral, it searches recent
    STM records for affirmations like "you are my friend".  When an
    affirmation is found, the relationship memory is updated to mark
    the user as a friend.  The function returns a dictionary with
    ``reply`` and ``affinity`` fields.
    """
    # Default reply and affinity
    reply = "I'm not entirely sure yet, but I value our connection."
    affinity = "unknown"
    # Load relationship states and increment conversation count
    if rel_mem_api:
        try:
            resp = rel_mem_api({"op": "SNAPSHOT"})
            states = resp.get("payload", {}).get("states", {}) if resp.get("ok") else {}
            user_state = states.get(user_id) or {}
            # Extract affinity and normalise
            affinity = str(user_state.get("affinity") or "neutral").lower()
            # Increment conversation count for this user.  This tracks
            # how many relation evaluations have occurred and can be
            # used by downstream affect modules.  Errors during update
            # are ignored to avoid blocking the reply.
            try:
                curr_cnt = int(user_state.get("conversation_count", 0) or 0)
            except Exception:
                curr_cnt = 0
            try:
                rel_mem_api({
                    "op": "UPDATE",
                    "payload": {"user_id": user_id, "update": {"conversation_count": curr_cnt + 1}}
                })
            except Exception:
                pass
        except Exception:
            affinity = "unknown"
    # Determine reply based on affinity
    if affinity == "friend":
        reply = "Yes — I consider you a friend too."
    elif affinity == "ally":
        reply = "I see us as allies and I appreciate your support."
    elif affinity == "trust":
        reply = "I trust you and enjoy our partnership."
    else:
        # Unknown or neutral: check for affirmations in persistent brain storage first
        found_affirmation = False
        try:
            # Try loading memory_librarian to check BRAIN storage
            from brains.cognitive.memory_librarian.service import memory_librarian  # type: ignore
            brain_resp = memory_librarian.service_api({
                "op": "BRAIN_GET",
                "payload": {
                    "scope": "BRAIN",
                    "origin_brain": "memory_librarian",
                    "key": "relationship_status"
                }
            })
            if brain_resp.get("ok") and brain_resp.get("payload", {}).get("found"):
                rel_text = str(brain_resp.get("payload", {}).get("data", {}).get("value", "")).lower()
                # Check if the stored relationship text indicates friendship
                if any(phrase in rel_text for phrase in ["friend", "ally", "we are"]):
                    found_affirmation = True
        except Exception:
            pass
        # Also check WM if not found in BRAIN
        if not found_affirmation:
            try:
                from brains.cognitive.memory_librarian.service import memory_librarian  # type: ignore
                wm_resp = memory_librarian.service_api({
                    "op": "WM_GET",
                    "payload": {"key": "relationship_status"}
                })
                if wm_resp.get("ok"):
                    entries = wm_resp.get("payload", {}).get("entries", [])
                    if entries:
                        rel_text = str(entries[0].get("value", "")).lower()
                        if any(phrase in rel_text for phrase in ["friend", "ally", "we are"]):
                            found_affirmation = True
            except Exception:
                pass
        # Fall back to checking STM records if not found in memory systems
        if not found_affirmation:
            records = _load_stm_records()
            if _detect_affirmation(records):
                found_affirmation = True
        # If affirmation found anywhere, update relationship memory
        if found_affirmation:
            try:
                if rel_mem_api:
                    rel_mem_api({"op": "UPDATE", "payload": {"user_id": user_id, "update": {"affinity": "friend"}}})
            except Exception:
                pass
            reply = "Yes — I consider you a friend too."
            affinity = "friend"
    return {"reply": reply, "affinity": affinity}


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle relation reasoning operations."""
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}
    if op == "REASON":
        query = str(payload.get("query") or "")
        user_id = str(payload.get("user_id") or "default_user")
        result = evaluate_relation(query, user_id)
        return success_response(op, mid, result)
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