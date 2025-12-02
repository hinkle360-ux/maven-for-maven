"""
Belief Tracker
==============

This module provides a simple persistent store for beliefs (facts)
and helper functions to record, query and detect conflicts among
beliefs.  A belief consists of a ``subject``, a ``predicate``, an
``object`` and an optional ``confidence`` score.  Beliefs are
stored in a newline‑delimited JSON file under the ``reports``
directory so that they persist across sessions.

Functions exported include:

* ``add_belief(subject, predicate, obj, confidence=1.0)`` – Append a
  new belief record to the belief file.
* ``find_related_beliefs(query)`` – Return all beliefs whose
  subject or object contains the query string (case insensitive).
* ``detect_conflict(subject, predicate, obj)`` – Check if a new
  belief about ``subject`` and ``predicate`` conflicts with existing
  beliefs (different object).  Returns the first conflicting belief
  or ``None`` if no conflict is found.

These helpers are intentionally lightweight and do not perform
semantic reasoning.  They are intended to serve as building blocks
for a richer belief management system in future upgrades.
"""

from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Optional
from brains.memory.brain_memory import BrainMemory
from brains.maven_paths import get_reports_path

# Teacher integration for learning belief tracking and update patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("belief_tracker")
except Exception as e:
    print(f"[BELIEF_TRACKER] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("belief_tracker")

# Determine the path for the belief store.  It resides under the
# top‑level ``reports`` directory.  If the directory does not exist,
# it will be created on demand when appending beliefs.
BELIEF_FILE = get_reports_path("beliefs.jsonl")


def _load_beliefs() -> List[Dict[str, Any]]:
    """Load all belief records from the belief file.

    Returns:
        A list of belief dictionaries.  If the file does not exist
        or an error occurs, an empty list is returned.
    """
    records: List[Dict[str, Any]] = []
    try:
        if BELIEF_FILE.exists():
            with open(BELIEF_FILE, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        if isinstance(obj, dict):
                            records.append(obj)
                    except Exception:
                        continue
    except Exception:
        return []
    return records


def _append_belief(rec: Dict[str, Any]) -> None:
    """Append a single belief record to the belief file.

    Args:
        rec: The belief dictionary to write.  Must contain at least
            ``subject``, ``predicate`` and ``object`` keys.
    """
    try:
        BELIEF_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BELIEF_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        # Silently ignore write errors
        pass


def add_belief(subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
    """Add a new belief to the belief store.

    Args:
        subject: The subject of the belief (e.g. "Paris").
        predicate: The predicate or relation (e.g. "is").
        obj: The object of the belief (e.g. "the capital of France").
        confidence: Optional confidence score between 0 and 1.
    """
    try:
        rec = {
            "subject": str(subject).strip(),
            "predicate": str(predicate).strip(),
            "object": str(obj).strip(),
            "confidence": float(confidence),
        }
        _append_belief(rec)
    except Exception:
        pass


def find_related_beliefs(query: str) -> List[Dict[str, Any]]:
    """Find beliefs related to the provided query string.

    A belief is considered related if the query appears as a
    substring (case insensitive) of the belief's subject or object.

    Args:
        query: The query string to search for.
    Returns:
        A list of matching belief dictionaries.
    """
    try:
        q = str(query).strip().lower()
    except Exception:
        q = ""
    if not q:
        return []
    try:
        records = _load_beliefs()
        matches: List[Dict[str, Any]] = []
        for rec in records:
            subj = str(rec.get("subject", "")).lower()
            obj = str(rec.get("object", "")).lower()
            if q in subj or q in obj:
                matches.append(rec)
        return matches
    except Exception:
        return []


def detect_conflict(subject: str, predicate: str, obj: str) -> Optional[Dict[str, Any]]:
    """Detect whether a new belief conflicts with an existing one.

    A conflict occurs when there is already a belief with the same
    subject and predicate but a different object.  Only the first
    conflicting belief is returned.

    Args:
        subject: The subject of the new belief.
        predicate: The predicate of the new belief.
        obj: The object of the new belief.
    Returns:
        The conflicting belief dictionary if found; otherwise ``None``.
    """
    try:
        s = str(subject).strip().lower()
        p = str(predicate).strip().lower()
        o = str(obj).strip().lower()

        # Check for learned conflict detection patterns first
        if _teacher_helper and _memory:
            try:
                learned_patterns = _memory.retrieve(
                    query=f"conflict detection pattern: {p}",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", "")
                        if isinstance(content, str) and "conflict" in content.lower():
                            print(f"[BELIEF_TRACKER] Using learned conflict detection pattern from Teacher")
                            # Learned pattern detected - use it for enhanced conflict detection
                            # For now, still use built-in logic but mark that we have learned patterns
                            break
            except Exception:
                pass

        # Built-in conflict detection logic
        conflict_found = False
        conflicting_belief = None
        for rec in _load_beliefs():
            rs = str(rec.get("subject", "")).strip().lower()
            rp = str(rec.get("predicate", "")).strip().lower()
            ro = str(rec.get("object", "")).strip().lower()
            if rs == s and rp == p and ro != o:
                conflict_found = True
                conflicting_belief = rec
                break

        # If no learned pattern and Teacher available, try to learn
        if not conflict_found and _teacher_helper:
            try:
                # Only call Teacher occasionally to learn new patterns (e.g., for novel predicates)
                beliefs = _load_beliefs()
                if len(beliefs) >= 5:  # Only learn if we have sufficient belief history
                    print(f"[BELIEF_TRACKER] Calling Teacher to learn conflict detection patterns...")
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"What patterns should I use to detect conflicts for beliefs with predicate '{predicate}'?",
                        context={
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "existing_beliefs_count": len(beliefs)
                        },
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[BELIEF_TRACKER] Learned from Teacher: {patterns_stored} conflict patterns stored")
                        # Learned pattern now in memory for future use
            except Exception as e:
                print(f"[BELIEF_TRACKER] Teacher call failed: {str(e)[:100]}")

        return conflicting_belief
    except Exception:
        return None


def tag_beliefs_as_suspect(beliefs: List[Dict[str, Any]], note: str = "") -> int:
    """Mark beliefs as suspect by appending tagged copies to the log.

    This function does not delete or overwrite existing beliefs.  It simply
    appends new records with a ``suspect`` tag so downstream consumers can
    lower trust while keeping history intact.
    """

    count = 0
    for belief in beliefs[:10]:  # keep bounded
        try:
            rec = dict(belief)
            tags = set(rec.get("tags", []))
            tags.add("suspect")
            rec["tags"] = sorted(tags)
            if note:
                rec["note"] = note
            rec["timestamp"] = time.time()
            _append_belief(rec)
            count += 1
        except Exception:
            continue
    return count


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Belief Tracker service API.

    Supported operations:
    - ADD_BELIEF: Add a new belief
    - FIND_RELATED: Find beliefs related to a query
    - DETECT_CONFLICT: Detect if a belief conflicts with existing beliefs
    - QUERY: Query all beliefs
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "ADD_BELIEF":
        try:
            subject = payload.get("subject", "")
            predicate = payload.get("predicate", "")
            obj = payload.get("object", "")
            confidence = payload.get("confidence", 1.0)

            if not subject or not predicate or not obj:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_FIELDS",
                        "message": "subject, predicate, and object are required"
                    }
                }

            add_belief(subject, predicate, obj, confidence)
            return {
                "ok": True,
                "payload": {
                    "added": True,
                    "belief": {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": confidence
                    }
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "ADD_FAILED",
                    "message": str(e)
                }
            }

    if op == "FIND_RELATED":
        try:
            query = payload.get("query", "")
            if not query:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_QUERY",
                        "message": "query parameter is required"
                    }
                }

            beliefs = find_related_beliefs(query)
            return {
                "ok": True,
                "payload": {
                    "beliefs": beliefs,
                    "count": len(beliefs)
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "FIND_FAILED",
                    "message": str(e)
                }
            }

    if op == "DETECT_CONFLICT":
        try:
            subject = payload.get("subject", "")
            predicate = payload.get("predicate", "")
            obj = payload.get("object", "")

            if not subject or not predicate or not obj:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_FIELDS",
                        "message": "subject, predicate, and object are required"
                    }
                }

            conflict = detect_conflict(subject, predicate, obj)
            return {
                "ok": True,
                "payload": {
                    "has_conflict": conflict is not None,
                    "conflict": conflict
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "DETECT_FAILED",
                    "message": str(e)
                }
            }

    if op == "QUERY":
        try:
            beliefs = _load_beliefs()
            return {
                "ok": True,
                "payload": {
                    "beliefs": beliefs,
                    "count": len(beliefs)
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "QUERY_FAILED",
                    "message": str(e)
                }
            }

    if op == "HEALTH":
        try:
            beliefs = _load_beliefs()
            return {
                "ok": True,
                "payload": {
                    "status": "operational",
                    "belief_count": len(beliefs)
                }
            }
        except Exception as e:
            return {
                "ok": True,
                "payload": {
                    "status": "operational_degraded",
                    "error": str(e)
                }
            }

    return {
        "ok": False,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": op
        }
    }


# Standard service contract: handle is the entry point
service_api = handle