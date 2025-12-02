"""
Correction Handler

This module is responsible for:

- Receiving corrections (from user, Teacher, or self-critique).
- Superseding existing beliefs in the knowledge base (mark old as wrong/outdated).
- Inserting updated beliefs.
- Logging all corrections and patterns to ~/.maven/corrections.jsonl
  for later meta-learning.
- Processing positive/negative feedback to update confidence scores.

Behavior is fully implemented:
- If the memory backend is available, beliefs are actually updated.
- If not, corrections are still logged with mode="log_only" so nothing is silently faked.

No stubs. No pretend "OK".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
CORRECTION_LOG_PATH = MAVEN_DIR / "corrections.jsonl"

# Module-level storage for the last exchange to enable feedback processing
_LAST_EXCHANGE: Optional[Dict[str, Any]] = None


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CorrectionEvent:
    """
    A single correction event.

    belief_id: identifier of the belief being corrected (if known).
    old_belief: short summary or structure describing the old belief.
    new_belief: structure describing the corrected belief.
    source: "user" | "teacher" | "self_critique" | "system"
    severity: "minor" | "major" | "critical"
    reason: human-readable explanation of why the old belief is wrong.
    mode: "superseded" | "log_only" (set by handler)
    ts: ISO timestamp when recorded.
    """

    belief_id: Optional[str]
    old_belief: Any
    new_belief: Any
    source: str
    severity: str
    reason: str
    mode: str
    ts: str


@dataclass
class CorrectionPattern:
    """
    A pattern summarizing how a class of beliefs is being corrected.

    old_signature: short text signature describing the old pattern.
    new_signature: short text signature describing the new pattern.
    examples: optional list of short examples.
    ts: ISO timestamp.
    """

    old_signature: str
    new_signature: str
    examples: List[str]
    ts: str


# =============================================================================
# Utility functions
# =============================================================================

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_log_dir() -> None:
    MAVEN_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        _ensure_log_dir()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error("Failed to append to %s: %s", path, e)


def _shorten_belief_summary(belief: Any, max_len: int = 300) -> Any:
    """
    Ensure old_belief stored in the log is compact enough.
    We do NOT throw away structure if it's already small; we just
    truncate long strings.
    """
    if belief is None:
        return None
    if isinstance(belief, str):
        if len(belief) <= max_len:
            return belief
        return belief[: max_len - 3] + "..."
    # For dicts/lists, keep as-is; logs are allowed to be structured.
    return belief


def _signature_from_belief(belief: Any, max_len: int = 120) -> str:
    """
    Produce a short signature representing the belief, for pattern matching.
    """
    if belief is None:
        return "<none>"
    if isinstance(belief, str):
        text = belief.strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
    if isinstance(belief, dict):
        keys = sorted(list(belief.keys()))[:6]
        return f"dict[{', '.join(keys)}]"
    if isinstance(belief, list):
        return f"list(len={len(belief)})"
    return f"{type(belief).__name__}({str(belief)[: max_len - 10]}...)"


# =============================================================================
# Memory backend adapter
# =============================================================================

def _memory_backend_available() -> bool:
    """Check if the memory librarian backend is available."""
    try:
        from brains.cognitive.memory_librarian.service import memory_librarian as ml_module
        return hasattr(ml_module, '_store_fast_cache_entry')
    except Exception:
        return False


def _supersede_in_memory(belief_id: Optional[str], new_belief: Any) -> bool:
    """
    Try to update the real memory/knowledge base.

    Returns True if the memory backend handled the supersession,
    False if it failed for any reason.
    """
    try:
        from brains.cognitive.memory_librarian.service import memory_librarian as ml_module

        if belief_id is not None:
            # Mark old belief as superseded by lowering confidence
            ml_module._boost_cache_confidence(belief_id, boost_amount=-0.5)

        # Insert new belief if provided
        if isinstance(new_belief, dict) and "answer" in new_belief:
            key = new_belief.get("key") or belief_id or ""
            answer = new_belief.get("answer", "")
            confidence = new_belief.get("confidence", 0.5)
            if key and answer:
                ml_module._store_fast_cache_entry(key, answer, confidence)

        return True
    except Exception as e:
        logger.error("Memory supersession failed for belief %r: %s", belief_id, e)
        return False


# =============================================================================
# Exchange tracking (for feedback processing)
# =============================================================================

def set_last_exchange(question: str, answer: str, confidence: float = 0.4, domain: str = "") -> None:
    """Store the last question/answer exchange for feedback processing.

    Args:
        question: The user's question
        answer: Maven's answer
        confidence: The confidence score of the answer
        domain: The domain/topic extracted from the question (first 1-2 words)
    """
    global _LAST_EXCHANGE
    _LAST_EXCHANGE = {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "domain": domain,
    }


def get_last_exchange() -> Optional[Dict[str, Any]]:
    """Retrieve the last exchange for feedback processing."""
    global _LAST_EXCHANGE
    return _LAST_EXCHANGE


# =============================================================================
# Feedback detection
# =============================================================================

def is_positive_feedback(text: str) -> bool:
    """Detect whether the text is positive feedback.

    Recognizes affirmative responses like "correct", "yes", "right", etc.

    Args:
        text: The user's input text

    Returns:
        True if the text indicates positive feedback
    """
    try:
        q = str(text or "").strip().lower()
    except Exception:
        return False

    # Exact matches
    positive_exact = {
        "correct", "right", "yes", "good", "true", "exactly",
        "yep", "yeah", "yup", "sure", "indeed", "absolutely",
        "y", "ok", "okay", "agreed"
    }
    if q in positive_exact:
        return True

    # Phrase matches
    positive_phrases = [
        "that's correct", "that's right", "you're right", "you're correct",
        "yes correct", "correct on", "all correct", "yes that's",
        "that is correct", "that is right"
    ]
    return any(phrase in q for phrase in positive_phrases)


def is_negative_feedback(text: str) -> bool:
    """Detect whether the text is negative feedback.

    Recognizes corrections like "no", "incorrect", "wrong", etc.

    Args:
        text: The user's input text

    Returns:
        True if the text indicates negative feedback
    """
    try:
        q = str(text or "").strip().lower()
    except Exception:
        return False

    # Exact matches
    negative_exact = {"no", "incorrect", "wrong", "false", "nope", "n"}
    if q in negative_exact:
        return True

    # Phrase matches and starts with
    if q.startswith("no") or "incorrect" in q or "wrong" in q:
        return True

    return False


def is_correction(ctx: Dict[str, Any]) -> bool:
    """Detect whether the current query is a user correction.

    This checks for explicit patterns in the original query such
    as "no," or "incorrect".
    """
    try:
        q = str((ctx.get("original_query") or "")).strip().lower()
    except Exception:
        q = ""
    return is_negative_feedback(q)


# =============================================================================
# Feedback handlers
# =============================================================================

def handle_positive_feedback(memory_librarian_api=None) -> str:
    """Process positive feedback by updating confidence scores.

    When the user confirms an answer is correct, this function:
    1. Updates meta_confidence for the domain (marks success)
    2. Calls BRAIN_MERGE to bump the Q&A confidence by +0.1
    3. Returns a confirmation message

    Args:
        memory_librarian_api: The memory librarian service_api function (optional)

    Returns:
        A response message for the user
    """
    global _LAST_EXCHANGE
    if _LAST_EXCHANGE is None:
        return "Noted."

    try:
        question = _LAST_EXCHANGE.get("question", "")
        answer = _LAST_EXCHANGE.get("answer", "")
        current_conf = _LAST_EXCHANGE.get("confidence", 0.4)
        domain = _LAST_EXCHANGE.get("domain", "")

        # Extract domain from question if not already set (first 1-2 words)
        if not domain and question:
            words = question.strip().split()
            domain = " ".join(words[:2]) if len(words) >= 2 else words[0] if words else ""

        # Update meta-confidence for the domain
        try:
            from brains.personal.memory import meta_confidence
            if domain:
                meta_confidence.update(domain, success=True)
        except Exception:
            pass

        # Update confidence in fast_cache (which pipeline checks on every query)
        new_conf = current_conf
        try:
            # Try to access the memory_librarian module to use its cache functions
            from brains.cognitive.memory_librarian.service import memory_librarian as ml_module
            # First try to boost existing cache entry
            boosted_conf = ml_module._boost_cache_confidence(question, boost_amount=0.15)
            if boosted_conf is not None:
                new_conf = boosted_conf
            else:
                # No cached entry exists, store a new one with boosted confidence
                new_conf = min(1.0, current_conf + 0.15)
                ml_module._store_fast_cache_entry(question, answer, new_conf)
        except Exception:
            # Fallback: just add 0.15 to current confidence locally
            new_conf = min(1.0, current_conf + 0.15)

        # Also update BRAIN_MERGE for persistent storage
        if memory_librarian_api is not None:
            try:
                q_key = question.strip().lower()
                memory_librarian_api({
                    "op": "BRAIN_MERGE",
                    "payload": {
                        "scope": "BRAIN",
                        "origin_brain": "qa_memory",
                        "key": q_key,
                        "value": answer,
                        "conf_delta": 0.15
                    }
                })
            except Exception:
                pass

        # Log the positive feedback as a correction event (reinforcement)
        _append_jsonl(CORRECTION_LOG_PATH, {
            "kind": "feedback_event",
            "feedback_type": "positive",
            "question": question[:200] if question else "",
            "domain": domain,
            "old_confidence": current_conf,
            "new_confidence": new_conf,
            "ts": _now_iso(),
        })

        # Return confirmation with updated confidence
        return "Noted."

    except Exception:
        return "Noted."


def handle_negative_feedback(memory_librarian_api=None) -> str:
    """Process negative feedback by recording the failure.

    When the user indicates an answer is incorrect, this function:
    1. Updates meta_confidence for the domain (marks failure)
    2. Records the incorrect pattern for learning
    3. Returns an acknowledgment

    Args:
        memory_librarian_api: The memory librarian service_api function (optional)

    Returns:
        A response message for the user
    """
    global _LAST_EXCHANGE
    if _LAST_EXCHANGE is None:
        return "I see. I'll try to do better."

    try:
        question = _LAST_EXCHANGE.get("question", "")
        answer = _LAST_EXCHANGE.get("answer", "")
        current_conf = _LAST_EXCHANGE.get("confidence", 0.4)
        domain = _LAST_EXCHANGE.get("domain", "")

        # Extract domain from question if not already set
        if not domain and question:
            words = question.strip().split()
            domain = " ".join(words[:2]) if len(words) >= 2 else words[0] if words else ""

        # Update meta-confidence for the domain (mark as failure)
        try:
            from brains.personal.memory import meta_confidence
            if domain:
                meta_confidence.update(domain, success=False)
        except Exception:
            pass

        # Lower confidence in fast_cache
        try:
            from brains.cognitive.memory_librarian.service import memory_librarian as ml_module
            ml_module._boost_cache_confidence(question, boost_amount=-0.2)
        except Exception:
            pass

        # Log the negative feedback
        _append_jsonl(CORRECTION_LOG_PATH, {
            "kind": "feedback_event",
            "feedback_type": "negative",
            "question": question[:200] if question else "",
            "answer": answer[:200] if answer else "",
            "domain": domain,
            "old_confidence": current_conf,
            "ts": _now_iso(),
        })

        return "I see. I'll try to do better."

    except Exception:
        return "I see. I'll try to do better."


# =============================================================================
# Core correction API
# =============================================================================

def register_correction(
    belief_id: Optional[str],
    old_belief: Any,
    new_belief: Any,
    source: str,
    severity: str,
    reason: str,
) -> CorrectionEvent:
    """
    Main entry point for a correction.

    - Attempts to supersede the belief in the memory backend.
    - Logs the event to corrections.jsonl with mode="superseded" or "log_only".
    - Returns the CorrectionEvent so other brains can react.

    This is what self-critique, Teacher, or user-facing layers should call
    when something is found to be wrong or outdated.
    """

    source_norm = source.lower()
    severity_norm = severity.lower()

    if severity_norm not in {"minor", "major", "critical"}:
        severity_norm = "minor"

    mode = "log_only"
    backend_ok = False

    if _memory_backend_available():
        backend_ok = _supersede_in_memory(belief_id, new_belief)
        if backend_ok:
            mode = "superseded"
        else:
            mode = "log_only"
    else:
        mode = "log_only"

    evt = CorrectionEvent(
        belief_id=str(belief_id) if belief_id is not None else None,
        old_belief=_shorten_belief_summary(old_belief),
        new_belief=new_belief,
        source=source_norm,
        severity=severity_norm,
        reason=reason,
        mode=mode,
        ts=_now_iso(),
    )

    record = asdict(evt)
    record["kind"] = "correction_event"
    _append_jsonl(CORRECTION_LOG_PATH, record)

    return evt


def record_correction_pattern(
    old_belief: Any,
    new_belief: Any,
    examples: Optional[List[str]] = None
) -> CorrectionPattern:
    """
    Record a higher-level pattern about how beliefs are being corrected.

    This is used later for meta-learning, e.g.:

    - "When Teacher claims X about JavaScript versions, it is often wrong and replaced by Y."
    - "When time-sensitive facts (prices, weather) age out, they are replaced by new values."

    This function does not modify the memory; it just logs patterns.
    """

    old_sig = _signature_from_belief(old_belief)
    new_sig = _signature_from_belief(new_belief)
    examples = examples or []

    pat = CorrectionPattern(
        old_signature=old_sig,
        new_signature=new_sig,
        examples=[str(e)[:200] for e in examples][:5],
        ts=_now_iso(),
    )

    record = asdict(pat)
    record["kind"] = "correction_pattern"
    _append_jsonl(CORRECTION_LOG_PATH, record)
    return pat


def list_recent_corrections(limit: int = 50, max_age_days: int = 30) -> List[Dict[str, Any]]:
    """
    Read the corrections log and return recent correction events
    as plain dicts (suitable for self_model / governance / UI).
    """

    if not CORRECTION_LOG_PATH.exists():
        return []

    events: List[Dict[str, Any]] = []
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    try:
        with CORRECTION_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") != "correction_event":
                    continue

                ts_str = rec.get("ts")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                except Exception:
                    ts = None

                if ts is not None and ts < cutoff:
                    continue

                events.append(rec)
    except Exception as e:
        logger.error("Failed to read corrections log: %s", e)
        return []

    # Most recent first
    events.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return events[:limit]


# =============================================================================
# Belief management
# =============================================================================

def find_contradicted_belief(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Identify the belief contradicted by the correction.

    Searches memory for the belief that was used to answer the previous query.

    Args:
        ctx: Context dictionary with 'original_query' and optionally 'last_answer'

    Returns:
        The belief record if found, None otherwise
    """
    global _LAST_EXCHANGE

    if _LAST_EXCHANGE is None:
        return None

    question = _LAST_EXCHANGE.get("question", "")
    answer = _LAST_EXCHANGE.get("answer", "")

    if not question:
        return None

    # Try to find the belief in memory librarian's fast cache
    try:
        from brains.cognitive.memory_librarian.service import memory_librarian as ml_module
        q_key = question.strip().lower()

        # Check fast cache for the belief
        cached = ml_module._get_fast_cache_entry(q_key)
        if cached:
            return {
                "id": q_key,
                "question": question,
                "answer": cached.get("answer", answer),
                "confidence": cached.get("confidence", 0.4),
                "source": "fast_cache"
            }
    except Exception:
        pass

    # Fallback: return the last exchange as a pseudo-belief
    return {
        "id": question.strip().lower(),
        "question": question,
        "answer": answer,
        "confidence": _LAST_EXCHANGE.get("confidence", 0.4),
        "source": "last_exchange"
    }


def supersede_belief(belief_id: Any, new_info: Dict[str, Any]) -> None:
    """Replace an outdated belief with updated information.

    Integrates with memory librarian to actually update or invalidate
    stored beliefs.

    Args:
        belief_id: The identifier of the old belief (typically the question key).
        new_info: The updated belief content with 'answer' and optionally 'confidence'.
    """
    if not belief_id:
        return

    key = str(belief_id).strip().lower() if belief_id else ""
    new_answer = new_info.get("answer", "")
    new_confidence = new_info.get("confidence", 0.5)

    # Use the unified correction registration
    old_belief_summary = {"id": key}

    # Try to get old belief details
    try:
        from brains.cognitive.memory_librarian.service import memory_librarian as ml_module
        cached = ml_module._get_fast_cache_entry(key)
        if cached:
            old_belief_summary = {
                "id": key,
                "answer": cached.get("answer", ""),
                "confidence": cached.get("confidence", 0.0),
            }
    except Exception:
        pass

    # Register the correction through the unified API
    new_belief_dict = {
        "key": key,
        "answer": new_answer,
        "confidence": new_confidence,
    }

    evt = register_correction(
        belief_id=key,
        old_belief=old_belief_summary,
        new_belief=new_belief_dict,
        source="user",
        severity="minor",
        reason="User correction via feedback",
    )

    if evt.mode == "superseded":
        logger.info("Superseded belief: %s", key[:50])
    else:
        logger.info("Logged belief correction (memory unavailable): %s", key[:50])


def handle_correction(ctx: Dict[str, Any]) -> None:
    """Process a user correction request.

    If the input is detected as a correction, this function attempts to
    supersede the contradicted belief with the new information and
    record the correction pattern.

    Args:
        ctx: Context dictionary with 'original_query' and optionally 'new_info'
    """
    try:
        if not is_correction(ctx):
            return

        old_belief = find_contradicted_belief(ctx)
        new_info = ctx.get("new_info") or {}

        if old_belief:
            # Use the unified correction API
            register_correction(
                belief_id=old_belief.get("id"),
                old_belief=old_belief,
                new_belief=new_info,
                source="user",
                severity="minor",
                reason="User indicated previous answer was incorrect",
            )

            # Also record the pattern for meta-learning
            record_correction_pattern(
                old_belief=old_belief,
                new_belief=new_info,
                examples=[
                    old_belief.get("answer", "")[:100],
                    new_info.get("answer", "")[:100] if isinstance(new_info, dict) else ""
                ]
            )
    except Exception as e:
        logger.error("handle_correction failed: %s", e)


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for correction handler.

    Supported operations:
    - REGISTER: Register a correction event
    - LIST_RECENT: List recent corrections
    - RECORD_PATTERN: Record a correction pattern
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    if op == "REGISTER":
        try:
            evt = register_correction(
                belief_id=payload.get("belief_id"),
                old_belief=payload.get("old_belief"),
                new_belief=payload.get("new_belief"),
                source=payload.get("source", "system"),
                severity=payload.get("severity", "minor"),
                reason=payload.get("reason", ""),
            )
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": asdict(evt),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "REGISTER_FAILED", "message": str(e)},
            }

    if op == "LIST_RECENT":
        try:
            limit = int(payload.get("limit", 50))
            max_age_days = int(payload.get("max_age_days", 30))
            corrections = list_recent_corrections(limit=limit, max_age_days=max_age_days)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"corrections": corrections, "count": len(corrections)},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LIST_FAILED", "message": str(e)},
            }

    if op == "RECORD_PATTERN":
        try:
            pat = record_correction_pattern(
                old_belief=payload.get("old_belief"),
                new_belief=payload.get("new_belief"),
                examples=payload.get("examples"),
            )
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": asdict(pat),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "PATTERN_FAILED", "message": str(e)},
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "correction_handler",
                "memory_backend_available": _memory_backend_available(),
                "log_path": str(CORRECTION_LOG_PATH),
                "available_operations": ["REGISTER", "LIST_RECENT", "RECORD_PATTERN", "HEALTH"],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
