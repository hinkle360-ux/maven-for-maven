"""
External Fact Verification

This module provides a concrete verification layer used by Self-Critique V2
(and potentially other brains) to check the factual reliability of an answer.

Input:
    question: str
    answer: str
    metadata: dict (may contain tools_used, sources, question_kind, etc.)

Output:
    dict with fields:
        supported: bool | None
        support_probability: float [0–1]
        contradiction_probability: float [0–1]
        notes: str
        evidence: list[dict]
        method: str ("metadata_only", "research_backend", "heuristic_only", "not_applicable")
        ts: iso timestamp

Behavior:
    - If the question is not factual / verifiable, returns supported=None and method="not_applicable".
    - If structured sources are present in metadata, aggregates them (metadata_only).
    - If a research backend is available, calls it and aggregates support/contradiction (research_backend).
    - If neither sources nor backend exist, uses honest heuristics (heuristic_only) and marks
      the result as low-confidence.

No stubs. If no backend, the module still runs and logs an explicit "no backend available".
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
FACT_VERIFICATION_LOG_PATH = MAVEN_DIR / "fact_verification.jsonl"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class VerificationEvidence:
    """A single piece of evidence from verification."""
    source_name: str          # e.g. "wikipedia", "research_backend", "metadata_source"
    url: Optional[str]
    snippet: str
    stance: str               # "support" | "contradict" | "unclear"
    confidence: float         # 0–1


@dataclass
class VerificationResult:
    """Complete result of fact verification."""
    supported: Optional[bool]  # True / False / None (not applicable / unknown)
    support_probability: float
    contradiction_probability: float
    notes: str
    evidence: List[VerificationEvidence]
    method: str                # "metadata_only" | "research_backend" | "heuristic_only" | "not_applicable"
    ts: str
    question: str
    answer: str
    metadata: Dict[str, Any]


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


# =============================================================================
# Question classification
# =============================================================================

_FACTUAL_WH_WORDS = ["who", "what", "when", "where", "which", "how many", "how much"]
_FACTUAL_PATTERNS = ["is", "are", "was", "were", "does", "do", "can", "could", "has", "have", "did"]
_OPINION_MARKERS = ["should", "would you", "do you think", "what's your opinion", "how do you feel"]
_TIME_SENSITIVE_MARKERS = ["today", "current", "latest", "now", "right now", "this week", "this month"]


def _is_factual_question(question: str, metadata: Dict[str, Any]) -> bool:
    """
    Heuristic: decide if question is factual and should be verified.

    We treat obviously opinion/creative/open-ended questions as not applicable.
    """
    q = question.strip().lower()
    if not q:
        return False

    kind = str(metadata.get("question_kind", "")).lower()
    if any(k in kind for k in ["opinion", "creative", "writing", "chat", "greeting"]):
        return False

    # Check for opinion markers
    if any(m in q for m in _OPINION_MARKERS):
        return False

    if any(w in q for w in _FACTUAL_WH_WORDS):
        return True

    # Simple yes/no or fact-check style questions
    if any(q.startswith(p + " ") for p in _FACTUAL_PATTERNS):
        return True

    # If metadata says it's research/external, treat as factual
    if any(k in kind for k in ["factual", "research", "web", "external"]):
        return True

    return False


def _is_time_sensitive(question: str, answer: str, metadata: Dict[str, Any]) -> bool:
    """Check if question/answer involves time-sensitive facts."""
    text = (question + " " + answer).lower()
    return any(m in text for m in _TIME_SENSITIVE_MARKERS)


# =============================================================================
# Metadata-based evidence extraction
# =============================================================================

def _extract_evidence_from_metadata(metadata: Dict[str, Any]) -> List[VerificationEvidence]:
    """
    Pull structured evidence from metadata["sources"] if present.

    Expects entries like:
        {
            "name": "wikipedia",
            "url": "...",
            "snippet": "...",
            "stance": "support" | "contradict" | "unclear",
            "confidence": 0.0–1.0,
        }

    But degrades gracefully if some keys are missing.
    """
    out: List[VerificationEvidence] = []
    sources = metadata.get("sources") or []
    if not isinstance(sources, list):
        return out

    for src in sources:
        if not isinstance(src, dict):
            continue
        name = str(src.get("name") or src.get("source") or "unknown")
        url = src.get("url")
        snippet = str(src.get("snippet") or src.get("text") or "")[:400]
        stance = str(src.get("stance") or "unclear").lower()
        if stance not in {"support", "contradict", "unclear"}:
            stance = "unclear"
        try:
            confidence = float(src.get("confidence", 0.5))
        except Exception:
            confidence = 0.5

        out.append(
            VerificationEvidence(
                source_name=name,
                url=url,
                snippet=snippet,
                stance=stance,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )

    return out


def _aggregate_evidence(evidence: List[VerificationEvidence]) -> Dict[str, float]:
    """
    Aggregate a list of evidence items into support/contradiction probabilities.
    """
    if not evidence:
        return {"support_p": 0.0, "contradiction_p": 0.0}

    support_weight = 0.0
    contradict_weight = 0.0
    total_weight = 0.0

    for ev in evidence:
        w = ev.confidence
        total_weight += w
        if ev.stance == "support":
            support_weight += w
        elif ev.stance == "contradict":
            contradict_weight += w

    if total_weight <= 0:
        return {"support_p": 0.0, "contradiction_p": 0.0}

    support_p = support_weight / total_weight
    contradiction_p = contradict_weight / total_weight
    return {"support_p": support_p, "contradiction_p": contradiction_p}


# =============================================================================
# Research backend adapter (optional)
# =============================================================================

def _research_backend_available() -> bool:
    """
    Detect if a dedicated research verification backend exists.

    Expected optional API (to be implemented in your research manager):

        from brains.research.manager import verify_answer_with_research

        verify_answer_with_research(question: str, answer: str, metadata: Dict[str, Any]) -> Dict[str, Any]

    If import or call fails, we return False and degrade gracefully.
    """
    try:
        from brains.research.manager import verify_answer_with_research  # noqa: F401
        return True
    except Exception:
        return False


def _call_research_backend(
    question: str,
    answer: str,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Call the research backend (if present) and normalize output.

    Expected backend output schema (you/Claude implement this):

        {
          "supporting_evidence": [
              {"source_name": str, "url": str|None, "snippet": str, "confidence": float},
              ...
          ],
          "contradicting_evidence": [
              {same fields},
              ...
          ],
          "uncertain_evidence": [optional list],
          "support_probability": float,
          "contradiction_probability": float,
          "notes": str,
        }

    If backend is missing or fails, returns None.
    """
    if not _research_backend_available():
        return None

    try:
        from brains.research.manager import verify_answer_with_research

        raw = verify_answer_with_research(question, answer, metadata)
    except Exception as e:
        logger.error("verify_answer_with_research failed: %s", e)
        return None

    if not isinstance(raw, dict):
        return None

    return raw


def _evidence_from_backend(raw: Dict[str, Any]) -> List[VerificationEvidence]:
    """Convert backend response to VerificationEvidence list."""
    out: List[VerificationEvidence] = []

    def convert_list(items: Any, stance: str) -> None:
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            name = str(it.get("source_name") or it.get("source") or "research_backend")
            url = it.get("url")
            snippet = str(it.get("snippet") or it.get("text") or "")[:400]
            try:
                conf = float(it.get("confidence", 0.7))
            except Exception:
                conf = 0.7
            out.append(
                VerificationEvidence(
                    source_name=name,
                    url=url,
                    snippet=snippet,
                    stance=stance,
                    confidence=max(0.0, min(1.0, conf)),
                )
            )

    convert_list(raw.get("supporting_evidence"), "support")
    convert_list(raw.get("contradicting_evidence"), "contradict")
    convert_list(raw.get("uncertain_evidence"), "unclear")

    return out


# =============================================================================
# Heuristic-only fallback
# =============================================================================

def _heuristic_verification(question: str, answer: str, metadata: Dict[str, Any]) -> VerificationResult:
    """
    When we have no sources and no backend, we still return a real result,
    but explicitly low-confidence and labeled as heuristic_only.
    """
    ts = _now_iso()
    q = question.lower()
    a = answer.lower()

    # Check for explicit uncertainty in the answer
    uncertainty_phrases = [
        "i don't know", "i do not know",
        "i'm not sure", "i am not sure",
        "cannot be determined", "cannot determine",
        "unclear", "uncertain",
    ]

    if any(p in a for p in uncertainty_phrases):
        supported = None
        support_p = 0.1
        contradiction_p = 0.1
        notes = "Answer explicitly expresses uncertainty; verification inconclusive (heuristic_only)."
    elif _is_time_sensitive(question, answer, metadata):
        # Time-sensitive answers without verification are risky
        supported = None
        support_p = 0.2
        contradiction_p = 0.4
        notes = "Time-sensitive claim without external verification; higher contradiction risk (heuristic_only)."
    else:
        # We cannot actually check against external data here.
        supported = None
        support_p = 0.3
        contradiction_p = 0.3
        notes = "No external sources or backend available; verification unknown (heuristic_only)."

    return VerificationResult(
        supported=supported,
        support_probability=support_p,
        contradiction_probability=contradiction_p,
        notes=notes,
        evidence=[],
        method="heuristic_only",
        ts=ts,
        question=question,
        answer=answer,
        metadata=metadata,
    )


# =============================================================================
# Public API
# =============================================================================

def verify_answer(question: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main verification entry point.

    Used by Self-Critique V2:

        from brains.research.verify import verify_answer

        result = verify_answer(question, answer, metadata)

    Returns a pure dict (JSON-serializable) with keys:

        supported: bool | None
        support_probability: float
        contradiction_probability: float
        notes: str
        evidence: list[dict]
        method: str
        ts: str
        question: str
        answer: str
        metadata: dict
    """
    if metadata is None:
        metadata = {}

    # 1. Check if this question is even suitable for factual verification
    if not _is_factual_question(question, metadata):
        ts = _now_iso()
        result = VerificationResult(
            supported=None,
            support_probability=0.0,
            contradiction_probability=0.0,
            notes="Question not classified as factual; verification not applicable.",
            evidence=[],
            method="not_applicable",
            ts=ts,
            question=question,
            answer=answer,
            metadata=metadata,
        )
        rec = asdict(result)
        rec["evidence"] = [asdict(ev) for ev in result.evidence]
        rec["kind"] = "fact_verification"
        _append_jsonl(FACT_VERIFICATION_LOG_PATH, rec)
        return rec

    # 2. Try metadata-only evidence first
    meta_evidence = _extract_evidence_from_metadata(metadata)
    if meta_evidence:
        agg = _aggregate_evidence(meta_evidence)
        support_p = agg["support_p"]
        contradiction_p = agg["contradiction_p"]

        if support_p > 0.5 and support_p >= contradiction_p:
            supported = True
        elif contradiction_p > 0.5 and contradiction_p > support_p:
            supported = False
        else:
            supported = None

        ts = _now_iso()
        result = VerificationResult(
            supported=supported,
            support_probability=support_p,
            contradiction_probability=contradiction_p,
            notes="Verification based on metadata sources only.",
            evidence=meta_evidence,
            method="metadata_only",
            ts=ts,
            question=question,
            answer=answer,
            metadata=metadata,
        )
        rec = asdict(result)
        rec["evidence"] = [asdict(ev) for ev in result.evidence]
        rec["kind"] = "fact_verification"
        _append_jsonl(FACT_VERIFICATION_LOG_PATH, rec)
        return rec

    # 3. No metadata sources; try research backend
    backend_raw = _call_research_backend(question, answer, metadata)
    if backend_raw is not None:
        evidence = _evidence_from_backend(backend_raw)
        agg = _aggregate_evidence(evidence)

        # Backend may already provide probabilities; combine conservatively.
        backend_support_p = float(backend_raw.get("support_probability", 0.0))
        backend_contra_p = float(backend_raw.get("contradiction_probability", 0.0))

        support_p = max(agg["support_p"], backend_support_p)
        contradiction_p = max(agg["contradiction_p"], backend_contra_p)

        if support_p > 0.5 and support_p >= contradiction_p:
            supported = True
        elif contradiction_p > 0.5 and contradiction_p > support_p:
            supported = False
        else:
            supported = None

        notes = str(backend_raw.get("notes", "Verification via research backend."))[:400]

        ts = _now_iso()
        result = VerificationResult(
            supported=supported,
            support_probability=max(0.0, min(1.0, support_p)),
            contradiction_probability=max(0.0, min(1.0, contradiction_p)),
            notes=notes,
            evidence=evidence,
            method="research_backend",
            ts=ts,
            question=question,
            answer=answer,
            metadata=metadata,
        )
        rec = asdict(result)
        rec["evidence"] = [asdict(ev) for ev in result.evidence]
        rec["kind"] = "fact_verification"
        _append_jsonl(FACT_VERIFICATION_LOG_PATH, rec)
        return rec

    # 4. Fallback: heuristic-only
    result = _heuristic_verification(question, answer, metadata)
    rec = asdict(result)
    rec["evidence"] = [asdict(ev) for ev in result.evidence]
    rec["kind"] = "fact_verification"
    _append_jsonl(FACT_VERIFICATION_LOG_PATH, rec)
    return rec


def list_recent_verifications(limit: int = 50, max_age_days: int = 7) -> List[Dict[str, Any]]:
    """
    Read the verification log and return recent verifications
    as plain dicts (suitable for self_model / governance / UI).
    """
    if not FACT_VERIFICATION_LOG_PATH.exists():
        return []

    events: List[Dict[str, Any]] = []
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    try:
        with FACT_VERIFICATION_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") != "fact_verification":
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
        logger.error("Failed to read verification log: %s", e)
        return []

    # Most recent first
    events.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return events[:limit]


def get_verification_statistics(days: int = 7) -> Dict[str, Any]:
    """
    Compute statistics from recent verifications for self_model / governance.
    """
    verifications = list_recent_verifications(limit=1000, max_age_days=days)

    if not verifications:
        return {
            "total_verifications": 0,
            "method_counts": {},
            "supported_count": 0,
            "contradicted_count": 0,
            "unknown_count": 0,
            "average_support_probability": 0.0,
            "average_contradiction_probability": 0.0,
        }

    method_counts: Dict[str, int] = {}
    supported_count = 0
    contradicted_count = 0
    unknown_count = 0
    total_support_p = 0.0
    total_contra_p = 0.0

    for v in verifications:
        method = v.get("method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

        supported = v.get("supported")
        if supported is True:
            supported_count += 1
        elif supported is False:
            contradicted_count += 1
        else:
            unknown_count += 1

        total_support_p += float(v.get("support_probability", 0.0))
        total_contra_p += float(v.get("contradiction_probability", 0.0))

    return {
        "total_verifications": len(verifications),
        "method_counts": method_counts,
        "supported_count": supported_count,
        "contradicted_count": contradicted_count,
        "unknown_count": unknown_count,
        "average_support_probability": total_support_p / len(verifications) if verifications else 0.0,
        "average_contradiction_probability": total_contra_p / len(verifications) if verifications else 0.0,
        "days_covered": days,
    }


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for fact verification.

    Supported operations:
    - VERIFY: Verify a question/answer pair
    - LIST_RECENT: List recent verifications
    - STATISTICS: Get verification statistics
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    if op == "VERIFY":
        try:
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            metadata = payload.get("metadata", {})

            if not question or not answer:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_INPUT", "message": "question and answer required"},
                }

            result = verify_answer(question, answer, metadata)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result,
            }
        except Exception as e:
            logger.exception("VERIFY operation failed")
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "VERIFY_FAILED", "message": str(e)},
            }

    if op == "LIST_RECENT":
        try:
            limit = int(payload.get("limit", 50))
            max_age_days = int(payload.get("max_age_days", 7))
            verifications = list_recent_verifications(limit=limit, max_age_days=max_age_days)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"verifications": verifications, "count": len(verifications)},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LIST_FAILED", "message": str(e)},
            }

    if op == "STATISTICS":
        try:
            days = int(payload.get("days", 7))
            stats = get_verification_statistics(days=days)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": stats,
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "STATISTICS_FAILED", "message": str(e)},
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "fact_verification",
                "research_backend_available": _research_backend_available(),
                "log_path": str(FACT_VERIFICATION_LOG_PATH),
                "available_operations": ["VERIFY", "LIST_RECENT", "STATISTICS", "HEALTH"],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
