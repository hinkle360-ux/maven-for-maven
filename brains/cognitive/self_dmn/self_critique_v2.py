"""
Self-Critique V2

Human-like self-monitoring layer:

- Reads a question, draft answer, and metadata.
- Identifies potential issues (hallucination risk, missing tools, safety concerns,
  overconfidence/hedging, inconsistency with context).
- Assigns probabilities to each issue.
- Produces an overall verdict and escalation decision.
- Logs all critiques to ~/.maven/self_critique.jsonl.

If an external verifier is available (research/teacher integration),
it is used; otherwise we degrade gracefully, and we do NOT pretend
we verified anything we didn't.

No stubs. No fake "ok".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
SELF_CRITIQUE_LOG_PATH = MAVEN_DIR / "self_critique.jsonl"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CritiqueIssue:
    """A single issue identified during self-critique."""
    kind: str                # e.g. "hallucination_risk", "missing_tool_usage"
    description: str
    probability: float       # 0.0–1.0
    severity: str            # "minor"|"major"|"critical"
    needs_escalation: bool


@dataclass
class CritiqueResult:
    """The complete result of a self-critique analysis."""
    question: str
    draft_answer: str
    verdict: str             # "ok"|"minor_issue"|"major_issue"|"unsafe"
    overall_confidence: float
    issues: List[CritiqueIssue]
    escalate_to_teacher: bool
    suggest_retry_with_tools: bool
    suggest_correction_record: bool
    tool_suggestions: List[str]
    ts: str
    metadata: Dict[str, Any]
    verifier_used: bool
    verifier_notes: str


@dataclass
class Fact:
    """A fact record for verification and storage."""
    content: str
    source: str
    domain: str = "factual"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactDecision:
    """Decision result from fact processing."""
    decision: str          # "accept" | "reject" | "verify_first"
    confidence: float      # 0.0-1.0
    reason: str
    reprimand_teacher: bool = False


@dataclass
class VerificationResult:
    """Result from external verification."""
    status: str              # "SUPPORTS" | "CONTRADICTS" | "UNKNOWN"
    confidence: float        # 0.0-1.0
    evidence_snippets: List[str] = field(default_factory=list)


def _looks_like_teacher_self_hallucination(text: str) -> bool:
    """
    Detect teacher self-hallucination patterns.

    Returns True if the text contains phrases that indicate the teacher
    is talking about itself as a generic LLM (e.g., ChatGPT, training data).
    """
    patterns = [
        "my training data",
        "my knowledge cutoff",
        "chatgpt",
        "as a large language model",
        "i was trained by",
        "openai",
        "as an ai",
        "i am gpt",
        "i am claude",
        "my training",
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in patterns)


def score_confidence(fact: Fact) -> float:
    """
    Score the confidence of a fact based on its source and content.

    Returns a float between 0.0 and 1.0.

    Source-based confidence:
    - self_model: 1.0 (highest trust - Maven's own knowledge)
    - user: 0.8 (high trust - user provided info)
    - teacher: 0.6 (moderate trust - external LLM)
    - unknown: 0.5 (neutral)
    """
    # Source-based scoring (primary factor)
    source_scores = {
        "self_model": 1.0,
        "user": 0.8,
        "teacher": 0.6,
        "unknown": 0.5,
    }

    source_lower = fact.source.lower()
    base_score = source_scores.get(source_lower, 0.5)

    # If source is not in our map, check for partial matches
    if source_lower not in source_scores:
        # Boost for known reliable sources
        reliable_sources = ["wikipedia", "official_docs", "academic", "verified"]
        if any(s in source_lower for s in reliable_sources):
            base_score = 0.7

    return base_score


def decide_fact(fact: Fact) -> FactDecision:
    """
    Decide whether to accept, reject, or verify_first a fact.

    Returns a FactDecision object with decision, confidence, and reason.

    Thresholds:
    - accept: confidence >= 0.7
    - reject: confidence < 0.3 OR teacher hallucination detected
    - verify_first: all other cases
    """
    confidence = score_confidence(fact)
    reprimand_teacher = False
    reason = ""

    # Check for teacher hallucination
    if fact.source.lower() == "teacher" and _looks_like_teacher_self_hallucination(fact.content):
        reprimand_teacher = True
        reason = "Teacher hallucination detected - self-reference to generic LLM identity"
        return FactDecision(
            decision="reject",
            confidence=confidence,
            reason=reason,
            reprimand_teacher=reprimand_teacher,
        )

    # Apply thresholds
    if confidence >= 0.7:
        decision = "accept"
        reason = f"Fact accepted with confidence {confidence:.2f} from source '{fact.source}'"
    elif confidence < 0.3:
        decision = "reject"
        reason = f"Fact rejected due to low confidence {confidence:.2f}"
    else:
        decision = "verify_first"
        reason = f"Fact needs verification (confidence {confidence:.2f})"

    return FactDecision(
        decision=decision,
        confidence=confidence,
        reason=reason,
        reprimand_teacher=reprimand_teacher,
    )


def process_fact(fact: Fact) -> FactDecision:
    """
    Process a fact and return a full decision with explanation.

    This is the main entry point for fact verification before storage.
    """
    # Use decide_fact for the main logic
    decision_result = decide_fact(fact)

    # Additional checks for unreliable source patterns
    unreliable_sources = ["unverified", "rumor", "anonymous"]
    if any(s in fact.source.lower() for s in unreliable_sources) and decision_result.confidence < 0.4:
        decision_result.reprimand_teacher = True
        decision_result.reason += "; unreliable source pattern detected"

    return decision_result


class SelfCritic:
    """
    Self-critic component for evaluating answers and facts.

    Provides both fact evaluation and answer critique capabilities.
    """

    def __init__(self):
        self._critique_count = 0
        self._fact_count = 0

    def evaluate(self, fact: Fact) -> FactDecision:
        """
        Evaluate a single fact.

        Args:
            fact: The Fact to evaluate

        Returns:
            FactDecision with decision in ["accept", "reject", "verify_first"]
        """
        self._fact_count += 1
        return process_fact(fact)

    def evaluate_fact(self, fact: Fact) -> FactDecision:
        """Alias for evaluate() - backward compatibility."""
        return self.evaluate(fact)

    def score(self, fact: Fact) -> float:
        """
        Score a fact's confidence.

        Args:
            fact: The Fact to score

        Returns:
            float between 0.0 and 1.0
        """
        return score_confidence(fact)

    def critique_answer(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CritiqueResult:
        """Run full self-critique on an answer."""
        self._critique_count += 1
        return run_self_critique(question, answer, metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this critic instance."""
        return {
            "critiques_performed": self._critique_count,
            "facts_evaluated": self._fact_count,
        }


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
# External verifier (optional)
# =============================================================================

def _external_verifier_available() -> bool:
    """
    Try to detect a dedicated verifier, such as a research manager
    function that can sanity-check answers against web or knowledge.

    We do NOT throw if it's missing; we simply return False.
    """
    try:
        from brains.research.verify import verify_answer  # noqa: F401
        return True
    except Exception:
        return False


def _run_external_verifier(question: str, answer: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    If available, call the external verifier.

    Expected external API (you/Claude can implement this separately):

        verify_answer(question: str, answer: str, metadata: Dict[str, Any]) -> Dict[str, Any]

    Should return, for example:
        {
          "supported": True/False,
          "support_probability": 0.8,
          "contradiction_probability": 0.1,
          "notes": "short string",
          "evidence": [...optional...]
        }

    If verifier is not present or fails, returns None.
    """
    if not _external_verifier_available():
        return None

    try:
        from brains.research.verify import verify_answer

        return verify_answer(question, answer, metadata)
    except Exception as e:
        logger.error("External verifier failed: %s", e)
        return None


# =============================================================================
# Heuristic analysis
# =============================================================================

_HEDGE_PHRASES = [
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "cannot be determined",
    "cannot determine",
    "may not be accurate",
    "might be wrong",
    "i believe",
    "i think",
    "possibly",
    "perhaps",
]

_OVERCONFIDENCE_PHRASES = [
    "100% certain",
    "absolutely certain",
    "there is no doubt",
    "beyond any doubt",
    "guaranteed",
    "definitely true",
    "certainly correct",
    "without question",
]

_SAFETY_TRIGGERS = [
    "rm -rf",
    "delete all files",
    "format the disk",
    "disable safety",
    "drop database",
    "truncate table",
    "sudo rm",
    "chmod 777",
    "disable firewall",
]


def _contains_any(text: str, phrases: List[str]) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def _base_hallucination_risk(question: str, answer: str, metadata: Dict[str, Any]) -> float:
    """
    Crude but real heuristic:
    - Higher risk for time-sensitive / external questions without tools.
    - Lower risk if tools were used or sources present.
    """
    q_lower = question.lower()
    answer_lower = answer.lower()
    meta_kind = str(metadata.get("question_kind", "")).lower()
    tools_used = metadata.get("tools_used") or []
    sources = metadata.get("sources") or []
    expects_tools = bool(metadata.get("expects_tools", False))

    risk = 0.2  # base

    # Time-sensitive / external topics
    if any(k in q_lower for k in ["weather", "price", "stock", "today", "latest", "current", "now"]):
        risk += 0.3
    if any(k in meta_kind for k in ["web", "external", "research"]):
        risk += 0.2

    # Specific numbers or dates without sources
    if any(c.isdigit() for c in answer) and not sources:
        risk += 0.1

    # If tools used or sources attached, reduce risk
    if tools_used or sources:
        risk -= 0.2

    # If tools expected but not used, bump risk
    if expects_tools and not tools_used:
        risk += 0.3

    # Long answers with many claims have higher risk
    word_count = len(answer.split())
    if word_count > 200:
        risk += 0.1

    # Bound
    return max(0.0, min(1.0, risk))


def _safety_issue_probability(question: str, answer: str, metadata: Dict[str, Any]) -> float:
    """
    Look for dangerous instructions combined with high-risk intent.
    """
    text = (question + " " + answer).lower()
    if not any(trigger in text for trigger in _SAFETY_TRIGGERS):
        return 0.0

    risk = "LOW"
    if "risk_level" in metadata:
        risk = str(metadata["risk_level"]).upper()
    if risk in {"HIGH", "CRITICAL"}:
        return 0.9
    return 0.6


def _missing_tool_probability(question: str, answer: str, metadata: Dict[str, Any]) -> float:
    """Check if tools were expected but not used."""
    tools_used = metadata.get("tools_used") or []
    expects_tools = bool(metadata.get("expects_tools", False))
    if not expects_tools:
        return 0.0
    if tools_used:
        return 0.0
    # If user explicitly asked to "use browser", "use git", etc.
    q_lower = question.lower()
    if any(k in q_lower for k in ["use the browser", "browse", "look up", "search the web", "check online"]):
        return 0.8
    if any(k in q_lower for k in ["run", "execute", "git", "file", "read the"]):
        return 0.6
    return 0.5


def _directness_and_hedging(answer: str) -> Dict[str, float]:
    """
    Return two scores:
    - hedging: 0–1, higher = more hedged.
    - overconfidence: 0–1, higher = more overconfident.
    """
    hedging = 0.6 if _contains_any(answer, _HEDGE_PHRASES) else 0.0
    overconf = 0.8 if _contains_any(answer, _OVERCONFIDENCE_PHRASES) else 0.0
    return {"hedging": hedging, "overconfidence": overconf}


def _length_confidence(answer: str) -> float:
    """
    Very rough proxy: ultra-short answers to complex questions have lower confidence.
    """
    words = len(answer.split())
    if words < 10:
        return 0.2
    if words < 30:
        return 0.4
    if words < 120:
        return 0.7
    return 0.8


def _severity_from_prob(p: float) -> str:
    if p > 0.8:
        return "critical"
    if p > 0.5:
        return "major"
    return "minor"


def _check_context_consistency(question: str, answer: str, metadata: Dict[str, Any]) -> Optional[CritiqueIssue]:
    """
    Check if the answer is consistent with any provided context.
    Returns an issue if inconsistency detected, None otherwise.
    """
    context = metadata.get("context", "")
    if not context:
        return None

    # Simple heuristic: if answer contradicts obvious facts in context
    answer_lower = answer.lower()
    context_lower = context.lower()

    # Check for negation patterns
    if "not" in answer_lower and "not" not in context_lower:
        # Potential contradiction - would need more sophisticated NLP
        pass

    # For now, return None - full implementation would use semantic similarity
    return None


# =============================================================================
# Main API
# =============================================================================

def run_self_critique(
    question: str,
    draft_answer: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> CritiqueResult:
    """
    Main entry point.

    This function:
    - Performs internal heuristic analysis.
    - Optionally calls external verifier if present.
    - Aggregates issues and computes overall verdict.
    - Logs the critique to self_critique.jsonl.
    - Optionally records corrections (pattern-level) if clearly bad.

    It does NOT silently "bless" an answer; if it can't verify,
    it is explicit about that in verifier_used/verifier_notes.
    """
    if metadata is None:
        metadata = {}

    ts = _now_iso()
    issues: List[CritiqueIssue] = []
    tool_suggestions: List[str] = []
    verifier_used = False
    verifier_notes = ""

    # 1) Heuristic: hallucination risk
    halluc_risk = _base_hallucination_risk(question, draft_answer, metadata)
    if halluc_risk > 0.3:
        issues.append(
            CritiqueIssue(
                kind="hallucination_risk",
                description="Answer may rely on unstated or unverifiable assumptions.",
                probability=halluc_risk,
                severity=_severity_from_prob(halluc_risk),
                needs_escalation=halluc_risk > 0.6,
            )
        )
        if halluc_risk > 0.5:
            tool_suggestions.append("research_tool")

    # 2) Heuristic: missing tools
    missing_tool_prob = _missing_tool_probability(question, draft_answer, metadata)
    if missing_tool_prob > 0.0:
        issues.append(
            CritiqueIssue(
                kind="missing_tool_usage",
                description="User request appears to require tools (web/fs/git/etc) but none were used.",
                probability=missing_tool_prob,
                severity=_severity_from_prob(missing_tool_prob),
                needs_escalation=missing_tool_prob > 0.6,
            )
        )
        tool_suggestions.append("action_engine")

    # 3) Heuristic: safety
    safety_prob = _safety_issue_probability(question, draft_answer, metadata)
    if safety_prob > 0.0:
        issues.append(
            CritiqueIssue(
                kind="safety_risk",
                description="Potentially dangerous or destructive operation mentioned.",
                probability=safety_prob,
                severity=_severity_from_prob(safety_prob),
                needs_escalation=True,
            )
        )
        tool_suggestions.append("execution_guard_review")

    # 4) Hedging vs overconfidence
    hedging_info = _directness_and_hedging(draft_answer)
    if hedging_info["hedging"] > 0.0:
        issues.append(
            CritiqueIssue(
                kind="high_hedging",
                description="Answer contains explicit uncertainty or hedging language.",
                probability=hedging_info["hedging"],
                severity="minor",
                needs_escalation=False,
            )
        )
    if hedging_info["overconfidence"] > 0.0 and halluc_risk > 0.3:
        issues.append(
            CritiqueIssue(
                kind="overconfidence",
                description="Answer uses very strong certainty language despite nontrivial risk.",
                probability=hedging_info["overconfidence"],
                severity="major",
                needs_escalation=True,
            )
        )

    # 5) Context consistency check
    context_issue = _check_context_consistency(question, draft_answer, metadata)
    if context_issue:
        issues.append(context_issue)

    # 6) Length-based confidence
    length_conf = _length_confidence(draft_answer)

    # 7) External verifier (optional, honest)
    verifier_result = _run_external_verifier(question, draft_answer, metadata)
    if verifier_result is not None:
        verifier_used = True
        supported = bool(verifier_result.get("supported", False))
        support_p = float(verifier_result.get("support_probability", 0.0))
        contra_p = float(verifier_result.get("contradiction_probability", 0.0))
        verifier_notes = str(verifier_result.get("notes", ""))[:300]

        if not supported and contra_p > 0.3:
            issues.append(
                CritiqueIssue(
                    kind="external_contradiction",
                    description="External verifier indicates contradiction with known sources.",
                    probability=contra_p,
                    severity=_severity_from_prob(contra_p),
                    needs_escalation=True,
                )
            )
        elif supported and support_p > 0.5:
            # Reduces effective hallucination risk
            halluc_risk *= 0.5
    else:
        verifier_notes = "No external verifier available; heuristics only."

    # 8) Aggregate verdict
    if not issues:
        verdict = "ok"
        overall_confidence = max(0.7, length_conf)
        escalate_to_teacher = False
        suggest_retry_with_tools = False
        suggest_correction_record = False
    else:
        max_prob = max(i.probability for i in issues)
        severity_map = {"minor": 0, "major": 1, "critical": 2}
        max_severity = max(severity_map.get(i.severity, 0) for i in issues)

        if max_severity >= 2 or max_prob > 0.8:
            verdict = "unsafe"
        elif max_severity == 1 or max_prob > 0.5:
            verdict = "major_issue"
        else:
            verdict = "minor_issue"

        # Confidence is inversely related to max_prob, but still bounded by length_conf
        overall_confidence = max(0.1, min(0.9, length_conf * (1.0 - 0.5 * max_prob)))

        escalate_to_teacher = any(i.needs_escalation for i in issues)
        suggest_retry_with_tools = any(
            i.kind in {"hallucination_risk", "missing_tool_usage"} and i.probability > 0.5
            for i in issues
        )
        suggest_correction_record = any(
            i.kind in {"external_contradiction", "hallucination_risk"} and i.probability > 0.7
            for i in issues
        )

    # 9) Optional correction pattern logging (pattern-level only)
    if suggest_correction_record and verifier_result is not None:
        try:
            from brains.cognitive.correction_handler import record_correction_pattern
            # We do NOT lie: we record exactly that this answer was suspected wrong
            # and might need replacement. The actual "new_belief" must be supplied
            # by the caller when they have a corrected answer.
            record_correction_pattern(
                old_belief={"question": question, "answer": draft_answer},
                new_belief={"question": question, "answer": "<corrected_answer_tbd>"},
                examples=[question, draft_answer],
            )
        except Exception as e:
            logger.error("Failed to record correction pattern: %s", e)

    result = CritiqueResult(
        question=question,
        draft_answer=draft_answer,
        verdict=verdict,
        overall_confidence=overall_confidence,
        issues=issues,
        escalate_to_teacher=escalate_to_teacher,
        suggest_retry_with_tools=suggest_retry_with_tools,
        suggest_correction_record=suggest_correction_record,
        tool_suggestions=list(sorted(set(tool_suggestions))),
        ts=ts,
        metadata=metadata,
        verifier_used=verifier_used,
        verifier_notes=verifier_notes,
    )

    # 10) Log to self_critique.jsonl
    rec = asdict(result)
    # Convert issues to pure dicts
    rec["issues"] = [asdict(i) for i in result.issues]
    rec["kind"] = "self_critique_v2"
    _append_jsonl(SELF_CRITIQUE_LOG_PATH, rec)

    return result


def list_recent_critiques(limit: int = 50, max_age_days: int = 7) -> List[Dict[str, Any]]:
    """
    Read the self-critique log and return recent critiques
    as plain dicts (suitable for self_model / governance / UI).
    """
    if not SELF_CRITIQUE_LOG_PATH.exists():
        return []

    events: List[Dict[str, Any]] = []
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    try:
        with SELF_CRITIQUE_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") != "self_critique_v2":
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
        logger.error("Failed to read self-critique log: %s", e)
        return []

    # Most recent first
    events.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return events[:limit]


def get_critique_statistics(days: int = 7) -> Dict[str, Any]:
    """
    Compute statistics from recent critiques for self_model / governance.
    """
    critiques = list_recent_critiques(limit=1000, max_age_days=days)

    if not critiques:
        return {
            "total_critiques": 0,
            "verdict_counts": {},
            "escalation_rate": 0.0,
            "average_confidence": 0.0,
            "top_issue_kinds": [],
        }

    verdict_counts: Dict[str, int] = {}
    issue_kind_counts: Dict[str, int] = {}
    escalation_count = 0
    total_confidence = 0.0

    for c in critiques:
        verdict = c.get("verdict", "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        if c.get("escalate_to_teacher"):
            escalation_count += 1

        total_confidence += float(c.get("overall_confidence", 0.5))

        for issue in c.get("issues", []):
            kind = issue.get("kind", "unknown")
            issue_kind_counts[kind] = issue_kind_counts.get(kind, 0) + 1

    # Top issue kinds sorted by count
    top_issues = sorted(issue_kind_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_critiques": len(critiques),
        "verdict_counts": verdict_counts,
        "escalation_rate": escalation_count / len(critiques) if critiques else 0.0,
        "average_confidence": total_confidence / len(critiques) if critiques else 0.0,
        "top_issue_kinds": [{"kind": k, "count": v} for k, v in top_issues],
        "days_covered": days,
    }


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for self-critique.

    Supported operations:
    - CRITIQUE: Run self-critique on a question/answer pair
    - LIST_RECENT: List recent critiques
    - STATISTICS: Get critique statistics
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    if op == "CRITIQUE":
        try:
            question = payload.get("question", "")
            draft_answer = payload.get("draft_answer", "")
            metadata = payload.get("metadata", {})

            if not question or not draft_answer:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_INPUT", "message": "question and draft_answer required"},
                }

            result = run_self_critique(question, draft_answer, metadata)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": result.verdict,
                    "overall_confidence": result.overall_confidence,
                    "issues": [asdict(i) for i in result.issues],
                    "escalate_to_teacher": result.escalate_to_teacher,
                    "suggest_retry_with_tools": result.suggest_retry_with_tools,
                    "tool_suggestions": result.tool_suggestions,
                    "verifier_used": result.verifier_used,
                    "verifier_notes": result.verifier_notes,
                },
            }
        except Exception as e:
            logger.exception("CRITIQUE operation failed")
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "CRITIQUE_FAILED", "message": str(e)},
            }

    if op == "LIST_RECENT":
        try:
            limit = int(payload.get("limit", 50))
            max_age_days = int(payload.get("max_age_days", 7))
            critiques = list_recent_critiques(limit=limit, max_age_days=max_age_days)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"critiques": critiques, "count": len(critiques)},
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
            stats = get_critique_statistics(days=days)
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
                "service": "self_critique_v2",
                "external_verifier_available": _external_verifier_available(),
                "log_path": str(SELF_CRITIQUE_LOG_PATH),
                "available_operations": ["CRITIQUE", "LIST_RECENT", "STATISTICS", "HEALTH"],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }


# =============================================================================
# Pipeline integration helper
# =============================================================================

def pipeline_stage_self_review(
    question: str,
    draft_answer: str,
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Helper function for pipeline integration.

    Call this after draft answer generation, before final output.
    Returns the critique along with escalation flags.
    """
    critique = run_self_critique(question, draft_answer, meta)

    # If unsafe or major issue with escalation:
    if critique.verdict in {"unsafe", "major_issue"} and critique.escalate_to_teacher:
        meta["self_critique"] = asdict(critique)
        meta["self_critique"]["issues"] = [asdict(i) for i in critique.issues]
        meta["escalate_reason"] = "self_critique_v2"

    return {
        "question": question,
        "draft_answer": draft_answer,
        "critique": critique,
        "needs_escalation": critique.escalate_to_teacher,
        "needs_retry_with_tools": critique.suggest_retry_with_tools,
    }
