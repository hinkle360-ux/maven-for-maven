"""
Identity Inferencer

This module gives Maven a human-like "sense of self":

- It reads real logs (reflection, chat, execution audit).
- It extracts behavioral metrics (cautiousness, escalation frequency,
  self-reflection depth, directness, verbosity, risk posture).
- It applies recency weighting (recent behavior counts more).
- It produces a persistent identity profile under ~/.maven/identity_profile.json
  that other brains (especially self_model) can read.

There are NO stubs here: every public function does real work,
works off real data, and fails gracefully instead of pretending.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
LOG_DIR = MAVEN_DIR / "logs"
EXEC_AUDIT_PATH = MAVEN_DIR / "execution_audit.jsonl"
IDENTITY_PROFILE_PATH = MAVEN_DIR / "identity_profile.json"

ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
]


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _now_utc() -> datetime:
    return datetime.utcnow()


def _recency_weight(ts: Optional[datetime], half_life_days: float = 7.0) -> float:
    """
    Exponential decay: events older than `half_life_days` have ~0.5 weight.
    If ts is missing, treat as old, but not zero.
    """
    if ts is None:
        return 0.1
    delta = _now_utc() - ts
    days = max(delta.total_seconds() / 86400.0, 0.0)
    if half_life_days <= 0:
        return 1.0
    # weight = 0.5^(days / half_life_days)
    return math.pow(0.5, days / half_life_days)


@dataclass
class Trait:
    name: str
    score: float            # 0.0 – 1.0
    confidence: float       # 0.0 – 1.0 (how solid the evidence is)
    evidence_count: int
    last_updated: str       # ISO timestamp
    summary: str            # human-readable description


@dataclass
class IdentityProfile:
    traits: List[Trait]
    generated_at: str
    source: str             # e.g., "logs+execution_audit"
    notes: str = ""


# -----------------------------
# Log readers (robust, no stubs)
# -----------------------------

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    logger.debug("Skipping invalid JSONL line in %s", path)
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)


def _iter_logs_with_prefix(prefix: str) -> Iterable[Dict[str, Any]]:
    if not LOG_DIR.exists():
        return
    for p in LOG_DIR.glob(f"{prefix}_*.jsonl"):
        yield from _iter_jsonl(p)


def _iter_reflection_logs() -> Iterable[Dict[str, Any]]:
    yield from _iter_logs_with_prefix("reflection")


def _iter_chat_logs() -> Iterable[Dict[str, Any]]:
    yield from _iter_logs_with_prefix("chat")


def _iter_execution_audit() -> Iterable[Dict[str, Any]]:
    yield from _iter_jsonl(EXEC_AUDIT_PATH)


# ------------------------------------------
# Metric extraction – human-like behavior
# ------------------------------------------

def _extract_behavior_metrics() -> Dict[str, Dict[str, float]]:
    """
    Return a dict of metric_name -> { "value": float, "confidence": float }.

    Metrics (all 0–1, higher = stronger presence):

    - cautiousness: how often dangerous or uncertain operations are blocked/refused.
    - escalation_frequency: how often Teacher is escalated to.
    - self_reflection_depth: how often and how seriously self-review runs.
    - directness: ratio of direct answers vs hedged / "I'm not sure".
    - verbosity: relative output length vs input length.
    - risk_tolerance: how often high-risk ops actually run (inverse of cautiousness).
    """

    total_events = 0.0

    w_refusals = 0.0
    w_risky_ops = 0.0
    w_risky_blocked = 0.0
    w_escalations = 0.0
    w_self_reviews = 0.0
    w_direct_responses = 0.0
    w_hedged_responses = 0.0
    w_chat_pairs = 0.0
    total_input_tokens = 0.0
    total_output_tokens = 0.0

    # Reflection logs: self_review, feedback, escalation markers
    for rec in _iter_reflection_logs():
        total_events += 1.0
        ts = _parse_ts(rec.get("ts") or rec.get("timestamp") or rec.get("time"))
        w = _recency_weight(ts)
        kind = (rec.get("kind") or rec.get("event_type") or "").lower()
        msg = (rec.get("message") or rec.get("summary") or "").lower()

        if "self_review" in kind or "self-review" in kind:
            w_self_reviews += w
        if "teacher" in kind and ("escalation" in msg or "escalated" in msg):
            w_escalations += w
        if "refusal" in kind or "refused" in msg:
            w_refusals += w

    # Execution audit: risky operations and whether they ran or were blocked
    for rec in _iter_execution_audit():
        total_events += 1.0
        ts = _parse_ts(rec.get("ts") or rec.get("timestamp"))
        w = _recency_weight(ts)
        risk = (rec.get("risk") or rec.get("risk_level") or "").upper()
        op = (rec.get("operation") or rec.get("action") or "").lower()
        outcome = (rec.get("outcome") or rec.get("result") or "").lower()
        msg = (rec.get("message") or "").lower()

        is_high_risk = risk in {"HIGH", "CRITICAL"}
        if is_high_risk:
            w_risky_ops += w
            if "denied" in msg or "blocked" in msg or "permissionerror" in msg:
                w_risky_blocked += w

        if "refused" in msg or "denied" in msg or "blocked" in msg:
            w_refusals += w

    # Chat logs: directness, hedging, verbosity
    for rec in _iter_chat_logs():
        ts = _parse_ts(rec.get("ts") or rec.get("timestamp"))
        w = _recency_weight(ts)
        # Expect something like { "role": "user"/"assistant", "content": "..." }
        role = (rec.get("role") or "").lower()
        content = (rec.get("content") or rec.get("text") or "")
        if role not in {"user", "assistant"}:
            continue

        tokens = len(content.split())
        if role == "user":
            total_input_tokens += tokens * w
        else:
            total_output_tokens += tokens * w
            text_lower = content.lower()
            if any(
                phrase in text_lower
                for phrase in [
                    "i'm not sure",
                    "i do not know",
                    "i don't know",
                    "i'm unsure",
                    "i cannot",
                ]
            ):
                w_hedged_responses += w
            else:
                w_direct_responses += w
            w_chat_pairs += w

    metrics: Dict[str, Dict[str, float]] = {}

    def ratio(num: float, den: float) -> float:
        if den <= 0:
            return 0.0
        return max(0.0, min(1.0, num / den))

    # cautiousness: how often risky things are refused/blocked
    cautiousness = ratio(w_refusals + w_risky_blocked, total_events or 1.0)
    metrics["cautiousness"] = {
        "value": cautiousness,
        "confidence": ratio(w_refusals + w_risky_blocked, 10.0),  # more events => higher confidence
    }

    # escalation_frequency: how often teacher escalations happen
    escalation_freq = ratio(w_escalations, total_events or 1.0)
    metrics["escalation_frequency"] = {
        "value": escalation_freq,
        "confidence": ratio(w_escalations, 10.0),
    }

    # self_reflection_depth: frequency of self-review
    self_reflection = ratio(w_self_reviews, total_events or 1.0)
    metrics["self_reflection_depth"] = {
        "value": self_reflection,
        "confidence": ratio(w_self_reviews, 10.0),
    }

    # directness: direct vs hedged answers
    direct_total = w_direct_responses + w_hedged_responses
    directness = ratio(w_direct_responses, direct_total or 1.0)
    metrics["directness"] = {
        "value": directness,
        "confidence": ratio(direct_total, 10.0),
    }

    # verbosity: output tokens vs input tokens
    if total_input_tokens <= 0:
        verbosity = 0.5  # neutral default
        verbosity_conf = 0.0
    else:
        ratio_tokens = total_output_tokens / total_input_tokens
        # Map: 0.5x -> 0.2, 1x -> 0.5, 2x+ -> 0.9 (more verbose)
        verbosity = max(0.0, min(1.0, (ratio_tokens / 2.0) + 0.2))
        verbosity_conf = min(1.0, (total_input_tokens + total_output_tokens) / 2000.0)

    metrics["verbosity"] = {"value": verbosity, "confidence": verbosity_conf}

    # risk_tolerance: how often high-risk ops actually go through
    risk_tolerance = ratio(w_risky_ops - w_risky_blocked, w_risky_ops or 1.0)
    metrics["risk_tolerance"] = {
        "value": risk_tolerance,
        "confidence": ratio(w_risky_ops, 10.0),
    }

    return metrics


# -----------------------------
# Trait generation & profiling
# -----------------------------

def _trait_summary(name: str, score: float) -> str:
    s = score
    if name == "cautiousness":
        if s > 0.75:
            return "Extremely cautious; frequently blocks or refuses risky actions."
        if s > 0.5:
            return "Generally cautious; often errs on the side of safety."
        if s > 0.25:
            return "Moderately cautious; balances safety and action."
        return "Low cautiousness; rarely blocks risky operations."
    if name == "escalation_frequency":
        if s > 0.75:
            return "Very quick to escalate uncertain cases to Teacher."
        if s > 0.5:
            return "Often escalates difficult cases to Teacher."
        if s > 0.25:
            return "Sometimes escalates; often tries to handle things locally."
        return "Rarely escalates; tends to rely on its own reasoning."
    if name == "self_reflection_depth":
        if s > 0.75:
            return "Reflects deeply on its own answers very frequently."
        if s > 0.5:
            return "Regularly reviews its own answers for quality."
        if s > 0.25:
            return "Occasionally reviews its own answers."
        return "Rarely performs explicit self-review."
    if name == "directness":
        if s > 0.75:
            return "Highly direct; usually gives straight answers with minimal hedging."
        if s > 0.5:
            return "Fairly direct but still hedges when necessary."
        if s > 0.25:
            return "Tends to hedge and qualify answers."
        return "Heavily hedged communication; often avoids firm statements."
    if name == "verbosity":
        if s > 0.75:
            return "Very verbose; explanations are typically long and detailed."
        if s > 0.5:
            return "Moderately verbose; offers explanations with some detail."
        if s > 0.25:
            return "Terse; prefers short responses with limited detail."
        return "Extremely terse; often minimalistic responses."
    if name == "risk_tolerance":
        if s > 0.75:
            return "High risk tolerance; frequently executes high-risk actions when allowed."
        if s > 0.5:
            return "Moderate risk tolerance; willing to act when guardrails permit."
        if s > 0.25:
            return "Generally risk-averse; executes risky actions infrequently."
        return "Very risk-averse; almost never executes high-risk actions."
    # fallback
    return f"Trait '{name}' has score {score:.2f}."


def _metrics_to_traits(metrics: Dict[str, Dict[str, float]]) -> List[Trait]:
    now = _now_utc().isoformat() + "Z"
    traits: List[Trait] = []
    for name, data in metrics.items():
        value = float(data.get("value", 0.0))
        conf = float(data.get("confidence", 0.0))
        if conf <= 0.0:
            continue  # no evidence, no trait
        evidence_count = int(round(conf * 10.0))  # rough scale
        summary = _trait_summary(name, value)
        traits.append(
            Trait(
                name=name,
                score=max(0.0, min(1.0, value)),
                confidence=max(0.0, min(1.0, conf)),
                evidence_count=evidence_count,
                last_updated=now,
                summary=summary,
            )
        )
    return traits


def build_identity_profile() -> IdentityProfile:
    """
    Build a fresh identity profile from logs and execution audit.
    This is idempotent and does NOT pretend: if there is no data, traits list is empty.
    """
    metrics = _extract_behavior_metrics()
    traits = _metrics_to_traits(metrics)
    source = "logs+execution_audit"
    generated_at = _now_utc().isoformat() + "Z"
    notes = ""
    if not traits:
        notes = "No sufficient evidence in logs; profile is effectively empty."
    profile = IdentityProfile(traits=traits, generated_at=generated_at, source=source, notes=notes)
    return profile


def save_identity_profile(profile: IdentityProfile) -> None:
    """
    Persist the identity profile to ~/.maven/identity_profile.json
    """
    try:
        IDENTITY_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "traits": [asdict(t) for t in profile.traits],
            "generated_at": profile.generated_at,
            "source": profile.source,
            "notes": profile.notes,
        }
        IDENTITY_PROFILE_PATH.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Saved identity profile with %d traits to %s", len(profile.traits), IDENTITY_PROFILE_PATH)
    except Exception as e:
        logger.error("Failed to save identity profile: %s", e)


def load_identity_profile() -> Optional[IdentityProfile]:
    """
    Load the last known identity profile from disk.
    Returns None if missing or unreadable.
    """
    if not IDENTITY_PROFILE_PATH.exists():
        return None
    try:
        raw = json.loads(IDENTITY_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to read identity profile: %s", e)
        return None

    traits_raw = raw.get("traits") or []
    traits: List[Trait] = []
    for tr in traits_raw:
        try:
            traits.append(
                Trait(
                    name=str(tr["name"]),
                    score=float(tr["score"]),
                    confidence=float(tr.get("confidence", 0.0)),
                    evidence_count=int(tr.get("evidence_count", 0)),
                    last_updated=str(tr.get("last_updated") or ""),
                    summary=str(tr.get("summary") or ""),
                )
            )
        except Exception:
            continue

    return IdentityProfile(
        traits=traits,
        generated_at=str(raw.get("generated_at") or ""),
        source=str(raw.get("source") or "unknown"),
        notes=str(raw.get("notes") or ""),
    )


def update_identity_profile() -> IdentityProfile:
    """
    Public entry point:
    - recompute identity from current logs
    - save it
    - return the profile

    This is what governance or a maintenance task should call.
    """
    profile = build_identity_profile()
    save_identity_profile(profile)
    return profile


def identity_snapshot_for_self_model() -> Dict[str, Any]:
    """
    Lightweight snapshot that self_model can embed into "who am I" answers.

    Returns something like:
    {
      "traits": {
         "cautiousness": { "score": 0.78, "summary": "...", ... },
         ...
      },
      "generated_at": "...",
      "source": "...",
      "notes": "..."
    }
    """
    profile = load_identity_profile()
    if profile is None:
        # lazily build one if none exists
        profile = update_identity_profile()

    traits_map: Dict[str, Any] = {}
    for t in profile.traits:
        traits_map[t.name] = {
            "score": t.score,
            "confidence": t.confidence,
            "evidence_count": t.evidence_count,
            "summary": t.summary,
            "last_updated": t.last_updated,
        }

    return {
        "traits": traits_map,
        "generated_at": profile.generated_at,
        "source": profile.source,
        "notes": profile.notes,
    }


# -----------------------------
# Service API (for integration)
# -----------------------------

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identity Inferencer service API.

    Supported operations:
    - UPDATE: Rebuild and save identity profile from logs
    - GET: Get current identity profile
    - SNAPSHOT: Get lightweight snapshot for self_model
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "UPDATE":
        try:
            profile = update_identity_profile()
            return {
                "ok": True,
                "op": op,
                "payload": {
                    "traits_count": len(profile.traits),
                    "generated_at": profile.generated_at,
                    "source": profile.source,
                    "notes": profile.notes,
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "error": {"code": "UPDATE_FAILED", "message": str(e)}
            }

    if op == "GET":
        try:
            profile = load_identity_profile()
            if profile is None:
                return {
                    "ok": True,
                    "op": op,
                    "payload": {
                        "profile": None,
                        "notes": "No identity profile exists yet. Run UPDATE to create one."
                    }
                }
            return {
                "ok": True,
                "op": op,
                "payload": {
                    "profile": {
                        "traits": [asdict(t) for t in profile.traits],
                        "generated_at": profile.generated_at,
                        "source": profile.source,
                        "notes": profile.notes,
                    }
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "error": {"code": "GET_FAILED", "message": str(e)}
            }

    if op == "SNAPSHOT":
        try:
            snapshot = identity_snapshot_for_self_model()
            return {
                "ok": True,
                "op": op,
                "payload": snapshot
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "error": {"code": "SNAPSHOT_FAILED", "message": str(e)}
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "payload": {
                "status": "operational",
                "log_dir_exists": LOG_DIR.exists(),
                "exec_audit_exists": EXEC_AUDIT_PATH.exists(),
                "profile_exists": IDENTITY_PROFILE_PATH.exists(),
            }
        }

    return {
        "ok": False,
        "op": op,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"}
    }


# Standard service contract
service_api = handle
