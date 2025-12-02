
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning personality trait patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("personality")
except Exception as e:
    print(f"[PERSONALITY] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for follow-up and conversation context tracking
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint,
        extract_continuation_intent
    )
except Exception as e:
    print(f"[PERSONALITY] Continuation helpers not available: {e}")
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("personality")

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

PREFS_DEFAULT = {"prefer_explain": True, "tone": "neutral", "verbosity_target": 1.0}

def _read_preferences() -> dict:
    results = _memory.retrieve(limit=1)

    # Look for preferences in retrieved results
    for result in results:
        if result.get("metadata", {}).get("kind") == "preferences":
            return result.get("content", PREFS_DEFAULT)

    # If no preferences found, store and return defaults
    _memory.store(
        content=PREFS_DEFAULT,
        metadata={
            "kind": "preferences",
            "source": "personality",
            "confidence": 1.0
        }
    )
    return dict(PREFS_DEFAULT)

def _counts() -> Dict[str, int]:
    """Return a mapping of record counts for each memory tier after rotation."""
    from api.memory import rotate_if_needed, ensure_dirs, count_lines  # type: ignore
    # Rotate memory before computing counts to avoid overflow warnings
    try:
        rotate_if_needed(BRAIN_ROOT)
    except Exception:
        pass
    t = ensure_dirs(BRAIN_ROOT)
    return {
        "stm": count_lines(t["stm"]),
        "mtm": count_lines(t["mtm"]),
        "ltm": count_lines(t["ltm"]),
        "cold": count_lines(t["cold"]),
    }

def _load_aggregate() -> Dict[str, Any]:
    results = _memory.retrieve(limit=1)

    # Look for aggregate in retrieved results
    for result in results:
        if result.get("metadata", {}).get("kind") == "aggregate":
            return result.get("content", {})

    return {}

def _save_aggregate(agg: Dict[str, Any]) -> None:
    _memory.store(
        content=agg,
        metadata={
            "kind": "aggregate",
            "source": "personality",
            "confidence": 0.9
        }
    )

def _update_aggregate_from_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    agg = _load_aggregate() or {"total_runs": 0, "allowed": 0, "denied": 0, "quarantined": 0,
                                "tone_stats": {}, "goal_stats": {}}
    tone = (payload.get("tone") or "unknown")
    goal = (payload.get("goal") or "unknown")
    decision = (payload.get("decision") or "ALLOW")
    agg["total_runs"] = int(agg.get("total_runs", 0)) + 1
    if str(decision).upper() == "ALLOW":
        agg["allowed"] = int(agg.get("allowed", 0)) + 1
    elif str(decision).upper() == "DENY":
        agg["denied"] = int(agg.get("denied", 0)) + 1
    else:
        agg["quarantined"] = int(agg.get("quarantined", 0)) + 1
    agg.setdefault("tone_stats", {})
    agg.setdefault("goal_stats", {})
    agg["tone_stats"][tone] = int(agg["tone_stats"].get(tone, 0)) + 1
    agg["goal_stats"][goal] = int(agg["goal_stats"].get(goal, 0)) + 1
    _save_aggregate(agg)
    return agg

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _suggest_from_aggregate(agg: Dict[str, Any], prefs: Dict[str, Any]) -> Dict[str, Any]:
    # Always return a structured proposal when we have at least 10 runs
    total = int(agg.get("total_runs", 0) or 0)
    if total < 10:
        return {"planner": {"explain_bias_delta": 0.0},
                "language": {"verbosity_bias_delta": 0.0, "tone": prefs.get("tone", "neutral")}}

    goals = agg.get("goal_stats", {}) or {}
    tones = agg.get("tone_stats", {}) or {}
    respond = int(goals.get("respond", 0))
    explain = int(goals.get("explain", 0))
    # We also count 'answer_question' as respond-style
    respond += int(goals.get("answer_question", 0))
    respond_ratio = respond / total if total else 0.0
    explain_ratio = explain / total if total else 0.0

    # Check for learned personality adaptation patterns first
    learned_pattern = None
    if _teacher_helper and _memory:
        try:
            learned_patterns = _memory.retrieve(
                query="personality adaptation pattern",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, dict) and "planner" in content:
                        learned_pattern = content
                        print(f"[PERSONALITY] Using learned adaptation pattern from Teacher")
                        break
        except Exception:
            pass

    # Use learned pattern if found, otherwise use built-in heuristics
    if learned_pattern:
        proposal = learned_pattern
    else:
        # Planner: nudge toward balance using tiny, bounded steps
        if explain_ratio >= 0.70:
            planner_delta = +0.05
        elif respond_ratio >= 0.70:
            planner_delta = -0.05
        else:
            planner_delta = 0.0

        # Language: move bias toward target (default 1.0 means neutral)
        target = float(prefs.get("verbosity_target", 1.0) or 1.0)
        if target > 1.0:
            lang_delta = +min(0.10, target - 1.0)
        elif target < 1.0:
            lang_delta = -min(0.10, 1.0 - target)
        else:
            # If neutral target, derive a mild delta from tone prevalence
            lang_delta = 0.05 if (tones.get("neutral", 0) / total if total else 0) >= 0.60 else 0.0

        # Tone suggestion = majority tone if it clears 60%
        tone_suggest = None
        if tones:
            top_tone, top_ct = max(tones.items(), key=lambda kv: kv[1])
            if (top_ct / total) >= 0.60:
                tone_suggest = top_tone
        proposal = {
            "planner": {"explain_bias_delta": _clip(planner_delta, -0.10, +0.10)},
            "language": {"verbosity_bias_delta": _clip(lang_delta, -0.10, +0.10)}
        }
        if tone_suggest:
            proposal["language"]["tone"] = tone_suggest

        # If no learned pattern and Teacher available, try to learn
        if _teacher_helper and total >= 20:  # Only learn if we have good sample size
            try:
                print(f"[PERSONALITY] No learned pattern, calling Teacher...")
                teacher_result = _teacher_helper.maybe_call_teacher(
                    question=f"What personality adaptation pattern should I use for respond_ratio={respond_ratio:.2f}, explain_ratio={explain_ratio:.2f}?",
                    context={
                        "total_runs": total,
                        "respond_ratio": respond_ratio,
                        "explain_ratio": explain_ratio,
                        "current_proposal": proposal,
                        "preferences": prefs
                    },
                    check_memory_first=True
                )

                if teacher_result and teacher_result.get("answer"):
                    patterns_stored = teacher_result.get("patterns_stored", 0)
                    print(f"[PERSONALITY] Learned from Teacher: {patterns_stored} adaptation patterns stored")
                    # Learned pattern now in memory for future use
            except Exception as e:
                print(f"[PERSONALITY] Teacher call failed: {str(e)[:100]}")

    return proposal

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import generate_mid, success_response, error_response  # type: ignore
    op = (msg or {}).get("op"," ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    if op == "HEALTH":
        return success_response(op, mid, {"status":"operational","memory_health": _counts()})

    if op == "PREFERENCES_SNAPSHOT":
        return success_response(op, mid, {"preferences": _read_preferences()})

    if op == "LEARN_FROM_RUN":
        # Store run record
        rec = {
            "goal": payload.get("goal"),
            "tone": payload.get("tone"),
            "verbosity": payload.get("verbosity_hint"),
            "decision": payload.get("decision"),
            "bank": payload.get("bank")
        }
        _memory.store(
            content=rec,
            metadata={
                "kind": "run_record",
                "source": "personality",
                "confidence": 0.8
            }
        )

        # Update aggregate directly
        agg = _update_aggregate_from_run(payload)

        # Store aggregate update
        _memory.store(
            content={"op": "AGG_UPDATE", "aggregate_total_runs": agg.get("total_runs", 0)},
            metadata={
                "kind": "aggregate_update",
                "source": "personality",
                "confidence": 0.9
            }
        )

        return success_response(op, mid, {"logged": True, "aggregate_total_runs": agg.get("total_runs", 0)})

    if op == "ADAPT_WEIGHTS_SUGGEST":
        agg = _load_aggregate()
        prefs = _read_preferences()

        # FOLLOW-UP DETECTION: Check if context indicates continuation
        query = payload.get("query", "")
        context = payload.get("context", {})
        is_cont = False
        try:
            if query:
                is_cont = is_continuation(query, context)
                if is_cont:
                    print("[PERSONALITY] Detected follow-up, maintaining tone consistency")
        except Exception as e:
            print(f"[PERSONALITY] Could not check continuation: {e}")

        suggestion = _suggest_from_aggregate(agg, prefs)

        # For continuations, suggest maintaining previous tone
        if is_cont:
            suggestion["maintain_tone"] = True

        # Store suggestion
        _memory.store(
            content={"op": "ADAPT_SUGGEST", "suggestion": suggestion, "is_continuation": is_cont},
            metadata={
                "kind": "adapt_suggest",
                "source": "personality",
                "confidence": 0.85
            }
        )

        return success_response(op, mid, {"suggestion": suggestion})

    return error_response(op, mid, "UNSUPPORTED_OP", op)

def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bid for attention based on personality and tone preferences.

    Provides routing hints with conversation context for continuations.
    """
    try:
        lang_info = ctx.get("stage_3_language", {}) or {}
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")
        conv_context = lang_info.get("conversation_context", {})

        routing_hint = create_routing_hint(
            brain_name="personality",
            action="apply_preferences",
            confidence=0.20,
            context_tags=["personality", "preferences"],
            metadata={
                "is_continuation": is_cont,
                "continuation_type": continuation_intent,
                "last_topic": conv_context.get("last_topic", "")
            }
        )

        return {
            "brain_name": "personality",
            "priority": 0.20,
            "reason": "personality_preferences",
            "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
        }
    except Exception:
        return {
            "brain_name": "personality",
            "priority": 0.20,
            "reason": "default",
            "evidence": {},
        }

# Standard service contract: handle is the entry point
service_api = handle
