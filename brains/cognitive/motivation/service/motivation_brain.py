"""
Motivation Brain
================

This is a skeleton implementation of a simple motivation brain.  It introduces
a light‑weight mechanism for identifying opportunities (questions or knowledge
gaps) and formulating goals to address them.  The goal of this brain is to
support higher‑level autonomous behaviour by suggesting topics worth
investigating.  In the current implementation it does not depend on any
external libraries and returns empty sets by default.

Operations:

  SCORE_OPPORTUNITIES
      Examine the provided context or evidence and return a list of
      opportunity descriptors.  Each opportunity is a dict with fields
      ``kind``, ``target`` and ``score``.  Higher scores indicate more
      pressing opportunities.  If no evidence is provided the list is empty.

  FORMULATE_GOALS
      Convert a list of opportunity descriptors into high‑level goals.  Each
      goal is a dict with ``goal_id``, ``type``, ``target``, ``priority``
      and ``rationale``.  Goal identifiers are prefixed with ``MOT-`` and
      include a simple monotonic counter.

Example usage:

    from brains.cognitive.motivation.service.motivation_brain import service_api
    resp = service_api({"op": "SCORE_OPPORTUNITIES", "payload": {"evidence": {...}}})
    opps = resp.get("payload", {}).get("opportunities", [])
    goals = service_api({"op": "FORMULATE_GOALS", "payload": {"opportunities": opps}})

This module is intentionally conservative and will evolve as autonomy features
are refined.  For now it returns empty lists when no useful information is
present.
"""

from __future__ import annotations
from typing import Dict, Any, List
import itertools
import json
from pathlib import Path
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning goal generation and motivation patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("motivation")
except Exception as e:
    print(f"[MOTIVATION] Teacher helper not available: {e}")
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
    print(f"[MOTIVATION] Continuation helpers not available: {e}")
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("motivation")

_counter = itertools.count(1)

_DEFAULT_STATE = {
    "helpfulness": 0.8,
    "truthfulness": 0.9,
    "curiosity": 0.5,
    "self_improvement": 0.5
}

def _get_state_path() -> Path:
    """Get the path to the motivation state file."""
    here = Path(__file__).resolve().parent
    return here.parent / "memory" / "motivation_state.json"

def _load_motivation_state() -> Dict[str, float]:
    """Load current motivation state using BrainMemory tier API."""
    try:
        results = _memory.retrieve(query="motivation_state", limit=1)
        if results:
            latest = results[0]
            data = latest.get("content", {})
            return data if isinstance(data, dict) else _DEFAULT_STATE.copy()
    except Exception:
        pass
    return _DEFAULT_STATE.copy()

def _save_motivation_state(state: Dict[str, float]) -> None:
    """Save motivation state using BrainMemory tier API."""
    try:
        _memory.store(
            content=state,
            metadata={"kind": "motivation_state", "source": "motivation", "confidence": 0.95}
        )
    except Exception:
        pass

def _generate_goal_id() -> str:
    return f"MOT-{next(_counter):04d}"

def _score_opportunities(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scan evidence for unanswered questions or gaps.

    This naive implementation looks for entries in the evidence with
    ``result_type`` set to "unanswered_question" and assigns a score based on
    the number of times the question has been asked.  An opportunity is a
    dict with keys ``kind``, ``target`` and ``score``.  If no suitable
    evidence is found an empty list is returned.
    """
    if not isinstance(evidence, dict):
        return []
    opps: List[Dict[str, Any]] = []
    results = (evidence.get("results") or []) if isinstance(evidence.get("results"), list) else []
    for rec in results:
        if not isinstance(rec, dict):
            continue
        # use a synthetic tag to mark unanswered questions; this can be
        # extended in future revisions
        if rec.get("result_type") == "unanswered_question":
            qtext = str(rec.get("query", "")).strip()
            if not qtext:
                continue
            # compute a simple score: the more times we've seen this question,
            # the higher the score.  Default to 1.0.
            try:
                count = int(rec.get("times", 1))
            except Exception:
                count = 1
            score = min(1.0, 0.1 * float(count)) + 0.5
            opps.append({"kind": "knowledge_gap", "target": qtext, "score": score})
    return opps

def _formulate_goals(opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform opportunities into concrete goals.

    A goal summarises a high‑level action for the agent to pursue.  Each goal
    contains a unique ID, a goal type, target information, a priority (0..1)
    and a human‑readable rationale.  If no opportunities are provided, an
    empty list is returned.
    """
    if not isinstance(opportunities, list):
        return []

    # Check for learned goal formulation patterns first
    learned_pattern = None
    if _teacher_helper and _memory and len(opportunities) > 0:
        try:
            # Get the type of the first opportunity to check for learned patterns
            first_kind = opportunities[0].get("kind", "unknown")
            learned_patterns = _memory.retrieve(
                query=f"goal formulation pattern: {first_kind}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.6:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, str) and len(content) > 10:
                        learned_pattern = content
                        print(f"[MOTIVATION] Using learned goal formulation pattern from Teacher")
                        break
        except Exception:
            pass

    goals: List[Dict[str, Any]] = []
    for opp in opportunities:
        if not isinstance(opp, dict):
            continue
        kind = opp.get("kind") or "unknown"
        target = opp.get("target") or ""
        score = float(opp.get("score", 0.5))
        gid = _generate_goal_id()

        # Use learned pattern if found, otherwise use default template
        if learned_pattern:
            rationale = learned_pattern.format(target=target, kind=kind) if "{target}" in learned_pattern else learned_pattern
        else:
            rationale = f"Address knowledge gap regarding '{target}'."

        goals.append({
            "goal_id": gid,
            "type": kind,
            "target": target,
            "priority": max(0.0, min(1.0, score)),
            "rationale": rationale
        })

    # If no learned pattern and Teacher available, try to learn
    if not learned_pattern and _teacher_helper and len(opportunities) > 0:
        try:
            first_kind = opportunities[0].get("kind", "unknown")
            first_target = opportunities[0].get("target", "")
            print(f"[MOTIVATION] No learned goal pattern, calling Teacher...")
            teacher_result = _teacher_helper.maybe_call_teacher(
                question=f"How should I formulate goals for {first_kind} opportunities like '{first_target}'?",
                context={
                    "opportunity_kind": first_kind,
                    "example_target": first_target,
                    "opportunities_count": len(opportunities)
                },
                check_memory_first=True
            )

            if teacher_result and teacher_result.get("answer"):
                patterns_stored = teacher_result.get("patterns_stored", 0)
                print(f"[MOTIVATION] Learned from Teacher: {patterns_stored} goal patterns stored")
                # Learned pattern now in memory for future use
        except Exception as e:
            print(f"[MOTIVATION] Teacher call failed: {str(e)[:100]}")

    return goals

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the motivation brain.

    Supports the following operations:

      - GET_STATE: Return current motivation drive vector
      - ADJUST_STATE: Modify motivation drives with bounded deltas
      - EVALUATE_QUERY: Compute motivation weights for a specific query
      - SCORE_OPPORTUNITIES: Identify potential knowledge gaps
      - FORMULATE_GOALS: Construct concrete goals from opportunities
      - SCORE_DRIVE: Compute overall motivation drive signal

    Unknown operations return an error response.
    """
    op = (msg or {}).get("op", "").upper()
    payload = (msg or {}).get("payload", {}) or {}

    if op == "GET_STATE":
        state = _load_motivation_state()
        return {"ok": True, "op": op, "payload": state}

    if op == "ADJUST_STATE":
        deltas = payload.get("deltas", {})
        current_state = _load_motivation_state()

        for key, delta in deltas.items():
            if key in current_state:
                try:
                    delta_val = float(delta)
                    delta_val = max(-0.2, min(0.2, delta_val))
                    new_val = current_state[key] + delta_val
                    current_state[key] = max(0.0, min(1.0, new_val))
                except Exception:
                    pass

        _save_motivation_state(current_state)
        return {"ok": True, "op": op, "payload": current_state}

    if op == "EVALUATE_QUERY":
        query = str(payload.get("query", ""))
        context = payload.get("context", {})

        state = _load_motivation_state()
        weights = state.copy()

        # FOLLOW-UP DETECTION: Check if this is a continuation
        is_cont = False
        try:
            is_cont = is_continuation(query, context)
            if is_cont:
                print("[MOTIVATION] Detected follow-up question, adjusting motivation")
                # For continuations, maintain current motivation levels
                # Don't re-boost drives that were already activated
        except Exception as e:
            print(f"[MOTIVATION] Could not check continuation: {e}")

        query_lower = query.lower()
        if any(word in query_lower for word in ["why", "how", "explain"]):
            # Boost curiosity less for continuations
            boost = 0.1 if is_cont else 0.2
            weights["curiosity"] = min(1.0, weights.get("curiosity", 0.5) + boost)
            weights["truthfulness"] = min(1.0, weights.get("truthfulness", 0.9) + 0.1)

        if "help" in query_lower or "please" in query_lower:
            boost = 0.08 if is_cont else 0.15
            weights["helpfulness"] = min(1.0, weights.get("helpfulness", 0.8) + boost)

        if any(word in query_lower for word in ["improve", "better", "enhance"]):
            boost = 0.1 if is_cont else 0.2
            weights["self_improvement"] = min(1.0, weights.get("self_improvement", 0.5) + boost)

        uncertainty = context.get("uncertainty", 0.0)
        if uncertainty > 0.7:
            weights["truthfulness"] = min(1.0, weights.get("truthfulness", 0.9) + 0.1)

        return {"ok": True, "op": op, "payload": {"weights": weights, "base_state": state}}

    if op == "SCORE_OPPORTUNITIES":
        evidence = payload.get("evidence") or {}
        opps = _score_opportunities(evidence)
        return {"ok": True, "op": op, "payload": {"opportunities": opps}}

    if op == "FORMULATE_GOALS":
        opps = payload.get("opportunities") or []
        goals = _formulate_goals(opps)
        return {"ok": True, "op": op, "payload": {"goals": goals}}

    if op == "SCORE_DRIVE":
        context = payload.get("context") or {}
        try:
            success = float(context.get("success_count", 0.0))
        except Exception:
            success = 0.0
        try:
            affect = float(context.get("affect_score", 0.0))
        except Exception:
            affect = 0.0
        try:
            contradictions = float(context.get("contradictions", 0.0))
        except Exception:
            contradictions = 0.0
        try:
            from api.utils import CFG  # type: ignore
            weights = (CFG.get("motivation") or {})
            w_success = float(weights.get("weight_success", 0.4))
            w_affect = float(weights.get("weight_affect", 0.4))
            w_contra = float(weights.get("weight_contradiction", 0.2))
        except Exception:
            w_success = 0.4
            w_affect = 0.4
            w_contra = 0.2
        drive = (w_success * success) + (w_affect * affect) - (w_contra * contradictions)
        try:
            drive = max(0.0, min(1.0, float(drive)))
        except Exception:
            drive = 0.0
        return {"ok": True, "op": op, "payload": {"drive": drive}}

    return {"ok": False, "op": op, "error": "unknown operation"}

def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bid for attention based on motivation needs.

    Provides routing hints with conversation context for continuations.
    """
    try:
        lang_info = ctx.get("stage_3_language", {}) or {}
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")
        conv_context = lang_info.get("conversation_context", {})

        # Motivation bids low by default
        routing_hint = create_routing_hint(
            brain_name="motivation",
            action="assess_goals",
            confidence=0.15,
            context_tags=["motivation", "goals"],
            metadata={
                "is_continuation": is_cont,
                "continuation_type": continuation_intent,
                "last_topic": conv_context.get("last_topic", "")
            }
        )

        return {
            "brain_name": "motivation",
            "priority": 0.15,
            "reason": "goal_assessment",
            "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
        }
    except Exception:
        return {
            "brain_name": "motivation",
            "priority": 0.15,
            "reason": "default",
            "evidence": {},
        }

# Standard service contract: handle is the entry point
service_api = handle