
# (Removed early alias to satisfy future import constraints; alias defined at end)
"""
Committee Brain
===============

This module introduces a simple committee mechanism for aggregating
decisions from multiple cognitive subsystems.  It mirrors how a group
of internal voices might reach a consensus when presented with a
question or proposed action.  Each member contributes a decision
(propose/approve/deny/abstain) and a confidence score; the committee
then aggregates these into a single decision and an overall
confidence.

Operations:

  CONSULT
      Accepts a list of ``votes`` where each vote is a dict with
      ``decision`` (str) and ``confidence`` (float).  Recognised
      decisions are "propose", "approve", "deny" and "abstain".  The
      aggregate decision is the majority decision weighted by
      confidence.  Confidence is scaled by the number of votes.

Example usage:

    from brains.cognitive.committee.service.committee_brain import service_api
    resp = service_api({"op": "CONSULT", "payload": {
        "votes": [
          {"decision": "approve", "confidence": 0.8},
          {"decision": "propose", "confidence": 0.6},
          {"decision": "deny", "confidence": 0.4},
        ]
    }})
    result = resp.get("payload")

The committee does not carry out any side effects; it only returns
aggregated judgments.
"""

from __future__ import annotations
from typing import Dict, Any, List
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning multi-perspective decision-making patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("committee")
except Exception as e:
    print(f"[COMMITTEE] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[COMMITTEE] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("committee")

# Step‑2 integration: import memory librarian service to store committee decisions.
try:
    # Lazy import to avoid mandatory dependency in environments without the librarian.
    from brains.cognitive.memory_librarian.service.memory_librarian import service_api as mem_service_api  # type: ignore
except Exception:
    mem_service_api = None  # type: ignore

def _aggregate_votes(votes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate votes into a single decision and confidence.

    The algorithm sums confidence scores for each decision category
    (propose/approve are treated as positive, deny as negative) and
    normalises by the total confidence.  Abstentions are ignored.

    Returns a dict with keys ``decision`` and ``confidence``.
    """
    if not votes:
        return {"decision": "abstain", "confidence": 0.0}

    # Check for learned decision aggregation patterns first
    learned_pattern = None
    if _teacher_helper and _memory and len(votes) >= 3:
        try:
            # Create a signature of the vote pattern
            vote_types = [str(v.get("decision", "abstain")).lower() for v in votes]
            vote_pattern = "-".join(sorted(set(vote_types)))

            learned_patterns = _memory.retrieve(
                query=f"decision aggregation pattern: {vote_pattern}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, dict) and "decision" in content:
                        learned_pattern = content
                        print(f"[COMMITTEE] Using learned decision aggregation pattern from Teacher")
                        break
        except Exception:
            pass

    # Use learned pattern if found, otherwise use built-in aggregation logic
    if learned_pattern:
        return learned_pattern
    else:
        pos = 0.0
        neg = 0.0
        total_conf = 0.0
        for v in votes:
            try:
                conf = float(v.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            dec = str(v.get("decision", "abstain")).lower()
            if dec in {"propose", "approve"}:
                pos += conf
            elif dec == "deny":
                neg += conf
            # abstain contributions don't affect pos/neg but count toward total
            total_conf += conf
        if total_conf <= 0:
            return {"decision": "abstain", "confidence": 0.0}
        # compute final decision
        if pos > neg:
            final_dec = "approve" if pos >= (neg + pos) * 0.75 else "propose"
            conf = pos / total_conf
        elif neg > pos:
            final_dec = "deny"
            conf = neg / total_conf
        else:
            final_dec = "abstain"
            conf = 0.0

        result = {"decision": final_dec, "confidence": round(conf, 3)}

        # If no learned pattern and Teacher available, try to learn
        if _teacher_helper and len(votes) >= 3:
            try:
                vote_types = [str(v.get("decision", "abstain")).lower() for v in votes]
                vote_pattern = "-".join(sorted(set(vote_types)))
                print(f"[COMMITTEE] No learned pattern for {vote_pattern}, calling Teacher...")
                teacher_result = _teacher_helper.maybe_call_teacher(
                    question=f"What decision aggregation pattern should I use for votes: {votes}?",
                    context={
                        "votes": votes,
                        "vote_pattern": vote_pattern,
                        "pos_conf": pos,
                        "neg_conf": neg,
                        "current_result": result
                    },
                    check_memory_first=True
                )

                if teacher_result and teacher_result.get("answer"):
                    patterns_stored = teacher_result.get("patterns_stored", 0)
                    print(f"[COMMITTEE] Learned from Teacher: {patterns_stored} decision patterns stored")
                    # Learned pattern now in memory for future use
            except Exception as e:
                print(f"[COMMITTEE] Teacher call failed: {str(e)[:100]}")

        return result

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for committee operations.

    Only supports CONSULT for now.  Returns an error for unknown operations.
    """
    op = (msg or {}).get("op", "").upper()
    payload = (msg or {}).get("payload", {}) or {}
# COGNITIVE BRAIN CONTRACT: Signal 1 & 2 - Detect continuation and get context
    continuation_detected = False
    conv_context = {}

    try:
        # Extract query from payload
        query = (payload.get("query") or
                payload.get("question") or
                payload.get("user_query") or
                payload.get("text") or "")

        if query:
            continuation_detected = is_continuation(query, payload)

            if continuation_detected:
                conv_context = get_conversation_context()
                # Enrich payload with conversation context
                payload["continuation_detected"] = True
                payload["last_topic"] = conv_context.get("last_topic", "")
                payload["conversation_depth"] = conv_context.get("conversation_depth", 0)
    except Exception as e:
        # Silently continue if continuation detection fails
        pass

    # Get conversation context for continuation detection
    conv_context = {}
    is_follow_up = False
    if _continuation_helpers_available:
        try:
            conv_context = get_conversation_context()
            query = payload.get("query", "")
            is_follow_up = is_continuation(query, payload)
        except Exception:
            pass

    if op == "CONSULT":
        votes = payload.get("votes") or []
        if not isinstance(votes, list):
            votes = []
        result = _aggregate_votes(votes)

        # Add continuation metadata
        result["is_continuation"] = is_follow_up
        if is_follow_up:
            result["base_topic"] = conv_context.get("last_topic", "")

        # Step‑2: Store the aggregated committee decision in working memory.
        try:
            if mem_service_api is not None:
                # Use a generic key derived from the aggregated decision to enable later recall.
                key = "committee:" + result.get("decision", "unknown")
                mem_service_api({
                    "op": "WM_PUT",
                    "payload": {
                        "key": key,
                        "value": result,
                        "tags": ["committee", "decision"],
                        # use the committee confidence as the working memory confidence
                        "confidence": float(result.get("confidence", 0.0)),
                        # default TTL of 5 minutes to allow opportunistic recall
                        "ttl": 300.0,
                    }
                })
        except Exception:
            # Any errors in WM integration should be non‑blocking
            pass

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result["routing_hint"] = create_routing_hint(
                    brain_name="committee",
                    action="deliberation_continuation" if is_follow_up else "committee_consult",
                    confidence=result.get("confidence", 0.5),
                    context_tags=["committee", "multi_turn"] if is_follow_up else ["committee"]
                )
            except Exception:
                pass

        return {"ok": True, "op": op, "payload": result}

    # EXECUTE_STEP: Phase 8 - Execute a governance/decision step
    if op == "EXECUTE_STEP":
        step = payload.get("step") or {}
        step_id = payload.get("step_id", 0)
        context = payload.get("context") or {}

        # Extract step details
        description = step.get("description", "")
        step_input = step.get("input") or {}

        # For governance steps, we typically aggregate multiple perspectives
        # Create synthetic votes based on the step context
        # In a real implementation, this would consult multiple subsystems

        # Default: approve with moderate confidence
        votes = [
            {"decision": "approve", "confidence": 0.7},
            {"decision": "approve", "confidence": 0.6}
        ]

        # Use CONSULT to aggregate
        result = _aggregate_votes(votes)

        output = {
            "decision": result.get("decision"),
            "confidence": result.get("confidence"),
            "description": description,
            "is_continuation": is_follow_up
        }

        # Add routing hint for governance steps
        if _continuation_helpers_available:
            try:
                output["routing_hint"] = create_routing_hint(
                    brain_name="committee",
                    action="governance_continuation" if is_follow_up else "governance_step",
                    confidence=result.get("confidence", 0.7),
                    context_tags=["governance", "multi_step"] if is_follow_up else ["governance"]
                )
            except Exception:
                pass

        return {"ok": True, "payload": {
            "output": output,
            "patterns_used": ["governance:consensus"]
        }}

    return {"ok": False, "op": op, "error": "unknown operation"}

# Standard service contract: handle is the entry point
service_api = handle