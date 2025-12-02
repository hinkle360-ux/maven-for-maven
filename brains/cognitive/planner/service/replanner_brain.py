"""
Replanner Brain
===============

This module provides a simple re‑planning mechanism for incomplete goals.  It
accepts a list of goals (as persisted in the personal goal memory) and
attempts to break compound tasks into smaller, more manageable actions.  The
replanner is intentionally conservative: it only splits goal titles on
conjunctions like "and" or "then" and discards empty segments.  When new
sub‑goals are generated they are added to the goal memory for future
execution.

Operations
----------

    REPLAN
        Given a list of goal dictionaries, return a list of newly created
        sub‑goals.  Each dictionary in the payload should include at least a
        ``title`` field.  Additional fields are ignored.  If no goals are
        provided or no splits are possible, an empty list is returned.

Example::

    from brains.cognitive.planner.service.replanner_brain import service_api
    goals = [ {"goal_id": "123", "title": "gather data and analyze it"} ]
    resp = service_api({"op": "REPLAN", "payload": {"goals": goals}})
    # resp["payload"]["new_goals"] might be something like
    # [ {"goal_id": "NEW-0001", "title": "gather data"}, {"goal_id": "NEW-0002", "title": "analyze it"} ]

The replanner uses the same ``goal_memory`` module as the planner to persist
new goals.  If goal memory is unavailable, replanning silently degrades to
returning the original goals unchanged.
"""

from __future__ import annotations

from typing import Dict, Any, List
import itertools

# Cognitive Brain Contract: Continuation awareness
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_enabled = True
except Exception as e:
    print(f"[REPLANNER] Continuation helpers not available: {e}")
    _continuation_enabled = False
    # Fallback stubs
    def is_continuation(*args, **kwargs): return False  # type: ignore
    def get_conversation_context(*args, **kwargs): return {}  # type: ignore
    def create_routing_hint(*args, **kwargs): return {}  # type: ignore

# Teacher integration for learning goal decomposition and replanning patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    from brains.memory.brain_memory import BrainMemory
    _teacher_helper = TeacherHelper("replanner")
    _memory = BrainMemory("replanner")
except Exception as e:
    print(f"[REPLANNER] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore
    _memory = None  # type: ignore

# Counter for assigning IDs to new sub‑goals.  New goals are prefixed with
# "REP-" to distinguish them from planner goals.  This is purely cosmetic and
# does not impact scheduling.
_replan_counter = itertools.count(1)


def _split_goal_title(title: str) -> List[str]:
    """Split a goal title on conjunctions and commas.

    Returns a list of non‑empty segments.  Only splits when multiple segments
    are detected.  If the title contains no conjunctions or only yields a
    single segment, the original title is returned in a single‑element list.

    Args:
        title: The goal title to split.

    Returns:
        A list of one or more strings representing sub‑tasks.
    """
    try:
        import re
        # Normalise whitespace
        t = (title or "").strip()

        # Check for learned splitting patterns first
        learned_split = None
        if _teacher_helper and _memory and len(t) > 10:
            try:
                title_preview = t[:50]
                learned_patterns = _memory.retrieve(
                    query=f"goal splitting pattern: {title_preview[:30]}",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", {})
                        if isinstance(content, dict) and "parts" in content:
                            learned_split = content.get("parts")
                            print(f"[REPLANNER] Using learned goal splitting pattern from Teacher")
                            break
            except Exception:
                pass

        # Use learned split if found, otherwise use regex heuristics
        if learned_split and isinstance(learned_split, list) and len(learned_split) > 0:
            return learned_split

        # First detect conditional pattern: if X then Y
        cond_match = re.search(r"\bif\s+(.+?)\s+then\s+(.+)", t, flags=re.IGNORECASE)
        if cond_match:
            cond = cond_match.group(1).strip()
            act = cond_match.group(2).strip()
            out: List[str] = []
            if cond:
                out.append(cond)
            if act:
                out.append(act)
            parts = out if out else [t]
        else:
            # Otherwise split on "and"/"then"/commas
            parts = [p.strip() for p in re.split(r"\b(?:and|then)\b|,", t, flags=re.IGNORECASE) if p and p.strip()]
            if len(parts) <= 1:
                parts = [t]

        # If no learned pattern and we split successfully and Teacher available, try to learn
        if not learned_split and len(parts) > 1 and _teacher_helper and len(t) > 10:
            try:
                print(f"[REPLANNER] No learned splitting pattern, calling Teacher...")
                teacher_result = _teacher_helper.maybe_call_teacher(
                    question=f"How should I split the goal '{t}' into sub-tasks?",
                    context={
                        "goal_title": t,
                        "current_split": parts,
                        "split_count": len(parts)
                    },
                    check_memory_first=True
                )

                if teacher_result and teacher_result.get("answer"):
                    patterns_stored = teacher_result.get("patterns_stored", 0)
                    print(f"[REPLANNER] Learned from Teacher: {patterns_stored} splitting patterns stored")
                    # Learned pattern now in memory for future use
            except Exception as e:
                print(f"[REPLANNER] Teacher call failed: {str(e)[:100]}")

        return parts
    except Exception:
        return [title or ""]


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the replanner brain.

    Supports the ``REPLAN`` operation which takes a list of goal dictionaries
    and returns new sub‑goals for any compound goal titles.  New goals are
    written to the personal goal memory.  Unknown operations return an
    error.

    Args:
        msg: A message dictionary containing an ``op`` key and optional
            ``payload``.

    Returns:
        A response dictionary with ``ok`` status and operation name.  On
        success, the payload contains a list of new goals under ``new_goals``.
        On failure, ``error`` describes the issue.
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
    context = (msg or {}).get("context", {}) or {}

    if op != "REPLAN":
        return {"ok": False, "op": op, "error": "unsupported_operation"}

    # Cognitive Brain Contract: Get conversation context
    conv_context = get_conversation_context() if _continuation_enabled else {}

    # Cognitive Brain Contract: Detect if this is a continuation of multi-step planning
    user_query = context.get("user_query", "")
    is_follow_up = is_continuation(user_query, context) if _continuation_enabled else False

    goals = payload.get("goals") or []
    new_goals: List[Dict[str, Any]] = []

    # Track replanning mode: iterative refinement vs. fresh planning
    replanning_mode = "iterative_refinement" if is_follow_up else "fresh_replan"

    # If this is a follow-up, check for previous replanning context
    previous_goals = []
    if is_follow_up:
        previous_goals = conv_context.get("last_replanned_goals", [])
        if previous_goals:
            print(f"[REPLANNER] Continuation detected: Refining {len(previous_goals)} previous goals")

    # Import goal memory lazily to avoid dependency issues when unused
    try:
        from brains.personal.memory import goal_memory  # type: ignore
    except Exception:
        goal_memory = None  # type: ignore

    for g in goals:
        title = str(g.get("title", "")).strip()
        if not title:
            continue
        parts = _split_goal_title(title)
        # Only split if multiple parts are returned.  When splitting, we
        # chain the resulting sub‑goals by specifying a dependency on
        # the immediately preceding sub‑goal.  The first segment has
        # no dependencies.  Each sub‑goal is persisted via the goal
        # memory and the returned record's identifier is used to set
        # dependencies for subsequent segments.
        if len(parts) > 1:
            # Determine if a conditional split should set conditions on
            # subsequent goals.  When the original title matches "if X then Y",
            # we apply a "success" condition to Y.  Otherwise, all parts
            # are unconditional.
            try:
                import re
                t = str(title).strip()
                cond_match = re.search(r"\bif\s+(.+?)\s+then\s+(.+)", t, flags=re.IGNORECASE)
                if cond_match:
                    cond_flags: List[Optional[str]] = [None, "success"]
                else:
                    cond_flags = [None for _ in parts]
            except Exception:
                cond_flags = [None for _ in parts]
            prev_id: str | None = None
            for part, cond_flag in zip(parts, cond_flags):
                created_rec: Dict[str, Any] | None = None
                # Persist the new goal in personal memory if available
                if goal_memory is not None:
                    try:
                        rec = goal_memory.add_goal(
                            part,
                            depends_on=[prev_id] if prev_id else None,
                            condition=cond_flag,
                        )
                        created_rec = rec
                        prev_id = rec.get("goal_id") if isinstance(rec, dict) else prev_id
                    except Exception:
                        created_rec = None
                # Create a new goal dict for return.  Use the created
                # record's goal_id if available; otherwise fall back to
                # synthetic REP prefix
                if created_rec and isinstance(created_rec, dict) and created_rec.get("goal_id"):
                    new_goal_id = created_rec["goal_id"]
                else:
                    new_goal_id = f"REP-{next(_replan_counter):04d}"
                new_goal = {
                    "goal_id": new_goal_id,
                    "title": part,
                    "status": "pending",
                }
                new_goals.append(new_goal)

    # Calculate confidence based on number of goals split successfully
    goals_split = sum(1 for g in goals if len(_split_goal_title(str(g.get("title", "")).strip())) > 1)
    confidence = min(0.95, 0.7 + (0.05 * goals_split)) if goals_split > 0 else 0.6

    # Cognitive Brain Contract: Add routing hint
    routing_hint = create_routing_hint(
        brain_name="replanner",
        action=replanning_mode,
        confidence=confidence,
        context_tags=["goal_decomposition", "multi_step_planning", "continuation"] if is_follow_up else ["goal_decomposition", "multi_step_planning"]
    ) if _continuation_enabled else {}

    result = {
        "ok": True,
        "op": op,
        "payload": {
            "new_goals": new_goals,
            "replanning_mode": replanning_mode,
            "goals_split": goals_split,
            "is_continuation": is_follow_up
        }
    }

    # Add routing hint to result
    if routing_hint:
        result["routing_hint"] = routing_hint

    return result

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