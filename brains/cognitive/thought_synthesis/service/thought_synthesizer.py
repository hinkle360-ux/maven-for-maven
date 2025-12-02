"""
Thought Synthesizer
===================

Step 2 implementation: combines partial thoughts and reasoning outputs
from the planner and reasoning brains into cohesive, higher level insights.

The synthesizer takes a plan (from planner), thought_steps (from reasoning),
memories, and context as input, and produces:
  - final_thoughts: a structured list of normalized thought steps
  - answer_skeleton: a template with slots for the language brain to fill

This enables the pipeline to have real internal "thinking" that drives
answer generation rather than ad-hoc templating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning thought synthesis and integration patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("thought_synthesis")
except Exception as e:
    print(f"[THOUGHT_SYNTHESIS] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("thought_synthesis")


def synthesize(thoughts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine a list of partial thoughts into a single synthesized thought.

    Legacy compatibility function. New code should use service_api with
    SYNTHESIZE operation instead.

    Args:
        thoughts: A list of thought dictionaries from various brains.
    Returns:
        A synthesized thought dictionary with final_thoughts and answer_skeleton.
        On error, includes an 'error' field with code and message.
    """
    # Delegate to service_api for actual synthesis
    if not thoughts:
        return {
            "final_thoughts": [],
            "answer_skeleton": {"kind": "error", "slots": {}},
            "error": {"code": "EMPTY_INPUT", "message": "No thoughts provided for synthesis"}
        }
    msg = {
        "op": "SYNTHESIZE",
        "payload": {
            "plan": None,
            "thought_steps": thoughts,
            "memories": [],
            "context": {},
        }
    }
    resp = service_api(msg)
    if resp.get("ok"):
        return resp.get("payload")
    # Service API returned an error
    error_info = resp.get("error", {"code": "UNKNOWN_ERROR", "message": "Synthesis failed"})
    return {
        "final_thoughts": [],
        "answer_skeleton": {"kind": "error", "slots": {}},
        "error": error_info
    }


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the thought synthesis brain.

    Supports operations:
      - SYNTHESIZE: Combine plan + thought_steps + memories into final_thoughts
        and answer_skeleton
      - HEALTH: Return operational status

    Args:
        msg: Request dictionary with 'op', 'mid', and 'payload'

    Returns:
        Response dictionary with 'ok', 'op', 'mid', and 'payload'
    """
    op = (msg or {}).get("op", "").upper()
    mid = (msg or {}).get("mid")
    payload = (msg or {}).get("payload") or {}

    if op == "SYNTHESIZE":
        plan = payload.get("plan")
        thought_steps = payload.get("thought_steps") or []
        memories = payload.get("memories") or []
        context = payload.get("context") or {}

        # ======================================================================
        # TASK 2: ENSURE CLEAN CANDIDATE FLOW
        # ======================================================================
        # Extract current question metadata for tagging all outputs.
        # This ensures thought_synthesis only produces thoughts for the
        # CURRENT question and tags them correctly for language_brain filtering.
        # ======================================================================
        original_query = context.get("original_query", "")

        # Compute question_signature and concept_key for this question
        try:
            from brains.learning.lesson_utils import canonical_concept_key
            current_concept_key = canonical_concept_key(original_query) if original_query else None
        except Exception:
            current_concept_key = None

        question_signature = original_query.lower().strip().rstrip("?").strip() if original_query else ""
        current_run_id = context.get("run_id") or context.get("pipeline_run_id")

        # Initialize final_thoughts as a list combining all inputs
        final_thoughts: List[Dict[str, Any]] = []

        # When plan exists: walk through steps and mark which succeeded
        if plan and isinstance(plan, dict):
            steps = plan.get("steps") or []
            plan_intent = plan.get("intent", "")

            # Add a meta-thought about the plan (with metadata for filtering)
            final_thoughts.append({
                "type": "plan_overview",
                "content": f"Executing plan with {len(steps)} steps for intent: {plan_intent}",
                "plan_id": plan.get("plan_id"),
                "step_count": len(steps),
                # TASK 2: Add metadata for candidate filtering
                "source_brain": "thought_synthesis",
                "concept_key": current_concept_key,
                "question_signature": question_signature,
                "run_id": current_run_id,
            })

            # Walk through plan steps and correlate with thought_steps
            for step in steps:
                step_kind = step.get("kind", "")
                step_id = step.get("id", "")
                step_status = step.get("status", "pending")

                # Determine if this step succeeded based on thought_steps
                succeeded = False
                for ts in thought_steps:
                    ts_type = ts.get("type", "")
                    # Match reasoning steps to plan steps
                    if step_kind == "retrieve" and ts_type == "recall":
                        succeeded = True
                        break
                    elif step_kind == "reason" and ts_type in ("inference", "recall"):
                        succeeded = True
                        break
                    elif step_kind in ("compose_answer", "compose_explanation", "compose_comparison", "compose_result"):
                        # These will be handled by language brain
                        succeeded = True
                        break

                final_thoughts.append({
                    "type": "plan_step",
                    "step_id": step_id,
                    "kind": step_kind,
                    "status": "succeeded" if succeeded else "pending",
                    "original_status": step_status,
                    # TASK 2: Add metadata for candidate filtering
                    "source_brain": "thought_synthesis",
                    "concept_key": current_concept_key,
                    "question_signature": question_signature,
                    "run_id": current_run_id,
                })

        # Check for learned synthesis patterns first
        learned_synthesis = None
        if _teacher_helper and _memory and len(thought_steps) >= 2:
            try:
                # Create signature of thought types
                thought_types = [ts.get("type", "unknown") for ts in thought_steps[:5]]
                thought_pattern = "-".join(sorted(set(thought_types)))

                learned_patterns = _memory.retrieve(
                    query=f"synthesis pattern: {thought_pattern}",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", "")
                        if isinstance(content, dict) and "final_thoughts" in content:
                            learned_synthesis = content
                            print(f"[THOUGHT_SYNTHESIS] Using learned synthesis pattern from Teacher")
                            final_thoughts = content.get("final_thoughts", final_thoughts)
                            break
            except Exception:
                pass

        # If no learned pattern, use built-in synthesis logic
        if not learned_synthesis:
            # Incorporate thought_steps from reasoning brain
            for ts in thought_steps:
                # Normalize the thought step (TASK 2: with metadata for filtering)
                normalized = {
                    "type": ts.get("type", "unknown"),
                    "content": ts.get("content", ""),
                    "confidence": ts.get("confidence", 0.5),
                    # TASK 2: Add metadata for candidate filtering
                    "source_brain": ts.get("source_brain", "reasoning"),
                    "concept_key": ts.get("concept_key") or current_concept_key,
                    "question_signature": ts.get("question_signature") or question_signature,
                    "run_id": ts.get("run_id") or current_run_id,
                }

                # Add type-specific fields
                if ts.get("type") == "recall":
                    normalized["source"] = ts.get("source", "unknown")
                    normalized["memory_type"] = ts.get("memory_type", "")
                elif ts.get("type") == "inference":
                    normalized["justification"] = ts.get("justification", "")
                elif ts.get("type") == "plan_hint":
                    normalized["hint"] = ts.get("content", "")
                elif ts.get("type") == "no_reasoning_path":
                    normalized["reason"] = ts.get("reason", "unknown")

                final_thoughts.append(normalized)

            # If no learned pattern and Teacher available, try to learn
            if _teacher_helper and len(thought_steps) >= 2:
                try:
                    thought_types = [ts.get("type", "unknown") for ts in thought_steps[:5]]
                    print(f"[THOUGHT_SYNTHESIS] No learned pattern, calling Teacher...")
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"How should I synthesize thoughts with types {thought_types}?",
                        context={
                            "thought_steps": thought_steps,
                            "plan": plan,
                            "thought_types": thought_types,
                            "current_final_thoughts": final_thoughts
                        },
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[THOUGHT_SYNTHESIS] Learned from Teacher: {patterns_stored} synthesis patterns stored")
                        # Learned pattern now in memory for future use
                except Exception as e:
                    print(f"[THOUGHT_SYNTHESIS] Teacher call failed: {str(e)[:100]}")

        # Build answer_skeleton based on final_thoughts
        answer_skeleton = _build_answer_skeleton(plan, final_thoughts, memories, context)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "final_thoughts": final_thoughts,
                "answer_skeleton": answer_skeleton,
            }
        }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {"status": "operational"}
        }

    # Unsupported operation
    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation {op} not supported"}
    }


def _build_answer_skeleton(
    plan: Optional[Dict[str, Any]],
    final_thoughts: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build an answer skeleton structure for the language brain to verbalize.

    Args:
        plan: The plan from the planner brain (may be None)
        final_thoughts: The synthesized list of thought steps
        memories: Retrieved memories used in answering
        context: Conversation context

    Returns:
        A dictionary with 'kind', 'slots', and optional metadata
    """
    # Determine answer kind based on plan intent or thought types
    kind = "direct_answer"  # default
    main_point = ""
    supporting_points: List[str] = []
    uncertainties: List[str] = []

    if plan:
        plan_intent = plan.get("intent", "")
        if plan_intent in ("EXPLAIN", "WHY", "HOW"):
            kind = "explanation"
        elif plan_intent == "COMPARE":
            kind = "comparison"
        elif plan_intent in ("PREFERENCE_QUERY", "IDENTITY_QUERY", "RELATIONSHIP_QUERY"):
            kind = "profile_summary"
        elif plan_intent in ("COMMAND", "REQUEST", "ANALYZE"):
            kind = "action_result"

    # Extract main point from recall thoughts
    recall_thoughts = [t for t in final_thoughts if t.get("type") == "recall"]
    if recall_thoughts:
        # Use the highest confidence recall as the main point
        recall_thoughts.sort(key=lambda t: t.get("confidence", 0), reverse=True)
        main_point = recall_thoughts[0].get("content", "")

        # Additional recalls become supporting points
        for rt in recall_thoughts[1:]:
            content = rt.get("content", "")
            if content and content != main_point:
                supporting_points.append(content)

    # Extract uncertainties from inference and no_reasoning_path thoughts
    for t in final_thoughts:
        t_type = t.get("type", "")
        if t_type == "inference":
            conf = t.get("confidence", 0)
            if conf < 0.6:
                uncertainties.append(t.get("content", ""))
        elif t_type == "no_reasoning_path":
            reason = t.get("reason", "")
            content = t.get("content", "")
            uncertainties.append(f"{reason}: {content}" if content else reason)

    # If no main point from recalls, try inference thoughts
    if not main_point:
        inference_thoughts = [t for t in final_thoughts if t.get("type") == "inference"]
        if inference_thoughts:
            inference_thoughts.sort(key=lambda t: t.get("confidence", 0), reverse=True)
            main_point = inference_thoughts[0].get("content", "")

    # Build the skeleton
    skeleton = {
        "kind": kind,
        "slots": {
            "main_point": main_point,
            "supporting_points": supporting_points,
            "uncertainties": uncertainties,
        }
    }

    # Add kind-specific metadata
    if kind == "comparison":
        skeleton["slots"]["subject_a"] = ""
        skeleton["slots"]["subject_b"] = ""
        skeleton["slots"]["differences"] = []
        skeleton["slots"]["similarities"] = []
    elif kind == "explanation":
        skeleton["slots"]["reasoning_chain"] = [t.get("content", "") for t in final_thoughts if t.get("type") in ("recall", "inference")]
    elif kind == "profile_summary":
        skeleton["slots"]["attributes"] = [t.get("content", "") for t in recall_thoughts]

    return skeleton


# Standard service contract: handle is the entry point
service_api = handle
