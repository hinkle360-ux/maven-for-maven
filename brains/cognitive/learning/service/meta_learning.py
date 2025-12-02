"""
Meta Learning
=============

This module is a skeleton for a meta‑learning layer that can adapt
Maven's weights and strategies based on past performance.  Meta
learning involves observing the outcomes of pipeline runs (such as
self‑critiques, corrections and user satisfaction) and updating
internal parameters to improve future behaviour.  Such a mechanism
would allow Maven to learn how to learn.

Currently this file defines only placeholder functions.  It can be
expanded in a future release to track metrics, compute gradients or
apply reinforcement learning strategies, all while remaining
compliant with the offline and governance constraints of the
platform.
"""

from __future__ import annotations

from typing import Any, Dict, List
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning meta-learning strategies
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("learning")
except Exception as e:
    print(f"[LEARNING] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("learning")


def record_run_metrics(ctx: Dict[str, Any]) -> None:
    """Record metrics from a completed pipeline run.

    Args:
        ctx: The context dictionary from a pipeline run.  It may
            contain self‑critique scores, final answers, confidence
            values and other metadata that could be useful for
            meta‑learning.
    """
    if not isinstance(ctx, dict) or not ctx:
        return

    # Store metrics in memory for pattern learning
    if _memory:
        try:
            # Extract key metrics
            metrics = {
                "confidence": ctx.get("confidence", 0.0),
                "success": ctx.get("success", False),
                "critique_score": ctx.get("critique_score", 0.0),
                "corrections": ctx.get("corrections", 0)
            }

            _memory.store(
                content=metrics,
                metadata={"kind": "run_metrics", "source": "learning", "confidence": 0.9}
            )

            # Learn from accumulated metrics periodically
            all_metrics = _memory.retrieve(query="run_metrics", limit=100, tiers=["stm", "mtm", "ltm"])

            if len(all_metrics) >= 20 and len(all_metrics) % 20 == 0:
                # Check for learned meta-learning strategies
                learned_patterns = _memory.retrieve(
                    query="meta-learning strategy",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                has_learned_strategy = any(
                    p.get("kind") == "learned_pattern" and p.get("confidence", 0) >= 0.7
                    for p in learned_patterns
                )

                if not has_learned_strategy and _teacher_helper:
                    try:
                        # Compute average metrics
                        avg_confidence = sum(m.get("content", {}).get("confidence", 0) for m in all_metrics) / len(all_metrics)
                        avg_success = sum(1 for m in all_metrics if m.get("content", {}).get("success", False)) / len(all_metrics)

                        print(f"[LEARNING] Reached {len(all_metrics)} metrics, calling Teacher...")
                        teacher_result = _teacher_helper.maybe_call_teacher(
                            question=f"What meta-learning strategies should I use with avg_confidence={avg_confidence:.2f}, avg_success={avg_success:.2f}?",
                            context={
                                "metrics_count": len(all_metrics),
                                "avg_confidence": avg_confidence,
                                "avg_success": avg_success,
                                "recent_metrics": metrics
                            },
                            check_memory_first=True
                        )

                        if teacher_result and teacher_result.get("answer"):
                            patterns_stored = teacher_result.get("patterns_stored", 0)
                            print(f"[LEARNING] Learned from Teacher: {patterns_stored} meta-learning strategies stored")
                    except Exception as e:
                        print(f"[LEARNING] Teacher call failed: {str(e)[:100]}")
        except Exception:
            pass


def update_parameters() -> None:
    """Update internal parameters based on recorded run metrics.

    This function would perform the actual meta‑learning step, such
    as adjusting weights or biases according to collected metrics.
    It is intentionally left blank pending a more detailed design.
    """
    # Check for learned parameter update strategies
    if _teacher_helper and _memory:
        try:
            learned_patterns = _memory.retrieve(
                query="parameter update strategy",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    print(f"[LEARNING] Using learned parameter update strategy from Teacher")
                    # Future: Apply learned parameter update logic here
                    return

            # If no learned strategy, call Teacher
            all_metrics = _memory.retrieve(query="run_metrics", limit=50, tiers=["stm", "mtm", "ltm"])
            if len(all_metrics) >= 10:
                print(f"[LEARNING] No learned update strategy, calling Teacher...")
                teacher_result = _teacher_helper.maybe_call_teacher(
                    question=f"What parameter update strategy should I use based on {len(all_metrics)} recorded metrics?",
                    context={
                        "metrics_count": len(all_metrics),
                        "recent_metrics": all_metrics[:5]
                    },
                    check_memory_first=True
                )

                if teacher_result and teacher_result.get("answer"):
                    patterns_stored = teacher_result.get("patterns_stored", 0)
                    print(f"[LEARNING] Learned from Teacher: {patterns_stored} update strategies stored")
        except Exception as e:
            print(f"[LEARNING] Teacher call failed: {str(e)[:100]}")


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Meta Learning service API.

    Supported operations:
    - RECORD_METRICS: Record metrics from a pipeline run
    - UPDATE_PARAMETERS: Update learning parameters
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "RECORD_METRICS":
        try:
            ctx = payload.get("context", {})
            record_run_metrics(ctx)
            return {
                "ok": True,
                "payload": {
                    "recorded": True
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "RECORD_FAILED",
                    "message": str(e)
                }
            }

    if op == "UPDATE_PARAMETERS":
        try:
            update_parameters()
            return {
                "ok": True,
                "payload": {
                    "updated": True
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "UPDATE_FAILED",
                    "message": str(e)
                }
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "payload": {
                "status": "operational",
                "note": "Meta learning is a stub awaiting full implementation"
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