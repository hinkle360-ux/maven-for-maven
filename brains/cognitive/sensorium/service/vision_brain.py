"""
vision_brain.py
================

This module defines a placeholder VisionBrain for Maven.  It lives in
the Sensorium stage and is intended as a foundation for future
multi‑modal processing.  The current implementation exposes a minimal
service API with an `ANALYZE_IMAGE` operation that returns an empty
feature set and a zero confidence score.

Developers can extend this stub by integrating lightweight image
pre‑processing and feature extraction (for example, colour histograms or
pretrained embeddings) while adhering to Maven’s offline and pure
stdlib constraints.
"""

from __future__ import annotations
from typing import Dict, Any

# Cognitive Brain Contract: Continuation awareness
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_enabled = True
except Exception as e:
    print(f"[VISION] Continuation helpers not available: {e}")
    _continuation_enabled = False
    # Fallback stubs
    def is_continuation(*args, **kwargs): return False  # type: ignore
    def get_conversation_context(*args, **kwargs): return {}  # type: ignore
    def create_routing_hint(*args, **kwargs): return {}  # type: ignore

# Teacher integration for learning image feature extraction patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    from brains.memory.brain_memory import BrainMemory
    _teacher_helper = TeacherHelper("vision")
    _memory = BrainMemory("vision")
except Exception as e:
    print(f"[VISION] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore
    _memory = None  # type: ignore

def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
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
    from api.utils import error_response  # type: ignore
    from api.utils import success_response  # type: ignore
    """Service API for the vision brain.

    Supported operations:

    * ``ANALYZE_IMAGE`` – Accepts a payload with an ``image`` field (bytes
      or path) and returns a placeholder dictionary containing an empty
      ``features`` list and a ``confidence`` score of 0.0.  A real
      implementation would populate these fields with extracted visual
      descriptors.

    Any unsupported operation returns an ``UNSUPPORTED_OP`` error.

    Args:
        msg: A message dictionary containing ``op``, ``mid`` and
            optional ``payload`` keys.

    Returns:
        A success_response on supported operations, or an error_response
        for unknown operations.
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}
    context = (msg or {}).get("context", {}) or {}

    if op == "ANALYZE_IMAGE":
        # Cognitive Brain Contract: Get conversation context
        conv_context = get_conversation_context() if _continuation_enabled else {}

        # Cognitive Brain Contract: Detect if this is a continuation
        user_query = context.get("user_query", "")
        is_follow_up = is_continuation(user_query, context) if _continuation_enabled else False

        # Check for learned image feature extraction patterns first
        learned_features = None
        if _teacher_helper and _memory:
            try:
                image_data = payload.get("image", "")
                # Create a signature from image metadata if available
                image_signature = str(type(image_data).__name__)[:20]
                learned_patterns = _memory.retrieve(
                    query=f"image feature pattern: {image_signature}",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    # FIX: Access metadata dict - BrainMemory stores metadata nested
                    rec_metadata = pattern_rec.get("metadata", {}) or {}
                    if rec_metadata.get("kind") == "learned_pattern" and rec_metadata.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", {})
                        if isinstance(content, dict) and "features" in content:
                            learned_features = content.get("features")
                            print(f"[VISION] Using learned image feature pattern from Teacher")
                            break
            except Exception:
                pass

        # Use learned features if found, otherwise use stub response
        if learned_features:
            out = {
                "features": learned_features,
                "confidence": 0.8,
                "detail": "Using learned image feature extraction pattern",
            }
        else:
            out = {
                "features": [],
                "confidence": 0.0,
                "detail": "Vision brain stub; implement feature extraction here",
            }

        # Calculate confidence for routing hint
        feature_confidence = out.get("confidence", 0.0)

        # Cognitive Brain Contract: Add routing hint
        routing_hint = create_routing_hint(
            brain_name="vision",
            action="analyze_image_continuation" if is_follow_up else "analyze_image_fresh",
            confidence=feature_confidence,
            context_tags=["visual_processing", "image_analysis", "continuation"] if is_follow_up else ["visual_processing", "image_analysis"]
        ) if _continuation_enabled else {}

        # Add routing hint and continuation flag to result
        if routing_hint:
            out["routing_hint"] = routing_hint
        out["is_continuation"] = is_follow_up

        return success_response(op, mid, out)
    return error_response(op, mid, "UNSUPPORTED_OP", op)

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