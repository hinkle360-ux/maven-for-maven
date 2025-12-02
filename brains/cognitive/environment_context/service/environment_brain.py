"""
Environment Context Service
==========================

This module implements a simple environment context brain for Maven.
It provides a single operation to query Maven's current location.  In
reality, Maven is a software system that runs wherever the user's
device executes the code.  To address questions like "Where are you?"
or "Where are we?", this service returns a succinct description of
Maven's digital environment.  If more detailed telemetry becomes
available in the future (e.g. device name, operating system, network
context), that information could be surfaced here under appropriate
privacy constraints.

The ``service_api`` function accepts a message dictionary with an
``op`` field.  Recognised operations are:

``GET_LOCATION``: Returns a payload with a ``location`` field
    describing the digital environment where Maven operates.

For unsupported operations, the service responds with an error
payload.  All responses include the original ``op`` and a unique
``mid`` identifier to facilitate tracing through the pipeline.

This brain is independent of any domain bank and therefore does not
participate in memory retrieval or semantic caching.  It simply
answers environment queries directly.
"""

from __future__ import annotations

from typing import Dict, Any

import os
import random
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning environment awareness and adaptation patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("environment_context")
except Exception as e:
    print(f"[ENVIRONMENT_CONTEXT] Teacher helper not available: {e}")
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
    print(f"[ENVIRONMENT_CONTEXT] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("environment_context")

# Helper to generate a unique message identifier.  Use random value
# to minimise collisions across threads.
def _gen_mid() -> str:
    return f"envctx-{random.randint(100000, 999999)}-{random.randint(1000, 9999)}"


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
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
    """
    Dispatch operations for the environment context brain.

    Parameters
    ----------
    msg : dict
        A message dictionary with keys ``op`` and (optionally)
        ``payload``.  ``op`` specifies the operation name (case
        insensitive).  Supported values are ``GET_LOCATION``.

    Returns
    -------
    dict
        A response containing ``ok``, ``op``, ``mid`` and ``payload`` or
        ``error`` fields.  On success, ``payload`` contains a
        ``location`` string.  On failure, ``error`` includes a code
        and message.
    """
    # Get conversation context for continuation detection
    conv_context = {}
    is_follow_up = False
    payload = (msg or {}).get("payload", {}) or {}
    if _continuation_helpers_available:
        try:
            conv_context = get_conversation_context()
            query = payload.get("query", "")
            is_follow_up = is_continuation(query, payload)
        except Exception:
            pass

    try:
        op = str((msg or {}).get("op", "")).upper()
    except Exception:
        op = ""
    mid = msg.get("mid") or _gen_mid()
    # Only one operation is supported: GET_LOCATION
    if op == "GET_LOCATION":
        # Check for learned environment descriptions first
        learned_location = None
        if _teacher_helper and _memory:
            try:
                learned_patterns = _memory.retrieve(
                    query="environment location description",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", "")
                        if isinstance(content, str) and len(content) > 10:
                            learned_location = content
                            print(f"[ENVIRONMENT_CONTEXT] Using learned location description from Teacher")
                            break
            except Exception:
                pass

        # Use learned description if found, otherwise use default
        if learned_location:
            location = learned_location
        else:
            # Return a fixed string describing the digital environment.
            # The phrasing emphasises that Maven lacks a physical body and
            # runs wherever the host system executes the code.
            location = "I exist in a digital environment on your device."

            # If no learned description and Teacher available, try to learn
            if _teacher_helper:
                try:
                    print(f"[ENVIRONMENT_CONTEXT] No learned location description, calling Teacher...")
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question="How should I describe my digital environment location?",
                        context={
                            "current_location": location,
                            "platform": os.name if hasattr(os, 'name') else "unknown"
                        },
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[ENVIRONMENT_CONTEXT] Learned from Teacher: {patterns_stored} location descriptions stored")
                        # Learned description now in memory for future use
                except Exception as e:
                    print(f"[ENVIRONMENT_CONTEXT] Teacher call failed: {str(e)[:100]}")

        result_payload = {
            "location": location,
            "is_continuation": is_follow_up
        }

        # Track context evolution for continuations
        if is_follow_up:
            result_payload["base_context"] = conv_context.get("last_topic", "")

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="environment_context",
                    action="context_evolution" if is_follow_up else "environment_query",
                    confidence=0.9,
                    context_tags=["environment", "context_tracking"] if is_follow_up else ["environment"]
                )
            except Exception:
                pass

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": result_payload
        }
    # Unsupported operation: return error response
    # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
    try:
        routing_hint = create_routing_hint(
            brain_name="environment",
            action="unknown",
            confidence=0.7,
            context_tags=[
                "unknown",
                "continuation" if continuation_detected else "fresh_query"
            ]
        )
        if isinstance(result, dict):
            result["routing_hint"] = routing_hint
        elif isinstance(payload_result, dict):
            payload_result["routing_hint"] = routing_hint
    except Exception:
        pass  # Routing hint generation is non-critical
    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": f"Unsupported operation: {op}"
        }
    }

# Standard service contract: handle is the entry point
service_api = handle
