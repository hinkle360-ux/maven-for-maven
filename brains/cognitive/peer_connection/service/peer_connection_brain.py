"""
Peer Connection Brain Service
----------------------------

This brain simulates establishing connections to peers for real‑time
communication.  It is minimal by design: the only supported operation
is ``CONNECT``, which accepts a ``peer_id`` in the payload and returns a
confirmation message.  If called with an unsupported operation or an
invalid payload, it returns an error.

This brain can be invoked from the language layer when a command like
"connect to peer 123" is parsed.  It does not perform any real network
operations; instead, it logs the intent and responds with a static
acknowledgement.  Future versions could integrate with an actual
transport layer.
"""

from __future__ import annotations
from typing import Dict, Any
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning peer communication and collaboration protocols
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("peer_connection")
except Exception as e:
    print(f"[PEER_CONNECTION] Teacher helper not available: {e}")
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
    print(f"[PEER_CONNECTION] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("peer_connection")

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
    Entry point for the peer connection brain.

    Args:
        msg: A dictionary containing ``op`` and an optional ``payload``.  The
            ``op`` field determines the action to perform.  Recognized
            operations:

            - ``CONNECT``: Establish a connection to a peer.  The ``payload``
              should include ``peer_id`` as a string or integer.  On
              success, the response payload contains a human‑friendly
              confirmation message.  If the peer_id is missing or invalid,
              returns an error.

    Returns:
        A dictionary with either a ``payload`` describing the result or an
        ``error`` indicating why the operation failed.  The top‑level
        ``ok`` field indicates whether the operation succeeded.
    """
    op = str((msg or {}).get("op", "")).upper()
    payload = (msg or {}).get("payload") or {}

    # Get conversation context for continuation detection
    conv_context = {}
    is_follow_up = False
    if _continuation_helpers_available:
        try:
            conv_context = get_conversation_context()
            query = payload.get("query", "") or payload.get("question", "")
            is_follow_up = is_continuation(query, payload)
        except Exception:
            pass
    # Handle peer delegation of tasks.  When asked to delegate a task to a
    # peer, create a new goal in the personal memory and return a
    # confirmation message.  The payload should include both ``peer_id``
    # and ``task`` (or ``goal``) fields.
    if op == "DELEGATE":
        peer_id = payload.get("peer_id")
        task = payload.get("task") or payload.get("goal")
        # Normalize peer_id to string and ensure it's non‑empty
        try:
            peer_str = str(peer_id).strip()
        except Exception:
            peer_str = ""
        try:
            task_str = str(task).strip()
        except Exception:
            task_str = ""
        if not peer_str or not task_str:
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="peer_connection",
                    action="delegate",
                    confidence=0.7,
                    context_tags=[
                        "delegate",
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
                "error": {
                    "code": "INVALID_DELEGATE",
                    "message": "Both peer_id and task are required for delegation."
                }
            }
        # Check for learned delegation patterns first
        learned_message = None
        if _teacher_helper and _memory:
            try:
                learned_patterns = _memory.retrieve(
                    query=f"delegation pattern: {task_str[:30]}",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", "")
                        if isinstance(content, str) and len(content) > 10:
                            learned_message = content
                            print(f"[PEER_CONNECTION] Using learned delegation pattern from Teacher")
                            break
            except Exception:
                pass

        # Store the delegated task using BrainMemory tier API
        try:
            rec = _memory.store(
                content={"task": task_str, "peer_id": peer_str, "status": "delegated"},
                metadata={"kind": "delegated_task", "source": "peer_connection", "confidence": 0.9}
            )
            goal_id = rec.get("id") if rec else None
        except Exception:
            goal_id = None

        # Use learned message if found, otherwise use default
        if learned_message:
            message = learned_message.format(task=task_str, peer=peer_str) if "{task}" in learned_message else learned_message
        else:
            message = f"Delegated task '{task_str}' to peer {peer_str}."

            # If no learned pattern and Teacher available, try to learn
            if _teacher_helper:
                try:
                    print(f"[PEER_CONNECTION] No learned delegation pattern, calling Teacher...")
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"What message should I use when delegating task '{task_str}' to peer {peer_str}?",
                        context={
                            "task": task_str,
                            "peer_id": peer_str,
                            "current_message": message
                        },
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[PEER_CONNECTION] Learned from Teacher: {patterns_stored} delegation patterns stored")
                        # Learned pattern now in memory for future use
                except Exception as e:
                    print(f"[PEER_CONNECTION] Teacher call failed: {str(e)[:100]}")

        result_payload = {
            "message": message,
            "goal_id": goal_id,
            "peer_id": peer_str,
            "task": task_str,
            "is_continuation": is_follow_up
        }

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="peer_connection",
                    action="peer_dialogue_continuation" if is_follow_up else "peer_delegation",
                    confidence=0.9,
                    context_tags=["peer", "delegation", "multi_turn"] if is_follow_up else ["peer", "delegation"]
                )
            except Exception:
                pass

        return {
            "ok": True,
            "payload": result_payload
        }

    # Handle peer queries. When asked to query a peer, return a stubbed
    # response and optionally record the question in a log.  The payload
    # should include ``peer_id`` and ``question`` fields.
    if op == "ASK":
        peer_id = payload.get("peer_id")
        question = payload.get("question") or payload.get("q")
        # Normalize fields to strings and strip whitespace
        try:
            peer_str = str(peer_id).strip()
        except Exception:
            peer_str = ""
        try:
            qstr = str(question).strip()
        except Exception:
            qstr = ""
        if not peer_str or not qstr:
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="peer_connection",
                    action="ask",
                    confidence=0.7,
                    context_tags=[
                        "ask",
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
                "error": {
                    "code": "INVALID_QUERY",
                    "message": "Both peer_id and question are required for a peer query."
                }
            }
        # Formulate a stubbed response.  A future implementation could
        # delegate the question to an actual agent.  Here we simply
        # acknowledge the query and echo back a generic placeholder answer.
        response_text = (
            f"Peer {peer_str} acknowledges your question about '{qstr}', "
            "but cannot answer right now."
        )
        # Persist the query using BrainMemory tier API for audit
        try:
            _memory.store(
                content={
                    "peer_id": peer_str,
                    "question": qstr,
                    "response": response_text,
                    "timestamp": __import__("time").time(),
                    "is_continuation": is_follow_up
                },
                metadata={"kind": "peer_query", "source": "peer_connection", "confidence": 0.85}
            )
        except Exception:
            pass

        result_payload = {
            "message": response_text,
            "peer_id": peer_str,
            "question": qstr,
            "is_continuation": is_follow_up
        }

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="peer_connection",
                    action="peer_dialogue_continuation" if is_follow_up else "peer_query",
                    confidence=0.85,
                    context_tags=["peer", "query", "dialogue"] if is_follow_up else ["peer", "query"]
                )
            except Exception:
                pass

        return {
            "ok": True,
            "payload": result_payload
        }

    # Only one operation is currently supported
    if op == "CONNECT":
        peer_id = payload.get("peer_id")
        # Normalize peer_id to string and ensure it's non‑empty
        peer_id_str: str
        try:
            peer_id_str = str(peer_id).strip()
        except Exception:
            peer_id_str = ""
        if not peer_id_str:
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="peer_connection",
                    action="connect",
                    confidence=0.7,
                    context_tags=[
                        "connect",
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
                "error": {
                    "code": "INVALID_PEER_ID",
                    "message": "A valid peer_id is required to establish a connection."
                }
            }
        # Simulate connecting to the peer.  In this offline environment no
        # actual network connection is made; we simply return a message.
        result_payload = {
            "message": f"Connected to peer {peer_id_str} for real‑time communication.",
            "peer_id": peer_id_str,
            "is_continuation": is_follow_up
        }

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="peer_connection",
                    action="peer_dialogue_continuation" if is_follow_up else "peer_connect",
                    confidence=1.0,
                    context_tags=["peer", "connection", "dialogue"] if is_follow_up else ["peer", "connection"]
                )
            except Exception:
                pass

        return {
            "ok": True,
            "payload": result_payload
        }
    # Unsupported operation
    # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
    try:
        routing_hint = create_routing_hint(
            brain_name="peer_connection",
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
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": op
        }
    }

# Standard service contract: handle is the entry point
service_api = handle