"""
Reasoning Trace Brain Service
==============================

Service API wrapper for the trace builder module.

Supported operations:
- NEW_TRACE: Create a new trace builder
- APPEND_STEP: Append a step to the current trace
- APPEND_ENTRY: Append a raw entry to the current trace
- VALIDATE: Validate determinism of the current trace
- PRODUCE: Produce final trace dictionary
- HEALTH: Health check
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from brains.cognitive.reasoning_trace.trace_builder import TraceBuilder, Step  # type: ignore

# Teacher integration for learning trace formatting and validation patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    from brains.memory.brain_memory import BrainMemory
    _teacher_helper = TeacherHelper("reasoning_trace")
    _memory = BrainMemory("reasoning_trace")
except Exception as e:
    print(f"[REASONING_TRACE] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore
    _memory = None  # type: ignore

# Global trace builder instance (could be improved with per-session tracking)
_current_trace: Optional[TraceBuilder] = None


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reasoning trace service API.

    Supported operations:
    - NEW_TRACE: Create a new trace builder
    - APPEND_STEP: Append a step (requires: step_id, step_type, description)
    - APPEND_ENTRY: Append a raw entry dict
    - VALIDATE: Validate determinism
    - PRODUCE: Produce final trace
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    global _current_trace

    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "NEW_TRACE":
        try:
            _current_trace = TraceBuilder()
            return {
                "ok": True,
                "payload": {
                    "created": True
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "CREATE_FAILED",
                    "message": str(e)
                }
            }

    if op == "APPEND_STEP":
        try:
            if _current_trace is None:
                _current_trace = TraceBuilder()

            step_id = payload.get("step_id", 0)
            step_type = payload.get("step_type", "")
            description = payload.get("description", "")
            tags = payload.get("tags", [])
            input_data = payload.get("input")

            if not step_type or not description:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_FIELDS",
                        "message": "step_type and description are required"
                    }
                }

            step = Step(step_id, step_type, description, tags, input_data)

            # If output and brain provided, mark as complete
            if "output" in payload or "brain" in payload:
                output = payload.get("output")
                brain = payload.get("brain", "unknown")
                patterns = payload.get("patterns", [])
                step.mark_complete(output, brain, patterns)

            # If error provided, mark as failed
            if "error" in payload:
                step.mark_failed(payload["error"])

            _current_trace.append_step(step)

            return {
                "ok": True,
                "payload": {
                    "appended": True,
                    "step_id": step_id
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "APPEND_FAILED",
                    "message": str(e)
                }
            }

    if op == "APPEND_ENTRY":
        try:
            if _current_trace is None:
                _current_trace = TraceBuilder()

            entry = payload.get("entry", {})
            if not entry:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_ENTRY",
                        "message": "entry parameter is required"
                    }
                }

            _current_trace.append_entry(entry)

            return {
                "ok": True,
                "payload": {
                    "appended": True
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "APPEND_FAILED",
                    "message": str(e)
                }
            }

    if op == "VALIDATE":
        try:
            if _current_trace is None:
                return {
                    "ok": False,
                    "error": {
                        "code": "NO_TRACE",
                        "message": "No trace available. Create one with NEW_TRACE first."
                    }
                }

            is_deterministic = _current_trace.validate_determinism()

            return {
                "ok": True,
                "payload": {
                    "deterministic": is_deterministic,
                    "validation_errors": _current_trace._validation_errors
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "VALIDATE_FAILED",
                    "message": str(e)
                }
            }

    if op == "PRODUCE":
        try:
            if _current_trace is None:
                return {
                    "ok": False,
                    "error": {
                        "code": "NO_TRACE",
                        "message": "No trace available. Create one with NEW_TRACE first."
                    }
                }

            trace = _current_trace.produce_trace()

            # Check for learned trace formatting patterns
            enhanced_trace = None
            if _teacher_helper and _memory and trace:
                try:
                    # Create signature from trace structure
                    trace_steps = trace.get("steps", []) if isinstance(trace, dict) else []
                    step_count = len(trace_steps)
                    trace_signature = f"steps:{step_count}"

                    learned_patterns = _memory.retrieve(
                        query=f"trace format pattern: {trace_signature}",
                        limit=3,
                        tiers=["stm", "mtm", "ltm"]
                    )

                    for pattern_rec in learned_patterns:
                        if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                            content = pattern_rec.get("content", {})
                            if isinstance(content, dict) and "format_enhancements" in content:
                                # Apply learned enhancements if applicable
                                enhanced_trace = content.get("enhanced_trace", trace)
                                print(f"[REASONING_TRACE] Using learned trace format from Teacher")
                                break
                except Exception:
                    pass

            return {
                "ok": True,
                "payload": {
                    "trace": enhanced_trace if enhanced_trace else trace
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "PRODUCE_FAILED",
                    "message": str(e)
                }
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "payload": {
                "status": "operational",
                "has_trace": _current_trace is not None
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
