"""
Attention Brain Service
=======================

Service API wrapper for the focus analyzer module.

Supported operations:
- UPDATE_FOCUS: Record a focus win for a brain
- GET_SUMMARY: Get focus statistics summary
- HEALTH: Health check
"""

from __future__ import annotations
from typing import Dict, Any

from brains.cognitive.attention.focus_analyzer import update_focus_stats, get_focus_summary  # type: ignore

# Teacher integration for learning focus summary patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    from brains.memory.brain_memory import BrainMemory
    _teacher_helper = TeacherHelper("attention_service")
    _memory = BrainMemory("attention_service")
except Exception as e:
    print(f"[ATTENTION_SERVICE] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore
    _memory = None  # type: ignore


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attention brain service API.

    Supported operations:
    - UPDATE_FOCUS: Record a focus win (requires: focus, optional: reason)
    - GET_SUMMARY: Get focus statistics summary
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "UPDATE_FOCUS":
        try:
            focus = payload.get("focus", "")
            reason = payload.get("reason")

            if not focus:
                return {
                    "ok": False,
                    "error": {
                        "code": "MISSING_FOCUS",
                        "message": "focus parameter is required"
                    }
                }

            update_focus_stats(focus, reason)
            return {
                "ok": True,
                "payload": {
                    "updated": True,
                    "focus": focus,
                    "reason": reason
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

    if op == "GET_SUMMARY":
        try:
            summary = get_focus_summary()

            # Check for learned summary formatting patterns
            enhanced_summary = None
            if _teacher_helper and _memory and summary:
                try:
                    # Create signature from summary data
                    summary_keys = sorted(summary.keys()) if isinstance(summary, dict) else []
                    summary_signature = "-".join(summary_keys[:3])

                    learned_patterns = _memory.retrieve(
                        query=f"summary format pattern: {summary_signature}",
                        limit=3,
                        tiers=["stm", "mtm", "ltm"]
                    )

                    for pattern_rec in learned_patterns:
                        if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                            content = pattern_rec.get("content", {})
                            if isinstance(content, dict) and "format" in content:
                                # Apply learned formatting if applicable
                                enhanced_summary = content.get("enhanced_summary", summary)
                                print(f"[ATTENTION_SERVICE] Using learned summary format from Teacher")
                                break
                except Exception:
                    pass

            return {
                "ok": True,
                "payload": {
                    "summary": enhanced_summary if enhanced_summary else summary
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "SUMMARY_FAILED",
                    "message": str(e)
                }
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "payload": {
                "status": "operational"
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
