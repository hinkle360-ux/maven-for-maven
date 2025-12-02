
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning system history summarization patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("system_history")
except Exception as e:
    print(f"[SYSTEM_HISTORY] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for follow-up and conversation context tracking
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[SYSTEM_HISTORY] Continuation helpers not available: {e}")
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    _continuation_helpers_available = False

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

# Initialize module-level memory for reuse
_memory = BrainMemory("system_history")

def _counts():
    from api.memory import ensure_dirs, count_lines  # type: ignore
    t = ensure_dirs(BRAIN_ROOT)
    return {"stm": count_lines(t["stm"]), "mtm": count_lines(t["mtm"]), "ltm": count_lines(t["ltm"]), "archive": count_lines(t.get("archive", t.get("cold", t.get("cold_storage", ""))))}

def _log_reflections(reflections: List[Dict[str, Any]]):
    for r in reflections or []:
        _memory.store(
            content={"op": "REFLECTION", "content": r.get("content", "")},
            metadata={
                "kind": "reflection",
                "source": r.get("source", "system_internal"),
                "confidence": r.get("confidence", 0.5)
            }
        )

def handle(msg):
    from api.utils import generate_mid, success_response, error_response  # type: ignore
    op = (msg or {}).get("op"," ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    if op == "HEALTH":
        return success_response(op, mid, {"status": "operational", "memory_health": _counts()})

    # ======================================================================
    # TASK 4: HANDLE CONVERSATION HISTORY QUESTIONS
    # ======================================================================
    # Questions like "what did I ask you first today" or "what did we talk
    # about yesterday" MUST be answered from actual conversation logs,
    # NOT from Teacher/LLM guesses.
    # ======================================================================
    if op == "QUERY_HISTORY":
        query = payload.get("query", "").lower()
        history_type = payload.get("history_type", "session")  # session, today, yesterday

        # Get conversation history from memory
        try:
            # Retrieve stored Q&A pairs from this session
            history_results = _memory.retrieve(
                query="qa_pair question",
                limit=100,
                tiers=["stm", "mtm"]  # Only recent memory
            )

            qa_pairs = []
            for rec in history_results:
                metadata = rec.get("metadata", {})
                content = rec.get("content", {})
                if metadata.get("kind") == "qa_pair" or metadata.get("kind") == "run_data":
                    question = ""
                    if isinstance(content, dict):
                        question = content.get("question", "") or content.get("original_query", "")
                    if question:
                        qa_pairs.append({
                            "question": question,
                            "timestamp": metadata.get("timestamp", 0)
                        })

            # Sort by timestamp (oldest first)
            qa_pairs.sort(key=lambda x: x.get("timestamp", 0))

            # Handle different history types
            if history_type == "first_today" or "first" in query:
                if qa_pairs:
                    first_q = qa_pairs[0].get("question", "")
                    return success_response(op, mid, {
                        "answer": f"Your first question was: '{first_q}'",
                        "found": True,
                        "question": first_q,
                        "source": "session_history"
                    })
                else:
                    return success_response(op, mid, {
                        "answer": "I only see the current run and don't have a record of earlier questions in this session.",
                        "found": False,
                        "source": "session_history"
                    })

            elif history_type == "yesterday" or "yesterday" in query:
                # Maven typically doesn't persist across sessions without explicit logging
                return success_response(op, mid, {
                    "answer": "I have no record of yesterday's conversation. My memory resets between sessions unless explicitly saved.",
                    "found": False,
                    "source": "session_history",
                    "note": "Cross-session history requires persistent logging which is not currently enabled."
                })

            elif history_type == "recent" or "recent" in query:
                if qa_pairs:
                    recent = qa_pairs[-5:]  # Last 5 questions
                    questions = [q.get("question", "") for q in recent]
                    return success_response(op, mid, {
                        "answer": f"Recent questions: {', '.join(questions[:3])}",
                        "found": True,
                        "questions": questions,
                        "source": "session_history"
                    })
                else:
                    return success_response(op, mid, {
                        "answer": "I don't have any recorded questions in this session yet.",
                        "found": False,
                        "source": "session_history"
                    })

            else:
                # Generic history query
                return success_response(op, mid, {
                    "answer": f"I have {len(qa_pairs)} recorded Q&A pairs in this session.",
                    "count": len(qa_pairs),
                    "source": "session_history"
                })

        except Exception as e:
            return error_response(op, mid, "HISTORY_ERROR", str(e))

    if op == "LOG_REFLECTIONS":
        _log_reflections(payload.get("reflections") or [])
        return success_response(op, mid, {"logged": len(payload.get("reflections") or [])})

    if op == "SET_TOPIC":
        """Store the current conversation topic for follow-up context."""
        topic = payload.get("topic", "")
        if topic:
            _memory.store(
                content={"topic": topic, "timestamp": payload.get("timestamp", "")},
                metadata={
                    "kind": "current_topic",
                    "source": "system_history",
                    "confidence": 0.95
                }
            )
            return success_response(op, mid, {"topic_set": topic})
        return error_response(op, mid, "MISSING_TOPIC", "No topic provided")

    if op == "GET_LAST_TOPIC":
        """Retrieve the most recent conversation topic for follow-up questions."""

        # FOLLOW-UP DETECTION: Check if current query is a continuation
        query = payload.get("query", "")
        context = payload.get("context", {})
        is_cont = False
        continuation_type = "unknown"

        if query and _continuation_helpers_available:
            try:
                is_cont = is_continuation(query, context)
                if is_cont:
                    print("[SYSTEM_HISTORY] Detected follow-up question, retrieving topic context")
            except Exception as e:
                print(f"[SYSTEM_HISTORY] Could not check continuation: {e}")

        try:
            # Retrieve the most recent topic from memory
            results = _memory.retrieve(
                query="current_topic",
                limit=1,
                tiers=["stm", "mtm"]  # Only check recent memory
            )
            last_topic = None
            for result in results:
                if result.get("kind") == "current_topic":
                    content = result.get("content", {})
                    topic = content.get("topic", "")
                    if topic:
                        last_topic = topic

                        # ROUTING LEARNING: Create routing hint for follow-up handling
                        routing_hint = {}
                        if _continuation_helpers_available:
                            try:
                                routing_hint = create_routing_hint(
                                    brain_name="system_history",
                                    action="provide_context",
                                    confidence=0.85 if is_cont else 0.50,
                                    context_tags=["follow_up", "topic_context"] if is_cont else ["topic_lookup"],
                                    metadata={
                                        "last_topic": topic,
                                        "is_continuation": is_cont
                                    }
                                )
                            except Exception as e:
                                print(f"[SYSTEM_HISTORY] Could not create routing hint: {e}")

                        return success_response(op, mid, {
                            "last_topic": topic,
                            "timestamp": content.get("timestamp", ""),
                            "is_continuation": is_cont,
                            "routing_hint": routing_hint
                        })
            return success_response(op, mid, {"last_topic": None, "is_continuation": is_cont})
        except Exception as e:
            return error_response(op, mid, "RETRIEVAL_ERROR", str(e))

    if op == "SUMMARIZE":
        """
        Summarize recent system and self-DMN activity into a compact health dashboard.

        This operation retrieves recent run data and audit records from BrainMemory,
        aggregates governance decisions, memory usage, bank frequencies, popular topics
        and a sample of recent Q/A. The summary is stored back to BrainMemory.
        """

        # Determine window of runs to analyze
        try:
            window = int(payload.get("window", 50))
        except Exception:
            window = 50

        # Check for learned summarization patterns first
        learned_summary = None
        if _teacher_helper and _memory:
            try:
                learned_patterns = _memory.retrieve(
                    query=f"summarization pattern: window={window}",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                        content = pattern_rec.get("content", {})
                        if isinstance(content, dict) and "aggregated" in content:
                            learned_summary = content
                            print(f"[SYSTEM_HISTORY] Using learned summarization pattern from Teacher")
                            break
            except Exception:
                pass

        # Use learned summary if found, otherwise compute from data
        if learned_summary:
            return success_response(op, mid, {"summary": learned_summary})

        # Retrieve run data from BrainMemory
        run_results = _memory.retrieve(limit=window)
        run_data_list = [r.get("content", {}) for r in run_results if r.get("metadata", {}).get("kind") == "run_data"]

        summary = {
            "aggregated": {
                "runs_analyzed": len(run_data_list),
                "decisions": {"ALLOW": 0, "DENY": 0, "QUARANTINE": 0, "RECOMPUTE": 0},
                "bank_usage": {},
                "top_likes": [],
            },
            "samples": []
        }

        # Load personal top likes
        try:
            import importlib
            personal = importlib.import_module("brains.personal.service.personal_brain")
            top_res = personal.service_api({"op": "TOP_LIKES", "payload": {"limit": 5}})
            top_items = (top_res.get("payload") or {}).get("items") or []
            summary["aggregated"]["top_likes"] = [
                {"subject": item.get("subject"), "score_boost": item.get("score_boost")} for item in top_items
            ]
        except Exception:
            summary["aggregated"]["top_likes"] = []

        # Count governance decisions and bank usage, and collect Q/A samples
        for data in run_data_list:
            # decisions from governance
            try:
                decision = str(((data.get("stage_8b_governance") or {}).get("decision") or {}).get("decision", "")).upper()
            except Exception:
                decision = ""
            if decision in {"ALLOW", "DENY", "QUARANTINE"}:
                summary["aggregated"]["decisions"][decision] += 1
            # bank usage
            bank = (data.get("stage_9_storage") or {}).get("bank")
            if bank:
                summary["aggregated"]["bank_usage"][bank] = summary["aggregated"]["bank_usage"].get(bank, 0) + 1
            # sample Q/A: use original query and final answer if available
            oq = data.get("original_query")
            ans = data.get("final_answer") or data.get("stage_10_finalize", {}).get("text")
            if oq and ans:
                summary["samples"].append({"query": oq, "answer": ans})

        # Retrieve audit data from BrainMemory
        audit_results = _memory.retrieve(limit=1000)
        recompute_count = 0
        for result in audit_results:
            if result.get("metadata", {}).get("kind") == "audit":
                obj = result.get("content", {})
                st = str(obj.get("status") or "")
                if st == "recompute" or st == "disputed":
                    recompute_count += 1
        summary["aggregated"]["decisions"]["RECOMPUTE"] = recompute_count

        # Store summary to BrainMemory
        _memory.store(
            content=summary,
            metadata={
                "kind": "health_dashboard",
                "source": "system_history",
                "confidence": 0.9
            }
        )

        return success_response(op, mid, {"summary": summary})

    return error_response(op, mid, "UNSUPPORTED_OP", op)

# Standard service contract: handle is the entry point
service_api = handle
