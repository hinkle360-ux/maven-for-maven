
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from brains.memory.brain_memory import BrainMemory

# Cognitive Brain Contract: Continuation awareness
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_enabled = True
except Exception as e:
    print(f"[SENSORIUM] Continuation helpers not available: {e}")
    _continuation_enabled = False
    # Fallback stubs
    def is_continuation(*args, **kwargs): return False  # type: ignore
    def get_conversation_context(*args, **kwargs): return {}  # type: ignore
    def create_routing_hint(*args, **kwargs): return {}  # type: ignore

# Semantic Normalizer integration for phrase/token canonicalization and intent classification
try:
    from brains.cognitive.sensorium.semantic_normalizer import (
        SemanticNormalizer,
        NormalizationResult,
        classify_intent,
        is_system_capability_query,
        is_self_identity_query,
    )
    _semantic_normalizer = SemanticNormalizer()
    _semantic_normalizer_enabled = True
except Exception as e:
    print(f"[SENSORIUM] Semantic normalizer not available: {e}")
    _semantic_normalizer = None  # type: ignore
    _semantic_normalizer_enabled = False
    # Fallback stubs
    def classify_intent(text): return None  # type: ignore
    def is_system_capability_query(text): return False  # type: ignore
    def is_self_identity_query(text): return False  # type: ignore

# Router-Teacher integration for LLM-assisted normalization
# This handles typos and weird phrasing that local normalization can't fix
try:
    from brains.cognitive.teacher.service.router_normalizer import (
        normalize_with_router_teacher,
        compute_weirdness_score,
        should_call_router_teacher,
        RouterNormalizationResult,
        IntentHints,
    )
    _router_normalizer_enabled = True
except Exception as e:
    print(f"[SENSORIUM] Router normalizer not available: {e}")
    _router_normalizer_enabled = False
    normalize_with_router_teacher = None  # type: ignore
    compute_weirdness_score = None  # type: ignore
    should_call_router_teacher = None  # type: ignore

# Teacher integration for learning sensory integration and processing patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("sensorium")
except Exception as e:
    print(f"[SENSORIUM] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("sensorium")

# In-run guard: track which norm types we've already asked Teacher for in this session
_norm_type_asked_this_run = set()

# Global storage for last normalized input (for introspection)
_last_normalized_input = {
    "raw_text": "",
    "normalized_text": "",
    "norm_type": "",
    "tokens": [],
    "timestamp": 0.0
}

def _read_weights(root: Path):
    from api.utils import CFG  # type: ignore
    # Use BrainMemory instead of direct file access
    try:
        results = _memory.retrieve(query="kind:weights", limit=1)
        if results:
            w = results[0].get("content", {})
        else:
            # Initialize with defaults and store
            w = CFG["weights_defaults"]
            _memory.store(
                content=w,
                metadata={"kind": "weights", "source": "sensorium", "confidence": 0.9}
            )
    except Exception:
        w = CFG["weights_defaults"]
    return w

HERE = Path(__file__).resolve().parent; BRAIN_ROOT = HERE.parent
def _counts():
    """Return a dictionary with the count of records in each memory tier.

    Invokes ``rotate_if_needed`` prior to counting to ensure that older
    records are rotated into deeper tiers before assessing memory health.
    """
    from api.memory import rotate_if_needed, ensure_dirs, count_lines  # type: ignore
    # Rotate memory before computing counts to mitigate overflow
    try:
        rotate_if_needed(BRAIN_ROOT)
    except Exception:
        pass
    t = ensure_dirs(BRAIN_ROOT)
    return {
        "stm": count_lines(t["stm"]),
        "mtm": count_lines(t["mtm"]),
        "ltm": count_lines(t["ltm"]),
        "archive": count_lines(t.get("archive", t.get("cold", t.get("cold_storage", "")))),
    }
def _normalize(text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    global _last_normalized_input
    import time
    from api.memory import compute_success_average, ensure_dirs, append_jsonl, rotate_if_needed  # type: ignore

    context = context or {}

    # ==========================================================================
    # Phase 0: Router-Teacher LLM Normalization (for weird/typo-heavy input)
    # ==========================================================================
    # If input has high "weirdness" (typos, missing spaces, etc.), call the
    # router-teacher LLM to fix typos and classify intent.
    # This replaces hard-coded typo patterns with LLM-based normalization.
    # ==========================================================================
    router_result = None
    llm_corrected_text = None
    llm_intent_hints = None

    if _router_normalizer_enabled and text:
        try:
            weirdness = compute_weirdness_score(text)
            # Lower threshold to 0.2 to catch more typos (like "eles" -> "else")
            if weirdness >= 0.2:
                print(f"[SENSORIUM] Weirdness={weirdness:.2f} - calling router-teacher LLM")

                # Build context for router-teacher
                router_context = {
                    "last_answer_subject": context.get("last_answer_subject"),
                    "last_web_search": context.get("last_web_search"),
                }

                router_result = normalize_with_router_teacher(text, router_context)

                if router_result and router_result.confidence >= 0.7:
                    llm_corrected_text = router_result.corrected_text
                    llm_intent_hints = router_result.intent_hints
                    print(f"[SENSORIUM] LLM corrected: '{text[:30]}...' -> '{llm_corrected_text[:30]}...'")

                    # Store the learned normalization as a pattern for future use
                    if llm_corrected_text != text.lower().strip():
                        try:
                            _memory.store(
                                content={
                                    "raw_text": text,
                                    "corrected_text": llm_corrected_text,
                                    "typo_corrections": router_result.typo_corrections,
                                    "intent_hints": router_result.intent_hints.to_dict() if hasattr(router_result.intent_hints, 'to_dict') else router_result.intent_hints,
                                    "confidence": router_result.confidence,
                                },
                                metadata={
                                    "kind": "learned_normalization",
                                    "pattern_type": "typo_correction",
                                    "source": "router_teacher",
                                    "confidence": router_result.confidence,
                                    # Mark as non-fact for safety
                                    "fact_type": "non_fact",
                                    "purpose": "normalization_learning",
                                    "writable_to_domain": False,
                                }
                            )
                            print(f"[SENSORIUM] Stored learned normalization: '{text[:20]}...' -> '{llm_corrected_text[:20]}...'")
                        except Exception as e:
                            print(f"[SENSORIUM] Failed to store normalization: {str(e)[:50]}")
        except Exception as e:
            print(f"[SENSORIUM] Router-teacher failed: {str(e)[:100]}")

    # Use LLM-corrected text if available, otherwise use original
    text_for_semantic = llm_corrected_text if llm_corrected_text else text

    # ==========================================================================
    # Phase 1: Semantic Normalization (phrase/token canonicalization)
    # ==========================================================================
    semantic_result = None
    if _semantic_normalizer_enabled and _semantic_normalizer and text_for_semantic:
        try:
            semantic_result = _semantic_normalizer.normalize(text_for_semantic)
            # Use semantically normalized text for downstream processing
            text_for_processing = semantic_result.normalized
        except Exception as e:
            print(f"[SENSORIUM] Semantic normalizer failed: {e}")
            text_for_processing = text_for_semantic
    else:
        text_for_processing = text_for_semantic

    # ==========================================================================
    # Phase 2: Weight-based normalization and type detection
    # ==========================================================================
    # Read weights and compute a learned bias based on recent success history.
    w = _read_weights(BRAIN_ROOT)
    parse_bias = float(w.get("parse_priority", 0.5))
    # Compute learned bias from recent successes (range [0,1])
    try:
        learned_bias = compute_success_average(BRAIN_ROOT)
    except Exception:
        learned_bias = 0.0

    # Check for learned normalization patterns first using norm_type classification
    learned_normalization = None
    norm_type = "casual_statement"  # Default classification
    if _teacher_helper and _memory and text_for_processing:
        try:
            # Classify into a small number of normalization pattern types
            s_lower = " ".join((text_for_processing or "").split()).lower()
            norm_type = "casual_statement"  # default

            # Simple classification based on text characteristics
            # Check for follow-up questions first (these need context from previous interaction)
            follow_up_patterns = [
                "tell me more", "more about", "what about", "can you expand",
                "explain further", "more details", "continue", "go on",
                "elaborate", "what else", "anything else", "more on",
                "explain that", "about that", "more info", "keep going"
            ]
            if any(pattern in s_lower for pattern in follow_up_patterns):
                norm_type = "follow_up_question"
            elif s_lower.startswith(("hello", "hi", "hey", "good morning", "good afternoon")):
                norm_type = "greeting"
            elif "?" in text_for_processing:
                norm_type = "question"
            elif any(s_lower.startswith(word) for word in ["please", "can you", "could you", "show me", "tell me", "give me"]):
                norm_type = "command"
            elif len(s_lower) < 5:
                norm_type = "unknown"

            # ============================================================
            # MEMORY-FIRST FIX: Search by metadata, not by content
            # ============================================================
            # Patterns are stored with metadata kind="normalization_pattern"
            # and pattern_type=norm_type. We must retrieve ALL records and
            # filter by metadata, because the content is a dict (not string).
            # ============================================================
            all_patterns = _memory.retrieve(
                query=None,  # Retrieve all, then filter by metadata
                limit=500,
                tiers=["stm", "mtm", "ltm"]
            )

            # Look for patterns with matching metadata (metadata is at top level of record)
            for pattern_rec in all_patterns:
                # FIX: Access metadata dict - BrainMemory stores metadata nested
                pattern_metadata = pattern_rec.get("metadata", {}) or {}
                # Match on: kind="normalization_pattern" AND pattern_type=norm_type
                if (pattern_metadata.get("kind") == "normalization_pattern" and
                    pattern_metadata.get("pattern_type") == norm_type):
                    content = pattern_rec.get("content", "")
                    if isinstance(content, dict) and "type" in content:
                        learned_normalization = content
                        print(f"[SENSORIUM] Found learned normalization pattern for {norm_type}, using memory")
                        break
        except Exception:
            pass

    # Use learned normalization if found, otherwise use built-in heuristics
    if learned_normalization:
        s = learned_normalization.get("normalized", " ".join((text_for_processing or "").split()).lower())
        input_type = learned_normalization.get("type", "text")
        lang = learned_normalization.get("language", "english")
    else:
        # Normalize the input string and detect simple type/language heuristics
        s = " ".join((text_for_processing or "").split()).lower()
        is_ascii = all(ord(ch) < 128 for ch in s)
        has_digit = any(ch.isdigit() for ch in s)
        input_type = "text" if parse_bias >= 0.5 else ("number" if s.isdigit() else ("mix" if has_digit else "text"))
        lang = "english" if is_ascii else "unknown"

        # If no learned pattern and Teacher available, try to learn (only once per norm_type)
        # Use in-run guard to prevent multiple calls for same norm_type in one session
        #
        # TASK 4: Teacher in sensorium is ONLY used for classification patterns.
        # These patterns MUST be marked as:
        # - fact_type="non_fact" (not world knowledge)
        # - purpose="classification_only" (only for input classification)
        # - writable_to_domain=False (never leak to domain banks)
        if _teacher_helper and norm_type and len(text_for_processing) > 10:
            if norm_type not in _norm_type_asked_this_run:
                try:
                    _norm_type_asked_this_run.add(norm_type)
                    print(f"[SENSORIUM] No learned pattern for {norm_type}, calling Teacher once...")
                    # TASK 4: Pass context that explicitly blocks fact storage
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"How should I normalize and classify {norm_type} text?",
                        context={
                            "norm_type": norm_type,
                            "current_type": input_type,
                            "current_lang": lang,
                            # TASK 4: These flags tell Teacher to NOT store facts
                            "classification_only": True,
                            "block_fact_storage": True,
                            "purpose": "input_classification"
                        },
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[SENSORIUM] Learned from Teacher: {patterns_stored} normalization patterns stored")
                        # TASK 4: Store the learned pattern with explicit classification-only metadata
                        # This ensures patterns are NEVER mistaken for world facts
                        try:
                            pattern_content = {
                                "normalized": s,
                                "type": input_type,
                                "language": lang,
                                "pattern_type": norm_type
                            }
                            _memory.store(
                                content=pattern_content,
                                metadata={
                                    "kind": "normalization_pattern",
                                    "pattern_type": norm_type,
                                    "source": "sensorium",
                                    "confidence": 0.9,
                                    # TASK 4: Explicit markers that this is NOT a fact
                                    "fact_type": "non_fact",
                                    "purpose": "classification_only",
                                    "writable_to_domain": False,
                                    "is_world_knowledge": False,
                                    "is_personal_fact": False,
                                }
                            )
                            print(f"[SENSORIUM_TASK4] Stored pattern as classification_only (not a fact)")
                        except Exception as store_err:
                            print(f"[SENSORIUM] Failed to store pattern: {str(store_err)[:100]}")
                except Exception as e:
                    print(f"[SENSORIUM] Teacher call failed: {str(e)[:100]}")

    # Augment weights with the learned bias for traceability
    try:
        w_with_bias = dict(w)
        w_with_bias["learned_bias"] = learned_bias
    except Exception:
        w_with_bias = w
        w_with_bias["learned_bias"] = learned_bias

    out = {
        "normalized": s,
        "type": input_type,
        "language": lang,
        "norm_type": norm_type,  # Include classification for downstream routing
        "confidence": 0.35,
        "weights_used": w_with_bias
    }

    # Include semantic normalization details if available
    tokens = []
    intent_kind = None
    if semantic_result is not None:
        out["semantic"] = {
            "original": semantic_result.original,
            "tokens": semantic_result.tokens,
            "applied_phrase_rules": semantic_result.applied_phrase_rules,
            "applied_token_rules": semantic_result.applied_token_rules,
            "intent_kind": semantic_result.intent_kind,
        }
        tokens = semantic_result.tokens
        intent_kind = semantic_result.intent_kind

    # Include intent_kind at the top level for easy access by downstream routing
    if intent_kind:
        out["intent_kind"] = intent_kind

    # ==========================================================================
    # Phase 3: LLM Intent Hints (from router-teacher)
    # ==========================================================================
    # If the router-teacher was called, include its intent hints in the output.
    # These hints allow the Integrator to hard-route queries without more
    # hard-coded patterns. The LLM's classification replaces typo whack-a-mole.
    # ==========================================================================
    if llm_intent_hints is not None:
        # Convert IntentHints to dict if needed
        hints_dict = llm_intent_hints.to_dict() if hasattr(llm_intent_hints, 'to_dict') else llm_intent_hints
        out["llm_intent_hints"] = hints_dict

        # Override intent_kind based on LLM hints for high-priority intents
        # These MUST route to specific tools, not Teacher
        #
        # ROUTING ORDER (from user spec):
        # 1. is_time_query (clock ONLY) -> time_now:GET_TIME
        # 2. is_date_or_day_query (date/day/today) -> time_now:GET_DATE
        # 3. is_calendar_query (month/year) -> time_now:GET_CALENDAR
        # 4. is_capability_query -> self_model
        # 5. is_self_identity -> self_model
        # 6. is_web_search -> research_manager

        if hints_dict.get("is_time_query", False):
            out["intent_kind"] = "time_query"
            out["time_query_type"] = "time"  # Clock time ONLY
            print(f"[SENSORIUM] LLM detected time_query (clock) -> will route to time_now:GET_TIME")
        elif hints_dict.get("is_date_or_day_query", False) or hints_dict.get("is_date_query", False):
            # is_date_or_day_query is the new preferred flag; is_date_query is legacy
            out["intent_kind"] = "time_query"
            out["time_query_type"] = "date"  # Date/day query
            print(f"[SENSORIUM] LLM detected date_or_day_query -> will route to time_now:GET_DATE")
        elif hints_dict.get("is_calendar_query", False):
            out["intent_kind"] = "time_query"
            out["time_query_type"] = "calendar"  # Month/year/calendar
            print(f"[SENSORIUM] LLM detected calendar_query -> will route to time_now:GET_CALENDAR")
        elif hints_dict.get("is_capability_query", False):
            out["intent_kind"] = "system_capability"
            out["routing_target"] = "self_model"
            out["routing_reason"] = "llm_detected_capability_query"
            print(f"[SENSORIUM] LLM detected capability_query -> routing to self_model")
        elif hints_dict.get("is_self_identity", False):
            out["intent_kind"] = "self_identity"
            out["routing_target"] = "self_model"
            out["routing_reason"] = "llm_detected_self_identity"
            print(f"[SENSORIUM] LLM detected self_identity -> routing to self_model")
        elif hints_dict.get("is_web_search", False):
            out["intent_kind"] = "web_search"
            print(f"[SENSORIUM] LLM detected web_search intent")

    # Include LLM-corrected text if different from original
    if llm_corrected_text and llm_corrected_text != text:
        out["llm_corrected_text"] = llm_corrected_text
        out["original_text"] = text

    # Store last normalized input for introspection
    import time as time_module
    _last_normalized_input = {
        "raw_text": text,
        "normalized_text": s,
        "norm_type": norm_type,
        "tokens": tokens,
        "timestamp": time_module.time()
    }

    # Persist normalization results using BrainMemory tier API
    try:
        # Store normalization operation with full context
        _memory.store(
            content={"op": "NORMALIZE", "input": text, "output": out, "success": None},
            metadata={"kind": "normalization_result", "source": "sensorium", "confidence": 0.35}
        )
        # Store normalization summary
        _memory.store(
            content={"op": "NORMALIZE", "type": input_type, "lang": lang},
            metadata={"kind": "normalization_summary", "source": "sensorium", "confidence": 0.35}
        )

        # Store last normalized message in a retrievable format
        # This allows "show me your last normalized user message" queries to work
        _memory.store(
            content={
                "raw_text": text,
                "normalized_text": s,
                "norm_type": norm_type,
                "tokens": tokens,
                "timestamp": time_module.time()
            },
            metadata={
                "kind": "last_normalized_message",
                "source": "sensorium",
                "confidence": 0.95,
                "retrievable": True
            }
        )
    except Exception:
        pass
    # Rotate records across tiers according to configured thresholds to prevent overflow
    try:
        rotate_if_needed(BRAIN_ROOT)
    except Exception:
        pass
    return out
def handle(msg):
    from api.utils import generate_mid, success_response, error_response  # type: ignore
    op=(msg or {}).get("op"," ").upper(); mid=msg.get("mid") or generate_mid(); payload=msg.get("payload") or {}
    context = (msg or {}).get("context", {}) or {}

    if op=="HEALTH":
        result = {"status":"operational","memory_health": _counts()}
        return success_response(op, mid, result)

    if op=="NORMALIZE":
        # Cognitive Brain Contract: Get conversation context
        conv_context = get_conversation_context() if _continuation_enabled else {}

        # Cognitive Brain Contract: Detect if this is a continuation
        text = str(payload.get("text",""))
        user_query = context.get("user_query", text)
        is_follow_up = is_continuation(user_query, context) if _continuation_enabled else False

        # Build context for router-teacher (includes conversation history for follow-up detection)
        normalize_context = {
            "last_answer_subject": conv_context.get("last_answer_subject"),
            "last_web_search": conv_context.get("last_web_search"),
            "is_follow_up": is_follow_up,
        }

        # Perform normalization (now with router-teacher support)
        norm_result = _normalize(text, normalize_context)

        # Calculate confidence based on normalization type
        norm_type = norm_result.get("norm_type", "casual_statement")
        confidence = 0.9 if norm_type == "follow_up_question" else 0.85

        # Check for system_capability or self_identity intent
        intent_kind = norm_result.get("intent_kind")
        context_tags = ["sensory_processing", "normalization"]

        if is_follow_up:
            context_tags.append("continuation")

        # PHASE 1: Route system_capability and self_identity queries to self_model
        if intent_kind == "system_capability":
            context_tags.append("system_capability")
            norm_result["routing_target"] = "self_model"
            norm_result["routing_reason"] = "system_capability_query"
            print(f"[SENSORIUM] Detected system_capability intent, routing to self_model")
        elif intent_kind == "self_identity":
            context_tags.append("self_identity")
            norm_result["routing_target"] = "self_model"
            norm_result["routing_reason"] = "self_identity_query"
            print(f"[SENSORIUM] Detected self_identity intent, routing to self_model")

        # Cognitive Brain Contract: Add routing hint
        routing_hint = create_routing_hint(
            brain_name="sensorium",
            action=f"normalize_{norm_type}" if not intent_kind else f"route_{intent_kind}",
            confidence=confidence,
            context_tags=context_tags
        ) if _continuation_enabled else {}

        # Add routing hint to result
        if routing_hint:
            norm_result["routing_hint"] = routing_hint
        norm_result["is_continuation"] = is_follow_up

        # Include LLM intent hints in the result for downstream routing
        # The Integrator will use these to hard-route queries without more hard-coded patterns
        llm_hints = norm_result.get("llm_intent_hints")
        if llm_hints:
            # Ensure intent hints are properly exposed at top level for Integrator
            norm_result["llm_intent_hints"] = llm_hints

            # Log LLM-assisted routing for debugging
            llm_corrected = norm_result.get("llm_corrected_text")
            if llm_corrected:
                print(f"[SENSORIUM] LLM-assisted: '{text[:25]}...' -> '{llm_corrected[:25]}...'")

        return success_response(op, mid, norm_result)

    if op == "DEBUG_NORMALIZE":
        # Debug operation for exposing detailed normalization info
        # Useful for "repeat my message" tests and debugging
        text = str(payload.get("text", ""))
        if not text:
            return error_response(op, mid, "MISSING_TEXT", "text parameter required")

        # Store original
        original_text = text

        # Semantic normalization (Phase 1)
        semantic_info = {}
        if _semantic_normalizer_enabled and _semantic_normalizer:
            try:
                semantic_result = _semantic_normalizer.normalize(text)
                semantic_info = {
                    "original": semantic_result.original,
                    "normalized": semantic_result.normalized,
                    "tokens": semantic_result.tokens,
                    "applied_phrase_rules": semantic_result.applied_phrase_rules,
                    "applied_token_rules": semantic_result.applied_token_rules,
                }
            except Exception as e:
                semantic_info = {"error": str(e)}

        # Full normalization (Phase 1 + Phase 2)
        norm_result = _normalize(text)

        debug_info = {
            "original_input": original_text,
            "final_normalized": norm_result.get("normalized", ""),
            "norm_type": norm_result.get("norm_type", "unknown"),
            "input_type": norm_result.get("type", "unknown"),
            "language": norm_result.get("language", "unknown"),
            "semantic_normalizer_enabled": _semantic_normalizer_enabled,
            "semantic_normalization": semantic_info,
            "weights_used": norm_result.get("weights_used", {}),
            "confidence": norm_result.get("confidence", 0.0),
            # For "repeat my message" tests - echo back the original
            "echo_original": original_text,
        }

        return success_response(op, mid, debug_info)

    if op == "GET_NORMALIZED_TEXT":
        # Simple operation that returns normalized text for echoing
        # Used by "repeat my message" type tests
        text = str(payload.get("text", ""))
        if not text:
            return error_response(op, mid, "MISSING_TEXT", "text parameter required")

        norm_result = _normalize(text)
        return success_response(op, mid, {
            "original": text,
            "normalized": norm_result.get("normalized", text),
            "norm_type": norm_result.get("norm_type", "casual_statement"),
        })

    if op == "GET_LAST_NORMALIZED":
        # Return the last normalized input for introspection
        # Used by "repeat my message back normalized" type requests
        if not _last_normalized_input.get("raw_text"):
            return error_response(op, mid, "NO_INPUT", "No input has been normalized yet")

        return success_response(op, mid, {
            "raw_text": _last_normalized_input.get("raw_text", ""),
            "normalized_text": _last_normalized_input.get("normalized_text", ""),
            "norm_type": _last_normalized_input.get("norm_type", ""),
            "tokens": _last_normalized_input.get("tokens", []),
            "timestamp": _last_normalized_input.get("timestamp", 0.0),
        })

    return error_response(op, mid, "UNSUPPORTED_OP", op)

# Standard service contract: handle is the entry point
service_api = handle
