"""
Router-Teacher / Normalizer Module
==================================

Purpose
-------
This module provides an LLM-assisted routing and normalization layer that:
1. Corrects typos and weird phrasing in user messages
2. Classifies intent hints (is_time_query, is_date_query, is_web_search, etc.)
3. Returns structured output that downstream routing can use

CRITICAL DESIGN PRINCIPLE:
The Teacher is NOT allowed to execute tools or answer domain questions here.
It ONLY rewrites/annotates the text. The actual routing decision remains
with the Integrator/Sensorium.

This replaces hard-coded typo patterns with LLM-based normalization while
keeping the LLM strictly in its place as a teacher/tool, not the actor.

Usage
-----
Called from Sensorium when:
- Local normalization confidence is low
- Text has "weirdness" (non-word ratio, unknown tokens, etc.)
- Explicit routing hints are needed

Input:
    raw_text: str - Original user message
    basic_normalized_text: str - Lowercase/stripped version
    last_answer_subject: Optional[str] - For follow-up context
    last_web_search: Optional[dict] - For web search follow-ups

Output (JSON):
    corrected_text: str - Fixed version (often same as input if confident)
    intent_hints: {
        is_time_query: bool
        is_date_query: bool
        is_calendar_query: bool
        is_web_search: bool
        is_capability_query: bool
        is_self_identity: bool
        is_follow_up: bool
    }
    confidence: float (0.0-1.0)
    notes: str - Free text for debugging
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Import LLM service for calling the router-teacher
try:
    from brains.tools.llm_service import llm_service as _llm
except Exception:
    _llm = None  # type: ignore


@dataclass
class IntentHints:
    """
    Intent classification hints from the router-teacher.

    These are high-level flags that the Integrator can use for
    hard-routing decisions (e.g., time_query -> time_now tool).

    IMPORTANT: Routing logic should check flags in this order:
    1. is_time_query (clock time ONLY, not date) -> time_now:GET_TIME
    2. is_date_or_day_query (date/day/today) -> time_now:GET_DATE
    3. is_calendar_query (month/year/calendar) -> time_now:GET_CALENDAR
    4. is_web_search -> research_manager
    5. is_capability_query -> self_model
    6. is_self_identity -> self_model
    7. Otherwise -> normal routing
    """
    is_time_query: bool = False          # Clock time ONLY (e.g., "what time is it")
    is_date_query: bool = False          # Date queries (legacy, use is_date_or_day_query)
    is_date_or_day_query: bool = False   # Date/day/today (e.g., "what is today", "what day is it")
    is_calendar_query: bool = False      # Month/year/calendar queries
    is_web_search: bool = False
    is_capability_query: bool = False
    is_self_identity: bool = False
    is_follow_up: bool = False
    is_greeting: bool = False
    is_command: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntentHints":
        # Handle both old format (is_date_query) and new format (is_date_or_day_query)
        is_date = bool(d.get("is_date_query", False))
        is_date_or_day = bool(d.get("is_date_or_day_query", False))
        # Merge: if either is True, set is_date_or_day_query
        merged_date = is_date or is_date_or_day

        return cls(
            is_time_query=bool(d.get("is_time_query", False)),
            is_date_query=is_date,  # Keep for backwards compat
            is_date_or_day_query=merged_date,  # Preferred flag
            is_calendar_query=bool(d.get("is_calendar_query", False)),
            is_web_search=bool(d.get("is_web_search", False)),
            is_capability_query=bool(d.get("is_capability_query", False)),
            is_self_identity=bool(d.get("is_self_identity", False)),
            is_follow_up=bool(d.get("is_follow_up", False)),
            is_greeting=bool(d.get("is_greeting", False)),
            is_command=bool(d.get("is_command", False)),
        )


@dataclass
class RouterNormalizationResult:
    """
    Result from the router-teacher normalization.

    This is the output contract that Sensorium expects.
    """
    corrected_text: str
    intent_hints: IntentHints
    confidence: float
    notes: str = ""
    typo_corrections: Dict[str, str] = field(default_factory=dict)
    raw_llm_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corrected_text": self.corrected_text,
            "intent_hints": self.intent_hints.to_dict(),
            "confidence": self.confidence,
            "notes": self.notes,
            "typo_corrections": self.typo_corrections,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouterNormalizationResult":
        return cls(
            corrected_text=str(d.get("corrected_text", "")),
            intent_hints=IntentHints.from_dict(d.get("intent_hints", {})),
            confidence=float(d.get("confidence", 0.0)),
            notes=str(d.get("notes", "")),
            typo_corrections=d.get("typo_corrections", {}),
        )


def compute_weirdness_score(text: str) -> float:
    """
    Compute a "weirdness" score for the input text.

    Higher score = more likely to need LLM normalization.

    Factors:
    - Non-word ratio (symbols, numbers mixed with text)
    - Very short messages (ambiguous)
    - Presence of unknown/gibberish tokens
    - Missing spaces between words
    - Multiple repeated characters

    Returns:
        float between 0.0 (normal) and 1.0 (very weird)
    """
    if not text or not text.strip():
        return 1.0  # Empty is weird

    text = text.strip()
    score = 0.0

    # Factor 1: Very short messages are ambiguous
    if len(text) < 5:
        score += 0.3
    elif len(text) < 10:
        score += 0.1

    # Factor 2: High ratio of non-alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    if len(text) > 0:
        alpha_ratio = alpha_count / len(text)
        if alpha_ratio < 0.5:
            score += 0.3
        elif alpha_ratio < 0.7:
            score += 0.1

    # Factor 3: Missing spaces (potential typos like "whatsthedate")
    words = text.split()
    if len(words) == 1 and len(text) > 8:
        # Single "word" that's very long - might be missing spaces
        score += 0.2

    # Factor 4: Repeated characters (e.g., "tiiiiime")
    if re.search(r'(.)\1{3,}', text):
        score += 0.2

    # Factor 5: Numbers mixed with text in unusual ways
    # (but not standard formats like "2024" or "10:30")
    if re.search(r'\d+[a-zA-Z]+\d+', text):
        score += 0.15

    # Factor 6: Common typo indicators
    typo_indicators = [
        r'\bto\s+day\b',  # "to day" instead of "today"
        r'\bwat\b',       # "wat" instead of "what"
        r'\bwats\b',      # "wats" instead of "whats"
        r'\b2day\b',      # "2day" instead of "today"
        r'\bur\b',        # "ur" instead of "your"
        r'\bu\b',         # "u" instead of "you"
    ]
    for pattern in typo_indicators:
        if re.search(pattern, text.lower()):
            score += 0.1

    # Cap at 1.0
    return min(1.0, score)


def build_router_normalization_prompt(
    raw_text: str,
    basic_normalized_text: str,
    last_answer_subject: Optional[str] = None,
    last_web_search: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build the prompt for the router-teacher LLM call.

    The prompt is carefully designed to:
    1. Fix obvious typos
    2. Classify intent
    3. NOT answer the question
    4. Return structured JSON
    """
    context_info = ""
    if last_answer_subject:
        context_info += f"\nLast conversation topic: {last_answer_subject}"
    if last_web_search:
        query = last_web_search.get("query", "")
        if query:
            context_info += f"\nLast web search query: {query}"

    prompt = f"""You are a message router/normalizer for Maven. Your ONLY job is to:
1. Fix obvious typos and normalize the text
2. Classify the intent (what type of query is this?)
3. Return your analysis as JSON

CRITICAL RULES:
- Do NOT answer the question
- Do NOT provide information
- ONLY normalize and classify
- If unsure about a typo, keep the original text

IMPORTANT DISTINCTIONS:
- is_time_query: ONLY for clock/hour questions ("what time is it", "current time")
- is_date_or_day_query: For date/day/today questions ("what day is it", "what is today", "what is the day", "which day", "today")
- is_calendar_query: For month/year/calendar ("what month", "what year", "calendar")

USER MESSAGE (raw): {raw_text}
USER MESSAGE (normalized): {basic_normalized_text}
{context_info}

Analyze this message and respond with ONLY this JSON (no other text):
{{
    "corrected_text": "<the message with typos fixed, or same as input if no typos>",
    "intent_hints": {{
        "is_time_query": <true ONLY for clock/hour questions>,
        "is_date_or_day_query": <true for date/day/today questions>,
        "is_calendar_query": <true for month/year/calendar questions>,
        "is_web_search": <true if this needs web search to answer>,
        "is_capability_query": <true if asking "can you X" about the system>,
        "is_self_identity": <true if asking "who are you" about the system>,
        "is_follow_up": <true if this continues a previous topic>,
        "is_greeting": <true if this is a greeting like hello/hi>,
        "is_command": <true if this is a command/instruction>
    }},
    "confidence": <0.0-1.0 how confident you are>,
    "typo_corrections": {{
        "<original>": "<corrected>"
    }},
    "notes": "<brief explanation of your reasoning>"
}}

EXAMPLES:

Input: "what is to day"
Output:
{{
    "corrected_text": "what is today",
    "intent_hints": {{
        "is_time_query": false,
        "is_date_or_day_query": true,
        "is_calendar_query": false,
        "is_web_search": false,
        "is_capability_query": false,
        "is_self_identity": false,
        "is_follow_up": false,
        "is_greeting": false,
        "is_command": false
    }},
    "confidence": 0.95,
    "typo_corrections": {{"to day": "today"}},
    "notes": "'to day' is a typo for 'today', this is a date/day query"
}}

Input: "wats the time"
Output:
{{
    "corrected_text": "what is the time",
    "intent_hints": {{
        "is_time_query": true,
        "is_date_or_day_query": false,
        "is_calendar_query": false,
        "is_web_search": false,
        "is_capability_query": false,
        "is_self_identity": false,
        "is_follow_up": false,
        "is_greeting": false,
        "is_command": false
    }},
    "confidence": 0.98,
    "typo_corrections": {{"wats": "what is"}},
    "notes": "'wats' is a typo for 'what is', this is a time query (clock)"
}}

Input: "what is the day"
Output:
{{
    "corrected_text": "what is the day",
    "intent_hints": {{
        "is_time_query": false,
        "is_date_or_day_query": true,
        "is_calendar_query": false,
        "is_web_search": false,
        "is_capability_query": false,
        "is_self_identity": false,
        "is_follow_up": false,
        "is_greeting": false,
        "is_command": false
    }},
    "confidence": 0.98,
    "typo_corrections": {{}},
    "notes": "Asking about day/date, not clock time"
}}

Now analyze the user's message and respond with JSON only:"""

    return prompt


def parse_router_response(response_text: str) -> Optional[RouterNormalizationResult]:
    """
    Parse the LLM response into a RouterNormalizationResult.

    Handles various response formats and extracts the JSON.
    """
    if not response_text:
        return None

    try:
        # Try to find JSON in the response
        # First, try direct JSON parse
        try:
            data = json.loads(response_text.strip())
            return RouterNormalizationResult.from_dict(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return RouterNormalizationResult.from_dict(data)

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*"corrected_text"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return RouterNormalizationResult.from_dict(data)

        # Try more permissive JSON extraction (nested objects)
        brace_count = 0
        start_idx = -1
        for i, c in enumerate(response_text):
            if c == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    try:
                        json_str = response_text[start_idx:i+1]
                        data = json.loads(json_str)
                        if "corrected_text" in data:
                            result = RouterNormalizationResult.from_dict(data)
                            result.raw_llm_response = response_text
                            return result
                    except json.JSONDecodeError:
                        continue

        return None

    except Exception as e:
        print(f"[ROUTER_NORMALIZER] Failed to parse response: {str(e)[:100]}")
        return None


def call_router_teacher(
    raw_text: str,
    basic_normalized_text: str,
    last_answer_subject: Optional[str] = None,
    last_web_search: Optional[Dict[str, Any]] = None,
) -> Optional[RouterNormalizationResult]:
    """
    Call the LLM in router-teacher mode.

    This is the main entry point for LLM-assisted normalization.

    Args:
        raw_text: Original user message
        basic_normalized_text: Lowercase/stripped version
        last_answer_subject: Previous conversation topic (for follow-ups)
        last_web_search: Last web search context (for follow-ups)

    Returns:
        RouterNormalizationResult or None if failed
    """
    if not _llm or not _llm.enabled:
        print("[ROUTER_NORMALIZER] LLM service not available")
        return None

    try:
        # Build the prompt
        prompt = build_router_normalization_prompt(
            raw_text=raw_text,
            basic_normalized_text=basic_normalized_text,
            last_answer_subject=last_answer_subject,
            last_web_search=last_web_search,
        )

        # Call LLM with strict parameters
        response = _llm.call(
            prompt=prompt,
            max_tokens=400,      # Keep responses short
            temperature=0.1,    # Low temperature for consistent classification
            context={"mode": "router_normalization"}
        )

        if not response.get("ok"):
            print(f"[ROUTER_NORMALIZER] LLM call failed: {response.get('error', 'unknown')}")
            return None

        response_text = response.get("text", "")

        # Parse the response
        result = parse_router_response(response_text)

        if result:
            result.raw_llm_response = response_text
            print(f"[ROUTER_NORMALIZER] Success: '{raw_text[:30]}...' -> '{result.corrected_text[:30]}...' "
                  f"(confidence={result.confidence:.2f})")
            return result
        else:
            print(f"[ROUTER_NORMALIZER] Failed to parse response: {response_text[:100]}...")
            return None

    except Exception as e:
        print(f"[ROUTER_NORMALIZER] Error: {str(e)[:100]}")
        return None


def should_call_router_teacher(
    text: str,
    weirdness_threshold: float = 0.3,
    confidence_threshold: float = 0.7
) -> bool:
    """
    Decide whether to call the router-teacher LLM.

    Called by Sensorium to determine if LLM normalization is needed.

    Args:
        text: The user message
        weirdness_threshold: Call LLM if weirdness >= this
        confidence_threshold: Call LLM if local classification confidence < this

    Returns:
        True if LLM should be called
    """
    weirdness = compute_weirdness_score(text)

    if weirdness >= weirdness_threshold:
        print(f"[ROUTER_NORMALIZER] Weirdness={weirdness:.2f} >= {weirdness_threshold}, recommending LLM call")
        return True

    return False


# =============================================================================
# Module-level convenience functions
# =============================================================================

def normalize_with_router_teacher(
    raw_text: str,
    context: Optional[Dict[str, Any]] = None
) -> Optional[RouterNormalizationResult]:
    """
    High-level function to normalize text with the router-teacher.

    This is the main entry point for other modules.

    Args:
        raw_text: Original user message
        context: Optional context with last_answer_subject, last_web_search, etc.

    Returns:
        RouterNormalizationResult or None if not needed/failed
    """
    if not raw_text or not raw_text.strip():
        return None

    context = context or {}

    # Basic normalization (lowercase, collapse spaces)
    basic_normalized = " ".join(raw_text.lower().split())

    # Check if we should call the router-teacher
    if not should_call_router_teacher(raw_text):
        return None

    # Call the router-teacher
    return call_router_teacher(
        raw_text=raw_text,
        basic_normalized_text=basic_normalized,
        last_answer_subject=context.get("last_answer_subject"),
        last_web_search=context.get("last_web_search"),
    )


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for router normalizer.

    Supported operations:
    - NORMALIZE: Normalize text and classify intent
    - CHECK_WEIRDNESS: Get weirdness score for text
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    if op == "NORMALIZE":
        raw_text = payload.get("raw_text", payload.get("text", ""))
        context = payload.get("context", {})

        if not raw_text:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_TEXT", "message": "raw_text required"}
            }

        result = normalize_with_router_teacher(raw_text, context)

        if result:
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result.to_dict()
            }
        else:
            # Return basic normalization if LLM not called/failed
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "corrected_text": raw_text,
                    "intent_hints": IntentHints().to_dict(),
                    "confidence": 0.5,
                    "notes": "LLM normalization not called or failed",
                    "typo_corrections": {}
                }
            }

    if op == "CHECK_WEIRDNESS":
        text = payload.get("text", "")
        if not text:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_TEXT", "message": "text required"}
            }

        score = compute_weirdness_score(text)
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "text": text,
                "weirdness_score": score,
                "should_call_llm": should_call_router_teacher(text)
            }
        }

    if op == "HEALTH":
        llm_available = _llm is not None and _llm.enabled if _llm else False
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "router_normalizer",
                "llm_available": llm_available,
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"}
    }


# Export public API
__all__ = [
    "IntentHints",
    "RouterNormalizationResult",
    "compute_weirdness_score",
    "call_router_teacher",
    "should_call_router_teacher",
    "normalize_with_router_teacher",
    "service_api",
]
