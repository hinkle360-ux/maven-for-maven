
from __future__ import annotations
import re
from typing import Dict, Any, List
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

# MAVEN MASTER SPEC: Per-brain memory tiers (STM→MTM→LTM→Archive)
try:
    from brains.memory.brain_memory import BrainMemory
    _memory = BrainMemory("pattern_recognition")
except Exception:
    _memory = None  # type: ignore

# Teacher integration for learning new pattern analysis techniques
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("pattern_recognition")
except Exception as e:
    print(f"[PATTERN_RECOGNITION] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for follow-up and conversation context tracking
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint,
        extract_continuation_intent,
        enhance_query_with_context
    )
except Exception as e:
    print(f"[PATTERN_RECOGNITION] Continuation helpers not available: {e}")
    # Provide fallback stubs
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore
    enhance_query_with_context = lambda query, context: query  # type: ignore

def _counts():
    """Return a mapping of record counts per memory tier (MAVEN SPEC compliant)."""
    if _memory:
        return _memory.get_stats()
    # Fallback for legacy compatibility
    try:
        from api.memory import rotate_if_needed, ensure_dirs, count_lines  # type: ignore
        rotate_if_needed(BRAIN_ROOT)
        t = ensure_dirs(BRAIN_ROOT)
        return {
            "stm": count_lines(t["stm"]),
            "mtm": count_lines(t["mtm"]),
            "ltm": count_lines(t["ltm"]),
            "archive": count_lines(t.get("archive", t.get("cold", ""))),
        }
    except Exception:
        return {"stm": 0, "mtm": 0, "ltm": 0, "archive": 0}

def _analyze(text: str) -> Dict[str, Any]:
    """
    Analyze a chunk of text and extract simple lexical and structural features.

    This primitive pattern recognition function was intentionally designed to keep
    the complexity of the "brain" minimal while still surfacing a handful of
    informative attributes about the input.  It now includes additional
    structural markers such as uppercase tokens, email addresses, and URLs in
    addition to the original digit, punctuation and repeating word checks.  A
    future version could incorporate more sophisticated statistical or
    linguistic analyses, but for now we deliberately favour simple heuristics
    that are easy to reason about and implement.

    Enhanced with continuation detection to recognize follow-up patterns and
    maintain topic continuity across conversation turns.

    Args:
        text: Raw user input string to inspect.

    Returns:
        A dictionary containing a `features` map with extracted booleans and
        counts, a `confidence` score for the extraction, conversation context,
        and a timestamp.
    """
    s = (text or "").strip()

    # -----------------------------------------------------------------
    # Continuation detection and context enrichment
    #
    # Detect if this is a follow-up that should be tagged as a continuation
    # pattern rather than a fresh new pattern.
    try:
        conv_context = get_conversation_context()
        is_cont = is_continuation(text)
        continuation_intent = extract_continuation_intent(text) if is_cont else "unknown"
    except Exception:
        conv_context = {}
        is_cont = False
        continuation_intent = "unknown"

    # Tokenize on alphanumerics and apostrophes; this preserves contractions
    words = [w for w in re.findall(r"[A-Za-z0-9']+", s)]
    repeats = sorted({w for w in words if words.count(w) > 1})
    # Core flags
    has_digit = any(c.isdigit() for c in s)
    has_punct = any(c in ".?!,;:" for c in s)
    # New structural markers
    # Uppercase words (heuristic: two or more uppercase letters)
    has_uppercase_word = any(w.isupper() and len(w) > 1 for w in words)
    # Simple email detection
    email_pattern = re.compile(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    has_email = bool(email_pattern.search(s))
    # Simple URL detection (http/https)
    url_pattern = re.compile(r"https?://\S+")
    has_url = bool(url_pattern.search(s))
    # Attempt to detect emojis by scanning for characters outside basic Latin
    # Unicode blocks.  Emojis live in ranges >0x1F600; we use a rough check
    # that any codepoint above 0x1F300 is likely an emoji or symbol.  This
    # avoids importing heavy external libraries.
    has_emoji = any(ord(ch) > 0x1F300 for ch in s)
    # Aggregate counts
    length = len(s)
    unique_words = len(set(words))
    avg_word_len = round(sum(len(w) for w in words) / len(words), 2) if words else 0.0
    shape = {
        "has_digit": has_digit,
        "has_punctuation": has_punct,
        "has_uppercase_word": has_uppercase_word,
        "has_email": has_email,
        "has_url": has_url,
        "has_emoji": has_emoji,
        "length": length,
        "word_count": len(words),
        "unique_words": unique_words,
        "avg_word_len": avg_word_len,
        "repeating_words": repeats[:5],
    }
    # Compute a learned bias based on recent successes to augment the output.
    from api.memory import compute_success_average, ensure_dirs, append_jsonl, rotate_if_needed  # type: ignore
    try:
        learned_bias = compute_success_average(BRAIN_ROOT)
    except Exception:
        learned_bias = 0.0

    # Build output with continuation context
    out = {
        "features": shape,
        "confidence": 0.5,
        "learned_bias": learned_bias,
        "is_continuation": is_cont,
        "continuation_intent": continuation_intent,
        "conversation_context": conv_context,
        "pattern_type": "continuation" if is_cont else "new_pattern"
    }
    # Log to memory with a placeholder success field for later marking.
    tiers = ensure_dirs(BRAIN_ROOT)
    try:
        append_jsonl(tiers["stm"], {"op": "ANALYZE", "input": s, "output": out, "success": None})
        append_jsonl(tiers["mtm"], {"op": "ANALYZE", "word_count": len(words)})
    except Exception:
        pass
    # Rotate records if memory exceeds configured thresholds
    try:
        rotate_if_needed(BRAIN_ROOT)
    except Exception:
        pass
    return out

def extract_patterns(records: List[Dict]) -> List[Dict]:
    """
    Extract deterministic patterns from a list of memory records.

    This is not clustering, not statistics, not ML.
    This is deterministic rule-based pattern induction.

    Extracts:
    - repeated user preferences
    - consistent relational structures
    - topic co-occurrence
    - repeated types of queries (intent frequencies)
    - domain-specific repeated explanations

    Args:
        records: List of memory records with content, tags, intent, etc.

    Returns:
        List of structured pattern records (not free text).
    """
    if not records or not isinstance(records, list):
        return []

    patterns = []

    # Track preferences (same pattern repeated)
    preference_map: Dict[str, List[Dict]] = {}
    intent_counter: Dict[str, int] = {}
    topic_map: Dict[str, int] = {}
    relation_types: Dict[str, int] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue

        content = str(rec.get("content", "")).lower().strip()
        tags = rec.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags_set = {str(t).lower() for t in tags}
        intent = str(rec.get("intent", "")).upper()
        verdict = str(rec.get("verdict", "")).upper()

        # Pattern 1: Preference clustering
        if verdict == "PREFERENCE" or "preference" in tags_set:
            # Extract preference subject
            if "like" in content:
                parts = content.split("like")
                if len(parts) > 1:
                    subject = parts[1].strip().split()[0] if parts[1].strip().split() else ""
                    if subject:
                        if subject not in preference_map:
                            preference_map[subject] = []
                        preference_map[subject].append(rec)

        # Pattern 2: Intent frequency tracking
        if intent:
            intent_counter[intent] = intent_counter.get(intent, 0) + 1

        # Pattern 3: Topic co-occurrence
        for tag in tags_set:
            if tag and len(tag) > 2:
                topic_map[tag] = topic_map.get(tag, 0) + 1

        # Pattern 4: Relationship patterns
        if "relationship" in tags_set or verdict == "RELATIONSHIP":
            # Extract relation type (friend, family, etc.)
            for word in ["friend", "family", "colleague", "mentor"]:
                if word in content:
                    relation_types[word] = relation_types.get(word, 0) + 1

    # Generate pattern records for preferences (≥2 occurrences)
    for subject, recs in preference_map.items():
        if len(recs) >= 2:
            patterns.append({
                "pattern_type": "preference_cluster",
                "subject": subject,
                "occurrences": len(recs),
                "consistency": min(1.0, len(recs) / 10.0),  # Deterministic scoring
                "examples": [r.get("content", "") for r in recs[:3]]
            })

    # Generate pattern records for recurring intents (≥3 occurrences)
    for intent, count in intent_counter.items():
        if count >= 3:
            patterns.append({
                "pattern_type": "recurring_intent",
                "intent": intent,
                "frequency": count,
                "consistency": min(1.0, count / 20.0)
            })

    # Generate pattern records for domain focus (≥5 occurrences)
    for topic, count in topic_map.items():
        if count >= 5:
            patterns.append({
                "pattern_type": "domain_focus",
                "topic": topic,
                "frequency": count,
                "consistency": min(1.0, count / 30.0)
            })

    # Generate pattern records for relationship types
    for rel_type, count in relation_types.items():
        if count >= 2:
            patterns.append({
                "pattern_type": "relation_structure",
                "relation_type": rel_type,
                "frequency": count,
                "consistency": min(1.0, count / 5.0)
            })

    # If we have Teacher helper and few patterns found, try to learn new pattern types
    if _teacher_helper and len(patterns) < 3 and len(records) >= 5:
        try:
            # Check if we've learned pattern analysis techniques before
            if _memory:
                learned_techniques = _memory.retrieve(
                    query="pattern analysis technique",
                    limit=5,
                    tiers=["stm", "mtm", "ltm"]
                )

                # Look for learned techniques
                for tech_rec in learned_techniques:
                    if tech_rec.get("kind") == "learned_pattern" and tech_rec.get("confidence", 0) >= 0.7:
                        content = tech_rec.get("content", "")
                        if isinstance(content, str) and "PATTERN" in content:
                            print(f"[PATTERN_RECOGNITION] Using learned technique from Teacher")
                            # Add a pattern indicating we used a learned technique
                            patterns.append({
                                "pattern_type": "learned_technique_applied",
                                "technique": content[:100],
                                "consistency": 0.8,
                                "frequency": 1
                            })
                            return patterns

            # If no learned technique, try calling Teacher
            print(f"[PATTERN_RECOGNITION] Few patterns found ({len(patterns)}), calling Teacher to learn...")

            # Summarize records for Teacher
            record_summary = f"{len(records)} records with various intents, tags, and content"
            if intent_counter:
                top_intents = sorted(intent_counter.items(), key=lambda x: x[1], reverse=True)[:3]
                record_summary += f", top intents: {', '.join(i for i, c in top_intents)}"

            teacher_result = _teacher_helper.maybe_call_teacher(
                question=f"What patterns should I look for in {record_summary}?",
                context={"record_count": len(records), "pattern_count": len(patterns)},
                check_memory_first=True
            )

            if teacher_result and teacher_result.get("answer"):
                answer = teacher_result["answer"]
                patterns_stored = teacher_result.get("patterns_stored", 0)

                print(f"[PATTERN_RECOGNITION] Learned from Teacher: {patterns_stored} techniques stored")
                print(f"[PATTERN_RECOGNITION] Answer: {answer[:100]}...")

                # Add a meta-pattern about the learned technique
                patterns.append({
                    "pattern_type": "teacher_learned_technique",
                    "technique": answer[:200],
                    "consistency": 0.8,
                    "frequency": 1,
                    "learned": True
                })

        except Exception as e:
            print(f"[PATTERN_RECOGNITION] Teacher call failed: {str(e)[:100]}")

    return patterns

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import generate_mid, success_response, error_response  # type: ignore
    op = (msg or {}).get("op"," ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    if op == "HEALTH":
        return success_response(op, mid, {"status": "operational", "memory_health": _counts()})
    if op == "ANALYZE":
        return success_response(op, mid, _analyze(str(payload.get("text",""))))
    if op == "EXTRACT_PATTERNS":
        records = payload.get("records", [])
        patterns = extract_patterns(records)
        return success_response(op, mid, {"patterns": patterns, "count": len(patterns)})
    return error_response(op, mid, "UNSUPPORTED_OP", op)

def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bid for attention based on pattern recognition needs.

    The pattern recognition brain bids higher when it detects:
    - Continuation patterns that need topic continuity
    - Recurring patterns that suggest user preferences
    - Structural patterns (emails, URLs, etc.)

    For continuations, suggests routing to maintain topic coherence.
    """
    try:
        # Get language analysis results
        lang_info = ctx.get("stage_3_language", {}) or {}
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")
        conv_context = lang_info.get("conversation_context", {})

        # Check if we have pattern analysis data
        pattern_info = ctx.get("pattern_recognition", {}) or {}
        features = pattern_info.get("features", {})

        # Higher priority for continuation patterns (topic continuity)
        if is_cont:
            routing_hint = create_routing_hint(
                brain_name="pattern_recognition",
                action="maintain_topic_continuity",
                confidence=0.65,
                context_tags=["continuation", "topic_continuity", continuation_intent],
                metadata={
                    "last_topic": conv_context.get("last_topic", ""),
                    "continuation_type": continuation_intent
                }
            )
            return {
                "brain_name": "pattern_recognition",
                "priority": 0.65,
                "reason": "continuation_pattern_detected",
                "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
            }

        # Medium priority for structural patterns (email, URL, etc.)
        has_structure = (
            features.get("has_email", False) or
            features.get("has_url", False) or
            features.get("has_emoji", False)
        )
        if has_structure:
            routing_hint = create_routing_hint(
                brain_name="pattern_recognition",
                action="analyze_structure",
                confidence=0.50,
                context_tags=["structural_pattern", "formatted_content"],
                metadata={"features": features}
            )
            return {
                "brain_name": "pattern_recognition",
                "priority": 0.50,
                "reason": "structural_pattern_detected",
                "evidence": {"routing_hint": routing_hint, "features": features},
            }

        # Low default priority
        routing_hint = create_routing_hint(
            brain_name="pattern_recognition",
            action="default",
            confidence=0.10,
            context_tags=["default"],
            metadata={}
        )
        return {
            "brain_name": "pattern_recognition",
            "priority": 0.10,
            "reason": "default",
            "evidence": {"routing_hint": routing_hint},
        }
    except Exception:
        # On error, return safe default
        return {
            "brain_name": "pattern_recognition",
            "priority": 0.10,
            "reason": "default",
            "evidence": {},
        }

# Standard service contract: handle is the entry point
service_api = handle
