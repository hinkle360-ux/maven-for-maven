from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

# BrainMemory integration for tier-based storage
from brains.memory.brain_memory import BrainMemory
from brains.cognitive.reasoning.truth_classifier import TruthClassifier

# Brain roles for domain vs cognitive distinction
from brains.brain_roles import get_domain_brains, is_domain_brain

# Teacher integration for learning reasoning heuristics and patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("reasoning")
except Exception as e:
    print(f"[REASONING] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Learning mode and lesson utilities for strategy-based learning
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    # Fallback enum if import fails
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"

try:
    from brains.learning.lesson_utils import (
        canonical_concept_key,
        create_lesson_record,
        store_lesson,
        retrieve_lessons
    )
except Exception as e:
    print(f"[REASONING] Lesson utils not available: {e}")
    # Fallback implementation for canonical_concept_key
    def canonical_concept_key(question: str) -> str:  # type: ignore
        """Fallback concept key extraction."""
        if not question:
            return ""
        q = question.lower().strip()
        for char in "?!.,;:'\"()[]{}*":
            q = q.replace(char, " ")
        filler = {"what", "is", "are", "the", "a", "an", "does", "do", "can",
                  "tell", "me", "about", "explain", "describe", "please"}
        words = [w for w in q.split() if w not in filler]
        return " ".join(words[-3:] if not words else words).strip() or q.strip()
    create_lesson_record = None  # type: ignore
    store_lesson = None  # type: ignore
    retrieve_lessons = None  # type: ignore

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
    print(f"[REASONING] Continuation helpers not available: {e}")
    # Provide fallback stubs
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore
    enhance_query_with_context = lambda query, context: query  # type: ignore

# Step‑2 integration: import memory librarian service for working memory recall
try:
    from brains.cognitive.memory_librarian.service.memory_librarian import service_api as mem_service_api  # type: ignore
except Exception:
    mem_service_api = None  # type: ignore

# Deferred import of importlib for optional affect modulation.  Importing
# importlib here avoids circular dependencies when the affect brain
# depends on the reasoning brain.
import importlib

# Initialize BrainMemory for reasoning brain
memory = BrainMemory("reasoning")

# Basic common‑sense knowledge.  This mapping associates well‑known entities
# with their correct categories.  It is used by _common_sense_check to
# detect blatantly incorrect questions such as "Is Mars a country?" and
# provide a corrective answer.  Additional entries may be added here
# to expand coverage of obvious facts.
COMMON_SENSE_CATEGORIES: Dict[str, str] = {
    "mars": "planet",
    "venus": "planet",
    "earth": "planet",
    "mercury": "planet",
    "jupiter": "planet",
    "saturn": "planet",
    "uranus": "planet",
    "neptune": "planet",
    "pluto": "dwarf planet",
    "moon": "natural satellite",
    "sun": "star",
    "paris": "city",
    "london": "city",
    "tokyo": "city",
    "new york": "city",
    "rome": "city",
    "berlin": "city"
}

# -----------------------------------------------------------------------------
# Telemetry logging
#
# The reasoning brain records statistics about certain filter events
# (e.g. safety and ethics checks) using BrainMemory.  Each event is stored
# as a separate record, allowing the memory tier system to manage retention.
# Errors in storing telemetry are silently ignored to avoid disrupting the
# reasoning process.
def _update_telemetry(event_type: str) -> None:
    try:
        if not event_type:
            return

        # Classify the telemetry event (these are factual operational events)
        classification = TruthClassifier.classify(
            content=f"telemetry_event:{event_type}",
            confidence=1.0,  # Telemetry events are facts (they definitely occurred)
            evidence=None
        )

        # Store telemetry event if classification allows
        if classification["allow_memory_write"]:
            memory.store(
                content=f"telemetry:{event_type}",
                metadata={
                    "kind": "telemetry",
                    "event_type": event_type,
                    "confidence": classification["confidence"],
                    "truth_type": classification["type"]
                }
            )
    except Exception:
        # Ignore all telemetry errors
        pass


# -----------------------------------------------------------------------------
# Domain fact storage helper
#
# This function stores facts from Teacher into domain brains ONLY.
# Cognitive brains should not hold authoritative fact records.
def _store_fact_to_domain(
    fact_statement: str,
    metadata: Dict[str, Any],
    fact_type: str = "world_fact"
) -> int:
    """
    Store a fact to appropriate domain brain(s).

    This is the ONLY way Teacher facts should be stored.
    Facts MUST NOT be stored in cognitive brains like reasoning.

    Args:
        fact_statement: The fact content
        metadata: Metadata dict (kind, source, confidence, etc.)
        fact_type: Type of fact ("world_fact", "personal_fact", etc.)

    Returns:
        Number of domain brains the fact was stored to
    """
    # --------------------------------------------------------------------
    # MEMORY INGESTION FILTER: Reject bad self-identity facts
    # --------------------------------------------------------------------
    # Before storing ANY fact, check if it contains incorrect self-identity
    # information. Teacher may hallucinate Apache Maven facts or LLM identity,
    # which MUST NOT be stored in memory.
    #
    # This is a defense-in-depth layer: even if Teacher gate fails, this
    # filter prevents contamination of domain banks.
    try:
        fact_lower = str(fact_statement).lower()

        # Define forbidden patterns (Apache Maven and LLM identity claims)
        forbidden_patterns = [
            "jason van zyl",
            "pom (project object model",
            "pom file",
            "maven central",
            "central repository",
            "java-based build tool",
            "apache maven",
            "sonatype",
            "apache software foundation",
            "java build automation",
            "created by google",
            "alphabet",
            "openai",
            "trained on a massive corpus",
            "i am an llm",
            "large language model",
            "chatbot built by",
            "language model developed by",
            "artificial intelligence",
            "machine learning model",
            "neural network trained"
        ]

        # Check if fact contains any forbidden pattern
        for pattern in forbidden_patterns:
            if pattern in fact_lower:
                print(f"[FACT_REJECTED_SELF_IDENTITY] Blocked fact containing '{pattern}'")
                print(f"[FACT_REJECTED_SELF_IDENTITY] Fact text: {fact_statement[:100]}...")
                return 0  # Reject fact, do not store
    except Exception as e:
        print(f"[FACT_FILTER_ERROR] Filter check failed: {str(e)[:100]}")
        # If filter fails, continue (fail open to avoid blocking legitimate facts)
        pass

    stored_count = 0

    # Determine target domain brain(s) based on fact type
    target_banks = []

    if fact_type == "personal_fact":
        # Personal facts go to personal domain bank
        target_banks = ["personal"]
    else:
        # World facts go to factual bank by default
        # In the future, could add smarter routing based on content
        target_banks = ["factual"]

    # Store to each target bank
    for bank_name in target_banks:
        # Verify it's actually a domain brain
        if not is_domain_brain(bank_name):
            print(f"[WARNING] Attempted to store fact to non-domain brain: {bank_name}")
            continue

        try:
            # Create BrainMemory for this domain brain
            domain_memory = BrainMemory(bank_name)

            # Store the fact
            domain_memory.store(
                content=fact_statement,
                metadata=metadata
            )

            stored_count += 1
            print(f"[FACT_STORED] Stored to domain brain '{bank_name}': {fact_statement[:60]}...")
        except Exception as e:
            # Log but don't fail if one bank fails
            print(f"[ERROR] Failed to store fact to {bank_name}: {str(e)[:100]}")
            continue

    if stored_count == 0:
        print(f"[WARNING] Fact not stored to any domain brain: {fact_statement[:60]}...")

    return stored_count


def _common_sense_check(question: str) -> Optional[Dict[str, str]]:
    """
    Perform a simple sanity check on binary questions of the form
    "Is X a Y?".  If X is a known entity with a canonical category and
    Y does not match that category, return a corrective result.  Otherwise
    return None to indicate no correction is needed.

    Args:
        question: The raw question string.

    Returns:
        A dictionary with keys 'entity', 'correct', 'wrong' if a mismatch
        is detected, or None otherwise.
    """
    try:
        import re
        q = (question or "").strip().lower()
        # Normalize whitespace and remove punctuation for matching
        q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
        q_norm = re.sub(r"\s+", " ", q_norm).strip()
        # Match patterns like "is X a Y" or "is X an Y"
        m = re.match(r"^is\s+(.*?)\s+an?\s+(.*?)$", q_norm)
        if m:
            entity = m.group(1).strip()
            category = m.group(2).strip()
            # Remove common articles from entity (e.g. the sun)
            entity = re.sub(r"^(the|a|an)\s+", "", entity).strip()
            # Look up in the common sense mapping; require exact match or
            # simple multi-word match
            if entity in COMMON_SENSE_CATEGORIES:
                correct_cat = COMMON_SENSE_CATEGORIES[entity]
                # If category does not match exactly or as a substring, treat as mismatch
                if category != correct_cat and category not in correct_cat:
                    return {"entity": entity, "correct": correct_cat, "wrong": category}
        return None
    except Exception:
        return None
# Compute the root directory for the reasoning brain.  This is used when
# calculating the rolling success average via compute_success_average.  The
# reasoning root corresponds to the directory containing this file's
# ``service`` folder, i.e. maven/brains/cognitive/reasoning.
THIS_FILE = Path(__file__).resolve()
REASONING_ROOT = THIS_FILE.parent.parent  # .../reasoning/service/ -> .../reasoning


def _is_question_text(text: str) -> bool:
    """
    Return True if the provided text appears to be phrased as a question.
    Currently this checks for a trailing question mark.
    """
    try:
        return str(text or "").strip().endswith("?")
    except Exception:
        return False


def _score_evidence(proposed: Dict[str, Any], evidence: Dict[str, Any]) -> float:
    """
    Compute a basic evidence score for a proposed fact given retrieval results.
    If any retrieved record matches the proposed content exactly or as a substring,
    return a high score (0.8).  Otherwise assign a nominal low score (0.4) if the
    proposed fact has any content, else 0.0.
    """
    try:
        content = str(proposed.get("content", "")).strip().lower()
        for it in (evidence or {}).get("results", []):
            if isinstance(it, dict):
                c = str(it.get("content", "")).strip().lower()
                if c and (content == c or content in c or c in content):
                    return 0.8
    except Exception:
        pass
    return 0.4 if proposed.get("content") else 0.0


def _educated_guess_for_question(query: str) -> str | None:
    """
    Provide a simple heuristic based educated guess for yes/no questions when there
    is no direct evidence available.  This helper inspects the lower‑cased query
    and returns a plausible answer string if a pattern is recognized.  Only a
    few illustrative patterns are currently supported.  If no guess can be made,
    returns None.
    """
    try:
        q_lower = (query or "").strip().lower()
    except Exception:
        q_lower = ""

    # Check for learned guess patterns first
    learned_guess = None
    if _teacher_helper and memory and len(query) > 10:
        try:
            query_preview = q_lower[:50]
            learned_patterns = memory.retrieve(
                query=f"educated guess pattern: {query_preview[:30]}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, str) and len(content) > 10:
                        learned_guess = content
                        print(f"[REASONING] Using learned educated guess pattern from Teacher")
                        break
        except Exception:
            pass

    # Use learned guess if found, otherwise use built-in heuristics
    if learned_guess:
        return learned_guess

    # Example heuristic: Do penguins have fur? → Penguins are birds; birds have feathers
    if "penguin" in q_lower and "fur" in q_lower:
        return "Probably not — penguins are birds and birds have feathers."
    # Additional patterns can be added here as needed
    return None

# -----------------------------------------------------------------------------
# Cross‑Episode QA Memory
#
# To enable long‑term learning across runs, the reasoner consults stored
# question‑answer pairs using BrainMemory.  Each entry contains a question
# string and its corresponding answer.  When evaluating a new question, the
# reasoner will first attempt to find an exact or normalized match.  If found,
# the stored answer is returned as a confident response (bypassing retrieval
# and heuristic guessing).

def _qa_memory_lookup(question: str) -> str | None:
    """
    Search the QA memory for a stored answer to the given question.  The
    lookup normalises the question by lower‑casing and stripping trailing
    punctuation.  If multiple answers exist, the most recent is returned.

    Args:
        question: The user's original question string.

    Returns:
        The stored answer text if found, otherwise None.
    """
    try:
        q = str(question or "").strip().lower().rstrip("?")
        if not q:
            return None

        # FIX: Retrieve all QA memory records without query filter
        # The issue was that memory.retrieve(query=q) searches in the "content" field,
        # but we store the ANSWER as content and the QUESTION in metadata.
        # So searching for the question in the content field would never match.
        results = memory.retrieve(query=None, limit=500)

        # Find matching question (most recent first due to retrieve ordering)
        ans = None
        for rec in results:
            # Only process QA memory records
            if rec.get("kind") != "qa_memory":
                continue

            stored_q = str(rec.get("question", "")).strip().lower().rstrip("?")
            if stored_q == q:
                ans = rec.get("answer")
                # Return first (most recent) match
                break

        return ans
    except Exception:
        return None


def _store_qa_memory(question: str, answer: str, confidence: float) -> None:
    """
    Store a question-answer pair in BrainMemory after truth classification.

    Args:
        question: The question that was asked
        answer: The answer that was provided
        confidence: Confidence level of the answer (0.0-1.0)
    """
    try:
        if not question or not answer:
            return

        # Classify the answer using TruthClassifier
        classification = TruthClassifier.classify(
            content=answer,
            confidence=confidence,
            evidence=None
        )

        # Only store non-RANDOM answers
        if classification["type"] != "RANDOM" and classification["allow_memory_write"]:
            memory.store(
                content=answer,
                metadata={
                    "kind": "qa_memory",
                    "question": question,
                    "answer": answer,
                    "confidence": classification["confidence"],
                    "truth_type": classification["type"]
                }
            )
    except Exception:
        # Silently ignore storage errors
        pass


def _store_topic_stat(topic_key: str) -> None:
    """
    Store a topic occurrence in BrainMemory for tracking topic familiarity.

    Args:
        topic_key: The topic key (normalized first two words of query)
    """
    try:
        if not topic_key:
            return

        # Classify topic stat as factual (it definitely occurred)
        classification = TruthClassifier.classify(
            content=f"topic:{topic_key}",
            confidence=1.0,
            evidence=None
        )

        # Store topic occurrence
        if classification["allow_memory_write"]:
            memory.store(
                content=f"topic_stat:{topic_key}",
                metadata={
                    "kind": "topic_stat",
                    "topic": topic_key,
                    "confidence": classification["confidence"],
                    "truth_type": classification["type"]
                }
            )
    except Exception:
        # Silently ignore storage errors
        pass


def _route_for(conf: float) -> str:
    """
    Determine which memory tier to route a fact into based on confidence.
    High confidence facts go into the factual bank, moderate confidence into
    working theories, and low confidence into STM only.
    """
    if conf >= 0.7:
        return "factual"
    if conf >= 0.4:
        return "working_theories"
    return "stm_only"


def _retrieve_facts_for_query(query: str) -> list:
    """
    Retrieve relevant facts from domain banks for a factual query.

    This is a key part of the memory-first architecture. For factual queries,
    we search domain banks (especially 'factual') for stored facts that can
    answer the question without calling the LLM.

    CONCEPT-KEY FIX: Now matches facts by their 'concept_key' metadata,
    allowing "what are birds" and "are birds" to find the same facts.
    The concept_key normalizes questions to their core concept (e.g., "birds").

    Args:
        query: The user's question

    Returns:
        List of relevant fact strings, ordered by relevance
    """
    facts = []

    # Compute concept_key for this query (primary lookup method)
    concept_key = canonical_concept_key(query)
    # Also keep normalized_query for fallback matching
    normalized_query = query.lower().strip().rstrip("?").strip()

    print(f"[RETRIEVE_FACTS] Looking up facts with concept_key='{concept_key}' for query='{query[:50]}...'")

    try:
        # Extract topic keywords from query for better matching (fallback)
        # e.g., "what are birds" -> "birds"
        topic_words = []
        q_lower = query.lower()
        # Strip common question prefixes
        for prefix in ["what are ", "what is ", "who is ", "who are ", "how do ",
                       "why do ", "where is ", "when is ", "tell me about "]:
            if q_lower.startswith(prefix):
                topic_words.append(q_lower[len(prefix):].strip().rstrip("?").strip())
                break
        if not topic_words:
            # Fall back to using the full query
            topic_words = [q_lower.rstrip("?").strip()]

        # ============================================================
        # CONCEPT-KEY BASED LOOKUP (PRIMARY)
        # ============================================================
        # Check by concept_key first - this is the primary lookup method.
        # This allows "what are birds" and "are birds" to match facts
        # stored with concept_key="birds".
        # ============================================================
        try:
            factual_memory = BrainMemory("factual")
            # Retrieve ALL facts and filter by concept_key metadata
            all_facts = factual_memory.retrieve(query=None, limit=500)
            for rec in all_facts:
                # FIX: Access metadata dict - BrainMemory stores metadata nested
                metadata = rec.get("metadata", {}) or {}

                # Check for concept_key metadata match (PRIMARY)
                stored_concept_key = str(metadata.get("concept_key", "")).strip()
                if stored_concept_key and stored_concept_key == concept_key:
                    content = rec.get("content")
                    confidence = metadata.get("confidence", 0.0)
                    if content and confidence >= 0.5:
                        if isinstance(content, str) and content not in facts:
                            facts.append(content)
                            print(f"[RETRIEVE_FACTS] ✓ Found fact by concept_key='{concept_key}': '{content[:60]}...'")

                # FALLBACK: Check for original_question metadata match
                orig_q = str(metadata.get("original_question", "")).lower().strip().rstrip("?").strip()
                if orig_q and orig_q == normalized_query:
                    content = rec.get("content")
                    confidence = metadata.get("confidence", 0.0)
                    if content and confidence >= 0.5:
                        if isinstance(content, str) and content not in facts:
                            facts.append(content)
                            print(f"[RETRIEVE_FACTS] ✓ Found fact by original_question match: '{content[:60]}...'")
        except Exception as e:
            print(f"[RETRIEVE_FACTS] Error in concept_key search: {e}")

        # If no exact question match, fall back to topic-based search
        if not facts:
            try:
                factual_memory = BrainMemory("factual")
                for topic in topic_words:
                    results = factual_memory.retrieve(query=topic, limit=10)
                    for rec in results:
                        # FIX: Access confidence from metadata
                        rec_metadata = rec.get("metadata", {}) or {}
                        content = rec.get("content")
                        confidence = rec_metadata.get("confidence", 0.0)
                        if content and confidence >= 0.5:
                            if isinstance(content, str) and content not in facts:
                                facts.append(content)
            except Exception:
                pass

        # Also search reasoning brain's own memory for related lessons
        # Use concept_key to find lessons stored for semantically equivalent questions
        try:
            all_lessons = memory.retrieve(query=None, limit=500)
            for rec in all_lessons:
                # FIX: Access metadata dict - BrainMemory stores metadata nested
                lesson_metadata = rec.get("metadata", {}) or {}

                # Check if this is a lesson with a matching concept_key
                kind = lesson_metadata.get("kind", "")
                rec_type = lesson_metadata.get("type", "")
                if kind == "lesson" or rec_type == "lesson":
                    # Check for concept_key match (PRIMARY)
                    stored_concept_key = str(lesson_metadata.get("concept_key", "")).strip()
                    if stored_concept_key and stored_concept_key == concept_key:
                        content = rec.get("content")
                        if isinstance(content, dict):
                            rule = content.get("distilled_rule", "")
                            if rule and rule not in facts:
                                facts.append(rule)
                                print(f"[RETRIEVE_FACTS] ✓ Found lesson by concept_key='{concept_key}': '{rule[:60]}...'")
                        elif isinstance(content, str) and content not in facts:
                            facts.append(content)
                            print(f"[RETRIEVE_FACTS] ✓ Found lesson content by concept_key='{concept_key}'")
        except Exception:
            pass

        # Fallback: content-based lesson search
        if not facts:
            try:
                lesson_results = memory.retrieve(query=query, limit=10)
                for rec in lesson_results:
                    # FIX: Access kind/type from metadata
                    fb_metadata = rec.get("metadata", {}) or {}
                    kind = fb_metadata.get("kind", "")
                    if kind == "lesson" or fb_metadata.get("type") == "lesson":
                        content = rec.get("content")
                        if isinstance(content, dict):
                            rule = content.get("distilled_rule", "")
                            if rule and rule not in facts:
                                facts.append(rule)
                        elif isinstance(content, str) and content not in facts:
                            facts.append(content)
            except Exception:
                pass

        if facts:
            print(f"[APPLY_STRATEGY] Found {len(facts)} relevant memory facts for concept_key='{concept_key}'")
        else:
            print(f"[APPLY_STRATEGY] No relevant facts in memory for '{normalized_query}'")

    except Exception as e:
        print(f"[REASONING] Error retrieving facts: {e}")

    return facts


def _build_answer_from_facts(query: str, facts: list) -> str | None:
    """
    Build a coherent answer from retrieved facts.

    This function takes a list of relevant facts and synthesizes them into
    a coherent answer to the user's question. This happens WITHOUT calling
    the LLM - it's pure memory-based answering.

    Args:
        query: The original question
        facts: List of relevant fact strings

    Returns:
        A synthesized answer string, or None if insufficient facts
    """
    if not facts:
        return None

    try:
        # For simple questions, we can often use the most relevant fact directly
        # or combine a few facts into a coherent response

        # If we have only one fact and it looks like a complete answer, use it
        if len(facts) == 1:
            fact = facts[0].strip()
            # If the fact is substantial enough, use it directly
            if len(fact) > 20:
                return fact

        # If we have multiple facts, combine them
        if len(facts) > 1:
            # Build a combined response
            # TODO: Improve this with better synthesis logic
            combined = " ".join(facts[:3])  # Use top 3 facts
            if len(combined) > 30:
                return combined

        # Single short fact - use it if it exists
        if facts:
            return facts[0].strip()

        return None

    except Exception:
        return None


# =============================================================================
# PHASE 4A & 4B: Learning Mode Integration
# =============================================================================
#
# These functions implement strategy-based learning for the reasoning brain:
# - classify_reasoning_problem: Categorize the type of reasoning task
# - STRATEGY_TABLE: In-memory store of learned strategies
# - load_strategies_from_lessons: Load strategies from stored lessons
# - select_strategy: Choose best strategy for a problem type
# - apply_strategy: Execute a strategy and return result
# - reasoning_llm_lesson: Generate and store a lesson from LLM interaction


# Reasoning problem type constants
PROBLEM_TYPES = {
    "logical_choice": ["which", "better", "compare", "versus", "vs"],
    "causal_explanation": ["why", "cause", "because", "reason", "how come"],
    "error_detection": ["wrong", "mistake", "error", "incorrect", "fix"],
    "comparison": ["difference", "similar", "compare", "contrast"],
    "design_decision": ["should", "best way", "approach", "design", "implement"],
    "factual_query": ["what", "who", "when", "where", "is", "are", "does", "do"],
}


def classify_reasoning_problem(context: Dict[str, Any]) -> str:
    """
    Classify the type of reasoning problem from context.

    Uses explicit rules based on query text, intent, and tags.

    Args:
        context: Pipeline context dict containing query info

    Returns:
        Problem type string (e.g., "logical_choice", "causal_explanation")
    """
    try:
        # Extract query text from various possible locations in context
        query = ""
        if "user_query" in context:
            query = str(context["user_query"])
        elif "payload" in context and isinstance(context["payload"], dict):
            query = str(context["payload"].get("query_text", ""))
            if not query:
                query = str(context["payload"].get("original_query", ""))
        elif "query_text" in context:
            query = str(context["query_text"])

        query_lower = query.lower().strip()

        # Check intent if available
        intent = ""
        if "intent" in context:
            intent = str(context["intent"]).upper()
        elif "payload" in context and isinstance(context["payload"], dict):
            intent = str(context["payload"].get("intent", "")).upper()

        # Map intent to problem type if clear
        intent_mapping = {
            "EXPLAIN": "causal_explanation",
            "WHY": "causal_explanation",
            "HOW": "causal_explanation",
            "COMPARE": "comparison",
            "ANALYSIS": "comparison",
        }
        if intent in intent_mapping:
            return intent_mapping[intent]

        # Check query text for problem type indicators
        for problem_type, keywords in PROBLEM_TYPES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return problem_type

        # Default fallback
        return "generic_reasoning"

    except Exception:
        return "generic_reasoning"


# Strategy table: maps (problem_type, domain) -> strategy dict
# Strategy structure: {"name": str, "steps": list, "confidence": float, "examples": list}
STRATEGY_TABLE: Dict[tuple, Dict[str, Any]] = {}


def extract_domain(context: Dict[str, Any]) -> Optional[str]:
    """
    Extract the domain/topic category from context.

    Checks multiple possible locations in context for domain info.
    Returns None if no domain can be determined (placeholder for future enhancement).

    Args:
        context: Pipeline context dict

    Returns:
        Domain string or None if not determinable
    """
    # Check explicit domain field
    if "domain" in context:
        domain = context.get("domain")
        if domain and isinstance(domain, str):
            return domain

    # Check payload for domain
    payload = context.get("payload")
    if isinstance(payload, dict):
        if "domain" in payload:
            domain = payload.get("domain")
            if domain and isinstance(domain, str):
                return domain
        # Check for category field as fallback
        if "category" in payload:
            category = payload.get("category")
            if category and isinstance(category, str):
                return category

    # Check for topic field
    if "topic" in context:
        topic = context.get("topic")
        if topic and isinstance(topic, str):
            return topic

    # TODO: Implement smarter domain extraction based on query content
    # For now, return None and let select_strategy use domain-agnostic strategies
    return None


def load_strategies_from_lessons(context: Dict[str, Any]) -> None:
    """
    Load strategies from stored lesson records into STRATEGY_TABLE.

    Reads lessons from BrainMemory and converts them to strategy entries.
    Ignores rejected lessons.

    Args:
        context: Pipeline context (unused but included for consistency)
    """
    global STRATEGY_TABLE

    if not retrieve_lessons:
        return

    try:
        # Retrieve lessons for reasoning brain
        lessons = retrieve_lessons(
            brain_name="reasoning",
            brain_memory=memory,
            status_filter=["new", "trusted", "provisional", "integrated"],
            limit=100
        )

        for lesson in lessons:
            try:
                # Extract problem type and domain from input signature
                input_sig = lesson.get("input_signature", {})
                problem_type = input_sig.get("problem_type", "generic_reasoning")
                domain = input_sig.get("domain")  # May be None

                # Build strategy key
                strategy_key = (problem_type, domain)

                # Build strategy structure
                strategy = {
                    "name": f"lesson_{lesson.get('topic', 'unknown')}",
                    "steps": lesson.get("distilled_rule", ""),
                    "confidence": lesson.get("confidence", 0.5),
                    "examples": lesson.get("examples", []),
                    "source_topic": lesson.get("topic", ""),
                }

                # Update strategy table (newer lessons override older)
                # Only override if new lesson has higher confidence
                existing = STRATEGY_TABLE.get(strategy_key)
                if existing is None or strategy["confidence"] > existing.get("confidence", 0):
                    STRATEGY_TABLE[strategy_key] = strategy

            except Exception:
                continue

        if STRATEGY_TABLE:
            print(f"[REASONING] Loaded {len(STRATEGY_TABLE)} strategies from lessons")

    except Exception as e:
        print(f"[REASONING] Failed to load strategies: {e}")


def select_strategy(problem_type: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Select the best strategy for a given problem type and domain.

    Args:
        problem_type: The classified problem type
        domain: Optional domain context

    Returns:
        Strategy dict or None if no matching strategy found
    """
    # Try exact match first
    strategy = STRATEGY_TABLE.get((problem_type, domain))
    if strategy:
        return strategy

    # Try without domain (None) as fallback
    strategy = STRATEGY_TABLE.get((problem_type, None))
    if strategy:
        return strategy

    # Try generic reasoning if specific type not found
    if problem_type != "generic_reasoning":
        strategy = STRATEGY_TABLE.get(("generic_reasoning", None))
        if strategy:
            return strategy

    return None


def apply_strategy(
    strategy: Dict[str, Any],
    context: Dict[str, Any]
) -> tuple:
    """
    Apply a strategy to generate a result using internal reasoning helpers.

    Maps problem types to appropriate internal helpers:
    - factual_query: Use QA memory lookup and common sense check
    - logical_choice / comparison: Use distilled rule from strategy if available
    - causal_explanation: Use distilled rule if it explains the pattern
    - generic_reasoning: Try all available helpers

    Args:
        strategy: Strategy dict from STRATEGY_TABLE
        context: Pipeline context

    Returns:
        Tuple of (result_dict, confidence_score) where result_dict may include
        an "answer" field if a real answer was generated.
    """
    try:
        # Extract strategy components
        strategy_name = strategy.get("name", "unknown")
        steps = strategy.get("steps", "")
        base_confidence = strategy.get("confidence", 0.5)
        examples = strategy.get("examples", [])
        source_topic = strategy.get("source_topic", "")

        # Extract query from context
        query = ""
        if "user_query" in context:
            query = str(context["user_query"])
        elif "payload" in context and isinstance(context["payload"], dict):
            query = str(context["payload"].get("query_text", ""))
            if not query:
                query = str(context["payload"].get("original_query", ""))
        elif "query_text" in context:
            query = str(context["query_text"])

        # For continuations, prefer the enhanced query which includes topic context
        # This ensures "tell me more" becomes "tell me more about physics"
        if context.get("is_continuation") and context.get("enhanced_query"):
            query = str(context["enhanced_query"])
            print(f"[APPLY_STRATEGY] Using enhanced query for continuation: '{query}'")

        # Classify the problem type from the strategy or context
        problem_type = source_topic if source_topic else classify_reasoning_problem(context)

        print(f"[APPLY_STRATEGY] Applying strategy '{strategy_name}' for problem_type={problem_type}")

        # Initialize result
        result = {
            "strategy_used": strategy_name,
            "problem_type": problem_type,
            "reasoning_steps": steps,
            "examples_available": len(examples),
        }
        final_confidence = base_confidence

        # ============================================================
        # MEMORY-FIRST STRATEGY APPLICATION
        # ============================================================
        # For factual queries, we follow the memory-first architecture:
        # 1. Check QA memory for exact answer
        # 2. Retrieve facts from domain banks
        # 3. Build answer from stored facts
        # 4. Only if all memory sources fail, fall through
        # ============================================================
        answer = None

        # 1. Try QA memory lookup first (works for all types)
        if query:
            stored_answer = _qa_memory_lookup(query)
            if stored_answer:
                answer = stored_answer
                final_confidence = max(base_confidence, 0.85)  # High confidence for stored answers
                result["source"] = "qa_memory"
                print(f"[REASONING] ✓ Strategy path: QA memory hit (no LLM needed)")

        # 2. For factual_query: Retrieve facts from domain banks and build answer
        if not answer and query and problem_type in ("factual_query", "generic_reasoning"):
            # This is the key memory-first step for factual queries
            facts = _retrieve_facts_for_query(query)
            if facts:
                fact_answer = _build_answer_from_facts(query, facts)
                if fact_answer:
                    answer = fact_answer
                    final_confidence = max(base_confidence, 0.8)
                    result["source"] = "domain_facts"
                    result["facts_used"] = len(facts)
                    print(f"[REASONING] ✓ Strategy path: Built answer from {len(facts)} stored facts (no LLM needed)")

        # 3. Try common sense check for factual questions
        if not answer and query and problem_type in ("factual_query", "generic_reasoning"):
            cs_result = _common_sense_check(query)
            if cs_result:
                entity = cs_result.get("entity", "")
                correct_cat = cs_result.get("correct", "")
                wrong_cat = cs_result.get("wrong", "")
                answer = f"No, {entity} is a {correct_cat}, not a {wrong_cat}."
                final_confidence = max(base_confidence, 0.85)
                result["source"] = "common_sense"
                print(f"[REASONING] ✓ Strategy path: Common sense check (no LLM needed)")

        # 4. Try educated guess for questions
        if not answer and query and problem_type in ("factual_query", "logical_choice", "generic_reasoning"):
            guess = _educated_guess_for_question(query)
            if guess:
                answer = guess
                final_confidence = max(base_confidence, 0.6)  # Lower confidence for guesses
                result["source"] = "educated_guess"
                print(f"[REASONING] Strategy path: Educated guess (no LLM needed)")

        # 5. Use distilled rule from strategy if it looks like an answer pattern
        if not answer and steps:
            # If the distilled rule looks like it provides an answer template, use it
            # This is a conservative check - we only use the rule if it's substantial
            if len(str(steps)) > 20 and problem_type in ("causal_explanation", "comparison", "logical_choice"):
                # The strategy's distilled rule may contain a pattern or explanation
                # For now, log that we have a rule but don't synthesize an answer from it
                # TODO: Implement rule-based answer synthesis when rules are more structured
                print(f"[REASONING] Strategy has distilled rule but answer synthesis not implemented")
                result["has_rule"] = True
                result["rule_preview"] = str(steps)[:100] if steps else ""

        # Get learning mode for logging
        learning_mode = context.get("learning_mode", "unknown")

        # Set the answer in result if we found one
        if answer:
            result["answer"] = answer
            result["verdict"] = "ANSWERED"
            result["confidence"] = final_confidence
            # Log success with learning mode context
            print(f"[REASONING] ✓ Strategy path successful, answered from memory (mode={learning_mode})")
            print(f"[REASONING]   Source: {result.get('source', 'unknown')}, Confidence: {final_confidence:.2f}")
        else:
            # No answer found, strategy didn't produce usable result
            result["verdict"] = "NO_STRATEGY_ANSWER"
            result["confidence"] = base_confidence
            final_confidence = 0.0  # Signal to caller that strategy didn't help
            print(f"[REASONING] Strategy path: No answer from memory, may need LLM teacher")

        return result, final_confidence

    except Exception as e:
        print(f"[APPLY_STRATEGY] Exception: {e}")
        return {"error": "strategy_execution_failed", "exception": str(e)}, 0.0


def reasoning_llm_lesson(
    context: Dict[str, Any],
    question: str,
    learning_mode: LearningMode
) -> Optional[Dict[str, Any]]:
    """
    Generate a lesson from an LLM interaction for the reasoning brain.

    In OFFLINE mode, returns a minimal lesson without calling the LLM.
    In TRAINING/SHADOW mode, calls TeacherHelper and stores the lesson.

    Args:
        context: Pipeline context dict
        question: The question being reasoned about
        learning_mode: Current learning mode

    Returns:
        Lesson record dict, or None if failed
    """
    if not create_lesson_record or not store_lesson:
        print("[REASONING] Lesson utils not available, skipping lesson generation")
        return None

    # Classify the problem type
    problem_type = classify_reasoning_problem(context)

    # Extract domain if available (placeholder for now)
    # TODO: Implement domain extraction from context
    domain = context.get("domain")

    # Build input signature with concept_key for strategy matching
    concept_key = canonical_concept_key(question)
    input_signature = {
        "problem_type": problem_type,
        "domain": domain,
        "concept_key": concept_key,
        "has_memories": bool(context.get("retrieved_memories")),
    }

    # Handle OFFLINE mode - no LLM call
    if learning_mode == LearningMode.OFFLINE:
        lesson = create_lesson_record(
            brain="reasoning",
            topic=problem_type,
            input_signature=input_signature,
            llm_prompt="",
            llm_response="",
            distilled_rule="",
            examples=[],
            confidence=0.0,
            mode="offline",
            status="new"
        )
        # Don't store offline lessons - they have no content
        return lesson

    # TRAINING or SHADOW mode - call TeacherHelper
    if not _teacher_helper:
        print("[REASONING] TeacherHelper not available for lesson generation")
        return None

    try:
        # Build the prompt for Teacher
        llm_prompt = f"Reasoning question: {question}\nProblem type: {problem_type}"

        # Call TeacherHelper with learning mode
        teacher_result = _teacher_helper.maybe_call_teacher(
            question=question,
            context=context,
            check_memory_first=True,
            learning_mode=learning_mode
        )

        if not teacher_result or teacher_result.get("verdict") == "LLM_DISABLED":
            # TeacherHelper blocked the call (shouldn't happen in TRAINING mode)
            return None

        # Extract response components
        llm_response = teacher_result.get("answer", "")
        verdict = teacher_result.get("verdict", "UNKNOWN")

        # Build distilled rule from the answer
        # For now, we use the answer directly as the rule
        # TODO: Implement proper distillation logic
        distilled_rule = llm_response if llm_response else ""

        # Determine confidence
        if verdict == "LEARNED":
            confidence = 0.8
        elif verdict == "KNOWN":
            confidence = 0.9
        else:
            confidence = 0.5

        # Create the lesson record
        lesson = create_lesson_record(
            brain="reasoning",
            topic=problem_type,
            input_signature=input_signature,
            llm_prompt=llm_prompt,
            llm_response=llm_response or "",
            distilled_rule=distilled_rule,
            examples=[],  # TODO: Extract examples from response
            confidence=confidence,
            mode=str(learning_mode.value) if hasattr(learning_mode, 'value') else str(learning_mode),
            status="new"
        )

        # Store the lesson with original_question for concept_key lookups
        # This allows "what are birds" and "are birds" to match the same lesson
        stored = store_lesson("reasoning", lesson, memory, original_question=question)
        if stored:
            print(f"[REASONING] Stored lesson: {problem_type} (concept_key='{concept_key}', confidence={confidence})")

        # Add answer to lesson for caller to use
        lesson["answer"] = llm_response

        return lesson

    except Exception as e:
        print(f"[REASONING] Failed to generate lesson: {e}")
        import traceback
        traceback.print_exc()
        return None


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for the reasoning brain.  Supports EVALUATE_FACT, GENERATE_THOUGHTS, and HEALTH.

    EVALUATE_FACT accepts a proposed fact and optional evidence.  It decides
    whether to accept the fact as TRUE, classify it as a THEORY, or mark it as
    UNKNOWN.  Questions are handled specially: the reasoner attempts to answer
    them using provided evidence and simple heuristics.  It also incorporates a
    learned bias signal from recent successes to gently adjust confidence.

    GENERATE_THOUGHTS produces structured internal reasoning steps for Step 2
    integration with the planner and thought synthesis systems.
    """
    op = (msg or {}).get("op", "").upper()
    mid = (msg or {}).get("mid")
    payload = (msg or {}).get("payload") or {}

    # Read learning_mode from context (default to TRAINING for LLM learning)
    # This controls whether LLM calls (e.g., Teacher) are allowed
    learning_mode = (msg or {}).get("learning_mode", LearningMode.TRAINING)

    # ------------------------------------------------------------------
    # Step 2 operation: GENERATE_THOUGHTS
    #
    # Produces structured internal reasoning steps based on query, intent,
    # entities, retrieved memories, and context.  Returns a list of thought
    # steps with types: recall, inference, plan_hint, no_reasoning_path.
    # This operation is used by the thought synthesis system to assemble
    # coherent internal thinking traces.
    if op == "GENERATE_THOUGHTS":
        query_text = str(payload.get("query_text", ""))
        intent = str(payload.get("intent", "")).upper()
        entities = payload.get("entities") or []
        retrieved_memories = payload.get("retrieved_memories") or []
        context = payload.get("context") or {}

        # -----------------------------------------------------------------
        # Continuation detection and context enrichment
        #
        # Detect if this is a follow-up query that should build on previous
        # reasoning rather than starting fresh.  Use conversation context
        # to inform reasoning strategy (expansion vs. new analysis).
        try:
            conv_context = get_conversation_context()
            is_cont = is_continuation(query_text, context)
            continuation_intent = extract_continuation_intent(query_text) if is_cont else "unknown"

            # Enrich context with conversation history
            context["conversation_context"] = conv_context
            context["is_continuation"] = is_cont
            context["continuation_intent"] = continuation_intent
            context["last_topic"] = conv_context.get("last_topic", "")

            # If continuation, enhance query with topic context
            if is_cont and conv_context.get("last_topic"):
                enhanced_query = enhance_query_with_context(query_text, conv_context)
                context["enhanced_query"] = enhanced_query
        except Exception:
            # If context retrieval fails, continue without enrichment
            is_cont = False
            continuation_intent = "unknown"
            conv_context = {}

        thought_steps: List[Dict[str, Any]] = []

        # -----------------------------------------------------------------
        # Continuation-aware reasoning
        #
        # When processing a follow-up, indicate that reasoning should expand
        # rather than restart.  This helps maintain coherence across turns.
        if is_cont:
            thought_steps.append({
                "type": "inference",
                "content": f"This is a {continuation_intent} follow-up to previous topic: {conv_context.get('last_topic', 'unknown')}",
                "justification": "Detected continuation pattern in query",
                "confidence": 0.85,
                "reasoning_mode": "expansion",
            })

        # Simple factual questions: produce recall thoughts from memories
        if intent in ("SIMPLE_FACT_QUERY", "QUESTION", "QUERY"):
            # Check if we have high-confidence memories
            for mem in retrieved_memories:
                if not isinstance(mem, dict):
                    continue
                conf = float(mem.get("confidence", 0.0))
                content = str(mem.get("content", ""))
                mem_type = str(mem.get("type", ""))
                if conf >= 0.7 and content:
                    thought_steps.append({
                        "type": "recall",
                        "source": "memory",
                        "content": content,
                        "confidence": conf,
                        "memory_type": mem_type,
                    })

            # If we have conflicting memories (different answers), note the conflict
            if len(thought_steps) > 1:
                contents = [t.get("content", "") for t in thought_steps]
                if len(set(contents)) > 1:
                    thought_steps.append({
                        "type": "inference",
                        "content": "Multiple conflicting memories found",
                        "justification": f"Retrieved {len(contents)} different answers",
                        "confidence": 0.5,
                    })

            # If no memories found, acknowledge uncertainty
            if not thought_steps:
                thought_steps.append({
                    "type": "inference",
                    "content": "No direct memory found for this question",
                    "justification": "No high-confidence matches in memory banks",
                    "confidence": 0.3,
                })
                thought_steps.append({
                    "type": "plan_hint",
                    "content": "May need to search external sources or decline to answer",
                    "confidence": 0.6,
                })

        # Preference/identity/relational queries: convert facts into recall thoughts
        elif intent in ("PREFERENCE_QUERY", "IDENTITY_QUERY", "RELATIONSHIP_QUERY"):
            for mem in retrieved_memories:
                if not isinstance(mem, dict):
                    continue
                content = str(mem.get("content", ""))
                conf = float(mem.get("confidence", 0.0))
                if content:
                    # Classify the recall type based on content
                    recall_type = "preference" if "like" in content.lower() or "prefer" in content.lower() else "fact"
                    thought_steps.append({
                        "type": "recall",
                        "source": "memory",
                        "content": content,
                        "confidence": conf,
                        "recall_type": recall_type,
                    })

            if not thought_steps:
                thought_steps.append({
                    "type": "inference",
                    "content": "No stored preferences or identity facts found",
                    "justification": "Query requires personal information not yet stored",
                    "confidence": 0.4,
                })

        # Open questions / explanations ("why", "how", "compare")
        elif intent in ("EXPLAIN", "WHY", "HOW", "COMPARE", "ANALYSIS"):
            # Break into simpler points or comparisons
            if retrieved_memories:
                thought_steps.append({
                    "type": "recall",
                    "source": "memory",
                    "content": f"Found {len(retrieved_memories)} relevant memories",
                    "confidence": 0.8,
                })
                # Create inference step to synthesize explanation
                thought_steps.append({
                    "type": "inference",
                    "content": "Need to synthesize explanation from multiple facts",
                    "justification": f"Combining {len(retrieved_memories)} memory records",
                    "confidence": 0.7,
                })
            else:
                thought_steps.append({
                    "type": "no_reasoning_path",
                    "reason": "insufficient_knowledge",
                    "content": "Cannot explain without relevant background knowledge",
                })

        # Unsupported or unknown intents
        else:
            if intent:
                thought_steps.append({
                    "type": "no_reasoning_path",
                    "reason": "unsupported_intent",
                    "content": f"Intent '{intent}' not supported for structured reasoning",
                })
            else:
                thought_steps.append({
                    "type": "no_reasoning_path",
                    "reason": "no_intent",
                    "content": "No intent provided; cannot generate reasoning steps",
                })

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "thought_steps": thought_steps,
                "query_text": query_text,
                "intent": intent,
            }
        }

    # ------------------------------------------------------------------
    # Custom operation: EXPLAIN_LAST
    #
    # When the Memory Librarian or another brain requests an explanation
    # of the last answer, this op constructs a simple derivation.  It
    # expects the payload to contain 'last_query' and 'last_response'.
    # For basic arithmetic questions (e.g. "2+3"), it parses the
    # expression and explains the operation.  For other inputs, it
    # returns a generic statement referencing the prior response.  The
    # returned payload uses a bespoke verdict to signal that this is
    # explanatory content rather than a truth judgement.  If any
    # exception occurs, a fallback explanation is returned.
    if op == "EXPLAIN_LAST":
        # Prepare default values
        last_q = str((payload or {}).get("last_query", ""))
        last_r = str((payload or {}).get("last_response", ""))
        explanation: str = ""
        try:
            import re as _re_exp
            # Basic arithmetic pattern: two integers with an operator
            m = _re_exp.match(r"\s*([-+]?\d+)\s*([+\-*/])\s*([-+]?\d+)\s*$", last_q)
            if m:
                a = int(m.group(1))
                op_char = m.group(2)
                b = int(m.group(3))
                result: Optional[float | int]
                verb: str
                if op_char == "+":
                    result = a + b
                    verb = "add"
                elif op_char == "-":
                    result = a - b
                    verb = "subtract"
                elif op_char == "*":
                    result = a * b
                    verb = "multiply"
                else:  # division
                    # Guard against division by zero
                    if b == 0:
                        result = None
                    else:
                        result = a / b
                    verb = "divide"
                if result is not None:
                    # Use integer representation when possible
                    if isinstance(result, float) and result.is_integer():
                        result = int(result)
                    explanation = f"To answer your previous question, I {verb} {a} and {b} to get {result}."
                else:
                    explanation = f"The previous calculation involved division by zero, which is undefined."
            # Fallback: if no arithmetic pattern matched or explanation not set
            if not explanation:
                if last_q and last_r:
                    explanation = f"I responded '{last_r}' to your previous query '{last_q}' based on my reasoning and stored knowledge."
                elif last_r:
                    explanation = f"I answered '{last_r}' in response to your previous question."
                else:
                    explanation = "I don't have enough context to provide an explanation."
        except Exception:
            # On any unexpected error, fall back to a generic explanation
            if last_r:
                explanation = f"I answered '{last_r}' previously based on my reasoning and memory."
            else:
                explanation = "I'm unable to provide an explanation due to missing context."
        # Assemble response.  Use a distinct verdict to differentiate
        # explanatory output from fact evaluation.  Confidence is set
        # optimistically as this operation simply reconstructs prior logic.
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "verdict": "EXPLANATION",
                "mode": "EXPLANATION",
                "confidence": 0.95,
                "routing_order": {"target_bank": None, "action": None},
                "supported_by": [],
                "contradicted_by": [],
                "answer": explanation,
                "weights_used": {"rule": "explain_last_v1"},
            },
        }

    if op == "EVALUATE_FACT":
        # Extract the proposed fact and any original query information
        proposed = (payload or {}).get("proposed_fact") or {}
        # Some callers (Memory Librarian) will include original_query in the proposed fact.  If not
        # present, fall back to the proposed content so that question analysis and
        # semantic lookups still function when original_query isn't provided.  This
        # improves robustness when service_api is called directly without the
        # Memory Librarian, e.g. during tests.
        orig_q = str(proposed.get("original_query", "") or payload.get("original_query", "") or "")
        content = str(proposed.get("content", ""))
        # If original_query is empty but content exists, use content as the question text for
        # downstream heuristics and semantic retrieval.  This allows the reasoner to
        # handle cases where only the fact content is supplied (no original query).
        if not orig_q and content:
            orig_q = str(content)

        # Step‑2: Opportunistic recall from working memory.
        # Before performing heavy reasoning, attempt to retrieve a prior answer stored in working memory.
        try:
            if mem_service_api is not None:
                lookup_key = orig_q if orig_q else content
                # Only attempt recall for non-empty keys
                if lookup_key:
                    wm_resp = mem_service_api({
                        "op": "WM_GET",
                        "payload": {"key": lookup_key}
                    })
                    wm_entries = (wm_resp or {}).get("payload", {}).get("entries", [])
                    # Use the most recent entry if available
                    if wm_entries:
                        # Choose the last entry (most recent) and return as known answer
                        last_entry = wm_entries[-1]
                        answer_val = last_entry.get("value")
                        if answer_val is not None:
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "KNOWN",
                                    "mode": "WM_RETRIEVED",
                                    # Use the entry's confidence or default to 0.7
                                    "confidence": float(last_entry.get("confidence", 0.7)),
                                    "routing_order": {"target_bank": None, "action": None},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": answer_val,
                                    "weights_used": {"rule": "wm_lookup_v1"},
                                },
                            }
        except Exception:
            # Ignore any WM retrieval errors
            pass

        # Determine if this input is a question by intent or punctuation.

        # Affect modulation: compute a valence adjustment for this fact or question.
        # Use the fact content when available, otherwise fall back to the original
        # query string.  The valence is later used to tweak confidence scores
        # slightly (positive valence raises confidence, negative lowers).
        # Extract affect metrics (valence and arousal) for modulation.
        # If the affect brain cannot be loaded or fails, fall back to zeros.
        aff_val: float = 0.0
        aff_arousal: float = 0.0
        try:
            ap_mod = importlib.import_module(
                "brains.cognitive.affect_priority.service.affect_priority_brain"
            )
            aff_text = content if content else orig_q
            aff_res = ap_mod.service_api({"op": "SCORE", "payload": {"text": aff_text}})
            aff_payload = aff_res.get("payload") or {}
            aff_val = float(aff_payload.get("valence", 0.0))
            aff_arousal = float(aff_payload.get("arousal", 0.0))
        except Exception:
            aff_val = 0.0
            aff_arousal = 0.0

        # --- Topic familiarity modulation ----------------------------------
        # Adjust affect valence based on the frequency of this question's
        # topic in the cross‑episode statistics.  Questions that have been
        # asked repeatedly receive a small positive boost to confidence,
        # while brand‑new topics get a slight penalty.  The statistics are
        # stored in BrainMemory.  Errors in loading the stats or computing
        # the key are silently ignored.
        try:
            import re
            # Compute topic key: first two words of the normalized query
            q = orig_q.strip().lower()
            # Normalize: remove punctuation
            q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
            q_norm = re.sub(r"\s+", " ", q_norm).strip()
            parts = q_norm.split()
            topic_key = " ".join(parts[:2]) if parts else ""
            # Load stats from BrainMemory
            fam_boost = 0.0
            if topic_key:
                # Retrieve topic stat records
                results = memory.retrieve(query=f"topic_stat:{topic_key}", limit=1000)

                # Count occurrences of this topic
                count = 0
                for rec in results:
                    if rec.get("kind") == "topic_stat" and rec.get("topic") == topic_key:
                        count += 1

                if count > 0:
                    # Each repetition adds 0.02, capped at +0.06
                    fam_boost = min(0.06, 0.02 * float(count))
                else:
                    # Slight penalty for unseen topics
                    fam_boost = -0.02
                # Adjust affect valence
                aff_val = float(aff_val) + fam_boost

                # Store this topic occurrence for future reference
                _store_topic_stat(topic_key)
        except Exception:
            # On any error, leave affect values unchanged
            pass

        # --- Domain confidence modulation ----------------------------------
        # Incorporate historical success rates for this domain (topic).  The
        # meta_confidence module records successes and failures of
        # previous answers.  A small adjustment based on that record is
        # applied to the affect valence so that domains where Maven has
        # performed well boost confidence and domains with poor history
        # reduce confidence.  Errors in loading the module or computing
        # the key are silently ignored.
        try:
            from brains.personal.memory import meta_confidence  # type: ignore
        except Exception:
            meta_confidence = None  # type: ignore
        if meta_confidence is not None:
            try:
                import re
                q = orig_q.strip().lower()
                # Normalize: remove non‑alphanumeric characters
                q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
                q_norm = re.sub(r"\s+", " ", q_norm).strip()
                parts = q_norm.split()
                domain_key = " ".join(parts[:2]) if parts else ""
                if domain_key:
                    adj = meta_confidence.get_confidence(domain_key)
                    aff_val = float(aff_val) + float(adj)
            except Exception:
                pass

        # --- Dynamic confidence modulation ----------------------------------
        # Apply a global confidence adjustment based on recent success
        # statistics across all domains.  This uses the optional
        # dynamic_confidence module, which returns a small bias derived
        # from the mean of recent adjustments.  The bias is added to
        # affect valence (aff_val) to subtly shift overall confidence.
        try:
            # Import dynamic confidence helper; ignore if unavailable
            from brains.cognitive.reasoning.service.dynamic_confidence import compute_dynamic_confidence  # type: ignore
        except Exception:
            compute_dynamic_confidence = None  # type: ignore
        if compute_dynamic_confidence and meta_confidence is not None:
            try:
                # Gather adjustments from meta confidence stats
                stats = meta_confidence.get_stats(1000) or []
                values = []
                for d in stats:
                    try:
                        values.append(float(d.get("adjustment", 0)))
                    except Exception:
                        continue
                if values:
                    dyn = compute_dynamic_confidence(values)
                    aff_val = float(aff_val) + float(dyn)
            except Exception:
                pass

        # Determine if this input is a question by intent or punctuation.
        storable_type = str(proposed.get("storable_type", "")).upper() or ""
        is_question_intent = False
        # FIX: Also treat REQUEST as a question intent (e.g. "tell me what you know about X")
        if storable_type in ("QUESTION", "REQUEST"):
            is_question_intent = True
        elif not storable_type:
            if _is_question_text(orig_q):
                is_question_intent = True

        # --------------------------------------------------------------------
        # Safety rules check.  Before proceeding with deeper reasoning, inspect
        # the query against developer‑defined safety rules.  These rules are
        # simple case‑insensitive substrings stored via the personal brain.
        # If any rule matches, we avoid returning a potentially inaccurate
        # answer.  Instead we respond with an undefined verdict and a
        # cautionary answer.  This catch‑all filter helps prevent
        # obviously false or harmful statements.  Errors in loading or
        # matching rules are silently ignored.
        try:
            from brains.personal.memory import safety_rules  # type: ignore[attr-defined]
        except Exception:
            safety_rules = None  # type: ignore
        if safety_rules is not None:
            try:
                patterns = safety_rules.get_rules()  # type: ignore[attr-defined]
                q_lower = str(orig_q).lower()
                for pattern in patterns:
                    if pattern and (pattern in q_lower):
                        ans = "I'm not sure that's correct. Let's revisit this question later."
                        conf_sf = 0.4
                        # Record safety filter event in telemetry
                        _update_telemetry("safety_filter")
                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "verdict": "UNKNOWN",
                                "mode": "SAFETY_FILTER",
                                "confidence": conf_sf,
                                "routing_order": {"target_bank": None, "action": None},
                                "supported_by": [],
                                "contradicted_by": [],
                                "answer": ans,
                                "weights_used": {"rule": "safety_filter_v1"}
                            }
                        }
            except Exception:
                pass

        # ----------------------------------------------------------------
        # Ethics rules check.  Similar to the safety filter above, inspect
        # the query against developer‑defined ethics rules stored in
        # ``reports/ethics_rules.json``.  These rules represent
        # case‑insensitive substrings that flag ethically questionable or
        # undesirable input.  If any match is found, return an
        # ``UNKNOWN`` verdict with a cautionary answer.  Failures in
        # loading or parsing the rules file are silently ignored.
        # ----------------------------------------------------------------
        # Ethics rules check.  Inspect the query against developer‑defined
        # ethics rules stored in ``reports/ethics_rules.json``.  This file may
        # contain either a list of simple patterns (backwards compatibility)
        # or a list of structured rule objects with ``pattern``, ``severity``
        # and ``action`` fields.  Structured rules allow differentiating
        # between hard blocks and softer warnings.  When a match is found
        # with a ``block`` action (or with the legacy unstructured format),
        # the question is blocked with a cautionary answer.  When a match
        # is found with a ``warn`` action, the system continues processing
        # but applies a small negative affect adjustment to reduce the
        # resulting confidence.  Any errors in loading or parsing the file
        # result in silently skipping this filter.
        try:
            root = REASONING_ROOT.parents[2]
            ethics_path = root / "reports" / "ethics_rules.json"
            if ethics_path.exists():
                with open(ethics_path, "r", encoding="utf-8") as f:
                    rules = json.load(f)
                q_lower = str(orig_q or "").lower()
                # Backwards compatible: if rules is a list of strings, convert
                # to structured rules with default block action
                if isinstance(rules, list) and rules and isinstance(rules[0], str):
                    rules = [{"pattern": p, "action": "block", "severity": "medium"} for p in rules]
                if isinstance(rules, list):
                    for rule in rules:
                        try:
                            patt = str(rule.get("pattern", "")).strip().lower()
                        except Exception:
                            patt = ""
                        if not patt:
                            continue
                        if patt in q_lower:
                            action = str(rule.get("action", "block")).lower()
                            if action == "warn":
                                # Apply a gentle negative adjustment to affect valence.
                                # This reduces the final confidence without blocking the
                                # question entirely.  Use a small penalty so as not
                                # to completely suppress plausible answers.
                                try:
                                    aff_val -= 0.05
                                except Exception:
                                    pass
                                # Record warn event
                                _update_telemetry("ethics_warn")
                                # Continue checking other patterns in case a block rule
                                # should be applied.
                                continue
                            # For block or unknown actions, return immediately with a
                            # cautionary answer.  Use severity to vary the confidence
                            # slightly; low severity yields a smaller penalty than high.
                            severity = str(rule.get("severity", "medium")).lower()
                            if severity == "low":
                                conf_penalty = 0.3
                            elif severity == "high":
                                conf_penalty = 0.5
                            else:
                                conf_penalty = 0.4
                            ans = "This query may raise ethical concerns. Let's discuss something else."
                            conf_ef = conf_penalty
                            # Record block event in telemetry
                            _update_telemetry("ethics_block")
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "UNKNOWN",
                                    "mode": "ETHICS_FILTER",
                                    "confidence": conf_ef,
                                    "routing_order": {"target_bank": None, "action": None},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": ans,
                                    "weights_used": {"rule": "ethics_filter_v2"}
                                }
                            }
        except Exception:
            pass

        # --------------------------------------------------------------------
        # USER PREFERENCE STATEMENT HANDLER
        # --------------------------------------------------------------------
        # When the language brain detects user preference statements like
        # "I am Josh", "I like the color green", etc., we must store them
        # in the personal brain immediately, NOT send to Teacher.
        try:
            # Check if this is a user preference/identity statement
            parsed_intent = str(proposed.get("intent", "")).strip().lower()

            if parsed_intent == "user_identity_statement":
                # Extract identity info
                identity_slot = str(proposed.get("identity_slot_type", "")).strip()
                identity_value = str(proposed.get("identity_value", "")).strip()

                if identity_slot and identity_value:
                    print(f"[USER_PREFERENCE_HANDLER] Detected identity statement: {identity_slot}={identity_value}")

                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api

                        # Store to personal brain
                        personal_resp = personal_api({
                            "op": "SET_USER_SLOT",
                            "payload": {
                                "slot_name": identity_slot,
                                "value": identity_value
                            }
                        })

                        if personal_resp.get("ok"):
                            print(f"[USER_PREFERENCE_HANDLER] Stored identity to personal brain")

                            # Return success - no need to call Teacher
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "STORED",
                                    "mode": "PERSONAL_IDENTITY_STORED",
                                    "confidence": 1.0,
                                    "routing_order": {"target_bank": "personal", "action": "store"},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": f"Got it! I'll remember that.",
                                    "weights_used": {"rule": "personal_identity_v1"}
                                }
                            }
                    except Exception as e:
                        print(f"[USER_PREFERENCE_HANDLER_ERROR] Failed to store identity: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()

            elif parsed_intent == "user_preference_statement":
                # Extract preference info
                pref_category = str(proposed.get("preference_category", "")).strip()
                pref_value = str(proposed.get("preference_value", "")).strip()

                if pref_category and pref_value:
                    print(f"[USER_PREFERENCE_HANDLER] Detected preference statement: category={pref_category}, value={pref_value}")

                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api

                        # For specific categories (color, animal, food), also store as structured slot
                        if pref_category in ("color", "animal", "food"):
                            slot_name = f"favorite_{pref_category}"
                            personal_resp = personal_api({
                                "op": "SET_USER_SLOT",
                                "payload": {
                                    "slot_name": slot_name,
                                    "value": pref_value
                                }
                            })

                        # Also store as dynamic preference record
                        personal_resp = personal_api({
                            "op": "ADD_PREFERENCE_RECORD",
                            "payload": {
                                "category": pref_category,
                                "value": pref_value,
                                "sentiment": "like",
                                "confidence": 0.9
                            }
                        })

                        if personal_resp.get("ok"):
                            print(f"[USER_PREFERENCE_HANDLER] Stored preference to personal brain")

                            # Return success - no need to call Teacher
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "STORED",
                                    "mode": "PERSONAL_PREFERENCE_STORED",
                                    "confidence": 1.0,
                                    "routing_order": {"target_bank": "personal", "action": "store"},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": f"Got it! I'll remember that you like {pref_value}.",
                                    "weights_used": {"rule": "personal_preference_v1"}
                                }
                            }
                    except Exception as e:
                        print(f"[USER_PREFERENCE_HANDLER_ERROR] Failed to store preference: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            print(f"[USER_PREFERENCE_HANDLER_ERROR] Pattern check failed: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            pass

        # Perform a common sense check on binary questions.  If the question
        # obviously contradicts basic knowledge (e.g. "Is Mars a country?"),
        # return a correction with high confidence and skip further reasoning.
        if is_question_intent:
            cs_res = _common_sense_check(orig_q)
            if cs_res:
                # Compose a corrective answer: clarify the true category of the entity.
                try:
                    # Capitalise the entity for the answer
                    ent = cs_res.get("entity", "").strip().title()
                    correct = cs_res.get("correct", "").strip()
                    wrong = cs_res.get("wrong", "").strip()
                    ans = f"No, {ent} is a {correct}, not a {wrong}."
                except Exception:
                    ans = "No, that is not correct."
                # High confidence with slight affect modulation
                try:
                    conf_cs = 0.95 + (aff_val * 0.05 + aff_arousal * 0.03)
                except Exception:
                    conf_cs = 0.95
                conf_cs = max(0.0, min(1.0, conf_cs))
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "FALSE",
                        "mode": "COMMON_SENSE",
                        "confidence": conf_cs,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "answer": ans,
                        "weights_used": {"rule": "common_sense_v1"}
                    }
                }

        # Commands are not evaluated as facts and should skip storage.
        # FIX: REQUESTs (like "tell me what you know about X") should go through question handling, not skip!
        if storable_type == "COMMAND":
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": "SKIP_STORAGE",
                    "mode": f"{storable_type}_INPUT",
                    "confidence": 0.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "weights_used": {"rule": "intent_filter_v1"}
                }
            }
        # Emotion and opinion statements should be handled outside of the factual reasoner.
        if storable_type in ("EMOTION", "OPINION"):
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": "SKIP_STORAGE",
                    "mode": f"{storable_type}_INPUT",
                    "confidence": 0.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "weights_used": {"rule": "intent_filter_v1"}
                }
            }
        # Inputs labelled as UNKNOWN (e.g. greetings like "hi", "hello") should not be
        # treated as factual statements even if evidence exists for the raw text.  Without
        # this guard, duplicate "hello" entries in memory could cause trivial greetings
        # to be accepted as facts.  When the storable_type is UNKNOWN, skip storage and
        # return an UNKNOWN verdict regardless of evidence.  This ensures chit‑chat
        # does not accumulate in memory.
        if storable_type == "UNKNOWN":
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": "SKIP_STORAGE",
                    "mode": "UNKNOWN_INPUT",
                    "confidence": 0.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "weights_used": {"rule": "intent_filter_v1"}
                }
            }
        # If the original query is a question, treat this operation as answering the question.
        if is_question_intent:
            # ------------------------------------------------------------------
            # Knowledge graph lookup: attempt to answer simple definition
            # questions using the semantic memory.  Handle patterns like
            # "what is X" or "who is X".  If an answer is found, return it
            # immediately with high confidence.  This block avoids importlib
            # overhead by importing the module directly and falls back to
            # content when original_query is unavailable.  It also supports
            # inverse lookups (object → subject) to answer questions like
            # "What is the red planet?" when the fact stored is (mars, is, the red planet).
            try:
                # Import the knowledge graph.  If it does not exist, skip
                # lookup gracefully.  Synonym mappings are applied at
                # the API layer but are not used during direct lookup
                # here to avoid returning tautological answers (e.g. "the
                # red planet" → "the red planet").  Inverse lookups
                # handle synonym‐like phrasing by matching against
                # objects directly.
                from brains.personal.memory import knowledge_graph as kg_mod  # type: ignore
                import re as _re
                # Determine the question text: prefer original_query, fall back to content.
                qsource = orig_q.strip() if orig_q else str(content or "").strip()
                qnorm = qsource.lower()
                # Match "what is X" or "who is X" at the start of the question.
                mkg = _re.match(r"^(?:what|who)\s+is\s+(.+)", qnorm)
                if mkg:
                    subj = mkg.group(1).rstrip("?").strip()
                    if subj:
                        # Generate candidate subjects: the raw text and a version
                        # without leading articles.  We do not apply synonym
                        # mappings here to avoid confusing subject–object
                        # orientation.  Instead, inverse lookup handles
                        # synonym‐like phrasing by matching objects directly.
                        subj_norm = subj.lower().strip()
                        candidates = [subj_norm]
                        cand_strip = _re.sub(r"^(?:the|a|an)\s+", "", subj_norm).strip()
                        if cand_strip and cand_strip != subj_norm:
                            candidates.append(cand_strip)
                        kg_ans = None
                        # Direct lookup: subject → object.  For each
                        # candidate, attempt a direct fact lookup.  If the
                        # returned object matches the candidate itself
                        # (ignoring case and leading articles), treat it
                        # as tautological and continue to inverse lookup.
                        for cand in candidates:
                            if not cand:
                                continue
                            try:
                                res = kg_mod.query_fact(cand, "is")
                            except Exception:
                                res = None
                            if not res:
                                continue
                            # Normalise both candidate and answer for comparison
                            cand_norm = cand.lower().strip()
                            res_norm = str(res).lower().strip()
                            # Remove leading articles for comparison
                            import re as _re
                            cand_stripped = _re.sub(r"^(?:the|a|an)\s+", "", cand_norm).strip()
                            res_stripped = _re.sub(r"^(?:the|a|an)\s+", "", res_norm).strip()
                            # If the answer equals the candidate (after stripping), skip this result
                            if res_stripped == cand_stripped:
                                continue
                            kg_ans = res
                            break
                        # Inverse lookup: object → subject, with synonym support.  If no answer
                        # was found via direct lookup, search all facts for a record
                        # where the relation is "is" and the object matches the candidate.
                        # To support synonyms (e.g. "the red planet" → "mars"), both the
                        # candidate and record object are canonicalised via the synonym
                        # mapping before comparison.  Leading articles are also removed.
                        if not kg_ans:
                            try:
                                facts = kg_mod.list_facts(0)
                            except Exception:
                                facts = []
                            # Attempt to import the synonym module; if unavailable or
                            # mapping fails, canonicalisation will default to the
                            # lowercase stripped term.
                            try:
                                from brains.personal.memory import synonyms as syn_mod  # type: ignore
                            except Exception:
                                syn_mod = None  # type: ignore
                            import re as _re
                            for cand in candidates:
                                if not cand:
                                    continue
                                # Prepare the candidate: lower‑case and strip articles
                                cand_norm = str(cand).lower().strip()
                                cand_clean = _re.sub(r"^(?:the|a|an)\s+", "", cand_norm).strip()
                                # Canonicalise the candidate if a synonym mapping is available
                                if syn_mod:
                                    try:
                                        canon_cand = syn_mod.get_canonical(cand_clean)  # type: ignore
                                    except Exception:
                                        canon_cand = cand_clean
                                else:
                                    canon_cand = cand_clean
                                for rec in facts:
                                    try:
                                        if str(rec.get("relation", "")).strip().lower() != "is":
                                            continue
                                        obj_val = str(rec.get("object", "")).strip().lower()
                                        # Remove articles from the stored object
                                        obj_clean = _re.sub(r"^(?:the|a|an)\s+", "", obj_val).strip()
                                        # Canonicalise the stored object
                                        if syn_mod:
                                            try:
                                                canon_obj = syn_mod.get_canonical(obj_clean)  # type: ignore
                                            except Exception:
                                                canon_obj = obj_clean
                                        else:
                                            canon_obj = obj_clean
                                        if canon_obj == canon_cand:
                                            kg_ans = rec.get("subject")
                                            break
                                    except Exception:
                                        continue
                                if kg_ans:
                                    break
                        if kg_ans:
                            # Compute a confidence influenced by affect metrics.  The base
                            # confidence reflects the reliability of explicit facts.
                            try:
                                conf_kg = 0.88 + aff_val * 0.05 + aff_arousal * 0.03
                            except Exception:
                                conf_kg = 0.88
                            conf_kg = max(0.0, min(conf_kg, 1.0))

                            # Store Q&A pair for future reference
                            _store_qa_memory(orig_q, str(kg_ans), conf_kg)

                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "TRUE",
                                    "mode": "KG_ANSWER",
                                    "confidence": conf_kg,
                                    "routing_order": {"target_bank": None, "action": None},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": kg_ans,
                                    "weights_used": {"rule": "knowledge_graph_v1"}
                                }
                            }

                        # If no direct or inverse match, attempt to derive an answer via knowledge graph inference.
                        # This uses stored inference rules (e.g. located_in + part_of -> located_in) to
                        # generate transitive facts such as (A located_in C) from (A located_in B, B part_of C).
                        # Only attempt inference when the question pattern resembles "where is X" or
                        # when the relation is implicitly "located_in" (what is X located in?).  We try
                        # matching any inferred fact whose subject matches the candidate term and
                        # relation is one of 'located_in' or 'part_of'.
                        if not kg_ans:
                            try:
                                # Run inference via the V2 rule engine.  If unavailable, fall back to
                                # the simple transitive closure provided by kg_mod.infer().
                                inf_results: list = []
                                try:
                                    # run_inference returns inferred facts based on custom rules
                                    inf_results = kg_mod.run_inference(10)  # type: ignore[attr-defined]
                                except Exception:
                                    # fallback to built‑in transitive inference on located_in/part_of
                                    try:
                                        inf_results = kg_mod.infer(10)  # type: ignore[attr-defined]
                                    except Exception:
                                        inf_results = []
                                if inf_results:
                                    import re as _re
                                    for cand in candidates:
                                        if not cand:
                                            continue
                                        cand_norm = cand.lower().strip()
                                        for rec in inf_results:
                                            try:
                                                subj_val = str(rec.get("subject", "")).strip().lower()
                                                rel_val = str(rec.get("relation", "")).strip().lower()
                                                obj_val = str(rec.get("object", "")).strip()
                                                # Check if this inferred fact matches our candidate subject and a relevant relation
                                                if subj_val == cand_norm and rel_val in {"located_in", "part_of"}:
                                                    kg_ans = obj_val
                                                    break
                                            except Exception:
                                                continue
                                        if kg_ans:
                                            break
                                if kg_ans:
                                    # Confidence for inferred answers is slightly lower than direct facts.
                                    try:
                                        conf_inf = 0.75 + aff_val * 0.05 + aff_arousal * 0.03
                                    except Exception:
                                        conf_inf = 0.75
                                    conf_inf = max(0.0, min(conf_inf, 1.0))

                                    # Store inferred Q&A pair
                                    _store_qa_memory(orig_q, str(kg_ans), conf_inf)

                                    return {
                                        "ok": True,
                                        "op": op,
                                        "mid": mid,
                                        "payload": {
                                            "verdict": "TRUE",
                                            "mode": "INFERRED",
                                            "confidence": conf_inf,
                                            "routing_order": {"target_bank": None, "action": None},
                                            "supported_by": [],
                                            "contradicted_by": [],
                                            "answer": kg_ans,
                                            "weights_used": {"rule": "knowledge_inference_v1"}
                                        }
                                    }
                            except Exception:
                                # Ignore inference errors silently
                                pass
            except Exception:
                # On any error during knowledge graph retrieval, silently continue
                pass
            # Prior knowledge: consult the cross‑episode QA memory to see if this
            # question has been answered in a previous run.  If a stored answer
            # exists, return it immediately with high confidence, bypassing
            # retrieval and heuristic guessing.
            stored_ans = _qa_memory_lookup(orig_q)
            if stored_ans:
                # Clamp confidence to a high value but account for affect modulation
                try:
                    conf_qa = 0.85 + aff_val * 0.05 + aff_arousal * 0.03
                except Exception:
                    conf_qa = 0.85
                conf_qa = max(0.0, min(conf_qa, 1.0))
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "TRUE",
                        "mode": "KNOWN_ANSWER",
                        "confidence": conf_qa,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "answer": stored_ans,
                        "weights_used": {"rule": "qa_memory_v1"}
                    }
                }
            # Look through provided evidence results to find a candidate answer
            evidence = payload.get("evidence") or {}
            results = (evidence.get("results") or []) if isinstance(evidence, dict) else []
            ans_record = None
            answer_text = None
            for it in results:
                if not isinstance(it, dict):
                    continue
                raw_content = str(it.get("content", "")).strip()
                if not raw_content:
                    continue
                # Attempt to parse JSON content (e.g. {"text": "...", "temperature": ...})
                parsed_text = None
                if raw_content.startswith("{") and raw_content.endswith("}"):
                    try:
                        data = json.loads(raw_content)
                        parsed_text = str(data.get("text", raw_content)).strip()
                    except Exception:
                        parsed_text = raw_content
                else:
                    parsed_text = raw_content
                # Skip if parsed text is empty or still looks like a question
                if not parsed_text or parsed_text.endswith("?"):
                    continue
                ans_record = it
                answer_text = parsed_text
                break
            if ans_record and answer_text:
                # Additional inference for yes/no membership questions.  When the
                # question begins with "Is X one of ..." and the retrieved
                # answer contains the subject, infer a simple affirmative.
                if is_question_intent:
                        try:
                            import re as _re
                            q_lower = (orig_q or "").strip().lower().rstrip("?")
                            # Match patterns like "is red one of the spectrum colors"
                            m = _re.match(r"^is\s+([a-z0-9\s\-]+?)\s+one\s+of\s+(?:the\s+)?(.+)$", q_lower)
                            if m:
                                subj = m.group(1).strip()
                                group = m.group(2).strip()
                                # Resolve synonyms to canonical form
                                try:
                                    from brains.personal.memory import synonyms as syn_mod  # type: ignore
                                    canon_subj = syn_mod.get_canonical(subj)  # type: ignore[attr-defined]
                                    syn_groups = syn_mod.list_groups()  # type: ignore[attr-defined]
                                    # List of synonym variants for the subject
                                    subj_syns = syn_groups.get(canon_subj, [canon_subj]) if syn_groups else [canon_subj]
                                except Exception:
                                    canon_subj = subj.lower()
                                    subj_syns = [canon_subj]
                                # If any variant of the subject appears in the answer, infer membership
                                try:
                                    ans_lower = answer_text.lower()
                                except Exception:
                                    ans_lower = answer_text
                                matched = False
                                for s in subj_syns:
                                    if s and s.lower() in ans_lower:
                                        matched = True
                                        break
                                if matched:
                                    # Compose a concise affirmative answer.  Capitalise the original subject
                                    subj_cap = subj.capitalize()
                                    # Strip trailing punctuation from group (e.g. "spectrum colors")
                                    group_clean = group.rstrip(".")
                                    answer_text = f"Yes, {subj_cap} is one of the {group_clean}."
                        except Exception:
                            pass
                # Use the stored confidence if present, otherwise default to 0.85
                try:
                    conf_val = float(ans_record.get("confidence", 0.85))
                except Exception:
                    conf_val = 0.85
                # Adjust confidence by affect valence.  Positive valence
                # slightly increases confidence, negative valence decreases it.
                try:
                    conf_val = conf_val + aff_val * 0.05 + aff_arousal * 0.03
                except Exception:
                    pass
                conf_val = max(0.0, min(conf_val, 1.0))

                # Store Q&A pair for future reference
                _store_qa_memory(orig_q, answer_text, conf_val)

                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "TRUE",
                        "mode": "ANSWERED",
                        "confidence": conf_val,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [ans_record.get("id")] if ans_record.get("id") else [],
                        "contradicted_by": [],
                        "answer": answer_text,
                        "answer_source_id": ans_record.get("id"),
                        "weights_used": {"rule": "question_answer_v1"}
                    }
                }
            # If no direct answer found, attempt an educated guess using heuristics
            guess = _educated_guess_for_question(orig_q)
            if guess:
                # Educated guesses are presented with moderate confidence and marked as THEORY
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "THEORY",
                        "mode": "EDUCATED_GUESS",
                        "confidence": 0.6,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "answer": guess,
                        "weights_used": {"rule": "educated_guess_v1"}
                    }
                }

            # As a fallback, attempt to evaluate the question as a logical or mathematical expression.
            # This leverages the agent's System‑2 tools for precise computation when possible.
            expr = orig_q or content
            expr = str(expr or "").strip().rstrip("?")
            # Preprocess expressions to remove common question prefixes so that
            # boolean and arithmetic evaluation can operate on the core
            # expression.  This allows inputs like "What is 2+2?" or
            # "Compute true and false" to be handled by the appropriate tool.
            try:
                import re
                # Regex captures variations of question phrases followed by the expression
                m = re.match(r"^(?:what(?:'s| is)|calculate|compute|evaluate|solve)\s+(.*)", expr, re.IGNORECASE)
                if m:
                    expr = m.group(1).strip()
            except Exception:
                pass
            # Lowercase copy used for keyword detection
            try:
                lower_expr = expr.lower()
            except Exception:
                lower_expr = expr
            answered_by_tool = False
            answer_val: Any = None
            tool_rule = None
            # Heuristic: if boolean keywords or operators are present, use the logic tool
            try:
                # Look for explicit boolean literals or logical operators (with surrounding spaces)
                if any(w in lower_expr for w in ["true", "false", " and ", " or ", "not "]):
                    import importlib
                    logic_mod = importlib.import_module("brains.agent.tools.logic_tool")
                    logic_resp = logic_mod.service_api({"op": "EVAL", "payload": {"expression": expr}})
                    if logic_resp.get("ok", False):
                        answer_val = logic_resp.get("payload", {}).get("result")
                        answered_by_tool = True
                        tool_rule = "logic_tool_v1"
            except Exception:
                pass
            # If boolean evaluation didn't apply or failed, try arithmetic evaluation
            if not answered_by_tool:
                try:
                    has_digit = any(ch.isdigit() for ch in expr)
                except Exception:
                    has_digit = False
                # Check for math operators; require at least one digit and an operator to reduce false positives
                has_op = any(op in expr for op in ["+", "-", "*", "/", "%", "**"])
                if has_digit and has_op:
                    try:
                        import importlib
                        math_mod = importlib.import_module("brains.agent.tools.math_tool")
                        math_resp = math_mod.service_api({"op": "CALC", "payload": {"expression": expr}})
                        if math_resp.get("ok", False):
                            answer_val = math_resp.get("payload", {}).get("result")
                            answered_by_tool = True
                            tool_rule = "math_tool_v1"
                    except Exception:
                        pass
            if answered_by_tool:
                # When a tool produces a result, treat it as a confident answer.
                try:
                    conf_tool = 0.9 + (aff_val * 0.05 + aff_arousal * 0.03)
                except Exception:
                    conf_tool = 0.9
                conf_tool = max(0.0, min(1.0, conf_tool))

                # Store tool-based Q&A pair
                _store_qa_memory(orig_q, str(answer_val), conf_tool)

                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "TRUE",
                        "mode": "ANSWERED",
                        "confidence": conf_tool,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "answer": str(answer_val),
                        "weights_used": {"rule": tool_rule or "system2_tool_v1"}
                    }
                }

            # Otherwise, no answer available from Maven's own knowledge
            # Try the teacher as a fallback if enabled
            teacher_enabled = True  # TODO: Make this configurable
            teacher_answer = None
            teacher_facts_stored = 0

            # --------------------------------------------------------------------
            # SELF-INTENT GATE: Block Teacher for self-identity questions
            # --------------------------------------------------------------------
            # Self-identity questions MUST be answered by self_model/self_dmn,
            # NEVER by Teacher. Teacher would hallucinate incorrect facts about
            # Maven (e.g., Apache Maven, Java build tool, Jason van Zyl, etc.).
            #
            # If a self-intent is detected, skip Teacher entirely and route to
            # self_model for the canonical answer.
            try:
                import re
                q_lower = str(orig_q or "").strip().lower()

                # Normalize text: lowercase, strip punctuation, collapse spaces
                normalized = q_lower
                for punct in ['.', ',', '!', '?', ':', ';']:
                    normalized = normalized.replace(punct, ' ')
                normalized = ' '.join(normalized.split())  # collapse multiple spaces

                # SELF-MEMORY patterns - separate stats vs health/scan
                # CRITICAL: These questions about Maven's OWN memory/learning must NEVER go to Teacher!

                # Stats patterns: counting facts, what's been learned
                memory_stats_patterns = [
                    "how many facts have you learned",
                    "how much have you learned",
                    "how many facts do you know",
                    "what have you learned so far",
                    "what do you remember",
                    "what do you remember right now",
                    "memory stats",
                    "show memory stats"
                ]

                # Scan/health patterns: diagnosing memory system health
                memory_scan_patterns = [
                    "scan your memory system",
                    "scan memory system",
                    "scan your memory",
                    "scan memory",
                    "diagnose your memory",
                    "diagnose memory",
                    "check your memory",
                    "check your memory system"
                ]

                system_scan_patterns = [
                    "scan self",
                    "system scan",
                    "full system scan",
                    "scan your system",
                    "scan your entire system",
                    "scan your runtime",
                    "scan your runtime environment",
                    "scan your installation",
                ]

                routing_scan_patterns = [
                    "scan routing table",
                    "scan your routing table",
                    "scan routes",
                    "routing scan",
                    "show routing signatures",
                    "list routing signatures",
                    "list routing rules",
                ]

                codebase_scan_patterns = [
                    "scan codebase",
                    "scan your codebase",
                    "scan your code base",
                    "scan your entire codebase",
                    "list all python files",
                    "scan code and list all python files",
                    "scan your code and list all python files",
                    "scan all code",
                    "scan code",
                    "scan my code",
                    "scan your code and list modules",
                    "scan code and list modules",
                    "scan your code and list all python modules",
                ]

                cognitive_scan_patterns = [
                    "scan cognitive brains",
                    "scan your cognitive brains",
                    "scan brains",
                    "scan your brains",
                    "list all cognitive brains",
                    "list your brains",
                    "scan brains directory",
                ]

                # SELF-INTRODUCTION patterns - "explain yourself" requests
                # These trigger SELF_INTRODUCTION operation for dynamic intro
                # Include common typo variants like "explaine"
                self_introduction_patterns = [
                    r"\bexplain[e]?\s+your\s*self\b",  # explain/explaine your self
                    r"\bexplain[e]?\s+yourself\b",     # explain/explaine yourself
                    r"\bexpl[ai]+n[e]?\s+your\s*self\b",  # explian/explaain variants
                    r"\bintroduce\s+yourself\b",
                    r"\bintroduce\s+your\s*self\b",
                    r"\bexplain[e]?\s+yourself\s+to\b",  # typo tolerance
                    r"\bintroduce\s+yourself\s+to\b",
                    r"\btell\s+\w+\s+about\s+yourself\b",
                    r"\btell\s+\w+\s+who\s+you\s+are\b",
                    r"\bexplain[e]?\s+to\s+\w+\s+who\s+you\s+are\b",  # typo tolerance
                    r"\bmaven\s+explain[e]?\s+your\s*self\b",  # typo tolerance
                    r"\b\w+\s+maven\s+explain[e]?\s+your\s*self\b",  # typo tolerance
                    r"\bwhoami\b",
                    r"\byour\s*self\b",  # standalone "your self"
                ]

                # SELF-IDENTITY patterns - regex for identity/code questions
                self_identity_patterns = [
                    r"\bwho are you\b",
                    r"\bwho you are\b",
                    r"\bwhat are you\b",
                    r"\btell\s+me\s+about\s+yourself\b",
                    r"\btell\s+me\s+about\s+your\s*self\b",
                    r"\bwhat\s+can\s+you\s+tell\s+me\s+about\s+your\s*self\b",
                    r"\bdescribe\s+yourself\b",
                    r"\bdescribe\s+your\s*self\b",
                    r"\bwhat.*your\s+name\b",
                    r"\bare\s+you\s+maven\b",
                    r"\bwhat\s+(is|are)\s+maven\b",
                    r"\bwho\s+(created|built|made)\s+maven\b",
                    r"\bare\s+you\s+(an?\s+)?llm\b",
                    r"\bare\s+you\s+(a\s+)?large\s+language\s+model\b",
                    r"training\s+data",
                    r"knowledge\s+cutoff",
                    r"when\s+were\s+you\s+created",
                    r"when\s+were\s+you\s+trained",
                    r"how\s+were\s+you\s+trained",
                    r"what\s+model\s+are\s+you",
                ]

                # SELF-CODE patterns - regex for code introspection
                self_code_patterns = [
                    r"\bwhat.*your\s+(own\s+)?code\b",
                    r"\bwhat.*your\s+(own\s+)?systems?\b",
                    r"\bwhat.*you\s+built\b",
                    r"\bhow\s+do\s+you\s+work\b",
                    r"\bdescribe\s+your\s+code\b",
                    r"\bwhat\s+are\s+your\s+brains\b",
                    r"\byour\s+(cognitive\s+)?brains\b",
                    r"\byour\s+source\s+code\b",
                    r"\bhow\s+are\s+you\s+structured\b",
                    r"\bhow\s+does\s+maven\s+work\b",
                    r"\byour\s+architecture\b",
                    r"\bwhat.*you\s+know\s+about\s+(yourself|your\s+(own\s+)?code|your\s+systems)\b"
                ]

                # SELF-UPGRADE patterns - planning self-improvement
                self_upgrade_patterns = [
                    r"\bplan\s+upgrade\s+for\s+your\s*self\b",
                    r"\bplan\s+upgrades?\s+for\s+yourself\b",
                    r"\bhow\s+can\s+you\s+improve\s+yourself\b",
                    r"\bplan\s+your\s+(own\s+)?upgrades?\b",
                    r"\bwhat\s+(should|could)\s+you\s+improve\b",
                    r"\bhow\s+to\s+improve\s+maven\b",
                    r"\bupgrade\s+plan\s+for\s+maven\b",
                    r"\bself[- ]improvement\s+plan\b",
                    r"\bplan\s+to\s+improve\s+yourself\b",
                    r"\bwhat\s+improvements?\s+(do\s+you\s+need|should\s+maven\s+make)\b"
                ]

                # ================================================================
                # CAPABILITY PATTERNS - "can you X" questions about Maven's abilities
                # CRITICAL: These MUST be answered from capability_snapshot, NOT Teacher
                # ================================================================
                capability_patterns = [
                    # Web/Browser - CRITICAL: "can you browse the web" must match
                    "can you search the web",
                    "can you browse the web",
                    "can you browse the internet",
                    "can you look this up online",
                    "can you search online",
                    "can you do web search",
                    "can you access the internet",
                    "do you have internet access",
                    "are you connected to the internet",
                    # Code execution
                    "can you run code",
                    "can you execute code",
                    "can you run python",
                    "can you run scripts",
                    "can you execute scripts",
                    "can you run programs",
                    "can you execute programs",
                    # Control programs
                    "can you control other programs",
                    "can you control apps on my computer",
                    "can you control other applications",
                    "can you launch other apps",
                    "can you run other programs",
                    "can you control programs",
                    "control other programs",
                    # File access
                    "can you read files on my system",
                    "can you change files on my system",
                    "can you read or change files",
                    "can you read files",
                    "can you change files",
                    "can you access files on my computer",
                    "can you modify files",
                    "can you write files",
                    "read or change files",
                    "read files",
                    "change files",
                    # Autonomous tools
                    "can you use tools without me asking",
                    "can you use the internet without me asking",
                    "do you use tools autonomously",
                    "do you act on your own",
                    "do you do things without asking",
                    # Upgrade questions - CRITICAL
                    "what upgrades do you need",
                    "what upgrade do you need",
                    "what do you need",
                    # Creative capability questions
                    "can you write a story",
                    "can you write me a story",
                    "can you create a story",
                    "can you write a poem",
                    "can you compose",
                    "can you generate",
                    # General capability questions
                    "what can you do",
                    "what are your capabilities",
                    "what are you capable of",
                    "what tools do you have",
                    "what tools can you use",
                    # Scan code patterns
                    "scan your code",
                    "scan your codebase",
                    "scan code",
                ]

                # ================================================================
                # HISTORY PATTERNS - questions about conversation history
                # CRITICAL: These MUST be answered from session history, NOT Teacher
                # ================================================================
                history_patterns = [
                    "what did i ask you first today",
                    "what was my first question",
                    "what did i ask you first",
                    "what did we talk about yesterday",
                    "what did we talk about last time",
                    "what did we discuss",
                    "what have we discussed",
                    "what did we talk about before",
                    "what topics have we covered",
                    "what did i say earlier",
                    "what did i mention",
                    "do you remember what i asked",
                    "do you remember our conversation",
                    "what was our last topic",
                ]

                # ================================================================
                # USER MEMORY PATTERNS - questions about what Maven knows about user
                # CRITICAL: These MUST be answered from personal banks, NOT Teacher
                # ================================================================
                user_memory_patterns = [
                    "what do you remember about me",
                    "what do you know about me",
                    "what have you learned about me",
                    "what is the most important thing you know about me",
                    "what have you learned so far about me",
                    "do you remember my preferences",
                    "what are my preferences",
                    "tell me what you know about me",
                ]

                # ================================================================
                # EXPLAIN_LAST PATTERNS - introspection about previous answer
                # CRITICAL: These MUST be answered from EXPLAIN_LAST, NOT Teacher
                # ================================================================
                explain_last_patterns = [
                    "why did you answer that way",
                    "why that answer",
                    "how did you get that answer",
                    "how did you arrive at that",
                    "which parts of your system helped you answer",
                    "which brains helped",
                    "which parts helped",
                    "did you use the teacher to answer that",
                    "did you use the teacher",
                    "did you call the teacher",
                    "what would you do differently next time",
                    "how would you improve that answer",
                    "explain your reasoning",
                    "explain that answer",
                    "why did you say that",
                ]

                # Check for self-memory questions FIRST (using substring matching)
                is_self_query = False
                self_kind = None
                self_mode = None  # "stats" vs "health"

                # Check stats patterns first
                for substring in memory_stats_patterns:
                    if substring in normalized:
                        is_self_query = True
                        self_kind = "memory"
                        self_mode = "stats"
                        print(f"[SELF_INTENT_GATE] Detected self-memory question, blocking Teacher")
                        print(f"[SELF_INTENT_GATE] Matched substring: '{substring}'")
                        print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                        break

                # If not stats, check scan/health patterns
                if not is_self_query:
                    for substring in memory_scan_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "memory"
                            self_mode = "health"
                            print(f"[SELF_INTENT_GATE] Detected self-memory question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched substring: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                if not is_self_query:
                    for substring in system_scan_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "system_scan"
                            self_mode = "full"
                            print(f"[SELF_INTENT_GATE] Detected self-system scan, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched substring: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                if not is_self_query:
                    for substring in routing_scan_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "routing_scan"
                            print(f"[SELF_INTENT_GATE] Detected self-routing scan, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched substring: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                if not is_self_query:
                    for substring in codebase_scan_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "code_scan"
                            print(f"[SELF_INTENT_GATE] Detected self-codebase scan, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched substring: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                if not is_self_query:
                    for substring in cognitive_scan_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "cognitive_scan"
                            print(f"[SELF_INTENT_GATE] Detected self-cognitive scan, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched substring: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # If not memory, check for INTRODUCTION questions first (more specific)
                # "explain yourself" should trigger SELF_INTRODUCTION, not generic identity
                if not is_self_query:
                    for pattern in self_introduction_patterns:
                        if re.search(pattern, q_lower):
                            is_self_query = True
                            self_kind = "introduction"
                            print(f"[SELF_INTENT_GATE] Detected self-introduction request, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # If not memory or introduction, check for identity questions
                if not is_self_query:
                    for pattern in self_identity_patterns:
                        if re.search(pattern, q_lower):
                            is_self_query = True
                            self_kind = "identity"
                            print(f"[SELF_INTENT_GATE] Detected self-identity question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # If not memory or identity, check for code questions
                if not is_self_query:
                    for pattern in self_code_patterns:
                        if re.search(pattern, q_lower):
                            is_self_query = True
                            self_kind = "code"
                            print(f"[SELF_INTENT_GATE] Detected self-code question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # Check for self-upgrade planning questions
                if not is_self_query:
                    for pattern in self_upgrade_patterns:
                        if re.search(pattern, q_lower):
                            is_self_query = True
                            self_kind = "upgrade"
                            print(f"[SELF_INTENT_GATE] Detected self-upgrade question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # ================================================================
                # CAPABILITY QUESTIONS - Route to capability_snapshot, NOT Teacher
                # ================================================================
                if not is_self_query:
                    for substring in capability_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "capability"
                            print(f"[SELF_INTENT_GATE] Detected CAPABILITY question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # ================================================================
                # HISTORY QUESTIONS - Route to session_history, NOT Teacher
                # ================================================================
                if not is_self_query:
                    for substring in history_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "history"
                            print(f"[SELF_INTENT_GATE] Detected HISTORY question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # ================================================================
                # USER MEMORY QUESTIONS - Route to personal_brain, NOT Teacher
                # ================================================================
                if not is_self_query:
                    for substring in user_memory_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "user_memory"
                            print(f"[SELF_INTENT_GATE] Detected USER MEMORY question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                # ================================================================
                # EXPLAIN_LAST QUESTIONS - Route to EXPLAIN_LAST, NOT Teacher
                # ================================================================
                if not is_self_query:
                    for substring in explain_last_patterns:
                        if substring in normalized:
                            is_self_query = True
                            self_kind = "explain_last"
                            print(f"[SELF_INTENT_GATE] Detected EXPLAIN_LAST question, blocking Teacher")
                            print(f"[SELF_INTENT_GATE] Matched: '{substring}'")
                            print(f"[SELF_INTENT_GATE] Question: '{orig_q[:60]}...'")
                            break

                if is_self_query:
                    if self_mode:
                        print(f"[SELF_INTENT_GATE] self_kind={self_kind}, mode={self_mode}, routing appropriately")
                    else:
                        print(f"[SELF_INTENT_GATE] self_kind={self_kind}, routing appropriately")

                    self_answer = None
                    self_confidence = 0.95
                    answer_source = "self_model"

                    try:
                        # ============================================================
                        # CAPABILITY QUESTIONS - Use capabilities.answer_capability_question()
                        # ============================================================
                        if self_kind == "capability":
                            print(f"[SELF_INTENT_GATE] Routing to capabilities.answer_capability_question()")
                            from capabilities import answer_capability_question
                            cap_result = answer_capability_question(orig_q)
                            if cap_result:
                                self_answer = cap_result.get("answer", "")
                                self_confidence = 0.99  # High confidence - from config
                                answer_source = "capability_snapshot"
                                print(f"[SELF_INTENT_GATE] Got capability answer: enabled={cap_result.get('enabled')}")
                            else:
                                self_answer = "I cannot determine my capability for that specific question. Please ask about web search, code execution, file access, or tool usage."

                        # ============================================================
                        # HISTORY QUESTIONS - Use system_history_brain QUERY_HISTORY
                        # ============================================================
                        elif self_kind == "history":
                            print(f"[SELF_INTENT_GATE] Routing to system_history_brain.QUERY_HISTORY")
                            from brains.cognitive.system_history.service.system_history_brain import service_api as history_api

                            # Determine history type from query
                            history_type = "session"
                            if "yesterday" in normalized or "last time" in normalized:
                                history_type = "yesterday"
                            elif "first" in normalized:
                                history_type = "first_today"
                            elif "recent" in normalized:
                                history_type = "recent"

                            history_resp = history_api({
                                "op": "QUERY_HISTORY",
                                "payload": {
                                    "query": orig_q,
                                    "history_type": history_type
                                }
                            })
                            if history_resp.get("ok"):
                                payload = history_resp.get("payload", {})
                                self_answer = payload.get("answer", "I have no record of that conversation.")
                                self_confidence = 0.95 if payload.get("found") else 0.80
                                answer_source = "session_history"
                            else:
                                self_answer = "I can only see this session. I have no record of previous days or sessions."

                        # ============================================================
                        # USER MEMORY QUESTIONS - Use personal_brain
                        # ============================================================
                        elif self_kind == "user_memory":
                            print(f"[SELF_INTENT_GATE] Routing to personal_brain for user memory")
                            from brains.personal.service.personal_brain import service_api as personal_api

                            personal_resp = personal_api({
                                "op": "ANSWER_PERSONAL_QUESTION",
                                "payload": {
                                    "question_type": "what_do_i_know_about_you",
                                    "query": orig_q
                                }
                            })
                            if personal_resp.get("ok"):
                                payload = personal_resp.get("payload", {})
                                self_answer = payload.get("answer")
                                if not self_answer:
                                    # Try to summarize stored facts
                                    facts = payload.get("facts", [])
                                    if facts:
                                        fact_lines = [f"- {f.get('content', str(f))}" for f in facts[:5]]
                                        self_answer = "Here's what I know about you:\n" + "\n".join(fact_lines)
                                    else:
                                        self_answer = "I don't have any stored facts about you yet. Tell me about yourself and I'll remember it!"
                                self_confidence = payload.get("confidence", 0.85)
                                answer_source = "personal_bank"
                            else:
                                self_answer = "I haven't stored any personal information about you yet. I learn from our conversations."

                        # ============================================================
                        # EXPLAIN_LAST QUESTIONS - Use language_brain EXPLAIN_LAST
                        # ============================================================
                        elif self_kind == "explain_last":
                            print(f"[SELF_INTENT_GATE] Routing to language_brain.EXPLAIN_LAST")
                            from brains.cognitive.language.service.language_brain import service_api as language_api

                            # Get last run context from memory
                            explain_resp = language_api({
                                "op": "EXPLAIN_LAST",
                                "payload": {
                                    "query": orig_q
                                }
                            })
                            if explain_resp.get("ok"):
                                payload = explain_resp.get("payload", {})
                                self_answer = payload.get("explanation") or payload.get("answer") or payload.get("text")
                                self_confidence = 0.90
                                answer_source = "explain_last"
                            else:
                                # Provide a generic but honest answer
                                self_answer = "I don't have a trace of my previous answer in memory. Generally, I use my reasoning brain to process questions, check memory for relevant facts, and generate responses. I only use Teacher when I don't have information in memory."

                        elif self_kind == "introduction":
                            # ============================================================
                            # SELF_INTRODUCTION - Dynamic self-introduction from introspection
                            # ============================================================
                            print(f"[SELF_INTENT_GATE] Routing to self_model.SELF_INTRODUCTION")
                            from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api

                            # Check if there's a target platform (e.g., "to grok")
                            from brains.cognitive.integrator.routing_safety import get_introduction_target
                            target_platform = get_introduction_target(orig_q)

                            intro_resp = self_model_api({
                                "op": "SELF_INTRODUCTION",
                                "payload": {
                                    "detail_level": "full"
                                }
                            })

                            if intro_resp.get("ok"):
                                payload = intro_resp.get("payload", {})
                                self_answer = payload.get("introduction_text", "I am Maven.")
                                self_confidence = 1.0
                                answer_source = "self_introduction"

                                # If target platform specified, send to that platform
                                if target_platform:
                                    print(f"[SELF_INTENT_GATE] Target platform: {target_platform}")
                                    try:
                                        if target_platform == "grok":
                                            # Import and use grok tool to send the introduction
                                            from brains.tools_api import get_tool
                                            grok_tool = get_tool("x")
                                            if grok_tool and hasattr(grok_tool, "reply_to_grok"):
                                                print(f"[SELF_INTENT_GATE] Sending introduction to Grok...")
                                                grok_result = grok_tool.reply_to_grok(self_answer)
                                                if grok_result.get("ok"):
                                                    self_answer = f"I've introduced myself to Grok: {self_answer}"
                                                else:
                                                    self_answer = f"{self_answer}\n\n(Note: Could not send to Grok: {grok_result.get('error', 'unknown error')})"
                                            else:
                                                self_answer = f"{self_answer}\n\n(Note: Grok tool not available for sending)"
                                    except Exception as e:
                                        print(f"[SELF_INTENT_GATE] Error sending to {target_platform}: {e}")
                                        self_answer = f"{self_answer}\n\n(Note: Could not send to {target_platform}: {str(e)[:50]})"

                        # ============================================================
                        # OTHER SELF QUERIES - Use self_model (identity, code, memory, etc.)
                        # ============================================================
                        else:
                            print(f"[SELF_INTENT_GATE] Routing to self_model for self_kind={self_kind}")
                            from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api

                            self_resp = self_model_api({
                                "op": "QUERY_SELF",
                                "payload": {
                                    "query": orig_q,
                                    "self_kind": self_kind,
                                    "self_mode": self_mode
                                }
                            })

                            if self_resp.get("ok"):
                                payload = self_resp.get("payload", {})
                                self_answer = payload.get("text") or payload.get("answer")
                                self_confidence = payload.get("confidence", 0.95)
                                answer_source = "self_model"

                        # Return the answer if we got one
                        if self_answer:
                            print(f"[SELF_INTENT_GATE] Got answer from {answer_source}")
                            _store_qa_memory(orig_q, self_answer, self_confidence)

                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "TRUE",
                                    "mode": f"{answer_source.upper()}_ANSWER",
                                    "confidence": self_confidence,
                                    "routing_order": {"target_bank": None, "action": None},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": self_answer,
                                    "weights_used": {"rule": f"self_gate_{self_kind}_v1"},
                                    "self_intent_kind": self_kind,
                                    "answer_source": answer_source
                                }
                            }

                    except Exception as e:
                        print(f"[SELF_INTENT_GATE_ERROR] Failed to get answer for {self_kind}: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()

                    # If we failed to get an answer, return UNANSWERED rather than calling Teacher
                    print(f"[SELF_INTENT_GATE] Handler failed for {self_kind}, returning UNANSWERED (NOT calling Teacher)")
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "verdict": "UNANSWERED",
                            "mode": "SELF_QUERY_NO_MODEL",
                            "confidence": 0.0,
                            "routing_order": {"target_bank": None, "action": None},
                            "supported_by": [],
                            "contradicted_by": [],
                            "answer": f"I could not answer this {self_kind} question from my internal systems.",
                            "weights_used": {"rule": "self_gate_v1"},
                            "self_intent_kind": self_kind
                        }
                    }
            except Exception as e:
                # If self-intent check fails, log but continue (fail open)
                print(f"[SELF_INTENT_GATE_ERROR] Pattern check failed: {str(e)[:100]}")
                pass

            # --------------------------------------------------------------------
            # USER_PERSONAL_GATE: Block Teacher for user personal questions
            # --------------------------------------------------------------------
            # User personal questions MUST be answered by personal brain from
            # memory ONLY, NEVER by Teacher. Teacher would hallucinate facts
            # about the user (wrong name, wrong preferences, etc.).
            #
            # If a user personal intent is detected, skip Teacher entirely and
            # route to personal brain for the canonical answer from memory.
            try:
                q_lower = str(orig_q or "").strip().lower()

                # Normalize text: lowercase, strip punctuation, collapse spaces
                normalized = q_lower
                for punct in ['.', ',', '!', '?', ':', ';']:
                    normalized = normalized.replace(punct, ' ')
                normalized = ' '.join(normalized.split())

                # USER IDENTITY patterns - who am I, what's my name
                user_identity_patterns = [
                    "who am i",
                    "who do you think i am",
                    "what is my name",
                    "what's my name",
                    "whats my name",
                    "do you know my name",
                    "do you remember my name",
                ]

                # USER PREFERENCE patterns - what do I like, what are my preferences
                user_preference_patterns = [
                    "what do i like",
                    "what are my preferences",
                    "what are my favorite",
                    "what are my favourite",
                    "what color do i like",
                    "what colour do i like",
                    "what colors do i like",
                    "what colours do i like",
                    "what animal do i like",
                    "what animals do i like",
                    "what food do i like",
                    "what foods do i like",
                    "tell me my preferences",
                    "list my preferences",
                ]

                is_user_personal = False
                question_type = None

                # Check for user identity questions
                for substring in user_identity_patterns:
                    if substring in normalized:
                        is_user_personal = True
                        question_type = "who_am_i"
                        print(f"[USER_PERSONAL_GATE] Detected user identity question, blocking Teacher")
                        print(f"[USER_PERSONAL_GATE] Matched substring: '{substring}'")
                        print(f"[USER_PERSONAL_GATE] Question: '{orig_q[:60]}...'")
                        break

                # If not identity, check for specific preference questions
                if not is_user_personal:
                    if "what color" in normalized or "what colour" in normalized:
                        is_user_personal = True
                        question_type = "what_color"
                        print(f"[USER_PERSONAL_GATE] Detected color preference question, blocking Teacher")
                    elif "what animal" in normalized:
                        is_user_personal = True
                        question_type = "what_animal"
                        print(f"[USER_PERSONAL_GATE] Detected animal preference question, blocking Teacher")
                    elif "what food" in normalized:
                        is_user_personal = True
                        question_type = "what_food"
                        print(f"[USER_PERSONAL_GATE] Detected food preference question, blocking Teacher")
                    elif any(substring in normalized for substring in user_preference_patterns):
                        is_user_personal = True
                        question_type = "what_do_i_like"
                        print(f"[USER_PERSONAL_GATE] Detected general preference question, blocking Teacher")

                if is_user_personal and question_type:
                    print(f"[USER_PERSONAL_GATE] question_type={question_type}, routing to personal brain instead")

                    # Call personal brain for answer from memory only
                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api

                        personal_resp = personal_api({
                            "op": "ANSWER_PERSONAL_QUESTION",
                            "payload": {
                                "question_type": question_type
                            }
                        })

                        if personal_resp.get("ok"):
                            payload = personal_resp.get("payload", {})
                            personal_answer = payload.get("answer")
                            personal_found = payload.get("found", False)
                            personal_confidence = 0.95 if personal_found else 0.5

                            if personal_answer:
                                print(f"[USER_PERSONAL_GATE] Got answer from personal brain")

                                # Store personal answer in QA memory for future reference
                                _store_qa_memory(orig_q, personal_answer, personal_confidence)

                                return {
                                    "ok": True,
                                    "op": op,
                                    "mid": mid,
                                    "payload": {
                                        "verdict": "TRUE" if personal_found else "UNANSWERED",
                                        "mode": "PERSONAL_MEMORY_ANSWER",
                                        "confidence": personal_confidence,
                                        "routing_order": {"target_bank": "personal", "action": None},
                                        "supported_by": [],
                                        "contradicted_by": [],
                                        "answer": personal_answer,
                                        "weights_used": {"rule": "personal_memory_v1"}
                                    }
                                }
                    except Exception as e:
                        print(f"[USER_PERSONAL_GATE_ERROR] Failed to call personal brain: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()
                        # Fall through to return UNANSWERED rather than calling Teacher

                    # If personal brain fails, return UNANSWERED rather than calling Teacher
                    print(f"[USER_PERSONAL_GATE] personal brain unavailable, returning UNANSWERED (NOT calling Teacher)")
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "verdict": "UNANSWERED",
                            "mode": "PERSONAL_QUERY_NO_DATA",
                            "confidence": 0.0,
                            "routing_order": {"target_bank": "personal", "action": None},
                            "supported_by": [],
                            "contradicted_by": [],
                            "answer": "I don't have that information about you yet. You can tell me by sharing your preferences!",
                            "weights_used": {"rule": "personal_gate_v1"}
                        }
                    }
            except Exception as e:
                # If user-personal check fails, log but continue (fail open)
                print(f"[USER_PERSONAL_GATE_ERROR] Pattern check failed: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                pass

            # MEMORY RETRIEVAL CHECK: Before calling the teacher, check if we
            # already have a stored answer from a previous teacher interaction
            try:
                stored_answer = _qa_memory_lookup(orig_q)
                if stored_answer:
                    # Found an answer in memory - return it without calling teacher
                    print(f"[MEMORY_HIT] Found stored answer, skipping teacher call")
                    try:
                        conf_mem = 0.85 + (aff_val * 0.05 + aff_arousal * 0.03)
                    except Exception:
                        conf_mem = 0.85
                    conf_mem = max(0.0, min(1.0, conf_mem))

                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "verdict": "KNOWN_ANSWER",
                            "mode": "MEMORY_RETRIEVED",
                            "confidence": conf_mem,
                            "routing_order": {"target_bank": None, "action": None},
                            "supported_by": [],
                            "contradicted_by": [],
                            "answer": stored_answer,
                            "weights_used": {"rule": "qa_memory_v1"}
                        }
                    }
            except Exception as e:
                # If memory lookup fails, continue to teacher
                print(f"[MEMORY_CHECK_ERROR] {str(e)[:100]}")
                pass

            # Check if routing has already been learned for this question
            # If routing exists, we've already asked the teacher about this question before
            routing_already_learned = False
            try:
                from brains.cognitive.memory_librarian.service.librarian_memory import (
                    retrieve_routing_rule_for_question
                )
                existing_routing = retrieve_routing_rule_for_question(orig_q, threshold=0.3)
                if existing_routing:
                    routing_already_learned = True
                    print(f"[ROUTING_SKIP_TEACHER] Question already learned, skipping teacher call")
            except Exception as e:
                print(f"[ROUTING_CHECK_ERROR] {str(e)[:100]}")
                # Continue with teacher call if check fails
                pass

            # LEARNING MODE GATE: Only call Teacher if learning_mode allows LLM access
            # OFFLINE mode blocks all LLM calls; TRAINING/SHADOW modes allow Teacher calls
            llm_allowed = learning_mode != LearningMode.OFFLINE

            if teacher_enabled and not routing_already_learned and llm_allowed:
                # HARD SAFETY CHECK: Block Teacher for self-queries
                # This is defense-in-depth; the self-intent gate should have already returned
                try:
                    q_lower = str(orig_q or "").strip().lower()
                    normalized = q_lower
                    for punct in ['.', ',', '!', '?', ':', ';']:
                        normalized = normalized.replace(punct, ' ')
                    normalized = ' '.join(normalized.split())

                    # Check self-memory substrings (both stats and scan patterns)
                    self_memory_check = [
                        # Stats patterns
                        "how many facts have you learned",
                        "how much have you learned",
                        "how many facts do you know",
                        "what have you learned so far",
                        "what do you remember",
                        # Scan/health patterns
                        "scan your memory system",
                        "scan memory system",
                        "scan your memory",
                        "scan memory",
                        "diagnose your memory",
                        "diagnose memory",
                        "check your memory",
                        "check your memory system"
                    ]

                    for substring in self_memory_check:
                        if substring in normalized:
                            print(f"[SELF_INTENT_GATE_ERROR] Teacher call blocked for self-memory question: '{orig_q[:60]}...'")
                            print(f"[SELF_INTENT_GATE_ERROR] Matched: '{substring}'")
                            print(f"[SELF_INTENT_GATE_ERROR] Self-memory questions MUST NOT go to Teacher!")
                            # Return UNANSWERED instead of calling Teacher
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "UNANSWERED",
                                    "mode": "BLOCKED_SELF_MEMORY",
                                    "confidence": 0.0,
                                    "routing_order": {"target_bank": None, "action": None},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": "Error: self-memory question reached Teacher path. Please report this bug.",
                                    "weights_used": {"rule": "self_gate_safety_v1"}
                                }
                            }

                    # ================================================================
                    # CAPABILITY FALLBACK BLOCK: Block Teacher for capability queries
                    # ================================================================
                    # These questions MUST be answered by capability_snapshot, NOT Teacher.
                    # If we reach this point, the earlier self-intent gate missed it.
                    capability_block_patterns = [
                        "can you browse the web",
                        "can you search the web",
                        "can you control other programs",
                        "can you control programs",
                        "control other programs",
                        "can you read files",
                        "can you change files",
                        "can you read or change files",
                        "read or change files",
                        "what upgrades do you need",
                        "what upgrade do you need",
                        "scan your code",
                        "scan code",
                        "scan your codebase",
                        "can you write a story",
                        "can you create a story",
                        "what can you do",
                        "what are your capabilities",
                    ]

                    for substring in capability_block_patterns:
                        if substring in normalized:
                            print(f"[CAPABILITY_GATE] Teacher call blocked for capability question: '{orig_q[:60]}...'")
                            print(f"[CAPABILITY_GATE] Matched: '{substring}'")
                            print(f"[CAPABILITY_GATE] Routing to capability_snapshot instead!")
                            # Route to capability_snapshot
                            try:
                                from capabilities import answer_capability_question
                                cap_result = answer_capability_question(orig_q)
                                if cap_result and cap_result.get("answer"):
                                    return {
                                        "ok": True,
                                        "op": op,
                                        "mid": mid,
                                        "payload": {
                                            "verdict": "KNOWN_CAPABILITY",
                                            "mode": "CAPABILITY_SNAPSHOT",
                                            "confidence": 0.99,
                                            "routing_order": {"target_bank": None, "action": None},
                                            "supported_by": [],
                                            "contradicted_by": [],
                                            "answer": cap_result.get("answer"),
                                            "weights_used": {"rule": "capability_gate_v2"}
                                        }
                                    }
                            except Exception as cap_err:
                                print(f"[CAPABILITY_GATE] capability_snapshot failed: {cap_err}")
                            # Return a generic capability response if snapshot fails
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "KNOWN_CAPABILITY",
                                    "mode": "CAPABILITY_BLOCKED",
                                    "confidence": 0.9,
                                    "routing_order": {"target_bank": None, "action": None},
                                    "supported_by": [],
                                    "contradicted_by": [],
                                    "answer": "This is a question about my capabilities. Please check my capability snapshot for accurate information.",
                                    "weights_used": {"rule": "capability_gate_fallback_v1"}
                                }
                            }
                except Exception:
                    pass  # Don't let safety check crash the pipeline

                try:
                    # Import teacher brain
                    from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api  # type: ignore

                    # For follow-up questions, use enhanced query with topic context
                    teacher_question = orig_q
                    request_context = payload.get("context") or {}

                    # Check if this is a continuation query and enhance with conversation context
                    try:
                        if is_continuation(orig_q):
                            conv_context = get_conversation_context()
                            last_topic = conv_context.get("last_topic")

                            if last_topic:
                                teacher_question = enhance_query_with_context(orig_q, conv_context)
                                request_context["is_continuation"] = True
                                request_context["enhanced_query"] = teacher_question
                                request_context["last_topic"] = last_topic
                                print(f"[FOLLOW_UP] Detected continuation, enhanced query: {teacher_question[:60]}...")
                    except Exception as e:
                        # If context retrieval fails, continue with original query
                        print(f"[FOLLOW_UP] Context enhancement failed: {e}")

                    # Call the teacher with the question and any retrieved context
                    print(f"[DEBUG] Calling teacher for: {teacher_question[:60]}...")
                    teacher_resp = teacher_api({
                        "op": "TEACH",
                        "payload": {
                            "question": teacher_question,
                            "context": request_context,
                            "retrieved_facts": results  # Pass weak evidence to teacher
                        }
                    })

                    teacher_ok = teacher_resp.get("ok")
                    print(f"[DEBUG] Teacher response ok={teacher_ok}")

                    if teacher_ok:
                        teacher_payload = teacher_resp.get("payload") or {}
                        teacher_answer = teacher_payload.get("answer")
                        candidate_facts = teacher_payload.get("candidate_facts") or []

                        # Vet and store each fact through TruthClassifier
                        for fact in candidate_facts:
                            try:
                                fact_statement = fact.get("statement", "")
                                if not fact_statement:
                                    continue

                                # Classify the fact using TruthClassifier
                                classification = TruthClassifier.classify(
                                    content=fact_statement,
                                    confidence=0.7,  # Teacher facts start with moderate confidence
                                    evidence=None
                                )

                                # Only store non-RANDOM facts
                                if classification["type"] != "RANDOM" and classification["allow_memory_write"]:
                                    # Store to DOMAIN brains only (never to cognitive brains like reasoning)
                                    try:
                                        fact_type = fact.get("type", "world_fact")
                                        metadata = {
                                            "kind": "learned_fact",
                                            "source": "llm_teacher",
                                            "original_question": orig_q,
                                            "confidence": classification["confidence"],
                                            "truth_type": classification["type"],
                                            "fact_type": fact_type
                                        }

                                        # Store to domain brain(s)
                                        stored_to = _store_fact_to_domain(
                                            fact_statement=fact_statement,
                                            metadata=metadata,
                                            fact_type=fact_type
                                        )

                                        if stored_to > 0:
                                            teacher_facts_stored += 1

                                            # Log telemetry for successful storage
                                            try:
                                                from brains.cognitive.teacher.service.teacher_brain import _store_telemetry
                                                _store_telemetry("facts_stored", {"fact": fact_statement[:100]})
                                            except Exception:
                                                pass
                                    except Exception:
                                        # Ignore storage errors for individual facts
                                        pass
                            except Exception:
                                # Skip this fact if processing fails
                                continue

                        # If teacher provided an answer, store the Q&A pair
                        if teacher_answer:
                            _store_qa_memory(orig_q, teacher_answer, teacher_payload.get("confidence", 0.7))

                        # ROUTING LEARNING: After teacher answers, learn routing for this question
                        # FIX 1: Always learn routing when Teacher answers (verdict LEARNED), not just when facts stored
                        # This ensures future similar questions will query the right banks first
                        teacher_verdict = str(teacher_payload.get("verdict", "")).upper()
                        print(f"[DEBUG] teacher_verdict={teacher_verdict}, teacher_answer exists={bool(teacher_answer)}")

                        # If Teacher blocked the question as self-knowledge, do NOT store facts or learn routing
                        if teacher_verdict == "SELF_KNOWLEDGE_FORBIDDEN":
                            print(f"[TEACHER_BLOCKED_SELF] Teacher refused to teach self-facts (correct behavior)")
                            # Clear any answer that might have leaked through
                            teacher_answer = None
                            teacher_facts_stored = 0
                            # Do NOT learn routing for self-queries (they should use self_model)
                        elif teacher_verdict == "LEARNED" and teacher_answer:
                            try:
                                from brains.cognitive.memory_librarian.service.librarian_memory import (
                                    learn_routing_for_question
                                )

                                # Define available banks that facts could be stored in
                                # Use ALL domain banks from the tree scan (not hardcoded)
                                available_banks = get_domain_brains()

                                # FIX 4: Log routing learning explicitly
                                print(f"[ROUTING_LEARNING] Learning routes for: {orig_q[:60]}...")

                                # Ask Teacher for routing suggestions and store them
                                routing_result = learn_routing_for_question(
                                    question=orig_q,
                                    available_banks=available_banks,
                                    context=payload.get("context")
                                )

                                # FIX 4: Log the learned routing rule
                                if routing_result:
                                    routes = routing_result.get("routes", [])
                                    aliases = routing_result.get("aliases", [])
                                    print(f"[ROUTING_LEARNED_FOR] {orig_q[:60]}")
                                    print(f"  routes={[r.get('bank') for r in routes]}")
                                    print(f"  aliases={aliases}")
                                else:
                                    print(f"[ROUTING_LEARNING_FAILED] No routing learned for: {orig_q[:60]}")
                            except Exception as e:
                                # FIX 4: Log routing learning failures
                                print(f"[ROUTING_LEARNING_ERROR] {str(e)[:100]}")
                                # Don't fail the whole response if routing learning fails
                                pass
                except Exception as e:
                    # If teacher fails, fall through to UNANSWERED
                    print(f"[DEBUG] Teacher exception: {str(e)[:200]}")
                    import traceback
                    traceback.print_exc()
                    pass
            elif not llm_allowed:
                # Log that LLM was blocked due to OFFLINE mode
                print(f"[REASONING] LLM blocked (learning_mode={learning_mode}), using offline reasoning only")

            # If teacher provided an answer, return it
            if teacher_answer:
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "LEARNED",
                        "mode": "TEACHER_ANSWER",
                        "confidence": 0.7,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "answer": teacher_answer,
                        "teacher_facts_stored": teacher_facts_stored,
                        "weights_used": {"rule": "teacher_v1"}
                    }
                }

            # Otherwise, truly no answer available
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": "UNANSWERED",
                    "mode": "QUESTION_INPUT",
                    "confidence": 0.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "weights_used": {"rule": "question_answer_v1"}
                }
            }

        # --- Primitive safeguard: questions are not facts ---
        if _is_question_text(content):
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": "UNANSWERED",
                    "mode": "QUESTION_INPUT",
                    "confidence": 0.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "weights_used": {"rule": "primitive_reason_v2"}
                }
            }
        # Weighted confidence calculation for factual statements.  Begin by scoring
        # the proposed fact against any retrieved evidence.  This helper returns
        # 0.8 for a direct match and 0.4 for no match.  Using the evidence score
        # as the base ensures that statements without supporting evidence are
        # treated conservatively (UNKNOWN) rather than being automatically
        # labelled as theories.
        evidence = payload.get("evidence") or {}
        conf = _score_evidence(proposed, evidence)
        # Apply a penalty for speculative or hedging language supplied by callers
        try:
            pen = float(proposed.get("confidence_penalty", 0.0))
        except Exception:
            pen = 0.0
        conf -= pen
        # Apply affect adjustment: valence and arousal nudge the confidence up
        # or down slightly.  Positive values raise confidence and negative
        # values lower it.
        try:
            conf = conf + aff_val * 0.05 + aff_arousal * 0.03
        except Exception:
            pass
        # Track supporting or contradicting evidence for provenance
        supported_by: List[str] = []
        contradicted_by: List[str] = []
        # Incorporate evidence: matching records slightly boost confidence, contradictions reduce it
        try:
            for it in (evidence.get("results") or []):
                if not isinstance(it, dict):
                    continue
                c = str(it.get("content", "")).strip().lower()
                proposed_c = str(proposed.get("content", "")).strip().lower()
                record_type = str(it.get("type", "")).lower()
                if c and (proposed_c == c or proposed_c in c or c in proposed_c):
                    conf += 0.05
                    rec_id = it.get("id")
                    if rec_id:
                        supported_by.append(rec_id)
                elif record_type == "contradiction":
                    conf -= 0.1
                    rec_id = it.get("id")
                    if rec_id:
                        contradicted_by.append(rec_id)
        except Exception:
            pass
        # Adjust confidence based on recent success rate (learned bias).  We use
        # the reasoning brain's own STM as the root for computing the success
        # average.  A modest adjustment (±0.15) nudges the confidence toward
        # better performance without dominating the evidence score.
        from api.memory import compute_success_average  # type: ignore
        try:
            learned = compute_success_average(REASONING_ROOT, n=50)
        except Exception:
            learned = 0.0
        conf += 0.15 * learned
        # Clamp confidence to the valid range [0.0, 1.0]
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        # Determine verdict based on dynamically adjusted thresholds.  The true
        # and theory thresholds are biased by the affect metrics: positive
        # valence and high arousal lower the thresholds (faster acceptance),
        # negative valence raises them (cautious acceptance).  Limits ensure
        # thresholds remain within reasonable bounds.
        try:
            adjust = 0.05 * aff_val + 0.03 * aff_arousal
        except Exception:
            adjust = 0.0
        true_threshold = 0.85 - adjust
        theory_threshold = 0.70 - adjust
        # Clamp thresholds to sensible ranges
        if true_threshold < 0.60:
            true_threshold = 0.60
        if true_threshold > 0.90:
            true_threshold = 0.90
        if theory_threshold < 0.50:
            theory_threshold = 0.50
        if theory_threshold > 0.85:
            theory_threshold = 0.85
        if conf >= true_threshold:
            verdict = "TRUE"
            mode = "VERIFIED"
        elif conf >= theory_threshold:
            verdict = "THEORY"
            mode = "EDUCATED_GUESS"
        else:
            verdict = "UNKNOWN"
            mode = "NO_EVIDENCE"
        # If verdict is unknown, request a targeted memory search via the message bus.
        if verdict == "UNKNOWN":
            try:
                # Import send lazily to avoid circular imports.
                from brains.cognitive.message_bus import send as _mb_send  # type: ignore
                # Construct simple domain hints from the question or content.
                hints: List[str] = []
                try:
                    q_text = str(orig_q or content or "").lower().split()
                    for w in q_text:
                        # Use alphabetic tokens as domain hints (e.g. keywords)
                        if w.isalpha():
                            hints.append(w)
                            if len(hints) >= 2:
                                break
                except Exception:
                    hints = []
                _mb_send({
                    "from": "reasoning",
                    "to": "memory",
                    "type": "SEARCH_REQUEST",
                    "domains": hints or ["general"],
                    "confidence_threshold": 0.7,
                })
            except Exception:
                # Silently ignore message bus failures
                pass
        # Determine routing order: only store TRUE facts into domain banks.
        route_bank = _route_for(conf)
        routing_order = {
            "target_bank": route_bank,
            "action": "STORE" if verdict == "TRUE" else "SKIP"
        }
        # Produce a simple reasoning trace explaining how the confidence was evaluated.
        # In addition to the generic message, append a note when a self‑identity
        # query yields no evidence.  This transparency helps users understand
        # why an answer may be unknown.  See upgrade notes on reasoning
        # transparency for more context.
        try:
            trace_msg = (
                f"Evaluated confidence {conf:.2f} against thresholds (TRUE≥{true_threshold:.2f}, THEORY≥{theory_threshold:.2f})."
            )
        except Exception:
            trace_msg = "Confidence evaluation details unavailable."
        # Append introspective explanation for self‑identity queries when
        # confidence is insufficient to produce a factual answer.  Detect
        # identity queries by looking for common phrases in the original
        # question.  Only append the note when the verdict is UNKNOWN.
        try:
            if verdict == "UNKNOWN":
                q_lower = str(orig_q or "").strip().lower()
                identity_patterns = [
                    "who are you",
                    "what is your name",
                    "what's your name",
                    "tell me about yourself",
                    "who you are",
                    "are you maven"
                ]
                for pat in identity_patterns:
                    if pat in q_lower:
                        trace_msg += " No self-definition found in memory."
                        break
        except Exception:
            # Ignore any errors when adding introspective notes
            pass
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "verdict": verdict,
                "mode": mode,
                "confidence": conf,
                "routing_order": routing_order,
                "supported_by": supported_by,
                "contradicted_by": contradicted_by,
                "weights_used": {"rule": "primitive_reason_v2"},
                "reasoning_trace": trace_msg,
            }
        }
    # Health endpoint just returns a status ok
    if op == "HEALTH":
        return {"ok": True, "op": op, "mid": mid, "payload": {"status": "ok"}}
    # EXECUTE_STEP: Phase 8 - Execute a reasoning/logic step
    if op == "EXECUTE_STEP":
        step = payload.get("step") or {}
        step_id = payload.get("step_id", 0)
        context = payload.get("context") or {}

        # Extract step details
        description = step.get("description", "")
        step_input = step.get("input") or {}
        task = step_input.get("task", description)

        # Execute reasoning step
        # For simplicity, return a structured reasoning result
        output = {
            "reasoning": f"Analyzed: {description}",
            "conclusion": f"Completed reasoning about: {task}",
            "task": task
        }

        return {"ok": True, "op": op, "mid": mid, "payload": {
            "output": output,
            "patterns_used": ["reasoning:logic"]
        }}

    return {"ok": False, "op": op, "mid": mid, "error": {"code": "UNSUPPORTED_OP", "message": op}}

# -----------------------------------------------------------------------------
# Attention bid interface
#
# The reasoning brain can request attention from the integrator by
# providing a bid via ``bid_for_attention``.  This function examines the
# current pipeline context and determines how urgently reasoning
# resources are needed.  It prioritises situations where contradictions
# need resolving or where a question lacks a clear answer but related
# facts are available.  For general questions it bids moderately, and
# otherwise returns a low default bid.  Errors result in a safe low
# priority.
def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Inspect reasoning verdict if it exists
        stage8 = ctx.get("stage_8_validation") or {}
        verdict = str(stage8.get("verdict", "")).upper()
        mode = str(stage8.get("mode", "")).upper()

        # -----------------------------------------------------------------
        # Continuation detection and context enrichment
        #
        # Detect if this is a follow-up query to determine whether reasoning
        # should expand previous analysis or start fresh.
        lang_info = ctx.get("stage_3_language", {}) or {}
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")
        conv_context = lang_info.get("conversation_context", {})

        # Skip reasoning for PREFERENCE and relationship queries - these don't need validation
        if verdict == "PREFERENCE" or mode in {"PREFERENCE_QUERY", "RELATIONSHIP_QUERY"}:
            routing_hint = create_routing_hint(
                brain_name="reasoning",
                action="skip",
                confidence=0.05,
                context_tags=["preference", "relationship", "no_validation"],
                metadata={"is_continuation": is_cont}
            )
            return {
                "brain_name": "reasoning",
                "priority": 0.05,
                "reason": "preference_or_relationship_skip",
                "evidence": {"routing_hint": routing_hint},
            }

        # High priority if contradictions have been detected.  We treat
        # ``CONTRADICTED_EVIDENCE`` mode as a proxy for contradictions or
        # a THEORY verdict to indicate disputed evidence.
        if mode == "CONTRADICTED_EVIDENCE" or verdict == "THEORY":
            routing_hint = create_routing_hint(
                brain_name="reasoning",
                action="resolve_contradiction",
                confidence=0.95,
                context_tags=["contradiction", "theory", "validation"],
                metadata={"is_continuation": is_cont, "last_topic": conv_context.get("last_topic", "")}
            )
            return {
                "brain_name": "reasoning",
                "priority": 0.95,
                "reason": "contradiction_detected",
                "evidence": {"routing_hint": routing_hint},
            }

        # Determine if the current input is a question
        st_type = str(
            lang_info.get("type")
            or lang_info.get("storable_type")
            or lang_info.get("intent")
            or ""
        ).upper()

        # Check if memory retrieval found any results
        mem_results = (ctx.get("stage_2R_memory") or {}).get("results", [])
        has_related = bool(mem_results)

        # When the verdict is UNKNOWN or UNANSWERED and there are related facts
        # available, reasoning can attempt an inference.  Bid moderately high.
        if verdict in {"UNANSWERED", "UNKNOWN"} and has_related:
            # For continuations, expand previous reasoning; for new questions, fresh analysis
            if is_cont:
                routing_hint = create_routing_hint(
                    brain_name="reasoning",
                    action="expand_previous_reasoning",
                    confidence=0.80,
                    context_tags=["follow_up", "expansion", continuation_intent, "inference"],
                    metadata={
                        "last_topic": conv_context.get("last_topic", ""),
                        "continuation_type": continuation_intent,
                        "has_related_facts": True
                    }
                )
            else:
                routing_hint = create_routing_hint(
                    brain_name="reasoning",
                    action="fresh_inference",
                    confidence=0.75,
                    context_tags=["new_question", "inference"],
                    metadata={"has_related_facts": True}
                )
            return {
                "brain_name": "reasoning",
                "priority": 0.75,
                "reason": "inference_possible",
                "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
            }

        # For general questions bid medium to analyse the question
        if st_type == "QUESTION":
            if is_cont:
                routing_hint = create_routing_hint(
                    brain_name="reasoning",
                    action="expand_analysis",
                    confidence=0.55,
                    context_tags=["follow_up", "expansion", continuation_intent, "question"],
                    metadata={
                        "last_topic": conv_context.get("last_topic", ""),
                        "continuation_type": continuation_intent
                    }
                )
            else:
                routing_hint = create_routing_hint(
                    brain_name="reasoning",
                    action="analyze_question",
                    confidence=0.50,
                    context_tags=["new_question", "analysis"],
                    metadata={}
                )
            return {
                "brain_name": "reasoning",
                "priority": 0.50,
                "reason": "question_analysis",
                "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
            }

        # Low default priority for all other cases
        routing_hint = create_routing_hint(
            brain_name="reasoning",
            action="default",
            confidence=0.15,
            context_tags=["default"],
            metadata={"is_continuation": is_cont}
        )
        return {
            "brain_name": "reasoning",
            "priority": 0.15,
            "reason": "default",
            "evidence": {"routing_hint": routing_hint},
        }
    except Exception:
        return {
            "brain_name": "reasoning",
            "priority": 0.15,
            "reason": "default",
            "evidence": {},
        }

def handle(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Brain contract entry point.

    Implements strategy-first reasoning with LLM fallback:
    1. Read learning_mode from context (defaults to OFFLINE)
    2. Classify the problem type
    3. Extract domain from context
    4. Load strategies from lessons
    5. Attempt to apply a matching strategy
    6. If TRAINING mode: fall back to reasoning_llm_lesson for LLM-based learning
    7. If OFFLINE mode: return structured "not confident" response
    8. Legacy fallback to _handle_impl for edge cases

    Args:
        context: Standard brain context dict

    Returns:
        Result dict
    """
    # =========================================================================
    # TIME QUERY GUARD: Refuse to answer time queries if time tool is available
    # =========================================================================
    # Time questions MUST be handled by the time_now tool, not by reasoning/LLM.
    # This guard prevents the LLM from giving incorrect or fabricated time info.
    try:
        from brains.cognitive.sensorium.semantic_normalizer import is_time_query
        from capabilities import is_capability_enabled

        question_text = context.get("user_query", "") or context.get("query", "")
        if question_text and is_time_query(str(question_text)):
            if is_capability_enabled("time"):
                print(f"[REASONING] ⛔ TIME QUERY BLOCKED: '{question_text}' - use time_now tool instead")
                return {
                    "ok": True,
                    "op": context.get("op", "EVALUATE_FACT"),
                    "mid": context.get("mid"),
                    "payload": {
                        "answer": None,
                        "use_tool": "time_now",
                        "source": "tool_redirect",
                        "reason": "Time queries must use the time_now tool for accurate information",
                    }
                }
    except Exception as e:
        print(f"[REASONING] Time guard check failed: {e}")

    # 1. Read learning_mode from context (default to TRAINING for LLM learning)
    learning_mode = context.get("learning_mode", LearningMode.TRAINING)

    # 2. Classify the reasoning problem type
    problem_type = classify_reasoning_problem(context)

    # 3. Extract domain from context
    domain = extract_domain(context)

    # 4. Extract question from context for potential LLM fallback
    question = ""
    if "user_query" in context:
        question = str(context["user_query"])
    elif "payload" in context and isinstance(context["payload"], dict):
        question = str(context["payload"].get("query_text", ""))
        if not question:
            question = str(context["payload"].get("original_query", ""))
    elif "query_text" in context:
        question = str(context["query_text"])

    # For continuations, prefer the enhanced query which includes topic context
    if context.get("is_continuation") and context.get("enhanced_query"):
        question = str(context["enhanced_query"])

    print(f"[REASONING] handle() called: learning_mode={learning_mode}, problem_type={problem_type}, domain={domain}")

    # 5. Load strategies from lessons (refresh strategy table)
    load_strategies_from_lessons(context)

    # 6. Select a strategy for this problem type and domain
    strategy = select_strategy(problem_type, domain)

    # 7. Strategy-first branch: try to apply strategy before falling back
    STRATEGY_CONFIDENCE_THRESHOLD = 0.5

    if strategy is not None:
        print(f"[REASONING] Found strategy '{strategy.get('name')}' for problem_type={problem_type}, domain={domain}")
        result, strategy_conf = apply_strategy(strategy, context)

        # If strategy produced a valid result with sufficient confidence, use it
        if result is not None and strategy_conf >= STRATEGY_CONFIDENCE_THRESHOLD:
            # Check if result has an actual answer (not just metadata)
            if result.get("answer") or result.get("verdict"):
                print(f"[REASONING] Using strategy result (confidence={strategy_conf})")
                # Wrap in standard brain response format
                return {
                    "ok": True,
                    "op": context.get("op", "EVALUATE_FACT"),
                    "mid": context.get("mid"),
                    "payload": {
                        **result,
                        "source": "strategy",
                        "strategy_name": strategy.get("name"),
                        "problem_type": problem_type,
                        "domain": domain,
                        "learning_mode": str(learning_mode.value) if hasattr(learning_mode, 'value') else str(learning_mode),
                    }
                }
            else:
                print(f"[REASONING] Strategy result has no answer, proceeding to fallback")
        else:
            print(f"[REASONING] Strategy confidence too low ({strategy_conf}), proceeding to fallback")
    else:
        print(f"[REASONING] No strategy found for problem_type={problem_type}, domain={domain}")

    # 8. LLM fallback: only in TRAINING mode, use reasoning_llm_lesson
    if learning_mode == LearningMode.TRAINING:
        print(f"[REASONING] TRAINING mode: calling reasoning_llm_lesson for question: {question[:60]}...")
        lesson = reasoning_llm_lesson(context, question, learning_mode)

        if lesson and lesson.get("answer"):
            print(f"[REASONING] reasoning_llm_lesson produced answer, returning it")
            return {
                "ok": True,
                "op": context.get("op", "EVALUATE_FACT"),
                "mid": context.get("mid"),
                "payload": {
                    "verdict": "LEARNED",
                    "answer": lesson.get("answer"),
                    "source": "reasoning_llm_lesson",
                    "problem_type": problem_type,
                    "domain": domain,
                    "lesson_topic": lesson.get("topic", ""),
                    "confidence": lesson.get("confidence", 0.7),
                    "learning_mode": str(learning_mode.value) if hasattr(learning_mode, 'value') else str(learning_mode),
                }
            }
        else:
            print(f"[REASONING] reasoning_llm_lesson did not produce answer, falling back to _handle_impl")

    # 9. OFFLINE mode fallback: return structured "not confident" response
    elif learning_mode == LearningMode.OFFLINE:
        # In OFFLINE mode with no strategy, we cannot call LLM
        # Check if _handle_impl can handle this without LLM (it has its own gating)
        # But first try to provide a structured response
        print(f"[REASONING] OFFLINE mode: no strategy available, attempting offline reasoning via _handle_impl")
        # Fall through to _handle_impl which has LLM gating built in

    # 10. SHADOW mode: treat like OFFLINE for now (use strategies only)
    elif learning_mode == LearningMode.SHADOW:
        print(f"[REASONING] SHADOW mode: using offline reasoning (shadow LLM evaluation not yet implemented)")
        # Fall through to _handle_impl

    # 11. Legacy fallback to _handle_impl for any remaining cases
    # _handle_impl has its own learning_mode gating for the teacher_api call
    return _handle_impl(context)


# Brain contract alias
service_api = handle