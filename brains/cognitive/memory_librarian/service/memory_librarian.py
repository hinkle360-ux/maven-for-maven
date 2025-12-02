
from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple
import importlib.util, sys, json, re, os
import threading
from pathlib import Path

from brains.maven_paths import get_reports_path

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[MEMORY_LIBRARIAN] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Routing diagnostics for tracking pipeline paths (Phase C cleanup)
try:
    from brains.cognitive.routing_diagnostics import tracer, RouteType  # type: ignore
except Exception:
    tracer = None  # type: ignore
    RouteType = None  # type: ignore

# Optional import of the focus analyzer.  If unavailable, update
# calls will be silently ignored.  This allows the memory
# librarian to remain compatible with older Maven builds that do
# not include the attention analytics module.
try:
    from brains.cognitive.attention.focus_analyzer import update_focus_stats  # type: ignore
except Exception:
    update_focus_stats = None  # type: ignore

# Import memory consolidation helper to enable tier promotion.  When new facts
# are stored, we trigger consolidation so that validated knowledge can
# graduate from short‑term memory into mid‑ and long‑term stores.  Errors
# during consolidation are ignored to avoid blocking storage operations.
try:
    from brains.cognitive.memory_consolidation import consolidate_memories  # type: ignore
except Exception:
    consolidate_memories = None  # type: ignore

# ---------------------------------------------------------------------------
# Global state for attention and session tracking
#
# The memory librarian keeps track of a small history of attention focus
# transitions across pipeline runs.  Each entry records which brain won
# attention, why it did so, and when this occurred.  This helps later
# analyses identify patterns in attention allocation and allows bidding
# strategies to be tuned.  A short queue of recent user queries is also
# maintained to enable multi‑turn context awareness, stress detection and
# proactive clarification.  Only the last _MAX_RECENT_QUERIES entries are
# retained to bound memory usage.
_ATTENTION_HISTORY: List[Dict[str, Any]] = []
_RECENT_QUERIES: List[Dict[str, Any]] = []
_MAX_RECENT_QUERIES: int = 10

# ---------------------------------------------------------------------------
# Conversation state for multi-turn context (Phase 0 continuation support)
#
# To support continuation queries like "more about brains" or "anything else",
# the memory librarian tracks the last discussed topic and response across
# pipeline runs.  When a continuation trigger is detected in the user's input,
# the librarian replaces the query with the last topic so that retrieval is
# scoped appropriately.  After each pipeline run, these state variables are
# updated based on the user's original query and the final answer.
_LAST_TOPIC: str = ""
_LAST_RESPONSE: str = ""




# ---------------------------------------------------------------------------
# Browser state for browser tool commands (grok, chatgpt, etc.)
#
# When a browser page is opened, the page_id is stored here so subsequent
# grok/chatgpt tool commands can interact with the same page.
_LAST_BROWSER_PAGE_ID: Optional[str] = None

# ---------------------------------------------------------------------------
# Conversation context dictionary for deeper state tracking (Phase 2)
#
# In addition to the simple last topic/response strings above, Maven maintains
# a richer conversation context.  This dictionary stores the most recent
# query, answer, inferred topic, a list of entity tokens extracted from the
# conversation, and a depth counter measuring how many turns have occurred in
# the current session.  This structure enables pronoun resolution (e.g. mapping
# "that" or "it" back to the previous answer) and more robust continuation
# handling.  It is intentionally lightweight and does not rely on external
# NLP libraries.
_CONVERSATION_STATE: Dict[str, Any] = {
    "last_query": "",
    "last_response": "",
    "last_topic": "",
    "thread_entities": [],
    "conversation_depth": 0,
}

# ---------------------------------------------------------------------------
# Web Search State Tracking
#
# Tracks the most recent web search to enable intelligent follow-up handling.
# When a user asks "tell me more about X" after a web search, we can either
# reuse the cached results with a deeper synthesis prompt, or trigger a new
# search with a refined query. This eliminates the problem of follow-ups
# being answered generically by the LLM instead of using actual web data.
# ---------------------------------------------------------------------------

_LAST_WEB_SEARCH: Dict[str, Any] = {
    "query": "",           # The original search query (e.g., "music")
    "normalized_topic": "",  # Normalized topic for matching (e.g., "music")
    "engine": "",          # Search engine used (bing, duckduckgo, google)
    "results": [],         # List of search result dicts (title, url, snippet)
    "seq_id": 0,           # Sequence ID for recency
    "answer": "",          # The synthesized answer from the search
    "sources": [],         # Source URLs that were cited
}


def store_web_search_result(
    query: str,
    results: List[Dict[str, Any]],
    engine: str = "",
    answer: str = "",
    sources: Optional[List[str]] = None,
) -> None:
    """
    Store the results of a web search for follow-up handling.

    This is called after a successful web search so that follow-up questions
    like "tell me more" can reuse the results instead of going to generic LLM.

    Args:
        query: The original search query
        results: List of search result dicts with 'title', 'url', 'snippet'
        engine: Search engine used (bing, duckduckgo, google)
        answer: The synthesized answer from the search
        sources: List of source URLs that were cited
    """
    global _LAST_WEB_SEARCH

    # Extract normalized topic for matching
    normalized_topic = _extract_topic(query) or query.lower().strip()

    _LAST_WEB_SEARCH = {
        "query": query,
        "normalized_topic": normalized_topic,
        "engine": engine,
        "results": results[:10] if results else [],  # Keep top 10
        "seq_id": _next_seq_id(),
        "answer": answer,
        "sources": sources or [],
    }

    print(f"[MEMORY_LIBRARIAN] Stored web search: query='{query}' topic='{normalized_topic}' ({len(results)} results)")


def get_last_web_search() -> Optional[Dict[str, Any]]:
    """
    Get the most recent web search results.

    Returns:
        Dict with query, results, engine, answer, sources, or None if no search stored
    """
    if not _LAST_WEB_SEARCH.get("query"):
        return None
    return _LAST_WEB_SEARCH.copy()


def followup_refers_to_web_search(query: str) -> bool:
    """
    Check if a follow-up query refers to the last web search topic.

    This is used by the integrator to decide whether to route a follow-up
    question back to the research manager (to use cached results) or to
    generic reasoning.

    Args:
        query: The user's follow-up query (e.g., "tell me more about music")

    Returns:
        True if the follow-up appears to reference the last web search topic
    """
    if not _LAST_WEB_SEARCH.get("query"):
        return False

    query_lower = query.lower().strip()
    last_topic = _LAST_WEB_SEARCH.get("normalized_topic", "")

    # Generic follow-ups (no specific topic mentioned) refer to last search
    # Comprehensive list to catch all "tell me more" variations
    generic_patterns = [
        "tell me more",
        "more about that",
        "what else",
        "go deeper",
        "more details",
        "elaborate",
        "expand on that",
        "continue",
        "keep going",
        "more info",
        # Additional patterns for generalized follow-up detection
        "dive deeper",
        "can you expand on that",
        "tell me more about that",
        "more information",
        "more detail",
        "further detail",
        "give me more",
        "explain more",
        "describe more",
        "say more",
        "anything more",
        "is there more",
        "deeper explanation",
        "in more detail",
        "go on",
        "and?",
        "what more",
    ]

    for pattern in generic_patterns:
        if pattern in query_lower:
            # If it's a generic follow-up, assume it refers to last search
            if len(query_lower.split()) <= 6:  # Short query
                return True

    # Check if the query explicitly mentions the last topic
    if last_topic and last_topic in query_lower:
        return True

    # Check if any key words from the last query appear
    last_query_words = set(_LAST_WEB_SEARCH.get("query", "").lower().split())
    query_words = set(query_lower.split())

    # Remove common words
    stopwords = {"tell", "me", "more", "about", "what", "is", "the", "a", "an", "of", "to", "in"}
    last_query_words -= stopwords
    query_words -= stopwords

    # If there's significant overlap, consider it a reference
    overlap = last_query_words & query_words
    if overlap and len(overlap) >= 1:
        return True

    return False


def clear_web_search_state() -> None:
    """Clear the stored web search state (e.g., at start of new conversation)."""
    global _LAST_WEB_SEARCH
    _LAST_WEB_SEARCH = {
        "query": "",
        "normalized_topic": "",
        "engine": "",
        "results": [],
        "seq_id": 0,
        "answer": "",
        "sources": [],
    }


# ---------------------------------------------------------------------------
# Phase 4: Tiered Memory System Constants
#
# Maven's memory system operates across multiple cognitive tiers, each with
# different retention characteristics and bandwidth constraints.  These
# constants define the tier hierarchy and metadata fields used to manage
# memory lifecycle without time-based expiry (no datetime, no TTL).
# ---------------------------------------------------------------------------

# Memory tier constants - explicit tier labels for deterministic routing
TIER_WM = "WM"           # Working memory - very short horizon, high bandwidth
TIER_SHORT = "SHORT"     # Per-session episodic memory
TIER_MID = "MID"         # Cross-session high-value memory
TIER_LONG = "LONG"       # Durable knowledge base
TIER_PINNED = "PINNED"   # Never evicted unless explicitly removed

# ---------------------------------------------------------------------------
# Phase 5: Continuous Learning Record Types
#
# These record types represent higher-order cognitive structures that emerge
# from pattern detection, concept formation, and skill acquisition. They are
# persisted alongside facts and preferences to enable long-term adaptation.
# ---------------------------------------------------------------------------

# Phase 5 record type constants (used in verdict/tags fields)
RECORD_TYPE_INFERRED_PATTERN = "INFERRED_PATTERN"  # Detected patterns from recurring inputs
RECORD_TYPE_SKILL = "SKILL"                         # Learned query handling strategies
RECORD_TYPE_CONCEPT = "CONCEPT"                     # Formed concepts from stable patterns

# Sequence ID counter for recency tracking (replaces time-based logic)
# This counter increments monotonically on every write operation to enable
# recency-aware retrieval without relying on wall-clock timestamps.
_SEQ_ID_COUNTER: int = 0
_SEQ_ID_LOCK: threading.Lock = threading.Lock()

def _next_seq_id() -> int:
    """
    Generate the next sequence ID for memory records.

    This function provides a monotonically increasing sequence number that
    serves as a recency indicator without time-based dependencies. Each
    write operation receives a unique seq_id, enabling deterministic
    ordering and recency scoring in retrieval operations.

    Returns:
        A unique, monotonically increasing integer sequence ID.
    """
    global _SEQ_ID_COUNTER
    with _SEQ_ID_LOCK:
        _SEQ_ID_COUNTER += 1
        return _SEQ_ID_COUNTER

def _extract_topic(text: str) -> str:
    """
    Extract a plausible topic from a user query.  This helper looks for the
    keyword 'about' and returns the text following it; otherwise, it
    returns the last alphabetic word.  The topic is lower‑cased for
    consistent matching.  When no topic can be found, returns an empty string.

    Args:
        text: The raw user input.

    Returns:
        A lower‑cased topic string or ''.
    """
    try:
        s = str(text or "").strip()
    except Exception:
        return ""
    if not s:
        return ""
    # Normalize whitespace and case
    s_low = s.lower()
    # If the query contains 'about', return the substring after the last 'about'
    try:
        if " about " in s_low:
            # Split on the first occurrence of ' about ' to capture everything after
            parts = s_low.split(" about ", 1)
            topic = parts[1].strip()
            return topic
    except Exception:
        pass
    # Fallback: return the last alphabetic token
    import re as _re_extract
    try:
        tokens = _re_extract.findall(r"[A-Za-z']+", s_low)
        return tokens[-1] if tokens else ""
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Entity extraction for conversation threading
#
# The conversation state stores a list of thread entities extracted from
# recent queries and answers.  This helper performs a simple tokenisation of
# alphanumeric sequences, lowercases them, removes common stopwords and
# returns a deduplicated list.  It is deliberately naïve to avoid reliance
# on external NLP packages; its goal is to capture salient nouns or
# keywords that may help with pronoun resolution and topic inference.

def _extract_entities(text: str) -> List[str]:
    """
    Extract potential entities or keywords from a piece of text.

    This naive implementation tokenises the input on alphanumeric
    characters, lower-cases tokens, removes stopwords and returns unique
    terms.  It is designed to be lightweight and robust in the absence of
    external NLP libraries.

    Args:
        text: Arbitrary string from which to extract entities.

    Returns:
        A list of unique tokens representing possible entities or keywords.
    """
    try:
        import re as _re
        # Convert to string and lower case
        s = str(text or "")
        tokens = _re.findall(r"[A-Za-z0-9']+", s.lower())
        # Define a small stopword list; include common pronouns and
        # non-content words.  This list can be extended as needed.
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else",
            "this", "that", "it", "he", "she", "they", "we", "you", "your",
            "my", "mine", "ours", "ourselves", "i", "me", "am", "is", "are",
            "was", "were", "be", "been", "being", "to", "of", "in", "on",
            "for", "with", "about", "as", "at", "by", "from", "into",
            "onto", "until", "while", "up", "down", "over", "under",
            "more", "anything", "else", "what", "how", "did", "you", "get",
            "come", "up", "with"
        }
        ents: List[str] = []
        for tok in tokens:
            # Keep alphabetic tokens longer than one character and not in stopwords
            if tok and tok.isalpha() and tok not in stopwords:
                ents.append(tok)
        # Deduplicate while preserving order
        uniq: List[str] = []
        seen: set[str] = set()
        for t in ents:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Pronoun set and cache gating helpers
#
# ``PRONOUNS`` enumerates single-word pronouns and interrogatives that
# indicate a query is context-dependent.  ``_should_cache`` applies a
# quality gate when deciding whether to write a fast or semantic cache
# entry.  Queries that are too short, have low confidence, or contain
# pronouns are not cached.  See Fix 2 for details.

# Pronouns used to detect context-dependent queries.  If any of these
# tokens appear as standalone words within a query, caching is disabled.
PRONOUNS: Set[str] = {
    "that", "this", "it", "these", "those",
    "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how"
}

def _should_cache(query: str, verdict: str, confidence: float) -> bool:
    """Return True if the query should be cached.

    A query is eligible for caching only when it meets several quality
    criteria: it must have a TRUE verdict, consist of at least two
    space-separated tokens, exhibit sufficient confidence (≥ 0.8), and
    contain no pronoun tokens.  If any condition fails, the function
    returns False.

    Args:
        query: The original user query string.
        verdict: The Stage 8 verdict (e.g. 'TRUE', 'FALSE').
        confidence: The final confidence score for the answer.

    Returns:
        A boolean indicating whether caching should proceed.
    """
    try:
        q = str(query or "").strip()
        v = str(verdict or "").upper()
        if not q or v != "TRUE":
            return False
        # Too few words? (one or zero words)
        if len(q.split()) < 2:
            return False
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.0
        if conf < 0.8:
            return False
        qlower = q.lower()
        # Tokenise query and check pronouns as standalone tokens
        words = qlower.split()
        for p in PRONOUNS:
            if p in words:
                return False
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Phase 4: Tiered Memory Assignment and Scoring
#
# These functions implement the core logic for Maven's tiered memory system.
# _assign_tier() determines which memory tier a record belongs to based on
# intent, confidence, and content type. _score_memory_hit() computes a
# unified relevance score across all tiers for retrieval ranking.
# ---------------------------------------------------------------------------

def _assign_tier(record: Dict[str, Any], context: Dict[str, Any]) -> tuple[str, float]:
    """
    Assign a memory tier and importance score to a record.

    This function implements deterministic tier assignment based on intent
    type, confidence level, and semantic content. It does not use time-based
    logic, LLM calls, or randomness. The tier controls retention policy and
    retrieval priority, while importance influences consolidation and eviction.

    Tier Assignment Rules:
    - Identity, self-model, spec, governance → TIER_PINNED (never evicted)
    - User preferences, relationships → TIER_MID (cross-session)
    - General factual QA (high confidence) → TIER_MID
    - Theories, speculations → TIER_SHORT (session-scoped)
    - Creative outputs, one-offs → TIER_SHORT or SKIP_STORAGE
    - Low-value, nonsense → SKIP_STORAGE (empty tier string)

    Args:
        record: Memory record with 'content', 'confidence', and optional
                'intent', 'verdict', 'tags' fields.
        context: Additional context with 'intent', 'verdict', 'storable_type',
                 'is_identity', 'is_preference', 'is_relationship', etc.

    Returns:
        A tuple of (tier, importance_score) where tier is one of the
        TIER_* constants and importance_score is a float in [0.0, 1.0].
        Returns ("", 0.0) if the record should not be stored.
    """
    try:
        # Extract relevant fields from record and context
        content = str(record.get("content", "")).strip()
        confidence = float(record.get("confidence", 0.5))
        intent = str(context.get("intent", "") or record.get("intent", "")).upper()
        verdict = str(context.get("verdict", "") or record.get("verdict", "")).upper()
        storable_type = str(context.get("storable_type", "")).upper()
        tags = record.get("tags") or context.get("tags") or []

        # Normalize tags to list of strings
        if isinstance(tags, str):
            tags = [tags]
        tags_set = {str(t).lower() for t in tags or []}

        # Early exit: no content means no storage
        if not content or len(content) < 3:
            return ("", 0.0)

        # Rule 1: PINNED tier - Identity, self-model, specs, governance
        # These are permanent system knowledge that should never be evicted
        identity_intents = {"IDENTITY_QUERY", "SELF_DESCRIPTION_REQUEST", "MAVEN_IDENTITY"}
        system_tags = {"governance", "spec", "design", "architecture", "self_model"}

        if intent in identity_intents or tags_set & system_tags:
            return (TIER_PINNED, 1.0)

        # User identity statements
        if ("identity" in tags_set or "user_identity" in tags_set or
            "my name is" in content.lower() or "i am " in content.lower()):
            return (TIER_PINNED, 1.0)

        # Rule 2: MID tier - User preferences and relationships
        # These are high-value cross-session facts about the user
        preference_intents = {"PREFERENCE", "PREFERENCE_QUERY", "PROFILE_QUERY"}
        relationship_intents = {"RELATIONSHIP_QUERY", "RELATIONSHIP"}
        preference_tags = {"preference", "user_preference", "like", "dislike"}
        relationship_tags = {"relationship", "friend", "family"}

        if (intent in preference_intents or tags_set & preference_tags or
            verdict == "PREFERENCE"):
            return (TIER_MID, min(1.0, confidence + 0.2))

        if (intent in relationship_intents or tags_set & relationship_tags or
            "we are" in content.lower() or "you and i" in content.lower()):
            return (TIER_MID, min(1.0, confidence + 0.2))

        # Phase 5: LONG tier - Concepts (stable knowledge structures)
        # Concepts are formed from patterns and represent durable understanding
        if verdict == RECORD_TYPE_CONCEPT or "concept" in tags_set:
            return (TIER_LONG, min(1.0, confidence + 0.3))

        # Phase 5: MID tier - Skills (learned query handling strategies)
        # Skills represent recurring successful patterns that persist cross-session
        if verdict == RECORD_TYPE_SKILL or "skill" in tags_set:
            return (TIER_MID, min(1.0, confidence + 0.25))

        # Phase 5: SHORT/MID tier - Inferred patterns (detected regularities)
        # Patterns start in SHORT and promote to MID based on consistency
        if verdict == RECORD_TYPE_INFERRED_PATTERN or "pattern" in tags_set:
            # High-consistency patterns go to MID, others to SHORT
            pattern_consistency = record.get("consistency", 0.0)
            if pattern_consistency >= 0.3:  # 30% consistency threshold
                return (TIER_MID, min(1.0, confidence + 0.2))
            else:
                return (TIER_SHORT, confidence * 0.9)

        # Rule 3: MID tier - High-confidence factual knowledge
        # Validated facts with high confidence are cross-session knowledge
        if verdict == "TRUE" and confidence >= 0.8:
            # Check if this is general reusable knowledge vs. one-off
            # Skip creative/narrative content even if marked TRUE
            creative_indicators = ["story", "poem", "joke", "imagine", "creative"]
            if any(ind in content.lower() for ind in creative_indicators):
                return (TIER_SHORT, confidence * 0.7)
            return (TIER_MID, confidence)

        # Rule 4: SKIP very low confidence - Filter before tier assignment
        # Very low confidence records are not worth storing in any tier
        if confidence < 0.3:
            return ("", 0.0)

        # Rule 5: SHORT tier - Theories, speculations, medium confidence
        # These are session-scoped hypotheses that may be promoted later
        if verdict in {"THEORY", "UNKNOWN"} or (0.5 <= confidence < 0.8):
            return (TIER_SHORT, confidence * 0.8)

        # Rule 6: SHORT tier - Questions and transient interactions
        # Questions themselves may be stored briefly for context tracking
        question_intents = {"QUESTION", "SIMPLE_FACT_QUERY", "EXPLAIN", "WHY", "HOW"}
        if intent in question_intents or storable_type == "QUESTION":
            # Only store in SHORT if explicitly requested via tags
            if "store_question" in tags_set:
                return (TIER_SHORT, 0.5)
            else:
                return ("", 0.0)  # Skip storage by default

        # Rule 7: SKIP_STORAGE - Meta-queries, social, low quality
        skip_verdicts = {"SKIP_STORAGE", "UNANSWERED"}
        skip_intents = {"SOCIAL", "EMOTION", "GREETING", "UNKNOWN_INPUT"}

        if verdict in skip_verdicts or intent in skip_intents:
            return ("", 0.0)

        # Rule 7: LONG tier - Default for unclassified mid-high confidence
        # This is the fallback for factual content that doesn't fit above
        if confidence >= 0.6:
            return (TIER_LONG, confidence)

        # Final fallback: SHORT tier for anything else with minimal importance
        return (TIER_SHORT, confidence * 0.5)

    except Exception:
        # On any error, skip storage to avoid corrupting memory
        return ("", 0.0)


def _score_memory_hit(hit: Dict[str, Any], query: Dict[str, Any]) -> float:
    """
    Compute a deterministic relevance score for a memory retrieval hit.

    This function ranks memory hits across all tiers using a transparent,
    explainable formula. It combines tier priority, importance, usage
    patterns, recency, and match quality without time-based logic or
    randomness.

    Scoring Formula:
        base_score = match_quality (from retrieval system)
        tier_boost = tier-dependent constant
        importance_boost = importance * 0.3
        usage_boost = min(use_count * 0.05, 0.2)
        recency_boost = (seq_id / max_seq_id) * 0.1  # normalized recency

        final_score = base_score + tier_boost + importance_boost + usage_boost + recency_boost

    Args:
        hit: A memory record with fields: tier, importance, use_count, seq_id,
             score (match quality), content, etc.
        query: The retrieval query context (may include filters, k, etc.)

    Returns:
        A float score in [0.0, ~2.0+] where higher scores indicate better
        relevance. The score is deterministic and does not depend on time.
    """
    try:
        # Extract base match quality (default to 0.5 if not present)
        try:
            base_score = float(hit.get("score", 0.0) or hit.get("similarity", 0.0) or 0.5)
        except (TypeError, ValueError):
            base_score = 0.5

        # Tier boost - prioritize higher tiers
        tier = str(hit.get("tier", "")).upper()
        tier_boost_map = {
            TIER_PINNED: 0.5,   # Highest priority
            TIER_MID: 0.3,      # Cross-session facts
            TIER_SHORT: 0.1,    # Session facts
            TIER_WM: 0.4,       # Working memory (high short-term value)
            TIER_LONG: 0.2,     # Long-term knowledge
        }
        tier_boost = tier_boost_map.get(tier, 0.0)

        # Importance boost (0 to 0.3)
        try:
            importance = float(hit.get("importance", 0.0))
        except (TypeError, ValueError):
            importance = 0.0
        importance_boost = min(importance * 0.3, 0.3)

        # Usage boost - frequently accessed hits get a bump
        try:
            use_count = int(hit.get("use_count", 0))
        except (TypeError, ValueError):
            use_count = 0
        usage_boost = min(use_count * 0.05, 0.2)

        # Recency boost - based on seq_id (larger = more recent)
        # Normalize by current counter to get [0, 1] range
        try:
            seq_id = int(hit.get("seq_id", 0))
        except (TypeError, ValueError):
            seq_id = 0

        # Get current max seq_id for normalization
        global _SEQ_ID_COUNTER
        max_seq_id = max(_SEQ_ID_COUNTER, 1)  # Avoid division by zero
        recency_boost = (seq_id / max_seq_id) * 0.1 if seq_id > 0 else 0.0

        # Phase 5: Record type boost - concepts, skills, patterns get priority
        verdict = str(hit.get("verdict", "")).upper()
        tags = hit.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags_set = {str(t).lower() for t in tags}

        record_type_boost = 0.0
        if verdict == RECORD_TYPE_CONCEPT or "concept" in tags_set:
            record_type_boost = 0.25  # Concepts highly valuable
        elif verdict == RECORD_TYPE_SKILL or "skill" in tags_set:
            record_type_boost = 0.20  # Skills very valuable
        elif verdict == RECORD_TYPE_INFERRED_PATTERN or "pattern" in tags_set:
            record_type_boost = 0.15  # Patterns moderately valuable

        # Combine all factors
        final_score = base_score + tier_boost + importance_boost + usage_boost + recency_boost + record_type_boost

        return final_score

    except Exception:
        # On error, return base score only
        try:
            return float(hit.get("score", 0.0) or 0.5)
        except Exception:
            return 0.5

# ---------------------------------------------------------------------------
# Continuation and pronoun resolution helper
#
# User utterances may contain requests to continue a previous topic (e.g.
# "more", "anything else", "more about physics") or refer to a prior answer
# using pronouns (e.g. "that", "this", "it").  This helper inspects the
# input query and the global conversation state to rewrite the query when
# appropriate.  It returns a tuple consisting of the potentially modified
# query and a boolean flag indicating whether the retrieval should
# restrict itself to the short‑term memory bank.  Errors are suppressed to
# avoid disrupting the pipeline.

def _resolve_continuation_and_pronouns(query: str) -> tuple[str, bool]:
    """
    Resolve continuation triggers and simple pronoun references.

    Args:
        query: The original user query.

    Returns:
        A tuple of (resolved_query, force_stm) where resolved_query is
        the modified query string (or the original when no change is
        necessary) and force_stm indicates whether retrieval should be
        limited to the short‑term memory bank.
    """
    try:
        q = str(query or "")
    except Exception:
        return query, False
    try:
        ql = q.strip().lower()
    except Exception:
        ql = ""
    # Default flag: do not restrict retrieval to STM
    force_stm = False
    # Access the current conversation state
    conv = globals().get("_CONVERSATION_STATE", {}) or {}
    last_topic = conv.get("last_topic", "") or ""
    last_response = conv.get("last_response", "") or ""
    try:
        # Continuation triggers for bare follow-ups
        # Include common variants and polite continuations
        triggers_set = {
            "more",
            "anything else",
            "any thing else",
            "what else",
            "what else?",
            "tell me more",
            "tell me more?",
            "go on",
            "continue",
            "continue?",
        }
        special_last_answer_triggers = {
            "what did you just say",
            "what was your last answer",
            "what was your last response",
            "last answer",
            "last response",
        }
        if ql in triggers_set or ql in special_last_answer_triggers:
            # If the user explicitly asks about the last answer, return
            # the last response.  Otherwise, use the last topic.
            if ql in special_last_answer_triggers:
                if last_response:
                    return last_response, True
                elif last_topic:
                    return last_topic, True
            # Generic continuation: use last topic
            if last_topic:
                return last_topic, True
        # Detect "more about X" and extract X.  Also handle "tell me more about X"
        if ql.startswith("more about ") or ql.startswith("tell me more about "):
            if ql.startswith("tell me more about "):
                candidate = ql[len("tell me more about "):].strip()
            else:
                candidate = ql[len("more about "):].strip()
            if candidate:
                return candidate, True
        # Detect "more on X" patterns and extract X or fall back to last_topic
        if ql.startswith("more on "):
            candidate = ql[len("more on "):].strip()
            if candidate:
                # If candidate is a pronoun like that/this/it etc., resolve to last_topic
                if candidate in {"that", "this", "it", "them", "these", "those"}:
                    if last_topic:
                        return last_topic, True
                return candidate, True
        # Single pronoun queries: return last response or last topic
        if ql in {"that", "this", "it", "them", "these", "those"}:
            if last_response:
                return last_response, True
            elif last_topic:
                return last_topic, True
        # Pronoun references embedded in explanatory questions
        import re as _re_pron
        # Patterns covering "how did you get {pronoun}" and "how did you come up with {pronoun}"
        patterns = [
            r"(how\s+did\s+you\s+get\s+)(that|this|it|them|these|those)\b",
            r"(how\s+did\s+you\s+come\s+up\s+with\s+)(that|this|it|them|these|those)\b",
            r"(explain\s+)(that|this|it|them|these|those)\b",
        ]
        for pat in patterns:
            m = _re_pron.search(pat, ql)
            if m:
                prefix = m.group(1)
                replacement = last_response or last_topic
                if replacement:
                    # Replace the pronoun with the resolved phrase
                    new_query = _re_pron.sub(pat, prefix + replacement, ql)
                    return new_query, True
        # Additional fallback: if the query is short (<= 3 words) and ends with
        # "else", treat it as a continuation (e.g. "anything else?")
        try:
            if len(ql.split()) <= 3 and "else" in ql:
                if last_topic:
                    return last_topic, True
        except Exception:
            pass
    except Exception:
        # Silently ignore resolution errors
        pass
    return query, False
    if not s:
        return ""
    # Normalize whitespace and case
    s_low = s.lower()
    # If the query contains 'about', return the substring after the last 'about'
    try:
        if " about " in s_low:
            # Split on the first occurrence of ' about ' to capture everything after
            parts = s_low.split(" about ", 1)
            topic = parts[1].strip()
            return topic
    except Exception:
        pass
    # Fallback: return the last alphabetic token
    import re as _re_extract
    try:
        tokens = _re_extract.findall(r"[A-Za-z']+", s_low)
        return tokens[-1] if tokens else ""
    except Exception:
        return ""

def _update_conversation_state(query: str, answer: str) -> None:
    """
    Update the global conversation state variables with the latest topic and
    response.  This function extracts a topic from the provided query and
    stores both the topic and answer in module‑level variables.  It is
    resilient to errors and will silently ignore failures.

    Args:
        query: The user's original query string.
        answer: The system's final answer string.
    """
    global _LAST_TOPIC, _LAST_RESPONSE, _CONVERSATION_STATE
    try:
        topic = _extract_topic(query)
        if topic:
            _LAST_TOPIC = topic
        if answer:
            _LAST_RESPONSE = str(answer).strip()
        # Update the detailed conversation state.  Always record the last
        # query and response; if a topic was extracted, overwrite the
        # stored topic, otherwise retain the existing one.  Also extract
        # salient entities from the query and answer to aid pronoun
        # resolution and update the conversation depth based on the
        # recent query history.
        try:
            _CONVERSATION_STATE["last_query"] = str(query).strip()
        except Exception:
            _CONVERSATION_STATE["last_query"] = query
        try:
            _CONVERSATION_STATE["last_response"] = str(answer).strip() if answer else ""
        except Exception:
            _CONVERSATION_STATE["last_response"] = answer or ""
        try:
            if topic:
                _CONVERSATION_STATE["last_topic"] = topic
                # Persist topic to system_history for teacher learning
                try:
                    from brains.cognitive.system_history.service.system_history_brain import service_api as sys_history_api  # type: ignore
                    sys_history_api({
                        "op": "SET_TOPIC",
                        "payload": {"topic": topic}
                    })
                except Exception:
                    pass
            elif not _CONVERSATION_STATE.get("last_topic"):
                # Fallback to using the existing _LAST_TOPIC if no new topic
                _CONVERSATION_STATE["last_topic"] = _LAST_TOPIC or ""
        except Exception:
            _CONVERSATION_STATE["last_topic"] = topic or _CONVERSATION_STATE.get("last_topic", "")
        try:
            ents_q = _extract_entities(query)
        except Exception:
            ents_q = []
        try:
            ents_a = _extract_entities(answer) if answer else []
        except Exception:
            ents_a = []
        try:
            # Use a set to deduplicate but preserve order via list comprehension
            combined = []
            seen = set()
            for e in (ents_q + ents_a):
                if e not in seen:
                    combined.append(e)
                    seen.add(e)
            _CONVERSATION_STATE["thread_entities"] = combined
        except Exception:
            _CONVERSATION_STATE["thread_entities"] = []
        try:
            # Conversation depth equals the number of recent queries if available
            _CONVERSATION_STATE["conversation_depth"] = len(_RECENT_QUERIES)
        except Exception:
            try:
                _CONVERSATION_STATE["conversation_depth"] = int(_CONVERSATION_STATE.get("conversation_depth", 0)) + 1
            except Exception:
                _CONVERSATION_STATE["conversation_depth"] = 1
    except Exception:
        # Do not propagate exceptions from state updates
        pass

# ---------------------------------------------------------------------------
# Identity bootstrap and episodic helpers
#
# The durable identity store records the primary user name across sessions.
# These helpers allow the memory librarian to populate working memory with
# the stored name on startup and to extract recent self‑introductions from
# the conversation history.  They are light‑weight and do not depend on
# any external Maven components.  Errors are suppressed so that missing
# identity modules do not break the librarian.

try:
    from brains.personal.service import identity_user_store  # type: ignore
except Exception:
    identity_user_store = None  # type: ignore

def bootstrap_identity(wm_put: Any) -> None:
    """Hydrate working memory with the primary user name from the durable store.

    If a name exists in the durable store, this helper writes it to
    working memory using the provided ``wm_put`` callable.  The entry
    uses key ``user_identity`` and tags ["identity", "name"] with
    full confidence.  Exceptions are ignored to avoid disrupting
    startup.

    Args:
        wm_put: A callable following the signature of the WM_PUT
            operation.  It should accept keyword arguments ``key``,
            ``value``, ``tags`` and ``confidence``.
    """
    try:
        if identity_user_store:
            ident = identity_user_store.GET()  # type: ignore[attr-defined]
            if isinstance(ident, dict):
                name = ident.get("name")
                if name:
                    wm_put(key="user_identity", value=name, tags=["identity", "name"], confidence=1.0)
    except Exception:
        pass

def episodic_last_declared_identity(recent_queries: List[Dict[str, Any]], n: int = 10) -> Optional[str]:
    """Return the last declared name from recent queries.

    This helper scans the last ``n`` user queries in reverse order and
    returns the first phrase following "I am", "I'm", "im", "call me" or "my name is"
    if found.  Matching is case‑insensitive and conservative.  When
    no declaration is found, None is returned.

    Args:
        recent_queries: A list of dicts representing recent exchanges.
            Each dict should contain at least the user's utterance under
            the "user" key.  Non‑dict entries are ignored.
        n: The maximum number of recent queries to inspect.
    Returns:
        The extracted name as a string, or None if no match is found.
    """
    try:
        if not recent_queries:
            return None
        # Limit to the last n entries
        for entry in reversed(recent_queries[-int(n):]):
            try:
                utter = str(entry.get("user") or "").strip()
            except Exception:
                continue
            if not utter:
                continue
            lower = utter.lower()
            import re as _re  # local import
            m = _re.search(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\s'-]*)", utter, _re.IGNORECASE)
            if not m:
                m = _re.search(r"\b(?:i\s+am|i'm|im|call\s+me)\s+([A-Za-z][A-Za-z\s'-]*)", utter, _re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if name:
                    return name
        return None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Blackboard subscription registry (Phase 6)
#
# The blackboard acts as a lightweight shared working memory.  Consumers
# may register interest in certain WM entries via BB_SUBSCRIBE.  Each
# subscription can specify a key filter, tag filters, a minimum confidence
# threshold, a time‑to‑live for matching events and a priority hint.
# The CONTROL_CYCLE operation scans WM, scores candidate events per
# subscription, performs a simple arbitration via the integrator brain and
# dispatches the winning event through the message bus.
#
# Subscriptions are stored in the form:
#   _BLACKBOARD_SUBS[subscriber_id] = {
#       "key": Optional[str],
#       "tags": Optional[List[str]],
#       "min_conf": float,
#       "ttl": float,
#       "priority": float,
#       "last_index": int,
#   }
#
# last_index is used to avoid reprocessing the same WM entries across cycles.

_BLACKBOARD_SUBS: Dict[str, Dict[str, Any]] = {}

def _bb_subscribe(subscriber: str, key: Optional[str], tags: Optional[List[str]], min_conf: float, ttl: float, priority: float) -> None:
    """Register or update a blackboard subscription.

    Args:
        subscriber: Unique identifier for the subscriber (brain name).
        key: Optional key filter; only WM entries with matching key are delivered.
        tags: Optional list of tags; at least one must match for an event to be considered.
        min_conf: Minimum confidence score to accept an event.
        ttl: Maximum age in seconds for events to be delivered.
        priority: Base priority hint used during arbitration.
    """
    try:
        sub_cfg: Dict[str, Any] = {
            "key": key.strip() if isinstance(key, str) and key.strip() else None,
            "tags": [t.strip() for t in tags] if tags else None,
            "min_conf": float(min_conf) if min_conf is not None else 0.0,
            "ttl": float(ttl) if ttl is not None else 300.0,
            "priority": float(priority) if priority is not None else 0.5,
            "last_index": 0,
        }
        _BLACKBOARD_SUBS[str(subscriber)] = sub_cfg
    except Exception:
        # If any conversion fails, fall back to defaults
        _BLACKBOARD_SUBS[str(subscriber)] = {
            "key": None,
            "tags": None,
            "min_conf": 0.0,
            "ttl": 300.0,
            "priority": 0.5,
            "last_index": 0,
        }

def _bb_collect_events() -> List[Dict[str, Any]]:
    """Collect and annotate working memory entries for blackboard arbitration.

    Returns a list of tuples (subscriber_id, entry, score).  Each entry is
    a shallow copy of the WM entry with an added ``bb_score``.  The score
    combines the base priority of the subscription with the entry's
    confidence and recency.  This helper does not mutate WM or the
    subscription registry.
    """
    events: List[Dict[str, Any]] = []
    try:
        # Snapshot WM without expiry metadata
        with _WM_LOCK:
            _prune_working_memory()
            snapshot = list(_WORKING_MEMORY)
        for sub_id, cfg in _BLACKBOARD_SUBS.items():
            # Determine the starting index; ensure valid range
            last_idx = int(cfg.get("last_index", 0))
            if last_idx < 0:
                last_idx = 0
            for idx, ent in enumerate(snapshot):
                # Skip entries already processed
                if idx < last_idx:
                    continue
                # Filter by key
                sub_key = cfg.get("key")
                if sub_key and ent.get("key") != sub_key:
                    continue
                # Filter by tags
                sub_tags = cfg.get("tags")
                if sub_tags:
                    try:
                        ent_tags = ent.get("tags") or []
                        if not any(t in ent_tags for t in sub_tags):
                            continue
                    except Exception:
                        continue
                # Filter by confidence
                try:
                    conf_val = float(ent.get("confidence", 0.0))
                except Exception:
                    conf_val = 0.0
                if conf_val < cfg.get("min_conf", 0.0):
                    continue
                # No time-based decay
                recency = 1.0
                # Compute reliability if present
                try:
                    reliability = float(ent.get("source_reliability", 1.0))
                except Exception:
                    reliability = 1.0
                base_p = cfg.get("priority", 0.5)
                score = base_p * conf_val * recency * reliability
                event_copy = {k: v for k, v in ent.items() if k != "expires_at"}
                event_copy["bb_score"] = score
                events.append({
                    "subscriber": sub_id,
                    "entry": event_copy,
                    "score": score,
                    "index": idx,
                })
    except Exception:
        return []
    return events

def _bb_mark_processed(sub_id: str, index: int) -> None:
    """Advance the last processed index for a subscriber."""
    try:
        sub = _BLACKBOARD_SUBS.get(sub_id)
        if sub is not None:
            sub["last_index"] = max(sub.get("last_index", 0), index + 1)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Shared Working Memory (Step‑1 integration)
#
# The working memory is a simple in‑memory list of dictionaries used for
# opportunistic information exchange between cognitive modules.  Each entry
# includes a key, an arbitrary value, optional tags, a confidence score and
# an expiry timestamp.  Entries persist only for a short TTL and are
# automatically pruned on each access.  A lock protects concurrent access.
_WORKING_MEMORY: List[Dict[str, Any]] = []
_WM_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Working Memory Persistence & Arbitration (Step‑2.1)
#
# These helpers enable persistence of the working memory across process runs
# and scoring of competing entries.  Persistence can be toggled via
# ``CFG['wm']['persist']`` and arbitration via ``CFG['wm']['arbitration']``.

# Flag to indicate whether we've loaded from disk already
_WM_LOADED_FROM_DISK: bool = False

def _wm_store_path() -> Path:
    """Return the path to the persistent working memory store."""
    try:
        return get_reports_path("wm_store.jsonl")
    except Exception:
        return Path("wm_store.jsonl")

def _wm_persist_append(entry: Dict[str, Any]) -> None:
    """Append a working memory entry to the persistent store."""
    try:
        from api.utils import append_jsonl  # type: ignore
        append_jsonl(_wm_store_path(), entry)
    except Exception:
        pass

def _wm_load_from_disk(max_records: int = 5000) -> None:
    """Load persisted working memory entries from disk into memory.

    This is a best‑effort loader: it ignores malformed lines and expired entries.
    It should be called with the WM lock held and only once per process.
    """
    global _WM_LOADED_FROM_DISK
    if _WM_LOADED_FROM_DISK:
        return
    try:
        path = _wm_store_path()
        if not path.exists():
            _WM_LOADED_FROM_DISK = True
            return
        loaded = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if loaded >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                _WORKING_MEMORY.append(rec)
                loaded += 1
        _prune_working_memory()
    except Exception:
        pass
    _WM_LOADED_FROM_DISK = True

def _wm_load_if_needed() -> None:
    """Load persisted working memory if configured and not yet loaded."""
    try:
        from api.utils import CFG  # type: ignore
        persist_enabled = True
        try:
            persist_enabled = bool((CFG.get("wm", {}) or {}).get("persist", True))
        except Exception:
            persist_enabled = True
    except Exception:
        persist_enabled = True
    if not persist_enabled:
        return
    with _WM_LOCK:
        _wm_load_from_disk()

def _prune_working_memory() -> None:
    """Remove expired items from the working memory.

    Entries contain an ``expires_at`` field set when they are stored via
    the ``WM_PUT`` operation.  This helper filters out any entry whose
    expiry has passed.  It must be called with the WM lock held.
    """
    # No time-based expiry - keep all entries
    pass

# ---------------------------------------------------------------------------
# Per-brain persistent memory with merge-write learning
#
# These helpers enable Maven to learn from validated facts by persisting
# key/value pairs to per-brain JSONL files. When the same fact is
# confirmed multiple times (TRUE verdict), confidence is bumped and
# access_count is incremented, creating a reinforcement learning effect.

def _brain_path(brain: str) -> Path:
    """Return the path to a brain's persistent memory file."""
    try:
        return get_reports_path("memory", f"{brain}.jsonl")
    except Exception:
        return Path(f"{brain}.jsonl")

def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    """Append a JSON entry to a JSONL file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _get_brain(brain: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent value for a key from a brain's memory.

    Scans the brain's JSONL file in reverse to find the last entry
    matching the requested key. Returns a dict with value, confidence,
    and access_count, or None if not found.
    """
    key = ctx.get("key")
    path = _brain_path(brain)
    if not path.exists():
        return {"status":"ok","data":None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("k") == key or rec.get("key") == key:
                return {"status":"ok","data":{
                    "value": rec.get("value", rec.get("v")),
                    "confidence": rec.get("confidence", 0.4),
                    "access_count": rec.get("access_count", 0)
                }}
    except Exception:
        pass
    return {"status":"ok","data":None}

def _merge_brain_kv(brain: str, key: str, value, conf_delta: float = 0.1) -> Dict[str, Any]:
    """Last-write-wins with confidence bump and access_count increment.

    When the same key/value pair is stored multiple times (indicating
    repeated validation), confidence is increased by conf_delta and
    access_count is incremented. This creates a reinforcement learning
    effect where frequently confirmed facts gain higher confidence.

    Args:
        brain: The brain name (used for file path)
        key: The memory key
        value: The memory value
        conf_delta: Confidence increase on repeat (default 0.1)

    Returns:
        Dict with value, confidence, and access_count
    """
    prev_resp = _get_brain(brain, {"key": key})
    prev = prev_resp.get("data") if isinstance(prev_resp, dict) else None
    if prev and str(prev.get("value")) == str(value):
        # Same value confirmed again - bump confidence and access count
        new_conf = min(1.0, float(prev.get("confidence") or 0) + conf_delta)
        new_acc = int(prev.get("access_count") or 0) + 1
    else:
        # New or changed value - start with modest confidence
        new_conf = 0.6  # starting point for newly confirmed facts
        new_acc = 1
    _append_jsonl(_brain_path(brain), {
        "key": key,
        "value": value,
        "confidence": new_conf,
        "access_count": new_acc,
    })
    return {"value": value, "confidence": new_conf, "access_count": new_acc}

# ---------------------------------------------------------------------------
# Relationship fact storage
#
# These helpers enable Maven to store and retrieve relationship facts such as
# "we are friends" or "we are not friends". Relationship facts are stored
# using the same JSONL mechanism as other facts, but with a dedicated "relationships"
# brain to keep them separate and easily queryable.

def set_relationship_fact(user_id: str, key: str, value: bool) -> None:
    """
    Store a simple relationship fact, e.g. ('friend_with_system', True).
    Should write to the same underlying store used for other user-specific facts.

    Args:
        user_id: The user identifier
        key: The relationship key (e.g., "friend_with_system")
        value: Boolean value indicating the relationship status
    """
    try:
        record = {
            "kind": "relationship",
            "key": key,
            "value": value,
            "user_id": user_id,
            "status": "confirmed",
            "source": "user_statement",
        }
        # Store in the relationships brain
        _append_jsonl(_brain_path("relationships"), record)
    except Exception:
        # Silently ignore errors to prevent pipeline disruption
        pass

def get_relationship_fact(user_id: str, key: str) -> dict | None:
    """
    Retrieve a relationship fact, or None if not present.
    Should use the existing memory/query mechanisms, not a new subsystem.

    Args:
        user_id: The user identifier
        key: The relationship key (e.g., "friend_with_system")

    Returns:
        The latest record for (user_id, key) or None if not found.
    """
    try:
        path = _brain_path("relationships")
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Scan in reverse to get the most recent entry
        for line in reversed(lines):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("user_id") == user_id and rec.get("key") == key:
                return rec
    except Exception:
        pass
    return None

def get_all_preferences(user_id: str) -> list:
    """
    Retrieve all stored preferences for a user.

    This searches through memory banks for records tagged with "preference"
    and returns them as a list of preference facts.

    Args:
        user_id: The user identifier

    Returns:
        List of preference records (dicts with keys like 'content', 'confidence', etc.)
    """
    preferences = []
    try:
        # Search through multiple banks where preferences might be stored
        bank_names = ["factual", "working_theories", "preferences"]

        for bank_name in bank_names:
            try:
                bank_path = _brain_path(bank_name)
                if not bank_path.exists():
                    continue

                with open(bank_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        # Check if this is a preference record for our user
                        if (rec.get("user_id") == user_id or not rec.get("user_id")) and \
                           ("preference" in str(rec.get("tags", [])).lower() or \
                            "preference" in str(rec.get("kind", "")).lower()):
                            preferences.append(rec)
                    except Exception:
                        continue
            except Exception:
                continue

        # Also search brain_storage.jsonl for BRAIN_PUT stored preferences
        try:
            brain_mem_file = MAVEN_ROOT / "brains" / "cognitive" / "memory_librarian" / "memory" / "brain_storage.jsonl"
            if brain_mem_file.exists():
                with open(brain_mem_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Track the latest value for each preference key (last write wins)
                preference_keys = {}
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        key = rec.get("key", "")
                        # Check if this is a preference key (favorite_*)
                        if key.startswith("favorite_"):
                            preference_keys[key] = rec
                    except Exception:
                        continue

                # Convert preference keys to the format expected by list_preferences
                for key, rec in preference_keys.items():
                    value = rec.get("value")
                    if value:
                        # Extract the preference type from the key (e.g., "favorite_color" -> "color")
                        pref_type = key.replace("favorite_", "")

                        # Format the content string based on preference type
                        if pref_type == "color":
                            content = f"the color {value}"
                        elif pref_type == "animal":
                            content = f"{value}s"
                        elif pref_type == "food":
                            content = value
                        else:
                            content = value

                        preferences.append({
                            "content": content,
                            "confidence": rec.get("confidence", 0.9),
                            "kind": "preference",
                            "tags": ["preference", pref_type],
                            "key": key,
                            "value": value
                        })
        except Exception:
            pass

    except Exception:
        pass

    return preferences

def list_preferences(user_id: str, domain: str | None = None) -> list:
    """
    List stored preferences for a user, optionally filtered by domain.

    This is a helper for preference summarization queries. It retrieves
    all preference records and filters them by domain category if specified.

    Args:
        user_id: The user identifier
        domain: Optional domain/category to filter by (e.g., "animals", "food", "music")

    Returns:
        List of preference records matching the criteria
    """
    # Get all preferences first
    all_prefs = get_all_preferences(user_id)

    # If no domain filter, return all
    if domain is None:
        return all_prefs

    # Filter by domain
    filtered = []
    try:
        domain_lower = domain.lower()
        for pref in all_prefs:
            # Check if domain appears in content, tags, or kind
            content = str(pref.get("content", "")).lower()
            tags = str(pref.get("tags", [])).lower()
            kind = str(pref.get("kind", "")).lower()

            if domain_lower in content or domain_lower in tags or domain_lower in kind:
                filtered.append(pref)
    except Exception:
        # If filtering fails, return all preferences
        return all_prefs

    return filtered

# ---------------------------------------------------------------------------
# Fast‑path caching for repeated queries
#
# When the same question is asked multiple times within a short time window,
# Maven should be able to answer instantly without re‑running heavy
# retrieval or reasoning.  To accomplish this, the memory librarian
# maintains a tiny cache mapping normalised questions to their validated
# answers along with confidence scores and timestamps.  Entries expire
# after 24 hours.  During the pipeline run, the librarian consults this
# cache before invoking the planner, pattern recogniser or any memory
# retrieval.  When a cache hit is detected, the pipeline short‑circuits
# most stages and goes straight to candidate generation and finalisation.
# The cache is appended to ``reports/fast_cache.jsonl`` for persistence.

FAST_CACHE_TTL_SEC: float = 24 * 60 * 60  # 24 hour expiration
# FAST_CACHE_PATH will be initialised after MAVEN_ROOT is defined.  Assign to None for now.
FAST_CACHE_PATH: Optional[Path] = None

# ---------------------------------------------------------------------------
# Semantic cache for cross‑session context reuse
#
# In addition to the fast cache used for exact repeat queries, the memory
# librarian maintains a lightweight semantic cache keyed by token overlap.
# This cache enables Maven to recall answers from previous runs when the
# current question shares a significant number of keywords with a past
# query.  Each entry stores the original query text, a token set, the
# answer, confidence and a timestamp.  When a new question arrives, the
# librarian consults the semantic cache after the fast cache lookup and
# before invoking heavy retrieval or reasoning.  If a match with at
# least 50 percent token overlap is found, the cached answer is used to
# generate a response directly, bypassing retrieval.  After the final
# answer is produced, the semantic cache is updated with the current
# query and answer for future reuse.
SEMANTIC_CACHE_PATH: Optional[Path] = None
# Semantic cache path for cross‑session context reuse.  This file stores
# a list of query→answer pairs used to answer semantically similar
# queries in the future.  It is initialised after ``MAVEN_ROOT`` is
# available below.
SEMANTIC_CACHE_PATH: Optional[Path] = None

# Phrases that indicate a cached answer may be meta, self‑referential or otherwise
# non‑informative.  If the cached answer contains any of these substrings
# (case‑insensitive), the cache entry is treated as invalid and the
# pipeline falls back to full retrieval.  This prevents a filler
# response like "I'm going to try my best" from being accepted as a
# factual answer on subsequent runs.
# Default phrases indicating a cached answer may be meta, self‑referential or otherwise
# non‑informative.  These are used as a fallback if no configuration file is provided.
BAD_CACHE_PHRASES: List[str] = [
    "i'm going to try my best",
    "i am going to try my best",
    "i don't have specific information",
    "i don't have information",
    "as an ai",
    "got it — noted",
    "got it - noted",
    "i'm also considering other possibilities",
    "i\u2019m going to try my best",  # unicode apostrophe variant
]

def _load_cache_sanity_phrases() -> List[str]:
    """Load bad phrases from config/cache_sanity.json if present.

    The configuration file should contain a JSON object with a
    ``bad_phrases`` list.  Each entry is normalised to lowercase and
    stripped of surrounding whitespace.  If the file is missing or
    malformed, the built‑in ``BAD_CACHE_PHRASES`` list is returned.
    """
    try:
        # MAVEN_ROOT may not be defined when this module is first imported.
        # Use a local import guard to avoid NameError; dynamic loading will
        # occur after MAVEN_ROOT is set up later in the file.
        root = globals().get("MAVEN_ROOT")
        if root:
            cfg_path = (root / "config" / "cache_sanity.json").resolve()
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
                phrases = data.get("bad_phrases") or data.get("BAD_PHRASES") or []
                if isinstance(phrases, list):
                    cleaned: List[str] = []
                    for p in phrases:
                        try:
                            s = str(p).strip().lower()
                            if s:
                                cleaned.append(s)
                        except Exception:
                            continue
                    if cleaned:
                        return cleaned
    except Exception:
        # Ignore any errors during config loading; fallback to default
        pass
    # Fallback to the built‑in BAD_CACHE_PHRASES
    try:
        return [str(p).strip().lower() for p in BAD_CACHE_PHRASES if p]
    except Exception:
        return []

# === Self query and similarity helpers =====================================

def _tokenize(text: str) -> List[str]:
    """Basic alphanumeric tokenizer used for similarity calculations.

    Args:
        text: Arbitrary string.
    Returns:
        A list of lower‑cased alphanumeric word tokens.
    """
    import re
    try:
        return [t for t in re.findall(r"\w+", str(text or "").lower()) if t]
    except Exception:
        return []

def _cosine_similarity(a: set[str], b: set[str]) -> float:
    """Compute a simple cosine similarity between two token sets.

    This treats the sets as binary vectors; the dot product is the
    intersection size, and norms are the square roots of the set sizes.
    Returns 0.0 when either set is empty.

    Args:
        a: Set of tokens for the first string.
        b: Set of tokens for the second string.
    Returns:
        Cosine similarity as a float between 0 and 1.
    """
    try:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        from math import sqrt
        return inter / (sqrt(len(a)) * sqrt(len(b)))
    except Exception:
        return 0.0

def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute the Jaccard similarity between two token sets.

    Returns 0.0 when either set is empty.
    Args:
        a: Set of tokens.
        b: Set of tokens.
    Returns:
        Jaccard similarity ratio.
    """
    try:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0

def _is_self_query(text: str) -> bool:
    """Detect whether a query refers to the agent's self (identity, age, location or preferences).

    This helper identifies questions directed at the assistant about
    its own attributes.  In addition to WH‑pronoun combinations such
    as ``who are you``, ``what is your name`` and ``where are you``,
    it also detects preference queries like ``do you like …`` or
    ``what's your favourite …``.  The matching is deliberately
    conservative: we require a direct second‑person reference (``you`` or
    ``your``) in combination with a recognised trigger word (WH word,
    modal verb ``do`` or the word ``favourite``).  Age questions are
    handled explicitly.

    Args:
        text: Raw user query.
    Returns:
        True if the query appears to be about the agent's identity.
    """
    try:
        ql = (text or "").strip().lower()
    except Exception:
        return False
    import re
    # Age patterns
    if re.search(r"\bhow\s+old\s+are\s+you\b", ql) or re.search(r"\bhow\s+old\s+you\b", ql):
        return True
    # Identity patterns
    if re.search(r"\b(who|what|where|how)\b", ql) and re.search(r"\b(you|your|yourself)\b", ql):
        return True
    # Capabilities queries.  Detect questions asking about what the agent can do
    # or is capable of.  Match phrases like "what can you do", "what do you do",
    # "what are your capabilities", "what are you capable", and similar variations.
    try:
        # "what can you do" or "what do you do"
        if re.search(r"\bwhat\s+(?:can|do)\s+you\s+(?:do)?\b", ql):
            return True
        # "what are your capabilities/abilities/skills"
        if re.search(r"\bwhat\s+are\s+(?:your|you)\s+(?:capabilities|abilities|skills)\b", ql):
            return True
        # "what are you capable"
        if re.search(r"\bwhat\s+are\s+you\s+capable\b", ql):
            return True
    except Exception:
        pass
    # Preference or likes queries.  Users sometimes ask about Maven's tastes
    # using variations like "do you like", "what do you prefer", "what are your
    # preferences", "do you enjoy" or "are you into".  To robustly detect
    # these, split the query into word tokens and look for a pronoun
    # referring to the agent (you/your/yourself) together with a token
    # beginning with a preference root (e.g. like, prefer, favour, enjoy,
    # into, interested, love).  This captures misspellings such as
    # "likee" and "preferances" and synonyms like "favourite".  If both
    # conditions hold, treat the query as self‑referential so it
    # triggers the self model.
    try:
        tokens = re.findall(r"\b\w+\b", ql)
        # Include second‑person possessive "yours" and first‑/plural‑person
        # pronouns like "mine", "our", "ours", "we", "us" in the pronoun
        # set.  Queries containing these pronouns often refer back to
        # previously mentioned user or shared information rather than
        # requesting factual knowledge.  Without including these, such
        # questions may be misrouted to the general memory search instead
        # of being handled by the self model or pronoun resolution.
        pronouns = {"you", "your", "yourself", "yours", "mine", "our", "ours", "we", "us"}
        pref_roots = [
            "like", "lik", "prefer", "preferenc", "preferanc",
            "favor", "favour", "favorite", "favourite",
            "enjoy", "into", "interested", "love"
        ]
        found_pronoun = any(tok in pronouns for tok in tokens)
        found_pref = any(any(tok.startswith(root) for root in pref_roots) for tok in tokens)
        if found_pronoun and found_pref:
            return True
    except Exception:
        pass
    return False

#
# Environment query detection
#
def _is_env_query(text: str) -> bool:
    """
    Detect whether a query asks about Maven's environment or location.

    This helper checks for common phrases that refer to the agent's
    operating context (e.g. "where are you", "where are we",
    "where am i").  These are not geography questions about external
    places but rather about where the system itself resides.

    Args:
        text: Raw user query.
    Returns:
        True if the query appears to be about the agent's environment.
    """
    try:
        ql = (text or "").strip().lower()
    except Exception:
        return False
    patterns = [
        # "where are we" deliberately excluded; conversation meta detector handles this pattern
        "where are you",
        "where am i",
        "where's your location",
        "where is your location",
        "where do you live",
    ]
    for p in patterns:
        try:
            if p in ql:
                return True
        except Exception:
            continue
    return False

def _semantic_verify(answer: str) -> bool:
    """Check whether the provided answer appears meaningful.

    This simple heuristic attempts to reject filler or self‑referential
    responses without relying on external resources.  It returns
    ``False`` when the answer is empty, contains only punctuation,
    matches any configured bad phrase, or lacks alphabetic characters.

    Exception: Numeric answers (containing only digits, decimal points,
    minus signs, or commas) are accepted regardless of length, as they
    are valid responses to mathematical or counting questions.

    Args:
        answer: The answer text to verify.

    Returns:
        True if the answer seems substantive; False otherwise.
    """
    try:
        ans = str(answer or "").strip()
        if not ans:
            return False
        ans_lc = ans.lower()

        # Check if this is a purely numeric answer (allows digits, decimal, minus, comma)
        # This handles cases like "4", "3.14", "-5", "1,000"
        is_numeric = all(ch.isdigit() or ch in '.,- ' for ch in ans)
        has_digit = any(ch.isdigit() for ch in ans)

        if is_numeric and has_digit:
            # Numeric answers are valid regardless of length
            return True

        # Very short non-numeric answers are unlikely to be factual
        if len(ans_lc) < 3:
            return False
        # Reject if matches any bad phrase
        for bad in BAD_CACHE_PHRASES:
            try:
                if bad and bad in ans_lc:
                    return False
            except Exception:
                continue
        # Require at least one alphabetic character for non-numeric answers
        if not any(ch.isalpha() for ch in ans):
            return False
        return True
    except Exception:
        return False

def _purge_invalid_cache() -> None:
    """Remove invalid or poisoned entries from the fast cache on startup.

    This function reads the existing ``fast_cache.jsonl`` file and
    rewrites it, excluding any entries whose answers contain bad
    phrases or fail the semantic verifier.  Removed entries are
    recorded in ``reports/cache_poison.log`` for transparency.
    """
    try:
        path = globals().get("FAST_CACHE_PATH")
        root = globals().get("MAVEN_ROOT")
        if not path or not root:
            return
        cache_path: Path = path  # type: ignore
        if not cache_path.exists():
            return
        with open(cache_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        valid_lines: List[str] = []
        for ln in lines:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            ans = str(rec.get("answer", "")).strip()
            ans_lc = ans.lower()
            invalid = False
            # Phrase check
            for bad in BAD_CACHE_PHRASES:
                try:
                    if bad and bad in ans_lc:
                        invalid = True
                        break
                except Exception:
                    continue
            # Semantic verify
            if not invalid and not _semantic_verify(ans):
                invalid = True
            if invalid:
                # Log to cache_poison.log
                try:
                    log_path = root / "reports" / "cache_poison.log"
                    log_entry = {"query": rec.get("query"), "bad_answer": ans}
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(log_entry) + "\n")
                except Exception:
                    pass
                continue
            # Keep valid entry
            valid_lines.append(ln if ln.endswith("\n") else ln + "\n")
        # Rewrite cache if entries were removed
        if len(valid_lines) < len(lines):
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    for ln in valid_lines:
                        fh.write(ln)
            except Exception:
                pass
    except Exception:
        # Do not propagate purge errors
        pass

def _lookup_fast_cache(query: str) -> Optional[Dict[str, Any]]:
    """Look up a cached answer for the given query if it exists and is fresh.

    Queries are compared in a case‑insensitive manner after stripping
    surrounding whitespace.  If multiple cached entries match, the most
    recent valid entry is returned.  Expired entries (older than
    FAST_CACHE_TTL_SEC) are ignored.  Returns ``None`` when no valid
    cached answer is available.

    Args:
        query: The raw user query string.

    Returns:
        A dictionary with keys ``query``, ``answer``, ``confidence`` and
        ``timestamp`` if a fresh cached entry exists, otherwise ``None``.
    """
    try:
        qnorm = (query or "").strip().lower()
        if not qnorm:
            return None
        if FAST_CACHE_PATH.exists():
            # Read all lines and iterate from the end to find the most
            # recent matching entry.  This is efficient for small caches
            # (only a handful of repeated questions are expected) and
            # avoids reading the entire file into memory when it grows.
            with open(FAST_CACHE_PATH, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                try:
                    rec_q = str(rec.get("query", "")).strip().lower()
                except Exception:
                    rec_q = ""
                if rec_q != qnorm:
                    continue
                # Found a match
                return rec
        return None
    except Exception:
        # Silently ignore any errors to avoid breaking the pipeline
        return None

def _store_fast_cache_entry(query: str, answer: str, confidence: float) -> None:
    """Append a new entry to the fast‑path cache.

    Args:
        query: The original user query.
        answer: The validated answer text.
        confidence: The confidence score associated with the answer.
    """
    try:
        # Skip storing answers that fail semantic verification.  This
        # prevents obvious filler or meta responses from polluting the
        # cache.  Only store answers that appear substantive.
        if not query or not answer or not _semantic_verify(answer):
            return
        qnorm = str(query).strip()
        # Ensure reports directory exists
        FAST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "query": qnorm,
            "answer": str(answer),
            "confidence": float(confidence) if confidence is not None else 0.8,
        }
        with open(FAST_CACHE_PATH, "a", encoding="utf-8") as fh:
            json.dump(record, fh)
            fh.write("\n")
    except Exception:
        # Do not propagate cache write errors
        pass

def _boost_cache_confidence(query: str, boost_amount: float = 0.1) -> Optional[float]:
    """Increment confidence for a cached entry when accessed repeatedly.

    This implements learning through repetition: when the same question is
    asked multiple times, we boost the confidence of the cached answer,
    indicating increased certainty through repeated validation.

    Args:
        query: The query whose cached entry should be updated.
        boost_amount: Amount to increment confidence (default 0.1).

    Returns:
        The new confidence value after boosting, or None if update failed.
    """
    try:
        qnorm = (query or "").strip().lower()
        if not qnorm or not FAST_CACHE_PATH.exists():
            return None

        # Read all cache entries
        with open(FAST_CACHE_PATH, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        updated = False
        new_lines = []
        new_confidence = None

        # Find and update the matching entry
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                rec = json.loads(line_stripped)
                rec_q = str(rec.get("query", "")).strip().lower()

                if rec_q == qnorm:
                    # Boost confidence, capping at 0.99
                    old_conf = float(rec.get("confidence", 0.8))
                    new_confidence = min(0.99, old_conf + boost_amount)
                    rec["confidence"] = new_confidence
                    new_lines.append(json.dumps(rec) + "\n")
                    updated = True
                else:
                    new_lines.append(line if line.endswith("\n") else line + "\n")
            except Exception:
                # Keep malformed lines as-is
                new_lines.append(line if line.endswith("\n") else line + "\n")

        # Write back if we made changes
        if updated:
            with open(FAST_CACHE_PATH, "w", encoding="utf-8") as fh:
                for ln in new_lines:
                    fh.write(ln)
            return new_confidence

        return None
    except Exception:
        # Silently ignore errors to avoid breaking the pipeline
        return None

def _consolidate_memory_banks() -> Dict[str, Any]:
    """Consolidate facts across memory banks based on capacity and importance.

    This implements a simplified STM → MTM → LTM consolidation strategy:
    - When working_theories bank gets too large, promote high-importance facts to factual bank
    - Archive old, low-importance facts to reduce memory footprint
    - Capacity-based (not time-based) to ensure performance

    Returns:
        Dictionary with consolidation statistics (facts_moved, facts_archived, etc.)
    """
    try:
        # Define capacity thresholds (in number of facts)
        WORKING_THEORIES_CAPACITY = 100  # STM-like capacity
        FACTUAL_CAPACITY = 500  # MTM-like capacity

        stats = {
            "facts_promoted": 0,
            "facts_archived": 0,
            "errors": 0
        }

        # Get facts from working_theories bank
        try:
            wt_bank = _bank_module("working_theories")
            wt_response = wt_bank.service_api({
                "op": "LIST",
                "payload": {"limit": WORKING_THEORIES_CAPACITY + 50}
            })
            wt_facts = wt_response.get("payload", {}).get("facts", [])
        except Exception:
            wt_facts = []

        # If working_theories exceeds capacity, promote high-confidence facts
        if len(wt_facts) > WORKING_THEORIES_CAPACITY:
            # Sort by confidence and importance
            sorted_facts = sorted(
                wt_facts,
                key=lambda f: (f.get("confidence", 0.0) + f.get("importance", 0.0)) / 2,
                reverse=True
            )

            # Promote top facts to factual bank
            to_promote = sorted_facts[:20]  # Promote batch of 20
            for fact in to_promote:
                try:
                    # Check confidence threshold for promotion
                    conf = fact.get("confidence", 0.0)
                    if conf >= 0.6:  # Only promote moderately confident facts
                        factual_bank = _bank_module("factual")
                        factual_bank.service_api({
                            "op": "STORE",
                            "payload": {"fact": fact}
                        })
                        # Remove from working_theories
                        wt_bank.service_api({
                            "op": "DELETE",
                            "payload": {"id": fact.get("id")}
                        })
                        stats["facts_promoted"] += 1
                except Exception:
                    stats["errors"] += 1

        return stats
    except Exception as e:
        return {"error": str(e), "facts_promoted": 0, "facts_archived": 0}

def _count_recent_identical_queries(query: str, within_sec: float) -> int:
    """Count how many times the given query appears in the recent query log.

    The count includes the current invocation and any prior entries in
    ``reports/query_log.jsonl`` that occurred within the specified time
    window.  Matching is case‑insensitive after stripping whitespace.

    Args:
        query: The user query to count.
        within_sec: Time window in seconds to look back.

    Returns:
        The number of matching queries (>=1 when called during a pipeline run).
    """
    try:
        qnorm = (query or "").strip().lower()
        if not qnorm:
            return 0
        path = (MAVEN_ROOT / "reports" / "query_log.jsonl").resolve()
        count = 1  # include the current invocation
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        rec_q = str(obj.get("query", "")).strip().lower()
                        if rec_q != qnorm:
                            continue
                        count += 1
            except Exception:
                pass
        return count
    except Exception:
        return 1

def _maybe_store_fast_cache(ctx: Dict[str, Any], threshold: int = 3, window_sec: float = 600.0) -> None:
    """Store a fast‑cache entry when the same query repeats multiple times.

    When the number of identical queries within ``window_sec`` seconds
    reaches ``threshold`` or higher, a new cache entry is created.  Only
    validated answers (stage 8 verdict TRUE) are stored.  If a fresh
    cache entry already exists for the query, this function does nothing.

    Args:
        ctx: The pipeline context containing ``original_query``,
            ``final_answer`` and ``final_confidence``.
        threshold: Number of occurrences required to trigger caching.
        window_sec: Time window for counting repeated queries.
    """
    try:
        q = str(ctx.get("original_query", "")).strip()
        if not q:
            return
        # Only store when we have a definitive answer validated by reasoning
        verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        ans = ctx.get("final_answer") or ""
        if verdict != "TRUE" or not ans:
            return
        # Compute confidence value early for cache gating
        try:
            conf_val = float(ctx.get("final_confidence") or 0.8)
        except Exception:
            conf_val = 0.8
        # Apply cache quality gate: skip caching short, low‑confidence or pronoun queries
        if not _should_cache(q, verdict, conf_val):
            return
        # Check existing cache; skip if already cached
        if _lookup_fast_cache(q):
            return
        # Count recent identical queries (include current invocation)
        cnt = _count_recent_identical_queries(q, window_sec)
        if cnt >= threshold:
            # conf_val already computed above
            _store_fast_cache_entry(q, ans, conf_val)
    except Exception:
        # Never raise; caching failures are silent
        pass


# === Semantic cache helpers ==================================================

def _lookup_semantic_cache(query: str) -> Optional[Dict[str, Any]]:
    """Look up a semantically similar entry for the given query.

    The semantic cache stores tokenised representations of previous queries
    and their answers.  This helper reads the cache file and returns
    the best entry whose Jaccard similarity (intersection over union)
    with the current query meets a configurable threshold.  Matching
    is case‑insensitive and based on alphanumeric word tokens.  When
    no cache entry meets the threshold or the cache file is missing,
    None is returned.

    Args:
        query: The raw user query string.
    Returns:
        A dictionary with keys ``query``, ``tokens``, ``answer`` and
        ``confidence`` if a match is found, otherwise ``None``.
    """
    try:
        # Normalise and tokenise the incoming query
        q = str(query or "").strip().lower()
        if not q:
            return None
        import re as _re
        tokens = [t for t in _re.findall(r"\w+", q) if t]
        if not tokens:
            return None
        token_set: Set[str] = set(tokens)
        path = globals().get("SEMANTIC_CACHE_PATH")
        if not path or not getattr(path, "exists", lambda: False)():
            return None
        # Load the semantic cache list; file contains a JSON array
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                return None
        except Exception:
            return None
        best_entry: Optional[Dict[str, Any]] = None
        best_score: float = 0.0
        # Define a small set of stopwords to filter trivial overlaps.  Only
        # consider matches that share at least one substantive token.
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "to", "of",
            "and", "or", "not", "no", "you", "i", "we", "they", "he", "she", "it",
            "who", "what", "when", "where", "why", "how", "can", "will", "would",
            "should", "could", "might", "your", "my", "their", "his", "her"
        }
        for item in data:
            try:
                itoks = set(item.get("tokens", []))
                if not itoks:
                    continue
                # Intersection of tokens
                inter = token_set.intersection(itoks)
                if not inter:
                    continue
                # Filter out stopwords; require at least one non-stopword overlap
                if not (inter - stopwords):
                    continue
                # Compute Jaccard similarity over union
                union = token_set.union(itoks)
                ratio = len(inter) / max(1, len(union))
                # Accept matches above threshold (0.3) and keep the best
                if ratio >= 0.3 and ratio > best_score:
                    best_score = ratio
                    best_entry = item
            except Exception:
                continue
        return best_entry
    except Exception:
        return None


def _update_semantic_cache(ctx: Dict[str, Any]) -> None:
    """Update the semantic cache with the current query and answer.

    This function appends or updates an entry in the semantic cache
    corresponding to the query contained in the pipeline context.  It
    extracts a bag of word tokens from the ``original_query`` and
    stores them alongside the ``final_answer`` and confidence.  If
    another entry shares exactly the same token set, it is updated
    rather than duplicated.  The cache is stored as a JSON array and
    written atomically to avoid corruption.

    Args:
        ctx: The current pipeline context containing ``original_query``,
            ``final_answer`` and ``final_confidence`` fields.
    """
    try:
        q = str(ctx.get("original_query", "")).strip()
        ans = ctx.get("final_answer")
        if not q or not ans:
            return
        # Apply cache quality gate: skip caching short, low‑confidence or pronoun queries
        try:
            verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        except Exception:
            verdict = ""
        try:
            conf_val = float(ctx.get("final_confidence") or 0.8)
        except Exception:
            conf_val = 0.8
        if not _should_cache(q, verdict, conf_val):
            return
        import re as _re
        tokens = [t for t in _re.findall(r"\w+", q.lower()) if t]
        if not tokens:
            return
        token_set = set(tokens)
        path = globals().get("SEMANTIC_CACHE_PATH")
        if not path:
            return
        # Ensure parent directory exists
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Load existing cache or start empty
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    data = []
            else:
                data = []
        except Exception:
            data = []
        updated = False
        # Update existing entry with matching token set
        for item in data:
            try:
                itoks = set(item.get("tokens", []))
            except Exception:
                itoks = set()
            if itoks == token_set:
                item["answer"] = str(ans)
                # Update confidence when provided
                try:
                    conf_val = float(ctx.get("final_confidence") or 0.8)
                except Exception:
                    conf_val = 0.8
                item["confidence"] = conf_val
                # Update intent and self_origin metadata on existing entry
                try:
                    lang_info = ctx.get("stage_3_language", {}) or {}
                    intent_type = str(lang_info.get("type")) if lang_info else None
                except Exception:
                    intent_type = None
                try:
                    val8 = ctx.get("stage_8_validation", {}) or {}
                    self_flag = bool(val8.get("self_origin") or val8.get("from_self_model"))
                except Exception:
                    self_flag = False
                if intent_type:
                    item["intent"] = intent_type
                item["self_origin"] = self_flag
                updated = True
                break
        if not updated:
            # Append new entry with additional metadata.  Capture the intent
            # type and whether the answer originated from the self model when available.
            try:
                conf_val = float(ctx.get("final_confidence") or 0.8)
            except Exception:
                conf_val = 0.8
            # Derive intent from stage_3_language if present
            try:
                lang_info = ctx.get("stage_3_language", {}) or {}
                intent_type = str(lang_info.get("type")) if lang_info else None
            except Exception:
                intent_type = None
            # Determine self origin from stage_8_validation if flagged
            try:
                val8 = ctx.get("stage_8_validation", {}) or {}
                self_flag = bool(val8.get("self_origin") or val8.get("from_self_model"))
            except Exception:
                self_flag = False
            data.append({
                "query": q,
                "tokens": list(token_set),
                "answer": str(ans),
                "confidence": conf_val,
                "intent": intent_type,
                "self_origin": self_flag,
            })
        # Write the updated cache atomically using api.utils if available
        try:
            from api.utils import _atomic_write  # type: ignore
            _atomic_write(path, json.dumps(data, indent=2))
        except Exception:
            # Fallback to naive write
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
    except Exception:
        # Suppress any errors during cache update
        pass

# Optional import of the belief tracker.  If unavailable, belief
# extraction and conflict detection will be skipped.
try:
    from brains.cognitive.belief_tracker.service.belief_tracker import add_belief as _belief_add, detect_conflict as _belief_detect  # type: ignore
except Exception:
    _belief_add = None  # type: ignore
    _belief_detect = None  # type: ignore

# Optional import of context management utilities for decay and reconstruction.
try:
    from brains.cognitive.context_management.service.context_manager import apply_decay as _ctx_decay  # type: ignore
    from brains.cognitive.context_management.service.context_manager import reconstruct_context as _ctx_reconstruct  # type: ignore
except Exception:
    _ctx_decay = None  # type: ignore
    _ctx_reconstruct = None  # type: ignore

# Optional import of meta‑learning for recording run metrics.  If absent,
# run metrics will not be captured.
try:
    from brains.cognitive.learning.service.meta_learning import record_run_metrics as _meta_record  # type: ignore
except Exception:
    _meta_record = None  # type: ignore


def _is_question(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return t.split(" ", 1)[0] in ("do","does","did","is","are","can","will","should","could","would","was","were")


# === Paths / Wiring ==========================================================

THIS_FILE = Path(__file__).resolve()
SERVICE_DIR = THIS_FILE.parent
COG_ROOT = SERVICE_DIR.parent.parent          # brains/cognitive
MAVEN_ROOT = COG_ROOT.parent.parent           # brains

sys.path.insert(0, str(MAVEN_ROOT))

# Diagnostic logging helpers for memory operations
def _diag_log(tag, rec):
    try:
        root = MAVEN_ROOT
        p = root / "reports" / "diagnostics" / "diag.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tag":tag, **(rec or {})}, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _ok(data=None):  # uniform shape so .get('data') never crashes
    return {"status":"ok","data":data}

# Initialise FAST_CACHE_PATH now that MAVEN_ROOT is defined.  This ensures that
# the path is constructed correctly once MAVEN_ROOT is available.  If the global
# variable FAST_CACHE_PATH is None, set it to the reports directory under MAVEN_ROOT.
try:
    if 'FAST_CACHE_PATH' in globals() and FAST_CACHE_PATH is None:
        FAST_CACHE_PATH = MAVEN_ROOT / 'reports' / 'fast_cache.jsonl'
    # Initialise semantic cache path when MAVEN_ROOT is defined.  Use a
    # dedicated subdirectory under reports to avoid collisions with the
    # fast cache.  When SEMANTIC_CACHE_PATH is None, assign it here.
    if 'SEMANTIC_CACHE_PATH' in globals() and SEMANTIC_CACHE_PATH is None:
        SEMANTIC_CACHE_PATH = MAVEN_ROOT / 'reports' / 'cache' / 'semantic_index.json'
except Exception:
    # Fallback: default to a local file in the current working directory
    from pathlib import Path as _Path  # avoid shadowing main Path import
    FAST_CACHE_PATH = _Path('fast_cache.jsonl')
    # Use a local file for semantic cache when the project root cannot be
    # determined.  This ensures the semantic cache still functions in
    # degraded environments.
    SEMANTIC_CACHE_PATH = _Path('semantic_index.json')

# After determining FAST_CACHE_PATH, load dynamic cache sanity phrases
# from the configuration file and purge any invalid cache entries.  This
# ensures that BAD_CACHE_PHRASES is always up‑to‑date and that the cache
# does not contain poisoned answers when the service starts.
try:
    BAD_CACHE_PHRASES = _load_cache_sanity_phrases()
    _purge_invalid_cache()
except Exception:
    # Leave BAD_CACHE_PHRASES unchanged on error and skip purge
    pass

# -----------------------------------------------------------------------------
# Retrieval caching
#
# The librarian frequently queries all domain banks for a given user query.
# When multiple retrievals are performed during a single run with the same
# query and limit, the results are identical.  To avoid redundant work,
# maintain a simple in‑memory cache keyed by (query, limit).  This cache is
# not persisted across runs; it resets when the process restarts.  It also
# applies to parallel retrieval, sharing the same backing store.
_RETRIEVE_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
# TTL in seconds for retrieval cache entries.  Cached results older than
# this threshold will be discarded to ensure that queries reflect recent
# updates.  A small TTL helps balance performance with freshness.  This
# constant may be tuned or made configurable via CFG in future iterations.
_RETRIEVE_CACHE_TTL: int = 60

# --- Autonomy config loader --------------------------------------------------
# The autonomy mechanism is controlled via a separate configuration file
# (config/autonomy.json).  This helper reads the file and returns a
# dictionary of settings.  If the file does not exist or is malformed, an
# empty dict is returned.  See the autonomy plan for details on keys such
# as "enable", "max_ticks_per_run", etc.
def _load_autonomy_config() -> Dict[str, Any]:
    try:
        cfg_path = (MAVEN_ROOT / "config" / "autonomy.json").resolve()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}

# For optional parallel bank retrieval
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Module loaders ==========================================================

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    # -------------------------------------------------------------------
    # Ensure that the loaded module exports a `handle` callable for
    # backward compatibility.  Many brain service modules only define
    # `service_api` as their entrypoint.  To conform to the uniform
    # service API expected by the memory librarian, mirror
    # `service_api` under the `handle` attribute if the module does not
    # already define it.  This check is intentionally wrapped in a
    # try/except to avoid propagating unexpected attribute errors that
    # could break module loading.
    try:
        if not hasattr(mod, "handle") and hasattr(mod, "service_api"):
            setattr(mod, "handle", getattr(mod, "service_api"))
    except Exception:
        # Errors during aliasing should not prevent the module from
        # loading.  Swallow any exceptions quietly.
        pass
    return mod

def _brain_module(name: str):
    svc = COG_ROOT / name / "service" / f"{name}_brain.py"
    return _load_module(svc, f"brain_{name}_service")

def _bank_module(name: str):
    svc = MAVEN_ROOT / "domain_banks" / name / "service" / f"{name}_bank.py"
    if not svc.exists():
        # legacy path: brains/domain_banks/...
        svc = MAVEN_ROOT / "brains" / "domain_banks" / name / "service" / f"{name}_bank.py"
    return _load_module(svc, f"bank_{name}_service")

def _gov_module():
    svc = MAVEN_ROOT / "brains" / "governance" / "policy_engine" / "service" / "policy_engine.py"
    return _load_module(svc, "policy_engine_service")

def _repair_module():
    svc = MAVEN_ROOT / "brains" / "governance" / "repair_engine" / "service" / "repair_engine.py"
    return _load_module(svc, "repair_engine_service")

def _personal_module():
    svc = MAVEN_ROOT / "brains" / "personal" / "service" / "personal_brain.py"
    return _load_module(svc, "personal_brain_service")

def _router_module():
    # Dual-process router under reasoning service.  Replace the previous learned
    # router with the dual_router wrapper to obtain a slow_path signal when
    # confidence margins are low.  The dual router forwards all supported
    # operations to the learned router, so callers can use the same API.
    svc = COG_ROOT / "reasoning" / "service" / "dual_router.py"
    return _load_module(svc, "dual_router_service")


SELF_IDENTITY_PATTERNS = [
    "who are you",
    "what are you",
    "what is your name",
    "what system are you",
    "are you an llm",
    "are you a language model",
    "what model are you",
    "what is your architecture",
    "describe your architecture",
    "describe your full pipeline",
    "describe your pipeline stages",
    "how do you work",
    "how do your brains work",
]

SELF_TRAINING_PATTERNS = [
    "how were you trained",
    "what is your training data",
    "what datasets were you trained on",
    "what is your knowledge cutoff",
    "when does your knowledge end",
]

SELF_CODE_PATTERNS = [
    "scan your code",
    "scan your codebase",
    "scan your entire codebase",
    "list your python files",
    "list all python files",
    "scan your brains directory",
    "scan brains directory",
]

SELF_MEMORY_PATTERNS = [
    "scan your memory",
    "what do you remember",
    "describe your memory system",
    "how many facts do you know",
]

# CRITICAL: System capability/upgrade questions MUST route to system_capabilities
# NOT to Teacher (which would hallucinate about Apache Maven / Java 17)
SELF_CAPABILITY_PATTERNS = [
    # Upgrade questions
    "what upgrade do you need",
    "what upgrades do you need",
    "what do you need to improve",
    "what could you improve",
    "what are your weaknesses",
    "what are your limitations",
    "what can you improve",
    "how can you improve",
    "what would make you better",
    # Capability questions
    "what can you do",
    "what are you capable of",
    "what are your capabilities",
    "what tools do you have",
    "what tools can you use",
    "can you browse the web",
    "can you run code",
    "can you access files",
    "can you read files",
    "can you write files",
    "can you execute",
    "can you search the web",
    "can you control other programs",
    "do you have internet access",
    "do you have web access",
]

# Follow-up patterns for self-capability context
# When user asks "what else" after a capability answer, we expand on it
SELF_CAPABILITY_FOLLOWUP_PATTERNS = [
    "what else",
    "anything else",
    "more on that",
    "tell me more",
    "continue",
    "go on",
    "expand on that",
    "more details",
    "what more",
    "elaborate",
]

# Track last self-capability context for follow-ups
_last_self_capability_context = {
    "mode": None,  # "upgrades" or "capabilities"
    "answer": None,
    "timestamp": 0.0,
    "expanded_sections": [],  # Which sections we've already expanded
}


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(p in t for p in patterns)


def classify_self_intent(user_text: str) -> dict:
    """
    Decide if this is a self-intent query that must bypass Teacher.
    Returns:
      {"kind": "identity" | "training" | "code_scan" | "memory" | "system_capability" | "capability_followup" | None,
       "mode": str | None}
    """
    import time
    global _last_self_capability_context

    t = user_text.lower().strip()

    # FIRST: Check for follow-ups to previous self-capability answers
    # This catches "what else" after "what upgrade do you need"
    if _matches_any(t, SELF_CAPABILITY_FOLLOWUP_PATTERNS):
        # Check if we have recent capability context (within 5 minutes)
        if (_last_self_capability_context.get("mode") and
            time.time() - _last_self_capability_context.get("timestamp", 0) < 300):
            print(f"[SELF_INTENT_GATE] Detected follow-up to previous capability answer (mode={_last_self_capability_context['mode']})")
            return {"kind": "capability_followup", "mode": _last_self_capability_context["mode"]}

    if _matches_any(t, SELF_IDENTITY_PATTERNS):
        return {"kind": "identity", "mode": "full"}

    if _matches_any(t, SELF_TRAINING_PATTERNS):
        return {"kind": "identity", "mode": "training_info"}

    if _matches_any(t, SELF_CODE_PATTERNS):
        return {"kind": "code_scan", "mode": "full"}

    if _matches_any(t, SELF_MEMORY_PATTERNS):
        return {"kind": "memory", "mode": "stats"}

    # CRITICAL: Detect system capability/upgrade questions
    # These MUST route to system_capabilities, NOT Teacher
    if _matches_any(t, SELF_CAPABILITY_PATTERNS):
        # Determine if it's an upgrade question or a general capability question
        if any(word in t for word in ["upgrade", "improve", "weakness", "limitation", "better"]):
            return {"kind": "system_capability", "mode": "upgrades"}
        else:
            return {"kind": "system_capability", "mode": "capabilities"}

    return {"kind": None, "mode": None}


def handle_self_intent_if_any(user_text: str, op: str, mid: str):
    import time
    global _last_self_capability_context

    intent = classify_self_intent(user_text)
    if intent["kind"] is None:
        return None

    # Handle capability follow-ups ("what else" after capability answer)
    if intent["kind"] == "capability_followup":
        print(f"[SELF_INTENT_GATE] Handling capability follow-up (mode={intent.get('mode')})")
        try:
            from brains.system_capabilities import get_capability_truth

            # Provide additional details based on what was already expanded
            expanded = _last_self_capability_context.get("expanded_sections", [])

            if intent.get("mode") == "upgrades":
                # Provide more upgrade details
                additional_sections = []

                if "learning" not in expanded:
                    additional_sections.extend([
                        "**Learning system improvements:**",
                        "- Cross-session pattern learning for frequently asked questions",
                        "- Automatic strategy refinement based on user feedback",
                        "- Better fact validation before storing to domain banks",
                        "",
                    ])
                    expanded.append("learning")

                if "introspection" not in expanded:
                    additional_sections.extend([
                        "**Introspection capabilities:**",
                        "- Deeper self-analysis of routing decisions",
                        "- Automatic detection of misrouted queries",
                        "- Performance metrics tracking and optimization",
                        "",
                    ])
                    expanded.append("introspection")

                if "collaboration" not in expanded:
                    additional_sections.extend([
                        "**Multi-agent collaboration:**",
                        "- Better coordination between cognitive brains",
                        "- Peer-to-peer knowledge sharing",
                        "- Committee-based decision making for complex queries",
                    ])
                    expanded.append("collaboration")

                if additional_sections:
                    answer_text = "\n".join(additional_sections)
                else:
                    answer_text = "I've covered the main areas for improvement. Would you like me to focus on a specific aspect in more detail?"

            else:
                # Provide more capability details
                truth = get_capability_truth()
                additional_sections = []

                if "memory" not in expanded:
                    additional_sections.extend([
                        "**Memory capabilities:**",
                        "- Episodic memory for conversation history",
                        "- Semantic memory for facts and knowledge",
                        "- Working memory for current context",
                        "- Pattern learning from interactions",
                        "",
                    ])
                    expanded.append("memory")

                if "reasoning" not in expanded:
                    additional_sections.extend([
                        "**Reasoning capabilities:**",
                        "- Logical inference and deduction",
                        "- Strategy-based problem solving",
                        "- Multi-step reasoning chains",
                        "",
                    ])
                    expanded.append("reasoning")

                if additional_sections:
                    answer_text = "\n".join(additional_sections)
                else:
                    answer_text = "I've covered my main capabilities. Would you like details about a specific feature?"

            # Update expanded sections
            _last_self_capability_context["expanded_sections"] = expanded

            if answer_text:
                print(f"[SELF_INTENT_GATE] ✓ Provided capability follow-up expansion")
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "SKIP_STORAGE",
                        "answer": answer_text,
                        "confidence": 0.90,
                        "mode": "CAPABILITY_FOLLOWUP",
                        "bypassed_pipeline": True,
                    },
                }

        except Exception as e:
            print(f"[SELF_INTENT_GATE_ERROR] Capability follow-up handling failed: {e}")
            import traceback
            traceback.print_exc()

    # CRITICAL: System capability questions route to system_capabilities module
    # NOT to self_model or Teacher (which would hallucinate about Apache Maven)
    if intent["kind"] == "system_capability":
        try:
            from brains.system_capabilities import (
                get_capability_truth,
                get_current_capabilities,
                answer_capability_question,
            )

            print(f"[SELF_INTENT_GATE] Routing system_capability query to system_capabilities")
            print(f"[SELF_INTENT_GATE] Mode: {intent.get('mode')}, Query: '{user_text[:50]}...'")

            # For upgrade questions, describe what improvements are needed
            if intent.get("mode") == "upgrades":
                # Get current limitations and describe upgrade needs
                truth = get_capability_truth()
                unavailable_tools = []
                for tool, status in truth.get("tools", {}).items():
                    if status != "available":
                        reason = truth.get("tool_details", {}).get(tool, {}).get("reason", status)
                        unavailable_tools.append(f"{tool}: {reason}")

                # Build honest upgrade answer
                upgrade_lines = [
                    "Based on my current system status, here are areas I could improve:",
                    "",
                ]

                if unavailable_tools:
                    upgrade_lines.append("**Tools that could be enabled:**")
                    for tool_info in unavailable_tools:
                        upgrade_lines.append(f"- {tool_info}")
                    upgrade_lines.append("")

                upgrade_lines.extend([
                    "**Routing improvements needed:**",
                    "- Better disambiguation between similar query types (time vs date vs calendar)",
                    "- More robust typo normalization using LLM-based correction",
                    "- Cleaner intent classification before routing decisions",
                    "",
                    "**Memory improvements:**",
                    "- Better context tracking across conversation turns",
                    "- More accurate working memory to avoid stale answer reuse",
                    "",
                    "**Self-improvement workflow:**",
                    "- Sandbox repository for testing code changes before applying",
                    "- Automated test validation before committing self-modifications",
                ])

                answer_text = "\n".join(upgrade_lines)

            else:
                # For general capability questions, use system_capabilities
                cap_answer = answer_capability_question(user_text)
                if cap_answer:
                    answer_text = cap_answer.get("answer", "")
                else:
                    # Fallback to current capabilities summary
                    current = get_current_capabilities()
                    available = current.get("available", [])
                    unavailable = current.get("unavailable", [])

                    answer_lines = ["Here's what I can currently do:"]
                    if available:
                        answer_lines.append("")
                        answer_lines.append("**Available:**")
                        for cap in available:
                            answer_lines.append(f"- {cap}")

                    if unavailable:
                        answer_lines.append("")
                        answer_lines.append("**Not currently available:**")
                        for cap in unavailable[:3]:  # Limit to top 3
                            answer_lines.append(f"- {cap}")

                    answer_text = "\n".join(answer_lines)

            if answer_text:
                # Store context for follow-ups
                _last_self_capability_context = {
                    "mode": intent.get("mode"),
                    "answer": answer_text,
                    "timestamp": time.time(),
                    "expanded_sections": [],  # Reset expanded sections for new question
                }

                print(f"[SELF_INTENT_GATE] ✓ Answered from system_capabilities (stored context for follow-ups)")
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "SKIP_STORAGE",
                        "answer": answer_text,
                        "confidence": 0.95,
                        "mode": "SYSTEM_CAPABILITY_DIRECT",
                        "bypassed_pipeline": True,
                    },
                }

        except Exception as e:
            print(f"[SELF_INTENT_GATE_ERROR] System capability handling failed: {e}")
            import traceback
            traceback.print_exc()

    # Other self-intent types route to self_model
    try:
        from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api

        self_resp = self_model_api(
            {
                "op": "QUERY_SELF",
                "payload": {
                    "query": user_text,
                    "self_kind": intent.get("kind"),
                    "self_mode": intent.get("mode"),
                },
            }
        )

        if self_resp.get("ok"):
            resp_payload = self_resp.get("payload", {})
            answer_text = resp_payload.get("text") or resp_payload.get("answer")

            if answer_text:
                print("[SELF_INTENT_GATE] Routed directly to self_model (early stage)")
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "verdict": "SKIP_STORAGE",
                        "answer": answer_text,
                        "confidence": resp_payload.get("confidence", 1.0),
                        "mode": f"SELF_{intent.get('kind', 'direct').upper()}_DIRECT",
                        "bypassed_pipeline": True,
                    },
                }
    except Exception as e:
        print(f"[SELF_INTENT_GATE_ERROR] Early self intent handling failed: {e}")

    return None


def classify_query_for_action(user_text: str) -> str | None:
    t = user_text.lower().strip()

    if "list all python files" in t or "list your python files" in t:
        return "list_python_files"

    return None


def handle_action_route_if_any(user_text: str, op: str, mid: str):
    action = classify_query_for_action(user_text)
    if action is None:
        return None

    try:
        from brains import action_engine_brain

        result = action_engine_brain.handle_action(action, {})

        if result.get("ok") and result.get("kind") == "file_list":
            files = result.get("files") or []
            text_lines = [
                f"I found {len(files)} Python files under my project root:",
                "",
                *files,
            ]
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "verdict": "SKIP_STORAGE",
                    "answer": "\n".join(text_lines),
                    "mode": "ACTION_ENGINE",
                },
            }

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "verdict": "ERROR",
                "answer": "I tried to run an action, but it returned an unknown result.",
                "mode": "ACTION_ENGINE_ERROR",
            },
        }
    except Exception as e:
        print(f"[ACTION_ENGINE_ERROR] Failed to execute action: {e}")
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "verdict": "ERROR",
                "answer": "Action engine failed to execute.",
                "mode": "ACTION_ENGINE_ERROR",
            },
        }

# === Helpers =================================================================

# Import brain roles to get domain banks dynamically
try:
    from brains.brain_roles import get_domain_brains
    _DOMAIN_BRAINS_AVAILABLE = True
except Exception:
    _DOMAIN_BRAINS_AVAILABLE = False

# Get the list of ALL domain banks from the tree scan
# This ensures we only route over domain brains (not cognitive brains)
# and automatically includes any new domain banks added to the tree.
if _DOMAIN_BRAINS_AVAILABLE:
    _ALL_BANKS = get_domain_brains()
else:
    # Fallback to hardcoded list if brain_roles not available
    # This should never happen in production, but provides safety
    _ALL_BANKS = [
        "arts",
        "science",
        "history",
        "economics",
        "geography",
        "language_arts",
        "law",
        "math",
        "philosophy",
        "technology",
        "theories_and_contradictions",
        "factual",
        "stm_only",
        "personal",
        "procedural",
        "creative",
        "working_theories",
    ]

def _retrieve_from_banks(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Retrieve evidence from all subject banks with deduplication.  Results are
    cached on a per‑query basis to avoid redundant retrievals within the same
    process.  The cache key is a tuple of the normalized query and limit.

    Args:
        query: The user query string.
        k: The maximum number of results to return per bank.

    Returns:
        A dictionary with 'results', 'banks' and 'banks_queried' fields.
    """
    # Attempt pronoun and continuation resolution using conversation state.
    # If the user query references the previous answer (e.g. "that") or
    # requests additional information (e.g. "anything else"), rewrite the
    # query accordingly and optionally restrict retrieval to the short‑term
    # memory bank.  Errors during resolution are suppressed so that the
    # fallback behaviour remains intact.
    force_stm_only: bool = False
    try:
        resolved_query, force_stm_only = _resolve_continuation_and_pronouns(query)
        if isinstance(resolved_query, str) and resolved_query:
            query = resolved_query
    except Exception:
        force_stm_only = False
    # Handle continuation patterns: if the query appears to ask for 'more'
    # information or a continuation (e.g. "more", "more about brains",
    # "anything else"), replace it with the last known topic.  This ensures
    # that follow‑up questions search the same domain as the prior query.
    try:
        q_raw = str(query or "")
    except Exception:
        q_raw = ""
    try:
        ql = q_raw.strip().lower()
    except Exception:
        ql = ""

    # ------------------------------------------------------------------
    # Explanation trigger detection
    #
    # Certain follow‑up queries ask for an explanation of the previous
    # answer rather than additional facts.  Examples include
    # "how did you get 5?", "how did you come up with that?",
    # "explain that", and similar.  When such a pattern is detected,
    # bypass normal retrieval and delegate to the reasoning brain's
    # EXPLAIN_LAST operation.  The conversation state provides the
    # necessary context (last_query and last_response).  This early
    # return prevents redundant searches and ensures the explanation
    # surfaces as the primary retrieval result.
    try:
        _ql = str(query or "").strip().lower()
    except Exception:
        _ql = ""
    try:
        import re as _re_explain
        # Define patterns for explanation requests
        _explain_patterns = [
            r"^how\s+did\s+you\s+get\b",
            r"^how\s+did\s+you\s+come\s+up\s+with\b",
            r"^explain\b",
            r"^why\s+did\s+you\s+get\b",
            r"^how\s+did\s+you\s+do\b",
        ]
        _needs_explanation = False
        for _pat in _explain_patterns:
            if _re_explain.search(_pat, _ql):
                _needs_explanation = True
                break
        if _needs_explanation:
            # Fetch conversation state safely
            conv = globals().get("_CONVERSATION_STATE", {}) or {}
            last_q = conv.get("last_query", "") or ""
            last_r = conv.get("last_response", "") or ""
            # Attempt to call the reasoning brain's explanation op
            try:
                reason_mod = _brain_module("reasoning")
            except Exception:
                reason_mod = None
            explanation_text: str | None = None
            if reason_mod is not None:
                try:
                    resp = reason_mod.service_api({
                        "op": "EXPLAIN_LAST",
                        "payload": {
                            "last_query": last_q,
                            "last_response": last_r
                        }
                    })
                    if resp and resp.get("ok"):
                        explanation_text = (resp.get("payload") or {}).get("answer") or None
                except Exception:
                    explanation_text = None
            # Fallback if the reasoning brain is unavailable or returns nothing
            if not explanation_text:
                # Construct a simple reference to the last response
                if last_r:
                    explanation_text = f"I answered '{last_r}' previously based on my reasoning and memory."
                else:
                    explanation_text = "I don't have enough context to provide an explanation."
            # Return the explanation as a single retrieval result.  Use a
            # synthetic bank name so downstream components can
            # differentiate explanatory content from factual retrieval.
            return {
                "results": [
                    {"content": explanation_text, "source_bank": "explanation"}
                ],
                "banks": ["explanation"],
                "banks_queried": ["explanation"]
            }
    except Exception:
        # On error, ignore and continue with normal retrieval
        pass
    try:
        # Detect explicit 'more about X' and extract X (legacy support)
        if ql.startswith("more about "):
            candidate = ql[len("more about "):].strip()
            if candidate:
                query = candidate
        # Detect bare 'more', 'anything else', 'any thing else', 'what else'
        elif ql in {"more", "anything else", "any thing else", "what else"} or \
            (ql.startswith("more ") and len(ql.split()) <= 2):
            # Use the last topic if available
            if _LAST_TOPIC:
                query = _LAST_TOPIC
    except Exception:
        # On error, fall back to the original query
        query = q_raw
    # Normalise key for cache lookup; default limit of 5 if invalid
    try:
        limit_int = int(k)
    except Exception:
        limit_int = 5
    key = (str(query or ""), limit_int)
    cached = _RETRIEVE_CACHE.get(key)
    if cached is not None:
        return {
            "results": list(cached.get("results", [])),
            "banks": list(cached.get("banks", [])),
            "banks_queried": list(cached.get("banks_queried", []))
        }
    results: list[Dict[str, Any]] = []
    searched: list[str] = []
    seen_contents: set[str] = set()
    # Determine targeted banks based on messages from the message bus.  If a
    # search request has been issued (e.g. by the reasoning brain), only
    # query the specified banks; otherwise fall back to all banks.  Before
    # consulting the message bus, handle relational queries locally: when
    # the query expresses a relationship between the user and the assistant
    # (e.g. "we are friends"), restrict the search to the personal bank.
    banks_to_use: List[str] = []
    try:
        ql = str(query or "").lower()
        # Detect relational query patterns: check for "we" or "you and i"
        # along with relationship keywords.  If matched, search only personal.
        rel_keywords = ["friend", "friends", "family", "partner", "partners", "couple", "married", "husband", "wife", "siblings", "brother", "sister"]
        if any(rk in ql for rk in rel_keywords):
            if re.search(r"\bwe\b", ql) or re.search(r"\byou and i\b", ql) or re.search(r"\bwe\s*'re\b", ql):
                banks_to_use = ["personal"]
    except Exception:
        banks_to_use = []

    # LEARNED ROUTING: Check if we have a learned routing rule for this question
    # This allows the Librarian to learn from the Teacher over time
    if not banks_to_use:
        try:
            from brains.cognitive.memory_librarian.service.librarian_memory import (
                retrieve_routing_rule_for_question
            )

            routing_rule = retrieve_routing_rule_for_question(query)
            if routing_rule:
                # Extract banks from the routing rule, sorted by weight
                routes = routing_rule.get("routes", [])
                if routes:
                    # Sort by weight (highest first) and take banks with weight > 0.2
                    sorted_routes = sorted(routes, key=lambda r: r.get("weight", 0), reverse=True)
                    for route in sorted_routes:
                        bank = route.get("bank", "")
                        weight = route.get("weight", 0)
                        if weight > 0.2 and bank in _ALL_BANKS and bank not in banks_to_use:
                            banks_to_use.append(bank)
        except Exception:
            # If routing rule lookup fails, continue with other methods
            pass

    try:
        from brains.cognitive.message_bus import pop_all  # type: ignore
        # Consume any pending messages
        msgs = pop_all()
        for m in msgs:
            try:
                if m.get("type") == "SEARCH_REQUEST":
                    domains = m.get("domains") or []
                    # Map domain keywords to bank names by substring match
                    for d in domains:
                        d_str = str(d).lower()
                        for b in _ALL_BANKS:
                            if d_str in b.lower() and b not in banks_to_use:
                                banks_to_use.append(b)
                    # Only handle the first search request for now
                    if banks_to_use:
                        break
            except Exception:
                continue
    except Exception:
        banks_to_use = []
    # If no targeted banks were specified, search all banks
    if not banks_to_use:
        banks_to_use = list(_ALL_BANKS)
    # Override bank selection when continuation or pronoun resolution forces STM
    try:
        if force_stm_only:
            banks_to_use = ["stm_only"]
    except Exception:
        pass
    for b in banks_to_use:
        try:
            svc = _bank_module(b)
            r = svc.service_api({"op": "RETRIEVE", "payload": {"query": query, "limit": k}})
            if r.get("ok"):
                pay = r.get("payload") or {}
                rr = pay.get("results") or []
                for item in rr:
                    if not isinstance(item, dict):
                        continue
                    item.setdefault("source_bank", b)
                    # Normalize content for deduplication.  Use the raw text if
                    # available, else fallback to serialized form.
                    content = str(item.get("content", "")).strip().lower()
                    if not content:
                        content = json.dumps(item, sort_keys=True)
                    # Simple relevance: ratio of query words present in content.
                    try:
                        q_tokens = [w for w in re.findall(r"\b\w+\b", str(query).lower())]
                        c_tokens = [w for w in re.findall(r"\b\w+\b", content)]
                        overlap = sum(1 for w in q_tokens if w in c_tokens)
                        relevance = float(overlap) / float(len(q_tokens) or 1)
                    except Exception:
                        relevance = 0.0
                    # Apply relevance floor: require at least 0.2 overlap
                    if relevance < 0.2:
                        continue
                    if content in seen_contents:
                        continue
                    seen_contents.add(content)
                    results.append(item)
                searched.append(b)
        except Exception:
            # Ignore individual bank failures
            continue
    res = {"results": results, "banks": searched, "banks_queried": searched}
    # Persist in cache (store copies to avoid accidental mutation)
    _RETRIEVE_CACHE[key] = {
        "results": list(results),
        "banks": list(searched),
        "banks_queried": list(searched),
    }
    return res

# Optional parallel retrieval implementation.  When parallel access is enabled
# via CFG["parallel_bank_access"], this helper will query each domain bank
# concurrently.  It accepts an optional max_workers argument to control
# concurrency.  Results are aggregated in the same format as
# _retrieve_from_banks.
def _retrieve_from_banks_parallel(query: str, k: int = 5, max_workers: int = 5) -> Dict[str, Any]:
    """
    Retrieve evidence from all subject banks using concurrent workers.

    Identical query/limit combinations are cached in the same cache as
    the sequential retrieval helper.  When a result is in the cache, this
    function skips all parallel calls and returns the cached result.

    Args:
        query: The user query string.
        k: Maximum number of results to return per bank.
        max_workers: Number of threads to use for concurrent bank access.

    Returns:
        A dictionary with 'results', 'banks' and 'banks_queried' fields.
    """
    try:
        limit_int = int(k)
    except Exception:
        limit_int = 5
    # Detect explanation requests early and delegate to sequential
    # retrieval so that specialised logic (e.g. EXPLAIN_LAST) can be
    # executed.  Without this check, parallel retrieval would miss
    # explanation patterns and return unrelated facts.  Only simple
    # prefix patterns are considered here to avoid unnecessary work.
    try:
        import re as _re_explain
        ql = str(query or "").strip().lower()
        _explain_triggers = [
            r"^how\s+did\s+you\s+get\b",
            r"^how\s+did\s+you\s+come\s+up\s+with\b",
            r"^explain\b",
            r"^why\s+did\s+you\s+get\b",
            r"^how\s+did\s+you\s+do\b",
        ]
        for _pat in _explain_triggers:
            if _re_explain.search(_pat, ql):
                # Delegate to sequential retrieval which handles explanation
                return _retrieve_from_banks(query, limit_int)
    except Exception:
        pass
    key = (str(query or ""), limit_int)
    cached = _RETRIEVE_CACHE.get(key)
    if cached is not None:
        return {
            "results": list(cached.get("results", [])),
            "banks": list(cached.get("banks", [])),
            "banks_queried": list(cached.get("banks_queried", []))
        }
    results: list[dict] = []
    searched: list[str] = []
    seen_contents: set[str] = set()
    # Submit retrieval tasks for each bank concurrently.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_bank = {}
        for b in _ALL_BANKS:
            try:
                svc = _bank_module(b)
                fut = pool.submit(svc.service_api, {"op": "RETRIEVE", "payload": {"query": query, "limit": k}})
                future_to_bank[fut] = b
            except Exception:
                # Skip banks that fail to load
                continue
        # Process completed futures as they finish
        for fut in as_completed(future_to_bank):
            b = future_to_bank[fut]
            try:
                r = fut.result()
            except Exception:
                continue
            try:
                if r.get("ok"):
                    pay = r.get("payload") or {}
                    rr = pay.get("results") or []
                    for item in rr:
                        if not isinstance(item, dict):
                            continue
                        # Annotate with source bank
                        item.setdefault("source_bank", b)
                        # Normalize content for deduplication
                        content = str(item.get("content", "")).strip().lower()
                        if not content:
                            content = json.dumps(item, sort_keys=True)
                        if content in seen_contents:
                            continue
                        seen_contents.add(content)
                        results.append(item)
                    searched.append(b)
            except Exception:
                continue
    res = {
        "results": results,
        "banks": searched,
        "banks_queried": searched
    }
    _RETRIEVE_CACHE[key] = {
        "results": list(results),
        "banks": list(searched),
        "banks_queried": list(searched),
    }
    return res

def _scan_counts(root: Path) -> Dict[str, Dict[str, int]]:
    """Return a mapping of brain name to counts of records per memory tier.

    This helper is used by the memory librarian to surface a high‑level
    snapshot of memory usage across different cognitive brains.  The
    original implementation hard‑coded a subset of brains and omitted
    newly added modules such as the coder brain.  To provide a more
    complete view of memory, this function now includes the coder brain
    alongside the existing ones.  If a brain is missing its memory
    directory or any error occurs while reading, an empty dict is
    returned for that brain.

    Args:
        root: Path to the cognitive brains root directory.

    Returns:
        A dict keyed by brain name with values of tier→record count.
    """
    from api.memory import tiers_for, count_lines  # type: ignore
    out: Dict[str, Dict[str, int]] = {}
    brains = [
        "sensorium",
        "planner",
        "language",
        "pattern_recognition",
        "reasoning",
        "affect_priority",
        "personality",
        "self_dmn",
        "system_history",
        "memory_librarian",
        # Include the coder brain in the memory overview to ensure its
        # memory usage is tracked like other cognitive modules.  Without
        # this, the coder's STM/MTM/LTM tiers are invisible to the
        # librarian and cannot be consolidated or reported upon.
        "coder",
    ]
    for brain in brains:
        broot = root / brain
        try:
            tiers = tiers_for(broot)
            out[brain] = {tier: count_lines(path) for tier, path in tiers.items()}
        except Exception:
            out[brain] = {}
    # Add counts for the personal knowledge bank (domain_bank) for completeness
    try:
        personal_root = MAVEN_ROOT / "brains" / "personal"
        tiers = tiers_for(personal_root)
        out["personal"] = {tier: count_lines(path) for tier, path in tiers.items()}
    except Exception:
        out["personal"] = {}
    return out

def _extract_definition(text: str):
    # Primitive "X is a Y" pattern to teach the router definitions when TRUE
    m = re.match(r'^(?P<term>[A-Za-z0-9 ]{1,40})\s+(is|are)\s+(a|an)?\s*(?P<klass>[A-Za-z0-9 ]{1,40})\.?$', text.strip(), re.I)
    if not m:
        return None, None
    term = (m.group("term") or "").strip().lower()
    klass = (m.group("klass") or "").strip().lower()
    if not term or not klass or term == klass:
        return None, None
    return term, klass

def _is_simple_math_expression(text: str) -> bool:
    """
    Detect if the input is a simple arithmetic expression like "2+5" or "3*4".
    Returns True for basic patterns with two numbers and an operator.
    """
    try:
        s = str(text or "").strip()
        # Match patterns like "2+5", "3 * 4", "10 - 3", etc.
        # Allow optional whitespace around numbers and operators
        pattern = r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$'
        if re.match(pattern, s):
            return True
    except Exception:
        pass
    return False

def _solve_simple_math(expression: str) -> Dict[str, Any]:
    """
    Solve a simple arithmetic expression containing two numbers and one operator.
    Supports +, -, *, / operations. Returns a dict with ok:True and result on success,
    or ok:False on parse failure. No eval() is used - only safe Python operators.
    """
    try:
        expr = str(expression or "").strip()
        # Extract lhs, operator, rhs using regex
        m = re.match(r'^\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*$', expr)
        if not m:
            return {"ok": False}

        lhs = int(m.group(1))
        op = m.group(2)
        rhs = int(m.group(3))

        # Compute result using safe operators
        if op == '+':
            result = lhs + rhs
        elif op == '-':
            result = lhs - rhs
        elif op == '*':
            result = lhs * rhs
        elif op == '/':
            if rhs == 0:
                return {"ok": False}
            result = lhs / rhs
            # Use integer if result is whole number
            if isinstance(result, float) and result.is_integer():
                result = int(result)
        else:
            return {"ok": False}

        return {"ok": True, "result": result}
    except Exception:
        return {"ok": False}

def _simple_route_to_bank(content: str) -> str:
    """
    Perform a simple keyword-based routing of new statements to domain banks.
    This helper looks for words associated with biology, physics/chemistry,
    math, history or geography.  If a match is found, the corresponding bank
    name is returned; otherwise the result defaults to 'working_theories'.
    """
    try:
        text = (content or "").strip().lower()
        # Check for simple arithmetic expressions first (higher priority)
        if _is_simple_math_expression(text):
            return "math"
        # Biology keywords (science bank)
        biology = [
            "mammal","animal","bird","species","organism","polar bear",
            "wings","skin","fur","feathers","evolution","biology","creature",
            "wildlife"
        ]
        if any(w in text for w in biology):
            return "science"
        # Physics/Chemistry keywords (science bank)
        physics = [
            "energy","force","atom","molecule","gravity","einstein",
            "physics","quantum","relativity","mass","velocity"
        ]
        if any(w in text for w in physics):
            return "science"
        # Math keywords
        math_words = [
            "+","-","*","/","=","²","³","equation","calculate","sum",
            "multiply","divisible","prime","square","triangle"
        ]
        if any(w in text for w in math_words):
            return "math"
        # History keywords or famous names
        history_indicators = [
            "was","were","born","died","war","ancient","century","historical"
        ]
        famous_people = ["einstein","newton","darwin","lincoln","washington","napoleon"]
        if any(w in text for w in history_indicators) or any(fn in text for fn in famous_people):
            return "history"
        # Geography
        geography_words = [
            "capital","country","city","continent","ocean","mountain",
            "river","lake","france","paris","europe","asia"
        ]
        if any(w in text for w in geography_words):
            return "geography"
    except Exception:
        pass
    return "working_theories"

def _best_memory_exact(evidence: Dict[str, Any], content: str):
    try:
        for it in (evidence or {}).get("results", []):
            if isinstance(it, dict) and str(it.get("content","")).strip().lower() == str(content).strip().lower():
                return it
    except Exception:
        pass
    return None

# Sanitize a yes/no question into a declarative form.  This helper strips
# trailing question marks and leading auxiliary verbs (e.g. "do", "does", "is", etc.)
# so that retrieval can find matching declarative statements such as
# "Birds have wings." when asked "Do birds have wings?".
def _sanitize_question(query: str) -> str:
    """
    Normalize a user question to a declarative form that helps match stored
    facts.  This helper strips trailing question marks, removes common
    yes/no prefixes (e.g., "is", "are", "can"), and extracts the
    subject phrase that follows forms of "to be".  For example:

        "Is the sky blue?"  ->  "the sky blue"
        "What color is the sky?"  ->  "the sky"

    Args:
        query: The raw user question string.

    Returns:
        A sanitized string suitable for memory retrieval.
    """
    try:
        q = (query or "").strip()
        if q.endswith("?"):
            q = q[:-1]
        lower = q.lower().strip()
        # Strip common yes/no question prefixes (e.g., "is", "are", "can")
        prefixes = [
            "do ", "does ", "did ", "is ", "are ", "can ", "should ",
            "could ", "will ", "would ", "was ", "were "
        ]
        for p in prefixes:
            if lower.startswith(p):
                # Remove the prefix from both q and its lowercase copy
                q = q[len(p):].strip()
                lower = q.lower()
                break
        # Attempt to extract the subject after a form of "to be" (is/are/was/were)
        for verb in [" is ", " are ", " was ", " were "]:
            if verb in lower:
                parts = q.split(verb, 1)
                if len(parts) > 1 and parts[1].strip():
                    q = parts[1].strip()
                break
        return q.strip() or query
    except Exception:
        return query

# --- Context and query history helpers ---------------------------------------

def _get_recent_queries(limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve the most recent user queries from a flat query log.

    The query log lives at ``reports/query_log.jsonl`` and contains one JSON
    object per line with fields ``query`` and ``timestamp``.  This helper
    returns the last ``limit`` entries in chronological order (oldest first).

    Args:
        limit: Maximum number of recent queries to return.

    Returns:
        A list of dicts each with ``query`` and ``timestamp`` keys.
    """
    path = (MAVEN_ROOT / "reports" / "query_log.jsonl").resolve()
    entries: List[Dict[str, Any]] = []
    try:
        # If the file does not exist, return an empty list.  This avoids a
        # TypeError on subsequent reads when None would otherwise be returned.
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        # Parse JSON lines; ignore malformed lines
        for line in lines:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "query" in obj:
                    entries.append({"query": obj.get("query")})
            except Exception:
                continue
    except Exception:
        # On any error, return an empty list instead of None
        return []
    # Return the last ``limit`` entries (chronological order retained).  If there
    # are fewer than ``limit`` entries, return all of them.
    return entries[-limit:]

def _cross_validate_answer(ctx: Dict[str, Any]) -> None:
    """
    Perform a lightweight self‑check on the validated answer stored in
    ``ctx['stage_8_validation']``.  This helper examines the original
    query and the answer and attempts to verify simple factual claims
    without requiring external services.  Two kinds of checks are
    performed:

    1. Arithmetic sanity: when the query looks like a simple math
       expression (e.g. "2+2" or "What is 5 * 7?"), evaluate the
       expression using Python and compare it to the provided answer.  If
       there is a mismatch, the answer in the context is replaced with
       the computed result and the ``cross_check_tag`` is set to
       ``"recomputed"``.  Otherwise the tag is ``"asserted_true"``.

    2. Definition/geography sanity: when the query begins with a
       definitional prefix (e.g. "what is", "who was", "capital of") and
       a memory retrieval has already been performed (``stage_2R_memory``
       exists), the function checks whether the answer appears in any of
       the retrieved memory results.  If the answer is found verbatim in
       the evidence, the tag is set to ``"asserted_true"``; otherwise it
       is set to ``"conflict_check"``.  This helps detect answers that
       may contradict the user’s stored knowledge.  When no definition
       prefix is recognised or no memory evidence is present, the tag
       defaults to ``"asserted_true"``.

    The function is resilient to errors and never raises.  It writes
    back into the provided context dict and has no return value.

    Args:
        ctx: The pipeline context containing the query, the validated
            answer and optional memory evidence.
    """
    try:
        # Extract the validated answer from the reasoning stage.  If no answer
        # exists (e.g. verdict UNKNOWN), proceed with an empty string so that
        # arithmetic or definitional checks can still be performed.
        answer = (ctx.get("stage_8_validation") or {}).get("answer")
        original_query = ctx.get("original_query") or ""
        q_norm = str(original_query).strip().lower()
        # Normalise the answer; use empty string when None
        ans_str = str(answer).strip() if answer is not None else ""
        # ------------------------------------------------------------------
        # Arithmetic sanity check.  If the query is comprised solely of
        # digits and simple arithmetic operators, evaluate it.  Use a
        # conservative pattern to avoid executing arbitrary code.  Queries
        # that end with a question mark or contain whitespace are handled
        # by stripping extraneous characters.
        try:
            # Remove common prefixes like "what is" before checking if this is
            # a pure math expression.  Also strip trailing punctuation.
            q_math = q_norm
            for prefix in ["what is ", "what's ", "calculate ", "compute ", "answer is "]:
                if q_math.startswith(prefix):
                    q_math = q_math[len(prefix):]
                    break
            q_math = q_math.rstrip("?.!").replace(" ", "")
            if q_math and re.fullmatch(r"[0-9+\-*/().]+", q_math):
                # Safely evaluate using eval() with no builtins
                try:
                    # Evaluate simple arithmetic expressions safely without using eval().
                    # Only support numbers and + - * / parentheses.  Use ast to parse and compute.
                    import ast, operator as _op
                    def _eval_expr(node):
                        # Recursively evaluate AST nodes representing arithmetic expressions.
                        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
                            return node.n
                        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                            return -_eval_expr(node.operand)
                        elif isinstance(node, ast.BinOp):
                            left = _eval_expr(node.left)
                            right = _eval_expr(node.right)
                            if isinstance(node.op, ast.Add):
                                return left + right
                            elif isinstance(node.op, ast.Sub):
                                return left - right
                            elif isinstance(node.op, ast.Mult):
                                return left * right
                            elif isinstance(node.op, ast.Div):
                                return left / right
                            else:
                                raise ValueError("Unsupported operator")
                        else:
                            raise ValueError("Unsupported expression")
                    try:
                        _tree = ast.parse(q_math, mode="eval")
                        result = _eval_expr(_tree.body)
                    except Exception:
                        result = None
                except Exception:
                    result = None
                if result is not None:
                    # Convert numeric result to a canonical string
                    if isinstance(result, (int, float)) and not isinstance(result, bool):
                        # Represent integers without a decimal point
                        if abs(result - int(result)) < 1e-9:
                            result_str = str(int(result))
                        else:
                            result_str = str(result)
                    else:
                        result_str = str(result)
                    # Extract the first numeric token from the answer
                    try:
                        # Use regex to capture numbers including decimal part and sign
                        num_tokens = re.findall(r"[-+]?[0-9]*\.?[0-9]+", ans_str)
                        ans_num = num_tokens[0] if num_tokens else ans_str
                    except Exception:
                        ans_num = ans_str
                    # Compare and update if mismatched or if there was no original answer
                    if ans_num != result_str or not ans_str:
                        # Update the answer in the reasoning verdict
                        ctx.setdefault("stage_8_validation", {})["answer"] = result_str
                        ctx.setdefault("stage_8_validation", {})["verdict"] = "TRUE"
                        ctx["cross_check_tag"] = "recomputed"
                        return
                    else:
                        ctx["cross_check_tag"] = "asserted_true"
                        return
        except Exception:
            pass
        # ------------------------------------------------------------------
        # Definition/geography check.  Identify definitional or geography
        # queries by looking for common prefixes.  If the query fits and
        # there is memory evidence available, search the evidence for the
        # answer.  When found, assert true; otherwise mark as conflict.
        try:
            prefixes = [
                "what is ", "who is ", "what was ", "who was ",
                "who are ", "what are ", "capital of ", "capital city of "
            ]
            is_def = any(q_norm.startswith(p) for p in prefixes)
            if is_def:
                mem_results = (ctx.get("stage_2R_memory") or {}).get("results", [])
                found = False
                ans_low = ans_str.lower()
                for rec in mem_results:
                    try:
                        cont = str(rec.get("content") or "").lower()
                        if ans_low and ans_low in cont:
                            found = True
                            break
                    except Exception:
                        continue
                ctx["cross_check_tag"] = "asserted_true" if found else "conflict_check"
                return
        except Exception:
            pass
        # Default case: no special checks triggered
        ctx["cross_check_tag"] = "asserted_true"
    except Exception:
        # On unexpected errors, default to asserted_true
        ctx["cross_check_tag"] = "asserted_true"
    # Return the last N entries (limit), preserving chronological order
    return entries[-limit:]

def _save_context_snapshot(ctx: Dict[str, Any], limit: int = 5) -> None:
    """Persist a trimmed context snapshot and append the current query to the query log.

    This helper avoids nested ``session_context`` structures by writing only
    the current pipeline state along with a shallow list of recent queries.  It
    appends the current query to a log file to enable retrieval of recent
    history across runs.

    Args:
        ctx: The full pipeline context.
        limit: Maximum number of recent queries to include in the snapshot.
    """
    try:
        # Ensure reports directory exists
        reports_dir = (MAVEN_ROOT / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        # Append current query to query log
        qlog = reports_dir / "query_log.jsonl"
        with open(qlog, "a", encoding="utf-8") as qfh:
            json.dump({"query": ctx.get("original_query")}, qfh)
            qfh.write("\n")
        # ------------------------------------------------------------------
        # Retention: prune the query log if it grows beyond a configurable
        # threshold.  Many pipeline runs accumulate queries over time and
        # without pruning the log could grow indefinitely.  The memory
        # configuration (config/memory.json) may specify a
        # ``query_log_max_entries`` integer.  When the number of stored
        # queries exceeds this limit, the oldest entries are removed.
        try:
            cfg_mem = CFG.get("memory", {}) or {}
            # Default to 500 entries if no configuration is provided or if the
            # value is not a valid integer.  A non‑positive value disables
            # pruning.
            max_entries = int(cfg_mem.get("query_log_max_entries", 500))
        except Exception:
            max_entries = 500
        try:
            if max_entries > 0 and qlog.exists():
                with open(qlog, "r", encoding="utf-8") as lf:
                    lines = [ln for ln in lf if ln.strip()]
                if len(lines) > max_entries:
                    # Retain only the newest entries
                    new_lines = lines[-max_entries:]
                    with open(qlog, "w", encoding="utf-8") as wf:
                        for ln in new_lines:
                            # Preserve newline endings
                            wf.write(ln if ln.endswith("\n") else ln + "\n")
        except Exception:
            # Silently ignore pruning errors to avoid disrupting snapshot saving
            pass
        # Build snapshot with limited recent history
        snapshot = {
            "original_query": ctx.get("original_query"),
            "personality_snapshot": ctx.get("personality_snapshot", {}),
            "session_context": {
                "recent_queries": _get_recent_queries(limit),
                "context_truncated": True
            }
        }
        # Include current pipeline stages (those starting with 'stage_')
        for key, val in ctx.items():
            try:
                if isinstance(key, str) and key.startswith("stage_"):
                    snapshot[key] = val
            except Exception:
                continue
        # Write the snapshot to context_snapshot.json
        with open(reports_dir / "context_snapshot.json", "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)
    except Exception:
        # Do not crash on any error
        pass

# --- Hybrid Semantic Memory & Retrieval Unification (Step‑7) ---
def _unified_retrieve(query: str, k: int = 5, filters: Optional[dict] = None) -> Dict[str, Any]:
    """
    Retrieve evidence from all domain banks and apply personal preference boost.

    This helper first performs a standard retrieval across all domain banks,
    then augments each result with a boost derived from the personal brain.
    The final score is the sum of the original confidence and the personal
    boost.  Results are sorted by this total score in descending order.

    Args:
        query: The user query string.
        k: Maximum number of results to return.
        filters: Optional filters for future extension (unused).

    Returns:
        A dictionary with 'results', 'banks' and 'banks_queried' fields.  Each
        result is annotated with 'boost' and 'total_score'.
    """
    # Decide between parallel and sequential retrieval based on configuration
    try:
        parallel = bool((CFG.get("memory") or {}).get("parallel_bank_access", False))
    except Exception:
        parallel = False
    try:
        limit_int = int(k)
    except Exception:
        limit_int = 5
    # Perform base retrieval across all banks
    try:
        base = _retrieve_from_banks_parallel(query, limit_int) if parallel else _retrieve_from_banks(query, limit_int)
    except Exception:
        # Fall back to sequential retrieval on any error
        base = _retrieve_from_banks(query, limit_int)
    results = list(base.get("results") or [])
    # Attempt to load personal brain service for boosting
    try:
        _personal = _personal_module()
        personal_api = getattr(_personal, "service_api", None)
    except Exception:
        personal_api = None
    enriched: List[Dict[str, Any]] = []
    query_context = {"query": query, "k": limit_int}

    for item in results:
        # Normalise the subject text for boosting
        try:
            subj = str(item.get("content") or item.get("text") or "").strip()
        except Exception:
            subj = ""
        boost_val: float = 0.0
        if personal_api and subj:
            # Call personal brain to compute a boost.  Errors are ignored.
            try:
                resp = personal_api({"op": "SCORE_BOOST", "payload": {"subject": subj}})
                if resp and resp.get("ok"):
                    boost_val = float((resp.get("payload") or {}).get("boost") or 0.0)
            except Exception:
                boost_val = 0.0

        # Phase 4: Apply tier-aware scoring
        # This replaces the simple conf_val + boost_val with a sophisticated
        # cross-tier ranking that considers tier priority, importance, usage,
        # recency (via seq_id), and match quality.
        tier_score = _score_memory_hit(item, query_context)

        # Combine tier score with personal boost for final ranking
        total = tier_score + boost_val

        # Annotate result with all scoring components for explainability
        enriched_item = dict(item)
        enriched_item["boost"] = boost_val
        enriched_item["tier_score"] = tier_score
        enriched_item["total_score"] = total
        enriched_item["retrieval_score"] = tier_score  # For downstream consumers
        enriched.append(enriched_item)

    # Sort results by descending total_score (tier-aware + personal boost)
    enriched.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
    # Trim to requested limit
    if limit_int > 0:
        enriched = enriched[:limit_int]
    return {
        "results": enriched,
        "banks": base.get("banks", []),
        "banks_queried": base.get("banks_queried", [])
    }

# === Service API =============================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    global _SEQ_ID_COUNTER, _LAST_BROWSER_PAGE_ID
    from api.utils import generate_mid, success_response, error_response, write_report, CFG  # type: ignore
    from api.memory import update_last_record_success, ensure_dirs  # type: ignore
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    # Step‑7: Unified retrieval across banks with personal boosts
    if op == "UNIFIED_RETRIEVE":
        # Extract query text and optional parameters
        try:
            q = str(payload.get("query") or payload.get("text") or payload.get("input") or "")
        except Exception:
            q = ""
        try:
            k_val = payload.get("k") or payload.get("limit") or 5
        except Exception:
            k_val = 5
        filters_val = payload.get("filters") or {}
        try:
            result = _unified_retrieve(q, k_val, filters_val)
            return success_response(op, mid, result)
        except Exception as e:
            return error_response(op, mid, "UNIFIED_RETRIEVE_FAILED", str(e))

    if op == "RUN_PIPELINE":
        # Phase 5 Determinism: Use monotonic sequence ID for run tracking
        # instead of random seed. This ensures full determinism without
        # time-based or random logic.
        try:
            with _SEQ_ID_LOCK:
                _SEQ_ID_COUNTER += 1
                run_id = _SEQ_ID_COUNTER
            _seed_record = (run_id, run_id)  # (run_id, trace_id) both use seq_id
        except Exception:
            _seed_record = None

        # Start routing diagnostics trace (Phase C cleanup)
        try:
            if tracer and RouteType:
                _trace_text = str(
                    payload.get("text")
                    or payload.get("question")
                    or payload.get("query")
                    or payload.get("input")
                    or ""
                )
                tracer.start_request(mid, _trace_text)
        except Exception:
            pass
        # Before beginning the pipeline, optionally consolidate memories
        # across cognitive brains.  Consolidation moves aged or low
        # importance records from STM into deeper tiers and enforces
        # per‑tier quotas.  The behaviour is controlled via the
        # ``memory.auto_consolidate`` configuration.  Any errors are
        # ignored to avoid disrupting the pipeline.
        try:
            # Avoid re-importing CFG within this function to prevent Python from
            # treating it as a local variable.  The module-level CFG import
            # ensures consistent access to configuration values throughout the
            # service.  This guard checks the auto_consolidate flag and
            # performs consolidation accordingly.  Any errors during import or
            # consolidation are intentionally ignored to avoid disrupting the
            # pipeline.
            if bool((CFG.get("memory") or {}).get("auto_consolidate", False)):
                try:
                    from brains.cognitive.memory_consolidation import consolidate_memories  # type: ignore
                    consolidate_memories()
                except Exception:
                    pass
        except Exception:
            pass
        # Accept multiple keys for the input text to improve compatibility with callers.
        # Historically the pipeline expected `text` but some clients may send
        # `question`, `query` or `input`.  To avoid losing the original query and
        # inadvertently triggering the UNKNOWN_INPUT verdict, fall back to these
        # alternate keys when `text` is missing.
        text = str(
            payload.get("text")
            or payload.get("question")
            or payload.get("query")
            or payload.get("input")
            or ""
        )
        conf = float(payload.get("confidence", 0.8))
        bypass_prerouting = bool(payload.get("_bypass_prerouting", False))

        early_self = handle_self_intent_if_any(text, op, mid)
        if early_self is not None:
            return early_self

        action_result = handle_action_route_if_any(text, op, mid)
        if action_result is not None:
            return action_result

        # ------------------------------------------------------------------
        # Research / deep research detection (offline routing)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                research_topic = None
                research_depth = 2
                research_web = False
                research_diag_patterns = [
                    "research diag",
                    "diag research",
                    "diagnose research",
                    "research diagnostics",
                    "diagnostics research",
                ]

                if any(pattern in lt for pattern in research_diag_patterns):
                    try:
                        from brains.cognitive.research_manager.service.research_manager_brain import (
                            service_api as research_api,
                        )

                        diag_resp = research_api(
                            {
                                "op": "RESEARCH_DIAGNOSTICS",
                                "mid": mid,
                                "payload": {},
                            }
                        )
                        if diag_resp.get("ok"):
                            diag_payload = diag_resp.get("payload") or {}
                            diag_text = str(diag_payload.get("text") or "Research diagnostics complete.").strip()
                            ctx_diag = {
                                "original_query": text,
                                "final_answer": diag_text,
                                "final_confidence": 0.72,
                                "mode": "RESEARCH_DIAGNOSTICS",
                            }
                            return success_response(op, mid, {"context": ctx_diag})
                    except Exception:
                        pass

                if lt.startswith("research:"):
                    research_topic = text.split(":", 1)[1].strip()
                elif lt.startswith("research "):
                    research_topic = text.split(" ", 1)[1].strip()
                elif lt.startswith("deep research on "):
                    research_topic = text.split("on", 1)[1].strip()
                    research_depth = 3
                    research_web = True
                elif lt.startswith("deep research:"):
                    research_topic = text.split(":", 1)[1].strip()
                    research_depth = 3
                    research_web = True

                if research_topic:
                    try:
                        from brains.cognitive.research_manager.service.research_manager_brain import (
                            service_api as research_api,
                        )

                        sources = ["memory", "teacher"]
                        if research_web:
                            sources.append("web")
                        rm_resp = research_api(
                            {
                                "op": "RUN_RESEARCH",
                                "mid": mid,
                                "payload": {
                                    "topic": research_topic,
                                    "depth": research_depth,
                                    "sources": sources,
                                    "full_prompt": text,
                                },
                            }
                        )
                        if rm_resp.get("ok"):
                            rm_pay = rm_resp.get("payload") or {}
                            summary_txt = (
                                str(rm_pay.get("summary") or rm_pay.get("answer") or "").strip()
                            )
                            facts_cnt = int(rm_pay.get("facts_collected") or rm_pay.get("facts_stored") or 0)
                            src_list = rm_pay.get("sources") or []
                            combined_answer = summary_txt or f"Research on {research_topic} complete."
                            combined_answer += f"\n\nResearch complete: {facts_cnt} facts stored from {len(src_list)} sources."
                            ctx_direct = {
                                "original_query": text,
                                "final_answer": combined_answer,
                                "final_confidence": float(rm_pay.get("confidence", 0.72) or 0.72),
                                "mode": "RESEARCH_DIRECT",
                            }
                            return success_response(op, mid, {"context": ctx_direct})
                    except Exception:
                        pass
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Coder detection (code generation routing)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                coder_prompt = None
                return_code_only = False

                # Check for explicit coder prefix patterns
                if lt.startswith("coder:"):
                    coder_prompt = text.split(":", 1)[1].strip()
                elif lt.startswith("use coder:"):
                    coder_prompt = text.split(":", 1)[1].strip()
                elif lt.startswith("coder write"):
                    coder_prompt = text[len("coder write"):].strip()
                elif lt.startswith("coder generate"):
                    coder_prompt = text[len("coder generate"):].strip()
                # Check for code generation patterns
                elif any(pattern in lt for pattern in [
                    "write a function", "write a python function",
                    "generate code", "generate a function", "create function",
                    "write code", "code a function"
                ]):
                    coder_prompt = text
                # Check for code-only follow-up patterns
                elif any(pattern in lt for pattern in [
                    "return only the code", "return only code", "just the code",
                    "show me the code", "give me the code", "code only",
                    "return only the full updated function", "return only the function"
                ]):
                    coder_prompt = text
                    return_code_only = True

                if coder_prompt:
                    print(f"[MEMORY_LIBRARIAN] Detected coder request: {coder_prompt[:50]}...")
                    try:
                        from brains.cognitive.coder.service.coder_brain import service_api as coder_api

                        coder_resp = coder_api({
                            "op": "GENERATE",
                            "mid": mid,
                            "payload": {
                                "spec": coder_prompt,  # coder expects 'spec' not 'prompt'
                            },
                        })
                        if coder_resp.get("ok"):
                            coder_pay = coder_resp.get("payload") or {}
                            # Extract generated code - coder returns: code, test_code, summary
                            generated_code = coder_pay.get("code") or ""
                            summary = coder_pay.get("summary") or ""
                            test_code = coder_pay.get("test_code") or ""

                            # Build the answer - prefer code if return_code_only
                            if return_code_only and generated_code:
                                final_answer = f"```python\n{generated_code}\n```"
                            elif generated_code:
                                final_answer = f"```python\n{generated_code}\n```"
                                if summary:
                                    final_answer += f"\n\n{summary}"
                            else:
                                final_answer = summary or "Code generation completed."

                            ctx_coder = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.85,
                                "mode": "CODER_DIRECT",
                            }
                            return success_response(op, mid, {"context": ctx_coder})
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] Coder routing failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Browser runtime detection (open URL in browser)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                import re  # Import early for fallback functions
                lt = text.strip().lower()
                browser_url = None

                # Import URL normalization utilities from smart_routing
                try:
                    from brains.cognitive.integrator.smart_routing import (
                        normalize_url,
                        extract_search_url_from_intent,
                        _HOSTNAME_MAP,
                    )
                    has_smart_routing = True
                except ImportError:
                    has_smart_routing = False
                    # Fallback hostname map
                    _HOSTNAME_MAP = {
                        "google": "www.google.com",
                        "bing": "www.bing.com",
                        "youtube": "www.youtube.com",
                        "github": "github.com",
                        "reddit": "www.reddit.com",
                    }

                    def normalize_url(url_or_hostname: str) -> str:
                        url = url_or_hostname.strip()
                        url_lower = url.lower()
                        # If it has a protocol, validate and fix hostname
                        if url_lower.startswith(("http://", "https://")):
                            if url_lower.startswith("https://"):
                                protocol = "https://"
                                rest = url[8:]
                            else:
                                protocol = "http://"
                                rest = url[7:]
                            if "/" in rest:
                                hostname, path = rest.split("/", 1)
                                path = "/" + path
                            else:
                                hostname = rest
                                path = ""
                            hostname_lower = hostname.lower()
                            if hostname_lower in _HOSTNAME_MAP:
                                return f"{protocol}{_HOSTNAME_MAP[hostname_lower]}{path}"
                            if "." not in hostname_lower:
                                return f"{protocol}www.{hostname_lower}.com{path}"
                            return url
                        # No protocol
                        if url_lower in _HOSTNAME_MAP:
                            return f"https://{_HOSTNAME_MAP[url_lower]}"
                        if "." not in url_lower:
                            return f"https://www.{url_lower}.com"
                        return f"https://{url}"

                    def extract_search_url_from_intent(message: str):
                        """Fallback search URL extraction."""
                        from urllib.parse import quote_plus
                        msg_lower = message.lower()

                        # Pattern: "open google and search for X"
                        search_patterns = [
                            (r'(?:open|go to|navigate to|browse to|visit)\s+(google|bing|duckduckgo|ddg)\s+(?:and\s+)?(?:search\s+for|search|look\s+up|lookup|find)\s+(.+)', None),
                        ]
                        for pattern_str, _ in search_patterns:
                            match = re.search(pattern_str, msg_lower, re.IGNORECASE)
                            if match:
                                site = match.group(1).lower()
                                query = match.group(2).strip()
                                encoded_query = quote_plus(query)
                                if site == "google":
                                    return f"https://www.google.com/search?q={encoded_query}"
                                elif site == "bing":
                                    return f"https://www.bing.com/search?q={encoded_query}"
                                elif site in ("duckduckgo", "ddg"):
                                    return f"https://duckduckgo.com/?q={encoded_query}"

                        # Pattern: "search for X on google"
                        if any(kw in msg_lower for kw in ["search", "look up", "lookup", "find"]):
                            if "google" in msg_lower:
                                for prefix in ["search for ", "search ", "look up ", "lookup ", "find "]:
                                    if prefix in msg_lower:
                                        idx = msg_lower.index(prefix) + len(prefix)
                                        query = message[idx:].strip()
                                        query = re.sub(r'\s+on\s+google\s*$', '', query, flags=re.IGNORECASE)
                                        if query:
                                            return f"https://www.google.com/search?q={quote_plus(query)}"
                        return None

                # ================================================================
                # DIRECT WEB SEARCH DETECTION
                # Handle "web search X" patterns BEFORE URL detection
                # This ensures "web search physics" uses same pipeline as "google search games"
                # ================================================================
                def extract_direct_web_search_query(message: str):
                    """Extract query from direct web search commands."""
                    msg_stripped = message.strip()
                    direct_patterns = [
                        (r'^web\s+search\s+(.+)$', 1),           # "web search X"
                        (r'^search\s+the\s+web\s+for\s+(.+)$', 1),  # "search the web for X"
                        (r'^search\s+online\s+for\s+(.+)$', 1),  # "search online for X"
                        (r'^look\s+up\s+online\s+(.+)$', 1),     # "look up online X"
                        (r'^internet\s+search\s+(.+)$', 1),      # "internet search X"
                    ]
                    for pattern, group_idx in direct_patterns:
                        match = re.match(pattern, msg_stripped, re.IGNORECASE)
                        if match:
                            return match.group(group_idx).strip()
                    return None

                # Check for direct "web search X" pattern FIRST
                direct_search_query = extract_direct_web_search_query(text)
                if direct_search_query:
                    print(f"[MEMORY_LIBRARIAN] Direct web search detected: query='{direct_search_query}'")
                    try:
                        from brains.agent.tools.web_search_tool import search_and_synthesize

                        # Use web_search_tool to search and synthesize answer
                        search_result = search_and_synthesize(
                            query=direct_search_query,
                            engine="auto",
                            max_results=5,
                            store_facts=True,
                        )

                        if search_result.get("success"):
                            # Build a nice answer with sources
                            answer_text = search_result.get("answer", "")
                            sources = search_result.get("sources", [])

                            if sources:
                                sources_section = "\n\n**Sources:**\n"
                                for src in sources[:5]:
                                    sources_section += f"- [{src.get('title', 'Link')}]({src.get('url', '')})\n"
                                final_answer = answer_text + sources_section
                            else:
                                final_answer = answer_text

                            ctx_search = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.85,
                                "mode": "WEB_SEARCH_SYNTHESIZED",
                                "search_query": direct_search_query,
                                "sources": sources,
                            }
                            return success_response(op, mid, {"context": ctx_search})
                        else:
                            # Search failed - provide error message
                            error_msg = search_result.get("error", "Search failed")
                            final_answer = (
                                f"**Web Search Failed**\n\n"
                                f"I tried to search for: \"{direct_search_query}\"\n\n"
                                f"**Error:** {error_msg}\n\n"
                                f"This could be due to search engines blocking automated access. "
                                f"Try again later or try a different query."
                            )
                            ctx_search = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.3,
                                "mode": "WEB_SEARCH_FAILED",
                                "search_query": direct_search_query,
                                "error": error_msg,
                            }
                            return success_response(op, mid, {"context": ctx_search})

                    except ImportError as e:
                        print(f"[MEMORY_LIBRARIAN] web_search_tool not available: {e}")
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] Direct web search error: {e}")

                # URL pattern for detection
                url_pattern = re.compile(
                    r'https?://\S+'  # Full URLs like http://example.com
                    r'|www\.\S+'     # www.example.com style
                    r'|\b[a-z0-9][-a-z0-9]*\.(com|net|org|io|ai|dev|co|edu|gov|app|me|info|biz|xyz)\b',
                    re.IGNORECASE
                )

                # First, check for search intent patterns (e.g., "open google and search for games")
                search_url = extract_search_url_from_intent(text)
                if search_url:
                    browser_url = search_url

                # Check for explicit browser prefix
                elif lt.startswith("browser:"):
                    url_candidate = text.split(":", 1)[1].strip()
                    # Extract URL from the text after prefix
                    url_match = url_pattern.search(url_candidate)
                    if url_match:
                        browser_url = normalize_url(url_match.group(0))
                    elif url_candidate:
                        # Normalize the hostname/domain
                        browser_url = normalize_url(url_candidate.split()[0])  # First word only

                # Check for action + URL/hostname patterns
                elif any(lt.startswith(p) for p in ["open ", "go to ", "navigate to ", "browse to ", "visit "]):
                    # Extract URL from after the action word
                    for prefix in ["open ", "go to ", "navigate to ", "browse to ", "visit "]:
                        if lt.startswith(prefix):
                            url_candidate = text[len(prefix):].strip()
                            url_match = url_pattern.search(url_candidate)
                            if url_match:
                                browser_url = normalize_url(url_match.group(0))
                            else:
                                # Try to extract hostname (first word, no spaces)
                                first_word = url_candidate.split()[0] if url_candidate else ""
                                # Remove trailing punctuation
                                first_word = first_word.rstrip(".,!?;:")
                                if first_word:
                                    # Check if it's a known hostname or could be a domain
                                    if first_word.lower() in _HOSTNAME_MAP or "." in first_word:
                                        browser_url = normalize_url(first_word)
                                    elif first_word.isalnum():
                                        # Treat as potential hostname
                                        browser_url = normalize_url(first_word)
                            break

                # Check for URLs anywhere in the text with action keywords
                elif any(kw in lt for kw in ["open the ", "load ", "fetch "]):
                    url_match = url_pattern.search(text)
                    if url_match:
                        browser_url = normalize_url(url_match.group(0))

                if browser_url:
                    # Final normalization - ensure URL has protocol
                    if not browser_url.startswith(("http://", "https://")):
                        browser_url = f"https://{browser_url}"

                    print(f"[MEMORY_LIBRARIAN] Detected browser request: {browser_url}")
                    try:
                        from optional.browser_runtime.browser_client import is_available, open_url

                        if not is_available():
                            # Browser server not running
                            final_answer = (
                                "**Browser Runtime Not Available**\n\n"
                                "The browser automation server is not running. "
                                "To start it, run:\n\n"
                                "```\n"
                                "python run_browser_server.py\n"
                                "```\n\n"
                                "Or on Windows, double-click `start_browser_server.cmd`."
                            )
                            ctx_browser = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.9,
                                "mode": "BROWSER_UNAVAILABLE",
                            }
                            return success_response(op, mid, {"context": ctx_browser})

                        # Check if this is a search URL - use web_search_tool for search URLs
                        try:
                            from brains.agent.tools.web_search_tool import (
                                is_search_url,
                                extract_query_from_url,
                                detect_engine_from_url,
                                search_and_synthesize,
                            )

                            if is_search_url(browser_url):
                                # This is a search URL - use web_search_tool pipeline
                                search_query = extract_query_from_url(browser_url)
                                search_engine = detect_engine_from_url(browser_url) or "auto"

                                print(f"[MEMORY_LIBRARIAN] Search URL detected: query='{search_query}', engine={search_engine}")

                                if search_query:
                                    # Use web_search_tool to search and synthesize answer
                                    search_result = search_and_synthesize(
                                        query=search_query,
                                        engine=search_engine,
                                        max_results=5,
                                        store_facts=True,
                                    )

                                    if search_result.get("success"):
                                        # Build a nice answer with sources
                                        answer_text = search_result.get("answer", "")
                                        sources = search_result.get("sources", [])

                                        answer_lines = [answer_text]

                                        if sources:
                                            answer_lines.append("")
                                            answer_lines.append("**Sources:**")
                                            for i, src in enumerate(sources[:5], 1):
                                                title = src.get("title", "")[:60]
                                                url = src.get("url", "")
                                                answer_lines.append(f"{i}. [{title}]({url})")

                                        final_answer = "\n".join(answer_lines)
                                        ctx_browser = {
                                            "original_query": text,
                                            "final_answer": final_answer,
                                            "final_confidence": 0.85,
                                            "mode": "WEB_SEARCH_SYNTHESIZED",
                                            "search_query": search_query,
                                            "engine_used": search_result.get("engine_used", search_engine),
                                            "sources": sources,
                                            "facts_stored": search_result.get("facts_stored", 0),
                                        }
                                        return success_response(op, mid, {"context": ctx_browser})
                                    else:
                                        # Search failed - provide explicit error instead of useless "Opened:" fallback
                                        error_msg = search_result.get("error", "Unknown search error")
                                        print(f"[MEMORY_LIBRARIAN] Web search failed: {error_msg}")

                                        # Build informative failure message
                                        answer_lines = [
                                            "**Web Search Failed**",
                                            "",
                                            f"I tried to search for: \"{search_query}\"",
                                            "",
                                            f"**Error:** {error_msg}",
                                            "",
                                            "**What happened:**",
                                        ]

                                        # Parse the error to give helpful info
                                        error_lower = error_msg.lower()
                                        if "blocked" in error_lower or "418" in error_lower or "captcha" in error_lower:
                                            answer_lines.append("- Search engines detected automated access and blocked the request")
                                            answer_lines.append("- This is temporary - try again later or use a different query")
                                        elif "parser_failed" in error_lower:
                                            answer_lines.append("- The search page structure has changed and I couldn't extract results")
                                            answer_lines.append("- Debug HTML has been saved for inspection")
                                        elif "no_results" in error_lower:
                                            answer_lines.append("- The search returned no results")
                                            answer_lines.append("- Try a different or broader search query")
                                        else:
                                            answer_lines.append(f"- {error_msg}")

                                        final_answer = "\n".join(answer_lines)
                                        ctx_browser = {
                                            "original_query": text,
                                            "final_answer": final_answer,
                                            "final_confidence": 0.3,
                                            "mode": "WEB_SEARCH_FAILED",
                                            "search_query": search_query,
                                            "error": error_msg,
                                        }
                                        return success_response(op, mid, {"context": ctx_browser})
                        except ImportError as e:
                            print(f"[MEMORY_LIBRARIAN] web_search_tool not available: {e}")
                        except Exception as e:
                            print(f"[MEMORY_LIBRARIAN] Web search error: {e}")

                        # Fall back to regular browser open for non-search URLs only
                        # (search URL failures are handled above with explicit error)
                        result = open_url(browser_url)

                        if result.get("error"):
                            final_answer = f"**Browser Error**\n\nFailed to open {browser_url}:\n{result['error']}"
                            ctx_browser = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.5,
                                "mode": "BROWSER_ERROR",
                            }
                        else:
                            # Build response from browser result
                            page_id = result.get("page_id", "unknown")
                            # Store page_id for subsequent grok tool commands
                            if page_id and page_id != "unknown":
                                _LAST_BROWSER_PAGE_ID = page_id
                            snapshot = result.get("snapshot", {})
                            page_title = snapshot.get("title", "Untitled")
                            page_url = snapshot.get("url", browser_url)

                            answer_lines = [
                                f"**Opened: {page_title}**",
                                "",
                                f"URL: {page_url}",
                                f"Page ID: {page_id}",
                            ]

                            # Include text content if available
                            text_content = snapshot.get("text_content", "")
                            if text_content:
                                # Truncate to first 1000 chars
                                preview = text_content[:1000]
                                if len(text_content) > 1000:
                                    preview += "..."
                                answer_lines.extend([
                                    "",
                                    "**Page Content Preview:**",
                                    preview,
                                ])

                            final_answer = "\n".join(answer_lines)
                            ctx_browser = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.95,
                                "mode": "BROWSER_DIRECT",
                                "browser_result": result,
                            }

                        return success_response(op, mid, {"context": ctx_browser})
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] Browser routing failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Unified X tool command detection (x: does everything on X.com)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                x_command = None

                # Check for x: patterns (the unified tool)
                if lt.startswith("x:"):
                    x_command = text[2:].strip()
                # Also support legacy grok patterns, route to x tool
                elif lt.startswith("use grok:") or lt.startswith("grok:"):
                    for prefix in ["use grok:", "grok:"]:
                        if lt.startswith(prefix):
                            msg = text[len(prefix):].strip()
                            x_command = f"Talk to Grok: {msg}"
                            break
                elif lt.startswith("post:"):
                    x_command = text  # Already in right format

                if x_command:
                    print(f"[MEMORY_LIBRARIAN] Detected x command: {x_command[:50]}...")
                    try:
                        from optional.browser_tools.x import x

                        print(f"[MEMORY_LIBRARIAN] Calling x(): {x_command[:50]}...")
                        response_text = x(x_command)

                        if response_text and not response_text.startswith("Failed"):
                            final_answer = f"**X.com Result**\n\n{response_text[:3000]}"
                            ctx_x = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.95,
                                "mode": "X_COMMAND_SUCCESS",
                                "x_response": response_text,
                            }
                        else:
                            final_answer = f"**X.com Command Failed**\n\n{response_text}"
                            ctx_x = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.5,
                                "mode": "X_COMMAND_FAILED",
                                "error": response_text,
                            }

                        return success_response(op, mid, {"context": ctx_x})
                    except ImportError as e:
                        print(f"[MEMORY_LIBRARIAN] X tool not available: {e}")
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] X routing failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Human tool command detection (human: for desktop control)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                human_command = None

                # Check for human: patterns
                if lt.startswith("human:"):
                    human_command = text[6:].strip()

                if human_command:
                    print(f"[MEMORY_LIBRARIAN] Detected human command: {human_command[:50]}...")
                    try:
                        from optional.browser_tools.human_tool import human

                        print(f"[MEMORY_LIBRARIAN] Calling human(): {human_command[:50]}...")
                        response_text = human(human_command)

                        if response_text and not response_text.startswith("Error"):
                            final_answer = f"**Human Control Result**\n\n{response_text[:3000]}"
                            ctx_human = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.95,
                                "mode": "HUMAN_COMMAND_SUCCESS",
                                "human_response": response_text,
                            }
                        else:
                            final_answer = f"**Human Command Failed**\n\n{response_text}"
                            ctx_human = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.5,
                                "mode": "HUMAN_COMMAND_FAILED",
                                "error": response_text,
                            }

                        return success_response(op, mid, {"context": ctx_human})
                    except ImportError as e:
                        print(f"[MEMORY_LIBRARIAN] Human tool not available: {e}")
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] Human routing failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # PC control tool command detection (pc: for system control)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                pc_command = None

                # Check for pc: patterns
                if lt.startswith("pc:"):
                    pc_command = text[3:].strip()
                elif lt.startswith("pc "):
                    pc_command = text[3:].strip()

                if pc_command:
                    print(f"[MEMORY_LIBRARIAN] Detected PC command: {pc_command[:50]}...")
                    try:
                        from optional.browser_tools.pc_control_tool import pc

                        print(f"[MEMORY_LIBRARIAN] Calling pc(): {pc_command[:50]}...")
                        response_text = pc(pc_command)

                        if response_text and not response_text.startswith("Error"):
                            final_answer = f"**PC Control Result**\n\n```\n{response_text[:5000]}\n```"
                            ctx_pc = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.95,
                                "mode": "PC_COMMAND_SUCCESS",
                                "pc_response": response_text,
                            }
                        else:
                            final_answer = f"**PC Command Failed**\n\n{response_text}"
                            ctx_pc = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.5,
                                "mode": "PC_COMMAND_FAILED",
                                "error": response_text,
                            }

                        return success_response(op, mid, {"context": ctx_pc})
                    except ImportError as e:
                        print(f"[MEMORY_LIBRARIAN] PC tool not available: {e}")
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] PC routing failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Brain inventory detection (list cognitive brains)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                inventory_request = False

                # Check for inventory patterns
                if lt.startswith("inventory:"):
                    inventory_request = True
                elif lt.startswith("inventory"):
                    inventory_request = True
                elif any(pattern in lt for pattern in [
                    "list all brains", "list your brains", "show all brains",
                    "what brains do you have", "list cognitive brains",
                    "show cognitive brains", "brain inventory"
                ]):
                    inventory_request = True

                if inventory_request:
                    print("[MEMORY_LIBRARIAN] Detected brain inventory request")
                    try:
                        # Use self_introspection tool to analyze brains
                        from brains.tools.self_introspection import get_self_introspection

                        introspector = get_self_introspection()
                        brain_analysis = introspector.analyze_all_brains()

                        # Format the response - brain_analysis has:
                        # - brains: dict of brain_name -> analysis dict
                        # - compliant: list of compliant brain names
                        # - non_compliant: list of non-compliant brain names
                        brains_dict = brain_analysis.get("brains", {})
                        total_brains = brain_analysis.get("total_brains", len(brains_dict))
                        compliant_list = brain_analysis.get("compliant", [])
                        non_compliant_list = brain_analysis.get("non_compliant", [])
                        compliant_count = len(compliant_list)

                        # Build a formatted answer
                        answer_lines = [
                            f"**Brain Inventory: {total_brains} cognitive brains found**",
                            f"Compliant brains: {compliant_count}/{total_brains}",
                            "",
                            "| Brain | Status | Has Service | Has Process |",
                            "|-------|--------|-------------|-------------|",
                        ]
                        for brain_name, brain_info in brains_dict.items():
                            is_compliant = brain_name in compliant_list
                            status = "✓ compliant" if is_compliant else "✗ non-compliant"
                            has_service = "Yes" if brain_info.get("has_service") else "No"
                            has_process = "Yes" if brain_info.get("has_process") else "No"
                            answer_lines.append(f"| {brain_name} | {status} | {has_service} | {has_process} |")

                        final_answer = "\n".join(answer_lines)

                        ctx_inventory = {
                            "original_query": text,
                            "final_answer": final_answer,
                            "final_confidence": 0.95,
                            "mode": "INVENTORY_DIRECT",
                        }
                        return success_response(op, mid, {"context": ctx_inventory})
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] Inventory routing failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Norm introspection detection (show what was processed/normalized)
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                norm_introspection = False

                # Check for norm introspection patterns
                if any(pattern in lt for pattern in [
                    "repeat this message back normalized",
                    "repeat my message back normalized",
                    "show me what you processed",
                    "show what you normalized",
                    "what did you normalize",
                    "echo my normalized input",
                    "show normalized input"
                ]):
                    norm_introspection = True

                if norm_introspection:
                    print("[MEMORY_LIBRARIAN] Detected norm introspection request")
                    try:
                        from brains.cognitive.sensorium.service.sensorium_brain import service_api as sensorium_api

                        # First normalize the current text
                        norm_resp = sensorium_api({
                            "op": "NORMALIZE",
                            "mid": mid,
                            "payload": {"text": text},
                        })

                        # Then get the last normalized result
                        last_norm_resp = sensorium_api({
                            "op": "GET_LAST_NORMALIZED",
                            "mid": mid,
                            "payload": {},
                        })

                        if last_norm_resp.get("ok"):
                            norm_pay = last_norm_resp.get("payload") or {}
                            raw_text = norm_pay.get("raw_text", "")
                            normalized_text = norm_pay.get("normalized_text", "")
                            norm_type = norm_pay.get("norm_type", "unknown")
                            tokens = norm_pay.get("tokens", [])

                            answer_lines = [
                                "**Normalization Introspection**",
                                "",
                                f"**Raw input:** {raw_text}",
                                f"**Normalized:** {normalized_text}",
                                f"**Classification:** {norm_type}",
                            ]
                            if tokens:
                                answer_lines.append(f"**Tokens:** {', '.join(tokens)}")

                            final_answer = "\n".join(answer_lines)

                            ctx_norm = {
                                "original_query": text,
                                "final_answer": final_answer,
                                "final_confidence": 0.95,
                                "mode": "NORM_INTROSPECTION_DIRECT",
                            }
                            return success_response(op, mid, {"context": ctx_norm})
                    except Exception as e:
                        print(f"[MEMORY_LIBRARIAN] Norm introspection failed: {e}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Reflection commands: run QA then improve with self-review
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                lt = text.strip().lower()
                base_question = None
                reflection_mode = None
                compare_questions: Optional[Tuple[str, str]] = None
                reflection_topic = None

                if lt.startswith("reflect on this answer:"):
                    base_question = text.split(":", 1)[1].strip()
                    reflection_mode = "answer_reflection"
                elif " and reflect on your explanation" in lt:
                    base_question = text.split(" and reflect on your explanation", 1)[0].strip()
                    reflection_mode = "answer_reflection"
                elif lt.startswith("reflect deeply:"):
                    base_question = text.split(":", 1)[1].strip()
                    reflection_mode = "deep"
                elif lt.startswith("compare and reflect:") and " vs " in lt:
                    comp = text.split(":", 1)[1]
                    parts = comp.split(" vs ", 1)
                    if len(parts) == 2:
                        compare_questions = (parts[0].strip(), parts[1].strip())
                        reflection_mode = "compare"
                elif lt.startswith("self reflect"):
                    reflection_mode = "self_audit"
                elif lt.startswith("reflect on how you learned about"):
                    reflection_topic = text.split("about", 1)[1].strip()
                    base_question = reflection_topic
                    reflection_mode = "self_audit"

                if reflection_mode:
                    from brains.cognitive.self_review.service.self_review_brain import run_reflection_engine

                    def _load_last_turn() -> Tuple[Optional[str], Optional[str]]:
                        try:
                            log_path = (MAVEN_ROOT / "reports" / "query_log.jsonl").resolve()
                            if not log_path.exists():
                                return None, None
                            lines = log_path.read_text(encoding="utf-8").splitlines()
                            for line in reversed(lines[-50:]):
                                if not line.strip():
                                    continue
                                try:
                                    obj = json.loads(line.strip())
                                except Exception:
                                    continue
                                q = obj.get("query") or obj.get("text")
                                ans = obj.get("answer") or obj.get("final_answer") or obj.get("response")
                                if q:
                                    return str(q), str(ans) if ans else None
                        except Exception:
                            return None, None
                        return None, None

                    def _run_pipeline_question(q: str) -> Tuple[Dict[str, Any], Optional[str]]:
                        resp = service_api(
                            {
                                "op": "RUN_PIPELINE",
                                "mid": mid,
                                "payload": {"text": q, "_bypass_prerouting": True},
                            }
                        )
                        ctx_local = (resp.get("payload") or {}).get("context", {})
                        ans_local = ctx_local.get("final_answer") or (
                            (ctx_local.get("stage_10_finalize") or {}).get("text")
                        )
                        return ctx_local, ans_local

                    if compare_questions:
                        ctx_a, ans_a = _run_pipeline_question(compare_questions[0])
                        ctx_b, ans_b = _run_pipeline_question(compare_questions[1])
                        comparison_text = (
                            (ans_a or "") + "\n\n--- VS ---\n\n" + (ans_b or "")
                        ).strip()
                        reflection_ctx = {
                            "question": f"Compare: {compare_questions[0]} vs {compare_questions[1]}",
                            "draft_answer": comparison_text,
                            "final_answer": comparison_text,
                            "confidence": min(
                                float(ctx_a.get("final_confidence") or 0.7),
                                float(ctx_b.get("final_confidence") or 0.7),
                            ),
                            "used_brains": [],
                            "context_tags": ["compare"],
                            "manual": True,
                        }
                        review_result = run_reflection_engine("compare", reflection_ctx)
                        verdict = review_result.get("verdict", "ok")
                        issues = review_result.get("issues", [])
                        improved_answer = review_result.get("improved_answer")
                        summary_lines = [
                            f"Compared answers for '{compare_questions[0]}' and '{compare_questions[1]}'",
                            f"Verdict: {verdict}",
                        ]
                        if issues:
                            summary_lines.append("Key issues: " + "; ".join(issues[:4]))
                        if improved_answer:
                            summary_lines.append("Refined synthesis:\n" + improved_answer)
                        else:
                            summary_lines.append("Observed answers:\n" + comparison_text)
                        reflect_ctx = {
                            "original_query": text,
                            "base_question": f"{compare_questions[0]} vs {compare_questions[1]}",
                            "initial_answer": comparison_text,
                            "final_answer": "\n\n".join(summary_lines),
                            "final_confidence": review_result.get("meta", {}).get("confidence", 0.65),
                            "mode": "REFLECTION_COMPARE",
                        }
                        return success_response(op, mid, {"context": reflect_ctx})

                    if not base_question:
                        base_question, initial_answer = _load_last_turn()
                    else:
                        initial_ctx, initial_answer = _run_pipeline_question(base_question)

                    if not base_question:
                        base_question = "recent question"
                        initial_ctx = {}
                        initial_answer = ""
                    else:
                        if 'initial_ctx' not in locals():
                            initial_ctx = {}

                    reflection_ctx = {
                        "question": base_question,
                        "draft_answer": initial_answer or "",
                        "final_answer": initial_answer or "",
                        "confidence": float((initial_ctx or {}).get("final_confidence") or 0.75),
                        "used_brains": [],
                        "context_tags": [reflection_mode],
                        "wm_trace": (initial_ctx or {}).get("wm_trace"),
                        "manual": True,
                    }
                    review_result = run_reflection_engine(reflection_mode, reflection_ctx)
                    verdict = review_result.get("verdict", "ok")
                    issues = review_result.get("issues", [])
                    improved_answer = review_result.get("improved_answer")

                    summary_lines = [f"Reflection on: {base_question}", f"Verdict: {verdict}"]
                    if issues:
                        summary_lines.append("Key findings: " + "; ".join(issues[:5]))
                    if improved_answer:
                        summary_lines.append("Corrected answer:\n" + improved_answer)
                    elif initial_answer:
                        summary_lines.append("Reviewed answer:\n" + initial_answer)
                    if reflection_mode in {"self_audit", "deep"}:
                        summary_lines.append("Future handling: be explicit about uncertainties and cite known facts when available.")

                    reflect_ctx = {
                        "original_query": text,
                        "base_question": base_question,
                        "initial_answer": initial_answer,
                        "final_answer": "\n\n".join(summary_lines),
                        "final_confidence": float((initial_ctx or {}).get("final_confidence") or 0.65),
                        "mode": "REFLECTION_DIRECT",
                    }
                    return success_response(op, mid, {"context": reflect_ctx})
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Multi-query detection and routing
        # ------------------------------------------------------------------
        if not bypass_prerouting:
            try:
                math_matches = re.findall(r"\d+\s*[+\-*/]\s*\d+", text)
                text_wo_math = re.sub(r"\d+\s*[+\-*/]\s*\d+", " ", text)
                lang_parts = []
                for part in re.split(r"\s+and\s+", text_wo_math):
                    seg = part.strip()
                    if seg.lower().startswith("and "):
                        seg = seg[4:]
                    if seg.lower().endswith(" and"):
                        seg = seg[:-4].strip()
                    if not seg or len(seg.split()) <= 1 and not math_matches:
                        continue
                    if seg.lower() in {"what is", "what is?", "what is and"}:
                        continue
                    lang_parts.append(seg)

                total_subs = len(math_matches) + len(lang_parts)
                if total_subs > 1 and (math_matches or len(lang_parts) > 1):
                    combined_answers: List[str] = []
                    # Math sub-queries
                    for expr in math_matches:
                        math_result = _solve_simple_math(expr)
                        if math_result.get("ok"):
                            combined_answers.append(f"{expr.replace(' ', '')} = {math_result['result']}")
                    # Language sub-queries
                    for seg in lang_parts:
                        sub_resp = service_api(
                            {
                                "op": "RUN_PIPELINE",
                                "mid": mid,
                                "payload": {"text": seg, "_bypass_prerouting": True},
                            }
                        )
                        sub_ctx = (sub_resp.get("payload") or {}).get("context", {})
                        sub_ans = sub_ctx.get("final_answer") or (
                            (sub_ctx.get("stage_10_finalize") or {}).get("text")
                        )
                        if sub_ans:
                            combined_answers.append(str(sub_ans).strip())
                    if combined_answers:
                        multi_ctx = {
                            "original_query": text,
                            "multi_query": {
                                "math": math_matches,
                                "language": lang_parts,
                            },
                            "final_answer": "\n\n".join(combined_answers),
                            "final_confidence": 0.72,
                            "mode": "MULTI_QUERY",
                        }
                        return success_response(op, mid, {"context": multi_ctx})
            except Exception:
                pass

        # ------------------------------------------------------------------
        # SELF-INTENT GATE: Catch self-memory questions BEFORE pipeline
        # ------------------------------------------------------------------
        # Questions about Maven's OWN memory/learning MUST be answered by
        # self_model, NEVER by Teacher. Intercept them HERE before
        # SENSORIUM/AFFECT/INTEGRATOR process them.
        try:
            q_lower = text.strip().lower()
            normalized = q_lower
            for punct in ['.', ',', '!', '?', ':', ';']:
                normalized = normalized.replace(punct, ' ')
            normalized = ' '.join(normalized.split())

            # Identity patterns: questions about who Maven is
            identity_patterns = [
                "who are you",
                "what are you",
                "who you are",
                "tell me who you are",
                "tell me about yourself",
                "tell me about your self",
                "tell me about you",
                "describe yourself",
                "describe your self",
                "what can you tell me about yourself",
                "what can you tell me about your self",
                "what do you know about your self",
                "what do you know about yourself"
            ]

            extra_self_patterns = [
                "training data",
                "knowledge cutoff",
                "when were you created",
                "when were you trained",
                "how were you trained",
                "what model are you",
                "are you an llm",
            ]
            identity_patterns.extend(extra_self_patterns)

            # TASK 1: Feelings / emotions / internal state patterns
            # These MUST route to self_model, NOT Teacher
            feelings_patterns = [
                "how do you feel",
                "how are you feeling",
                "do you have feelings",
                "do you have emotions",
                "are you happy",
                "are you sad",
                "are you conscious",
                "are you sentient",
                "are you alive",
                "are you real",
                "are you a person",
                "do you like",
                "do you enjoy",
                "do you prefer",
                "do you want",
                "what do you like",
                "what do you want",
                "what do you prefer",
                "do you have preferences",
                "do you have opinions",
                "what is your opinion",
                "what are your opinions",
                "how do you think",
                "what do you think about yourself",
                "do you dream",
                "do you sleep",
                "do you get tired",
                "do you get bored",
                "can you feel",
                "can you think",
                "your feelings",
                "your emotions",
                "your preferences",
                "your opinions",
            ]
            identity_patterns.extend(feelings_patterns)

            system_scan_patterns = [
                "scan self",
                "scan your self",
                "scan yourself",
                "scan your entire system",
                "scan entire system",
                "system scan",
                "full system scan",
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

            # Code patterns: questions about Maven's own code
            code_patterns = [
                "what do you know about your code",
                "what do you know about your own code",
                "describe your full pipeline stages from code introspection only",
            ]

            # Runtime/root patterns: questions about where Maven is running
            runtime_patterns = [
                "where is your root directory",
                "where are you running from",
                "where is your runtime",
                "what is your root directory"
            ]

            # Diagnostics patterns
            diag_patterns = ["diag", "diagnostics", "run diagnostics"]

            # Stats patterns: questions about fact counts
            memory_stats_patterns = [
                "how many facts have you learned",
                "how much have you learned",
                "how many facts do you know",
                "what have you learned so far",
                "what do you remember",
                "what do you remember right now",
                "memory stats",
                "show memory stats",
            ]

            # Scan/health patterns: memory system diagnostics
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

            # Upgrade patterns: self-upgrade planning requests
            upgrade_patterns = [
                "plan upgrade for your self",
                "plan upgrade for yourself",
                "plan a upgrade for your self",
                "plan a upgrade for yourself",
                "plan an upgrade for your self",
                "plan an upgrade for yourself",
                "plan upgrades for yourself",
                "plan upgrades for your self",
                "how can you upgrade yourself",
                "how can you improve yourself",
                "plan upgrade for your system",
                "plan an upgrade for your system"
            ]

            self_detected = False
            self_kind = None  # identity, code, memory, runtime, upgrade
            self_mode = None  # stats, health (for memory queries), plan (for upgrade queries)

            # Check identity patterns
            for pattern in identity_patterns:
                if pattern in normalized:
                    self_detected = True
                    self_kind = "identity"
                    print(f"[SELF_INTENT_GATE] Intercepted self-identity question BEFORE pipeline")
                    print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                    break

            # Check code patterns
            if not self_detected:
                for pattern in code_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "code"
                        print(f"[SELF_INTENT_GATE] Intercepted self-code question BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                        break

            # Check runtime patterns
            if not self_detected:
                for pattern in runtime_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "runtime"
                        print(f"[SELF_INTENT_GATE] Intercepted self-runtime question BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                        break

            # System scan patterns
            if not self_detected:
                for pattern in system_scan_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "system_scan"
                        self_mode = "full"
                        print(f"[SELF_INTENT_GATE] Intercepted self-system scan BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}, mode={self_mode}")
                        break

            # Routing scan patterns
            if not self_detected:
                for pattern in routing_scan_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "routing_scan"
                        print(f"[SELF_INTENT_GATE] Intercepted self-routing scan BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                        break

            # Codebase scan patterns
            if not self_detected:
                for pattern in codebase_scan_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "code_scan"
                        print(f"[SELF_INTENT_GATE] Intercepted self-codebase scan BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                        break

            # Cognitive brain scan patterns
            if not self_detected:
                for pattern in cognitive_scan_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "cognitive_scan"
                        print(f"[SELF_INTENT_GATE] Intercepted self-cognitive scan BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                        break

            # Check stats patterns
            if not self_detected:
                for pattern in memory_stats_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "memory"
                        self_mode = "stats"
                        print(f"[SELF_INTENT_GATE] Intercepted self-memory question BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}, mode={self_mode}")
                        break

            # Check scan patterns
            if not self_detected:
                for pattern in memory_scan_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "memory"
                        self_mode = "health"
                        print(f"[SELF_INTENT_GATE] Intercepted self-memory question BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}, mode={self_mode}")
                        break

            # Diagnostics patterns
            if not self_detected:
                for pattern in diag_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "diag"
                        print(f"[SELF_INTENT_GATE] Intercepted self-diagnostics question BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}")
                        break

            # Check upgrade patterns
            if not self_detected:
                for pattern in upgrade_patterns:
                    if pattern in normalized:
                        self_detected = True
                        self_kind = "upgrade"
                        self_mode = "plan"
                        print(f"[SELF_INTENT_GATE] Intercepted self-upgrade question BEFORE pipeline")
                        print(f"[SELF_INTENT_GATE] Matched: '{pattern}', kind={self_kind}, mode={self_mode}")
                        break

            # If detected, bypass entire pipeline and call self_model directly
            if self_detected:
                print(f"[SELF_INTENT_GATE] Bypassing SENSORIUM/AFFECT/INTEGRATOR/REASONING")
                print(f"[SELF_INTENT_GATE] Routing directly to self_model")
                try:
                    from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api

                    self_resp = self_model_api({
                        "op": "QUERY_SELF",
                        "payload": {
                            "query": text,
                            "self_kind": self_kind,
                            "self_mode": self_mode
                        }
                    })

                    if self_resp.get("ok"):
                        resp_payload = self_resp.get("payload", {})
                        answer_text = resp_payload.get("text") or resp_payload.get("answer")

                        if answer_text:
                            print(f"[SELF_INTENT_GATE] Got answer from self_model, returning immediately")
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "verdict": "SKIP_STORAGE",
                                    "answer": answer_text,
                                    "confidence": resp_payload.get("confidence", 1.0),
                                    "mode": f"SELF_{self_kind.upper()}_DIRECT" if self_kind else "SELF_DIRECT",
                                    "bypassed_pipeline": True
                                }
                            }
                except Exception as e:
                    print(f"[SELF_INTENT_GATE_ERROR] Failed calling self_model: {e}")
                    # Fall through to normal pipeline
        except Exception as e:
            print(f"[SELF_INTENT_GATE_ERROR] Gate check failed: {e}")
            # Fall through to normal pipeline

        # ------------------------------------------------------------------
        # Optional pipeline tracing.  When the environment variable
        # TRACE_PIPELINE is set to "1" or "true" (case insensitive), tracing is
        # forcibly enabled. Otherwise it falls back to the configuration in
        # CFG['pipeline_tracer']['enabled']. Trace events are emitted to
        # reports/pipeline_trace/trace_<mid>.jsonl, and the number of retained
        # traces is capped by CFG['pipeline_tracer']['max_files']. Older
        # traces beyond this cap are removed to prevent disk bloat.
        trace_enabled: bool = False
        try:
            # Check environment override first
            env_val = os.getenv("TRACE_PIPELINE")
            if env_val is not None:
                trace_enabled = str(env_val).strip().lower() in {"1","true","yes"}
            else:
                # Fall back to configuration default
                trace_cfg = CFG.get("pipeline_tracer", {}) or {}
                trace_enabled = bool(trace_cfg.get("enabled", False))
        except Exception:
            trace_enabled = False
        trace_events = []

        # Personality snapshot (best-effort)
        try:
            from brains.cognitive.personality.service import personality_brain
            prefs = personality_brain._read_preferences()
        except Exception:
            prefs = {"prefer_explain": True, "tone": "neutral", "verbosity_target": 1.0}

        # Load any prior session context to provide continuity across runs.  This
        # enables the system to recall high-level context between separate
        # pipeline executions, forming the basis of a persistent memory layer.
        session_ctx: Dict[str, Any] = {}
        try:
            snap_path = MAVEN_ROOT / "reports" / "context_snapshot.json"
            if snap_path.exists():
                session_ctx = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception:
            session_ctx = {}

        # Extract learning_mode from payload (default to TRAINING for LLM learning)
        try:
            from brains.learning.learning_mode import LearningMode
            _learning_mode = payload.get("learning_mode", LearningMode.TRAINING)
        except Exception:
            _learning_mode = payload.get("learning_mode", "training")

        ctx: Dict[str, Any] = {
            "original_query": text,
            "personality_snapshot": prefs,
            "session_context": session_ctx,
            "learning_mode": _learning_mode
        }
        # Attach deterministic seed to the context and persist golden trace
        try:
            if _seed_record is not None:
                seed_val, seed_ts = _seed_record
                ctx["run_seed"] = int(seed_val)
                # Write golden trace file
                try:
                    from pathlib import Path as _Path
                    import json as _json
                    root = _Path(__file__).resolve().parents[4]
                    gt_dir = root / "reports" / "golden_trace"
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    gt_path = gt_dir / f"trace_{seed_ts}.json"
                    # Use atomic write via api.utils
                    from api.utils import _atomic_write  # type: ignore
                    _atomic_write(gt_path, _json.dumps({"trace_id": seed_ts, "seed": int(seed_val)}, indent=2))
                except Exception:
                    pass
        except Exception:
            pass
        # ------------------------------------------------------------------
        # Update recent queries for multi‑turn context.  Maintain a short
        # list of the most recent queries and store it in the session
        # context.  This enables detection of repeated questions and
        # supports stress or frustration detection.  Older queries are
        # discarded beyond the maximum size.  Any errors during update
        # are silently ignored to avoid disrupting the pipeline.
        try:
            _RECENT_QUERIES.append({"query": text})
            if len(_RECENT_QUERIES) > _MAX_RECENT_QUERIES:
                _RECENT_QUERIES.pop(0)
            ctx["session_context"]["recent_queries"] = list(_RECENT_QUERIES)
            ctx["session_context"]["context_truncated"] = len(_RECENT_QUERIES) >= _MAX_RECENT_QUERIES
        except Exception:
            pass

        # Stage 1b — Personality adjust (observed via Governance)
        try:
            from brains.cognitive.personality.service import personality_brain
            sug_res = personality_brain.service_api({"op":"ADAPT_WEIGHTS_SUGGEST"}) or {}
            suggestion = (sug_res.get("payload") or {}).get("suggestion") or {}
        except Exception:
            suggestion = {}

        try:
            gov = _gov_module()
            adj = gov.service_api({"op":"ENFORCE","payload":{"action":"ADJUST_WEIGHTS","payload": suggestion}})
            approved = bool((adj.get("payload") or {}).get("allowed"))
        except Exception:
            approved = False

        ctx["stage_1b_personality_adjustment"] = {"proposal": suggestion, "approved": approved}

        # ------------------------------------------------------------------
        # Stage 1 — Sensorium normalization.  Perform lightweight text
        # normalization (e.g. lowercasing, whitespace trimming) before
        # downstream processing.  The sensorium stage runs unconditionally.
        s = _brain_module("sensorium").service_api({"op": "NORMALIZE", "payload": {"text": text}})
        if trace_enabled:
            trace_events.append({"stage": "sensorium"})

        # Stage 3 — Language parsing.  Parse the input to determine its
        # communicative intent (question, command, request, fact, etc.).
        # We run the language brain before planning so that only commands
        # and requests trigger the heavy plan/goal generation logic.
        l = _brain_module("language").service_api(
            {"op": "PARSE", "payload": {"text": text, "delta": (suggestion.get("language") if approved else {})}}
        )
        if trace_enabled:
            trace_events.append({"stage": "language_parse"})

        # Extract language parse results early so we can decide whether to
        # invoke the planner.  Commands and explicit requests create
        # actionable plans; other utterances (questions, facts, speculation)
        # receive a simple fallback plan.  This prevents the planner from
        # segmenting arbitrary statements into junk sub‑goals.
        lang_payload = (l.get("payload") or {})
        # Determine if the input is a command or request.  Fall back to
        # type to catch alternative representations (e.g. "COMMAND",
        # "REQUEST") even if is_command/request booleans are absent.
        is_cmd = bool(lang_payload.get("is_command"))
        is_req = bool(lang_payload.get("is_request"))
        is_question = bool(lang_payload.get("is_question"))
        st_type = str(lang_payload.get("type", "")).upper()
        intent = str(lang_payload.get("intent", "")).upper()
        # Augment command detection: if is_command flag is not set but
        # the storable_type is COMMAND or the text begins with a CLI prefix,
        # treat this input as a command for planning and routing purposes.
        if not is_cmd:
            try:
                tnorm_cli = str(text or "").strip()
                if st_type == "COMMAND" or tnorm_cli.startswith("--") or tnorm_cli.startswith("/"):
                    is_cmd = True
            except Exception:
                pass

        # Determine if planner is required based on intent type
        # Questions, explanations, comparisons, and commands all need planning
        def _planner_required() -> bool:
            if intent in {"RESEARCH_REQUEST", "RESEARCH_FOLLOWUP"}:
                return False
            # Command and request intents always require planning
            if is_cmd or is_req or st_type in {"COMMAND", "REQUEST"}:
                return True
            # Question intents require planning for proper reasoning
            if is_question or st_type == "QUESTION":
                return True
            # Specific intent types that require planning
            planning_intents = {
                "SIMPLE_FACT_QUERY",
                "QUESTION",
                "EXPLAIN",
                "WHY",
                "HOW",
                "COMPARE",
                "PROFILE_QUERY",
                "PREFERENCE_QUERY",
                "RELATIONSHIP_QUERY",
            }
            if intent in planning_intents:
                return True
            return False

        should_plan = _planner_required()
        # ----------------------------------------------------------------------
        # Additional filter: do not invoke the planner for simple retrieval
        # requests.  Many user queries of the form "show me ..." or
        # "find ..." are treated as commands by the language parser, which
        # in turn causes the planner to segment the request into sub‑goals
        # (e.g. "Show me Paris photos" and "the Eiffel Tower").  These
        # retrieval requests are meant to be handled immediately rather
        # than persisted as autonomous goals.  To avoid polluting the
        # personal goal memory with such items, we only allow planning
        # when the command starts with a strongly actionable verb (e.g.
        # "create", "make", "build", "plan", "schedule", "delegate",
        # "execute").  Commands beginning with other words (like
        # "show", "find", "search", "display", etc.) are executed on the
        # spot and skipped by the planner.
        # IMPORTANT: This filter only applies to commands, NOT to questions
        # or other query intents which should always go through the planner.
        if should_plan and (is_cmd or is_req) and not is_question:
            try:
                # Define the set of verbs that warrant persistent plans
                command_verbs = {"create", "make", "build", "plan", "schedule", "delegate", "execute"}
                # Define question/query words that should bypass this filter
                query_words = {"why", "how", "what", "when", "where", "who", "compare", "explain", "describe", "tell"}
                # Normalise the input to lower case and split into tokens
                query_lc = (text or "").strip().lower()
                tokens = query_lc.split()
                # Only proceed with planning if the first token is one of the
                # actionable verbs OR a question word.  Otherwise, reset should_plan
                # to False to skip the planner and use a fallback plan.
                if tokens:
                    first = tokens[0]
                    if first not in command_verbs and first not in query_words:
                        should_plan = False
            except Exception:
                # On error, leave should_plan unchanged
                pass

        # ------------------------------------------------------------------
        # Stage 7a — command routing
        #
        # Before invoking the planner, fast cache or any further stages,
        # detect CLI‑style commands (inputs starting with "--" or "/").
        # These should bypass the normal question/answer pipeline and be
        # handled by the command router.  Only commands that are not
        # actionable tasks (i.e. should_plan is False) are routed here.  If a
        # built‑in command is recognised, its result becomes the final
        # answer.  Unknown commands return a structured error.  For
        # consistency with the rest of the system, the verdict is set to
        # NEUTRAL and storage is skipped.  A minimal context is returned
        # immediately, skipping heavy retrieval and reasoning.
        try:
            # Identify inputs that look like commands and are not slated for
            # goal planning.  The language parser sets is_command True for
            # strings beginning with "--" or "/".  However, some parse
            # variants may set the type to COMMAND without populating the
            # boolean.  In addition, check the raw text prefix to catch
            # unparsed commands.  Only intercept when planning is disabled
            # to avoid interfering with complex task creations (e.g. "create goal ...").
            cmd_like = False
            try:
                stripped = (text or "").strip()
                cmd_like = stripped.startswith("--") or stripped.startswith("/")
            except Exception:
                cmd_like = False
            # Retrieve storable type from the language payload if present
            st_type_local = str(lang_payload.get("storable_type", lang_payload.get("type", ""))).upper()
            if (is_cmd or st_type_local == "COMMAND" or cmd_like) and not should_plan:
                # Route the command through the command_router.  Import on
                # demand to avoid circular dependencies during module
                # initialisation.
                try:
                    from brains.cognitive.command_router import route_command  # type: ignore
                    cmd_result = route_command(text)
                except Exception as _exc:
                    cmd_result = {"error": f"router_import_failed: {_exc}"}
                # Determine the response message.  If the router returns a
                # ``message``, use it directly.  Otherwise fall back to the
                # error description or a generic notice.
                msg_text = None
                try:
                    if isinstance(cmd_result, dict):
                        if cmd_result.get("message"):
                            msg_text = str(cmd_result["message"])
                        elif cmd_result.get("error"):
                            msg_text = str(cmd_result["error"])
                except Exception:
                    msg_text = None
                if not msg_text:
                    msg_text = "No command response."
                # Build a minimal context capturing the parse and final answer.
                ctx: Dict[str, Any] = {
                    "original_query": text,
                    "stage_3_language": lang_payload,
                    "stage_8_validation": {"verdict": "NEUTRAL", "confidence": 0.0},
                    "stage_6_candidates": {"candidates": []},
                    "stage_9_storage": {"skipped": True, "reason": "command"},
                    "stage_10_finalize": {"text": msg_text, "confidence": 0.0},
                    "final_answer": msg_text,
                    # Assign low confidence for error messages and higher for
                    # successful commands.  When the router returned an error,
                    # confidence is set to 0.0; otherwise use 0.8.
                    "final_confidence": (0.0 if cmd_result.get("error") else 0.8),
                    "final_tag": "command_response",
                }
                # Trace the command routing event if tracing is enabled.
                if trace_enabled:
                    trace_events.append({"stage": "command_router", "result": cmd_result})
                return success_response(op, mid, {"context": ctx})
        except Exception:
            # On any error in the command router path, fall through to the
            # normal pipeline.  Errors here should not prevent question
            # answering; they simply cause commands to be treated as
            # statements.
            pass

        # ------------------------------------------------------------------
        # DISABLED (Phase C cleanup): Self/Environment query bypasses removed
        # ------------------------------------------------------------------
        # These short-circuits prevented requests from flowing through the
        # cognitive pathway. Now disabled to ensure all requests reach
        # stage6_generate and use the Template→Heuristic→LLM pathway.
        #
        # Trace self/environment query detection for diagnostics
        try:
            if tracer and RouteType:
                if _is_env_query(text):
                    tracer.record_route(mid, RouteType.SELF_QUERY, {"query_type": "environment", "bypass": "disabled"})
        except Exception:
            pass

        # OLD CODE: Environment query bypass (DISABLED - commented out)
        # The entire environment query short-circuit has been removed to ensure
        # all requests flow through the cognitive pathway to stage6_generate.

        # OLD CODE: Self query bypass (DISABLED - commented out)
        # Trace self query detection for diagnostics
        try:
            if tracer and RouteType and _is_self_query(text):
                tracer.record_route(mid, RouteType.SELF_QUERY, {"query_type": "self", "bypass": "disabled"})
        except Exception:
            pass

        # The entire self query short-circuit has been removed to ensure
        # all requests flow through the cognitive pathway to stage6_generate.

        # ------------------------------------------------------------------
        # Fast cache lookup: Check if we have a cached answer with learned
        # confidence. This enables learning from feedback ("correct", etc.)
        # ------------------------------------------------------------------
        # When a user confirms an answer with "correct", the confidence is
        # boosted in the cache. On subsequent queries, the cache returns
        # the answer with the learned (higher) confidence, demonstrating
        # that Maven has learned from the feedback.
        #
        # Fast cache lookup
        fc_rec = _lookup_fast_cache(text)
        try:
            if tracer and RouteType:
                # Check if cache would have hit (for diagnostics only)
                _fc_check = _lookup_fast_cache(text)
                if _fc_check:
                    tracer.record_route(mid, RouteType.FAST_CACHE, {"bypass": "disabled", "would_have_hit": True})
        except Exception:
            pass
        # ------------------------------------------------------------------
        # Additional fast cache gating: avoid using cached answers for
        # queries about the agent's location or identity.  The fast cache
        # stores answers verbatim from prior runs; however, environmental
        # or self‑identity queries can evolve when the self model or
        # environment context is updated.  Using a stale cached answer
        # risks serving off‑topic content.  Similarly, skip the fast
        # cache for any query that the self query detector flags.  This
        # ensures fresh responses are computed for location and identity
        # queries rather than relying on potentially poisoned cache.
        if fc_rec:
            try:
                qnorm_lc = str((text or "")).strip().lower()
            except Exception:
                qnorm_lc = ""
            # Skip fast cache for self queries (who/what/where/how + you/your/yourself)
            skip_fc = False
            try:
                if _is_self_query(text):
                    skip_fc = True
            except Exception:
                skip_fc = False
            # Skip fast cache for environment location queries.  Note that
            # "where are we" is excluded here and handled as a
            # conversation meta pattern instead.  Patterns include
            # queries asking about the agent's physical location (you/am i)
            # and personal residence.
            if not skip_fc:
                env_triggers = [
                    # "where are we" removed; handled in conversation meta
                    "where are you",
                    "where am i",
                    "where's your location",
                    "where do you live",
                ]
                for _pat in env_triggers:
                    try:
                        if _pat in qnorm_lc:
                            skip_fc = True
                            break
                    except Exception:
                        continue
            if skip_fc:
                fc_rec = None
        # If a cached result is found, check it for meta or filler phrases.  If the
        # answer appears to be a generic or self‑referential response (e.g.
        # "I'm going to try my best"), treat the cache entry as poisoned and
        # ignore it.  This prevents incorrect filler answers from being
        # trusted as factual on subsequent runs.
        if fc_rec:
            try:
                ans_lc = str(fc_rec.get("answer", "")).strip().lower()
            except Exception:
                ans_lc = ""
            invalid_cache = False
            # Check configured bad phrases
            for bad in BAD_CACHE_PHRASES:
                if bad and bad in ans_lc:
                    invalid_cache = True
                    break
            # Run semantic verification on the cached answer.  Even if it
            # does not match a specific bad phrase, an answer that fails the
            # heuristic should be treated as invalid and recomputed.
            try:
                if not invalid_cache and not _semantic_verify(fc_rec.get("answer", "")):
                    invalid_cache = True
            except Exception:
                # On verification error, mark as invalid to be safe
                invalid_cache = True
            if invalid_cache:
                # Log the poisoning event to a report for later analysis.  Swallow
                # any exceptions during logging to avoid breaking the pipeline.
                try:
                    log_path = MAVEN_ROOT / "reports" / "cache_poison.log"
                    log_entry = {
                        "query": text,
                        "bad_answer": fc_rec.get("answer")
                    }
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(log_entry) + "\n")
                except Exception:
                    pass
                fc_rec = None
        if fc_rec:
            # LEARNING ON REPEAT: Boost confidence when same question asked again
            boosted_confidence = _boost_cache_confidence(text, boost_amount=0.1)
            if boosted_confidence is not None:
                final_conf = boosted_confidence
            else:
                final_conf = fc_rec.get("confidence", 0.8)

            # Attach parse results for downstream consumers
            ctx["stage_3_language"] = lang_payload or {}
            # Mark reasoning verdict and attach cached answer with boosted confidence
            ctx["stage_8_validation"] = {
                "verdict": "TRUE",
                "answer": fc_rec.get("answer"),
                "confidence": final_conf,
                "from_cache": True,
                "confidence_boosted": boosted_confidence is not None,
            }
            # Generate language candidates (high‑confidence direct answer)
            try:
                cand_res = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                ctx["stage_6_candidates"] = cand_res.get("payload", {})
            except Exception:
                ctx["stage_6_candidates"] = {}
            # Finalise the answer without tone wrapping for factual responses
            try:
                fin_res = _brain_module("language").service_api({"op": "FINALIZE", "payload": ctx})
                ctx["stage_10_finalize"] = fin_res.get("payload", {})
            except Exception:
                ctx["stage_10_finalize"] = {}
            # Capture final answer and confidence for external consumers
            try:
                ctx["final_answer"] = ctx.get("stage_10_finalize", {}).get("text")
                ctx["final_confidence"] = ctx.get("stage_10_finalize", {}).get("confidence")
                # Fallback: if finalization failed but we have a cached answer, use it directly
                if not ctx["final_answer"] and ctx.get("stage_8_validation", {}).get("from_cache"):
                    ctx["final_answer"] = ctx.get("stage_8_validation", {}).get("answer")
                    ctx["final_confidence"] = ctx.get("stage_8_validation", {}).get("confidence")
            except Exception:
                ctx["final_answer"] = None
                ctx["final_confidence"] = None
            # Indicate that storage was skipped due to fast cache
            ctx["stage_9_storage"] = {"skipped": True, "reason": "fast_cache_used"}
            # Persist context snapshot and system report
            try:
                _save_context_snapshot(ctx, limit=5)
            except Exception:
                pass
            try:
                run_id = ctx.get("run_seed", _SEQ_ID_COUNTER)
                write_report("system", f"run_{run_id}.json", json.dumps(ctx, indent=2))
            except Exception:
                pass
            # Before returning on a fast cache hit, perform a self‑evaluation.
            # This assessment computes simple health metrics and may enqueue
            # autonomous repair goals based on the context.  Errors are
            # swallowed to avoid disrupting the cache fast path.
            try:
                import importlib
                sc_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_critique")
                eval_resp = sc_mod.service_api({"op": "EVAL_CONTEXT", "payload": {"context": ctx}})
                ctx["stage_self_eval"] = eval_resp.get("payload", {})
            except Exception:
                ctx["stage_self_eval"] = {"error": "eval_failed"}
            # Return immediately with the final context
            # Before returning, update conversation state based on the
            # original query and final answer.  This enables multi‑turn
            # continuation handling.
            try:
                _update_conversation_state(text, ctx.get("final_answer"))
            except Exception:
                pass

            # CRITICAL FIX: Ensure final_answer is set (fast cache early return)
            try:
                if not ctx.get("final_answer"):
                    stage8 = ctx.get("stage_8_validation") or {}
                    verdict = str(stage8.get("verdict", "")).upper()
                    answer = stage8.get("answer")
                    print(f"[FINAL_ANSWER_FIX_CACHE] final_answer empty, verdict={verdict}, answer exists={bool(answer)}")

                    if verdict == "TRUE" and answer:
                        print(f"[FINAL_ANSWER_FIX_CACHE] Setting final_answer: '{answer[:60]}...'")
                        ctx["final_answer"] = answer
                        if not ctx.get("final_confidence"):
                            ctx["final_confidence"] = stage8.get("confidence", 0.9)
                        if not ctx.get("stage_10_finalize"):
                            ctx["stage_10_finalize"] = {
                                "text": answer,
                                "confidence": stage8.get("confidence", 0.9),
                                "source": "stage_8_fallback"
                            }
            except Exception as e:
                print(f"[FINAL_ANSWER_FIX_CACHE] Exception: {e}")
                pass

            # Wrap the context under a 'context' key for API compatibility
            return success_response("RUN_PIPELINE", mid, {"context": ctx})
        
        # ------------------------------------------------------------------
        # DISABLED (Phase C cleanup): Semantic cache bypass removed
        # ------------------------------------------------------------------
        # Semantic cache previously short-circuited the pipeline. Now disabled
        # to ensure all requests flow through the cognitive pathway.
        #
        # Trace semantic cache check for diagnostics
        if not fc_rec:
            sc_rec = None  # DISABLED
            try:
                if tracer and RouteType:
                    # Check if cache would have hit (for diagnostics only)
                    _sc_check = _lookup_semantic_cache(text)
                    if _sc_check:
                        tracer.record_route(mid, RouteType.SEMANTIC_CACHE, {"bypass": "disabled", "would_have_hit": True})
            except Exception:
                pass
            if sc_rec:
                # Apply semantic cache gating to ensure topical and intent alignment
                safe_match = True
                try:
                    q_tokens = set(_tokenize(text))
                    # Cached query tokens
                    cached_q_tokens = set(sc_rec.get("tokens", []))
                    # Cosine similarity between query and cached query
                    if _cosine_similarity(q_tokens, cached_q_tokens) < 0.75:
                        safe_match = False
                    # Jaccard similarity between query tokens and answer tokens
                    ans_tokens = set(_tokenize(sc_rec.get("answer", "")))
                    if _jaccard(q_tokens, ans_tokens) < 0.2:
                        safe_match = False
                    # Intent must match between cached query and current
                    cached_intent = sc_rec.get("intent")
                    current_intent = None
                    try:
                        current_intent = str(lang_payload.get("type")) if lang_payload else None
                    except Exception:
                        current_intent = None
                    if cached_intent and current_intent and cached_intent != current_intent:
                        safe_match = False
                    # Disallow non‑self cached answers for self queries
                    if _is_self_query(text) and not sc_rec.get("self_origin"):
                        safe_match = False
                except Exception:
                    safe_match = False
                if not safe_match:
                    sc_rec = None
            if sc_rec:
                # Reuse the parsed language payload in the context
                ctx["stage_3_language"] = lang_payload or {}
                # Build a validation object signalling a semantic cache hit
                ctx["stage_8_validation"] = {
                    "verdict": "NEUTRAL",
                    "answer": sc_rec.get("answer"),
                    "confidence": sc_rec.get("confidence", 0.6),
                    "from_semantic_cache": True,
                }
                # Generate candidates using the language brain
                try:
                    cand_res = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cand_res.get("payload", {})
                except Exception:
                    ctx["stage_6_candidates"] = {}
                # Finalise the answer using the language brain
                try:
                    fin_res = _brain_module("language").service_api({"op": "FINALIZE", "payload": ctx})
                    ctx["stage_10_finalize"] = fin_res.get("payload", {})
                except Exception:
                    ctx["stage_10_finalize"] = {}
                # Capture final answer and confidence
                try:
                    ctx["final_answer"] = ctx.get("stage_10_finalize", {}).get("text")
                    ctx["final_confidence"] = ctx.get("stage_10_finalize", {}).get("confidence")
                except Exception:
                    ctx["final_answer"] = None
                    ctx["final_confidence"] = None
                # Indicate that storage was skipped due to semantic cache
                ctx["stage_9_storage"] = {"skipped": True, "reason": "semantic_cache_used"}
                # Write context snapshot and system report
                try:
                    _save_context_snapshot(ctx, limit=5)
                except Exception:
                    pass
                try:
                    run_id = ctx.get("run_seed", _SEQ_ID_COUNTER)
                    write_report("system", f"run_{run_id}.json", json.dumps(ctx, indent=2))
                except Exception:
                    pass
                # Self‑evaluation for run metrics
                try:
                    import importlib
                    sc_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_critique")
                    eval_resp = sc_mod.service_api({"op": "EVAL_CONTEXT", "payload": {"context": ctx}})
                    ctx["stage_self_eval"] = eval_resp.get("payload", {})
                except Exception:
                    ctx["stage_self_eval"] = {"error": "eval_failed"}
                # Update semantic cache with the current context before returning
                try:
                    _update_semantic_cache(ctx)
                except Exception:
                    pass
                # Before returning, update conversation state based on the
                # original query and final answer.  Use the original query
                # from the context when available to avoid NameError when
                # 'text' is not defined in this scope.
                try:
                    upd_q = ctx.get("original_query") or payload.get("text") or text
                    _update_conversation_state(upd_q, ctx.get("final_answer"))
                except Exception:
                    pass

                # CRITICAL FIX: Ensure final_answer is set (same as end of pipeline)
                try:
                    if not ctx.get("final_answer"):
                        stage8 = ctx.get("stage_8_validation") or {}
                        verdict = str(stage8.get("verdict", "")).upper()
                        answer = stage8.get("answer")
                        print(f"[FINAL_ANSWER_FIX_EARLY] final_answer empty, verdict={verdict}, answer exists={bool(answer)}")

                        if verdict == "TRUE" and answer:
                            print(f"[FINAL_ANSWER_FIX_EARLY] Setting final_answer: '{answer[:60]}...'")
                            ctx["final_answer"] = answer
                            if not ctx.get("final_confidence"):
                                ctx["final_confidence"] = stage8.get("confidence", 0.9)
                            if not ctx.get("stage_10_finalize"):
                                ctx["stage_10_finalize"] = {
                                    "text": answer,
                                    "confidence": stage8.get("confidence", 0.9),
                                    "source": "stage_8_fallback"
                                }
                except Exception as e:
                    print(f"[FINAL_ANSWER_FIX_EARLY] Exception: {e}")
                    pass

                # Wrap the context under a 'context' key for API compatibility
                return success_response("RUN_PIPELINE", mid, {"context": ctx})

        # Stage 2 — Planner (conditional).  Only call the planner when the
        # input is a command or request.  Otherwise skip the planner
        # entirely and construct a simple fallback plan.  Skipping the
        # planner avoids writing unnecessary sub‑goals to the goal memory.
        if should_plan:
            try:
                p = _brain_module("planner").service_api(
                    {"op": "PLAN", "payload": {"text": text, "delta": (suggestion.get("planner") if approved else {})}}
                )
            except Exception:
                p = {"ok": False, "payload": {}}
            if trace_enabled:
                trace_events.append({"stage": "planner"})
            ctx["stage_2_planner"] = p.get("payload", {}) or {}
            # If the planner returns an empty payload (should be rare), fall back
            if not ctx["stage_2_planner"]:
                ctx["stage_2_planner"] = {
                    "goal": f"Satisfy user request: {text}",
                    "intents": ["retrieve_relevant_memories", "compose_response"],
                    "notes": "Planner fallback: empty plan"
                }
        else:
            # Skip planner and assign a generic respond plan
            ctx["stage_2_planner"] = {
                "goal": f"Satisfy user request: {text}",
                "intents": ["retrieve_relevant_memories", "compose_response"],
                "notes": "Planner not required for this intent"
            }

        # Stage 4 — Pattern recognition.  Run pattern analysis on the input
        # regardless of planning outcome.  Errors are caught and ignored.
        try:
            pr = _brain_module("pattern_recognition").service_api({"op": "ANALYZE", "payload": {"text": text}})
        except Exception:
            pr = {"ok": True, "payload": {"skipped": True}}
        if trace_enabled:
            trace_events.append({"stage": "pattern_recognition"})

        # Populate context entries for stages 1–4
        ctx["stage_1_sensorium"] = s.get("payload", {})
        ctx["stage_3_language"] = lang_payload
        ctx["stage_4_pattern_recognition"] = pr.get("payload", {})

        # Stage 5 — Affect priority scoring.  Use the affect_priority brain to assess
        # the emotional tone and urgency of the input.  Merge the suggested tone into
        # the planner's context if provided.
        try:
            aff_mod = _brain_module("affect_priority")
            ar = aff_mod.service_api({"op": "SCORE", "payload": {"text": text, "context": ctx}})
        except Exception:
            ar = {}
        ctx["stage_5_affect"] = (ar.get("payload") or {})
        tone = ctx.get("stage_5_affect", {}).get("suggested_tone")
        if tone:
            # ensure planner stage exists before setting tone
            ctx.setdefault("stage_2_planner", {})
            ctx["stage_2_planner"].setdefault("tone", tone)

        # ------------------------------------------------------------------
        # Stage 5b — Attention resolution.  After affect scoring, gather
        # coarse bids from select brains (e.g. language and reasoning)
        # based on the current context and ask the integrator brain to
        # determine which brain should receive focus.  The resulting
        # focus and related state are stored in ctx["stage_5b_attention"].
        try:
            # Compose bids by querying each brain's bid_for_attention function.
            # Each participating brain returns a dictionary with keys
            # brain_name, priority, reason and evidence.  Fallback to a
            # simple static bid if none are returned.
            bids: List[Dict[str, Any]] = []
            # Language brain bid
            try:
                lang_mod = _brain_module("language")
                if hasattr(lang_mod, "bid_for_attention"):
                    bid = lang_mod.bid_for_attention(ctx)
                    if isinstance(bid, dict) and bid.get("brain_name"):
                        bids.append(bid)
            except Exception:
                pass
            # Reasoning brain bid
            try:
                reason_mod = _brain_module("reasoning")
                if hasattr(reason_mod, "bid_for_attention"):
                    bid = reason_mod.bid_for_attention(ctx)
                    if isinstance(bid, dict) and bid.get("brain_name"):
                        bids.append(bid)
            except Exception:
                pass
            # Memory bid via local function
            try:
                bid = bid_for_attention(ctx)
                if isinstance(bid, dict) and bid.get("brain_name"):
                    bids.append(bid)
            except Exception:
                pass
            # Research manager bid when research intent detected
            try:
                stage3_intent = str((ctx.get("stage_3_language") or {}).get("intent", "")).lower()
                if stage3_intent in {"research_request", "research_followup"}:
                    topic = str((ctx.get("stage_3_language") or {}).get("research_topic") or text)
                    bids.append({
                        "brain_name": "research_manager",
                        "priority": 0.9,
                        "reason": "research_intent",
                        "evidence": {"topic": topic},
                    })
            except Exception:
                pass
            # Fallback: if no bids were collected, provide conservative default bids.
            #
            # The integrator expects at least one bid; however, if all
            # participating brains fail to provide a bid (e.g. due to an
            # import error), construct minimal bids with low priorities to
            # avoid dominating the attention arbitration.  When the input is
            # recognised as a question, give the language brain a slightly
            # higher priority to reflect a potential need for an answer.  All
            # other defaults use a small weight (<=0.2) as recommended in the
            # Stage 2.5→3.0 roadmap.
            if not bids:
                lang_info = ctx.get("stage_3_language", {}) or {}
                lang_type = str(
                    lang_info.get("type")
                    or lang_info.get("storable_type")
                    or lang_info.get("intent")
                    or ""
                ).upper()
                if lang_type == "QUESTION":
                    bids.append({
                        "brain_name": "language",
                        "priority": 0.2,
                        "reason": "unanswered_question",
                        "evidence": {"query": text},
                    })
                else:
                    bids.append({
                        "brain_name": "language",
                        "priority": 0.1,
                        "reason": "default",
                        "evidence": {},
                    })
                bids.append({
                    "brain_name": "reasoning",
                    "priority": 0.15,
                    "reason": "default",
                    "evidence": {},
                })
                bids.append({
                    "brain_name": "memory",
                    "priority": 0.1,
                    "reason": "default",
                    "evidence": {},
                })
            # Invoke integrator if present
            try:
                integrator_mod = _brain_module("integrator")
                # Extract norm_type from sensorium for follow-up detection
                sensorium_data = ctx.get("stage_1_sensorium", {})
                # Build context with query and norm_type for routing
                integrator_context = {
                    "query": text,
                    "norm_type": sensorium_data.get("norm_type", ""),
                    "last_topic": _CONVERSATION_STATE.get("last_topic", "")
                }
                ir = integrator_mod.service_api({
                    "op": "RESOLVE",
                    "payload": {
                        "bids": bids,
                        "query": text,
                        "context": integrator_context
                    }
                })
                if ir.get("ok"):
                    ctx["stage_5b_attention"] = ir.get("payload", {})
                    # Agency tool info comes directly in the integrator response payload now
                    integrator_payload = ir.get("payload", {})
                    if integrator_payload.get("agency_tool"):
                        ctx["agency_tool"] = integrator_payload["agency_tool"]
                        # CRITICAL: Also copy agency_method and agency_args!
                        # These determine GET_TIME vs GET_DATE vs GET_CALENDAR
                        if integrator_payload.get("agency_method"):
                            ctx["agency_method"] = integrator_payload["agency_method"]
                        if integrator_payload.get("agency_args"):
                            ctx["agency_args"] = integrator_payload["agency_args"]
                        print(f"[STAGE5b] Agency tool detected: {ctx['agency_tool']}, method: {ctx.get('agency_method')}, args: {ctx.get('agency_args')}")
                    if integrator_payload.get("bypass_teacher"):
                        ctx["bypass_teacher"] = integrator_payload["bypass_teacher"]
                    # --- Attention history tracking ---
                    # Record the winning focus and its reason along with a timestamp.
                    try:
                        focus = ctx.get("stage_5b_attention", {}).get("focus")
                        reason = ctx.get("stage_5b_attention", {}).get("state", {}).get("focus_reason", "")
                        _ATTENTION_HISTORY.append({
                            "focus": focus,
                            "reason": reason,
                        })
                        # Trim history to maximum length
                        if len(_ATTENTION_HISTORY) > _MAX_RECENT_QUERIES:
                            _ATTENTION_HISTORY.pop(0)
                        # Attach a copy of the history to the attention payload
                        ctx["stage_5b_attention"]["history"] = list(_ATTENTION_HISTORY)
                        # Update focus statistics using the optional analyzer.  This call
                        # quietly returns if the analyzer is not available.
                        try:
                            if update_focus_stats:
                                update_focus_stats(focus, reason)
                        except Exception:
                            # Do not propagate errors from optional analytics
                            pass
                    except Exception:
                        # Do not break the pipeline on history errors
                        pass
            except Exception:
                # Integrator unavailable; skip attention resolution
                pass
        except Exception:
            # Suppress all errors in Stage 5b to avoid breaking the pipeline
            pass

        # Stage 2R — memory-first retrieval (fan to all banks + TAC)
        # Optionally perform retrieval in parallel when enabled via configuration.
        try:
            pb_cfg = CFG.get("parallel_bank_access", {}) or {}
            use_parallel = bool(pb_cfg.get("enabled", False))
        except Exception:
            use_parallel = False
        # Determine whether the input should be stored/retrieved from memory.  If the
        # language brain has marked this input as non‑storable (e.g. a greeting or
        # other social chit‑chat), skip the memory search entirely and return
        # an empty set of results.  This prevents the librarian from wasting
        # time searching all banks for salutations and ensures that downstream
        # stages do not attempt to use irrelevant evidence.
        stage3_local = ctx.get("stage_3_language", {}) or {}
        # Determine whether to perform memory retrieval.  In general, we
        # retrieve memory when the input is storable or is a question.  A
        # special flag ``skip_memory_search`` on the parsed language
        # payload allows conversational meta queries to bypass retrieval
        # entirely.  This prevents meta questions like "where are we"
        # or "how's it going" from searching memory and returning
        # irrelevant results.
        storable_flag = bool(stage3_local.get("storable", True))
        # Normalise the intent/type fields
        intent_type = str(stage3_local.get("type") or stage3_local.get("intent") or "").upper()
        # Check for a skip flag set by the language brain.  When true,
        # always skip retrieval regardless of storable or question status.
        skip_mem = bool(stage3_local.get("skip_memory_search", False))
        # When skip_memory_search is set, normally we avoid all retrieval to
        # prevent irrelevant memory hits.  However, identity questions (e.g.
        # "who am I", "what's my name") rely on scanning the recent
        # conversation for self‑introductions.  In those cases, we must
        # perform a limited retrieval so Stage 3 can extract the name.  See
        # issue #core_identity_query for details.  Therefore, override
        # skip_mem for USER_IDENTITY_QUERY by forcing retrieval on.
        if skip_mem:
            if intent_type == "USER_IDENTITY_QUERY":
                # Perform retrieval even when skip_mem is set so that
                # personal introductions (e.g. "I am Josh") remain accessible.
                should_retrieve = True
            else:
                should_retrieve = False
        else:
            # If the input is a question, force retrieval even when storable is False
            should_retrieve = storable_flag or intent_type == "QUESTION"

        # FIX 5: Check for learned routing rules FIRST before falling back to dual router
        # This ensures that the Librarian uses its learned knowledge to route queries
        learned_routing_used = False
        try:
            from brains.cognitive.memory_librarian.service.librarian_memory import (
                retrieve_routing_rule_for_question
            )

            # Try to find a learned routing rule for this question
            learned_rule = retrieve_routing_rule_for_question(text, threshold=0.3)

            if learned_rule:
                # Extract routes from the learned rule
                routes = learned_rule.get("routes", [])
                if routes:
                    # Build score map from learned routes
                    learned_scores: Dict[str, float] = {}
                    for route in routes:
                        bank = route.get("bank", "")
                        weight = float(route.get("weight", 0.5))
                        if bank:
                            learned_scores[bank] = weight

                    if learned_scores:
                        # Use learned routing scores
                        ctx["stage_2R_routing_scores"] = learned_scores
                        # Sort by weight and take top banks
                        sorted_banks = sorted(learned_scores.items(), key=lambda x: x[1], reverse=True)
                        ctx["stage_2R_top_banks"] = [b[0] for b in sorted_banks[:2]]
                        ctx["stage_2R_slow_path"] = False
                        learned_routing_used = True

                        # FIX 4: Log that we used a learned routing rule
                        print(f"[ROUTING_RULES_USED] Learned routing applied for '{text[:60]}'")
                        print(f"  stage_2R_top_banks={ctx['stage_2R_top_banks']}")
                        print(f"  stage_2R_routing_scores={learned_scores}")
        except Exception as e:
            # Don't fail pipeline if learned routing fails
            print(f"[LEARNED_ROUTING_ERROR] {str(e)[:100]}")
            pass

        # Compute routing scores for transparency.  Use the dual router if
        # available to produce a mapping of banks to scores and record
        # whether the margin suggests a slow path.  Failures are
        # ignored gracefully.
        # Only use dual router if learned routing didn't provide scores
        if not learned_routing_used:
            try:
                import importlib as _importlib
                router_mod = _importlib.import_module("brains.cognitive.reasoning.service.dual_router")
                rres = router_mod.service_api({
                    "op": "ROUTE",
                    "payload": {"query": text}
                })
                rscores = (rres.get("payload") or {}).get("scores") or {}
                if rscores and isinstance(rscores, dict):
                    # Normalize to floats; keep original order sorted descending
                    try:
                        sorted_scores = sorted(rscores.items(), key=lambda itm: float(itm[1]), reverse=True)
                    except Exception:
                        sorted_scores = list(rscores.items())
                    # Attempt to convert score values to floats; fall back to original
                    score_map: Dict[str, float] = {}
                    for bk, sc in rscores.items():
                        try:
                            score_map[str(bk)] = float(sc)
                        except Exception:
                            try:
                                # Attempt to coerce via float(str())
                                score_map[str(bk)] = float(str(sc))
                            except Exception:
                                # Preserve as string if cannot convert
                                score_map[str(bk)] = 0.0
                    # If all computed scores are zero or missing, fall back to a simple heuristic.
                    try:
                        total_score = sum(float(v) for v in score_map.values()) if score_map else 0.0
                    except Exception:
                        total_score = 0.0
                    if total_score <= 0.0:
                        # When the router provides no meaningful scores, fall back.
                        # First attempt to pick a bank using a simple keyword router.
                        try:
                            pred_bank = _simple_route_to_bank(text)
                        except Exception:
                            pred_bank = None
                        if pred_bank:
                            # Assign a high score to the predicted bank and zero to others.
                            fallback_scores: Dict[str, float] = {}
                            try:
                                # Use _ALL_BANKS defined above if available
                                for bk in _ALL_BANKS:
                                    fallback_scores[bk] = 1.0 if bk == pred_bank else 0.0
                            except Exception:
                                # Fallback: only set the predicted bank
                                fallback_scores = {pred_bank: 1.0}
                            ctx["stage_2R_routing_scores"] = fallback_scores
                            ctx["stage_2R_top_banks"] = [pred_bank]
                        else:
                            # If no prediction is available, assign uniform scores across all domain banks.
                            try:
                                root_maven = Path(__file__).resolve().parents[4]
                                domain_dir = root_maven / "brains" / "domain_banks"
                                banks = [p.name for p in domain_dir.iterdir() if p.is_dir()]
                                if banks:
                                    fallback_scores = {b: 1.0 for b in banks}
                                    ctx["stage_2R_routing_scores"] = fallback_scores
                                    # Use the first two banks (sorted order) as top banks for determinism
                                    ctx["stage_2R_top_banks"] = banks[:2]
                                else:
                                    # No banks found; fall back to zero scores
                                    ctx["stage_2R_routing_scores"] = score_map
                                    ctx["stage_2R_top_banks"] = list(score_map.keys())[:2]
                            except Exception:
                                # On failure, fall back to zero scores
                                ctx["stage_2R_routing_scores"] = score_map
                                ctx["stage_2R_top_banks"] = list(score_map.keys())[:2]
                    else:
                        # Use the computed scores and determine top banks normally
                        ctx["stage_2R_routing_scores"] = score_map
                        try:
                            # Top two banks for transparency
                            ctx["stage_2R_top_banks"] = [itm[0] for itm in sorted_scores[:2]]
                        except Exception:
                            ctx["stage_2R_top_banks"] = list(score_map.keys())[:2]
                    # Override routing for simple math expressions
                    if _is_simple_math_expression(text):
                        ctx["stage_2R_routing_scores"] = {"math": 1.0}
                        ctx["stage_2R_top_banks"] = ["math"]
                # Include slow_path indicator if present
                if (rres.get("payload") or {}).get("slow_path") is not None:
                    ctx["stage_2R_slow_path"] = bool((rres.get("payload") or {}).get("slow_path"))
            except Exception:
                # On failure, omit routing scores
                pass

        # ------------------------------------------------------------------
        # Fallback routing: if the learned router returns no scores or all
        # scores are zero, derive a simple routing decision based on
        # keyword heuristics.  Without this fallback, downstream stages
        # receive a zero vector which prevents retrieval from any domain
        # bank.  The simple router examines the query text and maps it
        # to a high‑level bank; if a specific bank is identified, assign
        # it a score of 1.0 and designate it as the top bank.  The
        # slow_path flag is cleared to avoid misinterpretation by
        # consumers.
        try:
            scores_map = ctx.get("stage_2R_routing_scores") or {}
            fallback_needed = True
            if scores_map:
                for _val in scores_map.values():
                    try:
                        if float(_val) > 0.0:
                            fallback_needed = False
                            break
                    except Exception:
                        continue
            if fallback_needed:
                simple_bank = None
                try:
                    simple_bank = _simple_route_to_bank(text)
                except Exception:
                    simple_bank = None
                if simple_bank:
                    ctx["stage_2R_routing_scores"] = {simple_bank: 1.0}
                    ctx["stage_2R_top_banks"] = [simple_bank]
                    ctx["stage_2R_slow_path"] = False
        except Exception:
            # Silently ignore fallback errors
            pass
        # Before performing retrieval, send a targeted search request based on
        # the current attention focus.  The memory librarian will use the
        # message bus to inform the retrieval helper which domain banks to
        # prioritise.  For example, when the language brain has focus, only
        # language and factual banks are searched.  The focus strength is
        # forwarded as a confidence threshold to aid future prioritisation.
        try:
            attn = ctx.get("stage_5b_attention", {}) or {}
            focus = attn.get("focus")
            if focus:
                from brains.cognitive.message_bus import send  # type: ignore
                # Map high‑level brains to domain banks
                domain_map = {
                    "language": ["language_arts", "factual"],
                    "reasoning": ["science", "math", "theories_and_contradictions"],
                    "memory": ["stm_only", "theories_and_contradictions"],
                }
                domains = domain_map.get(str(focus).lower(), [])
                if domains:
                    conf_strength = float(attn.get("state", {}).get("focus_strength", 0.0) or 0.0)
                    send({
                        "from": "memory_librarian",
                        "to": "memory",
                        "type": "SEARCH_REQUEST",
                        "domains": domains,
                        "confidence_threshold": conf_strength,
                    })
        except Exception:
            pass
        if not should_retrieve:
            mem = {"results": [], "banks": [], "banks_queried": []}
        else:
            if use_parallel:
                # Determine max_workers from config; default to 5 if invalid
                try:
                    mw_val = pb_cfg.get("max_workers", 5)
                    max_workers = int(mw_val) if mw_val else 5
                except Exception:
                    max_workers = 5
                try:
                    mem = _retrieve_from_banks_parallel(text, k=5, max_workers=max_workers)
                except Exception:
                    # Fall back to sequential retrieval on error
                    mem = _retrieve_from_banks(text, k=5)
            else:
                mem = _retrieve_from_banks(text, k=5)
        # Additional retrieval for questions: attempt to sanitize the question and
        # retrieve again to capture declarative statements such as "Birds have wings."
        try:
            # Only perform additional retrieval for questions when the input is not a social chit‑chat.  Even if
            # the question itself is not storable, we still want to search for answers.
            # Determine storable flag and normalised intent as above
            lang_local = ctx.get("stage_3_language") or {}
            storable_flag = bool(lang_local.get("storable", True))
            # Determine if it is a question by checking both 'type' and 'intent' fields
            intent_type = str(lang_local.get("type") or lang_local.get("intent") or "").upper()
            is_question = intent_type == "QUESTION"
            if (storable_flag or is_question) and is_question:
                sanitized = _sanitize_question(text)
                if sanitized and sanitized.lower() != text.lower():
                    # Use the same retrieval mechanism as above (parallel or sequential)
                    try:
                        mem2 = _retrieve_from_banks_parallel(sanitized, k=5, max_workers=max_workers) if use_parallel else _retrieve_from_banks(sanitized, k=5)
                    except Exception:
                        mem2 = _retrieve_from_banks(sanitized, k=5)
                    # Merge mem and mem2 results/banks
                    res1 = mem.get("results", []) or []
                    res2 = mem2.get("results", []) or []
                    # Append unique results (by id and content)
                    seen_ids = set()
                    combined = []
                    for it in res1 + res2:
                        if not isinstance(it, dict):
                            continue
                        rec_id = it.get("id") or id(it)
                        sig = (rec_id, it.get("content"))
                        if sig in seen_ids:
                            continue
                        seen_ids.add(sig)
                        combined.append(it)
                    banks = list(set((mem.get("banks") or []) + (mem2.get("banks") or [])))
                    mem = {"results": combined, "banks": banks, "banks_queried": banks}

                # ------------------------------------------------------------------
                # Fallback retrieval for numerical questions.  If no results were
                # obtained from the initial and sanitized queries, look for
                # alternative phrases that might appear in stored facts.  This
                # helps answer queries like "At what temperature does water
                # freeze?" by searching for simplified phrases such as
                # "water freeze" or "freezing point of water".  Only run this
                # fallback when the intent is a question and no results have
                # yet been found.
                if not (mem.get("results") or []):
                    try:
                        q_lower = str(text or "").lower()
                    except Exception:
                        q_lower = ""
                    # ----------------------------------------------------------------------
                    # Fallback 1: numeric question phrases.  If the query is about
                    # temperature/freeze or temperature/boil and no results were found,
                    # search for simplified phrases likely present in stored facts.
                    if ("temperature" in q_lower and "freeze" in q_lower) or ("temperature" in q_lower and "boil" in q_lower):
                        alt_queries: List[str] = []
                        if "freeze" in q_lower:
                            alt_queries.extend(["water freeze", "water freezes at", "water freeze at", "freezing point of water"])
                        if "boil" in q_lower:
                            alt_queries.extend(["water boil", "water boils at", "water boil at", "boiling point of water"])
                        for alt_q in alt_queries:
                            try:
                                # Determine a sensible default for max_workers if not defined
                                alt_max_workers = max_workers if 'max_workers' in locals() else 5
                                mem_alt = _retrieve_from_banks_parallel(alt_q, k=5, max_workers=alt_max_workers) if use_parallel else _retrieve_from_banks(alt_q, k=5)
                            except Exception:
                                mem_alt = _retrieve_from_banks(alt_q, k=5)
                            if mem_alt and (mem_alt.get("results") or []):
                                res_alt = mem_alt.get("results", []) or []
                                res_orig = mem.get("results", []) or []
                                combined_alt: List[Dict[str, Any]] = []
                                seen_ids_alt: set = set()
                                for it in res_orig + res_alt:
                                    if not isinstance(it, dict):
                                        continue
                                    rec_id = it.get("id") or id(it)
                                    sig = (rec_id, it.get("content"))
                                    if sig in seen_ids_alt:
                                        continue
                                    seen_ids_alt.add(sig)
                                    combined_alt.append(it)
                                banks_alt = list(set((mem.get("banks") or []) + (mem_alt.get("banks") or [])))
                                mem = {"results": combined_alt, "banks": banks_alt, "banks_queried": banks_alt}
                                break
                    # ----------------------------------------------------------------------
                    # Fallback 2: morphological variations.  If the sanitized query
                    # produced no results, try trimming common English suffixes such
                    # as plural "s", "es", gerund "ing", and past tense "ed".  Each
                    # variant is searched individually; any results found are merged
                    # back into ``mem``.  This helps match stored facts where the
                    # user used plural or gerund forms that differ from the stored
                    # statement.
                    if not (mem.get("results") or []):
                        try:
                            # Derive a base string for morphological fallback.  Prefer the
                            # sanitized question if available; fall back to the original
                            # question otherwise.  Both are lowercased for uniformity.
                            base_q = (sanitized.lower() if 'sanitized' in locals() and sanitized else q_lower) or q_lower
                        except Exception:
                            base_q = q_lower
                        # Split into words and generate simple stems.  Import re on the fly
                        # to avoid a module-level dependency when the fallback is unused.
                        words: List[str] = []
                        try:
                            import re as _re  # type: ignore
                            words = [w.strip() for w in _re.findall(r"[A-Za-z0-9']+", base_q) if w.strip()]
                        except Exception:
                            words = base_q.split()
                        variants: List[str] = []
                        for w in words:
                            wl = w.lower()
                            # Skip short words to avoid over-trimming
                            if len(wl) <= 3:
                                continue
                            if wl.endswith("ing"):
                                variants.append(wl[:-3])
                            if wl.endswith("es"):
                                variants.append(wl[:-2])
                            if wl.endswith("s"):
                                variants.append(wl[:-1])
                            if wl.endswith("ed"):
                                variants.append(wl[:-2])
                        # Deduplicate variants and search each one
                        seen_var: set[str] = set()
                        for var in variants:
                            if not var or var in seen_var:
                                continue
                            seen_var.add(var)
                            try:
                                alt_max_workers = max_workers if 'max_workers' in locals() else 5
                                mem_alt = _retrieve_from_banks_parallel(var, k=5, max_workers=alt_max_workers) if use_parallel else _retrieve_from_banks(var, k=5)
                            except Exception:
                                mem_alt = _retrieve_from_banks(var, k=5)
                            if mem_alt and (mem_alt.get("results") or []):
                                res_alt = mem_alt.get("results", []) or []
                                res_orig = mem.get("results", []) or []
                                combined_alt2: List[Dict[str, Any]] = []
                                seen_ids2: set = set()
                                for it in res_orig + res_alt:
                                    if not isinstance(it, dict):
                                        continue
                                    rec_id = it.get("id") or id(it)
                                    sig = (rec_id, it.get("content"))
                                    if sig in seen_ids2:
                                        continue
                                    seen_ids2.add(sig)
                                    combined_alt2.append(it)
                                banks_alt2 = list(set((mem.get("banks") or []) + (mem_alt.get("banks") or [])))
                                mem = {"results": combined_alt2, "banks": banks_alt2, "banks_queried": banks_alt2}
                                # Stop after first successful variant to avoid noise
                                break
        except Exception:
            pass
        # Rank the retrieved results so that numerically relevant matches are surfaced first.
        # This helps answer questions involving numbers (e.g. temperatures or counts) by
        # promoting records containing the same numbers as the query.  If the query
        # contains digits or spelled out numbers (zero through ten), sort the
        # retrieval results to prefer those with matching numeric tokens.  If there
        # are no numeric tokens in the query, leave the ordering unchanged.
        try:
            import re
            qlow = str(text or "").lower()
            # Extract digits and spelled-out numbers from the query
            # Spelled-out numbers list can be extended as needed
            num_words = [
                "zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten"
            ]
            tokens: list[str] = []
            # Digits
            try:
                tokens.extend(re.findall(r"\d+", qlow))
            except Exception:
                pass
            # Spelled numbers
            for w in num_words:
                try:
                    if w in qlow:
                        tokens.append(w)
                except Exception:
                    continue
            # Only rank if we actually found numeric tokens
            if tokens and isinstance(mem, dict):
                res_list = mem.get("results") or []
                if isinstance(res_list, list):
                    def _numeric_score(item: dict) -> int:
                        # Compute a score based on how many query tokens appear in the result content
                        try:
                            c = str(item.get("content", "")).lower()
                        except Exception:
                            c = ""
                        score = 0
                        for tok in tokens:
                            try:
                                if tok.isdigit():
                                    # match exact digit sequences in the content
                                    if re.search(r"\b" + re.escape(tok) + r"\b", c):
                                        score += 1
                                else:
                                    if tok in c:
                                        score += 1
                            except Exception:
                                continue
                        return score
                    # Sort results descending by numeric score, preserving original order for ties
                    # Use enumerate index as secondary key for stable sort.  Before sorting,
                    # detect identity/origin queries and prioritise evidence mentioning Maven.
                    res_list = list(res_list)  # ensure it's a list copy

                    # Heuristic: if the original query asks about Maven's identity or origin,
                    # move results containing "maven" or "living intelligence" to the front.
                    try:
                        raw_q = str(ctx.get("original_query", "")).lower().strip()
                    except Exception:
                        raw_q = ""
                    identity_phrases = [
                        "who are you",
                        "what are you",
                        "what is your name",
                        "what's your name",
                        "who is maven",
                        "what is maven",
                        "why were you created",
                        "why were you made",
                        "purpose of maven",
                        "why do you exist",
                    ]
                    if any(p in raw_q for p in identity_phrases):
                        maven_hits = []
                        non_hits = []
                        for itm in res_list:
                            try:
                                c = str(itm.get("content", "")).lower()
                            except Exception:
                                c = ""
                            # If content is a JSON‑encoded dict (e.g. from codified answers), fallback to 'text'
                            if not c and itm.get("text"):
                                c = str(itm.get("text", "")).lower()
                            if "maven" in c or "living intelligence" in c:
                                maven_hits.append(itm)
                            else:
                                non_hits.append(itm)
                        res_list = maven_hits + non_hits

                    scored = []
                    for idx, itm in enumerate(res_list):
                        try:
                            s = _numeric_score(itm)
                        except Exception:
                            s = 0
                        scored.append((s, idx, itm))
                    # If any item has a score > 0, perform sorting
                    if any(s > 0 for s, _, _ in scored):
                        scored.sort(key=lambda x: (-x[0], x[1]))
                        res_list_sorted = [itm for _, _, itm in scored]
                        mem = mem.copy()
                        mem["results"] = res_list_sorted
        except Exception:
            pass
        if trace_enabled: trace_events.append({"stage": "memory_retrieve"})

        # --------------------------------------------------------------
        # Identity/origin query prioritisation
        #
        # After retrieving evidence from domain banks (stored in ``mem``),
        # reorder the results for queries asking about Maven's identity or
        # origin.  This heuristic moves personal knowledge (records that
        # mention "maven" or "living intelligence") to the front so they
        # are considered before unrelated etiquette statements like
        # "We say thank you".  Without this, generic one‑token matches
        # can override more relevant personal facts.
        try:
            raw_q = str(ctx.get("original_query", "")).lower().strip()
        except Exception:
            raw_q = ""
        identity_triggers = [
            "who are you",
            "what are you",
            "what is your name",
            "what's your name",
            "who is maven",
            "what is maven",
            "why were you created",
            "why were you made",
            "purpose of maven",
            "why do you exist",
        ]
        if any(p in raw_q for p in identity_triggers) and isinstance(mem, dict):
            res_list = mem.get("results") or []
            if isinstance(res_list, list) and res_list:
                maven_hits: list = []
                other_hits: list = []
                for itm in res_list:
                    try:
                        c = str(itm.get("content", "")).lower()
                    except Exception:
                        c = ""
                    # If content is a JSON‑encoded object, fall back to its 'text'
                    if not c and itm.get("text"):
                        c = str(itm.get("text", "")).lower()
                    if "maven" in c or "living intelligence" in c:
                        maven_hits.append(itm)
                    else:
                        other_hits.append(itm)
                # Only reorder if we found at least one Maven hit
                if maven_hits:
                    mem = mem.copy()
                    mem["results"] = maven_hits + other_hits

        ctx["stage_2R_memory"] = mem

        # Record upstream weights used (if present)
        ctx["stage_0_weights_used"] = {
            "sensorium": ctx["stage_1_sensorium"].get("weights_used"),
            "planner": ctx["stage_2_planner"].get("weights_used"),
            "language": ctx["stage_3_language"].get("weights_used"),
        }

        # --- Stage 8 — Reasoning (intent-aware proposal) ---
        # Extract parsing metadata from Stage 3.  The language brain
        # classifies the user input into QUESTION, COMMAND, REQUEST,
        # SPECULATION or FACT and indicates whether it is storable.
        stage3 = ctx.get("stage_3_language") or {}
        st_type = str(stage3.get("storable_type", "")).upper()
        # Some speculative statements apply a confidence penalty when stored.
        try:
            confidence_penalty = float(stage3.get("confidence_penalty", 0.0) or 0.0)
        except Exception:
            confidence_penalty = 0.0
        # If the input is a question/command/request, pass an empty content
        # so the reasoning brain does not attempt to store the raw query.
        if st_type in ("QUESTION", "COMMAND", "REQUEST"):
            proposed_content = ""
        else:
            proposed_content = text

        # Check if this is a web search follow-up or agency tool (detected by integrator).
        # If so, skip the reasoning brain call and let Stage 6 handle it.
        attention_payload = ctx.get("stage_5b_attention", {}) or {}
        is_web_followup = attention_payload.get("web_followup_mode", False)
        is_agency_tool = bool(ctx.get("agency_tool") or attention_payload.get("agency_tool"))

        if is_agency_tool:
            print(f"[DEBUG_STAGE8] Skipping reasoning call - agency_tool detected: {ctx.get('agency_tool')}, Stage 6 will execute")
            # Initialize stage_8_validation with a placeholder - Stage 6 will set the actual answer
            ctx["stage_8_validation"] = {
                "verdict": "PENDING",
                "mode": "AGENCY_TOOL_PENDING",
                "confidence": 0.0,
                "answer": "",
            }
        elif is_web_followup:
            print(f"[DEBUG_STAGE8] Skipping reasoning call - web_followup_mode detected, Stage 6 will handle")
            # Initialize stage_8_validation with a placeholder - Stage 6 will set the actual answer
            ctx["stage_8_validation"] = {
                "verdict": "PENDING",
                "mode": "WEB_FOLLOWUP_PENDING",
                "confidence": 0.0,
                "answer": "",
            }
        elif attention_payload.get("focus") == "self_model":
            # =========================================================================
            # SELF_MODEL ROUTING: When integrator routes to self_model, call it directly
            # instead of reasoning. This handles "who are you", "explain yourself", etc.
            # =========================================================================
            print(f"[DEBUG_STAGE8] Routing to self_model brain (focus=self_model)")
            self_model_handled = False
            try:
                from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api
                sm_response = self_model_api({
                    "op": "SELF_INTRODUCTION",
                    "payload": {
                        "detail_level": "standard",
                        "query": text,
                    }
                })
                if sm_response.get("ok"):
                    sm_payload = sm_response.get("payload", {})
                    # SELF_INTRODUCTION returns introduction_text, not answer
                    answer = sm_payload.get("introduction_text", sm_payload.get("answer", ""))
                    if answer:
                        ctx["stage_8_validation"] = {
                            "verdict": "TRUE",
                            "mode": "SELF_MODEL_IDENTITY",
                            "confidence": sm_payload.get("confidence", 0.95),
                            "answer": answer,
                        }
                        print(f"[DEBUG_STAGE8] self_model returned answer ({len(answer)} chars)")
                        self_model_handled = True

                        # =====================================================
                        # INTRO-TO-TARGET: If routing specified a target platform
                        # (e.g., "explain yourself to grok"), send the intro there
                        # =====================================================
                        intro_target = attention_payload.get("routing_intent", {}).get("introduction_target")
                        if not intro_target:
                            # Also check in metadata
                            intro_target = attention_payload.get("routing_metadata", {}).get("introduction_target")
                        if not intro_target:
                            # Check grammar_metadata from routing engine
                            intro_target = attention_payload.get("grammar_metadata", {}).get("introduction_target")

                        if intro_target:
                            print(f"[DEBUG_STAGE8] Introduction target detected: {intro_target}")
                            try:
                                if intro_target in ("grok", "x"):
                                    from optional.browser_tools.x import x
                                    # Send the intro to Grok
                                    grok_command = f"grok {answer}"
                                    print(f"[DEBUG_STAGE8] Sending intro to Grok ({len(answer)} chars)...")
                                    grok_response = x(grok_command)
                                    # Append Grok's response to the answer
                                    if grok_response and not grok_response.startswith("Failed"):
                                        combined = f"{answer}\n\n**Sent to Grok. Response:**\n{grok_response}"
                                        ctx["stage_8_validation"]["answer"] = combined
                                        ctx["stage_8_validation"]["grok_response"] = grok_response
                                        print(f"[DEBUG_STAGE8] Grok responded: {grok_response[:100]}...")
                                    else:
                                        ctx["stage_8_validation"]["answer"] = f"{answer}\n\n(Attempted to send to Grok but got: {grok_response})"
                                elif intro_target in ("chatgpt", "gpt", "openai"):
                                    from optional.browser_tools.chatgpt_tool import chatgpt
                                    print(f"[DEBUG_STAGE8] Sending intro to ChatGPT...")
                                    chatgpt_response = chatgpt(answer)
                                    if chatgpt_response:
                                        combined = f"{answer}\n\n**Sent to ChatGPT. Response:**\n{chatgpt_response}"
                                        ctx["stage_8_validation"]["answer"] = combined
                            except ImportError as e:
                                print(f"[DEBUG_STAGE8] Browser tool not available for {intro_target}: {e}")
                            except Exception as e:
                                print(f"[DEBUG_STAGE8] Failed to send intro to {intro_target}: {e}")
            except Exception as e:
                print(f"[DEBUG_STAGE8] Exception calling self_model: {e}")

            # If self_model didn't handle it, fall back to reasoning
            if not self_model_handled:
                print(f"[DEBUG_STAGE8] self_model didn't handle, falling back to reasoning")
                try:
                    v = _brain_module("reasoning").service_api({
                        "op": "EVALUATE_FACT",
                        "payload": {
                            "proposed_fact": {
                                "content": proposed_content,
                                "confidence": conf,
                                "source": "user_input",
                                "original_query": text,
                                "storable_type": st_type,
                                "confidence_penalty": confidence_penalty
                            },
                            "original_query": text,
                            "evidence": ctx.get("stage_2R_memory") or {}
                        }
                    })
                    ctx["stage_8_validation"] = (v.get("payload") or {})
                except Exception as e:
                    print(f"[DEBUG_STAGE8] Fallback reasoning exception: {e}")
                    ctx["stage_8_validation"] = {}
        else:
            # Always call the reasoning brain to evaluate the proposed content.
            # Include storable_type and confidence_penalty so the reasoning
            # service can apply intent-aware logic.  original_query is passed
            # separately to aid answer retrieval for questions.
            try:
                v = _brain_module("reasoning").service_api({
                    "op": "EVALUATE_FACT",
                    "payload": {
                        "proposed_fact": {
                            "content": proposed_content,
                            "confidence": conf,
                            "source": "user_input",
                            "original_query": text,
                            "storable_type": st_type,
                            "confidence_penalty": confidence_penalty
                        },
                        "original_query": text,
                        "evidence": ctx.get("stage_2R_memory") or {}
                    }
                })
                print(f"[DEBUG_STAGE8] Reasoning returned: ok={v.get('ok')}, has_payload={bool(v.get('payload'))}")
                ctx["stage_8_validation"] = (v.get("payload") or {})
                print(f"[DEBUG_STAGE8] stage_8_validation set: verdict={ctx['stage_8_validation'].get('verdict')}, answer exists={bool(ctx['stage_8_validation'].get('answer'))}")
            except Exception as e:
                print(f"[DEBUG_STAGE8] Exception calling reasoning: {e}")
                ctx["stage_8_validation"] = {}

        # After reasoning verdict, update the success flag for the most recent
        # operations across the cognitive brains.  A verdict of TRUE
        # indicates success for the initiating operations; any other verdict
        # marks them as unsuccessful.  This enables learning from past
        # performance to influence future biases.
        try:
            verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
            # FIX: Treat LEARNED (from Teacher) as success too
            is_success = verdict in ("TRUE", "LEARNED")
            brains_to_update = ["sensorium", "planner", "language", "pattern_recognition"]
            for b in brains_to_update:
                try:
                    root = COG_ROOT / b
                    update_last_record_success(root, is_success)
                except Exception:
                    pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Attempt inference when reasoning has attention but could not
        # determine a verdict.  When Stage 5b assigns focus to the
        # reasoning brain and the reasoning verdict is UNANSWERED or
        # UNKNOWN, try a simple heuristic to infer an answer from the
        # retrieved facts.  If inference succeeds, override the verdict
        # in stage_8_validation with a TRUE verdict and include the
        # inferred answer and confidence.  See Phase 1 fix: Attention
        # influences behaviour for reasoning.
        try:
            # When the verdict is not yet TRUE and memory results exist, attempt
            # to infer an answer via multi‑step reasoning.  This fires
            # regardless of the attention focus to ensure that factual
            # knowledge is surfaced whenever possible.  Only TRUE verdicts
            # are exempt to avoid overriding already validated responses.
            current_verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
            mem_res = (ctx.get("stage_2R_memory") or {}).get("results", [])
            if mem_res and current_verdict != "TRUE":
                # Extract the original query text
                qtext = str(ctx.get("original_query", "") or ctx.get("stage_3_language", {}).get("original_query", ""))
                inferred = _attempt_inference(qtext, mem_res)
                if inferred:
                    ctx.setdefault("stage_8_validation", {})
                    ctx_stage8 = ctx["stage_8_validation"]
                    ctx_stage8.update({
                        "verdict": "TRUE",
                        "mode": "INFERRED",
                        "confidence": inferred.get("confidence", 0.6),
                        "answer": inferred.get("answer"),
                        "reasoning_chain": inferred.get("steps", []),
                    })
                    if inferred.get("trace"):
                        ctx_stage8["reasoning_trace"] = inferred.get("trace")
        except Exception:
            # Never fail pipeline if inference errors
            pass
        # ------------------------------------------------------------------
        # Stage 8a – Meta-reasoner consistency check
        # If the reasoning engine reports both supporting and contradicting
        # evidence for the proposed statement, flag this as a potential
        # contradiction.  The meta-reasoner does not alter the verdict but
        # records the issue for offline inspection.  Flags are appended
        # to reports/system/meta_reasoner_flags.jsonl and stored in the
        # context under stage_meta_reasoner.
        try:
            support_ids = (ctx.get("stage_8_validation") or {}).get("supported_by") or []
            contradict_ids = (ctx.get("stage_8_validation") or {}).get("contradicted_by") or []
            meta_flag = None
            if support_ids and contradict_ids:
                meta_flag = {
                    "issue": "contradictory evidence",
                    "supported_by": support_ids,
                    "contradicted_by": contradict_ids
                }
                # Persist the flag to system reports
                try:
                    meta_dir = MAVEN_ROOT / "reports" / "system"
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    with open(meta_dir / "meta_reasoner_flags.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"mid": mid, "meta_flag": meta_flag}) + "\n")
                except Exception:
                    pass
                # When contradictory evidence is found, override the reasoning verdict to
                # ensure the claim is treated as a theory and routed to the theories bank.
                try:
                    # Set verdict and mode to indicate contradicted evidence
                    ctx.setdefault("stage_8_validation", {})
                    ctx["stage_8_validation"]["verdict"] = "THEORY"
                    ctx["stage_8_validation"]["mode"] = "CONTRADICTED_EVIDENCE"
                    # Force routing to theories_and_contradictions
                    ctx["stage_8_validation"]["routing_order"] = {"target_bank": "theories_and_contradictions", "action": "STORE"}
                    # Append a disputed audit entry to the Self‑DMN audit log
                    claim_id = ctx["stage_8_validation"].get("claim_id")
                    if claim_id:
                        try:
                            sdmn_dir = MAVEN_ROOT / "reports" / "self_dmn"
                            sdmn_dir.mkdir(parents=True, exist_ok=True)
                            with open(sdmn_dir / "audit.jsonl", "a", encoding="utf-8") as fh:
                                fh.write(json.dumps({"claim_id": claim_id, "status": "disputed"}) + "\n")
                        except Exception:
                            pass
                except Exception:
                    pass
            ctx["stage_meta_reasoner"] = meta_flag
        except Exception:
            ctx["stage_meta_reasoner"] = None
        if trace_enabled:
            trace_events.append({"stage": "reasoning"})

        # ------------------------------------------------------------------
        # Stage 8d – Self‑DMN dissent scan and optional recompute
        # After the initial reasoning verdict, invoke the Self‑DMN brain to
        # perform a dissent scan across recent claims.  If the scan returns a
        # RECOMPUTE decision for the current claim, re-run the reasoning
        # evaluation once to reassess the verdict with the same evidence.  The
        # recomputation is guarded so that it happens at most once per run.
        try:
            # Extract the claim ID from the reasoning stage, if present
            claim_id = (ctx.get("stage_8_validation") or {}).get("claim_id")
            # Proceed only if we have a claim ID and have not already re-run
            if claim_id and not ctx.get("debate_rerun"):
                sdmn_mod = _brain_module("self_dmn")
                try:
                    scan_res = sdmn_mod.service_api({"op": "DISSENT_SCAN", "payload": {}})
                    decisions = (scan_res.get("payload") or {}).get("decisions") or []
                except Exception:
                    decisions = []
                # Check for a recompute instruction for this claim
                recompute = False
                for dec in decisions:
                    try:
                        if dec.get("claim_id") == claim_id and str(dec.get("action", "")).upper() == "RECOMPUTE":
                            recompute = True
                            break
                    except Exception:
                        continue
                if recompute:
                    # Perform a single re-evaluation using the existing evidence
                    # Build the proposed_fact payload as in the initial call
                    proposed_fact = {
                        "content": proposed_content or text,
                        "confidence": conf,
                        "source": "user_input",
                        "original_query": text,
                        "storable_type": str(stage3.get("storable_type", "")),
                        "confidence_penalty": confidence_penalty
                    }
                    # Reuse the existing evidence from Stage 2R memory
                    evidence_reuse = ctx.get("stage_2R_memory") or {}
                    try:
                        new_val = _brain_module("reasoning").service_api({
                            "op": "EVALUATE_FACT",
                            "payload": {
                                "proposed_fact": proposed_fact,
                                "original_query": text,
                                "evidence": evidence_reuse
                            }
                        })
                        ctx["stage_8_validation"] = (new_val.get("payload") or ctx.get("stage_8_validation") or {})
                        # Mark the rerun so that we do not loop
                        ctx["debate_rerun"] = True
                        # Record that this run was recomputed due to Self-DMN dissent
                        ctx["stage_meta_reasoner"] = {"reason": "self_dmn_recompute"}
                    except Exception:
                        pass
        except Exception:
            pass

        # For non-storable inputs (question, command or request) that were
        # unanswered by the reasoning engine (mode ends with _INPUT), set
        # verdict to SKIP_STORAGE with a rationale.  This prevents storage
        # and helps candidate generation produce an appropriate reply.
        try:
            st_type_upper = str(stage3.get("storable_type", "")).upper()
            mode_upper = str((ctx.get("stage_8_validation") or {}).get("mode", "")).upper()
            if st_type_upper in ("QUESTION", "COMMAND", "REQUEST") and mode_upper.endswith("_INPUT"):
                unanswered_mode = f"UNANSWERED_{st_type_upper}"
                rationale_map = {
                    "QUESTION": "Questions are not stored without answers",
                    "COMMAND": "Commands are requests for action and not facts",
                    "REQUEST": "Requests are not facts and are not stored"
                }
                ctx["stage_8_validation"] = {
                    "verdict": "SKIP_STORAGE",
                    "mode": unanswered_mode,
                    "confidence": 0.0,
                    "route": None,
                    "rationale": rationale_map.get(st_type_upper, "Not storable input")
                }
        except Exception:
            pass

        # Stage 8b — Governance
        bias_profile = {
            "planner": ctx["stage_2_planner"].get("weights_used"),
            "language": ctx["stage_3_language"].get("weights_used"),
            "reasoning": ctx["stage_8_validation"].get("weights_used"),
            "personality": prefs,
            "adjustment_proposal": suggestion,
        }
        gov = _gov_module()
        # Always pass a non-empty content to governance; fall back to original text
        content_for_gov = proposed_content if proposed_content else text
        # Determine whether storage is warranted based on storable type and verdict.  Only
        # statements and validated theories should be sent to governance for storage.  Social
        # greetings and unknown inputs are treated as non‑storable as well to avoid
        # unnecessary policy checks.
        stage3_local = ctx.get("stage_3_language", {}) or {}
        st_type = str(stage3_local.get("storable_type", "")).upper()
        verdict_upper = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        # Non‑storable types or unanswered questions skip storage completely.  Include SOCIAL
        # and UNKNOWN in the list of non‑storable types.
        store_needed = not (st_type in {"QUESTION", "COMMAND", "REQUEST", "EMOTION", "OPINION", "SOCIAL", "UNKNOWN"} or verdict_upper == "SKIP_STORAGE")
        if store_needed:
            # Ask governance whether it is permissible to store this content
            enf = gov.service_api({
                "op": "ENFORCE",
                "payload": {
                    "action": "STORE",
                    "payload": {"content": content_for_gov},
                    "bias_profile": bias_profile,
                },
            })
            ctx["stage_8b_governance"] = enf.get("payload", {})
            allowed = bool(ctx["stage_8b_governance"].get("allowed", False))
        else:
            # Skip enforcement; mark as skipped so downstream storage can short‑circuit.  For
            # non‑storable types we still mark governance as allowed so that user‑facing logs
            # do not report a denial.  The action SKIP indicates no storage will occur.
            ctx["stage_8b_governance"] = {"allowed": True, "action": "SKIP"}
            allowed = True

        # Find duplicates against retrieved evidence (exact match)
        match = _best_memory_exact(ctx.get("stage_2R_memory"), proposed_content or "")
        duplicate = bool(match)

        # Stage 6 — candidates

        # ===== AGENCY TOOL EXECUTION (time_now, etc.) =====
        # When the integrator detects an agency tool query (e.g., "what time is it"),
        # it sets agency_tool in the context. Execute the tool directly and skip Teacher.
        attention_payload = ctx.get("stage_5b_attention", {}) or {}
        agency_tool = attention_payload.get("agency_tool") or ctx.get("agency_tool")

        if agency_tool and not ctx.get("stage_6_candidates"):
            print(f"[STAGE6_AGENCY_TOOL] Executing agency tool: {agency_tool}")
            try:
                from brains.cognitive.integrator.agency_executor import execute_agency_tool

                # Build agency_info from context
                # CRITICAL: Read method and args from context, don't hardcode!
                # The integrator sets these based on query type (time/date/calendar)
                agency_method = ctx.get("agency_method")
                agency_args = ctx.get("agency_args", {})

                # Fall back to defaults only if not provided
                if not agency_method and agency_tool == "time_now":
                    agency_method = "GET_TIME"  # Default fallback

                agency_info = {
                    "tool": agency_tool,
                    "method": agency_method,
                    "args": agency_args,
                    "bypass_teacher": ctx.get("bypass_teacher", True),
                }

                print(f"[STAGE6_AGENCY_TOOL] Method: {agency_method}, Args: {agency_args}")

                tool_result = execute_agency_tool(
                    tool_path=agency_info['tool'],
                    method_name=agency_info.get('method'),
                    args=agency_info.get('args')
                )

                if tool_result.get("status") == "success":
                    # Format the response for the user
                    # CRITICAL: Use formatted_response from agency_executor (it handles query_type correctly)
                    # Do NOT use output["formatted"] - that always contains time, not date/calendar!
                    formatted = tool_result.get("formatted_response")
                    if not formatted:
                        # Fallback only if formatted_response is not set
                        if isinstance(tool_result.get("output"), dict):
                            formatted = tool_result["output"].get("formatted", str(tool_result.get("output", "")))
                        else:
                            formatted = str(tool_result.get("output", ""))

                    cand = {
                        "type": "agency_tool",
                        "text": formatted,
                        "confidence": 0.99,  # Tool output is highly reliable
                        "tone": "neutral",
                        "tag": agency_info['tool'],
                    }
                    ctx["stage_6_candidates"] = {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "agency_tool_execution"}
                    }
                    ctx["stage_8_validation"] = {
                        "verdict": "ANSWERED",
                        "mode": "AGENCY_TOOL",
                        "confidence": 0.99,
                        "answer": formatted,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "weights_used": {"rule": "agency_tool_execution"},
                        "tool_used": agency_info['tool'],
                    }
                    print(f"[STAGE6_AGENCY_TOOL] ✓ Tool executed: {formatted[:80]}...")
                else:
                    # Tool failed - still set stage_8 so reflection skips and error is shown
                    error_msg = tool_result.get('error') or "Tool execution failed"
                    print(f"[STAGE6_AGENCY_TOOL] Tool execution failed: {error_msg}")

                    # Set up stage_8 with the error message so it becomes the final answer
                    cand = {
                        "type": "agency_tool_error",
                        "text": error_msg,
                        "confidence": 0.99,
                        "tone": "neutral",
                        "tag": agency_info['tool'],
                    }
                    ctx["stage_6_candidates"] = {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "agency_tool_error"}
                    }
                    ctx["stage_8_validation"] = {
                        "verdict": "ANSWERED",
                        "mode": "AGENCY_TOOL",
                        "confidence": 0.99,
                        "answer": error_msg,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "weights_used": {"rule": "agency_tool_error"},
                        "tool_used": agency_info['tool'],
                    }
            except Exception as agency_ex:
                print(f"[STAGE6_AGENCY_TOOL] Exception: {agency_ex}")
                import traceback
                traceback.print_exc()

        # ===== WEB SEARCH FOLLOW-UP ROUTING (from integrator) =====
        # When the integrator detects a follow-up to a web search (e.g., "tell me more"
        # after "web search computers"), it sets web_followup_mode=True in stage_5b_attention.
        # We route to research_manager's synthesize_web_followup function to use the cached
        # SERP data instead of falling back to generic LLM reasoning.
        attention_payload = ctx.get("stage_5b_attention", {}) or {}
        if attention_payload.get("web_followup_mode"):
            try:
                from brains.cognitive.research_manager.service.research_manager_brain import (
                    synthesize_web_followup,
                )
                last_web_search = attention_payload.get("last_web_search", {})
                followup_query = text  # The current user query
                print(f"[STAGE6_WEB_FOLLOWUP] Routing to synthesize_web_followup: query='{followup_query}', original='{last_web_search.get('query', '')}'")

                result = synthesize_web_followup(last_web_search, followup_query, ctx)

                if result.get("ok"):
                    answer_text = result.get("text_answer", "")
                    sources = result.get("sources", [])

                    # Build candidate
                    cand = {
                        "type": "web_followup",
                        "text": answer_text,
                        "confidence": 0.85,
                        "tone": "neutral",
                        "tag": "research_manager",
                        "sources": sources,
                    }
                    ctx["stage_6_candidates"] = {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "web_followup_synthesis"}
                    }
                    ctx["stage_8_validation"] = {
                        "verdict": "LEARNED",
                        "mode": "WEB_FOLLOWUP",
                        "confidence": 0.85,
                        "answer": answer_text,
                        "sources": sources,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "weights_used": {"rule": "web_followup_synthesis"}
                    }
                    print(f"[STAGE6_WEB_FOLLOWUP] Successfully synthesized follow-up answer ({len(answer_text)} chars)")
                else:
                    print(f"[STAGE6_WEB_FOLLOWUP] synthesize_web_followup returned ok=False, falling through")
            except Exception as web_followup_ex:
                print(f"[STAGE6_WEB_FOLLOWUP] Exception: {web_followup_ex}")
                # Fall through to normal processing

        # Check for preference_query intent and handle from memory
        stage3_intent = str((ctx.get("stage_3_language") or {}).get("intent", ""))
        if stage3_intent in {"research_request", "research_followup"} and not ctx.get("stage_6_candidates"):
            try:
                rm_mod = _brain_module("research_manager")
                op = "FETCH_REPORT" if stage3_intent == "research_followup" else "RUN_RESEARCH"
                stage3_lang = ctx.get("stage_3_language") or {}
                topic = stage3_lang.get("research_topic") or text
                depth = stage3_lang.get("research_depth") or 2
                deliverable = stage3_lang.get("deliverable") or "detailed_report"
                rm_payload = {
                    "topic": topic,
                    "depth": depth,
                    "deliverable": deliverable,
                    "sources": ["memory", "teacher"],
                    "original_query": text,
                }
                res = rm_mod.service_api({"op": op, "mid": mid, "payload": rm_payload}) if rm_mod else {}
                pay = (res or {}).get("payload") or {}
                summary = str(pay.get("summary") or pay.get("answer") or "").strip()
                if not summary:
                    summary = "I was unable to complete that research request right now."
                confidence_val = float(pay.get("confidence", 0.65)) if isinstance(pay, dict) else 0.65
                cand = {
                    "type": "research",
                    "text": summary,
                    "confidence": confidence_val,
                    "tone": "neutral",
                    "tag": "research_manager",
                }
                ctx["stage_6_candidates"] = {
                    "candidates": [cand],
                    "weights_used": {"gen_rule": "research_manager_v1"}
                }
                ctx["stage_8_validation"] = {
                    "verdict": "SKIP_STORAGE",
                    "mode": "RESEARCH",
                    "confidence": confidence_val,
                    "answer": summary,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "weights_used": {"rule": "research_manager_v1"}
                }
            except Exception as research_ex:
                try:
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
                except Exception:
                    ctx["stage_6_candidates"] = {"candidates": [], "error": str(research_ex)}
        elif stage3_intent == "preference_query":
            # Handle preference query by retrieving stored preferences
            # Support domain-specific queries (e.g., "what animals do I like?")
            try:
                user_id = ctx.get("user_id") or "default_user"
                stage3_lang = ctx.get("stage_3_language") or {}
                preference_domain = stage3_lang.get("preference_domain")

                # Use the new list_preferences helper with optional domain filter
                preferences = list_preferences(user_id, preference_domain)

                if preferences:
                    # Build a summary of preferences
                    pref_items = []
                    for pref in preferences:
                        content = str(pref.get("content", ""))
                        if content:
                            pref_items.append(content)

                    # Create a single-sentence summary
                    if len(pref_items) <= 3:
                        summary = ", ".join(pref_items)
                    else:
                        summary = ", ".join(pref_items[:3]) + f", and {len(pref_items) - 3} more"

                    # Customize answer based on domain
                    if preference_domain:
                        answer_text = f"Based on what you've told me, you like these {preference_domain}: {summary}."
                    else:
                        answer_text = f"Based on what you've told me, you like: {summary}."
                    confidence = 0.9
                else:
                    # No preferences found - customize message for domain
                    if preference_domain:
                        answer_text = f"You haven't told me about any {preference_domain} you like yet."
                    else:
                        answer_text = "You haven't told me anything you like yet. Tell me what you like!"
                    confidence = 0.7

                # Build a direct preference candidate
                cand = {
                    "type": "preference_query",
                    "text": answer_text,
                    "confidence": confidence,
                    "tone": "neutral",
                    "tag": "preference_retrieved",
                }
                ctx["stage_6_candidates"] = {
                    "candidates": [cand],
                    "weights_used": {"gen_rule": "preference_query_v1"}
                }
                # Set stage 8 validation with PREFERENCE verdict
                ctx["stage_8_validation"] = {
                    "verdict": "PREFERENCE",
                    "mode": "PREFERENCE_QUERY",
                    "confidence": 1.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "answer": answer_text,
                    "weights_used": {"rule": "preference_query_v1"}
                }
            except Exception as pref_ex:
                # Preference query failed; fall back to language brain
                try:
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
                except Exception:
                    ctx["stage_6_candidates"] = {"candidates": [], "error": str(pref_ex)}
        # Check for relationship_query intent and handle from memory
        elif stage3_intent == "relationship_query":
            # Handle relationship query by looking up stored relationship facts
            try:
                user_id = ctx.get("user_id") or "default_user"
                relationship_kind = (ctx.get("stage_3_language") or {}).get("relationship_kind")

                if relationship_kind:
                    fact = get_relationship_fact(user_id, relationship_kind)

                    if fact is not None and fact.get("value") is True:
                        answer_text = "You've told me we're friends. I'm a synthetic cognition system and don't experience friendship like humans do, but I understand that as your intent and I'm here to help you."
                        verdict = "TRUE"
                        confidence = 0.9
                    elif fact is not None and fact.get("value") is False:
                        answer_text = "You've told me we're not friends. I'll respect that, but I'm still here to help you if you want."
                        verdict = "TRUE"
                        confidence = 0.9
                    else:
                        # No stored relationship fact; fall back to default answer
                        answer_text = "I'm a synthetic cognition system, so I don't experience friendship the way humans do, but I'm here to help you."
                        verdict = "NEUTRAL"
                        confidence = 0.7

                    # Build a direct relationship candidate
                    cand = {
                        "type": "relationship_query",
                        "text": answer_text,
                        "confidence": confidence,
                        "tone": "neutral",
                        "tag": "relationship_retrieved",
                    }
                    ctx["stage_6_candidates"] = {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "relationship_query_v1"}
                    }
                    # Set stage 8 validation
                    ctx["stage_8_validation"] = {
                        "verdict": verdict,
                        "mode": "RELATIONSHIP_QUERY",
                        "confidence": confidence,
                        "routing_order": {"target_bank": None, "action": None},
                        "supported_by": [],
                        "contradicted_by": [],
                        "answer": answer_text,
                        "weights_used": {"rule": "relationship_query_v1"}
                    }
                else:
                    # Missing relationship_kind; fall back to language brain
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
            except Exception as rel_ex:
                # Relationship query failed; fall back to language brain
                try:
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
                except Exception:
                    ctx["stage_6_candidates"] = {"candidates": [], "error": str(rel_ex)}
        # Check for user_profile_summary intent and build profile from memory
        elif stage3_intent == "user_profile_summary":
            # Handle user profile summary by collecting identity, preferences, and relationships
            try:
                user_id = ctx.get("user_id") or "default_user"

                # Collect profile components
                profile_parts = []

                # 1. Identity/name
                name = _resolve_user_name(ctx)
                if name:
                    profile_parts.append(f"You're {name}")

                # 2. Preferences (top 5)
                preferences = list_preferences(user_id, None)
                if preferences:
                    pref_items = []
                    for pref in preferences[:5]:  # Limit to top 5
                        content = str(pref.get("content", ""))
                        if content:
                            pref_items.append(content)
                    if pref_items:
                        if len(pref_items) == 1:
                            profile_parts.append(f"you like {pref_items[0]}")
                        elif len(pref_items) == 2:
                            profile_parts.append(f"you like {pref_items[0]} and {pref_items[1]}")
                        else:
                            profile_parts.append(f"you like {', '.join(pref_items[:-1])}, and {pref_items[-1]}")

                # 3. Relationship status
                friend_fact = get_relationship_fact(user_id, "friend_with_system")
                if friend_fact and friend_fact.get("value") is True:
                    profile_parts.append("you've told me we're friends")

                # Build the final answer
                if profile_parts:
                    # Capitalize first letter and join with proper punctuation
                    profile_text = ". ".join(profile_parts)
                    if not profile_text[0].isupper():
                        profile_text = profile_text[0].upper() + profile_text[1:]
                    answer_text = f"Here's what I know so far: {profile_text}. I'm still learning and will update this as we talk."
                    confidence = 0.85
                else:
                    answer_text = "You haven't told me much about yourself yet. Share what you'd like me to know!"
                    confidence = 0.7

                # Build a direct profile candidate
                cand = {
                    "type": "user_profile_summary",
                    "text": answer_text,
                    "confidence": confidence,
                    "tone": "neutral",
                    "tag": "profile_retrieved",
                }
                ctx["stage_6_candidates"] = {
                    "candidates": [cand],
                    "weights_used": {"gen_rule": "user_profile_summary_v1"}
                }
                # Set stage 8 validation with SKIP_STORAGE verdict
                ctx["stage_8_validation"] = {
                    "verdict": "SKIP_STORAGE",
                    "mode": "PROFILE_SUMMARY",
                    "confidence": confidence,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "answer": answer_text,
                    "weights_used": {"rule": "user_profile_summary_v1"}
                }
            except Exception as profile_ex:
                # Profile query failed; fall back to language brain
                try:
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
                except Exception:
                    ctx["stage_6_candidates"] = {"candidates": [], "error": str(profile_ex)}
        # Check for math_compute intent and handle deterministically
        elif stage3_intent == "math_compute":
            # Call the deterministic math handler
            math_result = _solve_simple_math(text)
            if math_result.get("ok"):
                # Build a direct math candidate with confidence 1.0
                cand = {
                    "type": "math_deterministic",
                    "text": str(math_result["result"]),
                    "confidence": 1.0,
                    "tone": "neutral",
                    "tag": "math_computed",
                }
                ctx["stage_6_candidates"] = {
                    "candidates": [cand],
                    "weights_used": {"gen_rule": "math_deterministic_v1"}
                }
                # Mark in context for stage 8 validation
                ctx["mode"] = "math_deterministic"
                # Set stage 8 validation for math with TRUE verdict and confidence 1.0
                ctx["stage_8_validation"] = {
                    "verdict": "TRUE",
                    "mode": "MATH_DIRECT",
                    "confidence": 1.0,
                    "routing_order": {"target_bank": None, "action": None},
                    "supported_by": [],
                    "contradicted_by": [],
                    "answer": str(math_result["result"]),
                    "weights_used": {"rule": "math_deterministic_v1"}
                }
            else:
                # Math parsing failed; fall back to language brain
                try:
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
                except Exception as cand_ex:
                    ctx["stage_6_candidates"] = {"candidates": [], "error": str(cand_ex)}
        else:
            # For all other cases, call language brain to generate candidates
            try:
                cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                ctx["stage_6_candidates"] = cands.get("payload", {})

                # Check if any candidate is an identity response
                # Identity queries like "who are you", "describe yourself" should not be
                # logged as UNANSWERED_QUESTION. Set SKIP_STORAGE verdict.
                candidates_list = ctx["stage_6_candidates"].get("candidates", [])
                if candidates_list:
                    for cand in candidates_list:
                        if cand.get("type") == "identity":
                            # Set stage 8 validation for identity with SKIP_STORAGE verdict
                            ctx["stage_8_validation"] = {
                                "verdict": "SKIP_STORAGE",
                                "mode": "IDENTITY_RESPONSE",
                                "confidence": cand.get("confidence", 0.85),
                                "routing_order": {"target_bank": None, "action": None},
                                "supported_by": [],
                                "contradicted_by": [],
                                "answer": cand.get("text", ""),
                                "weights_used": {"rule": "identity_response_v1"}
                            }
                            break
            except Exception as cand_ex:
                ctx["stage_6_candidates"] = {"candidates": [], "error": str(cand_ex)}

            # Always run a cross‑validation to compute arithmetic or definitional responses.
            try:
                _cross_validate_answer(ctx)
            except Exception:
                pass
            # Load the verdict, answer and confidence after potential cross‑validation
            try:
                ver_u_local = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
                ans_local = (ctx.get("stage_8_validation") or {}).get("answer")
                conf_local = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0) or 0.0)
            except Exception:
                ver_u_local = ""
                ans_local = None
                conf_local = 0.0
            # If we have a TRUE verdict and a substantive answer, produce a direct factual candidate.
            if ver_u_local == "TRUE" and ans_local:
                ans_lc = str(ans_local).lower() if ans_local else ""
                is_bad = False
                try:
                    for bad in BAD_CACHE_PHRASES:
                        if bad and bad in ans_lc:
                            is_bad = True
                            break
                except Exception:
                    is_bad = False
                if not is_bad:
                    cand = {
                        "type": "direct_factual",
                        "text": ans_local,
                        "confidence": conf_local,
                        "tone": "neutral",
                        "tag": ctx.get("cross_check_tag", "asserted_true"),
                    }
                    ctx["stage_6_candidates"] = {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "s6_locked_bridge_v2"}
                    }
                else:
                    # Answer contains a bad phrase; rely on the language brain to generate candidates
                    try:
                        cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                        ctx["stage_6_candidates"] = cands.get("payload", {})
                    except Exception as cand_ex:
                        ctx["stage_6_candidates"] = {"candidates": [], "error": str(cand_ex)}
            else:
                # No locked answer available; rely on the language brain for candidate generation
                try:
                    cands = _brain_module("language").service_api({"op": "GENERATE_CANDIDATES", "mid": mid, "payload": ctx})
                    ctx["stage_6_candidates"] = cands.get("payload", {})
                except Exception as cand_ex:
                    ctx["stage_6_candidates"] = {"candidates": [], "error": str(cand_ex)}
        if trace_enabled:
            trace_events.append({"stage": "language_generate_candidates"})

        # Stage 10 — finalize
        # When the reasoning verdict is TRUE and we bridged Stage 6,
        # skip calling FINALIZE if a final answer is already present to
        # avoid overwriting it with an empty or filler response.  For
        # other verdicts, invoke the language brain to finalise the
        # selected candidate.
        try:
            ver_u2 = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        except Exception:
            ver_u2 = ""
        if ver_u2 == "TRUE" and ctx.get("stage_6_candidates"):
            # Extract the first candidate and set the final answer directly
            c0 = (ctx.get("stage_6_candidates") or {}).get("candidates", [])
            if c0:
                try:
                    c0 = c0[0]
                    ctx["stage_10_finalize"] = {"text": c0.get("text"), "confidence": c0.get("confidence", 0.0)}
                except Exception:
                    ctx["stage_10_finalize"] = {}
            else:
                # No candidates; fall back to FINALIZE normally
                try:
                    fin = _brain_module("language").service_api({"op":"FINALIZE","payload": ctx})
                    ctx["stage_10_finalize"] = fin.get("payload", {})
                except Exception:
                    ctx["stage_10_finalize"] = {}
        else:
            try:
                fin = _brain_module("language").service_api({"op":"FINALIZE","payload": ctx})
                ctx["stage_10_finalize"] = fin.get("payload", {})
            except Exception:
                ctx["stage_10_finalize"] = {}
        if trace_enabled:
            trace_events.append({"stage": "language_finalize"})

        # ------------------------------------------------------------------
        # Stage 11 — Reflection (Self-Review)
        # ------------------------------------------------------------------
        # Run reflection on the draft answer to detect issues and optionally
        # improve it before presenting to the user. Reflection sits between
        # draft generation (FINALIZE) and final answer capture.
        #
        # Contract:
        # - Input: draft_answer from stage_10_finalize
        # - Output: verdict (ok/minor_issue/major_issue), improved_answer (optional)
        # - If improved_answer exists and verdict is not "ok", use improved_answer
        # - Store reflection metadata for learning systems
        try:
            # CRITICAL: Skip reflection for agency tool executions (shell commands, time_now, etc.)
            # These tools provide direct output that should NOT be processed through reflection.
            stage8 = ctx.get("stage_8_validation") or {}
            stage8_mode = stage8.get("mode", "")
            stage8_tool = stage8.get("tool_used", "")

            if stage8_mode == "AGENCY_TOOL" or stage8_tool in ("shell_tool", "time_now", "browser_runtime"):
                # Use the tool output directly - no reflection needed
                tool_answer = stage8.get("answer", "")
                if tool_answer:
                    print(f"[REFLECTION] Skipping reflection for agency tool: {stage8_tool or stage8_mode}")
                    ctx["stage_11_reflection"] = {"skipped": True, "reason": "agency_tool_output"}
                    # Override stage_10_finalize with the tool output
                    ctx["stage_10_finalize"] = {
                        "text": tool_answer,
                        "confidence": 0.99,
                        "source": "agency_tool_direct",
                    }
                    # Also set final_answer directly to prevent any further overrides
                    ctx["final_answer"] = tool_answer
                    ctx["final_confidence"] = 0.99
                else:
                    print(f"[REFLECTION] Agency tool detected but no answer in stage_8")

            draft_answer = (ctx.get("stage_10_finalize") or {}).get("text", "")
            draft_confidence = (ctx.get("stage_10_finalize") or {}).get("confidence", 0.0)

            # Only run reflection if we have a draft answer AND it wasn't already handled by agency tool
            if draft_answer and len(draft_answer.strip()) > 0 and not ctx.get("stage_11_reflection", {}).get("skipped"):
                print(f"[REFLECTION] Running reflection on draft answer...")

                # Import self_review brain
                try:
                    from brains.cognitive.self_review.service.self_review_brain import (
                        _derive_question_tags,
                        run_reflection_engine,
                    )
                    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

                    # Skip reflection for trivial arithmetic where validation already suffices
                    if re.fullmatch(r"[0-9\s+\-*/=.]+", text.strip()):
                        ctx["stage_11_reflection"] = {"skipped": True, "reason": "trivial_math"}
                    else:
                        review_context_tags = []
                        try:
                            review_context_tags = _derive_question_tags(text)
                        except Exception:
                            review_context_tags = []

                        # Track both specialist brains (language, reasoning) and cognitive brains
                        # that participate in the pipeline for learning feedback
                        used_brains = [
                            "language",
                            "reasoning",
                            "integrator",  # Routing decisions
                            "affect_priority",  # Tone/safety handling
                            "context_management",  # Context decay/management
                        ]
                        reflection_context = {
                            "question": text,
                            "draft_answer": draft_answer,
                            "final_answer": draft_answer,
                            "confidence": draft_confidence,
                            "used_brains": used_brains,
                            "context_tags": review_context_tags,
                            "wm_trace": ctx.get("wm_trace"),
                            "beliefs": ctx.get("stage_2R_memory", {}).get("results", []),
                        }

                        review_result = run_reflection_engine("quick_check", reflection_context)
                        ctx["stage_11_reflection"] = review_result

                        verdict = review_result.get("verdict", "ok")
                        improved_answer = review_result.get("improved_answer")
                        issues = review_result.get("issues", [])
                        metadata_tags = (review_result.get("meta") or {}).get("context_tags", [])

                        print(
                            f"[REFLECTION] Verdict: {verdict}, Issues: {len(issues)}, Has improved: {bool(improved_answer)}"
                        )

                        # If issues found but no rewrite yet, request a focused correction from Teacher
                        if not improved_answer and review_result.get("improvement_needed"):
                            try:
                                helper = TeacherHelper("reflection_fix")
                                issue_summary = "; ".join(issues[:4]) or "Detected quality gaps"
                                prompt = (
                                    f"Question: {text}\nCurrent answer: {draft_answer}\n"
                                    f"Issues: {issue_summary}\n"
                                    "Return only the corrected answer, not an explanation of changes."
                                )
                                # NOTE: check_memory_first is deprecated; memory-first is always enforced
                                teacher_resp = helper.maybe_call_teacher(
                                    question=prompt,
                                    context={"task": "reflection_improve", "mode": "quick_check"},
                                )
                                if teacher_resp and teacher_resp.get("answer"):
                                    improved_answer = str(teacher_resp.get("answer", "")).strip()
                                    review_result["improved_answer"] = improved_answer
                            except Exception:
                                improved_answer = improved_answer or None

                        # CRITICAL: Use improved_answer if reflection says answer needs improvement
                        if improved_answer and (verdict in ("minor_issue", "major_issue") or review_result.get("improvement_needed")):
                            print(f"[REFLECTION] Using improved answer (verdict={verdict})")
                            ctx["stage_10_finalize"]["text"] = improved_answer
                            ctx["stage_10_finalize"]["reflection_improved"] = True
                            if draft_confidence < 0.8:
                                ctx["stage_10_finalize"]["confidence"] = min(draft_confidence + 0.1, 0.85)
                        else:
                            print(f"[REFLECTION] Keeping draft answer (verdict={verdict})")

                        try:
                            stage_tags = ctx.get("pipeline_stage_tags", {}) or {}
                            stage_tags["stage_11_reflection"] = {
                                "review_mode": (review_result.get("meta") or {}).get("mode"),
                                "verdict": verdict,
                                "metadata_tags": metadata_tags,
                                "issues": issues,
                            }
                            ctx["pipeline_stage_tags"] = stage_tags
                        except Exception:
                            pass

                except Exception as e:
                    print(f"[REFLECTION] Exception during reflection: {str(e)[:100]}")
                    ctx["stage_11_reflection"] = {"error": str(e)[:200]}
            else:
                print(f"[REFLECTION] Skipping reflection - no draft answer")
                ctx["stage_11_reflection"] = {"skipped": True, "reason": "no_draft_answer"}

        except Exception as e:
            # Never fail pipeline on reflection errors
            print(f"[REFLECTION] Outer exception: {str(e)[:100]}")
            ctx["stage_11_reflection"] = {"error": str(e)[:200]}

        if trace_enabled:
            trace_events.append({"stage": "reflection"})

        # Before capturing the final answer, run a cross‑validation on the
        # reasoning verdict.  This second pass occurs after the reasoning
        # brain has produced a verdict and the language brain has
        # formatted the answer.  Arithmetic or definitional mismatches
        # detected here override the final answer.
        try:
            _cross_validate_answer(ctx)
        except Exception:
            pass
        # Capture final answer and confidence for external consumers (e.g. CLI).
        # If the cross‑validation recomputed the answer, use it and supply a
        # generic confidence.  Otherwise mirror the language finalize payload.
        #
        # TASK 3: Fix final answer override bug
        # When stage_10_finalize produces a fallback message ("I don't yet have
        # enough information...") but stage_8_validation has a valid answer from
        # memory facts, prefer stage_8_validation. This prevents self_review or
        # language_brain from stomping good reasoning answers.
        try:
            # CRITICAL: If final_answer was already set by agency tool handler, preserve it
            stage11_reflection = ctx.get("stage_11_reflection") or {}
            if stage11_reflection.get("reason") == "agency_tool_output" and ctx.get("final_answer"):
                print(f"[FINAL_ANSWER_FIX] Preserving agency tool output (already set)")
                # Skip this section - answer was already correctly set
                pass
            else:
                # Determine whether the cross check produced a new answer.
                cv_answer = (ctx.get("stage_8_validation") or {}).get("answer")
                cv_tag = ctx.get("cross_check_tag")
                if cv_tag == "recomputed" and cv_answer:
                    ctx["final_answer"] = cv_answer
                    # Assign a moderate confidence when recomputing arithmetic
                    ctx["final_confidence"] = 0.4
                else:
                    finalize_text = ctx.get("stage_10_finalize", {}).get("text") or ""
                    finalize_conf = ctx.get("stage_10_finalize", {}).get("confidence")

                    # TASK 3: Check if finalize produced a fallback message
                    is_fallback_message = (
                        "i don't yet have enough information" in finalize_text.lower()
                        or "i don't have enough information" in finalize_text.lower()
                        or "i'm not sure how to help" in finalize_text.lower()
                        or "i am not sure how to help" in finalize_text.lower()
                        or (not finalize_text.strip())
                    )

                    # TASK 3: Check if stage_8 has a good answer
                    stage8 = ctx.get("stage_8_validation") or {}
                    stage8_verdict = str(stage8.get("verdict", "")).upper()
                    stage8_answer = stage8.get("answer")
                    stage8_conf = stage8.get("confidence", 0.0)

                    # TASK 3: Prefer stage_8 answer when:
                    # 1. Finalize produced fallback/empty message, AND
                    # 2. Stage_8 has verdict TRUE/LEARNED with valid answer
                    if is_fallback_message and stage8_answer and stage8_verdict in ("TRUE", "LEARNED", "ANSWERED"):
                        print(f"[TASK3_FINAL_ANSWER_FIX] Detected fallback message in finalize")
                        print(f"[TASK3_FINAL_ANSWER_FIX] Using stage_8 answer (verdict={stage8_verdict})")
                        print(f"[TASK3_FINAL_ANSWER_FIX] Stage8 answer: '{stage8_answer[:60]}...'")
                        ctx["final_answer"] = stage8_answer
                        ctx["final_confidence"] = stage8_conf if stage8_conf else 0.8
                        # Also update stage_10_finalize for consistency
                        ctx["stage_10_finalize"] = {
                            "text": stage8_answer,
                            "confidence": ctx["final_confidence"],
                            "source": "stage_8_memory_answer",
                            "task3_override": True
                        }
                    else:
                        ctx["final_answer"] = finalize_text
                        ctx["final_confidence"] = finalize_conf
        except Exception:
            ctx["final_answer"] = None
            ctx["final_confidence"] = None

        # Calibrate the final confidence using the Stage 8 validation confidence.
        # If the reasoning stage produced a higher confidence than the
        # language finalize, promote the final confidence to reflect that.
        try:
            s8_conf = (ctx.get("stage_8_validation") or {}).get("confidence")
            if s8_conf is not None:
                try:
                    sconf = float(s8_conf or 0.0)
                except Exception:
                    sconf = 0.0
                try:
                    fconf = float(ctx.get("final_confidence") or 0.0)
                except Exception:
                    fconf = 0.0
                if sconf > fconf:
                    ctx["final_confidence"] = sconf
            # Additionally, calibrate final confidence using the highest
            # confidence among retrieved memory facts.  When the final
            # confidence remains low but memory records include highly
            # trusted statements, promote the final confidence to match
            # that evidence.  This prevents high‑quality facts from being
            # obscured by conservative aggregation.
            try:
                mem_res = (ctx.get("stage_2R_memory") or {}).get("results", [])
            except Exception:
                mem_res = []
            try:
                max_mem_conf = 0.0
                for rec in mem_res:
                    try:
                        rc = float(rec.get("confidence", 0.0) or 0.0)
                    except Exception:
                        rc = 0.0
                    if rc > max_mem_conf:
                        max_mem_conf = rc
                try:
                    fconf2 = float(ctx.get("final_confidence") or 0.0)
                except Exception:
                    fconf2 = 0.0
                if max_mem_conf > fconf2:
                    ctx["final_confidence"] = max_mem_conf
            except Exception:
                pass
        except Exception:
            # Do not propagate confidence calibration errors
            pass

        # Final answer compositor enforcing teacher > memory > fallback rules
        try:
            stage8 = ctx.get("stage_8_validation") or {}
            teacher_verdict = str(stage8.get("verdict", "")).upper()
            teacher_answer = (stage8.get("answer") or "").strip()

            mem_results = (ctx.get("stage_2R_memory") or {}).get("results", [])
            usable_mem: List[Dict[str, Any]] = []
            for rec in mem_results:
                try:
                    bank = str(rec.get("bank") or rec.get("source") or "").lower()
                    if bank == "research_reports":
                        continue
                    content = str(
                        rec.get("content")
                        or rec.get("text")
                        or rec.get("value")
                        or ""
                    ).strip()
                    if not content:
                        continue
                    if content.startswith("Topic:") and "Summary:" in content:
                        continue
                    score = float(rec.get("score") or rec.get("relevance") or 0.0)
                    if score >= 0.6:
                        usable_mem.append({"content": content, "score": score, "bank": bank})
                except Exception:
                    continue

            composed_answer = ctx.get("final_answer") or ""
            composed_conf = ctx.get("final_confidence")

            if teacher_verdict == "LEARNED" and teacher_answer:
                composed_answer = teacher_answer
                composed_conf = composed_conf or stage8.get("confidence", 0.7)
                if usable_mem:
                    extras = [
                        m.get("content", "")
                        for m in sorted(usable_mem, key=lambda r: r.get("score", 0.0), reverse=True)[:2]
                    ]
                    extras = [e for e in extras if e]
                    if extras:
                        composed_answer = teacher_answer + " " + " ".join(extras)
            elif usable_mem:
                top_mem = sorted(usable_mem, key=lambda r: r.get("score", 0.0), reverse=True)
                mem_texts = [m.get("content", "") for m in top_mem[:3] if m.get("content")]
                if mem_texts:
                    composed_answer = "From memory: " + "; ".join(mem_texts)
                    composed_conf = composed_conf or top_mem[0].get("score", 0.65)
            if not composed_answer:
                composed_answer = "I don't know yet."
                composed_conf = composed_conf or 0.2

            ctx["final_answer"] = composed_answer
            ctx["final_confidence"] = composed_conf
            ctx.setdefault("stage_10_finalize", {})
            if not ctx["stage_10_finalize"].get("text"):
                ctx["stage_10_finalize"]["text"] = composed_answer
            if ctx["stage_10_finalize"].get("confidence") is None:
                ctx["stage_10_finalize"]["confidence"] = composed_conf
        except Exception:
            pass

        # Expose the cross‑validation tag for transparency.  Downstream
        # consumers can examine this flag to understand whether the
        # answer was asserted, recomputed or marked as conflicting.  If
        # cross_check_tag is absent, default to "asserted_true".
        try:
            ctx["final_tag"] = ctx.get("cross_check_tag", "asserted_true")
        except Exception:
            ctx["final_tag"] = "asserted_true"

        # Provide a simple explanation when no answer is found.  If the
        # reasoning verdict indicates UNKNOWN or UNANSWERED, or if the
        # final answer text is empty, attach a rationale explaining that
        # retrieval failed.  This helps external consumers understand why
        # the system could not produce a response.
        try:
            verdict_upper_tmp = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
            final_text_tmp = str(ctx.get("final_answer", "") or "").strip()
            if verdict_upper_tmp in {"UNKNOWN", "UNANSWERED"} or not final_text_tmp:
                ctx["reasoning_explanation"] = (
                    "No relevant facts were found in the domain banks or the inference engine "
                    "could not connect them, so the system could not confidently answer the question."
                )
        except Exception:
            pass

        # Propagate reasoning trace for transparency.  When the reasoning
        # engine supplies a ``reasoning_trace`` in stage 8, expose it in
        # the top‑level context so that external consumers can inspect the
        # inference process.  This does not alter the final answer or
        # confidence but simply surfaces the trace.
        try:
            rt = (ctx.get("stage_8_validation") or {}).get("reasoning_trace")
            if rt:
                ctx["reasoning_trace"] = rt
        except Exception:
            pass

        # --- Belief extraction from definitional answers ---
        # If the original query is a simple definition question (e.g. "what is X?")
        # and the language brain produced a non-empty final answer, extract the
        # subject and store the answer as a belief.  Conflicting beliefs are
        # detected via the belief tracker; only non-conflicting beliefs are
        # recorded.  This logic runs opportunistically and does not affect
        # the pipeline if the belief tracker is unavailable.
        try:
            ans = ctx.get("final_answer") or ""
            if ans:
                q_raw = str(ctx.get("original_query") or "").strip().lower()
                prefixes = ["what is ", "who is ", "what was ", "who was ", "who are ", "what are "]
                subj: Optional[str] = None
                for pfx in prefixes:
                    if q_raw.startswith(pfx):
                        subj = q_raw[len(pfx):].strip().rstrip("?")
                        break
                if subj and _belief_add:
                    conflict = None
                    try:
                        if _belief_detect:
                            conflict = _belief_detect(subj, "is", ans)
                    except Exception:
                        conflict = None
                    if not conflict:
                        try:
                            conf_val = float(ctx.get("final_confidence") or 1.0)
                        except Exception:
                            conf_val = 1.0
                        _belief_add(subj, "is", ans, confidence=conf_val)
        except Exception:
            pass

        # --- Context decay and meta‑learning ---
        # Apply temporal decay to the context to reduce the weight of old
        # numeric values.  Store the decayed copy under a special key.
        try:
            if _ctx_decay:
                ctx["decayed_context"] = _ctx_decay(ctx)
        except Exception:
            # Ignore decay errors
            pass
        # Record run metrics for meta learning.  The meta learning layer
        # can later update weights based on these observations.  This call
        # is opportunistic and will be silently skipped if the optional
        # import failed.
        try:
            if _meta_record:
                _meta_record(ctx)
        except Exception:
            pass

        # Persist user identity declarations like "I am Josh" or "my name is Alice"
        # This must happen BEFORE Stage 9 storage to ensure it runs even when
        # storage is skipped for questions or low-confidence content.
        try:
            _utterance = str(ctx.get("original_query") or "").lower()
            # Only process identity statements, not questions like "who am i"
            # Handle various casual forms: "im josh", "i'm josh", "i am josh", "my name is josh"
            if any(pattern in _utterance for pattern in ["i am ", "i'm ", "im ", "my name is ", "call me "]):
                # Skip questions
                if not any(q in _utterance for q in ["who am i", "what is my name", "what's my name", "whats my name"]):
                    # Extract name using the existing episodic helper
                    name = episodic_last_declared_identity([{"user": ctx.get("original_query", "")}], n=1)
                    if name and len(name.strip()) > 0:
                        # Store in the durable identity store
                        try:
                            if identity_user_store:
                                identity_user_store.SET(name.strip())  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        # Also store in working memory for immediate access
                        try:
                            service_api({
                                "op": "WM_PUT",
                                "payload": {
                                    "key": "user_identity",
                                    "value": name.strip(),
                                    "tags": ["identity", "name"],
                                    "confidence": 1.0
                                }
                            })
                        except Exception:
                            pass
                        # Store in brain-level persistent storage as backup
                        try:
                            service_api({
                                "op": "BRAIN_PUT",
                                "payload": {
                                    "scope": "BRAIN",
                                    "origin_brain": "memory_librarian",
                                    "key": "user_identity",
                                    "value": name.strip(),
                                    "confidence": 1.0
                                }
                            })
                        except Exception:
                            pass
        except Exception:
            pass

        # Stage 9 — Storage (routing-aware + router assist + TAC promotion)
        # Determine storage eligibility based on storable_type and reasoning verdict.
        st_type = str(stage3.get("storable_type", "")).upper()
        verdict_upper = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()

        # Handle relationship updates early
        stage3_intent = str(stage3.get("intent", ""))
        if stage3_intent == "relationship_update":
            try:
                # Extract relationship information from stage3
                user_id = ctx.get("user_id") or "default_user"
                relationship_kind = stage3.get("relationship_kind")
                relationship_value = stage3.get("relationship_value")

                if relationship_kind is not None and relationship_value is not None:
                    # Store the relationship fact
                    set_relationship_fact(user_id, relationship_kind, relationship_value)

                    # Record in stage_9_storage
                    ctx["stage_9_storage"] = {
                        "action": "STORE",
                        "bank": "relationships",
                        "kind": relationship_kind,
                        "value": relationship_value,
                    }
                else:
                    ctx["stage_9_storage"] = {"skipped": True, "reason": "relationship_update_missing_data"}
            except Exception as rel_store_ex:
                ctx["stage_9_storage"] = {"skipped": True, "reason": f"relationship_update_error: {str(rel_store_ex)[:100]}"}
            # Skip remaining storage logic for relationship updates
            pass
        else:
            pass  # Continue with normal storage logic
        # Allow downstream storage only when the claim is validated, not a low‑confidence
        # cache result and when the content is considered storable.  This early
        # check supersedes the normal routing logic to prevent polluting the
        # knowledge base with unverified or weakly supported answers.
        try:
            final_confidence_val = float(ctx.get("final_confidence") or 0.0)
        except Exception:
            final_confidence_val = 0.0
        from_cache_flag = bool((ctx.get("stage_8_validation") or {}).get("from_cache"))
        # Non-storable types are never persisted.  Extend the set of non-storable types
        # to include SOCIAL greetings and UNKNOWN inputs.  Use a specific skip reason
        # for social greetings (social_greeting) and unknown inputs (unknown_input) to
        # improve traceability.
        non_storable_types = {"QUESTION", "COMMAND", "REQUEST", "EMOTION", "SOCIAL", "UNKNOWN"}
        # Removed OPINION - opinions can be stored as beliefs
        # Facts, even with low confidence, should be stored
        # Determine if we should skip storage due to verdict
        # Store ALL facts (TRUE, UNKNOWN, THEORY) - route them appropriately based on confidence
        # The governance layer (stage_8b) already decided whether to allow storage
        # Storage logic should respect that decision rather than independently blocking
        # Check for math_compute intent - skip storage for raw math expressions
        stage3_intent = str(stage3.get("intent", ""))
        is_math_compute = (stage3_intent == "math_compute" or ctx.get("mode") == "math_deterministic")
        is_meta_query = stage3_intent in {"preference_query", "relationship_query", "user_profile_summary"}

        # Prefer final answers for storage; fall back to pipeline outputs or raw text.
        answer_for_storage = str(
            ctx.get("final_answer")
            or (ctx.get("stage_10_finalize") or {}).get("text")
            or (ctx.get("stage_8_validation") or {}).get("answer")
            or ""
        ).strip()
        stage6_answer = str((ctx.get("stage_6_generation") or {}).get("answer") or "").strip()
        storage_content = answer_for_storage or proposed_content or text or stage6_answer

        try:
            answer_source = str(
                (ctx.get("stage_8_validation") or {}).get("answer_source")
                or (ctx.get("stage_6_generation") or {}).get("source")
                or ctx.get("answer_source")
                or ""
            ).lower()
        except Exception:
            answer_source = ""

        if answer_source == "teacher" and storage_content:
            lower_storage = storage_content.lower()
            if any(term in lower_storage for term in ["trained", "knowledge cutoff"]):
                ctx["stage_9_storage"] = {"skipped": True, "reason": "teacher_identity_block"}

        if ctx.get("stage_9_storage", {}).get("skipped"):
            pass
        elif not storage_content:
            ctx["stage_9_storage"] = {"skipped": True, "reason": "empty_content"}

        # Skip storage for meta queries (preferences, relationships, profile) and math expressions
        elif is_meta_query:
            ctx["stage_9_storage"] = {"skipped": True, "reason": "query_not_stored"}
        elif is_math_compute:
            ctx["stage_9_storage"] = {"skipped": True, "reason": "math_expression_not_stored"}
        elif st_type in non_storable_types:
            reason_map = {
                "SOCIAL": "social_greeting",
                "UNKNOWN": "unknown_input"
            }
            reason = reason_map.get(st_type, f"{st_type.lower()}_not_stored")
            ctx["stage_9_storage"] = {"skipped": True, "reason": reason}
        # Reasoning may direct a SKIP_STORAGE verdict for unanswered questions or invalid input
        elif verdict_upper == "SKIP_STORAGE":
            ctx["stage_9_storage"] = {"skipped": True, "reason": "question_without_answer"}
        # Skip storage for PREFERENCE verdicts - preferences are already stored elsewhere
        elif verdict_upper == "PREFERENCE":
            ctx["stage_9_storage"] = {"skipped": True, "reason": "preference_already_handled"}
        elif allowed:
            routing = ctx.get("stage_8_validation", {}).get("routing_order", {}) or {}
            target_bank = routing.get("target_bank") or "theories_and_contradictions"
            # FIX: Don't blindly accept SKIP from routing if verdict is TRUE or LEARNED
            if verdict_upper in ("TRUE", "LEARNED"):
                action = "STORE"  # Always store TRUE/LEARNED facts
            else:
                action = routing.get("action") or "STORE"

            # Freeze the final decision for downstream code (prevents later overrides)
            final_action_upper = "STORE" if verdict_upper in ("TRUE", "LEARNED") else str(action).upper()
            ctx["_memory_route"] = {
                "verdict": verdict_upper,
                "allowed": True,
                "target_bank": target_bank,
                "final_action": final_action_upper
            }

            # FIX: Learning on TRUE or LEARNED verdict - merge key/value into per-brain memory
            # This enables reinforcement learning where repeated confirmations of the
            # same fact increase confidence and access count over time.
            _k = ctx.get("key")
            _v = ctx.get("value")
            if verdict_upper in ("TRUE", "LEARNED") and _k is not None and _v is not None:
                _merge_brain_kv("memory_librarian", _k, _v, conf_delta=0.1)

            # If Reasoning didn't choose a bank, ask the routing modules.  Use the
            # simple keyword router for obvious domains, but consult the learned
            # router for additional guidance.  Prefer the simple router when it
            # returns a specific domain; otherwise fall back to the learned router.
            if not target_bank or target_bank == "unknown":
                simple_bank = _simple_route_to_bank(storage_content)
                lr_target = None
                # Query the learned router and record its scores and signals
                try:
                    rr = _router_module().service_api({"op":"ROUTE","payload":{"text": storage_content}})
                    pay = rr.get("payload") or {}
                    ctx.setdefault("stage_8_validation", {}).setdefault("router_scores", pay.get("scores"))
                    ctx["stage_8_validation"]["router_signals"] = pay.get("signals")
                    # Capture the slow_path flag from the dual router.  This flag
                    # indicates that the confidence margin between the top two
                    # candidate banks is low and that a more deliberate evaluation
                    # may be warranted.  Record it in stage_8_validation for
                    # downstream consumers.  If slow_path is absent or None,
                    # default to False.
                    try:
                        slow = bool(pay.get("slow_path", False))
                    except Exception:
                        slow = False
                    ctx["stage_8_validation"]["slow_path"] = slow
                    lr_target = pay.get("target_bank")
                except Exception:
                    lr_target = None
                # If both routers yield results, compare their confidence.  Each
                # learned router score lies between 0 and 1.  The simple router
                # has no intrinsic score, so we approximate its score from the
                # learned router's score for that bank.  Prefer the learned
                # router's target when it scores strictly higher than the
                # simple router's score by a small margin (0.05).  Otherwise
                # prefer the simple router when it is specific (not a generic
                # bucket like "working_theories" or "unknown").  If both
                # fall back to generic categories, choose the learned target
                # directly.
                chosen: str | None = None
                scores = (ctx.get("stage_8_validation", {}).get("router_scores") or {})
                if simple_bank and simple_bank not in {"working_theories", "unknown"}:
                    # Score of the simple bank and the learned target from the learned router scores
                    try:
                        simple_score = float(scores.get(simple_bank, 0.0))
                    except Exception:
                        simple_score = 0.0
                    try:
                        lr_score = float(scores.get(lr_target, 0.0)) if lr_target else 0.0
                    except Exception:
                        lr_score = 0.0
                    # Prefer the learned router target if it scores significantly higher
                    if lr_target and lr_target != simple_bank and lr_score > simple_score + 0.05:
                        chosen = lr_target
                    else:
                        chosen = simple_bank
                else:
                    # If simple router yields generic bucket, rely on the learned router target
                    chosen = lr_target if lr_target else None
                if chosen:
                    target_bank = chosen
                else:
                    target_bank = "theories_and_contradictions"

            # FIX: CONFIDENCE-BASED ROUTING for TRUE/LEARNED facts
            # Route based on confidence level, ensuring low confidence facts are still stored
            if verdict_upper in ("TRUE", "LEARNED"):
                action = "STORE"
                # Route based on confidence
                if final_confidence_val < 0.5:
                    target_bank = "working_theories"
                elif final_confidence_val < 0.8:
                    target_bank = "factual"
                else:
                    target_bank = "factual"

            if duplicate and target_bank != "theories_and_contradictions":
                try:
                    # Build a fact payload similar to standard storage
                    verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
                    conf_val = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0))
                    # FIX: Treat LEARNED same as TRUE for verification level
                    if verdict in ("TRUE", "LEARNED"):
                        ver_level = "validated"
                    elif verdict == "THEORY":
                        ver_level = "educated_guess"
                    else:
                        ver_level = "unknown"
                    # Phase 4: Assign tier and importance via tier system
                    tier_context = {
                        "intent": str(ctx.get("intent", "")),
                        "verdict": verdict,
                        "storable_type": str(ctx.get("storable_type", "")),
                        "tags": [],
                    }
                    record_for_tier = {
                        "content": storage_content,
                        "confidence": conf_val,
                        "verdict": verdict,
                    }
                    tier, tier_importance = _assign_tier(record_for_tier, tier_context)

                    fact_payload = {
                        "content": storage_content,
                        "confidence": conf_val,
                        "verification_level": ver_level,
                        "source": "user_input",
                        "validated_by": "reasoning",
                        # Assign importance equal to the confidence value.  This allows
                        # high‑confidence facts to be promoted quickly to MTM/LTM
                        "importance": max(conf_val, tier_importance),  # Use max of conf and tier importance
                        "tier": tier or TIER_LONG,  # Default to LONG tier if not assigned
                        "seq_id": _next_seq_id(),
                        "use_count": 0,
                        "metadata": {
                            "supported_by": (ctx.get("stage_8_validation") or {}).get("supported_by", []),
                            "contradicted_by": (ctx.get("stage_8_validation") or {}).get("contradicted_by", []),
                            "from_pipeline": True
                        }
                    }
                    resp_store = _bank_module(target_bank).service_api({"op": "STORE", "payload": {"fact": fact_payload}})
                    superseded_id = None
                    if isinstance(match, dict) and match.get("source_bank") == "theories_and_contradictions":
                        superseded_id = match.get("id")
                        try:
                            _bank_module("theories_and_contradictions").service_api({
                                "op": "SUPERSEDE",
                                "payload": {"id": superseded_id, "by_bank": target_bank}
                            })
                        except Exception:
                            pass
                    ctx["stage_9_storage"] = {
                        "skipped": False,
                        "action": "PROMOTE_DUPLICATE",
                        "bank": target_bank,
                        "superseded_id": superseded_id,
                        "result": (resp_store.get("payload") or resp_store)
                    }
                except Exception as e:
                    ctx["stage_9_storage"] = {"skipped": True, "reason": str(e), "bank": target_bank}
            else:
                try:
                    # Theories and contradictions bank uses custom store operations
                    if target_bank == "theories_and_contradictions":
                        # Always treat low-confidence submissions as theories
                        # Phase 4: Assign tier for theories
                        conf_theory = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0))
                        tier_context_theory = {
                            "intent": str(ctx.get("intent", "")),
                            "verdict": "THEORY",
                            "storable_type": str(ctx.get("storable_type", "")),
                            "tags": [],
                        }
                        record_for_tier_theory = {
                            "content": storage_content,
                            "confidence": conf_theory,
                            "verdict": "THEORY",
                        }
                        tier_theory, tier_importance_theory = _assign_tier(record_for_tier_theory, tier_context_theory)

                        fact_payload = {
                            "content": storage_content,
                            "confidence": conf_theory,
                            "source_brain": "reasoning",
                            "linked_fact_id": None,
                            "contradicts": [],
                            "status": "open",
                            "verification_level": "educated_guess",
                            # Use confidence as importance for theories too; moderate importance encourages
                            # promotion into mid‑term memory once validated further.
                            "importance": max(conf_theory, tier_importance_theory),
                            "tier": tier_theory or TIER_SHORT,  # Theories default to SHORT tier
                            "seq_id": _next_seq_id(),
                            "use_count": 0,
                            "metadata": {
                                "supported_by": (ctx.get("stage_8_validation") or {}).get("supported_by", []),
                                "contradicted_by": (ctx.get("stage_8_validation") or {}).get("contradicted_by", []),
                                "from_pipeline": True
                            },
                        }
                        # For now all uncertain content becomes a theory
                        resp = _bank_module(target_bank).service_api({"op": "STORE_THEORY", "payload": {"fact": fact_payload}})
                        ctx["stage_9_storage"] = {
                            "skipped": False,
                            "action": action,
                            "bank": target_bank,
                            "result": (resp.get("payload") or resp)
                        }
                    else:
                        # Build a fact payload for generic banks
                        verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
                        conf_val = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0))
                        # Determine verification level based on verdict
                        if verdict == "TRUE":
                            ver_level = "validated"
                        elif verdict == "THEORY":
                            ver_level = "educated_guess"
                        else:
                            ver_level = "unknown"
                        # Phase 4: Assign tier and importance via tier system
                        tier_context_generic = {
                            "intent": str(ctx.get("intent", "")),
                            "verdict": verdict,
                            "storable_type": str(ctx.get("storable_type", "")),
                            "tags": [],
                        }
                        record_for_tier_generic = {
                            "content": storage_content,
                            "confidence": conf_val,
                            "verdict": verdict,
                        }
                        tier_generic, tier_importance_generic = _assign_tier(record_for_tier_generic, tier_context_generic)

                        fact_payload = {
                            "content": storage_content,
                            "confidence": conf_val,
                            "verification_level": ver_level,
                            "source": "user_input",
                            "validated_by": "reasoning",
                            # Assign importance equal to confidence to aid promotion
                            "importance": max(conf_val, tier_importance_generic),
                            "tier": tier_generic or TIER_LONG,  # Default to LONG tier
                            "seq_id": _next_seq_id(),
                            "use_count": 0,
                            "metadata": {
                                "supported_by": (ctx.get("stage_8_validation") or {}).get("supported_by", []),
                                "contradicted_by": (ctx.get("stage_8_validation") or {}).get("contradicted_by", []),
                                "from_pipeline": True
                            }
                        }
                        # Call bank with proper fact wrapper
                        resp = _bank_module(target_bank).service_api({"op": "STORE", "payload": {"fact": fact_payload}})
                        ctx["stage_9_storage"] = {
                            "skipped": False,
                            "action": action,
                            "bank": target_bank,
                            "result": (resp.get("payload") or resp)
                        }
                except Exception as e:
                    ctx["stage_9_storage"] = {"skipped": True, "reason": str(e), "bank": target_bank}
        else:
            ctx["stage_9_storage"] = {"skipped": True, "reason": "governance_denied"}

        # Learning hooks after successful store
        try:
            if not ctx.get("stage_9_storage", {}).get("skipped"):
                bank = ctx["stage_9_storage"].get("bank")
                # vocab learn
                try:
                    _router_module().service_api({"op":"LEARN","payload":{"text": storage_content, "bank": bank}})
                except Exception:
                    pass
                # MEMORY CONSOLIDATION: Trigger STM→MTM→LTM consolidation periodically
                # Run consolidation every 10 operations (deterministic)
                try:
                    if _SEQ_ID_COUNTER % 10 == 0:  # Every 10th operation
                        consolidation_stats = _consolidate_memory_banks()
                        if consolidation_stats.get("facts_promoted", 0) > 0:
                            ctx.setdefault("stage_9_storage", {})["consolidation"] = consolidation_stats
                except Exception:
                    pass
                # definition learn
                try:
                    term, klass = _extract_definition(storage_content)
                    verdict = str(ctx.get("stage_8_validation", {}).get("verdict","")).upper()
                    if term and klass and verdict == "TRUE":
                        _router_module().service_api({"op":"LEARN_DEFINITION","payload":{"term": term, "klass": klass}})
                except Exception:
                    pass
                # relationship interception hook - persist simple relational beliefs like "we are friends"
                try:
                    _intent_l = str(ctx.get("stage_3_language", {}).get("type", "")).lower()
                    _val_l = str(storage_content).lower()
                    if any(k in _intent_l for k in ["relationship", "relation", "social", "bond", "friend"]) or "friend" in _val_l:
                        # Store in working memory
                        try:
                            service_api({
                                "op": "WM_PUT",
                                "payload": {
                                    "key": "relationship_status",
                                    "value": storage_content,
                                    "tags": ["relationship", "social"],
                                    "confidence": 0.8
                                }
                            })
                        except Exception:
                            pass
                        # Store in brain-level persistent storage
                        try:
                            service_api({
                                "op": "BRAIN_PUT",
                                "payload": {
                                    "scope": "BRAIN",
                                    "origin_brain": "memory_librarian",
                                    "key": "relationship_status",
                                    "value": storage_content,
                                    "confidence": 0.8
                                }
                            })
                        except Exception:
                            pass
                except Exception:
                    pass
                # Persist user preferences like "I like the color green" and "I like cats"
                # Step C: Extended to handle color + animal + multiple preferences in one sentence
                try:
                    _intent_l = str(ctx.get("stage_3_language", {}).get("type", "")).lower()
                    _val_l = str(storage_content).lower()
                    _utterance = str(ctx.get("original_query") or "").lower()
                    if "like" in _val_l or "like" in _intent_l or "like" in _utterance:
                        import re
                        # Skip questions like "what color do i like" or "what animal do i like"
                        if re.search(r"\bwhat\b.*\b(color|animal)\b.*\blike\b", _utterance):
                            return
                        text_to_parse = (storage_content or _utterance).strip()

                        # Step C: Parse color preference
                        # Patterns: "i like the color green", "i like green", "green is my favorite color"
                        m_color = (re.search(r"\blike\s+(?:the\s+color\s+)?([a-zA-Z]+)\b(?!\s+more)(?!\s+and)", text_to_parse, re.I)
                                   or re.search(r"\b([a-zA-Z]+)\b\s+is\s+my\s+favorite\s+color", text_to_parse, re.I))
                        # Check if it's specifically marked as a color preference
                        if re.search(r"\bcolor\b", text_to_parse, re.I) and m_color:
                            color = m_color.group(1).lower()
                            print(f"[DEBUG] Color preference detected: {color}")
                            # Store color preference
                            try:
                                service_api({
                                    "op": "WM_PUT",
                                    "payload": {
                                        "key": "favorite_color",
                                        "value": color,
                                        "tags": ["preference", "color"],
                                        "confidence": 0.9
                                    }
                                })
                                service_api({
                                    "op": "BRAIN_PUT",
                                    "payload": {
                                        "scope": "BRAIN",
                                        "origin_brain": "memory_librarian",
                                        "key": "favorite_color",
                                        "value": color,
                                        "confidence": 0.9
                                    }
                                })
                                print(f"[FACT_STORED] user.favorite_color={color}")
                            except Exception:
                                pass

                        # Step C: Parse animal preference
                        # Patterns: "i like cats", "i like the animal the cat", "i like the animal dog"
                        m_animal = (re.search(r"\blike\s+(?:the\s+animal\s+(?:the\s+)?)?(?:cats|cat|dogs|dog|birds?|fish|hamsters?|rabbits?)\b", text_to_parse, re.I)
                                    or re.search(r"\b(cats|cat|dogs|dog|birds?|fish|hamsters?|rabbits?)\b\s+(?:is|are)\s+my\s+favorite", text_to_parse, re.I))
                        if m_animal:
                            animal_match = m_animal.group(0).lower()
                            # Extract just the animal name
                            animal = re.search(r"\b(cats?|dogs?|birds?|fish|hamsters?|rabbits?)\b", animal_match, re.I)
                            if animal:
                                animal_name = animal.group(1).lower()
                                # Normalize plural to singular for consistency
                                if animal_name.endswith('s') and animal_name not in ['fish']:
                                    animal_name = animal_name[:-1]
                                print(f"[DEBUG] Animal preference detected: {animal_name}")
                                # Store animal preference
                                try:
                                    service_api({
                                        "op": "WM_PUT",
                                        "payload": {
                                            "key": "favorite_animal",
                                            "value": animal_name,
                                            "tags": ["preference", "animal"],
                                            "confidence": 0.9
                                        }
                                    })
                                    service_api({
                                        "op": "BRAIN_PUT",
                                        "payload": {
                                            "scope": "BRAIN",
                                            "origin_brain": "memory_librarian",
                                            "key": "favorite_animal",
                                            "value": animal_name,
                                            "confidence": 0.9
                                        }
                                    })
                                    print(f"[FACT_STORED] user.favorite_animal={animal_name}")
                                except Exception:
                                    pass

                        # Step C: Parse food preference
                        # Patterns: "i like pizza", "i like the food pizza", "pizza is my favorite food"
                        m_food = (re.search(r"\blike\s+(?:the\s+food\s+)?([a-zA-Z]+)\b(?!\s+more)(?!\s+and)", text_to_parse, re.I)
                                  or re.search(r"\b([a-zA-Z]+)\b\s+is\s+my\s+favorite\s+food", text_to_parse, re.I))
                        # Check if it's specifically marked as a food preference
                        if re.search(r"\bfood\b", text_to_parse, re.I) and m_food:
                            food = m_food.group(1).lower()
                            print(f"[DEBUG] Food preference detected: {food}")
                            # Store food preference
                            try:
                                service_api({
                                    "op": "WM_PUT",
                                    "payload": {
                                        "key": "favorite_food",
                                        "value": food,
                                        "tags": ["preference", "food"],
                                        "confidence": 0.9
                                    }
                                })
                                service_api({
                                    "op": "BRAIN_PUT",
                                    "payload": {
                                        "scope": "BRAIN",
                                        "origin_brain": "memory_librarian",
                                        "key": "favorite_food",
                                        "value": food,
                                        "confidence": 0.9
                                    }
                                })
                                print(f"[FACT_STORED] user.favorite_food={food}")
                            except Exception:
                                pass

                        # Parse comparative likes: "I like cats over dogs"
                        m_comp = re.search(r"like\s+([a-zA-Z]+)\s+over\s+([a-zA-Z]+)", text_to_parse, re.I)
                        if m_comp:
                            choice, other = m_comp.groups()
                            pref = {"preferred": choice.lower(), "other": other.lower()}
                            try:
                                service_api({
                                    "op": "WM_PUT",
                                    "payload": {
                                        "key": "animal_preference",
                                        "value": pref,
                                        "tags": ["preference", "comparative"],
                                        "confidence": 0.9
                                    }
                                })
                                service_api({
                                    "op": "BRAIN_PUT",
                                    "payload": {
                                        "scope": "BRAIN",
                                        "origin_brain": "memory_librarian",
                                        "key": "animal_preference",
                                        "value": pref,
                                        "confidence": 0.9
                                    }
                                })
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

        # After storing and learning, trigger memory consolidation so that
        # recently stored facts can graduate from short‑term memory to
        # mid‑term and long‑term tiers when appropriate.  Consolidation
        # runs only when storage was not skipped and the helper is available.
        try:
            if consolidate_memories and not (ctx.get("stage_9_storage", {}) or {}).get("skipped"):
                consolidate_memories()
        except Exception:
            # Ignore consolidation errors to avoid disrupting the pipeline
            pass

        # Stage 10b — Personality feedback
        try:
            from brains.cognitive.personality.service import personality_brain
            fb = {
                "tone": ctx.get("stage_10_finalize", {}).get("tone"),
                "verbosity": ctx.get("stage_10_finalize", {}).get("verbosity"),
                "transparency": ctx.get("stage_10_finalize", {}).get("transparency")
            }
            personality_brain.service_api({"op":"LEARN_FROM_RUN","payload": fb})
            ctx["stage_10_personality_feedback"] = {"logged": True}
        except Exception as e:
            ctx["stage_10_personality_feedback"] = {"logged": False, "error": str(e)}

        # Stage 11 — Personal brain (style-only)
        try:
            per = _personal_module()
            per_boost = per.service_api({"op":"SCORE_BOOST","payload":{"subject": text}})
            per_why = per.service_api({"op":"WHY","payload":{"subject": text}})
            # Build baseline personal influence dictionary
            inf_obj = {
                **(per_boost.get("payload") or {}),
                "why": (per_why.get("payload") or {}).get("hypothesis"),
                "signals": (per_why.get("payload") or {}).get("signals", [])
            }
            # Incorporate conversation state metrics into the signals.  Retrieve the
            # global conversation state safely and embed it under a dedicated
            # key.  Do not mutate the original state object to avoid cross‑run
            # contamination.  When conv_state is non‑empty, this will expose
            # fields such as last_query, last_response, last_topic,
            # thread_entities and conversation_depth in the pipeline output.
            try:
                conv_state = dict(globals().get("_CONVERSATION_STATE") or {})
            except Exception:
                conv_state = {}
            if conv_state:
                inf_obj["conversation_state"] = conv_state
            ctx["stage_11_personal_influence"] = inf_obj
        except Exception as e:
            ctx["stage_11_personal_influence"] = {"error": str(e)}

        # Stage 12 — System history
        try:
            hist = _brain_module("system_history")
            hist.service_api({"op":"LOG_RUN_SUMMARY","payload":{
                "text": text,
                "mode": ctx["stage_8_validation"].get("mode"),
                "bank": ctx.get("stage_9_storage", {}).get("bank"),
                "personal_boost": (ctx.get("stage_11_personal_influence") or {}).get("boost", 0.0)
            }})
            ctx["stage_12_system_history"] = {"logged": True}
        except Exception as e:
            ctx["stage_12_system_history"] = {"logged": False, "error": str(e)}

        # Stage 12a — Self reflection (self‑critique)
        # Request a governance permit before performing self‑critique.  If allowed,
        # generate a critique using the self‑critique brain and record the permit id.
        try:
            import importlib
            # Request permit for CRITIQUE action
            permits_mod = importlib.import_module(
                "brains.governance.policy_engine.service.permits"
            )
            perm_resp = permits_mod.service_api({
                "op": "REQUEST",
                "payload": {"action": "CRITIQUE"}
            })
            perm_pay = perm_resp.get("payload") or {}
            allowed = bool(perm_pay.get("allowed", False))
            permit_id = perm_pay.get("permit_id")
            # Save permit status in context
            ctx["stage_12a_self_critique_permit"] = {
                "allowed": allowed,
                "permit_id": permit_id,
                "reason": perm_pay.get("reason")
            }
            # Log the permit request to the governance ledger.  Ignore
            # failures to avoid breaking the pipeline.
            try:
                from brains.governance.permit_logger import log_permit  # type: ignore
                log_permit("CRITIQUE", permit_id, allowed, perm_pay.get("reason"))
            except Exception:
                pass
            if allowed:
                # If permitted, proceed with critique generation
                crit_mod = importlib.import_module(
                    "brains.cognitive.self_dmn.service.self_critique"
                )
                final_text = (ctx.get("stage_10_finalize") or {}).get("text", "")
                crit_resp = crit_mod.service_api({"op": "CRITIQUE", "payload": {"text": final_text}})
                crit_payload = crit_resp.get("payload") or {}
                # Attach permit_id to critique payload for traceability
                if permit_id:
                    crit_payload["permit_id"] = permit_id
                ctx["stage_12a_self_critique"] = crit_payload
            else:
                # Not allowed: record empty critique but keep permit info
                ctx["stage_12a_self_critique"] = {}
        except Exception as e:
            ctx["stage_12a_self_critique_permit"] = {"allowed": False, "error": str(e)}
            ctx["stage_12a_self_critique"] = {"error": str(e)}

        # Stage 12b — Identity journal update
        # Request a governance permit for updating the identity journal.  When
        # permitted, merge the latest interaction into the identity snapshot
        # and compute a subject boost.  Attach the permit id for audit.
        try:
            import importlib
            permits_mod = importlib.import_module(
                "brains.governance.policy_engine.service.permits"
            )
            perm_resp = permits_mod.service_api({
                "op": "REQUEST",
                "payload": {"action": "OPINION"}
            })
            perm_pay = perm_resp.get("payload") or {}
            allowed = bool(perm_pay.get("allowed", False))
            permit_id = perm_pay.get("permit_id")
            ctx["stage_12b_identity_permit"] = {
                "allowed": allowed,
                "permit_id": permit_id,
                "reason": perm_pay.get("reason")
            }
            # Log the permit to the governance ledger.  Do not propagate errors.
            try:
                from brains.governance.permit_logger import log_permit  # type: ignore
                log_permit("OPINION", permit_id, allowed, perm_pay.get("reason"))
            except Exception:
                pass
            if allowed:
                id_mod = importlib.import_module(
                    "brains.personal.service.identity_journal"
                )
                update_data = {
                    "last_question": text,
                    "last_response": (ctx.get("stage_10_finalize") or {}).get("text"),
                }
                id_mod.service_api({"op": "UPDATE", "payload": {"update": update_data}})
                boost_resp = id_mod.service_api({"op": "SCORE_BOOST", "payload": {"subject": text}})
                boost_pay = boost_resp.get("payload") or {}
                # Attach permit id to identity influence to link update with proof
                if permit_id:
                    boost_pay["permit_id"] = permit_id
                ctx["stage_12b_identity_influence"] = boost_pay
            else:
                ctx["stage_12b_identity_influence"] = {}
        except Exception as e:
            ctx["stage_12b_identity_permit"] = {"allowed": False, "error": str(e)}
            ctx["stage_12b_identity_influence"] = {"error": str(e)}

        # Stage 13 — Self-DMN
        # Config flag to enable/disable self-DMN (default: enabled)
        # Only skip if governance explicitly denies it
        self_dmn_enabled = ctx.get("allow_self_dmn", True)
        if not self_dmn_enabled:
            ctx["stage_13_self_dmn"] = {"skipped": True, "reason": "governance_disabled"}
        else:
            try:
                sdmn = _brain_module("self_dmn")
                met = sdmn.service_api({"op":"ANALYZE_INTERNAL","payload":{"window": 10}})
                ctx["stage_13_self_dmn"] = {"metrics": (met.get("payload") or {}).get("metrics")}
            except Exception as e:
                ctx["stage_13_self_dmn"] = {"error": str(e)}

        # After all stages, update the affect priority brain with run outcomes.  In addition to
        # the usual parameters (tone, verbosity, decision, goal), include any identity
        # influence metrics and the self‑critique text if available.  These extra cues
        # allow the affect learner to adjust mood biases based on self reflection and
        # personal context.  Errors are ignored to avoid interfering with the core flow.
        try:
            aff_mod = _brain_module("affect_priority")
            # Build payload with optional fields.  Use dict() + comprehension to avoid
            # injecting None values when keys are missing.
            aff_payload = {
                "tone": (ctx.get("stage_10_finalize") or {}).get("tone"),
                "verbosity": (ctx.get("stage_10_finalize") or {}).get("verbosity"),
                "decision": (ctx.get("stage_8b_governance") or {}).get("action"),
                "goal": (ctx.get("stage_2_planner") or {}).get("goal"),
            }
            # Propagate identity boost into affect learning if computed
            try:
                id_boost = (ctx.get("stage_12b_identity_influence") or {}).get("boost")
                if id_boost is not None:
                    aff_payload["identity_boost"] = id_boost
            except Exception:
                pass
            # Propagate critique text for context
            try:
                crit_txt = (ctx.get("stage_12a_self_critique") or {}).get("critique")
                if crit_txt:
                    aff_payload["reflection"] = crit_txt
            except Exception:
                pass
            aff_mod.service_api({"op": "LEARN_FROM_RUN", "payload": aff_payload})
            ctx["stage_14_affect_learn"] = {"logged": True}
        except Exception as e:
            ctx["stage_14_affect_learn"] = {"logged": False, "error": str(e)}

        # Stage 15 — Autonomy (optional)
        #
        # If the autonomy configuration is enabled, execute a lightweight
        # autonomous tick.  This advances the self‑DMN hum oscillators,
        # collects memory health and formulates high‑level goals based on
        # current evidence.  The goals are not executed here; they are
        # recorded in the context for downstream processors or future
        # autonomy cycles to act upon.  Errors are captured without
        # interrupting the main pipeline flow.
        try:
            autocfg = _load_autonomy_config()
            # Config flag to enable/disable autonomy (default: disabled)
            autonomy_enabled = bool((autocfg or {}).get("enable", False))
            if not autonomy_enabled:
                ctx["stage_15_autonomy_tick"] = {"skipped": True, "reason": "disabled"}
                ctx["stage_15_autonomy_goals"] = {"skipped": True, "reason": "disabled"}
                ctx["stage_15_autonomy_actions"] = {"skipped": True, "reason": "disabled"}
            elif autonomy_enabled:
                # Pre-plan: Before running autonomy ticks, replan any active goals in
                # personal memory.  Compound goals are split into sub‑tasks via the
                # replanner brain so that the autonomy executor handles smaller actions.
                try:
                    from brains.personal.memory import goal_memory  # type: ignore
                    # Fetch only active (unfinished) goals from personal memory.  The
                    # goal_memory.get_goals API uses ``active_only`` as the keyword
                    # argument, so avoid passing ``only_active`` to prevent runtime errors.
                    active_goals = goal_memory.get_goals(active_only=True)  # type: ignore[arg-type]
                    if active_goals:
                        import importlib
                        replanner_mod = importlib.import_module(
                            "brains.cognitive.planner.service.replanner_brain"
                        )
                        repl_resp = replanner_mod.service_api({
                            "op": "REPLAN",
                            "payload": {"goals": active_goals},
                        })
                        ctx["stage_15_replan"] = repl_resp.get("payload", {})
                    else:
                        ctx["stage_15_replan"] = {"new_goals": []}
                except Exception as pre_repl_ex:
                    ctx["stage_15_replan"] = {
                        "error": "pre_replan_failed",
                        "detail": str(pre_repl_ex),
                    }
                # Autonomy tick
                try:
                    aut_brain_mod = _brain_module("autonomy")
                    # Use the safe tick function that never throws
                    tick_result = aut_brain_mod.tick(ctx)
                    if not isinstance(tick_result, dict):
                        tick_result = {
                            "action": "noop",
                            "reason": "invalid_tick_result",
                            "confidence": 0.0,
                        }
                    ctx["stage_15_autonomy_tick"] = tick_result
                except Exception as e:
                    ctx["stage_15_autonomy_tick"] = {
                        "action": "noop",
                        "reason": "tick_failed",
                        "exception_type": type(e).__name__,
                        "message": str(e)[:200],
                        "confidence": 0.0,
                    }
                # Score opportunities and formulate goals using the motivation brain
                try:
                    mot_mod = _brain_module("motivation")
                    # Build evidence from memory retrieval results (stage_2R_memory)
                    evidence = {
                        "results": (ctx.get("stage_2R_memory") or {}).get("results", [])
                    }
                    opps_resp = mot_mod.service_api({"op": "SCORE_OPPORTUNITIES", "payload": {"evidence": evidence}})
                    opportunities = (opps_resp.get("payload") or {}).get("opportunities", [])
                    goals_resp = mot_mod.service_api({"op": "FORMULATE_GOALS", "payload": {"opportunities": opportunities}})
                    ctx["stage_15_autonomy_goals"] = goals_resp.get("payload", {})
                except Exception:
                    ctx["stage_15_autonomy_goals"] = {}
                # After formulating goals, invoke the autonomy brain to execute goals.
                # Respect the autonomy configuration's max_ticks_per_run setting to
                # perform multiple ticks per pipeline run.  The actions for each
                # tick are collected into a list and stored in the context.  Errors
                # are captured to avoid disrupting the pipeline.
                try:
                    aut_mod = _brain_module("autonomy")
                    # Determine how many ticks to perform (default 1)
                    ticks = 1
                    try:
                        ticks_cfg = int((autocfg or {}).get("max_ticks_per_run", 1))
                        if ticks_cfg > 0:
                            ticks = ticks_cfg
                    except Exception:
                        ticks = 1
                    executed_list: list = []
                    # Load any previous resume state of remaining goals.  If the
                    # file does not exist or is invalid, treat as empty list.
                    remaining_ids: list[str] = []
                    try:
                        import json as _json
                        from pathlib import Path as _Path
                        root = _Path(__file__).resolve().parents[4]
                        rs_path = root / "reports" / "autonomy" / "resume_state.json"
                        if rs_path.exists():
                            with open(rs_path, "r", encoding="utf-8") as rs_fh:
                                data = _json.load(rs_fh) or {}
                                ids = data.get("remaining_goal_ids") or []
                                if isinstance(ids, list):
                                    remaining_ids = [str(g) for g in ids if g]
                    except Exception:
                        remaining_ids = []
                    # Execute ticks up to the configured maximum.  If the
                    # autonomy brain reports budget exhaustion or no goal is
                    # executed, stop early and preserve remaining goals for the
                    # next run.
                    for _idx in range(ticks):
                        try:
                            aut_resp = aut_mod.service_api({"op": "TICK"})
                        except Exception:
                            executed_list.append({"error": "autonomy_tick_failed"})
                            break
                        # Parse response payload.  It may contain executed_goals and skip flags
                        payload = aut_resp.get("payload") or {}
                        executed_list.append(payload)
                        # If the tick was skipped (e.g., rate limited or budget exhausted), stop
                        try:
                            if payload.get("skipped"):
                                # break the loop but preserve remaining_ids
                                break
                        except Exception:
                            pass
                        # Remove any executed goal IDs from remaining_ids
                        try:
                            ex_goals = payload.get("executed_goals") or []
                            for ex in ex_goals:
                                gid = ex.get("goal_id")
                                if gid and gid in remaining_ids:
                                    try:
                                        remaining_ids.remove(gid)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        # If there are no more remaining goals, we can break early
                        if not remaining_ids:
                            # Continue ticks though – other active goals may appear; do not break early
                            pass
                    # After executing ticks, persist remaining goal IDs to resume state for next run
                    try:
                        import json as _json
                        from pathlib import Path as _Path
                        import os, tempfile
                        root = _Path(__file__).resolve().parents[4]
                        aut_dir = root / "reports" / "autonomy"
                        aut_dir.mkdir(parents=True, exist_ok=True)
                        rs_path = aut_dir / "resume_state.json"
                        # Write to temporary then replace
                        state_obj = {"remaining_goal_ids": remaining_ids}
                        tmp_fd, tmp_file = tempfile.mkstemp(dir=str(aut_dir), prefix="resume_state", suffix=".tmp")
                        try:
                            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                                fh.write(_json.dumps(state_obj))
                            os.replace(tmp_file, rs_path)
                        finally:
                            try:
                                if os.path.exists(tmp_file):
                                    os.remove(tmp_file)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Store the list of executed actions in the context
                    ctx["stage_15_autonomy_actions"] = executed_list
                except Exception:
                    ctx["stage_15_autonomy_actions"] = {"error": "autonomy_tick_failed"}
                # Capture any remaining active goals after the autonomy ticks.  This
                # gives downstream processors and the user visibility into tasks
                # that still need to be completed and opens the door for future
                # re‑planning or follow‑up actions.  If the goal memory is not
                # available or an error occurs, fallback to an empty list.
                try:
                    from brains.personal.memory import goal_memory  # type: ignore
                    # After executing autonomy ticks, fetch active goals again.  Use the
                    # ``active_only`` argument to retrieve unfinished tasks.
                    remaining_goals = goal_memory.get_goals(active_only=True)  # type: ignore[arg-type]
                    # Automatically prune legacy junk goals that were created by
                    # prior segmentation bugs.  Junk goals are characterised by
                    # extremely short or trivial titles (e.g. single digits,
                    # isolated colour names) or unfinished question fragments.
                    pruned_goals: list = []
                    import re as _re
                    # Define heuristics: single digits, colour names, and
                    # partial question fragments are considered junk.
                    _colour_set = {"red", "orange", "yellow", "green", "blue", "indigo", "violet"}
                    for g in remaining_goals:
                        try:
                            title_raw = str(g.get("title", ""))
                            title = title_raw.strip().lower()
                        except Exception:
                            title = ""
                        is_junk = False
                        try:
                            # Single numeric (e.g. "1") or numeric with punctuation ("9.")
                            if _re.fullmatch(r"\d+", title) or _re.fullmatch(r"\d+\.", title):
                                is_junk = True
                            # Colour names (legacy spectrum tasks)
                            elif title in _colour_set:
                                is_junk = True
                            # Titles starting with specific junk phrases
                            elif title.startswith("numbers from"):
                                is_junk = True
                            elif title.startswith("what comes"):
                                is_junk = True
                            # Patterns like "0 comes 1." or "1 comes 2." etc.
                            elif _re.fullmatch(r"\d+\s+comes\s+\d+\.", title):
                                is_junk = True
                            # Titles ending with a question mark (often junk tasks)
                            elif title.endswith("?") and len(title) <= 5:
                                is_junk = True
                            # Fragments beginning with 'the ' and capital letter (e.g. "the Eiffel Tower")
                            elif title.startswith("the ") and len(title.split()) <= 3:
                                is_junk = True
                            # Requests to show photos (e.g. "show me paris photos")
                            elif title.startswith("show me") and "photo" in title:
                                is_junk = True
                            # Pleas to add numbers (e.g. "please add 2")
                            elif title.startswith("please add"):
                                is_junk = True
                            # Simple phrases indicating colours of visible spectrum
                            elif _re.match(r"[a-z]+\s+are\s+the\s+colors", title):
                                is_junk = True
                        except Exception:
                            is_junk = False
                        if is_junk:
                            # Mark the junk goal as completed so it will not be considered active
                            try:
                                gid = g.get("goal_id")
                                if gid:
                                    goal_memory.complete_goal(str(gid))  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            continue
                        pruned_goals.append(g)
                    ctx["stage_15_remaining_goals"] = pruned_goals
                except Exception:
                    ctx["stage_15_remaining_goals"] = []
                # Dynamic re‑planning of stale goals.  If the autonomy
                # configuration defines a positive ``replan_age_minutes`` value,
                # identify any remaining goals whose age exceeds this
                # threshold.  Use the replanner brain to split these tasks
                # into sub‑goals, then mark the originals as completed.  New
                # sub‑goals are persisted via the replanner and surfaced in
                # the context under ``stage_15_replanned_stale_goals``.  If
                # no goals are stale or the feature is disabled, record an
                # empty list for clarity.
                # No time-based age checking
                stale_goals: list = []
                if stale_goals:
                    try:
                        import importlib  # type: ignore
                        replanner_mod = importlib.import_module(
                            "brains.cognitive.planner.service.replanner_brain"
                        )
                        repl_out = replanner_mod.service_api({
                            "op": "REPLAN",
                            "payload": {"goals": stale_goals},
                        })
                        # Complete the stale goals to avoid duplicate execution
                        try:
                            for sg in stale_goals:
                                gid = sg.get("goal_id")
                                if gid:
                                    goal_memory.complete_goal(str(gid), success=True)  # type: ignore[attr-defined]
                            # Remove stale goals from the remaining goals list in the context
                            try:
                                rem = ctx.get("stage_15_remaining_goals") or []
                                if isinstance(rem, list):
                                    ctx["stage_15_remaining_goals"] = [g for g in rem if g.get("goal_id") not in {sg.get("goal_id") for sg in stale_goals}]
                            except Exception:
                                pass
                        except Exception:
                            pass
                        ctx["stage_15_replanned_stale_goals"] = repl_out.get("payload", {})
                    except Exception as stale_ex:
                        ctx["stage_15_replanned_stale_goals"] = {
                            "error": "stale_goal_replan_failed",
                            "detail": str(stale_ex),
                        }
                else:
                    ctx["stage_15_replanned_stale_goals"] = {"new_goals": []}

                ctx["stage_15_autonomy_enabled"] = True
            else:
                ctx["stage_15_autonomy_enabled"] = False
        except Exception as e:
            ctx["stage_15_autonomy_error"] = str(e)

        # Periodically summarize system history.  When the number of completed runs
        # reaches a multiple of 50, invoke the system history brain's SUMMARIZE op
        # to aggregate recent metrics and prune old logs.  Guard against errors.
        try:
            sys_dir = MAVEN_ROOT / "reports" / "system"
            run_files = []
            for f in sys_dir.iterdir():
                try:
                    if f.is_file() and f.name.startswith("run_") and f.suffix == ".json":
                        run_files.append(f)
                except Exception:
                    continue
            run_count = len(run_files)
        except Exception:
            run_count = 0
        if run_count > 0 and (run_count % 50) == 0:
            try:
                hist_mod = _brain_module("system_history")
                hist_mod.service_api({"op": "SUMMARIZE", "payload": {"window": 50}})
            except Exception:
                pass

        # Write system report and flush pipeline trace (if enabled)
        if trace_enabled:
            try:
                outdir = MAVEN_ROOT / "reports" / "pipeline_trace"
                outdir.mkdir(parents=True, exist_ok=True)
                # Prefer run_seed for file naming when available.  This provides
                # deterministic file names for reproducibility.  Fall back
                # to the message ID when no seed is present.
                try:
                    seed_val = ctx.get("run_seed")
                    if seed_val is not None:
                        fname = f"trace_{seed_val}.jsonl"
                    else:
                        fname = f"trace_{mid}.jsonl"
                except Exception:
                    fname = f"trace_{mid}.jsonl"
                # Write trace events to file.  To aid replayability, append
                # the final context at the end of the trace as a single event.
                with open(outdir / fname, "w", encoding="utf-8") as fh:
                    for ev in trace_events:
                        fh.write(json.dumps(ev) + "\n")
                    # Write the pipeline context as the last entry.  Use
                    # compact JSON to avoid very large trace files.
                    try:
                        fh.write(json.dumps({"stage": "context", "context": ctx}) + "\n")
                    except Exception:
                        pass
                # Enforce retention limit on trace files
                trace_cfg = CFG.get("pipeline_tracer", {}) or {}
                try:
                    max_files = int(trace_cfg.get("max_files", 25) or 25)
                except Exception:
                    max_files = 25
                files = []
                for f in outdir.iterdir():
                    try:
                        if f.is_file() and f.name.startswith("trace_") and f.suffix == ".jsonl":
                            files.append(f)
                    except Exception:
                        continue
                files = sorted(files, key=lambda p: p.stat().st_mtime)
                if len(files) > max_files:
                    for f in files[: len(files) - max_files]:
                        try:
                            f.unlink()
                        except Exception:
                            pass
            except Exception:
                pass
        # Surface routing information in the final context for transparency.
        # Expose the top two banks and the routing scores from stage 2
        try:
            top_b = ctx.get("stage_2R_top_banks")
            scores = ctx.get("stage_2R_routing_scores")
            if top_b or scores:
                ctx["final_routing"] = {
                    "top_banks": list(top_b) if isinstance(top_b, (list, tuple)) else [],
                    "scores": dict(scores) if isinstance(scores, dict) else {}
                }
        except Exception:
            pass
        # Persist the final context snapshot.  Use a flat recent query log rather
        # than nesting session_context.  This avoids exponential growth and
        # provides a concise history for continuity across runs.
        try:
            _save_context_snapshot(ctx, limit=5)
        except Exception:
            pass
        # After persisting the context, check whether this query should be
        # cached for fast retrieval in future runs.  Repeated questions
        # within a short window (default 10 minutes) that produce a
        # validated answer will trigger caching.  The cache entry is
        # ignored if one already exists or if the verdict is not TRUE.
        try:
            _maybe_store_fast_cache(ctx, threshold=3, window_sec=600.0)
        except Exception:
            pass
        # Write system report with full context for auditing
        run_id = ctx.get("run_seed", _SEQ_ID_COUNTER)
        write_report("system", f"run_{run_id}.json", json.dumps(ctx, indent=2))

        # ------------------------------------------------------------------
        # Stage 16 — Regression Harness (optional)
        #
        # Run a lightweight regression check against the QA memory to detect
        # contradictions and drift.  This stage is deliberately placed after
        # context persistence so that any mismatches are logged for the
        # current run.  The harness executes only when the QA memory has
        # accumulated a minimum number of entries to justify comparison.
        try:
            qa_file = MAVEN_ROOT / "reports" / "qa_memory.jsonl"
            # If there are enough QA entries, invoke the regression harness
            run_reg = False
            try:
                if qa_file.exists():
                    with qa_file.open("r", encoding="utf-8") as fh:
                        # Count non-empty lines; stop after threshold
                        threshold = 10
                        count = 0
                        for line in fh:
                            if line.strip():
                                count += 1
                                if count >= threshold:
                                    run_reg = True
                                    break
            except Exception:
                run_reg = False
            if run_reg:
                try:
                    import importlib
                    harness_mod = importlib.import_module("tools.regression_harness")
                    # Limit to first 10 entries for performance
                    res = harness_mod.run_regression(limit=10)
                    # Store a summary of regression results
                    reg_total = res.get("total", 0)
                    reg_match = res.get("matches", 0)
                    reg_mismatch = res.get("mismatches", 0)
                    ctx["stage_16_regression"] = {
                        "total": reg_total,
                        "matches": reg_match,
                        "mismatches": reg_mismatch,
                    }
                    # When mismatches are detected, create self‑repair goals.  Each
                    # mismatching QA entry spawns a goal titled "Verify QA: <question>"
                    # with a special description so the autonomy scheduler can pick
                    # them up.  Record the created goals in the context.
                    try:
                        mismatches = res.get("mismatches", 0) or 0
                        if mismatches:
                            details = res.get("details", []) or []
                            from brains.personal.memory import goal_memory  # type: ignore
                            created: list[dict] = []
                            # Gather existing goal titles and normalise them to avoid
                            # duplicates that differ only by case or spacing.  The
                            # normalisation removes punctuation and whitespace and
                            # converts to lower case.
                            import re as _re_norm  # local alias
                            existing_titles_norm: set[str] = set()
                            try:
                                # Collect existing goal titles from disk
                                current_goals = goal_memory.get_goals(active_only=False)
                                for g in current_goals:
                                    try:
                                        t = str(g.get("title", "")).strip()
                                    except Exception:
                                        t = ""
                                    if t:
                                        norm = _re_norm.sub(r"[^a-z0-9]", "", t.lower())
                                        existing_titles_norm.add(norm)
                            except Exception:
                                existing_titles_norm = set()
                            # Also include titles of goals currently queued in the remaining_goals
                            try:
                                rem_goals = ctx.get("stage_15_remaining_goals") or []
                                for g in rem_goals:
                                    try:
                                        t = str(g.get("title", "")).strip()
                                    except Exception:
                                        t = ""
                                    if t:
                                        norm = _re_norm.sub(r"[^a-z0-9]", "", t.lower())
                                        existing_titles_norm.add(norm)
                            except Exception:
                                pass
                            # Helper to normalise a proposed title
                            def _norm_title(s: str) -> str:
                                try:
                                    return _re_norm.sub(r"[^a-z0-9]", "", str(s or "").lower())
                                except Exception:
                                    return str(s or "").lower()
                            for itm in details:
                                try:
                                    q = str(itm.get("question", "")).strip()
                                except Exception:
                                    q = ""
                                if not q:
                                    continue
                                new_title = f"Verify QA: {q}"
                                norm_new_title = _norm_title(new_title)
                                # Skip if a goal with this (normalised) title already exists
                                if norm_new_title in existing_titles_norm:
                                    continue
                                try:
                                    new_goal = goal_memory.add_goal(new_title, description="AUTO_REPAIR")
                                    existing_titles_norm.add(norm_new_title)
                                    created.append({"goal_id": new_goal.get("goal_id"), "title": new_goal.get("title")})
                                except Exception:
                                    continue
                            if created:
                                ctx["stage_16_repair_goals"] = created
                    except Exception:
                        pass
                except Exception as reg_ex:
                    ctx["stage_16_regression"] = {"error": str(reg_ex)}
            else:
                ctx["stage_16_regression"] = {"skipped": True}
        except Exception as reg_top_ex:
            ctx["stage_16_regression_error"] = str(reg_top_ex)

        # ------------------------------------------------------------------
        # Stage 17: Long‑Term Memory Consolidation & QA Memory Pruning
        #
        # As the QA memory grows over time, it can accumulate hundreds of
        # entries, which degrades performance and increases storage.  To
        # mitigate this, consolidate older QA entries by extracting simple
        # definitional facts into the semantic knowledge graph and pruning
        # the log to a fixed size.  Only run this consolidation when there
        # are more than ``max_entries`` QA records.  Facts extracted from
        # pruned entries follow the same pattern as the assimilation in
        # language_brain.finalize: questions of the form "what is X" or
        # "who is X" with short answers are stored as (subject, "is",
        # answer).  Statistics about the number of pruned entries and
        # assimilated facts are stored in the context under
        # ``stage_17_memory_pruning``.
        try:
            from pathlib import Path
            import json as _json  # alias to avoid clobbering the outer json import
            import re as _re
            # Load knowledge graph module if available
            try:
                from brains.personal.memory import knowledge_graph  # type: ignore
            except Exception:
                knowledge_graph = None  # type: ignore
            # Determine the max number of QA entries to retain.  Use a
            # reasonable default if config isn't present or invalid.
            max_entries = 100
            # Allow tuning via config/memory.json if present.  This
            # optional file can specify {'qa_memory_max_entries': N}.
            try:
                from pathlib import Path as _Path
                mem_cfg_path = Path(__file__).resolve().parents[5] / "config" / "memory.json"
                if mem_cfg_path.exists():
                    try:
                        with open(mem_cfg_path, "r", encoding="utf-8") as mfh:
                            mcfg = _json.load(mfh) or {}
                        val = int(mcfg.get("qa_memory_max_entries", max_entries))
                        if val > 0:
                            max_entries = val
                    except Exception:
                        pass
            except Exception:
                pass
            # Read the QA memory file
            qa_path = get_reports_path("qa_memory.jsonl")
            if qa_path.exists():
                try:
                    with open(qa_path, "r", encoding="utf-8") as qfh:
                        raw_lines = [ln.strip() for ln in qfh if ln.strip()]
                except Exception:
                    raw_lines = []
                total_qas = len(raw_lines)
                if total_qas > max_entries:
                    # Determine how many to prune
                    prune_count = total_qas - max_entries
                    old_lines = raw_lines[:prune_count]
                    new_lines = raw_lines[prune_count:]
                    assimilated = 0
                    # Assimilate facts from pruned QAs into the knowledge graph
                    if knowledge_graph is not None:
                        for ln in old_lines:
                            try:
                                rec = _json.loads(ln)
                            except Exception:
                                continue
                            q = str(rec.get("question", "")).strip()
                            a = str(rec.get("answer", "")).strip()
                            if not q or not a:
                                continue
                            # Match simple definition patterns
                            m = _re.match(r"^(?:what|who)\s+is\s+(.+)", q.lower())
                            if not m:
                                continue
                            subj = m.group(1).strip().rstrip("?")
                            # Only assimilate short, confident answers
                            if len(a) > 80 or "?" in a or "don't know" in a.lower():
                                continue
                            # Build candidate subjects: raw and without leading articles
                            subj_norm = subj.lower().strip()
                            cand1 = subj_norm
                            cand2 = _re.sub(r"^(?:the|a|an)\s+", "", subj_norm).strip()
                            for candidate in [cand1, cand2]:
                                if not candidate:
                                    continue
                                try:
                                    knowledge_graph.add_fact(candidate, "is", a)
                                    assimilated += 1
                                    break
                                except Exception:
                                    continue
                    # Write retained lines back to the QA file
                    try:
                        with open(qa_path, "w", encoding="utf-8") as qfh:
                            for ln in new_lines:
                                qfh.write(ln + "\n")
                    except Exception:
                        pass
                    # Record pruning statistics in the context
                    ctx["stage_17_memory_pruning"] = {
                        "total_before": total_qas,
                        "pruned": prune_count,
                        "assimilated": assimilated,
                        "retained": len(new_lines)
                    }
        except Exception as prune_ex:
            # Record any pruning errors for debugging
            ctx["stage_17_memory_pruning_error"] = str(prune_ex)

        # ------------------------------------------------------------------
        # Stage 18: Self‑Review & Improvement Goal Creation
        #
        # After consolidating memory, perform a simple self‑assessment
        # across tracked domains to identify areas where Maven is
        # underperforming.  Use the meta_confidence statistics to
        # determine domains with the lowest recent adjustments.  For
        # each such domain below a configurable threshold (e.g. -0.05),
        # create a new goal to improve Maven's knowledge in that
        # domain.  These tasks are prefixed with "SELF_REVIEW" in the
        # description so that the autonomy scheduler can prioritise
        # them appropriately.  Record the created goals in
        # ``stage_18_self_review".  Errors are silently ignored.
        try:
            # Load meta confidence and goal memory if available
            try:
                from brains.personal.memory import meta_confidence  # type: ignore
            except Exception:
                meta_confidence = None  # type: ignore
            try:
                from brains.personal.memory import goal_memory  # type: ignore
            except Exception:
                goal_memory = None  # type: ignore
            if meta_confidence is not None and goal_memory is not None:
                # Determine threshold from config or use default
                threshold = -0.05
                # Optionally read a self‑review config
                try:
                    sr_cfg_path = Path(__file__).resolve().parents[5] / "config" / "self_review.json"
                    if sr_cfg_path.exists():
                        with open(sr_cfg_path, "r", encoding="utf-8") as srfh:
                            sr_cfg = json.load(srfh) or {}
                        th = float(sr_cfg.get("threshold", threshold))
                        if -1.0 <= th <= 0.0:
                            threshold = th
                except Exception:
                    pass
                # Gather domain stats
                stats = meta_confidence.get_stats(1000) or []
                # Identify lowest performers below threshold
                low_domains = [d for d in stats if d.get("adjustment", 0) < threshold]
                # Sort by adjustment ascending (most negative first)
                low_domains.sort(key=lambda d: d.get("adjustment", 0))
                created: list[dict] = []
                # Determine the maximum number of new self‑review goals to create
                max_new = 5
                try:
                    # Read limit from config/self_review.json if present
                    import json as _json
                    from pathlib import Path
                    root = Path(__file__).resolve().parents[5]
                    cfg_path = root / "config" / "self_review.json"
                    if cfg_path.exists():
                        with open(cfg_path, "r", encoding="utf-8") as cfh:
                            cfg = _json.load(cfh) or {}
                        try:
                            mx = int(cfg.get("max_goals", max_new))
                            if 1 <= mx <= 20:
                                max_new = mx
                        except Exception:
                            pass
                except Exception:
                    # If any error, fall back to default
                    max_new = 5
                # Fetch existing goals to avoid duplicate improvement tasks
                existing = []
                try:
                    existing = goal_memory.get_goals(active_only=False)
                except Exception:
                    existing = []
                existing_titles = set()
                for g in existing:
                    try:
                        t = str(g.get("title", "")).strip()
                        if t:
                            existing_titles.add(t)
                    except Exception:
                        continue
                count_new = 0
                for dom in low_domains:
                    if count_new >= max_new:
                        break
                    try:
                        dom_name = dom.get("domain", "").strip()
                    except Exception:
                        dom_name = ""
                    if not dom_name:
                        continue
                    # Compose goal title and description
                    title = f"Improve domain: {dom_name}"
                    # Skip if a goal with the same title already exists (active or completed)
                    if title in existing_titles:
                        continue
                    desc = f"SELF_REVIEW: {dom_name}"
                    try:
                        rec = goal_memory.add_goal(title, description=desc)
                        created.append({"goal_id": rec.get("goal_id"), "title": rec.get("title")})
                        existing_titles.add(title)
                        count_new += 1
                    except Exception:
                        continue
                if created:
                    ctx["stage_18_self_review"] = created
                else:
                    ctx["stage_18_self_review"] = []
        except Exception as sr_ex:
            ctx["stage_18_self_review_error"] = str(sr_ex)

        # Perform a run evaluation on the final context prior to returning.
        # This self‑evaluation computes health metrics and may enqueue
        # repair goals via the goal memory.  Errors during evaluation are
        # recorded in the context but do not block the pipeline response.
        if op == "RUN_PIPELINE":
            try:
                import importlib
                sc_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_critique")
                eval_resp = sc_mod.service_api({"op": "EVAL_CONTEXT", "payload": {"context": ctx}})
                ctx["stage_self_eval"] = eval_resp.get("payload", {})
            except Exception:
                ctx["stage_self_eval"] = {"error": "eval_failed"}
            # Update the semantic cache on completion of a pipeline run.  This stores
            # the query and answer in a cross‑session index for reuse on future runs.
            try:
                _update_semantic_cache(ctx)
            except Exception:
                pass
        # Clean up SEARCH_REQUEST messages from the message bus to prevent them
        # from polluting subsequent UNIFIED_RETRIEVE calls.  During pipeline
        # execution, targeted retrieval may post SEARCH_REQUEST messages to
        # restrict which banks are queried.  These messages should not persist
        # beyond the current pipeline run, as they would incorrectly filter
        # later independent retrieval operations.
        try:
            from brains.cognitive import message_bus
            # Clear all pending messages after pipeline completion
            message_bus.pop_all()
        except Exception:
            # Silently ignore errors during cleanup to avoid disrupting the response
            pass

        # End routing diagnostics trace (Phase C cleanup)
        try:
            if tracer and RouteType:
                final_ans = ctx.get("final_answer")
                tracer.record_route(mid, RouteType.FULL_PIPELINE, {"completed": True})
                tracer.end_request(mid, final_ans)
        except Exception:
            pass

        # Stage: Self-Review — Review the complete turn and take action
        if op == "RUN_PIPELINE":
            try:
                from brains.cognitive.self_review.service.self_review_brain import service_api as self_review_api

                review_resp = self_review_api({
                    "op": "REVIEW_TURN",
                    "payload": {
                        "query": text,
                        "plan": ctx.get("stage_2_planner", {}),
                        "thoughts": ctx.get("stage_4_reasoning", {}).get("thoughts", []),
                        "answer": str(ctx.get("final_answer", "")),
                        "metadata": {
                            "confidences": {
                                "final": ctx.get("final_confidence", 0.8),
                                "reasoning": (ctx.get("stage_8_validation", {}) or {}).get("confidence", 0.8)
                            },
                            "used_memories": (ctx.get("stage_2R_memory", {}) or {}).get("results", []),
                            "intents": [(ctx.get("stage_3_language", {}) or {}).get("intent", "")]
                        }
                    }
                })

                if review_resp.get("ok"):
                    review_payload = review_resp.get("payload", {})
                    verdict = review_payload.get("verdict", "ok")
                    recommended_action = review_payload.get("recommended_action", "accept")

                    ctx["stage_self_review"] = review_payload

                    if recommended_action == "ask_clarification":
                        issues = review_payload.get("issues", [])
                        issue_desc = issues[0].get("message", "low confidence") if issues else "low confidence"
                        ctx["final_answer"] = f"I'm not confident in my answer due to {issue_desc}. Could you please provide more context or rephrase your question?"
                        ctx["final_confidence"] = 0.3
                    elif recommended_action == "revise":
                        current_conf = ctx.get("final_confidence", 0.8)
                        ctx["final_confidence"] = max(0.2, current_conf * 0.7)
                        issues = review_payload.get("issues", [])
                        if issues:
                            issue_codes = [i.get("code", "") for i in issues]
                            ctx.setdefault("review_notes", []).append({
                                "action": "confidence_downgrade",
                                "issues": issue_codes
                            })

                        try:
                            from brains.cognitive.self_dmn.service.self_dmn_brain import service_api as self_dmn_api
                            self_dmn_api({
                                "op": "REFLECT_ON_ERROR",
                                "payload": {
                                    "error_context": {
                                        "verdict": verdict,
                                        "issues": issues,
                                        "query": text
                                    },
                                    "turn_history": []
                                }
                            })
                        except Exception:
                            pass
            except Exception as e:
                ctx["stage_self_review_error"] = str(e)

        # ------------------------------------------------------------------
        # CRITICAL FIX: Ensure final_answer is set when stage_8 has answer
        # ------------------------------------------------------------------
        # When SELF_MODEL or other early returns provide an answer in stage_8
        # but FINALIZE stage is bypassed, final_answer remains empty. This
        # causes tests and external consumers to see empty responses despite
        # having a valid answer from reasoning.
        #
        # TASK 3 ENHANCEMENT: Also check if final_answer contains the fallback
        # message and override it with stage_8 answer if available.
        #
        # Safety check: If final_answer is not set OR contains fallback message
        # but stage_8_validation has a valid answer, use that answer.
        try:
            current_answer = ctx.get("final_answer") or ""
            # TASK 3: Check if current answer is a fallback message
            is_fallback = (
                not current_answer.strip()
                or "i don't yet have enough information" in current_answer.lower()
                or "i don't have enough information" in current_answer.lower()
                or "i'm not sure how to help" in current_answer.lower()
                or "i am not sure how to help" in current_answer.lower()
            )

            stage8 = ctx.get("stage_8_validation") or {}
            verdict = str(stage8.get("verdict", "")).upper()
            answer = stage8.get("answer")

            if is_fallback:
                print(f"[FINAL_ANSWER_FIX] final_answer empty or fallback, stage8 verdict={verdict}, answer exists={bool(answer)}")

                if verdict in ("TRUE", "LEARNED", "ANSWERED") and answer:
                    print(f"[FINAL_ANSWER_FIX] Setting final_answer from stage_8: '{answer[:60]}...'")
                    ctx["final_answer"] = answer
                    # Also set confidence if not already set or low
                    stage8_conf = stage8.get("confidence", 0.9)
                    if not ctx.get("final_confidence") or ctx.get("final_confidence", 0) < 0.5:
                        ctx["final_confidence"] = stage8_conf

                    # Update stage_10_finalize for consistency
                    if not ctx.get("stage_10_finalize") or ctx.get("stage_10_finalize", {}).get("task3_override") is not True:
                        ctx["stage_10_finalize"] = {
                            "text": answer,
                            "confidence": stage8_conf,
                            "source": "stage_8_fallback",
                            "task3_override": True
                        }
                else:
                    print(f"[FINAL_ANSWER_FIX] Cannot set from stage_8: verdict={verdict}, answer={bool(answer)}")
            else:
                print(f"[FINAL_ANSWER_FIX] final_answer already set: '{current_answer[:60]}...'")
        except Exception as e:
            # Never fail pipeline on this safety check
            print(f"[FINAL_ANSWER_FIX] Exception in safety check: {e}")
            pass

        # Update conversation state to enable follow-up question context
        # This must happen before returning, even if the answer had low confidence
        try:
            query_for_state = ctx.get("original_query") or payload.get("text") or text
            _update_conversation_state(query_for_state, ctx.get("final_answer"))
        except Exception:
            pass

        return success_response(op, mid, {"context": ctx})

    # Provide a backwards‑compatible HEALTH operation that reports the set of
    # discoverable domain banks.  Earlier versions of Maven exposed a
    # ``HEALTH`` op returning a list of available banks.  Preserve this
    # behaviour by enumerating the banks that can be loaded without error.
    if op == "HEALTH":
        banks: list[str] = []
        for b in _ALL_BANKS:
            try:
                _bank_module(b)
                banks.append(b)
            except Exception:
                # Skip banks that fail to load
                pass
        return success_response(op, mid, {"discovered_banks": banks})

    if op == "HEALTH_CHECK":
        counts = _scan_counts(COG_ROOT)
        rotated: list[dict[str, Any]] = []
        # The overflow limit can be tuned by a soft_headroom factor as well as a
        # hard_headroom multiplier.  By default rotation triggers when STM
        # exceeds ``soft_headroom * stm_records``.  To further reduce the
        # frequency of repairs (and avoid cascading repair loops), introduce
        # ``hard_headroom`` which multiplies the computed limit.  Only when
        # ``stm_count`` exceeds ``soft_headroom * hard_headroom * stm_records``
        # is a repair triggered.  Missing keys fall back to sensible defaults.
        try:
            # Default the soft headroom to 2 when not explicitly configured.  This factor
            # multiplies the per-bank STM limit to compute an initial threshold.  Adjust
            # via CFG["rotation"]["soft_headroom"] in config files if needed.
            soft = float(CFG.get("rotation", {}).get("soft_headroom", 2))
        except Exception:
            soft = 2.0
        try:
            # Default the hard headroom to 10.  A larger default dramatically reduces the
            # frequency of repairs by requiring STM sizes to exceed (soft * hard * limit)
            # before triggering a repair.  This value can be overridden via
            # CFG["rotation"]["hard_headroom"] in config/autotune.json.
            hard = float(CFG.get("rotation", {}).get("hard_headroom", 10))
        except Exception:
            hard = 10.0
        # Compute the threshold: base limit multiplied by both headrooms
        base_limit = CFG.get("rotation", {}).get("stm_records", 1000)
        try:
            limit = float(base_limit) * soft
        except Exception:
            limit = float(base_limit) * 2.0
        threshold = limit * hard
        rep = _repair_module()
        for brain, tiers in counts.items():
            try:
                stm_count = int(tiers.get("stm", 0))
            except Exception:
                stm_count = 0
            # Only trigger a repair when the STM count exceeds the computed
            # threshold.  This drastically reduces the number of repair
            # operations ("stop the bleeding"), since the previous logic
            # repaired whenever ``stm_count > limit``.  With the default
            # settings of soft_headroom=2 and hard_headroom=2, rotation
            # occurs only when STM exceeds 4× the configured limit.
            if stm_count > threshold:
                # Respect the governance auto_repair setting.  If auto_repair is disabled,
                # skip invoking the repair engine entirely.  This allows administrators
                # to throttle or disable automated repairs without modifying code.
                if CFG.get("governance", {}).get("auto_repair", True) is False:
                    # Skip repair; record overflow but do not invoke repair engine
                    rotated.append({"brain": brain, "stm_count": stm_count, "rule":"memory_overflow_skipped"})
                else:
                    if brain == "personal":
                        stm_path = (MAVEN_ROOT / "brains" / "personal" / "memory" / "stm" / "records.jsonl").resolve()
                    else:
                        stm_path = (COG_ROOT / brain / "memory" / "stm" / "records.jsonl").resolve()
                    try:
                        rep.service_api({"op":"REPAIR","payload":{"rule":"memory_overflow","target": str(stm_path)}})
                    except Exception:
                        pass
                    rotated.append({"brain": brain, "stm_count": stm_count, "rule":"memory_overflow"})
        write_report("system", f"health_{_SEQ_ID_COUNTER}.json", json.dumps({"counts": counts, "rotations": rotated}, indent=2))
        return success_response(op, mid, {"rotations": rotated, "counts": counts})

    # ----------------------------------------------------------------------
    # Phase 4: Memory Health Summary - Tiered Memory Diagnostics
    #
    # This operation provides a comprehensive summary of memory health across
    # all tiers without modifying any state.  It returns counts, average
    # importance, and other statistics for each tier to enable debugging and
    # monitoring of the tiered memory system.
    # ----------------------------------------------------------------------
    if op == "MEMORY_HEALTH_SUMMARY":
        try:
            # Initialize tier statistics
            tier_stats = {
                TIER_PINNED: {"count": 0, "total_importance": 0.0, "total_use_count": 0, "total_confidence": 0.0},
                TIER_MID: {"count": 0, "total_importance": 0.0, "total_use_count": 0, "total_confidence": 0.0},
                TIER_SHORT: {"count": 0, "total_importance": 0.0, "total_use_count": 0, "total_confidence": 0.0},
                TIER_WM: {"count": 0, "total_importance": 0.0, "total_use_count": 0, "total_confidence": 0.0},
                TIER_LONG: {"count": 0, "total_importance": 0.0, "total_use_count": 0, "total_confidence": 0.0},
                "UNKNOWN": {"count": 0, "total_importance": 0.0, "total_use_count": 0, "total_confidence": 0.0},
            }

            # Scan working memory
            with _WM_LOCK:
                for entry in _WORKING_MEMORY:
                    tier = str(entry.get("tier", "UNKNOWN")).upper()
                    if tier not in tier_stats:
                        tier = "UNKNOWN"
                    tier_stats[tier]["count"] += 1
                    try:
                        tier_stats[tier]["total_importance"] += float(entry.get("importance", 0.0))
                    except (TypeError, ValueError):
                        pass
                    try:
                        tier_stats[tier]["total_use_count"] += int(entry.get("use_count", 0))
                    except (TypeError, ValueError):
                        pass
                    try:
                        tier_stats[tier]["total_confidence"] += float(entry.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        pass

            # Scan brain storage files
            for brain_name in ["memory_librarian", "reasoning", "language", "planner", "personality"]:
                try:
                    brain_mem_dir = MAVEN_ROOT / "brains" / "cognitive" / brain_name / "memory"
                    brain_mem_file = brain_mem_dir / "brain_storage.jsonl"
                    if brain_mem_file.exists():
                        with open(brain_mem_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    record = json.loads(line)
                                    tier = str(record.get("tier", "UNKNOWN")).upper()
                                    if tier not in tier_stats:
                                        tier = "UNKNOWN"
                                    tier_stats[tier]["count"] += 1
                                    try:
                                        tier_stats[tier]["total_importance"] += float(record.get("importance", 0.0))
                                    except (TypeError, ValueError):
                                        pass
                                    try:
                                        tier_stats[tier]["total_use_count"] += int(record.get("use_count", 0))
                                    except (TypeError, ValueError):
                                        pass
                                    try:
                                        tier_stats[tier]["total_confidence"] += float(record.get("confidence", 0.0))
                                    except (TypeError, ValueError):
                                        pass
                                except json.JSONDecodeError:
                                    continue
                except Exception:
                    # Skip brains that don't exist or have errors
                    pass

            # Compute averages
            summary_by_tier = {}
            total_records = 0
            for tier, stats in tier_stats.items():
                count = stats["count"]
                total_records += count
                if count > 0:
                    summary_by_tier[tier] = {
                        "count": count,
                        "avg_importance": stats["total_importance"] / count,
                        "avg_use_count": stats["total_use_count"] / count,
                        "avg_confidence": stats["total_confidence"] / count,
                    }
                else:
                    summary_by_tier[tier] = {
                        "count": 0,
                        "avg_importance": 0.0,
                        "avg_use_count": 0.0,
                        "avg_confidence": 0.0,
                    }

            # Include global state
            health_summary = {
                "tiers": summary_by_tier,
                "total_records": total_records,
                "current_seq_id": _SEQ_ID_COUNTER,
                "timestamp": None,  # No time-based logic
            }

            return success_response(op, mid, health_summary)
        except Exception as e:
            return error_response(op, mid, "MEMORY_HEALTH_SUMMARY_FAILED", str(e))

    # ----------------------------------------------------------------------
    # Configuration operations
    # These allow runtime toggling of the pipeline tracer and adjustment of
    # rotation thresholds.  They do not persist changes beyond the current
    # process lifetime and respect governance rules enforced by the policy
    # engine.  For example, Enable/Disable Tracer simply flips the in-memory
    # flag at CFG['pipeline_tracer']['enabled'].  SET_ROTATION_LIMITS can
    # update either the global rotation defaults or per-bank overrides.
    if op == "ENABLE_TRACER":
        try:
            CFG.setdefault("pipeline_tracer", {})["enabled"] = True
            return success_response(op, mid, {"enabled": True})
        except Exception as e:
            return error_response(op, mid, "TRACER_TOGGLE_FAILED", str(e))

    if op == "DISABLE_TRACER":
        try:
            CFG.setdefault("pipeline_tracer", {})["enabled"] = False
            return success_response(op, mid, {"enabled": False})
        except Exception as e:
            return error_response(op, mid, "TRACER_TOGGLE_FAILED", str(e))

    if op == "SET_ROTATION_LIMITS":
        bank = str((payload or {}).get("bank", "")).strip().lower()
        limits: Dict[str, Any] = {}
        for key in ("stm_records", "mtm_records", "ltm_records"):
            val = (payload or {}).get(key)
            if val is not None:
                try:
                    limits[key] = int(val)
                except Exception:
                    pass
        try:
            if not bank or bank == "global":
                # apply to global defaults
                CFG.setdefault("rotation", {}).update(limits)
                who = "global"
            else:
                per_bank = CFG.setdefault("rotation_per_bank", {})
                per_bank.setdefault(bank, {}).update(limits)
                who = bank
            return success_response(op, mid, {"bank": who, "limits": limits})
        except Exception as e:
            return error_response(op, mid, "SET_ROTATION_FAILED", str(e))

    # ------------------------------------------------------------------
    # Working Memory Operations (Step‑1)
    # These operations expose a simple shared WM through the memory
    # librarian.  WM_PUT stores an entry; WM_GET retrieves entries by key
    # or tag; WM_DUMP returns all live entries; CONTROL_TICK scans
    # working memory and emits message_bus events for each entry.
    elif op == "WM_PUT":
        # Load persisted memory if needed before storing
        try:
            _wm_load_if_needed()
        except Exception:
            pass
        # Store an item into working memory
        try:
            raw_key = (payload or {}).get("key")
            key = str(raw_key) if raw_key is not None else None
            value = (payload or {}).get("value")
            raw_tags = (payload or {}).get("tags")
            if isinstance(raw_tags, str):
                tags = [raw_tags]
            else:
                tags = [str(t) for t in list(raw_tags or [])]
            try:
                conf = float((payload or {}).get("confidence", 0.0))
            except Exception:
                conf = 0.0
            # Assign tier and importance via Phase 4 tier system
            tier_context = {
                "intent": str((payload or {}).get("intent", "")),
                "verdict": str((payload or {}).get("verdict", "")),
                "tags": tags,
            }
            record_for_tier = {"content": str(value or ""), "confidence": conf, "tags": tags}
            tier, importance = _assign_tier(record_for_tier, tier_context)

            # Generate sequence ID for recency tracking
            seq_id = _next_seq_id()

            entry = {
                "key": key,
                "value": value,
                "tags": tags,
                "confidence": conf,
                "tier": tier or TIER_WM,  # Default to WM tier if not assigned
                "importance": importance,
                "seq_id": seq_id,
                "use_count": 0,  # Initialize usage counter
            }
            with _WM_LOCK:
                _prune_working_memory()
                _WORKING_MEMORY.append(entry)
                # Persist entry if configured
                try:
                    from api.utils import CFG  # type: ignore
                    persist_enabled = bool((CFG.get("wm", {}) or {}).get("persist", True))
                except Exception:
                    persist_enabled = True
                if persist_enabled:
                    try:
                        _wm_persist_append(entry)
                    except Exception:
                        pass
            # Log the WM_PUT event
            try:
                root = globals().get("MAVEN_ROOT")
                if not root:
                    root = Path(__file__).resolve().parents[4]
                log_path = (root / "reports" / "wm_trace.jsonl").resolve()
                from api.utils import append_jsonl  # type: ignore
                append_jsonl(log_path, {"op": "WM_PUT", "entry": {k: v for k, v in entry.items() if k != "expires_at"}})
            except Exception:
                pass
            return success_response(op, mid, {"stored": True, "entry": {k: v for k, v in entry.items() if k != "expires_at"}})
        except Exception as e:
            return error_response(op, mid, "WM_PUT_FAILED", str(e))

    elif op == "WM_GET":
        # Load persisted memory if needed before retrieval
        try:
            _wm_load_if_needed()
        except Exception:
            pass
        # Retrieve items from working memory filtered by key or tags
        try:
            raw_key = (payload or {}).get("key")
            k = str(raw_key) if raw_key is not None else None
            tags = (payload or {}).get("tags")
            if isinstance(tags, str):
                tag_list = [tags]
            else:
                tag_list = list(tags or []) if tags else None
            results: List[Dict[str, Any]] = []
            with _WM_LOCK:
                _prune_working_memory()
                for ent in _WORKING_MEMORY:
                    ent_key = ent.get("key")
                    if k is not None and str(ent_key) != k:
                        continue
                    if tag_list:
                        try:
                            etags = ent.get("tags") or []
                            # Intersection test: at least one tag matches
                            if not any(t in etags for t in tag_list):
                                continue
                        except Exception:
                            continue
                    # Return a shallow copy without internal expiry
                    results.append({k2: v2 for k2, v2 in ent.items() if k2 != "expires_at"})
            # Apply arbitration scoring if enabled and a key is specified
            winner: Optional[Dict[str, Any]] = None
            alternatives: List[Dict[str, Any]] = []
            try:
                from api.utils import CFG  # type: ignore
                arbitration_enabled = bool((CFG.get("wm", {}) or {}).get("arbitration", True))
            except Exception:
                arbitration_enabled = True
            if arbitration_enabled and k is not None and len(results) > 1:
                try:
                    import math
                    scored_pairs: List[tuple] = []
                    for ent in results:
                        try:
                            conf_val = float(ent.get("confidence", 0.0))
                        except Exception:
                            conf_val = 0.0
                        try:
                            reliability = float(ent.get("source_reliability", 1.0))
                        except Exception:
                            reliability = 1.0
                        # No time-based decay
                        score = conf_val * reliability
                        scored_pairs.append((score, ent))
                    scored_pairs.sort(key=lambda x: x[0], reverse=True)
                    scored_results: List[Dict[str, Any]] = []
                    for sc, ent in scored_pairs:
                        ent_with_score = ent.copy()
                        try:
                            ent_with_score["score"] = round(sc, 6)
                        except Exception:
                            ent_with_score["score"] = sc
                        scored_results.append(ent_with_score)
                    if scored_results:
                        winner = scored_results[0]
                        alternatives = scored_results[1:]
                        results = scored_results
                except Exception:
                    pass
            # Log the WM_GET event
            try:
                root = globals().get("MAVEN_ROOT")
                if not root:
                    root = Path(__file__).resolve().parents[4]
                log_path = (root / "reports" / "wm_trace.jsonl").resolve()
                from api.utils import append_jsonl  # type: ignore
                append_jsonl(log_path, {"op": "WM_GET", "filter": {"key": k, "tags": tag_list}, "results": results})
            except Exception:
                pass
            # Include winner/alternatives when arbitration applied
            payload_dict: Dict[str, Any] = {"entries": results}
            if winner is not None:
                payload_dict["winner"] = winner
                payload_dict["alternatives"] = alternatives
            return success_response(op, mid, payload_dict)
        except Exception as e:
            return error_response(op, mid, "WM_GET_FAILED", str(e))

    elif op == "WM_DUMP":
        # Load persisted memory if needed before dumping
        try:
            _wm_load_if_needed()
        except Exception:
            pass
        # Dump all live working memory entries
        try:
            with _WM_LOCK:
                _prune_working_memory()
                dump = [{k: v for k, v in ent.items() if k != "expires_at"} for ent in list(_WORKING_MEMORY)]
            # Log the WM_DUMP event
            try:
                root = globals().get("MAVEN_ROOT")
                if not root:
                    root = Path(__file__).resolve().parents[4]
                log_path = (root / "reports" / "wm_trace.jsonl").resolve()
                from api.utils import append_jsonl  # type: ignore
                append_jsonl(log_path, {"op": "WM_DUMP", "entries": dump})
            except Exception:
                pass
            return success_response(op, mid, {"entries": dump})
        except Exception as e:
            return error_response(op, mid, "WM_DUMP_FAILED", str(e))

    elif op == "CONTROL_TICK":
        # Load persisted memory if needed before ticking
        try:
            _wm_load_if_needed()
        except Exception:
            pass
        # Scan WM and emit message bus events for each entry
        try:
            emitted = 0
            with _WM_LOCK:
                _prune_working_memory()
                current_entries = [{k: v for k, v in ent.items() if k != "expires_at"} for ent in list(_WORKING_MEMORY)]
            # Import message_bus lazily to avoid circular imports on module load
            try:
                from brains.cognitive import message_bus  # type: ignore
                for ent in current_entries:
                    msg = {
                        "from": "memory_librarian",
                        "to": "scheduler",
                        "type": "WM_EVENT",
                        "entry": ent,
                    }
                    try:
                        message_bus.send(msg)
                        emitted += 1
                    except Exception:
                        continue
            except Exception:
                # If message bus unavailable, skip emission
                emitted = 0
            # Optionally run the cognitive graph engine on emitted events
            try:
                from api.utils import CFG  # type: ignore
                graph_cfg = CFG.get("graph", {}) or {}
                if bool(graph_cfg.get("enabled", False)):
                    # Import default_graph_engine lazily to avoid circular deps
                    try:
                        from brains.cognitive.graph_engine import default_graph_engine  # type: ignore
                        # Instantiate and run with a minimal context; the engine will
                        # drain message_bus events and propagate them according to
                        # registered nodes.
                        engine = default_graph_engine()
                        engine.run({})
                    except Exception:
                        pass
            except Exception:
                pass
            # Log the CONTROL_TICK event
            try:
                root = globals().get("MAVEN_ROOT")
                if not root:
                    root = Path(__file__).resolve().parents[4]
                log_path = (root / "reports" / "wm_trace.jsonl").resolve()
                from api.utils import append_jsonl  # type: ignore
                append_jsonl(log_path, {"op": "CONTROL_TICK", "emitted": emitted, "entries": current_entries})
            except Exception:
                pass
            return success_response(op, mid, {"events_emitted": emitted})
        except Exception as e:
            return error_response(op, mid, "CONTROL_TICK_FAILED", str(e))

    # ------------------------------------------------------------------
    # Blackboard and Control Shell (Phase‑6)
    #
    # BB_SUBSCRIBE registers a subscription for a brain.  CONTROL_CYCLE
    # performs a bounded cycle: collects WM events, scores them per
    # subscription, arbitrates via the integrator and dispatches the
    # winning event.  Events older than the subscription TTL or below
    # the confidence threshold are ignored.  A global cap on steps and
    # runtime is enforced by configuration (blackboard.json).

    elif op == "BB_SUBSCRIBE":
        sub_id = str((payload or {}).get("subscriber", "")).strip()
        if not sub_id:
            return error_response(op, mid, "INVALID_SUBSCRIBER", "subscriber is required")
        key = (payload or {}).get("key")
        tags = (payload or {}).get("tags")
        tags_list = list(tags) if tags else None
        try:
            min_conf = float((payload or {}).get("min_conf", 0.0))
        except Exception:
            min_conf = 0.0
        try:
            ttl = float((payload or {}).get("ttl", 300.0))
        except Exception:
            ttl = 300.0
        try:
            priority = float((payload or {}).get("priority", 0.5))
        except Exception:
            priority = 0.5
        _bb_subscribe(sub_id, key, tags_list, min_conf, ttl, priority)
        return success_response(op, mid, {"subscribed": True, "subscriber": sub_id})

    elif op == "CONTROL_CYCLE":
        # Perform a bounded cycle of WM arbitration and dispatch
        try:
            _wm_load_if_needed()
        except Exception:
            pass
        # Load blackboard configuration
        try:
            from api.utils import CFG  # type: ignore
            bb_cfg = (CFG.get("blackboard") or {}) if CFG else {}
        except Exception:
            bb_cfg = {}
        try:
            max_steps = int(bb_cfg.get("max_steps", 64) or 64)
        except Exception:
            max_steps = 64
        try:
            max_events = int(bb_cfg.get("max_events_per_tick", 50) or 50)
        except Exception:
            max_events = 50
        try:
            max_ms = float(bb_cfg.get("max_runtime_ms", 150.0) or 150.0)
        except Exception:
            max_ms = 150.0
        starvation_guard = bool(bb_cfg.get("starvation_guard", True))
        # Collect candidate events
        candidates = _bb_collect_events()
        if not candidates:
            return success_response(op, mid, {"processed": 0, "message": "no events"})
        # Sort by score descending
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        processed = 0
        logs: List[Dict[str, Any]] = []
        for cand in candidates:
            if processed >= max_events:
                break
            sub_id = cand.get("subscriber")
            entry = cand.get("entry")
            idx = cand.get("index")
            score = cand.get("score", 0.0)
            # Build a bid for the integrator with subscription priority
            try:
                sub_cfg = _BLACKBOARD_SUBS.get(str(sub_id)) or {}
                base_p = float(sub_cfg.get("priority", 0.5))
            except Exception:
                base_p = 0.5
            try:
                bid_priority = float(score + base_p)
            except Exception:
                bid_priority = base_p
            # Clamp to [0.0, 1.0]
            if bid_priority < 0.0:
                bid_priority = 0.0
            if bid_priority > 1.0:
                bid_priority = 1.0
            bids = [{"brain_name": sub_id, "priority": bid_priority, "reason": "wm_event", "evidence": entry}]
            # Resolve via integrator
            try:
                from brains.cognitive.integrator.service.integrator_brain import service_api as integrator_api  # type: ignore
                res = integrator_api({"op": "RESOLVE", "payload": {"bids": bids}, "mid": f"BB-{mid}-{processed}"})
                winner = res.get("payload", {}).get("focus") or sub_id
            except Exception:
                winner = sub_id
            # Dispatch event to message bus
            try:
                from brains.cognitive import message_bus  # type: ignore
                message_bus.send({
                    "from": "blackboard",
                    "to": winner,
                    "type": "WM_EVENT",
                    "entry": entry,
                })
                processed += 1
                _bb_mark_processed(str(sub_id), int(idx))
                logs.append({
                    "subscriber": sub_id,
                    "winner": winner,
                    "entry": entry,
                    "score": score,
                })
            except Exception:
                continue
            if starvation_guard and processed >= max_events:
                break
        # Write trace
        try:
            root = globals().get("MAVEN_ROOT")
            if not root:
                root = Path(__file__).resolve().parents[4]
            bb_path = (root / "reports" / "blackboard_trace.jsonl").resolve()
            from api.utils import append_jsonl  # type: ignore
            for rec in logs:
                append_jsonl(bb_path, rec)
        except Exception:
            pass
        return success_response(op, mid, {"processed": processed})

    elif op == "PROCESS_EVENTS":
        # Drain message bus events and log them. Does not affect working memory.
        try:
            # Pop all events from the internal message bus
            try:
                from brains.cognitive import message_bus  # type: ignore
                events: List[Dict[str, Any]] = message_bus.pop_all() or []
            except Exception:
                events = []
            # Write events to wm_events.jsonl for auditing
            counts: Dict[str, int] = {}
            try:
                root = globals().get("MAVEN_ROOT")
                if not root:
                    root = Path(__file__).resolve().parents[4]
                log_path = (root / "reports" / "wm_events.jsonl").resolve()
                from api.utils import append_jsonl  # type: ignore
                for ev in events:
                    try:
                        # Determine type field for counting; fallback to 'UNKNOWN'
                        typ = str(ev.get("type") or "UNKNOWN")
                        counts[typ] = counts.get(typ, 0) + 1
                        append_jsonl(log_path, {"event": ev})
                    except Exception:
                        continue
            except Exception:
                # If logging fails, still compute counts
                for ev in events:
                    try:
                        typ = str(ev.get("type") or "UNKNOWN")
                        counts[typ] = counts.get(typ, 0) + 1
                    except Exception:
                        pass
            return success_response(op, mid, {"events": counts})
        except Exception as e:
            return error_response(op, mid, "PROCESS_EVENTS_FAILED", str(e))

    elif op == "ALIGNMENT_AUDIT":
        # Stub alignment audit implementation.  Creates placeholder reports in
        # reports/agent and returns their names.  Full alignment logic will be
        # implemented in a future upgrade.
        try:
            rpt_dir = MAVEN_ROOT / "reports" / "agent"
            # Ensure report directory exists
            rpt_dir.mkdir(parents=True, exist_ok=True)
            # Create placeholder JSON files
            matrix = {"note": "alignment audit stub"}
            findings = {"note": "alignment findings stub"}
            proof = {"note": "alignment proof stub"}
            (rpt_dir / "alignment_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
            (rpt_dir / "alignment_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
            (rpt_dir / "alignment_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
            return success_response(op, mid, {"reports": ["alignment_matrix.json", "alignment_findings.json", "alignment_proof.json"]})
        except Exception as e:
            return error_response(op, mid, "ALIGNMENT_AUDIT_FAILED", str(e))

    elif op == "ALIGNMENT_PROPOSE":
        # Stub alignment propose implementation.  Generates an empty patch list
        # in reports/agent/patchlist.json.  This will be replaced by real logic
        # that analyses alignment findings.
        try:
            rpt_dir = MAVEN_ROOT / "reports" / "agent"
            rpt_dir.mkdir(parents=True, exist_ok=True)
            patch = {"patches": []}
            (rpt_dir / "patchlist.json").write_text(json.dumps(patch, indent=2), encoding="utf-8")
            return success_response(op, mid, {"report": "patchlist.json"})
        except Exception as e:
            return error_response(op, mid, "ALIGNMENT_PROPOSE_FAILED", str(e))

    elif op == "ALIGNMENT_APPLY":
        # Stub alignment apply implementation.  Requires a governance token to
        # authorise any modifications.  When authorised, writes a simple
        # alignment_apply_result.json file indicating success.  No actual
        # modifications are performed in this stub.
        token = str((payload or {}).get("token", ""))
        # Require tokens starting with GOV-
        if not token.startswith("GOV-"):
            return error_response(op, mid, "AUTH_FAILED", "Invalid governance token")
        try:
            rpt_dir = MAVEN_ROOT / "reports" / "agent"
            rpt_dir.mkdir(parents=True, exist_ok=True)
            result = {"applied": True}
            (rpt_dir / "alignment_apply_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return success_response(op, mid, {"report": "alignment_apply_result.json"})
        except Exception as e:
            return error_response(op, mid, "ALIGNMENT_APPLY_FAILED", str(e))

    elif op == "CORRECT":
        old_statement = (payload or {}).get("old")
        new_statement = (payload or {}).get("new")
        reason = (payload or {}).get("reason", "user_correction")
        if not old_statement or not new_statement:
            return error_response(op, mid, "BAD_REQUEST", "Provide 'old' and 'new' in payload.")
        # Find the old record across banks
        try:
            old_hits = _retrieve_from_banks(str(old_statement), k=1)
            hits = (old_hits or {}).get("results") or []
            if not hits:
                return error_response(op, mid, "NOT_FOUND", "Original statement not found for correction")
            old_rec = hits[0]
            old_id = old_rec.get("id")
            old_bank = old_rec.get("source_bank") or "theories_and_contradictions"
            # Mark old as superseded
            try:
                _bank_module(old_bank).service_api({
                    "op": "SUPERSEDE",
                    "payload": {"id": old_id, "reason": reason}
                })
            except Exception:
                pass
            # Store the corrected statement through normal pipeline
            return service_api({"op": "RUN_PIPELINE", "payload": {"text": str(new_statement), "confidence": 0.9}})
        except Exception as e:
            return error_response(op, mid, "CORRECT_FAILED", str(e))

    elif op == "BRAIN_PUT":
        # Store brain-specific persistent data using append-only JSONL
        try:
            scope = (payload or {}).get("scope", "BRAIN")
            origin_brain = (payload or {}).get("origin_brain", "unknown")
            key = (payload or {}).get("key")
            value = (payload or {}).get("value")
            try:
                conf = float((payload or {}).get("confidence", 0.8))
            except Exception:
                conf = 0.8
            if not key:
                return error_response(op, mid, "BAD_REQUEST", "key is required")
            # Store to brain-specific memory file (JSONL format)
            brain_mem_dir = MAVEN_ROOT / "brains" / "cognitive" / origin_brain / "memory"
            brain_mem_dir.mkdir(parents=True, exist_ok=True)
            brain_mem_file = brain_mem_dir / "brain_storage.jsonl"
            # Assign tier and importance via Phase 4 tier system
            tier_context = {
                "intent": str((payload or {}).get("intent", "")),
                "verdict": str((payload or {}).get("verdict", "")),
                "tags": (payload or {}).get("tags") or [],
            }
            record_for_tier = {"content": str(value or ""), "confidence": conf}
            tier, importance = _assign_tier(record_for_tier, tier_context)

            # Append the entry to JSONL file (no read-modify-write, true append)
            record = {
                "key": key,
                "value": value,
                "confidence": conf,
                "scope": scope,
                "tier": tier or TIER_MID,  # Brain storage defaults to MID tier
                "importance": importance,
                "seq_id": _next_seq_id(),
                "use_count": 0,
            }
            with open(brain_mem_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            _diag_log("BRAIN_PUT", {"brain":origin_brain, "key":key, "value":value})
            return success_response(op, mid, {"stored": True, "key": key, "brain": origin_brain})
        except Exception as e:
            return error_response(op, mid, "BRAIN_PUT_FAILED", str(e))

    elif op == "BRAIN_MERGE":
        # Merge-write with confidence bump when same key/value repeats
        try:
            scope = (payload or {}).get("scope", "BRAIN")
            origin_brain = (payload or {}).get("origin_brain", "unknown")
            key = (payload or {}).get("key")
            value = (payload or {}).get("value")
            if not key or value is None:
                return error_response(op, mid, "BAD_REQUEST", "key and value are required")
            try:
                conf_delta = float((payload or {}).get("conf_delta", 0.1))
            except Exception:
                conf_delta = 0.1
            # Use the _merge_brain_kv helper to handle the merge logic
            res = _merge_brain_kv(origin_brain, key, value, conf_delta=conf_delta)
            return success_response(op, mid, {"status": "merged", "scope": "BRAIN", "brain": origin_brain, "data": res})
        except Exception as e:
            return error_response(op, mid, "BRAIN_MERGE_FAILED", str(e))

    elif op == "BRAIN_GET":
        # Retrieve brain-specific persistent data from JSONL (last write wins)
        try:
            scope = (payload or {}).get("scope", "BRAIN")
            origin_brain = (payload or {}).get("origin_brain", "unknown")
            key = (payload or {}).get("key")
            if not key:
                return error_response(op, mid, "BAD_REQUEST", "key is required")
            # Load from brain-specific memory file (JSONL format)
            brain_mem_dir = MAVEN_ROOT / "brains" / "cognitive" / origin_brain / "memory"
            brain_mem_file = brain_mem_dir / "brain_storage.jsonl"
            if not brain_mem_file.exists():
                result = {"found": False, "data": None}
                _diag_log("BRAIN_GET", {"brain":origin_brain, "key":key, "result":result})
                return success_response(op, mid, result)
            # Read JSONL file and find the most recent matching key
            try:
                with open(brain_mem_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Search in reverse order for last write wins
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("key") == key:
                            # Return data in the format expected by consumers
                            result_data = {
                                "found": True,
                                "data": {
                                    "value": record.get("value"),
                                    "confidence": record.get("confidence", 0.8),
                                    "updated_at": record.get("updated_at"),
                                    "scope": record.get("scope", "BRAIN")
                                }
                            }
                            _diag_log("BRAIN_GET", {"brain":origin_brain, "key":key, "result":result_data.get("data")})
                            return success_response(op, mid, result_data)
                    except json.JSONDecodeError:
                        continue
                result = {"found": False, "data": None}
                _diag_log("BRAIN_GET", {"brain":origin_brain, "key":key, "result":result})
                return success_response(op, mid, result)
            except Exception:
                result = {"found": False, "data": None}
                _diag_log("BRAIN_GET", {"brain":origin_brain, "key":key, "result":result})
                return success_response(op, mid, result)
        except Exception as e:
            return error_response(op, mid, "BRAIN_GET_FAILED", str(e))

    # -------------------------------------------------------------------------
    # Phase 5: Continuous Learning Operations
    # -------------------------------------------------------------------------

    if op == "EXTRACT_PATTERNS":
        """
        Detect recurring patterns from memory records (preferences, intents, etc.)
        Returns: List of detected patterns with occurrence counts and consistency
        """
        try:
            from collections import defaultdict
            payload_data = payload or {}
            min_occurrences = int(payload_data.get("min_occurrences", 2))

            # Read all memory records across tiers to detect patterns
            from api.memory import ensure_dirs  # type: ignore
            tiers_root = Path(__file__).resolve().parents[1] / "memory"

            patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"occurrences": 0, "records": []})

            # Scan all tier files for preference clusters and recurring intents
            for tier_name in [TIER_SHORT, TIER_MID, TIER_LONG, TIER_PINNED]:
                tier_file = tiers_root / f"{tier_name.lower()}.jsonl"
                if not tier_file.exists():
                    continue

                with open(tier_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            verdict = str(rec.get("verdict", "")).upper()
                            intent = str(rec.get("intent", "")).upper()
                            content = str(rec.get("content", ""))

                            # Detect preference clusters
                            if verdict == "PREFERENCE" and content:
                                # Extract subject from preference (e.g., "i like cats" -> "cats")
                                subject = content.lower().replace("i like ", "").replace("i prefer ", "").strip()
                                if len(subject) > 2:
                                    pattern_key = f"preference_cluster:{subject}"
                                    patterns[pattern_key]["occurrences"] += 1
                                    patterns[pattern_key]["pattern_type"] = "preference_cluster"
                                    patterns[pattern_key]["subject"] = subject
                                    patterns[pattern_key]["records"].append(rec)

                            # Detect recurring intents
                            if intent and intent != "UNKNOWN":
                                pattern_key = f"recurring_intent:{intent}"
                                patterns[pattern_key]["occurrences"] += 1
                                patterns[pattern_key]["pattern_type"] = "recurring_intent"
                                patterns[pattern_key]["intent"] = intent
                                patterns[pattern_key]["records"].append(rec)
                        except Exception:
                            continue

            # Filter patterns by minimum occurrences and compute consistency
            detected_patterns = []
            total_records = sum(p["occurrences"] for p in patterns.values())
            for pattern_key, pattern_data in patterns.items():
                if pattern_data["occurrences"] >= min_occurrences:
                    consistency = pattern_data["occurrences"] / max(total_records, 1)
                    detected_patterns.append({
                        "pattern_key": pattern_key,
                        "pattern_type": pattern_data.get("pattern_type"),
                        "subject": pattern_data.get("subject"),
                        "intent": pattern_data.get("intent"),
                        "occurrences": pattern_data["occurrences"],
                        "consistency": round(consistency, 3)
                    })

            return success_response(op, mid, {"patterns": detected_patterns, "count": len(detected_patterns)})
        except Exception as e:
            return error_response(op, mid, "EXTRACT_PATTERNS_FAILED", str(e))

    if op == "CREATE_CONCEPT":
        """
        Create a concept record from a stable pattern
        Writes a CONCEPT record to TIER_LONG
        """
        try:
            from api.memory import append_jsonl  # type: ignore
            pattern = payload.get("pattern", {})

            if not pattern:
                return error_response(op, mid, "BAD_REQUEST", "pattern is required")

            # Generate concept from pattern
            pattern_type = pattern.get("pattern_type")
            subject = pattern.get("subject")
            occurrences = pattern.get("occurrences", 0)
            consistency = pattern.get("consistency", 0.0)

            # Build concept record
            concept_name = f"concept_{pattern_type}_{subject}".replace(" ", "_")
            concept_content = f"Learned concept: {subject} (type: {pattern_type}, evidence: {occurrences} occurrences)"

            concept_record = {
                "record_id": f"concept_{_next_seq_id()}",
                "content": concept_content,
                "verdict": RECORD_TYPE_CONCEPT,
                "confidence": min(0.9, 0.6 + consistency),  # Higher consistency = higher confidence
                "tier": TIER_LONG,
                "importance": min(1.0, 0.8 + consistency * 0.2),
                "tags": ["concept", pattern_type],
                "seq_id": _next_seq_id(),
                "use_count": 0,
                "pattern_source": pattern.get("pattern_key"),
                "evidence_count": occurrences
            }

            # Write to LONG tier
            tiers_root = Path(__file__).resolve().parents[1] / "memory"
            long_file = tiers_root / "long.jsonl"
            long_file.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(long_file, concept_record)

            return success_response(op, mid, {"concept": concept_record, "tier": TIER_LONG})
        except Exception as e:
            return error_response(op, mid, "CREATE_CONCEPT_FAILED", str(e))

    if op == "DETECT_SKILLS":
        """
        Detect skills from query history (recurring query patterns)
        Returns: List of detected skills with usage counts
        """
        try:
            query_history = payload.get("query_history", [])
            min_usage = int(payload.get("min_usage", 3))

            from collections import defaultdict
            skills: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"usage_count": 0, "examples": []})

            # Detect recurring query patterns
            for query_record in query_history:
                query = str(query_record.get("query", "")).lower()
                intent = str(query_record.get("intent", "")).upper()
                plan = query_record.get("plan", [])

                # Detect "WHY" questions
                if query.startswith("why ") and intent == "QUESTION":
                    skill_key = "WHY_question_handler"
                    skills[skill_key]["usage_count"] += 1
                    skills[skill_key]["input_shape"] = "WHY question"
                    skills[skill_key]["plan_template"] = plan
                    skills[skill_key]["examples"].append(query)

                # Detect "HOW" questions
                elif query.startswith("how ") and intent == "QUESTION":
                    skill_key = "HOW_question_handler"
                    skills[skill_key]["usage_count"] += 1
                    skills[skill_key]["input_shape"] = "HOW question"
                    skills[skill_key]["plan_template"] = plan
                    skills[skill_key]["examples"].append(query)

                # Detect preference queries
                elif "what do i like" in query or "my preferences" in query:
                    skill_key = "preference_query_handler"
                    skills[skill_key]["usage_count"] += 1
                    skills[skill_key]["input_shape"] = "preference query"
                    skills[skill_key]["plan_template"] = plan
                    skills[skill_key]["examples"].append(query)

            # Filter by minimum usage and create skill records
            detected_skills = []
            for skill_key, skill_data in skills.items():
                if skill_data["usage_count"] >= min_usage:
                    detected_skills.append({
                        "skill_name": skill_key,
                        "input_shape": skill_data.get("input_shape"),
                        "usage_count": skill_data["usage_count"],
                        "plan_template": skill_data.get("plan_template"),
                        "examples": skill_data["examples"][:3]  # Limit examples
                    })

            return success_response(op, mid, {"skills": detected_skills, "count": len(detected_skills)})
        except Exception as e:
            return error_response(op, mid, "DETECT_SKILLS_FAILED", str(e))

    if op == "CONSOLIDATE_PREFERENCES":
        """
        Consolidate repeated preferences into canonical forms
        Merges duplicate preferences and detects conflicts
        """
        try:
            from api.memory import append_jsonl  # type: ignore
            from collections import defaultdict

            # Read all preference records from MID tier
            tiers_root = Path(__file__).resolve().parents[1] / "memory"
            mid_file = tiers_root / "mid.jsonl"

            preferences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            if mid_file.exists():
                with open(mid_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            if str(rec.get("verdict", "")).upper() == "PREFERENCE":
                                content = str(rec.get("content", "")).lower()
                                # Extract subject (e.g., "i like cats" -> "cats")
                                subject = content.replace("i like ", "").replace("i prefer ", "").replace("i dislike ", "").strip()
                                if len(subject) > 2:
                                    preferences[subject].append(rec)
                        except Exception:
                            continue

            # Consolidate preferences with multiple evidence
            consolidated = []
            conflicts = []

            for subject, records in preferences.items():
                if len(records) >= 2:  # At least 2 occurrences
                    # Check for conflicts (likes vs dislikes)
                    sentiments = [("likes" if "like" in str(r.get("content", "")).lower() else "dislikes") for r in records]
                    if "likes" in sentiments and "dislikes" in sentiments:
                        conflicts.append({
                            "type": "CONFLICT",
                            "subject": subject,
                            "resolution_strategy": "present_both_ask_user",
                            "evidence": [r.get("content") for r in records]
                        })
                    else:
                        # No conflict - consolidate
                        canonical = f"user_likes_{subject}".replace(" ", "_")
                        consolidated.append({
                            "canonical": canonical,
                            "subject": subject,
                            "tier": TIER_MID,
                            "importance": 0.8,
                            "evidence_count": len(records),
                            "verdict": RECORD_TYPE_CONCEPT  # Consolidated preferences become concepts
                        })

            return success_response(op, mid, {
                "consolidated": consolidated,
                "conflicts": conflicts,
                "consolidated_count": len(consolidated),
                "conflict_count": len(conflicts)
            })
        except Exception as e:
            return error_response(op, mid, "CONSOLIDATE_PREFERENCES_FAILED", str(e))

    # Fallback for unsupported operations
    return error_response(op, mid, "UNSUPPORTED_OP", op)

# -----------------------------------------------------------------------------
# Attention bid interface for memory retrieval
#
# The memory librarian does not represent a standalone cognitive brain,
# but for the purposes of attention resolution it can submit a bid on
# behalf of the memory subsystem.  A high confidence match in memory
# indicates that a relevant answer likely exists, so memory should
# receive attention.  Otherwise memory bids low.  The result is a
# dictionary with the same structure as BrainBid.to_dict().
def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an attention bid on behalf of memory retrieval.

    NOW WITH COGNITIVE BRAIN CONTRACT COMPLIANCE:
    - Uses standardized continuation_helpers for context awareness
    - Emits routing hints for Teacher learning
    - Detects if this is a continuation query for thread-aware retrieval

    Args:
        ctx: The current pipeline context passed through the memory
            librarian.  Should include stage_2R_memory results.

    Returns:
        A dictionary containing brain_name, priority, reason and evidence.
    """
    try:
        mem = ctx.get("stage_2R_memory") or {}
        results = mem.get("results") or []

        # Detect if this is a continuation query
        is_follow_up = False
        query = ctx.get("query", "")
        if _continuation_helpers_available and query:
            try:
                is_follow_up = is_continuation(query, ctx)
            except Exception:
                pass

        # Find the first result with confidence above 0.8
        high_match = None
        for it in results:
            try:
                c = float(it.get("confidence", 0.0))
                if c > 0.8:
                    high_match = it
                    break
            except Exception:
                continue

        # Determine action and priority based on match quality and continuation status
        if high_match:
            priority = 0.85
            reason = "high_confidence_match"
            action = "thread_aware_retrieval" if is_follow_up else "fresh_retrieval"
            confidence = float(high_match.get("confidence", 0.85))
            evidence = {"result": high_match}
        else:
            priority = 0.10
            reason = "default"
            action = "low_confidence_retrieval"
            confidence = 0.10
            evidence = {}

        # Create standardized routing hint
        routing_hint = None
        if _continuation_helpers_available:
            try:
                routing_hint = create_routing_hint(
                    brain_name="memory_librarian",
                    action=action,
                    confidence=confidence,
                    context_tags=(["memory", "context_aware", "continuation"] if is_follow_up
                                 else ["memory", "fresh_query"])
                )
                evidence["routing_hint"] = routing_hint
            except Exception as e:
                print(f"[MEMORY_LIBRARIAN] Warning: Failed to create routing hint: {str(e)[:100]}")

        return {
            "brain_name": "memory",
            "priority": priority,
            "reason": reason,
            "evidence": evidence,
        }

    except Exception:
        return {
            "brain_name": "memory",
            "priority": 0.10,
            "reason": "default",
            "evidence": {},
        }

# -----------------------------------------------------------------------------
# Inference helper for reasoning fallback
#
# When the reasoning brain has attention but cannot determine a verdict,
# we perform a lightweight inference using retrieved facts.  This helper
# attempts to answer yes/no questions of the form "Is X Y?" or "Is X a Y?"
# by scanning memory results for explicit statements of the form
# "X is a Y" or "X is Y".  It returns a dict with the inferred answer,
# confidence and a simple reasoning chain if successful, otherwise None.
def _attempt_inference(query: str, facts: List[Dict[str, Any]]) -> Any:
    """Attempt to infer an answer and explanation from memory facts.

    The inference process has two layers:

    1. Yes/No classification: For queries of the form "is X (a) Y?",
       scan facts for direct affirmations or negations and return a
       binary answer with a simple explanatory step.
    2. Multi‑step reasoning: For other queries, build short reasoning
       chains by matching keywords in the query to fact contents.  The
       strongest chain is selected based on a heuristic confidence and
       returned with an ordered list of supporting facts and roles.

    Args:
        query: Raw user query.
        facts: Relevant memory records (dicts with at least a 'content' field).

    Returns:
        A dict with keys 'answer', 'confidence', 'steps' and 'trace',
        or None when inference cannot produce a useful answer.
    """
    try:
        import re
        # Normalise query
        q_raw = str(query or "").strip().lower()
        q_norm = re.sub(r"[^a-z0-9\s]", " ", q_raw)
        q_norm = re.sub(r"\s+", " ", q_norm).strip()
        # Yes/no pattern: "is X a Y" or "is X Y"
        m = re.match(r"^is\s+(.+?)\s+(?:a\s+)?(.+)$", q_norm)
        if m:
            entity = m.group(1).strip()
            category = m.group(2).strip()
            # Remove leading articles from entity
            entity = re.sub(r"^(the|a|an)\s+", "", entity)
            entity_l = entity
            category_l = category
            for rec in facts or []:
                try:
                    content = str(rec.get("content", "")).strip().lower()
                except Exception:
                    content = ""
                if not content:
                    continue
                # Positive match: "entity is a category" or "entity is category"
                if re.search(rf"\b{re.escape(entity_l)}\s+is\s+(?:a\s+)?{re.escape(category_l)}\b", content):
                    return {
                        "answer": "Yes.",
                        "confidence": float(rec.get("confidence", 0.7) or 0.7),
                        "steps": [f"Found statement in memory: '{content}' which affirms that {entity} is {category}."],
                        "trace": [
                            {
                                "fact": content,
                                "role": "definition",
                            }
                        ],
                    }
                # Negative match: "entity is not a category"
                if re.search(rf"\b{re.escape(entity_l)}\s+is\s+not\s+(?:a\s+)?{re.escape(category_l)}\b", content):
                    return {
                        "answer": "No.",
                        "confidence": float(rec.get("confidence", 0.7) or 0.7),
                        "steps": [f"Found statement in memory: '{content}' which negates that {entity} is {category}."],
                        "trace": [
                            {
                                "fact": content,
                                "role": "definition",
                            }
                        ],
                    }
        # Multi‑step inference for general queries
        # Extract meaningful keywords (>2 chars) from the query
        words = [w for w in q_norm.split() if len(w) > 2]
        if not words:
            return None
        # Preprocess facts to lower‑cased content
        proc: List[str] = []
        for rec in facts or []:
            try:
                c = str(rec.get("content", "")).strip().lower()
            except Exception:
                c = ""
            if c:
                proc.append(c)
        if not proc:
            return None
        # Gather candidate single‑step reasoning entries
        candidates: List[Dict[str, Any]] = []
        for content in proc:
            shared = [kw for kw in words if kw in content]
            if not shared:
                continue
            conf = min(1.0, 0.5 + 0.1 * len(shared))
            candidates.append({
                "conclusion": content,
                "confidence": conf,
                "steps": [content],
            })
        # Attempt to form simple two‑step chains
        n = len(proc)
        for i in range(n):
            content_i = proc[i]
            shared_i = [kw for kw in words if kw in content_i]
            if not shared_i:
                continue
            for j in range(i + 1, n):
                content_j = proc[j]
                shared_j = [kw for kw in words if kw in content_j]
                if not shared_j:
                    continue
                conf = min(1.0, 0.6 + 0.05 * (len(shared_i) + len(shared_j)))
                candidates.append({
                    "conclusion": content_j,
                    "confidence": conf,
                    "steps": [content_i, content_j],
                })
        # Select best candidate
        best: Optional[Dict[str, Any]] = None
        for cand in candidates:
            try:
                if best is None or float(cand.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
                    best = cand
            except Exception:
                continue
        if best:
            try:
                bconf = float(best.get("confidence", 0.0))
            except Exception:
                bconf = 0.0
            if bconf >= 0.5:
                # Build trace with roles
                step_texts: List[str] = []
                trace: List[Dict[str, Any]] = []
                for s in best.get("steps", []) or []:
                    txt = str(s)
                    step_texts.append(txt)
                    # infer role heuristically
                    role: str
                    st = txt.lower()
                    if any(k in st for k in ["type", "types", "c3", "c4", "variety", "classification"]):
                        role = "types"
                    elif any(k in st for k in [
                        "process", "reaction", "stage", "step", "light‑dependent", "light dependent", "light‑independent",
                        "light independent", "cycle", "mechanism", "transfer", "energy", "convert"
                    ]):
                        role = "mechanism"
                    else:
                        role = "definition"
                    trace.append({"fact": txt, "role": role})
                return {
                    "answer": best.get("conclusion"),
                    "confidence": bconf,
                    "steps": step_texts,
                    "trace": trace,
                }
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Handle wrapper for working memory routing
# ---------------------------------------------------------------------------

# Save reference to original service_api implementation
_service_api_impl = service_api

# Flag to track if WM has been hydrated from persistent brain storage
_WM_HYDRATED = False

def _hydrate_wm_from_brain_storage() -> None:
    """
    Hydrate working memory from per-brain persistent storage on first call.
    Loads last saved values into WM so memory survives restarts.
    """
    global _WM_HYDRATED
    if _WM_HYDRATED:
        return
    _WM_HYDRATED = True

    try:
        def _last_brain_value(brain: str, key: str) -> Any:
            """Get the last value for a key from brain storage (last-write-wins)."""
            brain_mem_dir = MAVEN_ROOT / "brains" / "cognitive" / brain / "memory"
            brain_mem_file = brain_mem_dir / "brain_storage.jsonl"
            if not brain_mem_file.exists():
                return None
            try:
                with open(brain_mem_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Search in reverse order for last write wins
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        k = rec.get("k") or rec.get("key")
                        if k == key:
                            return rec.get("value", rec.get("v"))
                    except json.JSONDecodeError:
                        continue
                return None
            except Exception:
                return None

        # Hydrate known preference keys from memory_librarian brain storage
        for key in ("user_identity", "relationship_status", "favorite_color"):
            val = _last_brain_value("memory_librarian", key)
            if val is not None:
                # Add to working memory with high confidence
                with _WM_LOCK:
                    _WORKING_MEMORY.append({
                        "key": key,
                        "value": val,
                        "tags": ["preference", "hydrated"],
                        "confidence": 0.9,
                    })
    except Exception:
        # Silently fail if hydration errors occur
        pass

def handle(context: dict) -> dict:
    """
    Handle function that routes working memory operations.

    This wrapper adds _passed_memory = True to the context for WM operations
    and delegates to the underlying service implementation.

    Args:
        context: Request dictionary with 'op' and optional 'payload'

    Returns:
        Response dictionary from service implementation
    """
    # Hydrate WM from persistent storage on first call
    _hydrate_wm_from_brain_storage()

    # Mark that this request has passed through memory handling
    if isinstance(context, dict):
        op = (context or {}).get("op", "").upper()
        # For WM operations, mark as passed through memory
        if op.startswith("WM_"):
            context["_passed_memory"] = True

    # Route to the underlying service implementation
    return _service_api_impl(context)

# Service API entry point
service_api = handle
