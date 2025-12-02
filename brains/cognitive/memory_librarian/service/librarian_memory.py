"""
librarian_memory.py
~~~~~~~~~~~~~~~~~~~

Memory tier system specifically for the Memory Librarian.

MAVEN MASTER SPEC COMPLIANCE:
- Memory Librarian has its own STM→MTM→LTM→Archive tiers
- Librarian sees all other brains' memory (read-only)
- Librarian routes facts per Reasoning Brain decisions
- Librarian handles dedupe and tier management
- Librarian does NOT write into other brains' heuristics

The Memory Librarian uses its own memory tiers to store:
- Routing decisions and rules
- Cross-brain dedupe information
- Memory consolidation metadata
- Librarian's internal heuristics

This module provides a dedicated BrainMemory instance for the librarian.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

# Import the universal memory tier system
try:
    from brains.memory.brain_memory import BrainMemory

    # Create Memory Librarian's own memory instance
    # Higher capacities than typical brains due to librarian's coordination role
    _librarian_memory = BrainMemory(
        brain_id="memory_librarian",
        stm_capacity=200,   # Higher than default (100)
        mtm_capacity=1000,  # Higher than default (500)
        ltm_capacity=5000   # Higher than default (2000)
    )

except Exception:
    _librarian_memory = None  # type: ignore


def store_routing_rule(
    rule: str,
    domain: str,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store a routing rule in the librarian's memory (goes to STM).

    MAVEN SPEC: All writes go to STM, automatic tier spill.

    Args:
        rule: The routing rule (e.g., "planet → science")
        domain: Target domain bank
        confidence: Confidence level (0.0-1.0)
        metadata: Optional additional metadata

    Returns:
        Storage result
    """
    if not _librarian_memory:
        return {"ok": False, "error": "Memory not initialized"}

    meta = metadata or {}
    meta.update({
        "type": "routing_rule",
        "domain": domain,
        "confidence": confidence
    })

    return _librarian_memory.store(
        content=rule,
        metadata=meta
    )


def store_learned_routing_rule(
    question: str,
    routes: List[Dict[str, Any]],
    aliases: List[str],
    source: str = "llm_teacher"
) -> Dict[str, Any]:
    """
    Store a learned routing rule from the LLM teacher in the librarian's memory.

    This function stores routing rules learned from the Teacher Brain's TEACH_ROUTING
    operation. The rule includes the original question, suggested banks with weights,
    and keyword aliases for future routing.

    Args:
        question: The question that triggered the routing learning
        routes: List of dicts with 'bank' and 'weight' keys
        aliases: List of alias phrases for similar questions
        source: Source of the routing rule (default: "llm_teacher")

    Returns:
        Storage result
    """
    if not _librarian_memory:
        return {"ok": False, "error": "Memory not initialized"}

    # Store the routing rule with structured metadata
    content = {
        "question": question,
        "routes": routes,
        "aliases": aliases
    }

    metadata = {
        "kind": "routing_rule",
        "type": "learned_routing",
        "source": source,
        "question_pattern": question.lower(),  # For matching
        "confidence": 0.8,  # Start with high confidence for LLM suggestions
        "route_count": len(routes),
        "alias_count": len(aliases)
    }

    return _librarian_memory.store(
        content=content,
        metadata=metadata
    )


def retrieve_routing_rule_for_question(question: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Retrieve a learned routing rule for a question.

    FIX 2: Enhanced with semantic matching including:
    - Approximate word-set overlap with normalization
    - Stemming-like suffix removal (e.g., "ducks" → "duck")
    - Partial phrase matching
    - Alias normalization

    Args:
        question: The question to find routing for
        threshold: Minimum similarity threshold (0.0-1.0)

    Returns:
        The best matching routing rule dict, or None if no match found
    """
    if not _librarian_memory:
        return None

    def _normalize_word(word: str) -> str:
        """Normalize word by removing common suffixes for better matching."""
        word = word.lower().strip()
        # Remove common suffixes for stemming-like behavior
        for suffix in ['ing', 'ed', 's', 'es', 'er', 'ly']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                # Only strip if it leaves a reasonable stem
                return word[:-len(suffix)]
        return word

    def _compute_semantic_score(q1_words: set, q2_words: set, q1_text: str, q2_text: str) -> float:
        """Compute semantic similarity score between two question word sets."""
        if not q1_words or not q2_words:
            return 0.0

        # Normalize words for better matching
        q1_normalized = {_normalize_word(w) for w in q1_words}
        q2_normalized = {_normalize_word(w) for w in q2_words}

        # Word overlap score (normalized stems)
        overlap = len(q1_normalized & q2_normalized)
        if overlap == 0:
            return 0.0

        # Jaccard similarity with normalized words
        union_size = len(q1_normalized | q2_normalized)
        jaccard = overlap / union_size if union_size > 0 else 0.0

        # Boost for substring/phrase matches
        phrase_bonus = 0.0
        if q1_text in q2_text or q2_text in q1_text:
            phrase_bonus = 0.2

        # Combined score
        return jaccard + phrase_bonus

    try:
        # Retrieve routing rules from STM and MTM (recent rules are most relevant)
        results = _librarian_memory.retrieve(
            query=question,
            limit=20,  # Increased from 10 to catch more candidates
            tiers=["stm", "mtm"]
        )

        # Filter for learned routing rules and find best match
        best_match = None
        best_score = threshold

        question_lower = question.lower()
        question_words = set(question_lower.split())

        for record in results:
            metadata = record.get("metadata", {})
            if metadata.get("kind") != "routing_rule":
                continue
            if metadata.get("type") != "learned_routing":
                continue

            content = record.get("content", {})
            if not isinstance(content, dict):
                continue

            # Calculate semantic similarity with stored question
            stored_question = content.get("question", "").lower()
            stored_words = set(stored_question.split())

            if not stored_words:
                continue

            # Check direct question match with semantic scoring
            score = _compute_semantic_score(question_words, stored_words, question_lower, stored_question)
            if score > best_score:
                best_score = score
                best_match = content

            # Check if any aliases match (with semantic scoring)
            aliases = content.get("aliases", [])
            for alias in aliases:
                alias_lower = alias.lower()
                alias_words = set(alias_lower.split())
                if alias_words:
                    alias_score = _compute_semantic_score(question_words, alias_words, question_lower, alias_lower)
                    # Boost alias matches slightly since they're explicitly provided
                    alias_score *= 1.1
                    if alias_score > best_score:
                        best_score = alias_score
                        best_match = content

        # FIX 4: Log routing rule matches
        if best_match:
            print(f"[ROUTING_RULE_MATCH] Found rule for '{question[:60]}' (score={best_score:.2f})")

        return best_match

    except Exception as e:
        print(f"[ROUTING_RULE_RETRIEVAL_ERROR] {str(e)[:100]}")
        return None


def learn_routing_for_question(
    question: str,
    available_banks: List[str],
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Ask the Teacher Brain for routing suggestions and store them.

    This function:
    1. Calls TeacherBrain with TEACH_ROUTING operation
    2. Gets suggested routes and aliases
    3. Stores them in librarian memory for future use
    4. Returns the routing info for immediate use

    Args:
        question: The question to learn routing for
        available_banks: List of available memory banks
        context: Optional context dict

    Returns:
        Dict with 'routes' and 'aliases', or None on error
    """
    try:
        # Import teacher brain module
        from pathlib import Path
        import sys

        # Try to load teacher brain
        try:
            from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api  # type: ignore
        except Exception:
            # If direct import fails, try dynamic loading
            try:
                cog_root = Path(__file__).parent.parent.parent
                teacher_path = cog_root / "teacher" / "service" / "teacher_brain.py"

                import importlib.util
                spec = importlib.util.spec_from_file_location("teacher_brain", teacher_path)
                if spec and spec.loader:
                    teacher_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(teacher_module)
                    teacher_api = teacher_module.service_api  # type: ignore
                else:
                    return None
            except Exception:
                return None

        # Call teacher for routing suggestion
        response = teacher_api({
            "op": "TEACH_ROUTING",
            "payload": {
                "question": question,
                "available_banks": available_banks,
                "context": context or {}
            }
        })

        if not response.get("ok"):
            return None

        payload = response.get("payload", {})
        routes = payload.get("routes", [])
        aliases = payload.get("aliases", [])

        # Store the learned routing rule
        if routes:  # Only store if we got valid routes
            store_learned_routing_rule(
                question=question,
                routes=routes,
                aliases=aliases,
                source="llm_teacher"
            )

        return {
            "routes": routes,
            "aliases": aliases
        }

    except Exception:
        return None


def store_dedupe_record(
    content_hash: str,
    original_location: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store a deduplication record in the librarian's memory.

    MAVEN SPEC: All writes go to STM, automatic tier spill.

    Args:
        content_hash: Hash of the content for deduplication
        original_location: Where the original is stored (brain_id/tier)
        metadata: Optional additional metadata

    Returns:
        Storage result
    """
    if not _librarian_memory:
        return {"ok": False, "error": "Memory not initialized"}

    meta = metadata or {}
    meta.update({
        "type": "dedupe_record",
        "original_location": original_location
    })

    return _librarian_memory.store(
        content={"hash": content_hash, "location": original_location},
        metadata=meta
    )


def retrieve_routing_rules(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve routing rules from librarian's memory.

    Args:
        domain: Optional filter by target domain

    Returns:
        List of routing rules
    """
    if not _librarian_memory:
        return []

    # Retrieve from STM and MTM (recent rules are most relevant)
    results = _librarian_memory.retrieve(tiers=["stm", "mtm"])

    # Filter by type and domain
    filtered = []
    for record in results:
        metadata = record.get("metadata", {})
        if metadata.get("type") != "routing_rule":
            continue
        if domain and metadata.get("domain") != domain:
            continue
        filtered.append(record)

    return filtered


def get_librarian_stats() -> Dict[str, Any]:
    """
    Get statistics about the librarian's own memory.

    Returns:
        Dictionary with tier counts and metadata
    """
    if not _librarian_memory:
        return {"ok": False, "error": "Memory not initialized"}

    stats = _librarian_memory.get_stats()
    stats["ok"] = True
    stats["brain_id"] = "memory_librarian"

    return stats


def compact_librarian_archive() -> int:
    """
    Compact the librarian's archive tier.

    Returns:
        Number of records in compacted archive
    """
    if not _librarian_memory:
        return 0

    return _librarian_memory.compact()


# Public API for memory_librarian.py to import
__all__ = [
    "_librarian_memory",
    "store_routing_rule",
    "store_learned_routing_rule",
    "retrieve_routing_rule_for_question",
    "learn_routing_for_question",
    "store_dedupe_record",
    "retrieve_routing_rules",
    "get_librarian_stats",
    "compact_librarian_archive"
]
