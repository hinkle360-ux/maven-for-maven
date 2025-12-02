"""
Text Normalizer for Routing
============================

This module normalizes user input BEFORE routing happens.
This allows routing to work correctly even with typos and misspellings.

Pipeline:
    raw text → normalize() → command grammar + router → brains/tools

The normalizer:
1. Lowercases and cleans whitespace
2. Fuzzy-matches tokens against command vocabulary
3. Optionally uses LLM for complex cases
4. Learns personal typos over time

Usage:
    from brains.routing.normalizer import normalize

    norm_text = normalize("x gork helo fom mavn")
    # Returns: "x grok hello from maven"
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# COMMAND VOCABULARY
# =============================================================================

# Tools and commands that routing cares about
TOOL_VOCABULARY = {
    # Browser/platform tools
    "x", "grok", "chatgpt", "browser", "browser_open", "captcha", "captcha_solver",
    # Research tools
    "research", "search", "web", "web_search", "web_fetch",
    # Execution tools
    "shell", "python", "python_sandbox", "code", "run", "execute",
    # File tools
    "fs", "file", "git", "reload",
    # Maven commands
    "maven", "whoami", "introduce", "scan", "status",
}

# Keywords that affect routing
KEYWORD_VOCABULARY = {
    # Command prefixes
    "use", "tool", "open", "post", "reply", "send", "tell",
    # Self-intent keywords
    "explain", "yourself", "self", "who", "what", "are", "you",
    "introduce", "describe", "capabilities", "able",
    # Research keywords
    "research", "search", "find", "lookup", "investigate",
    # Action keywords
    "run", "execute", "create", "write", "read", "list", "scan",
}

# Combined vocabulary for fuzzy matching
ROUTING_VOCABULARY = TOOL_VOCABULARY | KEYWORD_VOCABULARY

# Common self-intent words that need normalization
SELF_INTENT_WORDS = {
    "yourself", "self", "explain", "introduce", "describe",
    "who", "what", "are", "you", "maven",
}

# PROTECTED WORDS: Common English words that should NEVER be fuzzy-matched
# These are valid words that happen to be close to command vocabulary
PROTECTED_WORDS = {
    # Pronouns and determiners
    "your", "you", "the", "a", "an",
    "my", "me", "we", "us", "our", "their", "them", "its",
    "this", "that", "these", "those",
    # Conjunctions and prepositions
    "and", "or", "but", "for", "from", "to", "in", "on", "at",
    "about", "with", "into", "over", "under", "of", "by",
    # Common verbs (prevent matching to command words)
    "is", "it", "be", "are", "was", "were", "been",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might",
    "get", "got", "give", "go", "going", "goes", "went",
    "say", "said", "says", "tell", "told", "talk", "talking",
    "like", "likes", "want", "wants", "need", "needs",
    "know", "knows", "think", "thinks", "see", "sees", "saw",
    "make", "makes", "made", "take", "takes", "took",
    # Common adverbs and adjectives
    "just", "also", "very", "really", "actually", "probably",
    "real", "good", "bad", "new", "old", "big", "small",
    "first", "last", "next", "other", "more", "most", "some",
    # Time words
    "time", "now", "then", "when", "before", "after",
    # Other common words that could false-match
    "self",   # Important for "your self" - don't match to "shell"
    "hello", "hi", "hey",
    "back", "read", "human", "prove", "proved",
    "way", "work", "works", "thing", "things",
    "here", "there", "where", "how", "why", "which",
    "all", "any", "each", "every", "both", "few", "many",
    "let", "lets", "put", "set", "try", "use", "used",
}


# =============================================================================
# TYPO DICTIONARY (loads from file, grows over time)
# =============================================================================

# Built-in common typos (never changes)
BUILTIN_TYPOS: Dict[str, str] = {
    # Tool typos
    "gork": "grok",
    "grokc": "grok",
    "gorck": "grok",
    "grk": "grok",
    "chatgtp": "chatgpt",
    "chatgp": "chatgpt",
    "chatgot": "chatgpt",
    "broser": "browser",
    "borwser": "browser",
    "browswer": "browser",
    "capcha": "captcha",
    "captch": "captcha",
    "reseach": "research",
    "researhc": "research",
    "resarch": "research",
    "serch": "search",
    "searc": "search",
    "searhc": "search",
    # Keyword typos
    "explaine": "explain",
    "explian": "explain",
    "explaain": "explain",
    "expain": "explain",
    "explan": "explain",
    "youself": "yourself",
    "yourslef": "yourself",
    "yurself": "yourself",
    "introudce": "introduce",
    "intorduce": "introduce",
    "introduc": "introduce",
    "descirbe": "describe",
    "desribe": "describe",
    "waht": "what",
    "whta": "what",
    "teh": "the",
    "fom": "from",
    "helo": "hello",
    "hellow": "hello",
    "mavn": "maven",
    "maevn": "maven",
    # Common general typos
    "thnk": "think",
    "aobut": "about",
    "abotu": "about",
    "taht": "that",
    "jsut": "just",
    "yoru": "your",
    "thier": "their",
    "recieve": "receive",
    "occured": "occurred",
    "seperate": "separate",
    "definately": "definitely",
    "accomodate": "accommodate",
}

# Path to learned typos file
_TYPO_FILE_PATH = Path(__file__).parent.parent.parent / "memory" / "typos.json"

# Runtime typo dictionary (builtin + learned)
_typo_dict: Dict[str, str] = {}


def _load_typo_dictionary() -> Dict[str, str]:
    """Load typo dictionary from file + builtins."""
    global _typo_dict

    # Start with builtins
    _typo_dict = dict(BUILTIN_TYPOS)

    # Load learned typos if file exists
    try:
        if _TYPO_FILE_PATH.exists():
            with open(_TYPO_FILE_PATH, "r") as f:
                learned = json.load(f)
                if isinstance(learned, dict):
                    _typo_dict.update(learned)
                    print(f"[NORMALIZER] Loaded {len(learned)} learned typos")
    except Exception as e:
        print(f"[NORMALIZER] Failed to load typos.json: {e}")

    return _typo_dict


def save_typo(typo: str, correction: str) -> bool:
    """
    Save a new typo correction to the learned dictionary.

    Args:
        typo: The misspelled word
        correction: The correct spelling

    Returns:
        True if saved successfully
    """
    global _typo_dict

    if not typo or not correction:
        return False

    typo = typo.lower().strip()
    correction = correction.lower().strip()

    if typo == correction:
        return False

    # Add to runtime dict
    _typo_dict[typo] = correction

    # Persist to file
    try:
        # Load existing
        learned = {}
        if _TYPO_FILE_PATH.exists():
            with open(_TYPO_FILE_PATH, "r") as f:
                learned = json.load(f)

        # Add new typo
        learned[typo] = correction

        # Ensure directory exists
        _TYPO_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(_TYPO_FILE_PATH, "w") as f:
            json.dump(learned, f, indent=2, sort_keys=True)

        print(f"[NORMALIZER] Learned typo: {typo} -> {correction}")
        return True

    except Exception as e:
        print(f"[NORMALIZER] Failed to save typo: {e}")
        return False


# =============================================================================
# EDIT DISTANCE (Levenshtein)
# =============================================================================

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if chars match, 1 otherwise
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _fuzzy_match_token(token: str, vocabulary: Set[str], max_distance: int = 2) -> Optional[str]:
    """
    Find the best fuzzy match for a token in the vocabulary.

    Args:
        token: The token to match
        vocabulary: Set of valid words
        max_distance: Maximum edit distance to consider

    Returns:
        Best matching word, or None if no good match
    """
    if not token:
        return None

    token_lower = token.lower()

    # Exact match
    if token_lower in vocabulary:
        return token_lower

    # Adjust max distance based on token length
    # Short tokens (≤3 chars) only allow distance 1
    # Longer tokens allow distance 2
    effective_max = 1 if len(token) <= 3 else max_distance

    best_match = None
    best_distance = effective_max + 1

    for word in vocabulary:
        # Quick length check to skip obviously wrong matches
        if abs(len(word) - len(token)) > effective_max:
            continue

        distance = _levenshtein_distance(token_lower, word)
        if distance < best_distance:
            best_distance = distance
            best_match = word

    return best_match if best_distance <= effective_max else None


# =============================================================================
# MAIN NORMALIZER
# =============================================================================

@dataclass
class NormalizationResult:
    """Result of text normalization."""
    original: str
    normalized: str
    corrections: List[Tuple[str, str]] = field(default_factory=list)  # [(typo, correction), ...]
    was_modified: bool = False


def normalize(text: str, use_llm: bool = False) -> NormalizationResult:
    """
    Normalize user input for routing.

    This function:
    1. Cleans whitespace and punctuation
    2. Applies typo dictionary corrections
    3. Fuzzy-matches tokens against command vocabulary
    4. Optionally uses LLM for complex cases

    Args:
        text: Raw user input
        use_llm: Whether to use LLM for complex normalization (default False)

    Returns:
        NormalizationResult with original and normalized text
    """
    if not text:
        return NormalizationResult(original="", normalized="")

    # Ensure typo dict is loaded
    if not _typo_dict:
        _load_typo_dictionary()

    original = text
    corrections: List[Tuple[str, str]] = []

    # Step 1: Basic cleanup
    # - Lowercase
    # - Collapse multiple spaces
    # - Strip leading/trailing whitespace
    # - Preserve colons after command keywords (x:, research:, etc.)
    cleaned = text.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Step 2: Tokenize (preserve punctuation attached to tokens)
    # Split on spaces but keep punctuation with tokens
    tokens = cleaned.split(' ')

    # Step 3: Process each token
    normalized_tokens = []
    for i, token in enumerate(tokens):
        # Extract any trailing punctuation
        match = re.match(r'^(\w+)([^\w]*)$', token)
        if match:
            word = match.group(1)
            punct = match.group(2)
        else:
            word = token
            punct = ""

        word_lower = word.lower()

        # PROTECTED WORDS: Never fuzzy-match common English words
        # This prevents "your" -> "you", "self" -> "shell", etc.
        if word_lower in PROTECTED_WORDS:
            normalized_tokens.append(word_lower + punct)
            continue

        # Check typo dictionary first (explicit corrections)
        if word_lower in _typo_dict:
            correction = _typo_dict[word_lower]
            corrections.append((word, correction))
            normalized_tokens.append(correction + punct)
            continue

        # For routing-critical tokens (first 5), try fuzzy matching
        # But ONLY if the word is NOT a valid English word
        if i < 5:
            fuzzy_match = _fuzzy_match_token(word, ROUTING_VOCABULARY)
            if fuzzy_match and fuzzy_match != word_lower:
                corrections.append((word, fuzzy_match))
                normalized_tokens.append(fuzzy_match + punct)
                continue

        # Keep original token (lowercase)
        normalized_tokens.append(word_lower + punct)

    normalized = ' '.join(normalized_tokens)
    was_modified = normalized.lower() != original.lower() or len(corrections) > 0

    if corrections:
        print(f"[NORMALIZER] Corrections: {corrections}")

    return NormalizationResult(
        original=original,
        normalized=normalized,
        corrections=corrections,
        was_modified=was_modified,
    )


def normalize_for_routing(text: str) -> str:
    """
    Convenience function that returns just the normalized text.

    Args:
        text: Raw user input

    Returns:
        Normalized text suitable for routing
    """
    result = normalize(text)
    return result.normalized


# =============================================================================
# LEARNING FROM ROUTING ERRORS
# =============================================================================

def learn_from_routing_error(
    raw_input: str,
    normalized_input: str,
    expected_route: str,
    actual_route: str,
) -> None:
    """
    Learn from a routing error by extracting typo corrections.

    When routing goes wrong, we can often identify which word
    caused the problem and add it to the typo dictionary.

    Args:
        raw_input: The original user input
        normalized_input: What we normalized it to
        expected_route: What the route should have been
        actual_route: What route was actually taken
    """
    # This is called when we detect a routing mistake
    # We can analyze the difference between raw and expected
    # to learn new typo corrections

    raw_tokens = raw_input.lower().split()
    norm_tokens = normalized_input.lower().split()

    # Find tokens that differ
    for raw_tok, norm_tok in zip(raw_tokens, norm_tokens):
        if raw_tok != norm_tok:
            # This was a correction - if routing still failed,
            # maybe we need to correct to something else
            print(f"[NORMALIZER] Potential learning: {raw_tok} was corrected to {norm_tok}")
            print(f"[NORMALIZER]   Expected route: {expected_route}, got: {actual_route}")

    # For now, just log. In the future, we could:
    # 1. Ask the user what the correct word should be
    # 2. Use the expected route to infer the correct word
    # 3. Use an LLM to suggest corrections


# =============================================================================
# INITIALIZATION
# =============================================================================

# Load typo dictionary on module import
_load_typo_dictionary()


__all__ = [
    "normalize",
    "normalize_for_routing",
    "NormalizationResult",
    "save_typo",
    "learn_from_routing_error",
    "ROUTING_VOCABULARY",
]
