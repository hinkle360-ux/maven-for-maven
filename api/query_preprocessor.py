"""
Query Preprocessor
==================

Preprocesses user queries before they enter the Maven pipeline.

Functions:
- Split multi-part questions (e.g., "36+42 what are dogs and what is lightning")
- Extract leading math expressions
- Normalize compound queries

This fixes the regression where multi-part questions were being parroted back
instead of being answered individually.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any


def preprocess_query(text: str) -> List[Dict[str, Any]]:
    """
    Preprocess a user query and split it into sub-queries if needed.

    Handles:
    1. Leading math expressions: "36+42 what are dogs" -> ["36+42", "what are dogs"]
    2. Multiple questions with "and": "what are dogs and what is lightning" -> ["what are dogs", "what is lightning"]
    3. Mixed cases: "36+42 what are dogs and what is lightning" -> ["36+42", "what are dogs", "what is lightning"]

    Args:
        text: The raw user query

    Returns:
        List of sub-query dicts with 'text' and 'type' keys
    """
    if not text or not text.strip():
        return [{"text": text, "type": "unknown"}]

    queries: List[Dict[str, Any]] = []
    remaining = text.strip()

    # Step 1: Extract leading math expression (if any)
    math_pattern = r"^\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s+"
    math_match = re.match(math_pattern, remaining)

    if math_match:
        # Found leading math - extract it
        math_expr = math_match.group(0).strip()
        queries.append({
            "text": math_expr,
            "type": "math"
        })
        # Remove the math expression from remaining text
        remaining = remaining[math_match.end():].strip()

    # Step 2: If there's remaining text, split on "and" connectors
    if remaining:
        # Split on "and" that appears between question words
        # Pattern: Split on "and" when preceded/followed by question-like patterns
        parts = _split_on_and_connector(remaining)

        for part in parts:
            part = part.strip()
            if part:
                # Determine type
                part_type = _classify_query_type(part)
                queries.append({
                    "text": part,
                    "type": part_type
                })

    # If no queries were extracted, return the original text
    if not queries:
        return [{"text": text, "type": _classify_query_type(text)}]

    return queries


def _split_on_and_connector(text: str) -> List[str]:
    """
    Split text on 'and' connectors while preserving context.

    Handles:
    - "what are dogs and what is lightning" -> ["what are dogs", "what is lightning"]
    - "dogs and cats" -> ["dogs and cats"] (single entity, not split)

    Args:
        text: Text to split

    Returns:
        List of text segments
    """
    # Pattern: Split on " and " when followed by a question word
    # This prevents splitting simple lists like "dogs and cats"
    question_words = r"\b(what|where|when|why|how|who|which|whose|whom)\b"

    # First, check if there are multiple question words
    question_matches = list(re.finditer(question_words, text, re.IGNORECASE))

    if len(question_matches) <= 1:
        # Only one or zero question words - don't split
        return [text]

    # Multiple question words found - split on "and" between them
    parts: List[str] = []
    current_start = 0

    # Find all " and " occurrences
    and_pattern = r"\s+and\s+"
    and_matches = list(re.finditer(and_pattern, text, re.IGNORECASE))

    if not and_matches:
        # No "and" found - return as-is
        return [text]

    # For each "and", check if it's between question phrases
    for and_match in and_matches:
        and_pos = and_match.start()

        # Check if there's a question word before and after this "and"
        has_question_before = any(qm.start() < and_pos for qm in question_matches)
        has_question_after = any(qm.start() > and_match.end() for qm in question_matches)

        if has_question_before and has_question_after:
            # Valid split point
            part = text[current_start:and_pos].strip()
            if part:
                parts.append(part)
            current_start = and_match.end()

    # Add remaining text
    if current_start < len(text):
        part = text[current_start:].strip()
        if part:
            parts.append(part)

    # If we didn't split anything, return original
    if not parts:
        return [text]

    return parts


def _classify_query_type(text: str) -> str:
    """
    Classify a query into a type.

    Args:
        text: Query text

    Returns:
        One of: "math", "question", "statement", "command", "unknown"
    """
    if not text:
        return "unknown"

    text_lower = text.strip().lower()

    # Check for pure math expression
    if re.match(r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*$", text):
        return "math"

    # Check for question
    if "?" in text:
        return "question"

    # Check for question words
    question_words = ["what", "where", "when", "why", "how", "who", "which", "whose", "whom"]
    if any(text_lower.startswith(word) for word in question_words):
        return "question"

    # Check for commands
    command_words = ["tell me", "show me", "give me", "explain", "describe"]
    if any(text_lower.startswith(cmd) for cmd in command_words):
        return "command"

    # Default to statement
    return "statement"


def should_process_as_multi_query(queries: List[Dict[str, Any]]) -> bool:
    """
    Determine if queries should be processed separately or as a single unit.

    Args:
        queries: List of query dicts from preprocess_query

    Returns:
        True if queries should be processed individually
    """
    if len(queries) <= 1:
        return False

    # Process separately if:
    # 1. There's a math expression mixed with questions
    # 2. There are multiple distinct questions

    has_math = any(q.get("type") == "math" for q in queries)
    has_questions = sum(1 for q in queries if q.get("type") in ["question", "command"]) > 0

    # If math mixed with questions, split
    if has_math and has_questions:
        return True

    # If multiple questions, split
    question_count = sum(1 for q in queries if q.get("type") in ["question", "command"])
    if question_count > 1:
        return True

    return False
