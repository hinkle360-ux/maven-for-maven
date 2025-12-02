"""
Routing Examples Store
======================

This module stores routing examples for learning from mistakes and
providing few-shot examples to the router LLM.

Each routing example contains:
- input: The original user message
- intent: Detected intent
- tools: Tools used (or that should have been used)
- brains: Brains routed to
- verdict: ok / wrong_tool / wrong_brain / overkill / underkill
- source: grammar / router_llm / learned / feedback

Examples with positive verdicts ("ok") can be used as few-shot examples.
Examples with negative verdicts are used to train routing patterns.

Storage:
- Examples are stored in reports/routing_examples.jsonl
- Recent examples are loaded at startup for the router LLM
- Patterns are extracted and fed to routing_learning

Usage:
    from brains.cognitive.routing_examples import (
        store_routing_example,
        get_recent_examples,
        get_examples_by_intent,
        record_routing_feedback,
    )

    # Store a successful routing
    store_routing_example(
        input_text="x grok hello",
        intent="browser_tool",
        tools=["x"],
        brains=["language"],
        verdict="ok",
        source="grammar"
    )

    # Record feedback on a routing mistake
    record_routing_feedback(
        input_text="x grok hello",
        actual_tools=[],
        actual_brains=["language"],
        correct_tools=["x"],
        correct_brains=["language"],
        verdict="wrong_tool"
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from brains.maven_paths import get_reports_path


@dataclass
class RoutingExample:
    """A single routing example."""

    input_text: str
    normalized_text: str = ""
    intent: str = ""
    tools: List[str] = field(default_factory=list)
    brains: List[str] = field(default_factory=list)
    subcommand: Optional[str] = None
    verdict: str = "ok"  # ok, wrong_tool, wrong_brain, overkill, underkill
    source: str = "unknown"  # grammar, router_llm, learned, feedback
    confidence: float = 1.0
    timestamp: str = ""
    signature: str = ""  # Normalized signature for pattern matching
    is_gold: bool = False  # True if from grammar (always correct)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.normalized_text:
            import re
            self.normalized_text = re.sub(r'\s+', ' ', self.input_text.strip().lower())
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute a routing signature for pattern matching."""
        parts = []

        # Add intent
        if self.intent:
            parts.append(f"intent:{self.intent}")

        # Add tool indicators
        if self.tools:
            parts.append(f"tools:{','.join(sorted(self.tools))}")

        # Add key words from input
        words = self.normalized_text.split()[:5]  # First 5 words
        if words:
            parts.append(f"words:{','.join(words)}")

        return "|".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingExample":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _get_examples_path() -> Path:
    """Get path to routing examples file."""
    return get_reports_path("routing_examples.jsonl")


def store_routing_example(
    input_text: str,
    intent: str = "",
    tools: Optional[List[str]] = None,
    brains: Optional[List[str]] = None,
    subcommand: Optional[str] = None,
    verdict: str = "ok",
    source: str = "unknown",
    confidence: float = 1.0,
    is_gold: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingExample:
    """
    Store a routing example.

    Args:
        input_text: The user's original message
        intent: Detected intent (e.g., "browser_tool", "research")
        tools: List of tools used
        brains: List of brains routed to
        subcommand: Optional subcommand (e.g., "grok" for x tool)
        verdict: Result of routing (ok, wrong_tool, wrong_brain, etc.)
        source: Where this routing came from (grammar, router_llm, etc.)
        confidence: Confidence score
        is_gold: True if from grammar (always correct)
        metadata: Additional metadata

    Returns:
        The stored RoutingExample
    """
    example = RoutingExample(
        input_text=input_text,
        intent=intent,
        tools=tools or [],
        brains=brains or [],
        subcommand=subcommand,
        verdict=verdict,
        source=source,
        confidence=confidence,
        is_gold=is_gold,
        metadata=metadata or {},
    )

    # Append to file
    try:
        path = _get_examples_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example.to_dict()) + "\n")

    except Exception as e:
        print(f"[ROUTING_EXAMPLES] Error storing example: {e}")

    return example


def get_recent_examples(
    limit: int = 50,
    verdict_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[RoutingExample]:
    """
    Get recent routing examples.

    Args:
        limit: Maximum number of examples to return
        verdict_filter: Only return examples with this verdict
        source_filter: Only return examples from this source

    Returns:
        List of RoutingExample objects
    """
    examples = []

    try:
        path = _get_examples_path()
        if not path.exists():
            return []

        # Read all lines (we'll optimize later for large files)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse from end (most recent first)
        for line in reversed(lines):
            if len(examples) >= limit:
                break

            try:
                data = json.loads(line.strip())
                example = RoutingExample.from_dict(data)

                # Apply filters
                if verdict_filter and example.verdict != verdict_filter:
                    continue
                if source_filter and example.source != source_filter:
                    continue

                examples.append(example)

            except (json.JSONDecodeError, KeyError):
                continue

    except Exception as e:
        print(f"[ROUTING_EXAMPLES] Error loading examples: {e}")

    return examples


def get_examples_by_intent(intent: str, limit: int = 20) -> List[RoutingExample]:
    """
    Get examples with a specific intent.

    Args:
        intent: The intent to filter by
        limit: Maximum number of examples

    Returns:
        List of matching examples
    """
    examples = []

    try:
        path = _get_examples_path()
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("intent") == intent:
                        examples.append(RoutingExample.from_dict(data))
                        if len(examples) >= limit:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue

    except Exception as e:
        print(f"[ROUTING_EXAMPLES] Error loading examples: {e}")

    return examples


def get_gold_examples(limit: int = 20) -> List[RoutingExample]:
    """
    Get gold examples (from grammar, always correct).

    These are ideal for few-shot learning.

    Args:
        limit: Maximum number of examples

    Returns:
        List of gold examples
    """
    return get_recent_examples(limit=limit, source_filter="grammar")


def record_routing_feedback(
    input_text: str,
    actual_tools: List[str],
    actual_brains: List[str],
    correct_tools: Optional[List[str]] = None,
    correct_brains: Optional[List[str]] = None,
    verdict: str = "wrong_tool",
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingExample:
    """
    Record feedback on a routing mistake.

    This creates a correction example that can be used to improve routing.

    Args:
        input_text: The original user message
        actual_tools: Tools that were actually used
        actual_brains: Brains that were actually routed to
        correct_tools: What tools should have been used
        correct_brains: What brains should have been routed to
        verdict: Type of error (wrong_tool, wrong_brain, overkill, underkill)
        metadata: Additional context

    Returns:
        The stored correction example
    """
    correction_metadata = {
        "actual_tools": actual_tools,
        "actual_brains": actual_brains,
        "is_correction": True,
        **(metadata or {}),
    }

    return store_routing_example(
        input_text=input_text,
        intent="correction",
        tools=correct_tools or [],
        brains=correct_brains or [],
        verdict=verdict,
        source="feedback",
        confidence=1.0,
        is_gold=False,
        metadata=correction_metadata,
    )


def get_few_shot_examples(limit: int = 8) -> List[Dict[str, Any]]:
    """
    Get examples formatted for few-shot prompting.

    Prioritizes:
    1. Gold examples (from grammar)
    2. OK examples from router_llm
    3. Recent OK examples from any source

    Args:
        limit: Maximum number of examples

    Returns:
        List of dicts with "input" and "output" keys
    """
    few_shot = []

    # First, get gold examples
    gold = get_gold_examples(limit=limit // 2)
    for ex in gold:
        few_shot.append({
            "input": ex.input_text,
            "output": {
                "brains": ex.brains,
                "tools": ex.tools,
                "subcommand": ex.subcommand,
                "confidence": ex.confidence,
                "reason": f"Grammar pattern: {ex.source}"
            }
        })

    # Then get other OK examples
    remaining = limit - len(few_shot)
    if remaining > 0:
        ok_examples = get_recent_examples(limit=remaining * 2, verdict_filter="ok")
        for ex in ok_examples:
            if len(few_shot) >= limit:
                break
            # Avoid duplicates
            if any(f["input"] == ex.input_text for f in few_shot):
                continue
            few_shot.append({
                "input": ex.input_text,
                "output": {
                    "brains": ex.brains,
                    "tools": ex.tools,
                    "subcommand": ex.subcommand,
                    "confidence": ex.confidence,
                    "reason": f"Source: {ex.source}"
                }
            })

    return few_shot[:limit]


def get_routing_stats() -> Dict[str, Any]:
    """
    Get statistics about routing examples.

    Returns:
        Dict with counts by verdict, source, intent
    """
    stats = {
        "total": 0,
        "by_verdict": {},
        "by_source": {},
        "by_intent": {},
        "gold_count": 0,
        "error_rate": 0.0,
    }

    try:
        path = _get_examples_path()
        if not path.exists():
            return stats

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    stats["total"] += 1

                    verdict = data.get("verdict", "unknown")
                    source = data.get("source", "unknown")
                    intent = data.get("intent", "unknown")

                    stats["by_verdict"][verdict] = stats["by_verdict"].get(verdict, 0) + 1
                    stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
                    stats["by_intent"][intent] = stats["by_intent"].get(intent, 0) + 1

                    if data.get("is_gold"):
                        stats["gold_count"] += 1

                except (json.JSONDecodeError, KeyError):
                    continue

        # Calculate error rate
        if stats["total"] > 0:
            errors = sum(
                stats["by_verdict"].get(v, 0)
                for v in ["wrong_tool", "wrong_brain", "overkill", "underkill"]
            )
            stats["error_rate"] = errors / stats["total"]

    except Exception as e:
        print(f"[ROUTING_EXAMPLES] Error computing stats: {e}")

    return stats


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RoutingExample",
    "store_routing_example",
    "get_recent_examples",
    "get_examples_by_intent",
    "get_gold_examples",
    "record_routing_feedback",
    "get_few_shot_examples",
    "get_routing_stats",
]
