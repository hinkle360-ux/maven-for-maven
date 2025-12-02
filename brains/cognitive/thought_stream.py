"""
Thought Stream Synthesis
=======================

This module provides a simple synthesiser for combining thought
fragments produced by different brains into a coherent final answer.
It is intended as a replacement for Stage 10 in the roadmap where the
language, reasoning and memory outputs are merged.  The current
implementation collects fragments from the pipeline context, resolves
contradictions (naively) and concatenates them into a single response.

Future iterations could implement more sophisticated conflict
resolution and rhetorical structuring, but this basic version
suffices for demonstration.
"""

from __future__ import annotations

from typing import Dict, Any, List


def _resolve_contradictions(fragments: List[str]) -> List[str]:
    """Naively resolve contradictions between fragments.

    The current heuristic simply removes duplicate lines and
    discards empty fragments.  A more advanced version might use
    semantic similarity or weighting to select the most plausible
    statements.  Here we preserve order while deduplicating.

    Args:
        fragments: A list of thought fragment strings.
    Returns:
        A list of unique, non‑empty fragments.
    """
    seen = set()
    resolved: List[str] = []
    for frag in fragments:
        if not frag:
            continue
        text = frag.strip()
        if text and text not in seen:
            resolved.append(text)
            seen.add(text)
    return resolved


def stage_10_synthesize(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Combine thought fragments from different stages into a final answer.

    This function expects the pipeline context to contain keys such
    as ``stage_6_language_fragment``, ``stage_8_reasoning_fragment``
    and ``stage_2R_memory_fragment``.  It collects these strings,
    resolves basic contradictions and concatenates them with spaces.

    Args:
        ctx: The pipeline context dictionary.
    Returns:
        A dictionary with a single key ``answer`` containing the
        synthesised response.
    """
    fragments: List[str] = []
    try:
        frag = ctx.get("stage_6_language_fragment")
        if isinstance(frag, str):
            fragments.append(frag)
    except Exception:
        pass
    try:
        frag = ctx.get("stage_8_reasoning_fragment")
        if isinstance(frag, str):
            fragments.append(frag)
    except Exception:
        pass
    try:
        frag = ctx.get("stage_2R_memory_fragment")
        if isinstance(frag, str):
            fragments.append(frag)
    except Exception:
        pass
    resolved = _resolve_contradictions(fragments)
    final_text = " ".join(resolved).strip()
    return {"answer": final_text}