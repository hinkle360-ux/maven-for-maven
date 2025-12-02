"""
Inference Engine
================

This module provides a simple inference engine for performing basic
logical reasoning over a set of facts.  It attempts to connect
multiple pieces of knowledge to answer queries that cannot be
resolved directly from a single fact.  The implementation is
intentionally lightweight and relies on string matching; future
versions could integrate a proper logic framework or knowledge graph.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple


def find_reasoning_chains(query: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify potential reasoning chains for the query.

    This stub scans the provided facts for statements containing
    overlapping keywords with the query.  It returns a list of
    candidate chains, each with a simple confidence score and a
    conclusion.  The logic is deliberately shallow and does not
    perform true logical inference.

    Args:
        query: The user query string.
        facts: A list of fact dictionaries with a 'content' field.
    Returns:
        A list of reasoning chain dictionaries.
    """
    chains: List[Dict[str, Any]] = []
    q = (query or "").lower()
    keywords = [w for w in q.split() if len(w) > 2]
    for fact in facts:
        try:
            content = str(fact.get("content", "")).lower()
        except Exception:
            content = ""
        if not content:
            continue
        shared = [kw for kw in keywords if kw in content]
        if shared:
            # Confidence proportional to number of shared keywords
            conf = min(1.0, 0.5 + 0.1 * len(shared))
            chains.append({
                "conclusion": content,
                "confidence": conf,
                "reasoning_steps": [content],
            })
    return chains


def attempt_inference(query: str, facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Attempt to derive an answer from multiple facts.

    Finds reasoning chains and returns the highest confidence result
    above a threshold.  Returns None if no suitable chain is found.

    Args:
        query: The user query.
        facts: A list of relevant fact dictionaries.
    Returns:
        A dictionary with keys 'answer', 'confidence' and 'steps',
        or None if no chain meets the confidence threshold.
    """
    chains = find_reasoning_chains(query, facts)
    # Select the chain with the highest confidence score
    best: Optional[Dict[str, Any]] = None
    for chain in chains:
        try:
            conf = float(chain.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if (best is None) or (conf > float(best.get("confidence", 0.0) or 0.0)):
            best = chain
    # Helper to classify a fact into a semantic role based on keyword heuristics
    def _classify_role(fact_text: str) -> str:
        try:
            txt = (fact_text or "").lower()
        except Exception:
            txt = ""
        mech_keywords = ["two", "three", "light", "dark", "stages", "stage", "phase", "cycle", "dependent", "independent"]
        type_keywords = ["c3", "c4", "cam", "types", "forms", "kind"]
        for kw in type_keywords:
            if kw in txt:
                return "types"
        for kw in mech_keywords:
            if kw in txt:
                return "mechanism"
        return "definition"
    # Accept chains with moderate confidence (>=0.5) rather than requiring
    # very high certainty.  This promotes multi‑hop reasoning on thin
    # evidence and surfaces tentative explanations when direct answers are
    # unavailable.  When accepted, annotate each step with its role.
    if best and float(best.get("confidence", 0.0) or 0.0) >= 0.5:
        steps = list(best.get("reasoning_steps", []))
        trace: List[Dict[str, Any]] = []
        for s in steps:
            role = _classify_role(str(s))
            trace.append({"fact": s, "role": role})
        return {
            "answer": best.get("conclusion"),
            "confidence": best.get("confidence"),
            "steps": steps,
            "trace": trace,
        }
    return None


# -----------------------------------------------------------------------------
# Next‑Action Prediction
#
# In addition to deriving answers from facts, the inference engine can
# provide a rudimentary next‑action predictor.  Given a pipeline context,
# it examines the current state and returns a list of candidate actions
# that the system could take to make progress on the user query.  This is
# intentionally simple and meant for demonstration; real planning would
# integrate with an action engine and task decomposer.

from typing import Dict, Any


def predict_next_steps(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Suggest the next logical steps based on the pipeline context.

    The heuristic looks for common failure modes (e.g. unanswered
    queries, low confidence) and proposes actions such as repeating
    searches, invoking external tools or asking clarifying questions.
    Each action is represented as a dictionary with a ``type`` and
    optional ``details``.

    Args:
        ctx: The current pipeline context.
    Returns:
        A list of candidate actions to execute.  The list may be
        empty if no follow‑up is necessary.
    """
    actions: List[Dict[str, Any]] = []
    try:
        # If the query was unanswered, suggest performing a broader search
        verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        if verdict == "UNANSWERED":
            actions.append({
                "type": "RETRY_SEARCH",
                "details": {
                    "breadth": "broad",
                    "reason": "Initial query unanswered"
                }
            })
        # If final confidence is low, suggest cross‑checking with another brain
        conf = float((ctx.get("stage_10_finalize") or {}).get("confidence", 0.0) or 0.0)
        if conf < 0.5:
            actions.append({
                "type": "CROSS_CHECK",
                "details": {
                    "target": "reasoning",
                    "reason": "Low confidence in current answer"
                }
            })
        # If memory search returned few results, propose expanding domains
        mem_results = ctx.get("stage_2R_memory_fragment") or ""
        if isinstance(mem_results, str) and len(mem_results.split()) < 5:
            actions.append({
                "type": "EXPAND_SEARCH_DOMAIN",
                "details": {
                    "domains": ["general"],
                    "reason": "Few memory results obtained"
                }
            })
    except Exception:
        # Fall back to an empty action list on error
        return []
    return actions