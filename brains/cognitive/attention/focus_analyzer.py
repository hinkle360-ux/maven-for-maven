"""
Focus Analyzer
==============

This module collects simple statistics about which cognitive brain
receives focus during the attention arbitration stage (Stage 5b).
By recording the winning focus and its accompanying reason, the
analyzer can later provide a summary of how frequently each brain
is selected and what justifications are most common.  These
statistics are intended for offline analysis and adaptive
strategies rather than real‑time decision making.  They are kept
in memory for the duration of the process and are not persisted
automatically.

Usage examples::

    from brains.cognitive.attention.focus_analyzer import update_focus_stats, get_focus_summary
    update_focus_stats("language", "unanswered_question")
    summary = get_focus_summary()
    # {'language': {'count': 1, 'proportion': 1.0, 'top_reasons': [('unanswered_question', 1)]}}

This lightweight helper adds no external dependencies and can be
safely ignored by callers that do not wish to analyse attention
patterns.  If persistence or more sophisticated analytics are
required, this module can be extended in a future upgrade.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning attention management and priority strategies
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("attention")
except Exception as e:
    print(f"[ATTENTION] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize memory at module level for reuse
_memory = BrainMemory("attention")

# Internal dictionary keyed by brain name.  Each entry stores a
# count of how many times the brain received focus and a nested
# dictionary of reasons with their occurrence counts.
_focus_stats: Dict[str, Dict[str, Any]] = {}


def update_focus_stats(focus: str, reason: str | None = None) -> None:
    """Record a focus win for the given brain.

    When a brain wins the attention bidding process, call this
    function with the brain name and (optionally) the reason for
    winning.  Both values should be lowercase strings for
    consistency.  The internal counters are incremented and stored
    in ``_focus_stats``.

    Args:
        focus: The name of the brain that received focus, e.g.
            ``"language"`` or ``"reasoning"``.  If falsy, the call
            is ignored.
        reason: A short string describing why the brain was chosen.
            May be ``None`` or empty to indicate no reason provided.
    """
    try:
        if not focus:
            return
        key = str(focus).strip().lower()
        info = _focus_stats.get(key, {"count": 0, "reasons": {}})
        # Increment total count
        info["count"] = int(info.get("count", 0)) + 1
        # Update reason counts if provided
        if reason:
            rkey = str(reason).strip().lower()
            reasons = info.get("reasons", {})
            reasons[rkey] = int(reasons.get(rkey, 0)) + 1
            info["reasons"] = reasons
        _focus_stats[key] = info

        # Store focus event in memory for pattern learning
        if _memory:
            try:
                _memory.store(
                    content={"focus": key, "reason": rkey if reason else "no_reason"},
                    metadata={"kind": "focus_event", "source": "attention", "confidence": 0.9}
                )
            except Exception:
                pass

        # Learn from focus patterns periodically
        total_focuses = sum(int(v.get("count", 0)) for v in _focus_stats.values())
        if _teacher_helper and total_focuses >= 50 and total_focuses % 50 == 0:
            try:
                # Check for learned attention strategies
                learned_patterns = _memory.retrieve(
                    query="attention strategy pattern",
                    limit=3,
                    tiers=["stm", "mtm", "ltm"]
                )

                has_learned_strategy = any(
                    p.get("kind") == "learned_pattern" and p.get("confidence", 0) >= 0.7
                    for p in learned_patterns
                )

                if not has_learned_strategy:
                    print(f"[ATTENTION] Reached {total_focuses} focus events, calling Teacher...")
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"What attention strategies should I use based on focus patterns: {_focus_stats}?",
                        context={
                            "focus_stats": _focus_stats,
                            "total_focuses": total_focuses,
                            "current_focus": key,
                            "current_reason": rkey if reason else None
                        },
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[ATTENTION] Learned from Teacher: {patterns_stored} attention strategies stored")
            except Exception as e:
                print(f"[ATTENTION] Teacher call failed: {str(e)[:100]}")

    except Exception:
        # Do not propagate errors – stats are best effort only
        pass


def get_focus_summary() -> Dict[str, Any]:
    """Return a summary of focus statistics.

    The summary contains, for each brain that has received focus at
    least once, the total number of wins, the proportion of wins
    relative to the total, and a list of the top three reasons
    ranked by frequency.  If no statistics have been recorded,
    returns an empty dictionary.

    Returns:
        A dictionary keyed by brain names with summary statistics.
    """
    try:
        total = sum(int(info.get("count", 0)) for info in _focus_stats.values())
        if total <= 0:
            return {}
        summary: Dict[str, Any] = {}
        for brain, info in _focus_stats.items():
            count = int(info.get("count", 0))
            # Compute proportion; protect against divide by zero
            proportion = (count / total) if total else 0.0
            # Sort reasons by descending count
            reasons: Dict[str, int] = info.get("reasons", {})
            top_reasons: List[Tuple[str, int]] = sorted(
                reasons.items(), key=lambda kv: kv[1], reverse=True
            )[:3]
            summary[brain] = {
                "count": count,
                "proportion": proportion,
                "top_reasons": top_reasons,
            }
        return summary
    except Exception:
        # On error, return empty to avoid breaking callers
        return {}