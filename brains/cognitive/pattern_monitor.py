"""
Pattern Monitor - Utilities for Viewing Brain Learning
======================================================

This module provides utilities to monitor how cognitive brain patterns
evolve over time through learning.

Features:
- List all patterns for a brain with scores/counts
- Compare pattern states before/after learning
- Generate learning reports
- Identify top-performing and struggling patterns
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from brains.cognitive.pattern_store import get_pattern_store, Pattern


def list_brain_patterns(brain_name: str, sort_by: str = "score") -> List[Dict[str, Any]]:
    """
    List all patterns for a brain with their learning statistics.

    Args:
        brain_name: Name of the brain (e.g., "integrator", "affect_priority")
        sort_by: How to sort ("score", "success_count", "failure_count", "signature")

    Returns:
        List of pattern dicts with stats
    """
    store = get_pattern_store()
    patterns = store.get_patterns_by_brain(brain_name)

    # Sort patterns
    if sort_by == "score":
        patterns.sort(key=lambda p: p.score, reverse=True)
    elif sort_by == "success_count":
        patterns.sort(key=lambda p: p.success_count, reverse=True)
    elif sort_by == "failure_count":
        patterns.sort(key=lambda p: p.failure_count, reverse=True)
    elif sort_by == "signature":
        patterns.sort(key=lambda p: p.signature)

    return [
        {
            "signature": p.signature,
            "score": round(p.score, 3),
            "success_count": p.success_count,
            "failure_count": p.failure_count,
            "total_uses": p.success_count + p.failure_count,
            "success_rate": (p.success_count / (p.success_count + p.failure_count)
                            if (p.success_count + p.failure_count) > 0 else 0.0),
            "frozen": p.frozen,
            "last_updated": p.last_updated
        }
        for p in patterns
    ]


def generate_learning_report(brain_name: str) -> str:
    """
    Generate a human-readable learning report for a brain.

    Args:
        brain_name: Name of the brain

    Returns:
        Formatted report string
    """
    patterns = list_brain_patterns(brain_name, sort_by="score")

    if not patterns:
        return f"No patterns found for brain '{brain_name}'"

    report = []
    report.append(f"\n{'='*70}")
    report.append(f"LEARNING REPORT: {brain_name.upper()}")
    report.append(f"{'='*70}\n")

    # Summary stats
    total_patterns = len(patterns)
    frozen_count = sum(1 for p in patterns if p["frozen"])
    learnable_count = total_patterns - frozen_count

    total_successes = sum(p["success_count"] for p in patterns)
    total_failures = sum(p["failure_count"] for p in patterns)
    total_uses = total_successes + total_failures

    avg_score = sum(p["score"] for p in patterns) / total_patterns if patterns else 0.0

    report.append(f"Summary:")
    report.append(f"  Total patterns: {total_patterns}")
    report.append(f"  Frozen (safety): {frozen_count}")
    report.append(f"  Learnable: {learnable_count}")
    report.append(f"  Total pattern uses: {total_uses}")
    report.append(f"  Total successes: {total_successes}")
    report.append(f"  Total failures: {total_failures}")
    report.append(f"  Overall success rate: {total_successes / total_uses * 100:.1f}%" if total_uses > 0 else "  Overall success rate: N/A")
    report.append(f"  Average pattern score: {avg_score:.3f}")

    # Top performers
    report.append(f"\nTop Performing Patterns:")
    top_patterns = [p for p in patterns if p["total_uses"] > 0][:5]
    for i, p in enumerate(top_patterns, 1):
        report.append(f"  {i}. {p['signature']}")
        report.append(f"     Score: {p['score']:.3f}, Uses: {p['total_uses']}, Success rate: {p['success_rate']*100:.1f}%")

    # Struggling patterns
    struggling = [p for p in patterns if p["total_uses"] > 0 and p["score"] < 0.2]
    if struggling:
        report.append(f"\nStruggling Patterns (score < 0.2):")
        for p in struggling:
            report.append(f"  - {p['signature']}: score={p['score']:.3f}, {p['failure_count']} failures")

    # Frozen patterns
    frozen_patterns = [p for p in patterns if p["frozen"]]
    if frozen_patterns:
        report.append(f"\nFrozen Safety Patterns:")
        for p in frozen_patterns:
            report.append(f"  - {p['signature']}: score={p['score']:.3f} (locked)")

    # All patterns table
    report.append(f"\nAll Patterns:")
    report.append(f"  {'Signature':<30} {'Score':>7} {'Succ':>5} {'Fail':>5} {'Rate':>6} {'Frozen':<7}")
    report.append(f"  {'-'*70}")
    for p in patterns:
        frozen_marker = "✓" if p["frozen"] else ""
        success_rate_pct = f"{p['success_rate']*100:.1f}%" if p["total_uses"] > 0 else "N/A"
        report.append(f"  {p['signature']:<30} {p['score']:>7.3f} {p['success_count']:>5} {p['failure_count']:>5} {success_rate_pct:>6} {frozen_marker:<7}")

    report.append(f"\n{'='*70}")

    return "\n".join(report)


def compare_all_brains() -> str:
    """
    Generate a comparison report across all brains.

    Returns:
        Formatted comparison report
    """
    brains = ["integrator", "affect_priority", "research_manager", "context_management"]

    report = []
    report.append(f"\n{'='*70}")
    report.append(f"CROSS-BRAIN LEARNING COMPARISON")
    report.append(f"{'='*70}\n")

    comparison_data = []
    for brain_name in brains:
        patterns = list_brain_patterns(brain_name)
        if patterns:
            total_uses = sum(p["total_uses"] for p in patterns)
            total_successes = sum(p["success_count"] for p in patterns)
            avg_score = sum(p["score"] for p in patterns) / len(patterns)
            success_rate = (total_successes / total_uses * 100) if total_uses > 0 else 0.0

            comparison_data.append({
                "brain": brain_name,
                "patterns": len(patterns),
                "uses": total_uses,
                "avg_score": avg_score,
                "success_rate": success_rate
            })

    if comparison_data:
        report.append(f"{'Brain':<25} {'Patterns':>10} {'Uses':>8} {'Avg Score':>12} {'Success Rate':>14}")
        report.append(f"{'-'*70}")

        for data in comparison_data:
            report.append(f"{data['brain']:<25} {data['patterns']:>10} {data['uses']:>8} {data['avg_score']:>12.3f} {data['success_rate']:>13.1f}%")

    report.append(f"\n{'='*70}")

    return "\n".join(report)


def get_pattern_history(brain_name: str, signature: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed history for a specific pattern.

    Args:
        brain_name: Name of the brain
        signature: Pattern signature

    Returns:
        Pattern details or None if not found
    """
    store = get_pattern_store()
    pattern = store.get_best_pattern(brain_name, signature, score_threshold=-1.0)

    if not pattern:
        return None

    return {
        "id": pattern.id,
        "brain": pattern.brain,
        "signature": pattern.signature,
        "context_tags": pattern.context_tags,
        "action": pattern.action,
        "score": pattern.score,
        "success_count": pattern.success_count,
        "failure_count": pattern.failure_count,
        "total_uses": pattern.success_count + pattern.failure_count,
        "success_rate": (pattern.success_count / (pattern.success_count + pattern.failure_count)
                        if (pattern.success_count + pattern.failure_count) > 0 else 0.0),
        "frozen": pattern.frozen,
        "last_updated": pattern.last_updated
    }
