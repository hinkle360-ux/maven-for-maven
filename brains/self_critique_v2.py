"""
Re-export of self_critique_v2 module for backward compatibility.

The actual implementation lives in brains/cognitive/self_dmn/self_critique_v2.py
This module provides a simpler import path: brains.self_critique_v2
"""

from brains.cognitive.self_dmn.self_critique_v2 import (
    # Data classes
    CritiqueIssue,
    CritiqueResult,
    Fact,
    FactDecision,
    VerificationResult,
    # Functions
    score_confidence,
    decide_fact,
    process_fact,
    run_self_critique,
    list_recent_critiques,
    get_critique_statistics,
    pipeline_stage_self_review,
    service_api,
    # Class
    SelfCritic,
    # Private functions exposed for testing
    _looks_like_teacher_self_hallucination,
)

__all__ = [
    "CritiqueIssue",
    "CritiqueResult",
    "Fact",
    "FactDecision",
    "VerificationResult",
    "score_confidence",
    "decide_fact",
    "process_fact",
    "run_self_critique",
    "list_recent_critiques",
    "get_critique_statistics",
    "pipeline_stage_self_review",
    "service_api",
    "SelfCritic",
    "_looks_like_teacher_self_hallucination",
]
