"""
teacher_proposal.py
~~~~~~~~~~~~~~~~~~~

TeacherProposal schema for proposal-only mode.

CORE PRINCIPLE: Teacher as Advisor, Not Oracle
=============================================
Teacher never directly writes authoritative facts to memory banks.
Instead, it returns structured proposals that brains can evaluate and
decide whether to accept.

This module defines the TeacherProposal structure:
- candidate_questions: Questions Teacher suggests asking to clarify intent
- hypotheses: Statements Teacher believes might be true (with confidence)
- strategy_suggestions: Suggested approaches for handling the problem

The key change is that all proposals have:
- source = "teacher" (always)
- status = "proposal" (never "committed" from Teacher side)

Brains are responsible for:
1. Evaluating proposals against their own beliefs
2. Checking compatibility with existing knowledge
3. Deciding what to store and with what confidence
4. Promoting tentative facts to committed status after confirmation

Usage:
    from brains.cognitive.teacher.service.teacher_proposal import (
        TeacherProposal,
        CandidateQuestion,
        Hypothesis,
        StrategySuggestion,
        create_proposal_from_response,
    )

    # Teacher returns a proposal
    proposal = create_proposal_from_response(llm_response)

    # Brain evaluates and decides what to accept
    for hypothesis in proposal.hypotheses:
        if brain.is_compatible(hypothesis):
            brain.store_tentative(hypothesis)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import json


class HypothesisKind(str, Enum):
    """Classification of hypothesis type."""
    FACTUAL = "factual"              # World fact (e.g., "Paris is the capital of France")
    PROCEDURAL = "procedural"        # How-to knowledge (e.g., "To bake bread, first...")
    SELF_DESCRIPTION = "self_description"  # About Maven itself (SHOULD BE BLOCKED)
    CAPABILITY = "capability"        # About what Maven can do (SHOULD BE BLOCKED)
    USER_RELATED = "user_related"    # About the user
    PATTERN = "pattern"              # A reusable pattern/rule


class QuestionPurpose(str, Enum):
    """Purpose of a candidate question."""
    CLARIFY_USER_INTENT = "clarify_user_intent"  # Clarify what user meant
    PROBE_CAPABILITY = "probe_capability"         # Ask about capability
    CHECK_CONFLICT = "check_conflict"             # Check for belief conflicts
    GATHER_CONTEXT = "gather_context"             # Gather more context


@dataclass
class CandidateQuestion:
    """A question Teacher suggests asking to clarify/expand."""
    text: str
    purpose: QuestionPurpose

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "purpose": self.purpose.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateQuestion":
        return cls(
            text=data.get("text", ""),
            purpose=QuestionPurpose(data.get("purpose", "clarify_user_intent"))
        )


@dataclass
class Hypothesis:
    """A hypothesis that Teacher proposes (never committed directly).

    CRITICAL: status is ALWAYS "proposal" from Teacher.
    Only brains can promote hypotheses to "tentative" or "committed".
    """
    statement: str
    confidence: float  # 0.0 to 1.0
    kind: HypothesisKind
    source: str = "teacher"  # Always "teacher"
    status: str = "proposal"  # Always "proposal" from Teacher
    supporting_evidence: List[str] = field(default_factory=list)
    original_question: Optional[str] = None
    concept_key: Optional[str] = None

    def __post_init__(self):
        # Enforce proposal status from Teacher
        if self.source == "teacher":
            self.status = "proposal"
        # Clamp confidence to valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "confidence": self.confidence,
            "kind": self.kind.value,
            "source": self.source,
            "status": self.status,
            "supporting_evidence": self.supporting_evidence,
            "original_question": self.original_question,
            "concept_key": self.concept_key
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        return cls(
            statement=data.get("statement", ""),
            confidence=float(data.get("confidence", 0.5)),
            kind=HypothesisKind(data.get("kind", "factual")),
            source=data.get("source", "teacher"),
            status=data.get("status", "proposal"),
            supporting_evidence=data.get("supporting_evidence", []),
            original_question=data.get("original_question"),
            concept_key=data.get("concept_key")
        )

    def is_safe_to_evaluate(self) -> bool:
        """Check if this hypothesis can be evaluated by brains.

        CRITICAL: Self-description and capability hypotheses should
        NEVER be adopted from Teacher - they must come from self_model.
        """
        if self.kind in (HypothesisKind.SELF_DESCRIPTION, HypothesisKind.CAPABILITY):
            return False
        return True


@dataclass
class StrategySuggestion:
    """A strategy suggestion for handling a problem type."""
    for_problem_type: str  # e.g., "factual_query", "causal_explanation"
    description: str
    pattern_key: Optional[str] = None  # For pattern learning
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "for_problem_type": self.for_problem_type,
            "description": self.description,
            "pattern_key": self.pattern_key,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySuggestion":
        return cls(
            for_problem_type=data.get("for_problem_type", ""),
            description=data.get("description", ""),
            pattern_key=data.get("pattern_key"),
            confidence=float(data.get("confidence", 0.7))
        )


@dataclass
class TeacherProposal:
    """Complete proposal returned by Teacher.

    Teacher NEVER directly writes to memory. Instead, it returns this
    proposal structure for brains to evaluate and decide upon.

    Attributes:
        candidate_questions: Questions to ask for clarification
        hypotheses: Proposed facts/statements with confidence
        strategy_suggestions: Suggested approaches for the problem
        answer: Optional direct answer text (for display, not storage)
        raw_response: Original LLM response for debugging
        verdict: Summary of proposal quality
    """
    candidate_questions: List[CandidateQuestion] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    strategy_suggestions: List[StrategySuggestion] = field(default_factory=list)
    answer: Optional[str] = None
    raw_response: str = ""
    verdict: str = "PROPOSAL"  # PROPOSAL, NO_ANSWER, ERROR
    original_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_questions": [q.to_dict() for q in self.candidate_questions],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "strategy_suggestions": [s.to_dict() for s in self.strategy_suggestions],
            "answer": self.answer,
            "raw_response": self.raw_response,
            "verdict": self.verdict,
            "original_question": self.original_question
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherProposal":
        return cls(
            candidate_questions=[
                CandidateQuestion.from_dict(q)
                for q in data.get("candidate_questions", [])
            ],
            hypotheses=[
                Hypothesis.from_dict(h)
                for h in data.get("hypotheses", [])
            ],
            strategy_suggestions=[
                StrategySuggestion.from_dict(s)
                for s in data.get("strategy_suggestions", [])
            ],
            answer=data.get("answer"),
            raw_response=data.get("raw_response", ""),
            verdict=data.get("verdict", "PROPOSAL"),
            original_question=data.get("original_question")
        )

    def get_safe_hypotheses(self) -> List[Hypothesis]:
        """Get only hypotheses that are safe for brains to evaluate.

        Filters out self_description and capability hypotheses.
        """
        return [h for h in self.hypotheses if h.is_safe_to_evaluate()]

    def get_highest_confidence_hypothesis(self) -> Optional[Hypothesis]:
        """Get the hypothesis with highest confidence (if safe)."""
        safe_hypotheses = self.get_safe_hypotheses()
        if not safe_hypotheses:
            return None
        return max(safe_hypotheses, key=lambda h: h.confidence)

    def has_content(self) -> bool:
        """Check if proposal has any meaningful content."""
        return bool(
            self.hypotheses or
            self.strategy_suggestions or
            self.answer
        )


def classify_hypothesis_kind(statement: str, original_question: str = "") -> HypothesisKind:
    """Classify the kind of hypothesis based on content.

    CRITICAL: Detects self-description and capability statements
    so they can be blocked from being stored.
    """
    statement_lower = statement.lower()
    question_lower = original_question.lower()

    # Self-description patterns (about Maven/system)
    self_patterns = [
        "i am", "i'm", "my name is", "i was created", "i was built",
        "i do not have", "i cannot", "i am an ai", "i am a language model",
        "maven is", "maven can", "maven has", "this system"
    ]
    if any(p in statement_lower for p in self_patterns):
        return HypothesisKind.SELF_DESCRIPTION

    # Capability patterns
    capability_patterns = [
        "can search", "can browse", "can run", "can execute",
        "can control", "can access", "can read", "can write",
        "has access to", "is able to", "capable of"
    ]
    if any(p in statement_lower for p in capability_patterns):
        return HypothesisKind.CAPABILITY

    # User-related patterns
    user_patterns = ["user", "your", "you are", "you have"]
    if any(p in statement_lower for p in user_patterns):
        return HypothesisKind.USER_RELATED

    # Procedural patterns (how-to)
    procedural_patterns = [
        "to do this", "first", "then", "step", "process",
        "method", "approach", "technique"
    ]
    if any(p in statement_lower for p in procedural_patterns):
        return HypothesisKind.PROCEDURAL

    # Pattern patterns
    pattern_patterns = ["pattern", "rule", "principle", "guideline"]
    if any(p in statement_lower for p in pattern_patterns):
        return HypothesisKind.PATTERN

    # Default to factual
    return HypothesisKind.FACTUAL


def create_proposal_from_response(
    response_text: str,
    answer: Optional[str] = None,
    facts: Optional[List[Dict[str, Any]]] = None,
    patterns: Optional[List[Dict[str, Any]]] = None,
    original_question: str = ""
) -> TeacherProposal:
    """Create a TeacherProposal from raw Teacher response.

    This function converts the old-style Teacher response into the
    new proposal format, ensuring all hypotheses are marked as
    status="proposal" and source="teacher".

    Args:
        response_text: Raw LLM response
        answer: Extracted answer text
        facts: List of extracted facts from Teacher
        patterns: List of extracted patterns from Teacher
        original_question: The question that was asked

    Returns:
        TeacherProposal with all content properly tagged
    """
    hypotheses: List[Hypothesis] = []
    strategy_suggestions: List[StrategySuggestion] = []

    # Convert facts to hypotheses
    if facts:
        for fact in facts:
            statement = fact.get("statement", "")
            if not statement:
                continue

            kind = classify_hypothesis_kind(statement, original_question)

            hypothesis = Hypothesis(
                statement=statement,
                confidence=float(fact.get("confidence", 0.7)),
                kind=kind,
                source="teacher",
                status="proposal",  # ALWAYS proposal from Teacher
                original_question=original_question,
                concept_key=fact.get("concept_key")
            )
            hypotheses.append(hypothesis)

    # Convert patterns to strategy suggestions
    if patterns:
        for pattern in patterns:
            if isinstance(pattern, dict):
                description = pattern.get("pattern", str(pattern))
                pattern_key = pattern.get("pattern_key")
                problem_type = pattern.get("mode", "general")
            else:
                description = str(pattern)
                pattern_key = None
                problem_type = "general"

            suggestion = StrategySuggestion(
                for_problem_type=problem_type,
                description=description,
                pattern_key=pattern_key,
                confidence=0.7
            )
            strategy_suggestions.append(suggestion)

    # Determine verdict
    has_content = bool(hypotheses or strategy_suggestions or answer)
    verdict = "PROPOSAL" if has_content else "NO_ANSWER"

    return TeacherProposal(
        hypotheses=hypotheses,
        strategy_suggestions=strategy_suggestions,
        answer=answer,
        raw_response=response_text,
        verdict=verdict,
        original_question=original_question
    )


def create_empty_proposal(original_question: str = "", reason: str = "NO_ANSWER") -> TeacherProposal:
    """Create an empty proposal for cases where Teacher cannot help."""
    return TeacherProposal(
        verdict=reason,
        original_question=original_question
    )


def create_blocked_proposal(original_question: str, reason: str) -> TeacherProposal:
    """Create a proposal indicating the question was blocked.

    Used for self-identity and capability questions that Teacher
    should never answer.
    """
    return TeacherProposal(
        verdict="BLOCKED",
        original_question=original_question,
        raw_response=f"[BLOCKED: {reason}]"
    )


# Export public API
__all__ = [
    "TeacherProposal",
    "CandidateQuestion",
    "Hypothesis",
    "StrategySuggestion",
    "HypothesisKind",
    "QuestionPurpose",
    "create_proposal_from_response",
    "create_empty_proposal",
    "create_blocked_proposal",
    "classify_hypothesis_kind",
]
