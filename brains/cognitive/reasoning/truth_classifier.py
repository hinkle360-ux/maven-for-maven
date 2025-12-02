"""
truth_classifier.py
~~~~~~~~~~~~~~~~~~~

Centralized truth/guess classification for Maven's reasoning system.

P0 GOVERNANCE REQUIREMENT:
This module provides the ONLY authorized way to classify statements into
truth categories. All brains that produce "final answers" must use this
classification system.

TRUTH TYPES:
- FACT: High confidence (>= 0.7), backed by evidence or common sense
- EDUCATED: Moderate confidence (0.4-0.7), reasonable inference
- UNKNOWN: Low confidence (< 0.4), insufficient evidence
- RANDOM: Pure guess, requires explicit permission flag

MEMORY GOVERNANCE:
- FACT: Normal write to memory tiers
- EDUCATED: Write with low confidence tag
- UNKNOWN: Write to STM only, do not persist
- RANDOM: Block write to STM unless explicitly allowed for ephemeral scratch

STDLIB ONLY, OFFLINE, WINDOWS 10 COMPATIBLE
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Literal
from enum import Enum


class TruthType(Enum):
    """Truth classification categories."""
    FACT = "FACT"
    EDUCATED = "EDUCATED"
    UNKNOWN = "UNKNOWN"
    RANDOM = "RANDOM"


class TruthClassifier:
    """
    Centralized classifier for truth/guess categorization.

    This is the ONLY authorized way to classify truth in Maven.
    All cognitive brains that produce final answers must use this.
    """

    # Confidence thresholds for classification
    FACT_THRESHOLD = 0.7
    EDUCATED_THRESHOLD = 0.4

    @staticmethod
    def classify(
        content: str,
        confidence: float,
        evidence: Optional[Dict[str, Any]] = None,
        allow_random: bool = False
    ) -> Dict[str, Any]:
        """
        Classify a statement into a truth category.

        Args:
            content: The statement or answer to classify
            confidence: Confidence score (0.0-1.0)
            evidence: Optional evidence supporting the statement
            allow_random: If True, allow RANDOM classification (default: False)

        Returns:
            Classification dict with:
                - type: TruthType (FACT/EDUCATED/UNKNOWN/RANDOM)
                - confidence: float
                - rationale: str explaining the classification
                - allow_memory_write: bool (can this be written to memory?)
                - memory_tier: Optional[str] (which tier if writable)

        Raises:
            ValueError: If classification parameters are invalid
        """
        # Validate inputs
        if not isinstance(content, str) or not content.strip():
            return TruthClassifier._make_classification(
                truth_type=TruthType.UNKNOWN,
                confidence=0.0,
                rationale="Empty or invalid content",
                allow_memory_write=False
            )

        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return TruthClassifier._make_classification(
                truth_type=TruthType.UNKNOWN,
                confidence=0.0,
                rationale="Invalid confidence score",
                allow_memory_write=False
            )

        # Check for random guess (confidence near 0, no evidence)
        if confidence < 0.1 and not evidence:
            if not allow_random:
                # Block random guesses unless explicitly allowed
                return TruthClassifier._make_classification(
                    truth_type=TruthType.UNKNOWN,
                    confidence=confidence,
                    rationale="Random guess blocked (no permission flag)",
                    allow_memory_write=False
                )
            else:
                return TruthClassifier._make_classification(
                    truth_type=TruthType.RANDOM,
                    confidence=confidence,
                    rationale="Random guess (ephemeral only)",
                    allow_memory_write=False  # Never write random guesses
                )

        # Classify based on confidence and evidence
        has_evidence = evidence and len(evidence.get("results", [])) > 0

        if confidence >= TruthClassifier.FACT_THRESHOLD:
            return TruthClassifier._make_classification(
                truth_type=TruthType.FACT,
                confidence=confidence,
                rationale=f"High confidence ({confidence:.2f})" +
                         (" with supporting evidence" if has_evidence else ""),
                allow_memory_write=True,
                memory_tier="factual"
            )

        elif confidence >= TruthClassifier.EDUCATED_THRESHOLD:
            return TruthClassifier._make_classification(
                truth_type=TruthType.EDUCATED,
                confidence=confidence,
                rationale=f"Moderate confidence ({confidence:.2f}), educated inference",
                allow_memory_write=True,
                memory_tier="working_theories"
            )

        else:
            return TruthClassifier._make_classification(
                truth_type=TruthType.UNKNOWN,
                confidence=confidence,
                rationale=f"Low confidence ({confidence:.2f}), insufficient evidence",
                allow_memory_write=True,  # Can write to STM only
                memory_tier="stm_only"
            )

    @staticmethod
    def _make_classification(
        truth_type: TruthType,
        confidence: float,
        rationale: str,
        allow_memory_write: bool,
        memory_tier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized classification result.

        Returns:
            Dict with classification details
        """
        return {
            "type": truth_type.value,
            "confidence": confidence,
            "rationale": rationale,
            "allow_memory_write": allow_memory_write,
            "memory_tier": memory_tier,
            "classifier_version": "1.0.0"
        }

    @staticmethod
    def should_store_in_memory(classification: Dict[str, Any]) -> bool:
        """
        Determine if a classification allows memory storage.

        Args:
            classification: Result from classify()

        Returns:
            True if this can be stored in memory tiers
        """
        return classification.get("allow_memory_write", False)

    @staticmethod
    def get_memory_tier(classification: Dict[str, Any]) -> Optional[str]:
        """
        Get the appropriate memory tier for a classification.

        Args:
            classification: Result from classify()

        Returns:
            Tier name (factual/working_theories/stm_only) or None
        """
        if not TruthClassifier.should_store_in_memory(classification):
            return None
        return classification.get("memory_tier")


def classify_truth(
    content: str,
    confidence: float,
    evidence: Optional[Dict[str, Any]] = None,
    allow_random: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for truth classification.

    This is the primary API that all brains should use.

    Args:
        content: Statement to classify
        confidence: Confidence score (0.0-1.0)
        evidence: Optional supporting evidence
        allow_random: Allow RANDOM classification (default: False)

    Returns:
        Classification dict from TruthClassifier.classify()
    """
    return TruthClassifier.classify(
        content=content,
        confidence=confidence,
        evidence=evidence,
        allow_random=allow_random
    )


# Export key components
__all__ = [
    "TruthType",
    "TruthClassifier",
    "classify_truth"
]
