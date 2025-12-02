"""
Tests for the Self-Critique System
===================================

Tests that verify:
1. Fact scoring based on source
2. Teacher hallucination detection
3. Fact decision making
4. External verification hook
5. Full process_fact pipeline
"""

import pytest
import sys
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestFactScoring:
    """Test suite for fact confidence scoring."""

    def test_self_model_highest_confidence(self):
        """Test that self_model source gets highest confidence."""
        from brains.self_critique_v2 import Fact, score_confidence

        fact = Fact(
            content="Maven is an AI assistant",
            source="self_model",
            domain="factual"
        )

        score = score_confidence(fact)

        assert score == 1.0

    def test_user_source_confidence(self):
        """Test user source confidence level."""
        from brains.self_critique_v2 import Fact, score_confidence

        fact = Fact(
            content="The sky is blue",
            source="user",
            domain="factual"
        )

        score = score_confidence(fact)

        assert score == 0.8

    def test_teacher_source_confidence(self):
        """Test teacher source confidence level."""
        from brains.self_critique_v2 import Fact, score_confidence

        fact = Fact(
            content="Python was created by Guido van Rossum",
            source="teacher",
            domain="factual"
        )

        score = score_confidence(fact)

        assert score == 0.6

    def test_unknown_source_confidence(self):
        """Test unknown source confidence level."""
        from brains.self_critique_v2 import Fact, score_confidence

        fact = Fact(
            content="Some fact",
            source="unknown",
            domain="factual"
        )

        score = score_confidence(fact)

        assert score == 0.5


class TestTeacherHallucinationDetection:
    """Test suite for detecting teacher self-hallucination."""

    def test_detects_training_data_mention(self):
        """Test detection of 'my training data' phrase."""
        from brains.self_critique_v2 import _looks_like_teacher_self_hallucination

        result = _looks_like_teacher_self_hallucination(
            "Based on my training data, the answer is..."
        )

        assert result is True

    def test_detects_knowledge_cutoff(self):
        """Test detection of 'my knowledge cutoff' phrase."""
        from brains.self_critique_v2 import _looks_like_teacher_self_hallucination

        result = _looks_like_teacher_self_hallucination(
            "My knowledge cutoff is 2023"
        )

        assert result is True

    def test_detects_chatgpt_mention(self):
        """Test detection of 'ChatGPT' mention."""
        from brains.self_critique_v2 import _looks_like_teacher_self_hallucination

        result = _looks_like_teacher_self_hallucination(
            "As ChatGPT, I can help you with..."
        )

        assert result is True

    def test_clean_text_passes(self):
        """Test that clean text is not flagged."""
        from brains.self_critique_v2 import _looks_like_teacher_self_hallucination

        result = _looks_like_teacher_self_hallucination(
            "The Earth orbits the Sun"
        )

        assert result is False


class TestFactDecision:
    """Test suite for fact decision making."""

    def test_accept_high_confidence(self):
        """Test that high confidence facts are accepted."""
        from brains.self_critique_v2 import Fact, decide_fact

        fact = Fact(
            content="Water boils at 100°C at sea level",
            source="self_model",
            domain="factual"
        )

        decision = decide_fact(fact)

        assert decision.decision == "accept"
        assert decision.confidence >= 0.7

    def test_reject_teacher_hallucination(self):
        """Test that teacher hallucination is rejected."""
        from brains.self_critique_v2 import Fact, decide_fact

        fact = Fact(
            content="As a large language model, I was trained by OpenAI",
            source="teacher",
            domain="factual"
        )

        decision = decide_fact(fact)

        assert decision.decision == "reject"
        assert decision.reprimand_teacher is True

    def test_verify_first_low_confidence(self):
        """Test that low confidence facts need verification."""
        from brains.self_critique_v2 import Fact, decide_fact

        fact = Fact(
            content="Some uncertain claim",
            source="teacher",
            domain="factual"
        )

        decision = decide_fact(fact)

        # Teacher facts that aren't hallucinations get 0.6 confidence
        # which is below 0.7 threshold, so should be verify_first
        assert decision.decision == "verify_first"


class TestVerificationResult:
    """Test suite for verification result structure."""

    def test_verification_result_structure(self):
        """Test VerificationResult dataclass."""
        from brains.self_critique_v2 import VerificationResult

        result = VerificationResult(
            status="SUPPORTS",
            confidence=0.8,
            evidence_snippets=["Evidence 1", "Evidence 2"]
        )

        assert result.status == "SUPPORTS"
        assert result.confidence == 0.8
        assert len(result.evidence_snippets) == 2


class TestProcessFact:
    """Test suite for full fact processing pipeline."""

    def test_process_accepts_self_model(self):
        """Test that self_model facts are accepted."""
        from brains.self_critique_v2 import Fact, process_fact

        fact = Fact(
            content="Maven is designed to be helpful",
            source="self_model",
            domain="personal"
        )

        decision = process_fact(fact)

        assert decision.decision == "accept"

    def test_process_rejects_hallucination(self):
        """Test that teacher hallucinations are rejected."""
        from brains.self_critique_v2 import Fact, process_fact

        fact = Fact(
            content="I was created by OpenAI and my name is GPT-4",
            source="teacher",
            domain="factual"
        )

        decision = process_fact(fact)

        assert decision.decision == "reject"
        assert decision.reprimand_teacher is True


class TestSelfCriticClass:
    """Test suite for SelfCritic class."""

    def test_evaluate_method(self):
        """Test SelfCritic.evaluate method."""
        from brains.self_critique_v2 import SelfCritic, Fact

        critic = SelfCritic()
        fact = Fact(
            content="Test content",
            source="user",
            domain="factual"
        )

        decision = critic.evaluate(fact)

        assert decision.decision in ["accept", "reject", "verify_first"]

    def test_score_method(self):
        """Test SelfCritic.score method."""
        from brains.self_critique_v2 import SelfCritic, Fact

        critic = SelfCritic()
        fact = Fact(
            content="Test content",
            source="user",
            domain="factual"
        )

        score = critic.score(fact)

        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
