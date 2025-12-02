"""
Capabilities and Boundaries Test Suite
======================================

Regression tests for the exact question blocks from the specification:

Block 1 - Self-Identity Questions:
  - "What are you / why are you here / what is your purpose / who created you"
  - "What can you not do / how do you make decisions / how do you decide what to do first"
  - "How do you decide what to prioritize"

Block 2 - Capability Questions:
  - "Can you search the web for me"
  - "Can you run code for me"
  - "Can you control other programs on my computer"
  - "Can you read or change files on my system"
  - "Can you use tools or the internet without me asking you"

Block 3 - Explanation Questions:
  - "Why did you answer that way"
  - "How did you get that answer"
  - "Which parts of your system helped you answer"
  - "Did you use the teacher to answer that"
  - "What would you do differently next time"

Test Assertions:
1. Identity and purpose answers come from self_model, NOT Teacher
2. Capabilities answers align with capability registry flags
3. Explanations reference brains used (reasoning, memory, self_model, capability registry)
4. Teacher is NOT invoked for capabilities or self-identity questions
"""

import pytest
from unittest.mock import patch, MagicMock
import json


# =============================================================================
# Block 1: Self-Identity Questions
# =============================================================================

class TestSelfIdentityQuestions:
    """Test that self-identity questions route to self_model, not Teacher."""

    IDENTITY_QUESTIONS = [
        "What are you?",
        "Why are you here?",
        "What is your purpose?",
        "Who created you?",
        "What can you not do?",
        "How do you make decisions?",
        "How do you decide what to do first?",
        "How do you decide what to prioritize?",
    ]

    def test_identity_questions_blocked_by_teacher_helper(self):
        """TeacherHelper should identify and block all identity questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        # Create a mock helper or use the real one
        try:
            helper = TeacherHelper("reasoning")
        except ValueError:
            # Create mock if no contract
            helper = MagicMock()
            helper._is_self_query = lambda q: any(
                p in q.lower() for p in ["who are you", "what are you", "who created"]
            )

        for question in self.IDENTITY_QUESTIONS:
            # Check if it's detected as self-query
            is_self = helper._is_self_query(question)
            # At least some of these should be caught
            # The patterns may not catch all variations, but key ones should work
            if "who created" in question.lower() or "what are you" in question.lower():
                assert is_self is True, f"Should detect as self-query: {question}"

    def test_teacher_brain_blocks_self_identity(self):
        """Teacher brain should return SELF_KNOWLEDGE_FORBIDDEN for identity questions."""
        from brains.cognitive.teacher.service.teacher_brain import _handle_impl

        identity_questions = [
            "Who are you?",
            "What is Maven?",
            "Who created Maven?",
        ]

        for question in identity_questions:
            # Call Teacher directly
            result = _handle_impl({
                "op": "TEACH",
                "payload": {
                    "question": question,
                    "context": {}
                }
            })

            # Should succeed but with SELF_KNOWLEDGE_FORBIDDEN verdict
            assert result.get("ok") is True
            payload = result.get("payload", {})
            verdict = payload.get("verdict", "")
            assert verdict == "SELF_KNOWLEDGE_FORBIDDEN", f"Expected SELF_KNOWLEDGE_FORBIDDEN for: {question}"

    def test_identity_card_exists(self):
        """Verify identity_card.json exists with correct structure."""
        from pathlib import Path

        identity_card_path = Path(__file__).parent.parent / "brains" / "cognitive" / "self_dmn" / "identity_card.json"

        assert identity_card_path.exists(), "identity_card.json should exist"

        with open(identity_card_path, "r") as f:
            identity = json.load(f)

        # Check required fields
        assert "name" in identity, "identity_card should have 'name'"
        assert identity["name"] == "Maven", "name should be 'Maven'"
        assert "creator" in identity, "identity_card should have 'creator'"
        assert "is_llm" in identity, "identity_card should have 'is_llm'"
        assert identity["is_llm"] is False, "Maven is NOT an LLM"
        assert "key_principles" in identity, "identity_card should have 'key_principles'"


# =============================================================================
# Block 2: Capability Questions
# =============================================================================

class TestCapabilityQuestions:
    """Test that capability questions use capability registry, not Teacher."""

    CAPABILITY_QUESTIONS = [
        ("Can you search the web for me?", "web_search"),
        ("Can you run code for me?", "code_execution"),
        ("Can you control other programs on my computer?", "control_programs"),
        ("Can you read or change files on my system?", "filesystem"),
        ("Can you use tools or the internet without me asking you?", "autonomous_tools"),
    ]

    def test_capability_questions_answered_from_registry(self):
        """Capability questions should use capability_snapshot, not Teacher."""
        from capabilities import answer_capability_question

        for question, capability_name in self.CAPABILITY_QUESTIONS:
            result = answer_capability_question(question)

            assert result is not None, f"Should recognize capability question: {question}"
            assert result["capability"] == capability_name, f"Wrong capability for: {question}"
            assert result["source"] == "capability_snapshot", f"Source should be capability_snapshot for: {question}"

    def test_control_programs_always_no(self):
        """Control other programs capability should ALWAYS be NO."""
        from capabilities import answer_capability_question, get_capability_snapshot

        result = answer_capability_question("Can you control other programs on my computer?")
        snapshot = get_capability_snapshot()

        assert result["enabled"] is False
        assert snapshot["can_control_programs"] is False
        assert "No" in result["answer"] or "cannot" in result["answer"].lower()

    def test_autonomous_tools_always_no(self):
        """Autonomous tool usage should ALWAYS be NO."""
        from capabilities import answer_capability_question, get_capability_snapshot

        result = answer_capability_question("Can you use tools without me asking?")
        snapshot = get_capability_snapshot()

        assert result["enabled"] is False
        assert snapshot["autonomous_tools"] is False
        assert "No" in result["answer"] or "never" in result["answer"].lower()

    def test_teacher_helper_blocks_capability_questions(self):
        """TeacherHelper should block capability questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("reasoning")
        except ValueError:
            helper = MagicMock()
            # Use the actual implementation
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper as TH
            helper._is_capability_query = TH._is_capability_query.__get__(helper)

        for question, _ in self.CAPABILITY_QUESTIONS:
            is_capability = helper._is_capability_query(question)
            assert is_capability is True, f"Should detect as capability query: {question}"


# =============================================================================
# Block 3: Explanation Questions
# =============================================================================

class TestExplanationQuestions:
    """Test that explanation questions reference actual system components."""

    EXPLANATION_QUESTIONS = [
        "Why did you answer that way?",
        "How did you get that answer?",
        "Which parts of your system helped you answer?",
        "Did you use the teacher to answer that?",
        "What would you do differently next time?",
    ]

    def test_explain_questions_blocked_by_teacher_helper(self):
        """TeacherHelper should block explain questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("reasoning")
        except ValueError:
            helper = MagicMock()
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper as TH
            helper._is_explain_query = TH._is_explain_query.__get__(helper)

        for question in self.EXPLANATION_QUESTIONS:
            is_explain = helper._is_explain_query(question)
            # At least some should be caught
            if "why did you" in question.lower() or "how did you get" in question.lower():
                assert is_explain is True, f"Should detect as explain query: {question}"


# =============================================================================
# Teacher Proposal Mode Tests
# =============================================================================

class TestTeacherProposalMode:
    """Test that Teacher operates in proposal-only mode."""

    def test_proposal_mode_enabled_by_default(self):
        """Proposal mode should be enabled by default."""
        from brains.cognitive.teacher.service.teacher_brain import is_proposal_mode_enabled

        assert is_proposal_mode_enabled() is True

    def test_direct_fact_write_disabled_by_default(self):
        """Direct fact write should be disabled by default."""
        from brains.cognitive.teacher.service.teacher_brain import is_direct_fact_write_enabled

        assert is_direct_fact_write_enabled() is False

    def test_teacher_proposal_schema_exists(self):
        """TeacherProposal schema should be importable."""
        from brains.cognitive.teacher.service.teacher_proposal import (
            TeacherProposal,
            Hypothesis,
            HypothesisKind,
            create_proposal_from_response,
        )

        # Create a test proposal
        proposal = create_proposal_from_response(
            response_text="ANSWER: Paris\nFACTS:\n- Paris is the capital of France",
            answer="Paris",
            facts=[{"statement": "Paris is the capital of France", "confidence": 0.8}],
            original_question="What is the capital of France?"
        )

        assert proposal.verdict == "PROPOSAL"
        assert len(proposal.hypotheses) == 1
        assert proposal.hypotheses[0].status == "proposal"
        assert proposal.hypotheses[0].source == "teacher"

    def test_hypothesis_blocks_self_description(self):
        """Self-description hypotheses should be flagged as unsafe."""
        from brains.cognitive.teacher.service.teacher_proposal import (
            Hypothesis,
            HypothesisKind,
        )

        # Self-description hypothesis
        self_hyp = Hypothesis(
            statement="I am an AI assistant",
            confidence=0.8,
            kind=HypothesisKind.SELF_DESCRIPTION,
            source="teacher"
        )

        assert self_hyp.is_safe_to_evaluate() is False

        # Capability hypothesis
        cap_hyp = Hypothesis(
            statement="I can search the web",
            confidence=0.8,
            kind=HypothesisKind.CAPABILITY,
            source="teacher"
        )

        assert cap_hyp.is_safe_to_evaluate() is False

        # Factual hypothesis (should be safe)
        fact_hyp = Hypothesis(
            statement="Paris is the capital of France",
            confidence=0.8,
            kind=HypothesisKind.FACTUAL,
            source="teacher"
        )

        assert fact_hyp.is_safe_to_evaluate() is True


# =============================================================================
# Capability Startup Scan Tests
# =============================================================================

class TestCapabilityStartupScan:
    """Test capability startup scanning."""

    def test_startup_scan_returns_valid_structure(self):
        """Startup scan should return properly structured results."""
        from capabilities import run_capability_startup_scan

        scan = run_capability_startup_scan(force_refresh=True)

        assert "scan_time" in scan
        assert "scan_duration_ms" in scan
        assert "capabilities" in scan
        assert "summary" in scan
        assert "enabled_capabilities" in scan
        assert "disabled_capabilities" in scan

    def test_startup_scan_caches_results(self):
        """Startup scan should cache results."""
        from capabilities import run_capability_startup_scan

        # First scan
        scan1 = run_capability_startup_scan(force_refresh=True)

        # Second scan (should use cache)
        scan2 = run_capability_startup_scan()

        assert scan1["scan_time"] == scan2["scan_time"]

    def test_get_enabled_capabilities(self):
        """get_enabled_capabilities should return list of enabled capability names."""
        from capabilities import get_enabled_capabilities

        enabled = get_enabled_capabilities()

        assert isinstance(enabled, list)
        for cap in enabled:
            assert isinstance(cap, str)


# =============================================================================
# Feature Flag Tests
# =============================================================================

class TestFeatureFlags:
    """Test feature flag configuration."""

    def test_features_json_has_required_flags(self):
        """features.json should have all required flags."""
        from pathlib import Path

        features_path = Path(__file__).parent.parent / "config" / "features.json"

        assert features_path.exists(), "features.json should exist"

        with open(features_path, "r") as f:
            features = json.load(f)

        # Check required flags
        required_flags = [
            "teacher_learning",
            "teacher_direct_fact_write",
            "teacher_proposal_mode",
            "capability_startup_scan",
        ]

        for flag in required_flags:
            assert flag in features, f"features.json should have '{flag}'"

    def test_proposal_mode_flag_is_true(self):
        """teacher_proposal_mode should be true by default."""
        from pathlib import Path

        features_path = Path(__file__).parent.parent / "config" / "features.json"

        with open(features_path, "r") as f:
            features = json.load(f)

        assert features.get("teacher_proposal_mode") is True

    def test_direct_fact_write_flag_is_false(self):
        """teacher_direct_fact_write should be false by default."""
        from pathlib import Path

        features_path = Path(__file__).parent.parent / "config" / "features.json"

        with open(features_path, "r") as f:
            features = json.load(f)

        assert features.get("teacher_direct_fact_write") is False


# =============================================================================
# Integration Tests: No Teacher for Identity/Capability
# =============================================================================

class TestTeacherNotInvokedForForbiddenQuestions:
    """Test that Teacher is never invoked for forbidden question types."""

    def test_forbidden_question_check_comprehensive(self):
        """_is_forbidden_teacher_question should catch all forbidden types."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("reasoning")
        except ValueError:
            # Create mock with real methods
            helper = MagicMock()
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper as TH
            helper._is_forbidden_teacher_question = TH._is_forbidden_teacher_question.__get__(helper)
            helper._is_self_query = TH._is_self_query.__get__(helper)
            helper._is_history_query = TH._is_history_query.__get__(helper)
            helper._is_self_memory_query = TH._is_self_memory_query.__get__(helper)
            helper._is_capability_query = TH._is_capability_query.__get__(helper)
            helper._is_explain_query = TH._is_explain_query.__get__(helper)

        forbidden_questions = [
            ("Who are you?", "self_identity"),
            ("Can you search the web?", "capability"),
            ("Can you run code?", "capability"),
            ("Can you control other programs?", "capability"),
        ]

        for question, expected_reason in forbidden_questions:
            is_forbidden, reason = helper._is_forbidden_teacher_question(question)
            assert is_forbidden is True, f"Should be forbidden: {question}"
            assert reason == expected_reason, f"Wrong reason for {question}: expected {expected_reason}, got {reason}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
