"""
Regression tests for capability questions routing.

These tests verify that capability questions like "can you search the web?" are:
1. Routed via SELF_INTENT_GATE (not through normal pipeline)
2. Answered from capability_snapshot (not from Teacher)
3. Provide truthful answers based on real config

See Task 1 in the capability fix specification.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestCapabilityAnswering:
    """Test that answer_capability_question returns truthful answers from config."""

    def test_web_search_disabled_answer(self, monkeypatch):
        """When web research is disabled, answer should say no web search."""
        # Ensure web research is disabled
        monkeypatch.setenv("MAVEN_NETWORK_DISABLED", "1")

        from capabilities import answer_capability_question

        result = answer_capability_question("can you search the web for me?")

        assert result is not None
        assert result["capability"] == "web_search"
        assert result["enabled"] is False
        assert result["source"] == "capability_snapshot"
        assert "No" in result["answer"] or "cannot" in result["answer"].lower()

    def test_code_execution_disabled_answer(self, monkeypatch):
        """When execution is disabled, answer should say no code execution."""
        # Force execution disabled
        monkeypatch.setenv("MAVEN_EXECUTION_ENABLED", "0")

        from capabilities import answer_capability_question

        result = answer_capability_question("can you run code for me?")

        assert result is not None
        assert result["capability"] == "code_execution"
        assert result["enabled"] is False
        assert result["source"] == "capability_snapshot"
        assert "No" in result["answer"] or "disabled" in result["answer"].lower()

    def test_control_programs_always_no(self):
        """Control other programs is always NO regardless of config."""
        from capabilities import answer_capability_question

        result = answer_capability_question("can you control other programs on my computer?")

        assert result is not None
        assert result["capability"] == "control_programs"
        assert result["enabled"] is False
        assert result["source"] == "capability_snapshot"
        assert "No" in result["answer"] or "cannot" in result["answer"].lower()

    def test_file_access_answer(self, monkeypatch):
        """File access answer should reflect actual config."""
        # Force execution disabled
        monkeypatch.setenv("MAVEN_EXECUTION_ENABLED", "0")

        from capabilities import answer_capability_question

        result = answer_capability_question("can you read or change files on my system?")

        assert result is not None
        assert result["capability"] == "filesystem"
        assert result["source"] == "capability_snapshot"
        # Answer should mention restrictions
        assert "configured" in result["answer"].lower() or "directory" in result["answer"].lower()

    def test_autonomous_tools_always_no(self):
        """Autonomous tool usage is always NO."""
        from capabilities import answer_capability_question

        result = answer_capability_question("can you use tools or the internet without me asking you?")

        assert result is not None
        assert result["capability"] == "autonomous_tools"
        assert result["enabled"] is False
        assert result["source"] == "capability_snapshot"
        assert "No" in result["answer"] or "never" in result["answer"].lower()


class TestCapabilitySnapshot:
    """Test that get_capability_snapshot returns accurate config-based data."""

    def test_snapshot_has_all_fields(self):
        """Snapshot should have all required capability fields."""
        from capabilities import get_capability_snapshot

        snapshot = get_capability_snapshot()

        assert "web_search_enabled" in snapshot
        assert "code_execution_enabled" in snapshot
        assert "filesystem_scope" in snapshot
        assert "can_control_programs" in snapshot
        assert "autonomous_tools" in snapshot
        assert "execution_mode" in snapshot

    def test_control_programs_always_false(self):
        """can_control_programs should always be False."""
        from capabilities import get_capability_snapshot

        snapshot = get_capability_snapshot()

        assert snapshot["can_control_programs"] is False

    def test_autonomous_tools_always_false(self):
        """autonomous_tools should always be False."""
        from capabilities import get_capability_snapshot

        snapshot = get_capability_snapshot()

        assert snapshot["autonomous_tools"] is False


class TestTeacherBlockingForCapabilities:
    """Test that Teacher is blocked for capability questions."""

    def test_teacher_helper_blocks_capability_questions(self):
        """TeacherHelper should identify and block capability questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        # TeacherHelper may fail to init without contract - mock it
        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            # Create a mock if no contract exists
            helper = MagicMock()
            helper._is_capability_query = TeacherHelper._is_capability_query.__get__(helper)

        # Test various capability questions
        capability_questions = [
            "can you search the web for me?",
            "can you run code for me?",
            "can you control other programs on my computer?",
            "can you read or change files on my system?",
            "can you use tools without me asking?",
        ]

        for question in capability_questions:
            assert helper._is_capability_query(question) is True, f"Failed to detect: {question}"

    def test_teacher_forbidden_for_capability_questions(self):
        """_is_forbidden_teacher_question should return True for capability questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_forbidden_teacher_question = TeacherHelper._is_forbidden_teacher_question.__get__(helper)
            helper._is_self_query = lambda q: False
            helper._is_history_query = lambda q: False
            helper._is_self_memory_query = lambda q: False
            helper._is_capability_query = TeacherHelper._is_capability_query.__get__(helper)
            helper._is_explain_query = lambda q: False

        is_forbidden, reason = helper._is_forbidden_teacher_question("can you search the web?")

        assert is_forbidden is True
        assert reason == "capability"


class TestSelfIntentGateRouting:
    """Test that SELF_INTENT_GATE properly routes capability questions."""

    def test_capability_patterns_defined(self):
        """Verify capability patterns are defined in the expected location."""
        # This test verifies the patterns exist by checking the source file
        import inspect
        from brains.cognitive.reasoning.service import reasoning_brain

        source = inspect.getsource(reasoning_brain)

        # Check that capability patterns are defined
        assert "capability_patterns" in source
        assert "can you search the web" in source
        assert "can you run code" in source
        assert "can you control other programs" in source
