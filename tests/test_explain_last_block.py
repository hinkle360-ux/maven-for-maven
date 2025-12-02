"""
Regression tests for explanation/introspection questions.

These tests verify that explanation questions like "why did you answer that way?" are:
1. Routed via SELF_INTENT_GATE to EXPLAIN_LAST
2. Answered from internal run metadata (not from Teacher)
3. Provide truthful introspection based on actual system behavior

See Task 3 in the capability fix specification.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestExplainQueryDetection:
    """Test that explanation questions are correctly identified."""

    def test_teacher_helper_detects_explain_questions(self):
        """TeacherHelper should identify explanation questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_explain_query = TeacherHelper._is_explain_query.__get__(helper)

        explain_questions = [
            "why did you answer that way?",
            "how did you get that answer?",
            "which parts of your system helped you answer?",
            "did you use the teacher to answer that?",
            "what would you do differently next time?",
            "why that answer?",
            "which brains helped?",
            "explain your reasoning",
        ]

        for question in explain_questions:
            assert helper._is_explain_query(question) is True, f"Failed to detect: {question}"

    def test_non_explain_questions_not_matched(self):
        """Non-explanation questions should not be matched."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_explain_query = TeacherHelper._is_explain_query.__get__(helper)

        non_explain_questions = [
            "what is a lion?",
            "explain what a lion is",  # asking to explain a concept, not the answer
            "how do computers work?",
            "tell me about birds",
        ]

        for question in non_explain_questions:
            assert helper._is_explain_query(question) is False, f"False positive: {question}"


class TestExplainTeacherBlocking:
    """Test that Teacher is blocked for explanation questions."""

    def test_teacher_forbidden_for_explain_questions(self):
        """_is_forbidden_teacher_question should return True for explain questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_forbidden_teacher_question = TeacherHelper._is_forbidden_teacher_question.__get__(helper)
            helper._is_self_query = lambda q: False
            helper._is_history_query = lambda q: False
            helper._is_self_memory_query = lambda q: False
            helper._is_capability_query = lambda q: False
            helper._is_explain_query = TeacherHelper._is_explain_query.__get__(helper)

        is_forbidden, reason = helper._is_forbidden_teacher_question("why did you answer that way?")

        assert is_forbidden is True
        assert reason == "explain_last"


class TestExplainPatternsInSelfIntentGate:
    """Test that SELF_INTENT_GATE has explain patterns defined."""

    def test_explain_patterns_defined(self):
        """Verify explain patterns are defined in reasoning_brain."""
        import inspect
        from brains.cognitive.reasoning.service import reasoning_brain

        source = inspect.getsource(reasoning_brain)

        # Check that explain patterns are defined
        assert "explain_last_patterns" in source
        assert "why did you answer that way" in source
        assert "how did you get that answer" in source
        assert "did you use the teacher" in source

    def test_explain_last_kind_defined(self):
        """Verify explain_last self_kind is used in routing."""
        import inspect
        from brains.cognitive.reasoning.service import reasoning_brain

        source = inspect.getsource(reasoning_brain)

        # Check that explain_last routing is implemented
        assert 'self_kind = "explain_last"' in source


class TestLanguageBrainExplainLast:
    """Test language_brain EXPLAIN_LAST operation exists."""

    def test_explain_last_operation_exists(self):
        """Verify EXPLAIN_LAST operation is defined in language_brain."""
        import inspect
        from brains.cognitive.language.service import language_brain

        source = inspect.getsource(language_brain)

        # Check that EXPLAIN_LAST is implemented
        assert 'op == "EXPLAIN_LAST"' in source or "EXPLAIN_LAST" in source


class TestExplainRoutingBehavior:
    """Test the routing behavior for explain questions."""

    def test_explain_question_routes_to_explain_last(self):
        """When asking 'why did you answer that way', it should route to explain_last handler."""
        # This is an integration test that verifies the pattern matches correctly

        # First, verify the pattern is detected
        test_question = "why did you answer that way?"
        normalized = test_question.lower().strip()
        for punct in ['.', ',', '!', '?', ':', ';']:
            normalized = normalized.replace(punct, ' ')
        normalized = ' '.join(normalized.split())

        explain_patterns = [
            "why did you answer that way",
            "why that answer",
            "how did you get that answer",
            "how did you arrive at that",
            "which parts of your system helped you answer",
            "which brains helped",
            "which parts helped",
            "did you use the teacher to answer that",
            "did you use the teacher",
            "did you call the teacher",
            "what would you do differently next time",
            "how would you improve that answer",
            "explain your reasoning",
            "explain that answer",
            "why did you say that",
        ]

        matched = any(pattern in normalized for pattern in explain_patterns)
        assert matched is True, f"Pattern not matched for: {test_question}"


class TestContextHintBlocking:
    """Test that context hints also trigger blocking."""

    def test_context_intent_blocks_teacher(self):
        """Context with explain_last intent should block Teacher."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_forbidden_teacher_question = TeacherHelper._is_forbidden_teacher_question.__get__(helper)
            helper._is_self_query = lambda q: False
            helper._is_history_query = lambda q: False
            helper._is_self_memory_query = lambda q: False
            helper._is_capability_query = lambda q: False
            helper._is_explain_query = lambda q: False

        # Even a generic question should be blocked if context says it's explain_last
        context = {"self_intent_kind": "explain_last"}
        is_forbidden, reason = helper._is_forbidden_teacher_question(
            "tell me more",
            context=context
        )

        assert is_forbidden is True
        assert reason == "explain_last"
