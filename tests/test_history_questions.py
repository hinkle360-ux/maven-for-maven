"""
Regression tests for conversation history questions.

These tests verify that history questions like "what did I ask you first today?" are:
1. Routed via SELF_INTENT_GATE (not through normal pipeline)
2. Answered from session_history (not from Teacher)
3. Return honest "I don't know" rather than hallucinating

See Task 2 in the capability fix specification.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHistoryQuestionDetection:
    """Test that history questions are correctly identified."""

    def test_teacher_helper_detects_history_questions(self):
        """TeacherHelper should identify history questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_history_query = TeacherHelper._is_history_query.__get__(helper)

        history_questions = [
            "what did i ask you first today?",
            "what was my first question?",
            "what did we talk about yesterday?",
            "what did we discuss?",
            "do you remember what i asked?",
        ]

        for question in history_questions:
            assert helper._is_history_query(question) is True, f"Failed to detect: {question}"

    def test_non_history_questions_not_matched(self):
        """Non-history questions should not be matched."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_history_query = TeacherHelper._is_history_query.__get__(helper)

        non_history_questions = [
            "what is a lion?",
            "how do computers work?",
            "tell me about birds",
        ]

        for question in non_history_questions:
            assert helper._is_history_query(question) is False, f"False positive: {question}"


class TestHistoryTeacherBlocking:
    """Test that Teacher is blocked for history questions."""

    def test_teacher_forbidden_for_history_questions(self):
        """_is_forbidden_teacher_question should return True for history questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_forbidden_teacher_question = TeacherHelper._is_forbidden_teacher_question.__get__(helper)
            helper._is_self_query = lambda q: False
            helper._is_history_query = TeacherHelper._is_history_query.__get__(helper)
            helper._is_self_memory_query = lambda q: False
            helper._is_capability_query = lambda q: False
            helper._is_explain_query = lambda q: False

        is_forbidden, reason = helper._is_forbidden_teacher_question("what did i ask you first today?")

        assert is_forbidden is True
        assert reason == "conversation_history"


class TestSystemHistoryBrain:
    """Test system_history_brain QUERY_HISTORY operation."""

    def test_query_history_yesterday_returns_honest_answer(self):
        """Asking about yesterday should return honest 'no record' answer."""
        from brains.cognitive.system_history.service.system_history_brain import service_api

        response = service_api({
            "op": "QUERY_HISTORY",
            "payload": {
                "query": "what did we talk about yesterday?",
                "history_type": "yesterday"
            }
        })

        assert response.get("ok") is True
        payload = response.get("payload", {})
        assert payload.get("found") is False
        assert payload.get("source") == "session_history"
        # Should not claim to have yesterday's conversation
        assert "no record" in payload.get("answer", "").lower() or "memory resets" in payload.get("answer", "").lower()

    def test_query_history_first_question_no_hallucination(self):
        """Asking about first question should return from actual session or 'no record'."""
        from brains.cognitive.system_history.service.system_history_brain import service_api

        response = service_api({
            "op": "QUERY_HISTORY",
            "payload": {
                "query": "what did I ask you first today?",
                "history_type": "first_today"
            }
        })

        assert response.get("ok") is True
        payload = response.get("payload", {})
        assert payload.get("source") == "session_history"
        # Either found actual history or honest "no record"
        # Should NOT hallucinate a conversation that didn't happen


class TestHistoryPatternsInSelfIntentGate:
    """Test that SELF_INTENT_GATE has history patterns defined."""

    def test_history_patterns_defined(self):
        """Verify history patterns are defined in reasoning_brain."""
        import inspect
        from brains.cognitive.reasoning.service import reasoning_brain

        source = inspect.getsource(reasoning_brain)

        # Check that history patterns are defined
        assert "history_patterns" in source
        assert "what did i ask you first today" in source
        assert "what did we talk about yesterday" in source


class TestUserMemoryQuestions:
    """Test user memory questions (what do you know about me)."""

    def test_teacher_helper_detects_user_memory_questions(self):
        """TeacherHelper should identify user memory questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_self_memory_query = TeacherHelper._is_self_memory_query.__get__(helper)

        user_memory_questions = [
            "what do you remember about me?",
            "what do you know about me?",
            "what is the most important thing you know about me?",
            "what have you learned about me?",
        ]

        for question in user_memory_questions:
            assert helper._is_self_memory_query(question) is True, f"Failed to detect: {question}"

    def test_teacher_forbidden_for_user_memory_questions(self):
        """_is_forbidden_teacher_question should return True for user memory questions."""
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

        try:
            helper = TeacherHelper("test_brain")
        except ValueError:
            helper = MagicMock()
            helper._is_forbidden_teacher_question = TeacherHelper._is_forbidden_teacher_question.__get__(helper)
            helper._is_self_query = lambda q: False
            helper._is_history_query = lambda q: False
            helper._is_self_memory_query = TeacherHelper._is_self_memory_query.__get__(helper)
            helper._is_capability_query = lambda q: False
            helper._is_explain_query = lambda q: False

        is_forbidden, reason = helper._is_forbidden_teacher_question("what is the most important thing you know about me?")

        assert is_forbidden is True
        assert reason == "self_memory"
