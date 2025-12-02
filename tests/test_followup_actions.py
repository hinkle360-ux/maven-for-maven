"""
test_followup_actions.py
~~~~~~~~~~~~~~~~~~~~~~~~

PHASE 2 Tests: Follow-up action execution.

Tests that "do it please" type confirmations properly execute the
previous action request like "can you write me a story about birds".

Test Scenarios:
1. "can you write me a story about birds" -> "do it please" -> story is generated
2. Context manager stores and retrieves action requests
3. Planner recognizes confirmation messages and retrieves pending actions
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestConfirmationDetection:
    """Test detection of confirmation messages like 'do it please'."""

    def test_is_confirmation_message_basic(self):
        """Test basic confirmation message detection."""
        from brains.cognitive.context_management.service.context_manager import (
            is_confirmation_message
        )

        # These should be detected as confirmations
        confirmations = [
            "do it",
            "do it please",
            "yes please",
            "go ahead",
            "ok, do it",
            "please do",
            "proceed",
            "yes, do it",
            "do that",
        ]

        for msg in confirmations:
            assert is_confirmation_message(msg), f"Expected '{msg}' to be a confirmation"

    def test_is_confirmation_message_not_confirmation(self):
        """Test that non-confirmations are not detected."""
        from brains.cognitive.context_management.service.context_manager import (
            is_confirmation_message
        )

        # These should NOT be detected as confirmations
        not_confirmations = [
            "what is the weather?",
            "tell me about birds",
            "how do I do that?",
            "can you help me?",
            "I want to do something",
            "please tell me more about this topic",  # Too long
        ]

        for msg in not_confirmations:
            assert not is_confirmation_message(msg), f"'{msg}' should not be a confirmation"


class TestActionRequestExtraction:
    """Test extraction of actionable requests from user text."""

    def test_extract_write_content_request(self):
        """Test extraction of 'write a story' type requests."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request
        )

        # Write content requests
        write_requests = [
            "can you write me a story about birds",
            "write a poem about love",
            "create a short story about cats",
            "generate a song about summer",
        ]

        for request in write_requests:
            result = extract_action_request(request)
            assert result is not None, f"Should extract action from '{request}'"
            assert result.request_type == "write_content", \
                f"Expected 'write_content' for '{request}', got '{result.request_type}'"
            assert result.original_text == request

    def test_extract_search_request(self):
        """Test extraction of search requests."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request
        )

        result = extract_action_request("can you search for information about Python?")
        assert result is not None
        assert result.request_type == "search"

    def test_no_action_for_questions(self):
        """Regular questions should not be extracted as action requests."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request
        )

        # These should NOT be extracted as action requests
        questions = [
            "what is the weather?",
            "who is the president?",
            "how tall is Mount Everest?",
        ]

        for question in questions:
            result = extract_action_request(question)
            assert result is None, f"'{question}' should not be an action request"


class TestActionRequestStorage:
    """Test storing and retrieving action requests."""

    def test_store_and_retrieve_action_request(self):
        """Test that action requests can be stored and retrieved."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request,
            store_action_request,
            get_last_action_request,
            clear_action_request,
        )

        # Clear any existing action
        clear_action_request()

        # Extract and store an action request
        action = extract_action_request("can you write me a story about birds")
        assert action is not None
        store_action_request(action)

        # Retrieve it
        retrieved = get_last_action_request()
        assert retrieved is not None
        assert retrieved.request_type == "write_content"
        assert "story" in retrieved.original_text.lower()

        # Clean up
        clear_action_request()

    def test_get_follow_up_context(self):
        """Test getting follow-up context after storing an action."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request,
            store_action_request,
            get_follow_up_context,
            clear_action_request,
        )

        # Clear any existing action
        clear_action_request()

        # Store an action
        action = extract_action_request("can you write me a story about birds")
        store_action_request(action)

        # Get follow-up context
        context = get_follow_up_context()
        assert context is not None
        assert context["is_follow_up"] is True
        assert "story" in context["original_request"].lower()
        assert context["request_type"] == "write_content"

        # Clean up
        clear_action_request()

    def test_follow_up_context_empty_after_executed(self):
        """Follow-up context should be None after action is marked executed."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request,
            store_action_request,
            get_follow_up_context,
            mark_action_executed,
            clear_action_request,
        )

        # Clear and store
        clear_action_request()
        action = extract_action_request("can you write me a story about birds")
        store_action_request(action)

        # Mark as executed
        mark_action_executed()

        # Should now return None
        context = get_follow_up_context()
        assert context is None

        # Clean up
        clear_action_request()


class TestPlannerFollowUp:
    """Test planner handling of follow-up messages."""

    def test_planner_handle_followup_with_pending_action(self):
        """Planner should handle follow-up with pending action."""
        from brains.cognitive.context_management.service.context_manager import (
            extract_action_request,
            store_action_request,
            clear_action_request,
        )

        try:
            from brains.cognitive.planner.service.planner_brain import handle
        except ImportError:
            pytest.skip("planner_brain not available")

        # Store a pending action
        clear_action_request()
        action = extract_action_request("can you write me a story about birds")
        store_action_request(action)

        # Send follow-up confirmation to planner
        msg = {
            "op": "HANDLE_FOLLOWUP",
            "payload": {
                "text": "do it please"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})
        assert payload.get("is_followup") is True
        assert "story" in payload.get("original_request", "").lower()
        assert len(payload.get("steps", [])) > 0

        # Clean up
        clear_action_request()

    def test_planner_handle_followup_without_pending_action(self):
        """Planner should handle follow-up without pending action."""
        from brains.cognitive.context_management.service.context_manager import (
            clear_action_request,
        )

        try:
            from brains.cognitive.planner.service.planner_brain import handle
        except ImportError:
            pytest.skip("planner_brain not available")

        # Clear any pending actions
        clear_action_request()

        # Send follow-up confirmation with no pending action
        msg = {
            "op": "HANDLE_FOLLOWUP",
            "payload": {
                "text": "do it please"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})
        assert payload.get("is_followup") is False
        assert payload.get("reason") == "no_pending_action"

    def test_planner_extract_action(self):
        """Planner should extract and store action requests."""
        from brains.cognitive.context_management.service.context_manager import (
            clear_action_request,
        )

        try:
            from brains.cognitive.planner.service.planner_brain import handle
        except ImportError:
            pytest.skip("planner_brain not available")

        # Clear any pending actions
        clear_action_request()

        # Send an actionable request
        msg = {
            "op": "EXTRACT_ACTION",
            "payload": {
                "text": "can you write me a story about birds"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})
        assert payload.get("is_action_request") is True
        assert payload.get("request_type") == "write_content"

        # Clean up
        clear_action_request()


class TestContextManagerServiceAPI:
    """Test context manager service API for follow-up handling."""

    def test_check_confirmation_operation(self):
        """Test CHECK_CONFIRMATION operation."""
        from brains.cognitive.context_management.service.context_manager import handle

        # Test with a confirmation message
        msg = {
            "op": "CHECK_CONFIRMATION",
            "payload": {
                "text": "do it please"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})
        assert payload.get("is_confirmation") is True

    def test_extract_action_request_operation(self):
        """Test EXTRACT_ACTION_REQUEST operation."""
        from brains.cognitive.context_management.service.context_manager import (
            handle,
            clear_action_request,
        )

        # Clear first
        clear_action_request()

        msg = {
            "op": "EXTRACT_ACTION_REQUEST",
            "payload": {
                "text": "can you write me a story about birds"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})
        assert payload.get("is_action_request") is True
        assert payload.get("action_request") is not None

        # Clean up
        clear_action_request()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
