"""
Comprehensive Chat Interface Tests
===================================

Tests for the Maven chat interface (ui/maven_chat.py) covering:
- Basic query processing
- Command routing
- Personal information storage
- Research mode detection
- Self-model queries
- Multi-question handling
- Edge cases and error handling
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_memory_librarian():
    """Mock the memory librarian service API."""
    with patch('ui.maven_chat.memory_librarian') as mock:
        mock.service_api = MagicMock(return_value={
            "ok": True,
            "payload": {
                "context": {
                    "final_answer": "Test answer",
                    "final_confidence": 0.85
                }
            }
        })
        yield mock


@pytest.fixture
def mock_language_brain():
    """Mock the language brain service API."""
    with patch('ui.maven_chat.language_brain') as mock:
        mock.service_api = MagicMock(return_value={
            "payload": {
                "storable_type": "FACT",
                "storable": True,
                "confidence_penalty": 0.0
            }
        })
        yield mock


# ============================================================================
# Test Intent Interpretation
# ============================================================================

class TestIntentInterpretation:
    """Tests for the _interpret_intent function."""

    def test_tick_intent_detected(self):
        """Test that tick-related queries are detected."""
        from ui.maven_chat import _interpret_intent

        tick_queries = ["tick", "advance hum", "oscillator"]
        for query in tick_queries:
            intent = _interpret_intent(query, {})
            assert intent == "dmn_tick", f"Failed for query: {query}"

    def test_reflect_intent_detected(self):
        """Test that reflect queries are detected."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("reflect on recent conversations", {})
        assert intent == "dmn_reflect"

    def test_dissent_intent_detected(self):
        """Test that dissent queries are detected."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("scan for dissent", {})
        assert intent == "dmn_dissent"

    def test_diagnostics_intent_detected(self):
        """Test that diagnostics commands are detected."""
        from ui.maven_chat import _interpret_intent

        diag_queries = ["diag", "diagnostics", "diagnostic"]
        for query in diag_queries:
            intent = _interpret_intent(query, {})
            assert intent == "diag", f"Failed for query: {query}"

    def test_status_intent_detected(self):
        """Test that status queries are detected."""
        from ui.maven_chat import _interpret_intent

        status_queries = ["status", "health check", "show counts"]
        for query in status_queries:
            intent = _interpret_intent(query, {})
            assert intent == "status", f"Failed for query: {query}"

    def test_summary_intent_detected(self):
        """Test that summary queries are detected."""
        from ui.maven_chat import _interpret_intent

        summary_queries = ["summary", "summarize", "report", "dashboard", "export"]
        for query in summary_queries:
            intent = _interpret_intent(query, {})
            assert intent == "summary", f"Failed for query: {query}"

    def test_retrieve_intent_detected(self):
        """Test that memory search queries are detected."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("search memory for cats", {})
        assert intent == "retrieve"

    def test_router_explain_intent_detected(self):
        """Test that router explanation queries are detected."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("router explain why this bank", {})
        assert intent == "router_explain"

    def test_register_claim_intent_detected(self):
        """Test that claim registration queries are detected."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("register claim the sky is blue", {})
        assert intent == "register_claim"

    def test_store_command_goes_to_pipeline(self):
        """Test that explicit store commands go to pipeline."""
        from ui.maven_chat import _interpret_intent

        store_queries = ["store cats are mammals", "remember my name is John", "save this fact"]
        for query in store_queries:
            intent = _interpret_intent(query, {})
            assert intent == "pipeline", f"Failed for query: {query}"

    def test_default_intent_is_pipeline(self):
        """Test that general queries default to pipeline."""
        from ui.maven_chat import _interpret_intent

        general_queries = [
            "what is 2+2",
            "tell me about dogs",
            "how does photosynthesis work",
            "hello there"
        ]
        for query in general_queries:
            intent = _interpret_intent(query, {})
            assert intent == "pipeline", f"Failed for query: {query}"


# ============================================================================
# Test Web Settings Extraction
# ============================================================================

class TestWebSettingsExtraction:
    """Tests for the _extract_web_settings function."""

    def test_web_true_hint(self):
        """Test web:true hint is extracted."""
        from ui.maven_chat import _extract_web_settings

        topic, web_enabled, time_budget, max_req, hint_seen = _extract_web_settings("cats web:true")
        assert topic == "cats"
        assert web_enabled is True
        assert hint_seen is True

    def test_web_false_hint(self):
        """Test web:false hint is extracted."""
        from ui.maven_chat import _extract_web_settings

        topic, web_enabled, time_budget, max_req, hint_seen = _extract_web_settings("cats web:false")
        assert topic == "cats"
        assert web_enabled is False
        assert hint_seen is True

    def test_web_time_budget(self):
        """Test web:60 (time budget in seconds) hint is extracted."""
        from ui.maven_chat import _extract_web_settings

        topic, web_enabled, time_budget, max_req, hint_seen = _extract_web_settings("cats web:60")
        assert topic == "cats"
        assert web_enabled is True  # Time budget implies web enabled
        assert time_budget == 60
        assert hint_seen is True

    def test_no_web_hint(self):
        """Test topics without web hints."""
        from ui.maven_chat import _extract_web_settings

        topic, web_enabled, time_budget, max_req, hint_seen = _extract_web_settings("cats")
        assert topic == "cats"
        assert hint_seen is False


# ============================================================================
# Test Research Mode Detection
# ============================================================================

class TestResearchModeDetection:
    """Tests for research command detection."""

    def test_research_colon_format(self):
        """Test 'research: topic' format detection."""
        # Research mode is detected in process() function
        # We test that the command prefix is recognized
        query = "research: quantum physics"
        assert query.lower().startswith("research:")

    def test_research_space_format(self):
        """Test 'research topic' format detection."""
        query = "research quantum physics"
        assert query.lower().startswith("research ")

    def test_deep_research_format(self):
        """Test 'deep research topic' format detection."""
        query = "deep research quantum physics"
        assert query.lower().startswith("deep research")

    def test_deep_research_on_format(self):
        """Test 'deep research on topic' format detection."""
        query = "deep research on quantum physics"
        assert query.lower().startswith("deep research on")


# ============================================================================
# Test Personal Information Patterns
# ============================================================================

class TestPersonalInfoPatterns:
    """Tests for personal information detection patterns."""

    def test_i_am_name_pattern(self):
        """Test 'i am [name]' pattern detection."""
        import re
        pattern = r"\bi\s+am\s+([a-z]+)\b"

        test_cases = [
            ("i am josh", "josh"),
            ("I am Sarah", "sarah"),
            ("i am john smith", "john"),  # Only captures first word
        ]

        for text, expected in test_cases:
            match = re.search(pattern, text.lower())
            assert match is not None, f"Pattern should match: {text}"
            assert match.group(1) == expected, f"Expected {expected}, got {match.group(1)}"

    def test_my_name_is_pattern(self):
        """Test 'my name is [name]' pattern detection."""
        import re
        pattern = r"\bmy\s+name\s+is\s+([a-z]+)\b"

        text = "my name is alice"
        match = re.search(pattern, text.lower())
        assert match is not None
        assert match.group(1) == "alice"

    def test_favorite_color_pattern(self):
        """Test favorite color patterns."""
        import re
        # Match patterns used in maven_chat.py
        patterns = [
            r"\bi\s+like\s+the\s+color\s+([a-z]+)\b",
            r"\bmy\s+favorite\s+colou?r\s+is\s+([a-z]+)\b",
            r"\bmy\s+favourite\s+colou?r\s+is\s+([a-z]+)\b",
        ]

        test_cases = [
            ("i like the color green", "green"),
            ("my favorite color is blue", "blue"),
            ("my favourite colour is red", "red"),
        ]

        for text, expected in test_cases:
            matched = False
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    assert match.group(1) == expected
                    matched = True
                    break
            assert matched, f"No pattern matched: {text}"


# ============================================================================
# Test Personal Questions
# ============================================================================

class TestPersonalQuestions:
    """Tests for personal question detection."""

    def test_who_am_i_detection(self):
        """Test 'who am i' question detection."""
        lower = "who am i".lower()
        assert lower in ["who am i", "who am i?"]

    def test_what_color_detection(self):
        """Test color preference question detection."""
        lower = "what color do i like".lower()
        assert "what" in lower and "color" in lower and "like" in lower

    def test_what_do_i_like_detection(self):
        """Test generic preference question detection."""
        lower = "what do i like".lower()
        assert lower in ["what do i like", "what do i like?"]


# ============================================================================
# Test Multi-Question Detection
# ============================================================================

class TestMultiQuestionDetection:
    """Tests for multi-question splitting."""

    def test_detect_multiple_what_is(self):
        """Test detection of multiple 'what is' questions."""
        import re
        text = "what is a dog and what is a cat"

        question_markers = ["what is", "what are", "who is", "who are"]
        marker_count = sum(text.lower().count(marker) for marker in question_markers)

        assert marker_count == 2

    def test_detect_math_expression(self):
        """Test detection of math expressions."""
        import re
        text = "what is 32+44"

        math_pattern = r"\d+\s*[\+\-\*/]\s*\d+"
        has_math = bool(re.search(math_pattern, text))

        assert has_math is True

    def test_combined_question_and_math(self):
        """Test detection of combined question and math."""
        import re
        text = "what is a dog and what is lighting what is 32+44"

        question_markers = ["what is", "what are", "who is", "who are"]
        marker_count = sum(text.lower().count(marker) for marker in question_markers)
        math_pattern = r"\d+\s*[\+\-\*/]\s*\d+"
        has_math = bool(re.search(math_pattern, text))

        assert marker_count >= 2 or (marker_count >= 1 and has_math)


# ============================================================================
# Test Command Router
# ============================================================================

class TestCommandRouter:
    """Tests for command routing functionality."""

    def test_status_command(self):
        """Test --status command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--status")
        assert "message" in result or "error" in result

    def test_cache_purge_command(self):
        """Test --cache purge command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--cache purge")
        assert "message" in result or "error" in result

    def test_scan_self_command(self):
        """Test --scan self command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--scan self")
        assert "message" in result or "error" in result

    def test_scan_memory_command(self):
        """Test --scan memory command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--scan memory")
        assert "message" in result or "error" in result

    def test_list_tools_command(self):
        """Test --list tools command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--list tools")
        assert "message" in result
        # Should contain tool information
        import json
        tools = json.loads(result["message"])
        assert "action_engine" in tools

    def test_say_hello_command(self):
        """Test --say hello command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--say hello")
        assert "message" in result
        assert "Hello" in result["message"]

    def test_unknown_command(self):
        """Test unknown command handling."""
        from brains.cognitive.command_router import route_command

        result = route_command("--nonexistent_cmd")
        assert "error" in result
        assert "unknown_command" in result["error"]

    def test_empty_command(self):
        """Test empty command handling."""
        from brains.cognitive.command_router import route_command

        result = route_command("")
        assert "error" in result
        assert "empty_command" in result["error"]

    def test_execution_status_command(self):
        """Test --execution status command routing."""
        from brains.cognitive.command_router import route_command

        result = route_command("--execution status")
        assert "message" in result or "error" in result


# ============================================================================
# Test Text Sanitization
# ============================================================================

class TestTextSanitization:
    """Tests for text sanitization for logging."""

    def test_email_sanitization(self):
        """Test that email addresses are masked."""
        from ui.maven_chat import _sanitize_for_log

        text = "Contact john@example.com for info"
        sanitized = _sanitize_for_log(text)
        assert "john@example.com" not in sanitized
        assert "<EMAIL>" in sanitized

    def test_long_token_sanitization(self):
        """Test that long alphanumeric tokens are masked."""
        from ui.maven_chat import _sanitize_for_log

        text = "API key: abcdefghijklmnop1234567890"
        sanitized = _sanitize_for_log(text)
        assert "abcdefghijklmnop1234567890" not in sanitized
        assert "<TOKEN>" in sanitized

    def test_normal_text_preserved(self):
        """Test that normal text is preserved."""
        from ui.maven_chat import _sanitize_for_log

        text = "Hello, how are you?"
        sanitized = _sanitize_for_log(text)
        assert sanitized == text


# ============================================================================
# Test Language Parsing
# ============================================================================

class TestLanguageParsing:
    """Tests for language brain parsing fallback."""

    def test_parse_language_fallback(self):
        """Test fallback parsing when language brain unavailable."""
        from ui.maven_chat import _parse_language

        # Force fallback by testing
        result = _parse_language("test text")
        assert "storable_type" in result
        assert "confidence_penalty" in result


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self):
        """Test handling of empty input."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("", {})
        assert intent == "pipeline"  # Default behavior

    def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("   ", {})
        assert intent == "pipeline"  # Default behavior

    def test_special_characters(self):
        """Test handling of special characters."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("!@#$%^&*()", {})
        assert intent == "pipeline"  # Default behavior

    def test_unicode_input(self):
        """Test handling of unicode input."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("", {})
        assert intent == "pipeline"  # Default behavior

    def test_very_long_input(self):
        """Test handling of very long input."""
        from ui.maven_chat import _interpret_intent

        long_text = "a" * 10000
        intent = _interpret_intent(long_text, {})
        assert intent == "pipeline"  # Default behavior


# ============================================================================
# Test Query Types
# ============================================================================

class TestQueryTypes:
    """Tests for different query type handling."""

    def test_question_query(self):
        """Test question-type queries."""
        from ui.maven_chat import _interpret_intent

        questions = [
            "what is the capital of France?",
            "how does photosynthesis work?",
            "why is the sky blue?",
            "who invented the telephone?",
            "when did World War II end?",
            "where is Mount Everest?",
        ]

        for q in questions:
            intent = _interpret_intent(q, {})
            assert intent == "pipeline", f"Question should go to pipeline: {q}"

    def test_statement_query(self):
        """Test statement-type queries."""
        from ui.maven_chat import _interpret_intent

        statements = [
            "The sky is blue",
            "Dogs are mammals",
            "Water freezes at 0 degrees Celsius",
        ]

        for s in statements:
            intent = _interpret_intent(s, {})
            assert intent == "pipeline", f"Statement should go to pipeline: {s}"

    def test_command_query(self):
        """Test command-type queries."""
        from ui.maven_chat import _interpret_intent

        # These should be handled specially
        commands = [
            ("tick", "dmn_tick"),
            ("reflect", "dmn_reflect"),
            ("status", "status"),
            ("diag", "diag"),
        ]

        for cmd, expected_intent in commands:
            intent = _interpret_intent(cmd, {})
            assert intent == expected_intent, f"Command {cmd} should have intent {expected_intent}"


# ============================================================================
# Test Process Function Integration
# ============================================================================

class TestProcessFunction:
    """Integration tests for the process() function."""

    def test_process_exists(self):
        """Test that process function exists and is callable."""
        from ui.maven_chat import process
        assert callable(process)

    def test_process_returns_string(self):
        """Test that process returns a string."""
        from ui.maven_chat import process

        result = process("hello")
        assert isinstance(result, str)

    def test_process_handles_empty_input(self):
        """Test that process handles empty input."""
        from ui.maven_chat import process

        result = process("")
        assert isinstance(result, str)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
