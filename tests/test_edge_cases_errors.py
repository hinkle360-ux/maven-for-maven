"""
Edge Cases and Error Handling Tests
=====================================

Comprehensive tests for edge cases and error handling:
- Empty/null inputs
- Invalid parameters
- Malformed queries
- Unicode and special characters
- Very long inputs
- Concurrent access
- Error recovery
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import json


# ============================================================================
# Test Empty/Null Inputs
# ============================================================================

class TestEmptyNullInputs:
    """Tests for empty and null input handling."""

    def test_empty_string_pipeline(self):
        """Test pipeline with empty string."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("", 0.5)
        assert isinstance(result, dict)
        # Should handle gracefully

    def test_whitespace_only_pipeline(self):
        """Test pipeline with whitespace-only input."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("   \t\n  ", 0.5)
        assert isinstance(result, dict)

    def test_empty_command_router(self):
        """Test command router with empty command."""
        from brains.cognitive.command_router import route_command

        result = route_command("")
        assert "error" in result
        assert "empty_command" in result["error"]

    def test_process_empty_input(self):
        """Test process function with empty input."""
        from ui.maven_chat import process

        result = process("")
        assert isinstance(result, str)

    def test_interpret_intent_empty(self):
        """Test intent interpretation with empty input."""
        from ui.maven_chat import _interpret_intent

        intent = _interpret_intent("", {})
        assert intent == "pipeline"  # Default

    def test_web_settings_empty(self):
        """Test web settings extraction with empty topic."""
        from ui.maven_chat import _extract_web_settings

        topic, web_enabled, time_budget, max_req, hint_seen = _extract_web_settings("")
        assert topic == ""


# ============================================================================
# Test Invalid Parameters
# ============================================================================

class TestInvalidParameters:
    """Tests for invalid parameter handling."""

    def test_negative_confidence(self):
        """Test pipeline with negative confidence."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("test query", -0.5)
        assert isinstance(result, dict)
        # Should handle or clip to valid range

    def test_confidence_over_one(self):
        """Test pipeline with confidence > 1."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("test query", 1.5)
        assert isinstance(result, dict)

    def test_action_engine_missing_params(self):
        """Test action engine with missing required parameters."""
        from brains.cognitive.action_engine.service.action_engine import service_api

        result = service_api({
            "op": "READ_FILE",
            "payload": {}  # Missing required 'path'
        })
        assert isinstance(result, dict)
        assert result.get("ok") is False

    def test_command_router_invalid_subcommand(self):
        """Test command router with invalid subcommand."""
        from brains.cognitive.command_router import route_command

        result = route_command("--cache invalid_subcommand")
        assert "error" in result

    def test_self_model_invalid_op(self):
        """Test self model with invalid operation."""
        from brains.cognitive.self_model.service.self_model_brain import service_api

        result = service_api({"op": "INVALID_OP"})
        assert isinstance(result, dict)
        assert result.get("ok") is False


# ============================================================================
# Test Malformed Queries
# ============================================================================

class TestMalformedQueries:
    """Tests for malformed query handling."""

    def test_partial_command(self):
        """Test partial command prefix."""
        from brains.cognitive.command_router import route_command

        result = route_command("-")  # Single dash
        assert isinstance(result, dict)

    def test_unclosed_quotes_in_query(self):
        """Test query with unclosed quotes."""
        from ui.maven_chat import process

        result = process('what is "the meaning of life')
        assert isinstance(result, str)

    def test_json_like_string(self):
        """Test JSON-like string in query."""
        from ui.maven_chat import process

        result = process('{"key": "value"}')
        assert isinstance(result, str)

    def test_html_in_query(self):
        """Test HTML tags in query."""
        from ui.maven_chat import process

        result = process("<script>alert('test')</script>")
        assert isinstance(result, str)


# ============================================================================
# Test Unicode and Special Characters
# ============================================================================

class TestUnicodeSpecialChars:
    """Tests for unicode and special character handling."""

    def test_unicode_query(self):
        """Test query with unicode characters."""
        from ui.maven_chat import process

        result = process("")
        assert isinstance(result, str)

    def test_emoji_query(self):
        """Test query with emojis."""
        from ui.maven_chat import process

        result = process("Hello 😀 How are you?")
        assert isinstance(result, str)

    def test_cjk_characters(self):
        """Test query with CJK characters."""
        from ui.maven_chat import process

        result = process("你好世界")  # Chinese: Hello World
        assert isinstance(result, str)

    def test_arabic_characters(self):
        """Test query with Arabic characters."""
        from ui.maven_chat import process

        result = process("مرحبا بالعالم")  # Arabic: Hello World
        assert isinstance(result, str)

    def test_special_punctuation(self):
        """Test query with special punctuation."""
        from ui.maven_chat import process

        result = process("What's the meaning of «this»?")
        assert isinstance(result, str)

    def test_control_characters(self):
        """Test query with control characters."""
        from ui.maven_chat import process

        result = process("test\x00\x01\x02")
        assert isinstance(result, str)


# ============================================================================
# Test Very Long Inputs
# ============================================================================

class TestLongInputs:
    """Tests for very long input handling."""

    def test_very_long_query(self):
        """Test very long query string."""
        from ui.maven_chat import process

        long_query = "a" * 10000
        result = process(long_query)
        assert isinstance(result, str)

    def test_long_repeated_words(self):
        """Test query with many repeated words."""
        from ui.maven_chat import process

        repeated_query = " ".join(["test"] * 1000)
        result = process(repeated_query)
        assert isinstance(result, str)

    def test_long_command(self):
        """Test very long command."""
        from brains.cognitive.command_router import route_command

        long_cmd = "--say " + "a" * 5000
        result = route_command(long_cmd)
        assert isinstance(result, dict)


# ============================================================================
# Test Error Recovery
# ============================================================================

class TestErrorRecovery:
    """Tests for error recovery mechanisms."""

    def test_pipeline_error_recovery(self):
        """Test pipeline recovers from errors."""
        from brains.pipeline.pipeline_runner import run_pipeline

        # Multiple queries should not affect each other
        results = [
            run_pipeline("normal query 1", 0.9),
            run_pipeline("", 0.5),  # Empty
            run_pipeline("normal query 2", 0.9),
        ]

        for result in results:
            assert isinstance(result, dict)

    def test_process_function_recovery(self):
        """Test process function recovers from errors."""
        from ui.maven_chat import process

        # Series of queries including edge cases
        queries = [
            "hello",
            "",
            "what is 2+2",
            "🎉",
            "normal query",
        ]

        for query in queries:
            result = process(query)
            assert isinstance(result, str)


# ============================================================================
# Test Boundary Conditions
# ============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_confidence_zero(self):
        """Test pipeline with zero confidence."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("test", 0.0)
        assert isinstance(result, dict)

    def test_confidence_one(self):
        """Test pipeline with confidence exactly 1."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("test", 1.0)
        assert isinstance(result, dict)

    def test_single_character_query(self):
        """Test single character query."""
        from ui.maven_chat import process

        result = process("?")
        assert isinstance(result, str)

    def test_numeric_only_query(self):
        """Test numeric-only query."""
        from ui.maven_chat import process

        result = process("12345")
        assert isinstance(result, str)


# ============================================================================
# Test Memory Operations
# ============================================================================

class TestMemoryOperations:
    """Tests for memory-related operations."""

    def test_brain_memory_empty_query(self):
        """Test brain memory with empty query."""
        from brains.memory.brain_memory import BrainMemory

        try:
            memory = BrainMemory("test_bank")
            # Should handle empty query
            results = memory.query("")
            assert isinstance(results, list)
        except Exception as e:
            # Some exceptions are acceptable for invalid input
            assert isinstance(e, (ValueError, TypeError, KeyError, AttributeError))

    def test_tier_manager_invalid_brain(self):
        """Test tier manager with non-existent brain."""
        from brains.memory.tier_manager import TierManager
        from pathlib import Path

        # Non-existent brain path
        nonexistent_path = Path("/nonexistent/path")

        try:
            tm = TierManager("nonexistent", nonexistent_path)
            # May either work with empty data or raise
            counts = tm.get_tier_counts()
            assert isinstance(counts, dict)
        except Exception as e:
            # Some exceptions are acceptable
            assert isinstance(e, (FileNotFoundError, ValueError, OSError))


# ============================================================================
# Test Command Router Edge Cases
# ============================================================================

class TestCommandRouterEdgeCases:
    """Edge case tests for command router."""

    def test_multiple_dashes(self):
        """Test command with multiple leading dashes."""
        from brains.cognitive.command_router import route_command

        result = route_command("---status")
        assert isinstance(result, dict)

    def test_mixed_prefix(self):
        """Test command with mixed prefix."""
        from brains.cognitive.command_router import route_command

        result = route_command("/-status")
        assert isinstance(result, dict)

    def test_command_with_newlines(self):
        """Test command with embedded newlines."""
        from brains.cognitive.command_router import route_command

        result = route_command("--say hello\nworld")
        assert isinstance(result, dict)

    def test_say_with_numbers(self):
        """Test say command with numbers."""
        from brains.cognitive.command_router import route_command

        result = route_command("--say 12345")
        assert isinstance(result, dict)
        assert "message" in result


# ============================================================================
# Test Sanitization
# ============================================================================

class TestSanitization:
    """Tests for input sanitization."""

    def test_sanitize_preserves_normal_text(self):
        """Test that normal text is preserved."""
        from ui.maven_chat import _sanitize_for_log

        text = "Hello, this is a normal query."
        assert _sanitize_for_log(text) == text

    def test_sanitize_masks_email(self):
        """Test that email addresses are masked."""
        from ui.maven_chat import _sanitize_for_log

        text = "Contact me at test@example.com"
        sanitized = _sanitize_for_log(text)
        assert "test@example.com" not in sanitized
        assert "<EMAIL>" in sanitized

    def test_sanitize_masks_long_tokens(self):
        """Test that long tokens are masked."""
        from ui.maven_chat import _sanitize_for_log

        text = "Secret key: abcdefghijklmnop12345678"
        sanitized = _sanitize_for_log(text)
        assert "abcdefghijklmnop12345678" not in sanitized
        assert "<TOKEN>" in sanitized

    def test_sanitize_empty_string(self):
        """Test sanitization of empty string."""
        from ui.maven_chat import _sanitize_for_log

        assert _sanitize_for_log("") == ""


# ============================================================================
# Test Multi-Question Edge Cases
# ============================================================================

class TestMultiQuestionEdgeCases:
    """Tests for multi-question handling edge cases."""

    def test_single_question_marker(self):
        """Test text with single question marker."""
        import re

        text = "what is a dog"
        markers = ["what is", "what are"]
        count = sum(text.lower().count(m) for m in markers)
        assert count == 1

    def test_many_question_markers(self):
        """Test text with many question markers."""
        import re

        text = "what is a dog what are cats what is weather what are birds"
        markers = ["what is", "what are"]
        count = sum(text.lower().count(m) for m in markers)
        assert count == 4

    def test_math_only_query(self):
        """Test query with only math expression."""
        import re

        text = "32+44"
        math_pattern = r"\d+\s*[\+\-\*/]\s*\d+"
        has_math = bool(re.search(math_pattern, text))
        assert has_math is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
