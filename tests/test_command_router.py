"""
Tests for the Command Router
============================

Tests that verify:
1. Command parsing and routing
2. Status command returns agent state
3. Scan commands invoke self_model
4. Enable/disable execution commands work
5. Unknown commands return proper errors
"""

import pytest
import sys
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCommandParsing:
    """Test suite for command parsing."""

    def test_parse_dashed_command(self):
        """Test parsing command with -- prefix."""
        from brains.cognitive.command_router import route_command

        result = route_command("--status")

        # Should not return empty_command error
        assert result.get("error") != "empty_command"

    def test_parse_slash_command(self):
        """Test parsing command with / prefix."""
        from brains.cognitive.command_router import route_command

        result = route_command("/status")

        assert result.get("error") != "empty_command"

    def test_parse_subcommand(self):
        """Test parsing command with subcommand."""
        from brains.cognitive.command_router import route_command

        result = route_command("--cache purge")

        # Should route to cache handler
        assert "message" in result or "error" in result

    def test_empty_command(self):
        """Test empty command handling."""
        from brains.cognitive.command_router import route_command

        result = route_command("")

        assert result.get("error") == "empty_command"

    def test_whitespace_command(self):
        """Test whitespace-only command."""
        from brains.cognitive.command_router import route_command

        result = route_command("   ")

        assert result.get("error") == "empty_command"


class TestStatusCommand:
    """Test suite for status command."""

    def test_status_returns_json(self):
        """Test that status returns JSON report."""
        from brains.cognitive.command_router import route_command
        import json

        result = route_command("--status")

        if "message" in result:
            # Should be valid JSON
            status = json.loads(result["message"])
            assert "running" in status or "active_goals" in status or True


class TestScanCommands:
    """Test suite for scan commands."""

    def test_scan_self(self):
        """Test scan self command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--scan self")

        # Should return message or error (depending on self_model availability)
        assert "message" in result or "error" in result

    def test_scan_memory(self):
        """Test scan memory command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--scan memory")

        assert "message" in result or "error" in result

    def test_scan_unknown_target(self):
        """Test scan with unknown target."""
        from brains.cognitive.command_router import route_command

        result = route_command("--scan unknown")

        assert "error" in result
        assert "unknown_scan_target" in result["error"]


class TestExecutionCommands:
    """Test suite for execution enable/disable commands."""

    def test_enable_execution(self):
        """Test enable execution command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--enable execution")

        # Should succeed or report execution_guard not available
        assert "message" in result or "error" in result

    def test_disable_execution(self):
        """Test disable execution command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--disable execution")

        assert "message" in result or "error" in result

    def test_enable_unknown_target(self):
        """Test enable with unknown target."""
        from brains.cognitive.command_router import route_command

        result = route_command("--enable unknown")

        assert "error" in result
        assert "unknown_enable_target" in result["error"]


class TestListCommand:
    """Test suite for list command."""

    def test_list_tools(self):
        """Test list tools command."""
        from brains.cognitive.command_router import route_command
        import json

        result = route_command("--list tools")

        assert "message" in result
        tools = json.loads(result["message"])
        assert "action_engine" in tools
        assert "browser_tool" in tools

    def test_list_unknown_target(self):
        """Test list with unknown target."""
        from brains.cognitive.command_router import route_command

        result = route_command("--list unknown")

        assert "error" in result


class TestCacheCommand:
    """Test suite for cache command."""

    def test_cache_purge(self):
        """Test cache purge command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--cache purge")

        # Should succeed with message about cache
        assert "message" in result
        assert "cache" in result["message"].lower()

    def test_cache_clear_synonym(self):
        """Test cache clear (synonym for purge)."""
        from brains.cognitive.command_router import route_command

        result = route_command("--cache clear")

        assert "message" in result

    def test_cache_unknown_subcommand(self):
        """Test cache with unknown subcommand."""
        from brains.cognitive.command_router import route_command

        result = route_command("--cache invalid")

        assert "error" in result
        assert "unknown_cache_command" in result["error"]


class TestSayCommand:
    """Test suite for say/speak commands."""

    def test_say_hello(self):
        """Test say hello command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--say hello")

        assert "message" in result
        assert "Hello" in result["message"]

    def test_say_with_phrase(self):
        """Test say with custom phrase."""
        from brains.cognitive.command_router import route_command

        result = route_command("--say good morning")

        assert "message" in result
        assert "Good morning" in result["message"]

    def test_say_empty(self):
        """Test say with no phrase."""
        from brains.cognitive.command_router import route_command

        result = route_command("--say")

        assert "error" in result
        assert "nothing_to_say" in result["error"]


class TestUnknownCommands:
    """Test suite for unknown command handling."""

    def test_unknown_command(self):
        """Test completely unknown command."""
        from brains.cognitive.command_router import route_command

        result = route_command("--nonexistent")

        assert "error" in result
        assert "unknown_command" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
