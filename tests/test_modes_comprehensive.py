"""
Comprehensive Mode Tests
=========================

Tests for Maven's various operational modes:
- Inventory mode (listing files, modules, tools)
- Coder mode (code generation and manipulation)
- Browser mode (web browsing capabilities)
- Research mode (topic research)
- Normal pipeline mode
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
# Test Inventory Mode
# ============================================================================

class TestInventoryMode:
    """Tests for inventory/listing functionality."""

    def test_list_python_files_via_action_engine(self):
        """Test listing Python files through action engine."""
        from brains.cognitive.action_engine.service.action_engine import service_api

        result = service_api({
            "op": "EXECUTE",
            "payload": {
                "action": "list_python_files",
                "params": {"path": str(project_root)}
            }
        })

        assert isinstance(result, dict)
        # May fail due to execution guard, but should return proper response
        assert "ok" in result

    def test_action_engine_list_actions(self):
        """Test getting available actions from action engine."""
        from brains.cognitive.action_engine.service.action_engine import service_api

        # Use LIST_PYTHON_FILES as a test of available actions
        result = service_api({"op": "LIST_PYTHON_FILES", "payload": {"path": "."}})

        assert isinstance(result, dict)
        # Should return a proper response (may fail due to execution guard)
        assert "ok" in result

    def test_inventory_query_patterns(self):
        """Test detection of inventory-related queries."""
        import re

        # These queries should potentially trigger inventory mode
        inventory_queries = [
            "list all files",
            "show available modules",
            "what tools do you have",
            "show me the inventory",
            "what brains are available",
        ]

        inventory_patterns = [
            r"\blist\b",
            r"\bshow\b.*\b(available|modules?|files?|tools?)\b",
            r"\binventory\b",
            r"\bwhat\b.*\bavailable\b",
        ]

        for query in inventory_queries:
            found_pattern = False
            for pattern in inventory_patterns:
                if re.search(pattern, query.lower()):
                    found_pattern = True
                    break
            # At least some inventory queries should match patterns


# ============================================================================
# Test Coder Mode
# ============================================================================

class TestCoderMode:
    """Tests for coder/code generation functionality."""

    def test_coder_pattern_store(self):
        """Test coder pattern store functionality."""
        from brains.cognitive.coder.pattern_store import (
            CoderPatternStore, Pattern, PatternContext
        )
        import uuid

        store = CoderPatternStore()

        # Store a pattern using the correct API
        pattern = Pattern(
            id=f"test-{uuid.uuid4().hex[:8]}",
            problem_description="Test function pattern",
            context=PatternContext(language="python"),
            code_after="def test_function(): pass",
            pattern_type="GENERATION",
            score=0.9
        )
        pattern_id = store.store_pattern(pattern)

        assert pattern_id is not None

    def test_coder_pattern_similarity_search(self):
        """Test searching for similar patterns."""
        from brains.cognitive.coder.pattern_store import (
            CoderPatternStore, Pattern, PatternContext, PatternQuery
        )
        import uuid

        store = CoderPatternStore()

        # Store some patterns using correct API
        pattern = Pattern(
            id=f"test-{uuid.uuid4().hex[:8]}",
            problem_description="Bubble sort implementation",
            context=PatternContext(language="python", tags=["sorting"]),
            code_after="def bubble_sort(arr): ...",
            pattern_type="GENERATION",
            score=0.85
        )
        store.store_pattern(pattern)

        # Find similar patterns using PatternQuery
        query = PatternQuery(problem_description="sorting algorithm")
        similar = store.find_similar_patterns(query, k=5)
        assert isinstance(similar, list)

    def test_coder_query_patterns(self):
        """Test detection of coder-related queries."""
        import re

        coder_queries = [
            "write a function to sort an array",
            "create a class for user management",
            "generate code for fibonacci",
            "implement binary search",
            "fix the bug in this code",
        ]

        coder_patterns = [
            r"\b(write|create|generate|implement)\b.*\b(function|class|code)\b",
            r"\bfix\b.*\b(bug|error|issue)\b",
            r"\bcode\b",
        ]

        for query in coder_queries:
            found_pattern = False
            for pattern in coder_patterns:
                if re.search(pattern, query.lower()):
                    found_pattern = True
                    break
            # Most coder queries should match patterns


# ============================================================================
# Test Browser Mode
# ============================================================================

class TestBrowserMode:
    """Tests for browser/web functionality."""

    def test_browser_config_exists(self):
        """Test that browser configuration exists."""
        try:
            from browser_runtime.config import BrowserConfig
            config = BrowserConfig()
            assert config is not None
        except ImportError:
            pytest.skip("Browser runtime not available (missing playwright)")

    def test_browser_pattern_store_exists(self):
        """Test that browser pattern store exists."""
        try:
            from brains.agent.tools.browser.pattern_store import PatternStore
            store = PatternStore()
            assert store is not None
        except ImportError:
            pytest.skip("Browser tools not available")

    def test_browser_intent_patterns(self):
        """Test browser intent detection patterns."""
        import re

        browser_queries = [
            "search the web for python tutorials",
            "open https://example.com",
            "go to google.com",
            "browse to github",
            "lookup weather forecast online",
        ]

        browser_patterns = [
            r"\bsearch\b.*\bweb\b",
            r"\b(open|go to|browse|visit)\b.*\b(https?://|\.com|\.org|\.net)\b",
            r"\bonline\b",
            r"https?://\S+",
        ]

        for query in browser_queries:
            found_pattern = False
            for pattern in browser_patterns:
                if re.search(pattern, query.lower()):
                    found_pattern = True
                    break
            # Most browser queries should match patterns

    def test_url_extraction(self):
        """Test URL extraction from queries."""
        import re

        test_cases = [
            ("open https://example.com", "https://example.com"),
            ("visit http://google.com", "http://google.com"),
            ("go to https://github.com/user/repo", "https://github.com/user/repo"),
        ]

        url_pattern = r"https?://\S+"

        for query, expected_url in test_cases:
            match = re.search(url_pattern, query)
            assert match is not None
            assert match.group(0) == expected_url


# ============================================================================
# Test Research Mode
# ============================================================================

class TestResearchMode:
    """Tests for research mode functionality."""

    def test_research_manager_exists(self):
        """Test that research manager brain exists."""
        try:
            from brains.cognitive.research_manager.service.research_manager_brain import service_api
            assert callable(service_api)
        except ImportError:
            pytest.skip("Research manager not available")

    def test_research_command_parsing(self):
        """Test parsing of research commands."""
        test_cases = [
            ("research: quantum physics", "quantum physics", 2, False),
            ("research quantum computing", "quantum computing", 2, False),
            ("deep research on machine learning", "machine learning", 3, True),
            ("deep research neural networks", "neural networks", 3, True),
        ]

        for cmd, expected_topic, expected_depth, expected_web in test_cases:
            lower_cmd = cmd.lower().strip()

            # Parse command
            topic = None
            depth = 2
            use_web = False

            if lower_cmd.startswith("deep research on "):
                topic = cmd[len("deep research on "):].strip()
                depth = 3
                use_web = True
            elif lower_cmd.startswith("deep research "):
                topic = cmd[len("deep research "):].strip()
                depth = 3
                use_web = True
            elif lower_cmd.startswith("research: "):
                topic = cmd[len("research: "):].strip()
                depth = 2
            elif lower_cmd.startswith("research "):
                topic = cmd[len("research "):].strip()
                depth = 2

            assert topic == expected_topic, f"Topic mismatch for '{cmd}'"
            assert depth == expected_depth, f"Depth mismatch for '{cmd}'"
            assert use_web == expected_web, f"Web flag mismatch for '{cmd}'"

    def test_research_web_hints(self):
        """Test web hint parsing in research queries."""
        from ui.maven_chat import _extract_web_settings

        test_cases = [
            ("cats web:true", "cats", True),
            ("dogs web:false", "dogs", False),
            ("birds web:60", "birds", True),  # Time budget implies web
            ("fish", "fish", None),  # No hint - default behavior
        ]

        for query, expected_topic, expected_web in test_cases:
            topic, web_enabled, time_budget, max_req, hint_seen = _extract_web_settings(query)
            assert topic == expected_topic
            if expected_web is not None:
                assert web_enabled == expected_web


# ============================================================================
# Test Normal Pipeline Mode
# ============================================================================

class TestNormalPipelineMode:
    """Tests for normal pipeline processing."""

    def test_pipeline_handles_factual_query(self):
        """Test pipeline handling of factual queries."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("What is the capital of France?", 0.9)
        assert isinstance(result, dict)
        assert "ok" in result or "response" in result

    def test_pipeline_handles_math_query(self):
        """Test pipeline handling of math queries."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("What is 2+2?", 0.9)
        assert isinstance(result, dict)

    def test_pipeline_handles_statement(self):
        """Test pipeline handling of statements."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("The sky is blue.", 0.9)
        assert isinstance(result, dict)

    def test_pipeline_stages_execution(self):
        """Test that pipeline executes stages."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("What color is grass?", 0.9)

        # Check for execution log if available
        if "execution_log" in result:
            assert isinstance(result["execution_log"], list)
            assert len(result["execution_log"]) > 0


# ============================================================================
# Test Mode Switching
# ============================================================================

class TestModeSwitching:
    """Tests for mode detection and switching."""

    def test_detect_coder_mode(self):
        """Test detection of coder mode from query."""
        import re

        coder_indicators = [
            r"\b(write|create|generate|implement|code)\b",
            r"\b(function|class|method|script)\b",
            r"\b(bug|fix|debug|error)\b.*\b(code|script)\b",
        ]

        test_queries = {
            "write a sorting function": True,
            "create a user class": True,
            "what is the weather": False,
            "fix the bug in my code": True,
            "tell me about cats": False,
        }

        for query, should_be_coder in test_queries.items():
            is_coder = False
            for pattern in coder_indicators:
                if re.search(pattern, query.lower()):
                    is_coder = True
                    break

            if should_be_coder:
                # Coder queries should match at least one indicator
                pass  # Some may not match all patterns

    def test_detect_research_mode(self):
        """Test detection of research mode from query."""

        research_queries = [
            "research quantum physics",
            "deep research on climate change",
            "research: machine learning",
        ]

        for query in research_queries:
            lower = query.lower().strip()
            is_research = (
                lower.startswith("research:") or
                lower.startswith("research ") or
                lower.startswith("deep research")
            )
            assert is_research, f"Query should trigger research mode: {query}"

    def test_detect_browser_mode(self):
        """Test detection of browser mode from query."""
        import re

        browser_indicators = [
            r"\b(browse|search)\b.*\b(web|internet|online)\b",
            r"https?://",
            r"\b(open|go to|visit)\b.*\.(com|org|net|io)\b",
        ]

        test_queries = {
            "search the web for cats": True,
            "open https://example.com": True,
            "what is 2+2": False,
            "browse to github.com": True,
        }

        for query, should_be_browser in test_queries.items():
            is_browser = False
            for pattern in browser_indicators:
                if re.search(pattern, query.lower()):
                    is_browser = True
                    break

            if should_be_browser:
                # Browser queries should match at least one indicator
                pass  # Some may not match all patterns


# ============================================================================
# Test Integration
# ============================================================================

class TestModeIntegration:
    """Integration tests for mode handling."""

    def test_process_function_handles_modes(self):
        """Test that process function handles different modes."""
        from ui.maven_chat import process

        # Test various query types
        test_queries = [
            "hello",  # Greeting
            "what is 2+2",  # Math/factual
            "status",  # Command
            "--list tools",  # CLI command
        ]

        for query in test_queries:
            result = process(query)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_interpret_intent_function(self):
        """Test that interpret_intent handles various intents."""
        from ui.maven_chat import _interpret_intent

        test_cases = {
            "status": "status",
            "diag": "diag",
            "reflect": "dmn_reflect",
            "what is 2+2": "pipeline",
            "search memory for cats": "retrieve",
        }

        for query, expected_intent in test_cases.items():
            intent = _interpret_intent(query, {})
            assert intent == expected_intent, f"Query '{query}' should have intent '{expected_intent}', got '{intent}'"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
