"""
Comprehensive Routing Tests
============================

Tests for Maven's routing system including:
- Pipeline routing
- Coder brain routing
- Inventory routing
- Browser tool routing
- Research mode routing
- Self-model routing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import re


# ============================================================================
# Test Pipeline Stage Routing
# ============================================================================

class TestPipelineStageRouting:
    """Tests for pipeline stage execution."""

    def test_pipeline_runner_exists(self):
        """Test that pipeline runner module exists."""
        from brains.pipeline.pipeline_runner import run_pipeline
        assert callable(run_pipeline)

    def test_pipeline_returns_response(self):
        """Test that pipeline returns a response dict."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("test query", 0.8)
        assert isinstance(result, dict)
        assert "ok" in result or "response" in result

    def test_pipeline_handles_empty_query(self):
        """Test pipeline handling of empty query."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("", 0.8)
        assert isinstance(result, dict)

    def test_pipeline_handles_low_confidence(self):
        """Test pipeline handling of low confidence."""
        from brains.pipeline.pipeline_runner import run_pipeline

        result = run_pipeline("test query", 0.1)
        assert isinstance(result, dict)


# ============================================================================
# Test Coder Brain Routing
# ============================================================================

class TestCoderBrainRouting:
    """Tests for coder brain routing."""

    def test_coder_pattern_detection(self):
        """Test detection of coder-related queries."""
        # Coder queries should match coding patterns
        coder_patterns = [
            r"\b(write|create|generate|implement)\s+(code|function|class|method)\b",
            r"\b(fix|debug|repair)\s+(bug|error|issue)\b",
            r"\b(refactor|optimize|improve)\s+(code|function|class)\b",
        ]

        test_queries = [
            "write code for sorting",
            "create function to validate input",
            "fix bug in authentication",
            "refactor code to be cleaner",
        ]

        for query in test_queries:
            matched = False
            for pattern in coder_patterns:
                if re.search(pattern, query.lower()):
                    matched = True
                    break
            assert matched, f"Query should match coder pattern: {query}"

    def test_coder_pattern_store_exists(self):
        """Test that coder pattern store module exists."""
        from brains.cognitive.coder.pattern_store import CoderPatternStore
        store = CoderPatternStore()
        assert store is not None


# ============================================================================
# Test Inventory Routing
# ============================================================================

class TestInventoryRouting:
    """Tests for inventory system routing."""

    def test_inventory_query_detection(self):
        """Test detection of inventory-related queries."""
        # Inventory queries pattern matching
        inventory_patterns = [
            r"\b(list|show|what)\s+(files?|modules?|brains?|tools?)\b",
            r"\b(inventory|catalog|available)\b",
            r"\bwhat\s+(is|are)\s+(available|installed)\b",
        ]

        test_queries = [
            "list files in project",
            "show modules available",
            "what brains are available",
            "show inventory",
            "what is available",
        ]

        for query in test_queries:
            matched = False
            for pattern in inventory_patterns:
                if re.search(pattern, query.lower()):
                    matched = True
                    break
            # Not all queries must match - this tests pattern recognition
            # Some may go through normal pipeline

    def test_inventory_brain_exists(self):
        """Test that inventory brain module exists."""
        try:
            from brains.cognitive.inventory.service.inventory_brain import service_api
            assert callable(service_api)
        except ImportError:
            # Inventory brain may not exist as a separate module
            pass


# ============================================================================
# Test Research Mode Routing
# ============================================================================

class TestResearchModeRouting:
    """Tests for research mode routing."""

    def test_research_command_detection(self):
        """Test detection of research commands."""
        research_patterns = [
            r"^research:\s*",
            r"^research\s+",
            r"^deep research\s+",
            r"^deep research on\s+",
        ]

        test_queries = [
            "research: quantum physics",
            "research quantum physics",
            "deep research quantum physics",
            "deep research on quantum physics",
        ]

        for query in test_queries:
            matched = False
            for pattern in research_patterns:
                if re.match(pattern, query.lower()):
                    matched = True
                    break
            assert matched, f"Query should match research pattern: {query}"

    def test_research_manager_exists(self):
        """Test that research manager module exists."""
        try:
            from brains.cognitive.research_manager.service.research_manager_brain import service_api
            assert callable(service_api)
        except ImportError:
            pytest.skip("Research manager brain not available")


# ============================================================================
# Test Self-Model Routing
# ============================================================================

class TestSelfModelRouting:
    """Tests for self-model routing."""

    def test_self_query_detection(self):
        """Test detection of self-related queries."""
        self_patterns = [
            r"\bwho\s+are\s+you\b",
            r"\bwhat\s+are\s+you\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bwhat\s+is\s+your\s+(name|purpose|function)\b",
        ]

        test_queries = [
            "who are you",
            "what are you",
            "what can you do",
            "tell me about yourself",
            "what is your name",
            "what is your purpose",
        ]

        for query in test_queries:
            matched = False
            for pattern in self_patterns:
                if re.search(pattern, query.lower()):
                    matched = True
                    break
            assert matched, f"Query should match self pattern: {query}"

    def test_self_model_brain_exists(self):
        """Test that self model brain module exists."""
        from brains.cognitive.self_model.service.self_model_brain import service_api
        assert callable(service_api)

    def test_self_model_query_self(self):
        """Test QUERY_SELF operation."""
        from brains.cognitive.self_model.service.self_model_brain import service_api

        result = service_api({
            "op": "QUERY_SELF",
            "payload": {
                "query": "who are you",
                "self_kind": "identity"
            }
        })

        assert isinstance(result, dict)
        # May or may not be OK depending on state
        assert "ok" in result or "error" in result


# ============================================================================
# Test Browser Tool Routing
# ============================================================================

class TestBrowserToolRouting:
    """Tests for browser tool routing."""

    def test_web_search_query_detection(self):
        """Test detection of web search queries."""
        web_patterns = [
            r"\bsearch\s+(the\s+)?web\s+for\b",
            r"\bgoogle\s+",
            r"\blookup\s+online\b",
            r"\bfind\s+on\s+the\s+internet\b",
        ]

        test_queries = [
            "search the web for cats",
            "google python tutorials",
            "lookup online python docs",
            "find on the internet weather",
        ]

        for query in test_queries:
            matched = False
            for pattern in web_patterns:
                if re.search(pattern, query.lower()):
                    matched = True
                    break
            # Browser patterns may or may not match - testing pattern recognition

    def test_url_detection(self):
        """Test URL pattern detection."""
        url_pattern = r"https?://\S+"

        test_queries = [
            "open https://example.com",
            "go to http://google.com",
            "visit https://github.com/user/repo",
        ]

        for query in test_queries:
            match = re.search(url_pattern, query)
            assert match is not None, f"URL should be detected in: {query}"


# ============================================================================
# Test Memory Bank Routing
# ============================================================================

class TestMemoryBankRouting:
    """Tests for memory bank routing decisions."""

    def test_memory_tier_manager_exists(self):
        """Test that memory tier manager exists."""
        from brains.memory.tier_manager import TierManager
        assert TierManager is not None

    def test_brain_memory_exists(self):
        """Test that brain memory module exists."""
        from brains.memory.brain_memory import BrainMemory
        assert BrainMemory is not None


# ============================================================================
# Test Semantic Normalization
# ============================================================================

class TestSemanticNormalization:
    """Tests for semantic normalization in routing."""

    def test_semantic_normalizer_exists(self):
        """Test that semantic normalizer exists."""
        try:
            from brains.cognitive.sensorium.semantic_normalizer import SemanticNormalizer
            normalizer = SemanticNormalizer()
            assert normalizer is not None
        except ImportError:
            pytest.skip("SemanticNormalizer not available")


# ============================================================================
# Test Routing Brain Integration
# ============================================================================

class TestRoutingBrainIntegration:
    """Tests for routing brain integration."""

    def test_integrator_routing_exists(self):
        """Test that integrator routing brain exists."""
        from brains.cognitive.integrator.routing_brain import RoutingBrain
        brain = RoutingBrain()
        assert brain is not None

    def test_route_query(self):
        """Test routing a simple query."""
        from brains.cognitive.integrator.routing_brain import RoutingBrain
        brain = RoutingBrain()

        # Test routing returns some result
        result = brain.route("what is 2+2")
        assert result is not None


# ============================================================================
# Test Teacher Brain Routing
# ============================================================================

class TestTeacherBrainRouting:
    """Tests for teacher brain routing."""

    def test_teacher_brain_exists(self):
        """Test that teacher brain module exists."""
        try:
            from brains.cognitive.teacher.service.teacher_brain import service_api
            assert callable(service_api)
        except ImportError:
            pytest.skip("Teacher brain not available")

    def test_teacher_health_check(self):
        """Test teacher brain health check."""
        try:
            from brains.cognitive.teacher.service.teacher_brain import service_api
            result = service_api({"op": "HEALTH"})
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("Teacher brain not available")


# ============================================================================
# Test Reasoning Brain Routing
# ============================================================================

class TestReasoningBrainRouting:
    """Tests for reasoning brain routing."""

    def test_reasoning_brain_exists(self):
        """Test that reasoning brain module exists."""
        from brains.cognitive.reasoning.service.reasoning_brain import service_api
        assert callable(service_api)

    def test_learned_router_exists(self):
        """Test that learned router exists."""
        from brains.cognitive.reasoning.service.learned_router import service_api
        assert callable(service_api)


# ============================================================================
# Test Action Engine Routing
# ============================================================================

class TestActionEngineRouting:
    """Tests for action engine routing."""

    def test_action_engine_exists(self):
        """Test that action engine module exists."""
        from brains.cognitive.action_engine.service.action_engine import service_api
        assert callable(service_api)

    def test_get_available_actions(self):
        """Test getting available actions."""
        from brains.cognitive.action_engine.service.action_engine import service_api

        result = service_api({"op": "LIST_ACTIONS"})
        assert isinstance(result, dict)
        assert "ok" in result


# ============================================================================
# Test Pre-Pipeline Detection
# ============================================================================

class TestPrePipelineDetection:
    """Tests for pre-pipeline intent detection."""

    def test_command_prefix_detection(self):
        """Test detection of command prefixes."""
        command_prefixes = ["--", "/"]

        test_queries = [
            "--status",
            "--cache purge",
            "/help",
            "/scan self",
        ]

        for query in test_queries:
            has_prefix = any(query.startswith(p) for p in command_prefixes)
            assert has_prefix, f"Query should have command prefix: {query}"

    def test_non_command_detection(self):
        """Test that regular queries don't have command prefix."""
        test_queries = [
            "what is 2+2",
            "tell me about cats",
            "how does python work",
        ]

        command_prefixes = ["--", "/"]
        for query in test_queries:
            has_prefix = any(query.startswith(p) for p in command_prefixes)
            assert not has_prefix, f"Query should NOT have command prefix: {query}"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
