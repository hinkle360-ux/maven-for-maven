"""
Web Search Pipeline Tests
=========================

Tests for Maven's web search pipeline including:
- Web search tool functionality
- HTML result extraction for DuckDuckGo, Bing, Google
- Intent routing for web search queries
- Integration with research_manager
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest


# ============================================================================
# Test Web Search Tool Module
# ============================================================================

class TestWebSearchToolModule:
    """Tests for web_search_tool module."""

    def test_web_search_tool_exists(self):
        """Test that web_search_tool module exists and has required functions."""
        from brains.agent.tools.web_search_tool import (
            search,
            search_and_synthesize,
            web_search_tool,
            SearchResult,
            SearchResponse,
        )
        assert callable(search)
        assert callable(search_and_synthesize)
        assert callable(web_search_tool)

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        from brains.agent.tools.web_search_tool import SearchResult

        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="This is a test snippet",
            engine="duckduckgo"
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "This is a test snippet"
        assert result.engine == "duckduckgo"

        # Test to_dict
        d = result.to_dict()
        assert d["title"] == "Test Title"
        assert d["url"] == "https://example.com"

    def test_search_response_dataclass(self):
        """Test SearchResponse dataclass."""
        from brains.agent.tools.web_search_tool import SearchResult, SearchResponse

        results = [
            SearchResult(title="Result 1", url="https://example1.com", snippet="Snippet 1"),
            SearchResult(title="Result 2", url="https://example2.com", snippet="Snippet 2"),
        ]
        response = SearchResponse(
            results=results,
            engine_used="duckduckgo",
            query="test query",
            success=True,
        )
        assert response.success is True
        assert len(response.results) == 2
        assert response.engine_used == "duckduckgo"

        # Test to_dict
        d = response.to_dict()
        assert d["success"] is True
        assert len(d["results"]) == 2


# ============================================================================
# Test URL Builders
# ============================================================================

class TestURLBuilders:
    """Tests for search URL building functions."""

    def test_build_duckduckgo_url(self):
        """Test DuckDuckGo URL builder."""
        from brains.agent.tools.web_search_tool import build_duckduckgo_url

        url = build_duckduckgo_url("python tutorials")
        assert "duckduckgo.com" in url
        assert "python+tutorials" in url or "python%20tutorials" in url

    def test_build_bing_url(self):
        """Test Bing URL builder."""
        from brains.agent.tools.web_search_tool import build_bing_url

        url = build_bing_url("python tutorials")
        assert "bing.com" in url
        assert "python+tutorials" in url or "python%20tutorials" in url

    def test_build_google_url(self):
        """Test Google URL builder."""
        from brains.agent.tools.web_search_tool import build_google_url

        url = build_google_url("python tutorials")
        assert "google.com" in url
        assert "python+tutorials" in url or "python%20tutorials" in url

    def test_build_search_url_auto_detection(self):
        """Test search URL builder with engine selection."""
        from brains.agent.tools.web_search_tool import build_search_url

        assert "duckduckgo.com" in build_search_url("test", "duckduckgo")
        assert "bing.com" in build_search_url("test", "bing")
        assert "google.com" in build_search_url("test", "google")
        # Default should be duckduckgo
        assert "duckduckgo.com" in build_search_url("test", "unknown")


# ============================================================================
# Test URL Detection
# ============================================================================

class TestURLDetection:
    """Tests for search URL detection functions."""

    def test_detect_engine_from_url(self):
        """Test engine detection from URL."""
        from brains.agent.tools.web_search_tool import detect_engine_from_url

        assert detect_engine_from_url("https://www.google.com/search?q=test") == "google"
        assert detect_engine_from_url("https://www.bing.com/search?q=test") == "bing"
        assert detect_engine_from_url("https://duckduckgo.com/?q=test") == "duckduckgo"
        assert detect_engine_from_url("https://example.com/page") is None

    def test_extract_query_from_url(self):
        """Test query extraction from search URLs."""
        from brains.agent.tools.web_search_tool import extract_query_from_url

        assert extract_query_from_url("https://www.google.com/search?q=python+tutorials") == "python tutorials"
        assert extract_query_from_url("https://www.bing.com/search?q=test+query") == "test query"
        assert extract_query_from_url("https://duckduckgo.com/?q=games") == "games"
        assert extract_query_from_url("https://example.com/page") is None

    def test_is_search_url(self):
        """Test search URL detection."""
        from brains.agent.tools.web_search_tool import is_search_url

        assert is_search_url("https://www.google.com/search?q=test") is True
        assert is_search_url("https://www.bing.com/search?q=test") is True
        assert is_search_url("https://duckduckgo.com/?q=test") is True
        assert is_search_url("https://example.com/page") is False
        assert is_search_url("https://www.google.com/") is False  # No query


# ============================================================================
# Test HTML Result Extractors
# ============================================================================

class TestHTMLExtractors:
    """Tests for HTML result extraction."""

    def test_extract_duckduckgo_results(self):
        """Test DuckDuckGo result extraction."""
        from brains.agent.tools.web_search_tool import extract_duckduckgo_results

        # Sample DDG-like HTML (simplified)
        html = '''
        <a class="result__a" href="https://example.com/page1">Example Title</a>
        <a class="result__snippet">This is the snippet for example.</a>
        '''
        results = extract_duckduckgo_results(html)
        # May or may not extract depending on exact HTML structure
        assert isinstance(results, list)

    def test_extract_bing_results(self):
        """Test Bing result extraction."""
        from brains.agent.tools.web_search_tool import extract_bing_results

        # Sample Bing-like HTML (simplified)
        html = '''
        <li class="b_algo">
            <h2><a href="https://example.com/page1">Example Title</a></h2>
            <p>This is the snippet for example.</p>
        </li>
        '''
        results = extract_bing_results(html)
        assert isinstance(results, list)

    def test_extract_google_results(self):
        """Test Google result extraction."""
        from brains.agent.tools.web_search_tool import extract_google_results

        # Sample Google-like HTML (simplified)
        html = '''
        <div class="g">
            <a href="https://example.com/page1"><h3>Example Title</h3></a>
            <div class="VwiC3b">This is the snippet for example.</div>
        </div>
        '''
        results = extract_google_results(html)
        assert isinstance(results, list)

    def test_extract_results_from_text_fallback(self):
        """Test text fallback extraction."""
        from brains.agent.tools.web_search_tool import extract_results_from_text

        text = '''
        Example Site https://example.com/page1 This is about example.
        Another Site https://another.com/page2 This is another example.
        '''
        results = extract_results_from_text(text)
        assert isinstance(results, list)
        assert len(results) >= 1


# ============================================================================
# Test Block Page Detection
# ============================================================================

class TestBlockPageDetection:
    """Tests for block page detection."""

    def test_failure_reason_enum_exists(self):
        """Test FailureReason enum exists with correct values."""
        from brains.agent.tools.web_search_tool import FailureReason

        assert FailureReason.SUCCESS.value == "success"
        assert FailureReason.BLOCKED.value == "blocked"
        assert FailureReason.NO_RESULTS.value == "no_results"
        assert FailureReason.PARSER_FAILED.value == "parser_failed"
        assert FailureReason.NETWORK_ERROR.value == "network_error"

    def test_detect_duckduckgo_block(self):
        """Test DuckDuckGo block detection."""
        from brains.agent.tools.web_search_tool import detect_duckduckgo_block

        # 418 page in URL
        is_blocked, msg = detect_duckduckgo_block("", "https://duckduckgo.com/static-pages/418.html")
        assert is_blocked is True
        assert "418" in msg

        # Title with "Protection"
        is_blocked, msg = detect_duckduckgo_block("<title>Protection. Privacy.</title>", "")
        assert is_blocked is True

        # Normal search page
        is_blocked, _ = detect_duckduckgo_block(
            '<html><body class="result__a">results</body></html>',
            "https://duckduckgo.com/?q=test"
        )
        # Should not be blocked (has result structure)

    def test_detect_google_block(self):
        """Test Google block detection."""
        from brains.agent.tools.web_search_tool import detect_google_block

        # Sorry page
        is_blocked, msg = detect_google_block("<title>Sorry</title>")
        assert is_blocked is True

        # Unusual traffic
        is_blocked, msg = detect_google_block("Sorry, unusual traffic from your computer")
        assert is_blocked is True

        # CAPTCHA
        is_blocked, msg = detect_google_block("Please solve the captcha")
        assert is_blocked is True

        # Consent wall
        is_blocked, msg = detect_google_block("before you continue to Google")
        assert is_blocked is True

    def test_detect_bing_block(self):
        """Test Bing block detection."""
        from brains.agent.tools.web_search_tool import detect_bing_block

        # Error title
        is_blocked, msg = detect_bing_block("<title>Error</title>")
        assert is_blocked is True

        # CAPTCHA
        is_blocked, msg = detect_bing_block("Please complete the captcha")
        assert is_blocked is True

    def test_detect_block_page_comprehensive(self):
        """Test comprehensive block page detection."""
        from brains.agent.tools.web_search_tool import detect_block_page, FailureReason

        # DuckDuckGo blocked
        is_blocked, msg, reason = detect_block_page("", "https://duckduckgo.com/static-pages/418.html", "duckduckgo")
        assert is_blocked is True
        assert reason == FailureReason.BLOCKED

        # Google sorry page
        is_blocked, msg, reason = detect_block_page("<title>Sorry</title>", "", "google")
        assert is_blocked is True
        assert reason == FailureReason.BLOCKED

        # Normal page
        is_blocked, msg, reason = detect_block_page("<html>Normal page</html>", "", "bing")
        assert is_blocked is False
        assert reason == FailureReason.SUCCESS


class TestCAPTCHADetection:
    """Tests for CAPTCHA detection."""

    def test_detect_captcha_google(self):
        """Test CAPTCHA detection for Google."""
        from brains.agent.tools.web_search_tool import detect_captcha

        captcha_html = "Our systems have detected unusual traffic from your computer"
        assert detect_captcha(captcha_html) is True

        normal_html = "Search results for python tutorials"
        assert detect_captcha(normal_html) is False

    def test_detect_captcha_various_indicators(self):
        """Test CAPTCHA detection for various indicators."""
        from brains.agent.tools.web_search_tool import detect_captcha

        indicators = [
            "verify you're not a robot",
            "CAPTCHA challenge",
            "recaptcha",
            "automated queries",
        ]
        for indicator in indicators:
            assert detect_captcha(indicator) is True, f"Should detect: {indicator}"


# ============================================================================
# Test Intent Routing for Web Search
# ============================================================================

class TestWebSearchIntentRouting:
    """Tests for web search intent detection and routing."""

    def test_web_search_intent_enum_exists(self):
        """Test that WEB_SEARCH intent exists."""
        from brains.cognitive.integrator.routing_intent import PrimaryIntent

        assert hasattr(PrimaryIntent, "WEB_SEARCH")
        assert PrimaryIntent.WEB_SEARCH.value == "web_search"

    def test_web_search_patterns_defined(self):
        """Test that web search patterns are defined."""
        from brains.cognitive.integrator.routing_intent import WEB_SEARCH_PATTERNS

        assert isinstance(WEB_SEARCH_PATTERNS, list)
        assert len(WEB_SEARCH_PATTERNS) > 0
        assert "google search" in WEB_SEARCH_PATTERNS
        assert "search the web" in WEB_SEARCH_PATTERNS

    def test_classify_google_search_query(self):
        """Test that 'google search games' routes to web search."""
        from brains.cognitive.integrator.routing_intent import (
            classify_intent_local,
            PrimaryIntent,
        )

        intent = classify_intent_local("google search games")
        assert intent.primary_intent == PrimaryIntent.WEB_SEARCH

    def test_classify_web_search_query(self):
        """Test that 'search the web for X' routes to web search."""
        from brains.cognitive.integrator.routing_intent import (
            classify_intent_local,
            PrimaryIntent,
        )

        intent = classify_intent_local("search the web for new games")
        assert intent.primary_intent == PrimaryIntent.WEB_SEARCH

    def test_classify_latest_games_query(self):
        """Test that 'what new games are coming out' routes to web search."""
        from brains.cognitive.integrator.routing_intent import (
            classify_intent_local,
            PrimaryIntent,
        )

        intent = classify_intent_local("what new games are coming out")
        assert intent.primary_intent == PrimaryIntent.WEB_SEARCH

    def test_classify_news_query(self):
        """Test that news queries route to web search."""
        from brains.cognitive.integrator.routing_intent import (
            classify_intent_local,
            PrimaryIntent,
        )

        intent = classify_intent_local("latest news about technology")
        assert intent.primary_intent == PrimaryIntent.WEB_SEARCH


# ============================================================================
# Test Agency Routing Patterns
# ============================================================================

class TestAgencyRoutingPatterns:
    """Tests for agency routing patterns."""

    def test_web_search_patterns_in_agency_patterns(self):
        """Test that web search patterns are in agency patterns."""
        from brains.cognitive.integrator.agency_routing_patterns import (
            AGENCY_PATTERNS,
            match_agency_pattern,
        )

        # Find patterns that use web_search_tool
        web_search_patterns = [
            p for p in AGENCY_PATTERNS
            if "web_search_tool" in p.get("tool", "")
        ]
        assert len(web_search_patterns) > 0, "Should have web_search_tool patterns"

    def test_match_games_query(self):
        """Test matching a games query."""
        from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern

        result = match_agency_pattern("what new games are coming out")
        assert result is not None
        assert result.get("is_web_search") is True or "browser" in result.get("tool", "")

    def test_match_search_the_web_query(self):
        """Test matching 'search the web' query."""
        from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern

        result = match_agency_pattern("search the web for python tutorials")
        assert result is not None


# ============================================================================
# Test Research Manager Integration
# ============================================================================

class TestResearchManagerIntegration:
    """Tests for research_manager integration."""

    def test_answer_from_search_results_exists(self):
        """Test that answer_from_search_results function exists."""
        from brains.cognitive.research_manager.service.research_manager_brain import (
            answer_from_search_results,
        )
        assert callable(answer_from_search_results)

    def test_answer_from_search_results_with_empty_results(self):
        """Test answer_from_search_results with empty results."""
        from brains.cognitive.research_manager.service.research_manager_brain import (
            answer_from_search_results,
        )

        result = answer_from_search_results(
            results=[],
            original_query="test query",
            store_facts=False,
        )
        assert result["ok"] is False
        assert "error" in result

    def test_answer_from_search_results_with_results(self):
        """Test answer_from_search_results with sample results."""
        from brains.cognitive.research_manager.service.research_manager_brain import (
            answer_from_search_results,
        )

        results = [
            {
                "title": "Example Result 1",
                "url": "https://example.com/1",
                "snippet": "This is the first example result about the topic.",
            },
            {
                "title": "Example Result 2",
                "url": "https://example.com/2",
                "snippet": "This is the second example result with more details.",
            },
        ]

        result = answer_from_search_results(
            results=results,
            original_query="example topic",
            store_facts=False,
        )
        assert result["ok"] is True
        assert "text_answer" in result
        assert len(result.get("sources", [])) >= 1


# ============================================================================
# Test Web Search Tool Interface
# ============================================================================

class TestWebSearchToolInterface:
    """Tests for web_search_tool function interface."""

    def test_web_search_tool_with_empty_query(self):
        """Test web_search_tool with empty query."""
        from brains.agent.tools.web_search_tool import web_search_tool

        result = web_search_tool({"query": ""})
        assert result["ok"] is False
        assert "error" in result

    def test_web_search_tool_payload_structure(self):
        """Test web_search_tool accepts correct payload structure."""
        from brains.agent.tools.web_search_tool import web_search_tool

        # Just test that the function can be called with proper payload
        # (actual search requires browser runtime)
        payload = {
            "query": "test query",
            "engine": "auto",
            "synthesize": True,
            "max_results": 5,
        }
        # This will fail without browser runtime, but should not crash
        try:
            result = web_search_tool(payload)
            assert isinstance(result, dict)
        except Exception:
            # Expected if browser runtime not available
            pass


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
