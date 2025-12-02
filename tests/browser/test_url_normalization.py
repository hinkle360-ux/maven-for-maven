"""
URL Normalization and Browser Routing Tests
============================================

Tests for:
- URL normalization (google -> https://www.google.com)
- Search URL extraction (open google and search for games -> search URL)
- Browser routing detection for natural language patterns
"""

import pytest
from unittest.mock import patch


class TestURLNormalization:
    """Tests for URL normalization in smart_routing.py"""

    def test_normalize_full_url_unchanged(self):
        """Full URLs with valid hostnames should be returned as-is."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("https://www.google.com") == "https://www.google.com"
        assert normalize_url("http://example.com/path") == "http://example.com/path"
        assert normalize_url("https://github.com/user/repo") == "https://github.com/user/repo"

    def test_normalize_url_with_bad_hostname(self):
        """URLs with protocol but short hostname should be fixed.

        This is the KEY bug fix - https://google should become https://www.google.com
        """
        from brains.cognitive.integrator.smart_routing import normalize_url

        # This was causing ERR_NAME_NOT_RESOLVED
        assert normalize_url("https://google") == "https://www.google.com"
        assert normalize_url("http://google") == "http://www.google.com"
        assert normalize_url("https://youtube") == "https://www.youtube.com"
        assert normalize_url("https://github") == "https://github.com"
        assert normalize_url("https://bing") == "https://www.bing.com"

    def test_normalize_url_with_bad_hostname_and_path(self):
        """URLs with bad hostname but valid path should fix hostname and preserve path."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("https://google/search?q=test") == "https://www.google.com/search?q=test"
        assert normalize_url("https://github/user/repo") == "https://github.com/user/repo"
        assert normalize_url("http://youtube/watch?v=abc") == "http://www.youtube.com/watch?v=abc"

    def test_normalize_url_unknown_hostname_with_protocol(self):
        """Unknown hostnames with protocol should get .com added."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("https://somesite") == "https://www.somesite.com"
        assert normalize_url("http://myapp") == "http://www.myapp.com"
        assert normalize_url("https://myapp/path") == "https://www.myapp.com/path"

    def test_normalize_known_hostname(self):
        """Known short hostnames should map to correct URLs."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("google") == "https://www.google.com"
        assert normalize_url("youtube") == "https://www.youtube.com"
        assert normalize_url("github") == "https://github.com"
        assert normalize_url("reddit") == "https://www.reddit.com"
        assert normalize_url("bing") == "https://www.bing.com"
        assert normalize_url("duckduckgo") == "https://duckduckgo.com"
        assert normalize_url("ddg") == "https://duckduckgo.com"

    def test_normalize_unknown_hostname_adds_com(self):
        """Unknown hostnames without TLD should get .com added."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("somesite") == "https://www.somesite.com"
        assert normalize_url("myapp") == "https://www.myapp.com"

    def test_normalize_domain_with_tld(self):
        """Domains with TLD should just get https:// prefix."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("example.com") == "https://example.com"
        assert normalize_url("mysite.org") == "https://mysite.org"
        assert normalize_url("www.example.com") == "https://www.example.com"

    def test_normalize_handles_whitespace(self):
        """Whitespace should be trimmed."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        assert normalize_url("  google  ") == "https://www.google.com"
        assert normalize_url("\n https://example.com \t") == "https://example.com"


class TestSearchURLExtraction:
    """Tests for search URL extraction from natural language."""

    def test_extract_open_google_and_search(self):
        """'open google and search for X' should extract search URL."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        result = extract_search_url_from_intent("open google and search for games")
        assert result == "https://www.google.com/search?q=games"

    def test_extract_go_to_google_search(self):
        """'go to google and search for X' should extract search URL."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        result = extract_search_url_from_intent("go to google and search for python tutorials")
        assert result == "https://www.google.com/search?q=python+tutorials"

    def test_extract_bing_search(self):
        """Bing search patterns should work."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        result = extract_search_url_from_intent("open bing and search for weather")
        assert result == "https://www.bing.com/search?q=weather"

    def test_extract_duckduckgo_search(self):
        """DuckDuckGo search patterns should work."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        result = extract_search_url_from_intent("navigate to duckduckgo and search for privacy tips")
        assert result == "https://duckduckgo.com/?q=privacy+tips"

    def test_extract_no_search_returns_none(self):
        """Non-search patterns should return None."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        assert extract_search_url_from_intent("open google") is None
        assert extract_search_url_from_intent("hello world") is None
        assert extract_search_url_from_intent("what is the weather") is None

    def test_extract_search_url_encodes_special_chars(self):
        """Search queries with special characters should be URL-encoded."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        result = extract_search_url_from_intent("open google and search for c++ programming")
        assert "c%2B%2B" in result or "c++" in result.lower()  # URL encoded or raw

    def test_extract_look_up_pattern(self):
        """'look up X on google' patterns should work."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        result = extract_search_url_from_intent("open google and look up recipes")
        assert result == "https://www.google.com/search?q=recipes"


class TestBrowserToolForcing:
    """Tests for browser tool forcing detection."""

    def test_force_browser_for_open_google(self):
        """'open google' should force browser tool."""
        from brains.cognitive.integrator.smart_routing import _should_force_browser_tool

        assert _should_force_browser_tool("open google", ["browser_runtime"]) is True

    def test_force_browser_for_full_url(self):
        """Full URLs should force browser tool."""
        from brains.cognitive.integrator.smart_routing import _should_force_browser_tool

        assert _should_force_browser_tool("open https://www.google.com", ["browser_runtime"]) is True
        assert _should_force_browser_tool("go to https://github.com", ["browser"]) is True

    def test_force_browser_for_search_intent(self):
        """Search intent patterns should force browser tool."""
        from brains.cognitive.integrator.smart_routing import _should_force_browser_tool

        assert _should_force_browser_tool("open google and search for games", ["browser_runtime"]) is True
        assert _should_force_browser_tool("search for python on google", ["browser"]) is True

    def test_no_force_without_browser_tool(self):
        """Should not force if browser tool not available."""
        from brains.cognitive.integrator.smart_routing import _should_force_browser_tool

        assert _should_force_browser_tool("open google", ["filesystem", "code"]) is False
        assert _should_force_browser_tool("open https://google.com", []) is False

    def test_force_browser_for_known_sites(self):
        """Known site names with action should force browser."""
        from brains.cognitive.integrator.smart_routing import _should_force_browser_tool

        assert _should_force_browser_tool("open youtube", ["browser_runtime"]) is True
        assert _should_force_browser_tool("go to github", ["browser"]) is True
        assert _should_force_browser_tool("navigate to reddit", ["browser_tool"]) is True

    def test_force_browser_for_browser_prefix(self):
        """'browser:' prefix should always force browser tool."""
        from brains.cognitive.integrator.smart_routing import _should_force_browser_tool

        assert _should_force_browser_tool("browser: google.com", ["browser_runtime"]) is True
        assert _should_force_browser_tool("browser: open this page", ["browser"]) is True


class TestKnownSiteDetection:
    """Tests for known site detection with action keywords."""

    def test_detect_google_with_open(self):
        """'open google' should be detected as known site with action."""
        from brains.cognitive.integrator.smart_routing import _has_known_site_with_action

        assert _has_known_site_with_action("open google") is True
        assert _has_known_site_with_action("go to google") is True

    def test_detect_youtube_with_action(self):
        """'go to youtube' should be detected."""
        from brains.cognitive.integrator.smart_routing import _has_known_site_with_action

        assert _has_known_site_with_action("visit youtube") is True
        assert _has_known_site_with_action("navigate to youtube") is True

    def test_no_detect_without_action(self):
        """Site name without action should not be detected."""
        from brains.cognitive.integrator.smart_routing import _has_known_site_with_action

        assert _has_known_site_with_action("google is a search engine") is False
        assert _has_known_site_with_action("I like youtube") is False


class TestSearchIntentDetection:
    """Tests for search intent detection."""

    def test_detect_search_intent(self):
        """Search keywords with search engine should be detected."""
        from brains.cognitive.integrator.smart_routing import _has_search_intent

        assert _has_search_intent("search for games on google") is True
        assert _has_search_intent("look up weather on bing") is True
        assert _has_search_intent("find recipes on duckduckgo") is True

    def test_no_detect_search_without_engine(self):
        """Search keywords without search engine should not be detected."""
        from brains.cognitive.integrator.smart_routing import _has_search_intent

        assert _has_search_intent("search for games") is False
        assert _has_search_intent("find me some food") is False

    def test_detect_combined_patterns(self):
        """Combined open + search patterns should be detected."""
        from brains.cognitive.integrator.smart_routing import _has_search_intent

        assert _has_search_intent("open google and search for games") is True
        assert _has_search_intent("go to bing and look up news") is True


class TestBrowserRouteIntegration:
    """Integration tests for the full browser routing flow."""

    def test_open_google_produces_valid_url(self):
        """'open google' should result in https://www.google.com, not https://google."""
        from brains.cognitive.integrator.smart_routing import normalize_url

        # This is the key test - "google" should normalize to a valid URL
        url = normalize_url("google")
        assert url == "https://www.google.com"
        assert "www.google.com" in url  # Must have proper hostname

    def test_search_intent_produces_search_url(self):
        """'open google and search for games' should produce search URL."""
        from brains.cognitive.integrator.smart_routing import extract_search_url_from_intent

        url = extract_search_url_from_intent("open google and search for games")
        assert url is not None
        assert "google.com/search" in url
        assert "games" in url
