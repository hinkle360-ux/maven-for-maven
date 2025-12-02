"""
Host Web Client Implementation
==============================

Concrete implementation of web search and fetch tools using Python's
standard library (urllib). This module performs actual network I/O
and should NOT be imported by core brains.

The host runtime creates instances of these tools and injects them
into the brain context via ToolRegistry.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from brains.tools_api import WebSearchResult, WebFetchResult


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML parser to extract readable text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


def _web_enabled() -> bool:
    """Check if web research is enabled via environment or config."""
    try:
        env_override = os.getenv("MAVEN_ENABLE_WEB_RESEARCH")
        if env_override is not None:
            return env_override.strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass

    # Try to load from config
    try:
        from api.utils import CFG
        legacy_cfg = bool((CFG.get("web_research") or {}).get("enabled", False))
        return bool(CFG.get("ENABLE_WEB_RESEARCH", True) or legacy_cfg)
    except Exception:
        return True  # Default to enabled


def _timeout_seconds() -> int:
    """Get timeout from config or default."""
    try:
        from api.utils import CFG
        return max(5, min(60, int(CFG.get("WEB_RESEARCH_MAX_SECONDS", 30))))
    except Exception:
        return 30


def _parse_search_results(html: str, max_results: int) -> List[Dict[str, str]]:
    """Parse DuckDuckGo HTML search results."""
    url_pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
    snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>'

    url_matches = re.findall(url_pattern, html)
    snippet_matches = re.findall(snippet_pattern, html)

    results: List[Dict[str, str]] = []
    for i, (url, title) in enumerate(url_matches[:max_results]):
        snippet = snippet_matches[i] if i < len(snippet_matches) else ""
        if url.startswith("//duckduckgo.com/l/"):
            try:
                redirect = re.search(r"uddg=([^&]+)", url)
                if redirect:
                    url = urllib.parse.unquote(redirect.group(1))
            except Exception:
                pass
        results.append({
            "url": url.strip(),
            "title": title.strip(),
            "snippet": snippet.strip()
        })
    return results


class HostWebSearchTool:
    """
    Host implementation of web search using DuckDuckGo HTML API.

    Satisfies the WebSearchTool protocol from brains.tools_api.
    """

    def __init__(self, user_agent: str = "Maven/2.0 Research Bot"):
        self.user_agent = user_agent

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        language: str = "en",
        timeout: int = 30
    ) -> WebSearchResult:
        """
        Perform a web search and return results.

        Uses DuckDuckGo HTML API for searching.
        """
        if not _web_enabled():
            return WebSearchResult(text="", url=None, confidence=0.0)

        timeout = timeout or _timeout_seconds()
        print(f"[HOST_WEB] Searching web for '{query}' (max_results={max_results})")

        try:
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except urllib.error.URLError as e:
            print(f"[HOST_WEB] Network error during search: {e}")
            return WebSearchResult(text="", url=None, confidence=0.0)
        except Exception as e:
            print(f"[HOST_WEB] Search failed: {e}")
            return WebSearchResult(text="", url=None, confidence=0.0)

        parsed_results = _parse_search_results(html, max_results=max_results)
        if not parsed_results:
            print("[HOST_WEB] No search results parsed")
            return WebSearchResult(text="", url=None, confidence=0.0)

        first_result = parsed_results[0]
        first_url = first_result["url"]
        snippet = first_result["title"] + " " + first_result["snippet"]

        # Try to fetch the first page for more content
        page_text = ""
        try:
            fetch_result = HostWebFetchTool(user_agent=self.user_agent).fetch(
                first_url,
                timeout=timeout,
                max_chars=8000
            )
            if fetch_result.success:
                page_text = fetch_result.text
        except Exception:
            pass

        text = page_text or snippet
        confidence = 0.4 if text else 0.0

        return WebSearchResult(
            text=text,
            url=first_url,
            confidence=confidence,
            raw_results=parsed_results
        )


class HostWebFetchTool:
    """
    Host implementation of web page fetching.

    Satisfies the WebFetchTool protocol from brains.tools_api.
    """

    def __init__(self, user_agent: str = "Maven/2.0 Research Bot"):
        self.user_agent = user_agent

    def fetch(
        self,
        url: str,
        *,
        timeout: int = 30,
        max_chars: int = 8000
    ) -> WebFetchResult:
        """
        Fetch a web page and extract text content.
        """
        if not _web_enabled():
            return WebFetchResult(
                text="",
                url=url,
                status_code=0,
                success=False,
                error="Web fetch disabled"
            )

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                html = response.read().decode("utf-8", errors="ignore")
                status_code = response.status

            parser = _HTMLTextExtractor()
            parser.feed(html)
            text = parser.text()[:max_chars]

            return WebFetchResult(
                text=text,
                url=url,
                status_code=status_code,
                success=True
            )
        except urllib.error.HTTPError as e:
            return WebFetchResult(
                text="",
                url=url,
                status_code=e.code,
                success=False,
                error=str(e)
            )
        except Exception as e:
            print(f"[HOST_WEB] Failed to fetch page: {e}")
            return WebFetchResult(
                text="",
                url=url,
                status_code=0,
                success=False,
                error=str(e)
            )
