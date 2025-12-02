"""
Web Search Tool
===============

Unified web search tool for Maven that:
1. Searches the web using multiple engines (Bing, DuckDuckGo, Google)
2. Extracts structured results (title, URL, snippet)
3. Integrates with research_manager for answer synthesis
4. Stores key facts into memory for future use

Priority order: Bing (most reliable) -> DuckDuckGo -> Google (best-effort)

Features:
- Comprehensive block page detection (BLOCKED, NO_RESULTS, PARSER_FAILED states)
- HTML snapshot debugging when extraction fails
- Robust Bing selectors with multiple fallback patterns
- Explicit error reporting instead of silent failures

Debug output: HTML snapshots saved to .debug/web_search/{engine}-{timestamp}.html
"""

from __future__ import annotations

import os
import re
import urllib.parse
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, field

# Import answer style for detail level control
try:
    from brains.cognitive.language.answer_style import (
        DetailLevel,
        infer_detail_level,
        get_web_search_synthesis_instruction,
    )
    _answer_style_available = True
except Exception:
    _answer_style_available = False
    # Provide fallback
    class DetailLevel:  # type: ignore
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"
    def infer_detail_level(question, context=None):  # type: ignore
        return DetailLevel.MEDIUM
    def get_web_search_synthesis_instruction(detail_level, is_followup=False):  # type: ignore
        return "- Synthesize the information into a clear, helpful answer\n- Be concise (2-3 sentences for simple queries, a short paragraph for complex ones)"


# ============================================================================
# Types
# ============================================================================

SearchEngine = Literal["duckduckgo", "bing", "google", "auto"]
SearchMode = Literal["browser", "api"]


class FailureReason(Enum):
    """Distinct failure states for web search extraction."""
    SUCCESS = "success"
    BLOCKED = "blocked"           # Engine returned block/418/captcha page
    NO_RESULTS = "no_results"     # Valid SERP but no results found
    PARSER_FAILED = "parser_failed"  # Could not parse the page structure
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


# Debug output directory
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".debug", "web_search")


def _ensure_debug_dir():
    """Ensure debug directory exists."""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        return True
    except Exception:
        return False


def _save_debug_html(engine: str, html: str, url: str = "", reason: str = ""):
    """Save HTML snapshot for debugging failed extractions."""
    if not _ensure_debug_dir():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{engine}-{timestamp}.html"
    filepath = os.path.join(DEBUG_DIR, filename)

    try:
        header = f"""<!--
DEBUG SNAPSHOT
Engine: {engine}
URL: {url}
Timestamp: {timestamp}
Reason: {reason}
HTML Length: {len(html)} chars
-->

"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + html)

        print(f"[WEB_SEARCH_DEBUG] Saved HTML snapshot: {filepath}")
        return filepath
    except Exception as e:
        print(f"[WEB_SEARCH_DEBUG] Failed to save snapshot: {e}")
        return None


def _log_html_preview(engine: str, html: str, max_chars: int = 500):
    """Log a preview of the HTML for debugging."""
    preview = html[:max_chars].replace("\n", " ").replace("\r", "")
    if len(html) > max_chars:
        preview += "..."
    print(f"[WEB_SEARCH_DEBUG] {engine} HTML preview ({len(html)} chars): {preview}")


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    engine: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "engine": self.engine,
        }


@dataclass
class SearchResponse:
    """Response from a web search."""
    results: List[SearchResult] = field(default_factory=list)
    engine_used: str = ""
    query: str = ""
    success: bool = False
    error: Optional[str] = None
    raw_page_id: Optional[str] = None
    failure_reason: FailureReason = FailureReason.UNKNOWN
    debug_html_path: Optional[str] = None  # Path to debug HTML if saved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "engine_used": self.engine_used,
            "query": self.query,
            "success": self.success,
            "error": self.error,
            "raw_page_id": self.raw_page_id,
            "failure_reason": self.failure_reason.value,
            "debug_html_path": self.debug_html_path,
        }


# ============================================================================
# URL Builders
# ============================================================================

def build_duckduckgo_url(query: str) -> str:
    """Build DuckDuckGo search URL."""
    return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"


def build_bing_url(query: str) -> str:
    """Build Bing search URL."""
    return f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"


def build_google_url(query: str) -> str:
    """Build Google search URL."""
    return f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"


def build_search_url(query: str, engine: str) -> str:
    """Build search URL for the specified engine."""
    engine = engine.lower()
    if engine == "google":
        return build_google_url(query)
    elif engine == "bing":
        return build_bing_url(query)
    else:
        return build_duckduckgo_url(query)


def detect_engine_from_url(url: str) -> Optional[str]:
    """Detect which search engine a URL belongs to."""
    url_lower = url.lower()
    if "duckduckgo.com" in url_lower:
        return "duckduckgo"
    elif "bing.com" in url_lower:
        return "bing"
    elif "google.com" in url_lower:
        return "google"
    return None


def extract_query_from_url(url: str) -> Optional[str]:
    """Extract the search query from a search engine URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        # All major engines use 'q' parameter
        if "q" in params:
            return params["q"][0]
    except Exception:
        pass
    return None


def is_search_url(url: str) -> bool:
    """Check if a URL is a search engine search page."""
    return detect_engine_from_url(url) is not None and extract_query_from_url(url) is not None


# ============================================================================
# HTML Result Extractors
# ============================================================================

def extract_duckduckgo_results(html: str) -> List[SearchResult]:
    """Extract search results from DuckDuckGo HTML."""
    results = []

    # DuckDuckGo result patterns - multiple selectors for different page versions
    # Pattern 1: Standard web results (data-nrn attribute based)
    pattern1 = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]*)</a>.*?'
        r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([^<]*)',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern 2: New DDG layout with article elements
    pattern2 = re.compile(
        r'<article[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>.*?<h2[^>]*>([^<]*)</h2>.*?'
        r'<span[^>]*>([^<]*)</span>.*?</article>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern 3: Simple link + text extraction
    pattern3 = re.compile(
        r'<a[^>]*href="(https?://(?!duckduckgo)[^"]+)"[^>]*>([^<]{10,})</a>',
        re.DOTALL | re.IGNORECASE
    )

    # Try each pattern
    for match in pattern1.finditer(html):
        url = match.group(1).strip()
        title = _clean_html(match.group(2).strip())
        snippet = _clean_html(match.group(3).strip())
        if url and title and not url.startswith("javascript:"):
            results.append(SearchResult(title=title, url=url, snippet=snippet, engine="duckduckgo"))

    if not results:
        for match in pattern2.finditer(html):
            url = match.group(1).strip()
            title = _clean_html(match.group(2).strip())
            snippet = _clean_html(match.group(3).strip())
            if url and title and not url.startswith("javascript:"):
                results.append(SearchResult(title=title, url=url, snippet=snippet, engine="duckduckgo"))

    # Fallback: extract from text content if HTML parsing fails
    if not results:
        # Look for URLs that aren't DDG internal links
        for match in pattern3.finditer(html):
            url = match.group(1).strip()
            title = _clean_html(match.group(2).strip())
            if url and title and "duckduckgo" not in url.lower():
                results.append(SearchResult(title=title, url=url, snippet="", engine="duckduckgo"))

    return results[:10]  # Limit to top 10


def extract_bing_results(html: str) -> List[SearchResult]:
    """Extract search results from Bing HTML."""
    results = []

    # Bing uses <li class="b_algo"> for organic results
    # The structure is typically:
    #   <li class="b_algo">
    #     <h2><a href="URL">Title</a></h2>
    #     <div class="b_caption">
    #       <p>Snippet text...</p>
    #     </div>
    #   </li>

    # STRATEGY 1: Split HTML by <li class="b_algo"> markers
    # This avoids regex issues with nested </li> tags
    split_pattern = r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>'
    parts = re.split(split_pattern, html, flags=re.IGNORECASE)

    # Skip first part (content before first b_algo)
    for block in parts[1:]:
        url = None
        title = None
        snippet = ""

        # Limit block to reasonable size (next major section marker)
        # This prevents bleeding into next result
        for end_marker in ['<li class="b_algo"', '<li class="b_ad"', '<footer', '</ol>', '</ul>']:
            marker_pos = block.lower().find(end_marker.lower())
            if marker_pos > 0:
                block = block[:marker_pos]
                break

        # TITLE EXTRACTION: Handle nested HTML tags within <a>
        # Pattern 1: h2 > a with any content inside (including nested tags)
        h2_match = re.search(
            r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL | re.IGNORECASE
        )
        if h2_match:
            url = h2_match.group(1).strip()
            # Clean nested HTML from title
            title = _clean_html(h2_match.group(2).strip())

        # Pattern 2: Just <a href> with h2 wrapper (different ordering)
        if not url:
            link_match = re.search(
                r'<a[^>]*href="(https?://(?!www\.bing\.com)[^"]+)"[^>]*>(.*?)</a>',
                block,
                re.DOTALL | re.IGNORECASE
            )
            if link_match:
                candidate_url = link_match.group(1).strip()
                candidate_title = _clean_html(link_match.group(2).strip())
                # Only use if it looks like a real title (5+ chars, not just icons)
                if candidate_title and len(candidate_title) >= 5:
                    url = candidate_url
                    title = candidate_title

        # Pattern 3: data-* attributes sometimes have URLs
        if not url:
            data_url_match = re.search(
                r'data-(?:url|href)="(https?://(?!www\.bing\.com)[^"]+)"',
                block,
                re.IGNORECASE
            )
            if data_url_match:
                url = data_url_match.group(1).strip()

        # Skip if no URL found
        if not url:
            continue

        # Skip Bing internal URLs
        if "bing.com" in url.lower() or "microsoft.com/bing" in url.lower():
            continue

        # SNIPPET EXTRACTION: Try multiple patterns
        snippet_patterns = [
            # Standard b_caption with nested content
            r'<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>.*?<p[^>]*>(.*?)</p>',
            # b_lineclamp class (line-clamped snippets)
            r'<p[^>]*class="[^"]*b_(?:lineclamp|paractl)[^"]*"[^>]*>(.*?)</p>',
            # Any div with algoSlug or snippet-like class
            r'<div[^>]*class="[^"]*(?:algoSlug|snippet|description)[^"]*"[^>]*>(.*?)</div>',
            # Caption div with span content
            r'<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>.*?<span[^>]*>(.*?)</span>',
            # Generic paragraph with substantial content
            r'<p[^>]*>(.{30,}?)</p>',
        ]

        for pattern in snippet_patterns:
            snippet_match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
            if snippet_match:
                candidate_snippet = _clean_html(snippet_match.group(1).strip())
                # Only use if it's substantial text (not just a URL or date)
                if len(candidate_snippet) >= 20 and not candidate_snippet.startswith("http"):
                    snippet = candidate_snippet
                    break

        # Generate title from URL if we have URL but no title
        if url and not title:
            # Extract domain as fallback title
            from urllib.parse import urlparse
            parsed = urlparse(url)
            title = parsed.netloc.replace("www.", "")

        if url and title:
            results.append(SearchResult(
                title=title[:200],  # Limit title length
                url=url,
                snippet=snippet[:500],  # Limit snippet length
                engine="bing"
            ))

    # STRATEGY 2: If block splitting failed, try direct h2 > a pattern on full HTML
    if not results:
        for match in re.finditer(
            r'<h2[^>]*>\s*<a[^>]*href="(https?://(?!www\.bing\.com)[^"]+)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL | re.IGNORECASE
        ):
            url = match.group(1).strip()
            title = _clean_html(match.group(2).strip())
            if url and title and "bing.com" not in url.lower():
                results.append(SearchResult(title=title[:200], url=url, snippet="", engine="bing"))

    # STRATEGY 3: Look for cite elements (often contain clean URLs for results)
    if not results:
        for match in re.finditer(
            r'<cite[^>]*>(https?://[^<]+)</cite>',
            html,
            re.IGNORECASE
        ):
            url = match.group(1).strip()
            if "bing.com" not in url.lower():
                # Try to find nearby title
                pos = match.start()
                nearby_html = html[max(0, pos-500):pos+100]
                title_match = re.search(r'<a[^>]*>([^<]{5,})</a>', nearby_html, re.IGNORECASE)
                title = _clean_html(title_match.group(1)) if title_match else url
                results.append(SearchResult(title=title[:200], url=url, snippet="", engine="bing"))

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique_results.append(r)

    return unique_results[:10]


def extract_google_results(html: str) -> List[SearchResult]:
    """Extract search results from Google HTML."""
    results = []

    # Google uses <div class="g"> for organic results
    # Pattern 1: Standard Google result with h3 and cite
    pattern1 = re.compile(
        r'<div[^>]*class="[^"]*g[^"]*"[^>]*>.*?'
        r'<a[^>]*href="([^"]+)"[^>]*>.*?<h3[^>]*>([^<]*(?:<[^>]+>[^<]*)*)</h3>.*?'
        r'(?:<div[^>]*class="[^"]*VwiC3b[^"]*"[^>]*>([^<]*(?:<[^>]+>[^<]*)*)</div>|'
        r'<span[^>]*>([^<]*)</span>)',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern 2: Simplified Google extraction
    pattern2 = re.compile(
        r'<a[^>]*href="(/url\?q=([^&]+)|https?://[^"]+)"[^>]*>.*?'
        r'<h3[^>]*>([^<]+)</h3>',
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern1.finditer(html):
        url = match.group(1).strip()
        # Skip Google internal URLs
        if url.startswith("/") and not url.startswith("/url"):
            continue
        # Handle /url?q= style redirects
        if url.startswith("/url?q="):
            url = urllib.parse.unquote(url[7:].split("&")[0])
        title = _clean_html(match.group(2).strip())
        snippet = _clean_html((match.group(3) or match.group(4) or "").strip())
        if url and title and url.startswith("http"):
            results.append(SearchResult(title=title, url=url, snippet=snippet, engine="google"))

    if not results:
        for match in pattern2.finditer(html):
            raw_url = match.group(1).strip()
            if raw_url.startswith("/url?q="):
                url = urllib.parse.unquote(match.group(2))
            else:
                url = raw_url
            title = _clean_html(match.group(3).strip())
            if url and title and url.startswith("http") and "google.com" not in url:
                results.append(SearchResult(title=title, url=url, snippet="", engine="google"))

    return results[:10]


def extract_results_from_html(html: str, engine: str) -> List[SearchResult]:
    """Extract search results from HTML based on the engine."""
    engine = engine.lower()
    if engine == "duckduckgo":
        return extract_duckduckgo_results(html)
    elif engine == "bing":
        return extract_bing_results(html)
    elif engine == "google":
        return extract_google_results(html)
    else:
        # Try each extractor in order
        results = extract_duckduckgo_results(html)
        if not results:
            results = extract_bing_results(html)
        if not results:
            results = extract_google_results(html)
        return results


def extract_results_from_text(text: str) -> List[SearchResult]:
    """
    Extract search results from plain text content.

    This is a fallback when HTML parsing fails. It looks for URL-like
    patterns and nearby text that could be titles/snippets.
    """
    results = []

    # Pattern to find URLs with surrounding context
    url_pattern = re.compile(
        r'([^\n]+?)\s*(https?://[^\s\n]+)\s*([^\n]*)',
        re.IGNORECASE
    )

    seen_urls = set()
    for match in url_pattern.finditer(text):
        before = match.group(1).strip()
        url = match.group(2).strip()
        after = match.group(3).strip()

        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Use text before URL as title, text after as snippet
        title = before if len(before) > 3 else after[:50]
        snippet = after if before else ""

        if url and (title or snippet):
            results.append(SearchResult(
                title=_clean_html(title[:100]),
                url=url,
                snippet=_clean_html(snippet[:300]),
                engine="text_fallback"
            ))

    return results[:10]


def _clean_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================================
# Block Page Detection
# ============================================================================

# Generic CAPTCHA/block indicators
CAPTCHA_INDICATORS = [
    "unusual traffic",
    "verify you're not a robot",
    "captcha",
    "recaptcha",
    "are you a robot",
    "verify that you are not a robot",
    "our systems have detected",
    "automated queries",
]


def detect_captcha(html: str, text: str = "") -> bool:
    """Detect if the page is a CAPTCHA/block page."""
    combined = (html + " " + text).lower()
    return any(indicator in combined for indicator in CAPTCHA_INDICATORS)


def detect_duckduckgo_block(html: str, url: str = "") -> tuple[bool, str]:
    """
    Detect DuckDuckGo block/error pages.

    Returns:
        Tuple of (is_blocked, reason_message)
    """
    url_lower = url.lower()
    html_lower = html.lower()

    # DuckDuckGo's 418/block page
    if "/static-pages/418.html" in url_lower:
        return True, "DUCKDUCKGO_BLOCKED: 418 page (rate limited or abuse detection)"

    # Check for error page indicators in URL
    if "static-pages" in url_lower and any(x in url_lower for x in ["error", "block", "403", "418"]):
        return True, f"DUCKDUCKGO_BLOCKED: Error page detected in URL"

    # Check page title for protection/error
    title_match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if any(x in title for x in ["protection", "blocked", "error", "403", "418", "access denied"]):
            return True, f"DUCKDUCKGO_BLOCKED: Title indicates block page: '{title_match.group(1)}'"

    # Check for empty/minimal DDG pages (no search results structure)
    if "duckduckgo" in html_lower:
        # DDG should have result containers
        has_results_structure = any(x in html_lower for x in [
            "result__a", "results--main", "web-result", "links_main"
        ])
        if not has_results_structure and len(html) < 5000:
            return True, "DUCKDUCKGO_BLOCKED: Page lacks normal search result structure"

    return False, ""


def detect_google_block(html: str, url: str = "") -> tuple[bool, str]:
    """
    Detect Google block/sorry pages.

    Returns:
        Tuple of (is_blocked, reason_message)
    """
    html_lower = html.lower()

    # Check page title
    title_match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if "sorry" in title:
            return True, "GOOGLE_BLOCKED: 'Sorry' page detected"
        if "unusual traffic" in title:
            return True, "GOOGLE_BLOCKED: Unusual traffic detection"
        # Detect if browser redirected to DuckDuckGo instead of Google
        if "duckduckgo" in title:
            return True, "GOOGLE_BLOCKED: Browser redirected to DuckDuckGo (check browser settings)"

    # Check for sorry page content
    if "sorry" in html_lower and "unusual traffic" in html_lower:
        return True, "GOOGLE_BLOCKED: Sorry/unusual traffic page"

    # Check for captcha form
    if "captcha" in html_lower or "recaptcha" in html_lower:
        return True, "GOOGLE_BLOCKED: CAPTCHA page"

    # Check for consent page (GDPR)
    if "consent.google.com" in html_lower or "before you continue" in html_lower:
        return True, "GOOGLE_BLOCKED: Consent/GDPR wall"

    # Detect if we got DuckDuckGo page instead of Google (browser redirect)
    if "duckduckgo" in html_lower and "google" not in html_lower:
        return True, "GOOGLE_BLOCKED: Got DuckDuckGo page instead of Google (browser redirect)"

    return False, ""


def detect_bing_block(html: str, url: str = "") -> tuple[bool, str]:
    """
    Detect Bing block/error pages.

    Returns:
        Tuple of (is_blocked, reason_message)
    """
    html_lower = html.lower()

    # Check page title
    title_match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if any(x in title for x in ["error", "blocked", "sorry", "403", "captcha"]):
            return True, f"BING_BLOCKED: Title indicates block page: '{title_match.group(1)}'"

    # Check for captcha
    if "captcha" in html_lower:
        return True, "BING_BLOCKED: CAPTCHA detected"

    # Check for empty result container - bing should have b_content or b_results
    has_results_structure = any(x in html_lower for x in [
        "b_algo", "b_results", "b_content"
    ])

    # If it's claiming to be Bing but has no results structure and is small
    if "bing" in html_lower and not has_results_structure and len(html) < 5000:
        return True, "BING_BLOCKED: Page lacks normal search result structure"

    return False, ""


def detect_block_page(html: str, url: str, engine: str) -> tuple[bool, str, FailureReason]:
    """
    Comprehensive block page detection for any engine.

    Returns:
        Tuple of (is_blocked, message, failure_reason)
    """
    engine_lower = engine.lower()

    # Engine-specific detection first
    if engine_lower == "duckduckgo":
        is_blocked, message = detect_duckduckgo_block(html, url)
        if is_blocked:
            return True, message, FailureReason.BLOCKED

    elif engine_lower == "google":
        is_blocked, message = detect_google_block(html, url)
        if is_blocked:
            return True, message, FailureReason.BLOCKED

    elif engine_lower == "bing":
        is_blocked, message = detect_bing_block(html, url)
        if is_blocked:
            return True, message, FailureReason.BLOCKED

    # Generic CAPTCHA detection as fallback
    if detect_captcha(html):
        return True, f"{engine.upper()}_CAPTCHA: Generic CAPTCHA indicators detected", FailureReason.BLOCKED

    return False, "", FailureReason.SUCCESS


# ============================================================================
# Main Search Function
# ============================================================================

# Fallback order when engine fails
# Bing first - usually least hostile to automation and we have robust selectors
# DuckDuckGo second - good privacy but often blocks automated access (418 pages)
# Google last - most aggressive blocking and consent walls
FALLBACK_ORDER = ["bing", "duckduckgo", "google"]


def search(
    query: str,
    engine: SearchEngine = "auto",
    mode: SearchMode = "browser",
    max_results: int = 5,
) -> SearchResponse:
    """
    Perform a web search and return structured results.

    Args:
        query: Natural language search query
        engine: Search engine to use ("duckduckgo", "bing", "google", "auto")
                "auto" tries DuckDuckGo first, then Bing, then Google
        mode: Search mode ("browser" for Playwright, "api" for direct API)
        max_results: Maximum number of results to return

    Returns:
        SearchResponse with structured results

    Example:
        >>> response = search("latest games 2024", engine="auto")
        >>> for result in response.results:
        ...     print(f"{result.title}: {result.url}")
    """
    print(f"[WEB_SEARCH] Starting search: query=\"{query}\", engine={engine}, mode={mode}")

    if not query or not query.strip():
        return SearchResponse(
            success=False,
            error="Empty query",
            query=query,
        )

    # Determine engine order
    if engine == "auto":
        engines_to_try = FALLBACK_ORDER.copy()
    else:
        engines_to_try = [engine.lower()]
        # Add fallbacks if primary fails
        for fallback in FALLBACK_ORDER:
            if fallback != engine.lower() and fallback not in engines_to_try:
                engines_to_try.append(fallback)

    # Try each engine until we get results
    last_error = None
    last_failure_reason = FailureReason.UNKNOWN
    engines_tried = []

    for current_engine in engines_to_try:
        print(f"[WEB_SEARCH] Trying engine: {current_engine}")
        engines_tried.append(current_engine)

        try:
            response = _execute_search(query, current_engine, mode)

            if response.success and response.results:
                print(f"[WEB_SEARCH] Success with {current_engine}: {len(response.results)} results")
                # Limit results
                response.results = response.results[:max_results]
                return response
            elif response.error:
                # More detailed logging based on failure reason
                reason = response.failure_reason
                if reason == FailureReason.BLOCKED:
                    print(f"[WEB_SEARCH] {current_engine} BLOCKED: {response.error}")
                elif reason == FailureReason.PARSER_FAILED:
                    print(f"[WEB_SEARCH] {current_engine} PARSER_FAILED: {response.error}")
                elif reason == FailureReason.NO_RESULTS:
                    print(f"[WEB_SEARCH] {current_engine} NO_RESULTS: {response.error}")
                else:
                    print(f"[WEB_SEARCH] {current_engine} failed: {response.error}")

                last_error = response.error
                last_failure_reason = response.failure_reason
                # Continue to next engine
                continue
        except Exception as e:
            print(f"[WEB_SEARCH] {current_engine} exception: {e}")
            last_error = str(e)
            last_failure_reason = FailureReason.UNKNOWN
            continue

    # All engines failed - provide comprehensive error message
    engines_str = ", ".join(engines_tried)
    final_error = f"All search engines failed ({engines_str}). Last error: {last_error}"
    print(f"[WEB_SEARCH] {final_error}")

    return SearchResponse(
        success=False,
        error=final_error,
        query=query,
        failure_reason=last_failure_reason,
    )


def _execute_search(query: str, engine: str, mode: str) -> SearchResponse:
    """Execute search with a specific engine."""

    if mode == "browser":
        return _execute_browser_search(query, engine)
    else:
        # API mode - not implemented yet, fall back to browser
        return _execute_browser_search(query, engine)


def _execute_browser_search(query: str, engine: str) -> SearchResponse:
    """Execute search using the browser runtime."""

    try:
        from optional.browser_runtime.browser_client import is_available, open_url

        if not is_available():
            return SearchResponse(
                success=False,
                error="Browser runtime not available",
                query=query,
                engine_used=engine,
                failure_reason=FailureReason.NETWORK_ERROR,
            )

        # Build search URL
        search_url = build_search_url(query, engine)
        print(f"[WEB_SEARCH] Opening: {search_url}")

        # Execute browser request
        result = open_url(search_url)

        if result.get("error"):
            return SearchResponse(
                success=False,
                error=result.get("error"),
                query=query,
                engine_used=engine,
                failure_reason=FailureReason.NETWORK_ERROR,
            )

        # Extract content
        snapshot = result.get("snapshot", {})
        page_id = result.get("page_id")
        # PageSnapshot model uses 'html' and 'text' fields, not 'html_content'/'text_content'
        html_content = snapshot.get("html", "") or snapshot.get("html_content", "")
        text_content = snapshot.get("text", "") or snapshot.get("text_content", "")
        final_url = snapshot.get("url", search_url)

        # Log HTML preview for debugging
        if html_content:
            _log_html_preview(engine, html_content)

        # Check for block pages (comprehensive detection)
        is_blocked, block_message, block_reason = detect_block_page(html_content, final_url, engine)
        if is_blocked:
            print(f"[WEB_SEARCH] {block_message}")
            # Save debug HTML for blocked pages
            debug_path = _save_debug_html(engine, html_content, final_url, block_message)
            return SearchResponse(
                success=False,
                error=block_message,
                query=query,
                engine_used=engine,
                raw_page_id=page_id,
                failure_reason=block_reason,
                debug_html_path=debug_path,
            )

        # Extract results from HTML first, then text as fallback
        results = []
        if html_content:
            results = extract_results_from_html(html_content, engine)
            print(f"[WEB_SEARCH] Extracted {len(results)} results from HTML")

        if not results and text_content:
            results = extract_results_from_text(text_content)
            print(f"[WEB_SEARCH] Extracted {len(results)} results from text fallback")

        if not results:
            # No results found - save debug HTML so we can inspect what went wrong
            debug_path = _save_debug_html(
                engine,
                html_content,
                final_url,
                f"PARSER_FAILED: No results extracted (HTML len={len(html_content)}, text len={len(text_content)})"
            )

            # Determine if this is NO_RESULTS or PARSER_FAILED
            # If the page has result structure markers but we got nothing, it's parser failure
            html_lower = html_content.lower()
            has_result_markers = False
            if engine == "bing":
                has_result_markers = "b_algo" in html_lower or "b_results" in html_lower
            elif engine == "google":
                has_result_markers = 'class="g"' in html_lower or "search-result" in html_lower
            elif engine == "duckduckgo":
                has_result_markers = "result__" in html_lower or "web-result" in html_lower

            if has_result_markers:
                failure_reason = FailureReason.PARSER_FAILED
                error_msg = f"PARSER_FAILED: {engine} page has result structure but extraction failed"
            else:
                failure_reason = FailureReason.NO_RESULTS
                error_msg = f"NO_RESULTS: No result markers found on {engine} page"

            print(f"[WEB_SEARCH] {error_msg}")
            return SearchResponse(
                success=False,
                error=error_msg,
                query=query,
                engine_used=engine,
                raw_page_id=page_id,
                failure_reason=failure_reason,
                debug_html_path=debug_path,
            )

        return SearchResponse(
            success=True,
            results=results,
            engine_used=engine,
            query=query,
            raw_page_id=page_id,
            failure_reason=FailureReason.SUCCESS,
        )

    except ImportError as e:
        return SearchResponse(
            success=False,
            error=f"Browser runtime not installed: {e}",
            query=query,
            engine_used=engine,
            failure_reason=FailureReason.NETWORK_ERROR,
        )
    except Exception as e:
        return SearchResponse(
            success=False,
            error=f"Browser search error: {e}",
            query=query,
            engine_used=engine,
            failure_reason=FailureReason.UNKNOWN,
        )


# ============================================================================
# Integration with Research Manager
# ============================================================================

def search_and_synthesize(
    query: str,
    engine: SearchEngine = "auto",
    max_results: int = 5,
    store_facts: bool = True,
) -> Dict[str, Any]:
    """
    Search the web and synthesize an answer using the research pipeline.

    This is the main entry point for web search queries. It:
    1. Performs the search
    2. Passes results to research_manager for synthesis
    3. Optionally stores key facts to memory
    4. Returns a natural language answer with sources

    Args:
        query: Natural language search query
        engine: Search engine preference
        max_results: Maximum results to process
        store_facts: Whether to store facts to memory

    Returns:
        Dict with:
        - answer: Natural language answer
        - sources: List of {title, url} sources
        - search_results: Raw search results
        - success: Whether search succeeded
    """
    print(f"[WEB_SEARCH_SYNTHESIZE] Query: {query}")

    # Step 1: Perform search
    search_response = search(query, engine=engine, max_results=max_results)

    if not search_response.success or not search_response.results:
        return {
            "success": False,
            "error": search_response.error or "No search results",
            "answer": f"I couldn't find information about '{query}'. {search_response.error or ''}",
            "sources": [],
            "search_results": [],
        }

    # Step 2: Format results for research manager
    results_for_research = [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
        }
        for r in search_response.results
    ]

    # Step 3: Try to synthesize answer via research_manager
    try:
        from brains.cognitive.research_manager.service.research_manager_brain import (
            _teacher_helper,
            _store_fact_record,
        )

        if _teacher_helper:
            answer = _synthesize_answer_with_teacher(query, results_for_research, _teacher_helper)
        else:
            answer = _synthesize_answer_simple(query, results_for_research)

        # Step 4: Store facts if enabled
        facts_stored = 0
        if store_facts:
            for result in search_response.results[:3]:
                if result.snippet:
                    stored = _store_fact_record(
                        content=f"{result.title}: {result.snippet}",
                        confidence=0.6,
                        source="web_search",
                        topic=query,
                        url=result.url,
                        metadata={"engine": search_response.engine_used}
                    )
                    if stored:
                        facts_stored += 1

        print(f"[WEB_SEARCH_SYNTHESIZE] Stored {facts_stored} facts")

    except ImportError:
        # Research manager not available, use simple synthesis
        answer = _synthesize_answer_simple(query, results_for_research)
        facts_stored = 0
    except Exception as e:
        print(f"[WEB_SEARCH_SYNTHESIZE] Error during synthesis: {e}")
        answer = _synthesize_answer_simple(query, results_for_research)
        facts_stored = 0

    # Format sources
    sources = [
        {"title": r.title, "url": r.url}
        for r in search_response.results[:5]
    ]

    source_urls = [s["url"] for s in sources]

    # Store web search results for follow-up handling
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import store_web_search_result
        store_web_search_result(
            query=query,
            results=results_for_research,
            engine=search_response.engine_used,
            answer=answer,
            sources=source_urls,
        )
    except Exception as e:
        print(f"[WEB_SEARCH_SYNTHESIZE] Failed to store for follow-up: {e}")

    return {
        "success": True,
        "answer": answer,
        "sources": sources,
        "search_results": results_for_research,
        "engine_used": search_response.engine_used,
        "facts_stored": facts_stored,
    }


def _synthesize_answer_with_teacher(
    query: str,
    results: List[Dict[str, Any]],
    teacher_helper,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Synthesize answer using Teacher with dynamic detail level.

    The response length is determined by infer_detail_level based on the
    question complexity and conversation context, replacing the hard-coded
    "2-3 sentences" instruction.
    """

    # Infer detail level based on query and context
    detail_level = infer_detail_level(query, context)

    # Build evidence bundle
    evidence_lines = []
    for i, r in enumerate(results[:5], 1):
        evidence_lines.append(f"{i}. {r['title']}")
        if r.get('snippet'):
            evidence_lines.append(f"   {r['snippet'][:200]}")
        evidence_lines.append(f"   Source: {r['url']}")
        evidence_lines.append("")

    evidence_text = "\n".join(evidence_lines)

    # Get synthesis instructions based on detail level
    synthesis_instructions = get_web_search_synthesis_instruction(detail_level, is_followup=False)

    prompt = f"""Based on the following web search results, provide an answer to: "{query}"

Search Results:
{evidence_text}

Instructions:
{synthesis_instructions}
- Do NOT make up information beyond what's in the results
- If the results don't fully answer the question, say so

Answer:"""

    try:
        result = teacher_helper.maybe_call_teacher(
            question=prompt,
            context={"topic": query, "task": "web_search_synthesis"},
        )

        if result and result.get("answer"):
            return str(result.get("answer")).strip()
    except Exception as e:
        print(f"[WEB_SEARCH] Teacher synthesis failed: {e}")

    # Fallback to simple synthesis
    return _synthesize_answer_simple(query, results)


def _synthesize_answer_simple(
    query: str,
    results: List[Dict[str, Any]],
) -> str:
    """Simple answer synthesis without Teacher."""

    if not results:
        return f"No results found for '{query}'."

    # Build a simple answer from snippets
    answer_parts = [f"Here's what I found about '{query}':\n"]

    for i, r in enumerate(results[:3], 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        if snippet:
            answer_parts.append(f"{i}. **{title}**: {snippet[:150]}...")
        else:
            answer_parts.append(f"{i}. **{title}**")

    return "\n".join(answer_parts)


# ============================================================================
# Tool Interface for Maven
# ============================================================================

def web_search_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maven tool interface for web search.

    Payload fields:
        query: str - The search query
        engine: str - "auto", "duckduckgo", "bing", "google"
        synthesize: bool - Whether to synthesize an answer (default: True)
        max_results: int - Maximum results (default: 5)

    Returns:
        Dict with answer, sources, and raw results
    """
    query = payload.get("query", "")
    engine = payload.get("engine", "auto")
    synthesize = payload.get("synthesize", True)
    max_results = payload.get("max_results", 5)

    if not query:
        return {"ok": False, "error": "Missing query"}

    if synthesize:
        result = search_and_synthesize(query, engine=engine, max_results=max_results)
        return {
            "ok": result.get("success", False),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "search_results": result.get("search_results", []),
            "engine_used": result.get("engine_used", ""),
            "error": result.get("error"),
        }
    else:
        response = search(query, engine=engine, max_results=max_results)
        return {
            "ok": response.success,
            "results": [r.to_dict() for r in response.results],
            "engine_used": response.engine_used,
            "error": response.error,
        }


# ============================================================================
# Service API (Standard Tool Interface)
# ============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for the web_search tool.

    Operations:
    - SEARCH: Search the web and return raw results
    - SEARCH_AND_SYNTHESIZE: Search and synthesize an answer
    - HEALTH: Health check

    Args:
        msg: Dict with "op" and optional "payload"

    Returns:
        Dict with "ok", "payload" or "error"
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "WEB_SEARCH"

    if op == "SEARCH":
        query = payload.get("query", "")
        engine = payload.get("engine", "auto")
        max_results = payload.get("max_results", 5)

        if not query:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_QUERY", "message": "Query is required"}
            }

        try:
            response = search(query, engine=engine, max_results=max_results)
            return {
                "ok": response.success,
                "op": op,
                "mid": mid,
                "payload": {
                    "results": [r.to_dict() for r in response.results],
                    "engine_used": response.engine_used,
                    "result_count": len(response.results),
                },
                "error": {"code": "SEARCH_FAILED", "message": response.error} if not response.success else None
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "SEARCH_ERROR", "message": str(e)}
            }

    if op == "SEARCH_AND_SYNTHESIZE":
        query = payload.get("query", "")
        engine = payload.get("engine", "auto")
        max_results = payload.get("max_results", 5)

        if not query:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_QUERY", "message": "Query is required"}
            }

        try:
            result = search_and_synthesize(query, engine=engine, max_results=max_results)
            return {
                "ok": result.get("success", False),
                "op": op,
                "mid": mid,
                "payload": {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "search_results": result.get("search_results", []),
                    "engine_used": result.get("engine_used", ""),
                },
                "error": {"code": "SYNTHESIS_FAILED", "message": result.get("error")} if not result.get("success") else None
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "SYNTHESIS_ERROR", "message": str(e)}
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service": "web_search",
                "capability": "web_search",
                "description": "Web search with multiple engines and answer synthesis",
                "engines": ["bing", "duckduckgo", "google"],
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"}
    }


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "web_search"
TOOL_CAPABILITY = "web_search"
TOOL_DESCRIPTION = "Search the web and synthesize answers"
TOOL_OPERATIONS = ["SEARCH", "SEARCH_AND_SYNTHESIZE", "HEALTH"]


def is_available() -> bool:
    """Check if the web search tool is available."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
        "requires_execution": False,
        "module": "brains.agent.tools.web_search_tool",
    }
