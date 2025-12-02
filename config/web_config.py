import os
from dataclasses import dataclass


SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# Web search mode: "browser" uses Playwright browser, "api" uses SerpAPI
# Browser mode is default and doesn't require SERPAPI_KEY
WEB_SEARCH_MODE = os.getenv("WEB_SEARCH_MODE", "browser").lower()

# Only warn about SERPAPI when API mode is explicitly requested
if WEB_SEARCH_MODE == "api" and not SERPAPI_KEY:
    print("[WEB_CONFIG] WARNING: SERPAPI_API_KEY not set but WEB_SEARCH_MODE=api; falling back to browser mode")


@dataclass
class WebResearchConfig:
    """Configuration defaults for web research limits and provider selection."""

    enabled: bool = True
    max_seconds: int = 1200
    max_requests: int = 20
    provider: str = "duckduckgo"
    serpapi_key: str = SERPAPI_KEY or ""
    # Search mode: "browser" (default) or "api" (requires SERPAPI_KEY)
    search_mode: str = WEB_SEARCH_MODE if (WEB_SEARCH_MODE != "api" or SERPAPI_KEY) else "browser"


WEB_RESEARCH_CONFIG = WebResearchConfig()
