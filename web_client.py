import json
import os
import typing as t
from html.parser import HTMLParser
from pathlib import Path
from urllib import error, parse, request

from config.web_config import SERPAPI_KEY, WEB_SEARCH_MODE

try:
    import serpapi  # type: ignore
except Exception:
    serpapi = None
    # Only warn about serpapi when API mode is explicitly requested
    if WEB_SEARCH_MODE == "api":
        print("[WEB_CLIENT] WARNING: serpapi package not installed but WEB_SEARCH_MODE=api; use browser mode instead")


class WebDocument(t.TypedDict):
    url: str
    title: str
    text: str
    source: str  # always "web"


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML parser to extract readable text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - glue
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


def _load_serpapi_key() -> str:
    """Load the SerpAPI key from env or config/api_keys.json."""

    if SERPAPI_KEY:
        return SERPAPI_KEY

    env_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if env_key:
        return env_key

    config_path = Path(__file__).resolve().parent / "config" / "api_keys.json"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        key = str(data.get("serpapi", "")).strip()
        if len(key) == 64:
            return key
        print("[WEB_CLIENT] Invalid or missing SerpAPI key")
    except FileNotFoundError:
        print("[WEB_CLIENT] config/api_keys.json not found")
    except Exception as e:  # pragma: no cover - config dependent
        print(f"[WEB_CLIENT] Failed to load SerpAPI key: {e}")
    return ""


def _fetch_page_text(url: str, timeout: float) -> str:
    """Fetch raw page text for the given URL, returning empty string on error."""

    try:
        req = request.Request(url, headers={"User-Agent": "Maven/2.0 Research Bot"})
        with request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        parser = _HTMLTextExtractor()
        parser.feed(html)
        return parser.text()[:8000]
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[WEB_CLIENT] Failed to fetch page text: {e}")
        return ""


def _parse_organic_results(payload: dict, max_results: int) -> list[dict]:
    try:
        organic = payload.get("organic_results") or []
        if isinstance(organic, list):
            return organic[:max_results]
    except Exception:
        pass
    return []


def search_web(
    query: str,
    max_results: int,
    per_request_timeout: float,
) -> t.List[WebDocument]:
    """
    Perform a web search and return up to max_results documents.

    This MUST:
    - Enforce per_request_timeout on the underlying HTTP client.
    - Return an empty list on any error (network, parse, etc.).
    - NEVER raise out of this function during normal operation.
    """

    try:
        per_request_timeout = float(per_request_timeout)
    except Exception:
        per_request_timeout = 5.0

    if serpapi is None:
        print("[WEB_CLIENT] SerpAPI client unavailable; skipping web search")
        return []

    api_key = _load_serpapi_key()
    if not api_key:
        print("[WEB_CLIENT] SerpAPI key not configured; skipping web search")
        return []

    try:
        print(f"[WEB_CLIENT] Searching SerpAPI: '{query}' (max_results={max_results})")
        client = serpapi.Client(api_key=api_key)
        payload = client.search(
            {
                "engine": "google",
                "q": query,
                "num": max_results,
            },
            timeout=per_request_timeout,
        )
    except error.URLError as e:  # pragma: no cover - network dependent
        print(f"[WEB_CLIENT] Network error during SerpAPI search: {e}")
        return []
    except Exception as e:
        print(f"[WEB_CLIENT] SerpAPI search failed: {e}")
        return []

    organic_results = _parse_organic_results(payload, max_results)
    if not organic_results:
        print("[WEB_CLIENT] No search results parsed")
        return []

    documents: list[WebDocument] = []
    for result in organic_results:
        try:
            title = str(result.get("title", "")).strip()
            url = str(result.get("link", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            text = _fetch_page_text(url, timeout=per_request_timeout) if url else ""
            if not text:
                text = snippet
            documents.append({
                "url": url,
                "title": title or url or query,
                "text": text,
                "source": "web",
            })
        except Exception as e:
            print(f"[WEB_CLIENT] Error processing search result: {e}")
            continue

    print(f"[WEB_CLIENT] Returning {len(documents)} web documents")
    return documents
