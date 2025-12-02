"""
smart_routing.py
~~~~~~~~~~~~~~~~

LLM-assisted smart routing for Maven.

This module provides the main routing pipeline that makes routing feel like
a smart chat assistant:
- Robust to phrasing: "help me debug this", "why is this breaking",
  "this code is weird" all route to the same path.
- Dynamic: adjusts based on context, prior turns, capabilities.
- LLM helps teach routing, but never lies about capabilities or internal state.

IMPORTANT: This module layers on TOP of existing routing. It doesn't replace
the integrator's existing patterns - it enhances them with LLM-assisted
classification.

Routing Priority Order:
1. SELF-INTENT GATE: capability/self/history questions -> self_model (BYPASSES ALL)
2. LOCAL PATTERN MATCH: Fast local classification
3. TEACHER CLASSIFICATION: LLM-assisted intent classification (if enabled)
4. CAPABILITY VALIDATION: Filter against actual capabilities
5. EXISTING PATTERNS: Fall back to agency_routing_patterns if needed
6. DEFAULT: Route to language brain

Usage:
    from brains.cognitive.integrator.smart_routing import (
        compute_routing_plan,
        classify_intent,
    )

    # Get a routing plan for a message
    plan = compute_routing_plan(
        message="help me debug this Python error",
        context_bundle={"recent_turns": [...]}
    )

    if plan:
        brains = plan.final_brains  # ["coder", "reasoning"]
        tools = plan.final_tools    # ["filesystem"]
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brains.cognitive.integrator.routing_intent import (
    PrimaryIntent,
    Urgency,
    Complexity,
    RoutingIntent,
    RoutingSuggestion,
    RoutingPlan,
    classify_intent_local,
    SELF_INTENT_PATTERNS,
)

from brains.cognitive.integrator.routing_safety import (
    is_self_intent_query,
    get_self_intent_type,
    enforce_capability_boundary,
    safe_routing_fallback,
)

logger = logging.getLogger(__name__)

# Routing log paths
HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
SMART_ROUTING_LOG_PATH = MAVEN_DIR / "smart_routing_decisions.jsonl"


# =============================================================================
# ROUTING DECISION LOGGING
# =============================================================================

@dataclass
class RoutingDecisionLog:
    """
    Log entry for a routing decision.

    This is stored for training data to distill Teacher routing into patterns.
    """
    decision_id: str
    timestamp: str
    user_text: str
    context_hash: str
    local_intent: Optional[Dict[str, Any]]
    teacher_suggestion: Optional[Dict[str, Any]]
    final_plan: Dict[str, Any]
    outcome: Optional[str] = None  # Set later by feedback
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_routing_log(log_entry: RoutingDecisionLog) -> None:
    """Append a routing decision to the log file."""
    try:
        _ensure_dir(SMART_ROUTING_LOG_PATH)
        with SMART_ROUTING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry.to_dict()) + "\n")
    except Exception as e:
        logger.error("Failed to write routing log: %s", e)


def _hash_context(context: Optional[Dict[str, Any]]) -> str:
    """Create a hash of context for logging."""
    if not context:
        return "no_context"
    try:
        # Use project/episode info if available
        project = context.get("project", "")
        episode = context.get("episode_id", "")
        if project or episode:
            return f"{project}:{episode}"
        return "context_present"
    except Exception:
        return "hash_error"


# =============================================================================
# CAPABILITY VALIDATION
# =============================================================================

def _get_registered_brains() -> set:
    """Get the set of registered cognitive brains."""
    try:
        from brains.brain_roles import get_cognitive_brains
        return set(get_cognitive_brains())
    except ImportError:
        # Fallback list if brain_roles not available
        return {
            "language", "reasoning", "self_model", "teacher", "integrator",
            "planner", "coder", "research_manager", "memory_librarian",
            "action_engine", "sensorium", "self_review", "self_dmn",
            "system_history", "pattern_recognition", "affect_priority",
        }


def _get_available_tools() -> List[str]:
    """Get list of available tools from capability snapshot."""
    try:
        from brains.system_capabilities import get_available_tools
        return get_available_tools()
    except ImportError:
        return []


def _get_capability_snapshot() -> Dict[str, Any]:
    """Get current capability snapshot."""
    try:
        from brains.system_capabilities import get_capability_summary
        return get_capability_summary()
    except ImportError:
        return {
            "execution_mode": "UNKNOWN",
            "tools_available": [],
            "web_research_enabled": False,
        }


def _validate_brains(suggested: List[str], registered: set) -> Tuple[List[str], List[str]]:
    """
    Validate suggested brains against registered brains.

    Returns:
        Tuple of (valid_brains, filtered_brains)
    """
    valid = []
    filtered = []
    for brain in suggested:
        if brain in registered:
            valid.append(brain)
        else:
            filtered.append(brain)
    return valid, filtered


def _validate_tools(suggested: List[str], available: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate suggested tools against available tools.

    Returns:
        Tuple of (valid_tools, filtered_tools)
    """
    available_set = set(available)
    valid = []
    filtered = []
    for tool in suggested:
        # Handle tool paths like "web_client.search" -> check "web_search"
        tool_base = tool.split(".")[0] if "." in tool else tool
        if tool in available_set or tool_base in available_set:
            valid.append(tool)
        else:
            filtered.append(tool)
    return valid, filtered


# =============================================================================
# URL/DOMAIN DETECTION FOR BROWSER TOOL
# =============================================================================

# Regex to detect URLs and domains in user messages
_URL_PATTERN = re.compile(
    r'https?://\S+'  # Full URLs like http://example.com
    r'|www\.\S+'     # www.example.com style
    r'|\b[a-z0-9][-a-z0-9]*\.(com|net|org|io|ai|dev|co|edu|gov|app|me|info|biz|xyz)\b',  # bare domains
    re.IGNORECASE
)

# Browser action keywords that indicate browser intent even without explicit URLs
_BROWSER_ACTION_KEYWORDS = [
    "open ", "go to ", "navigate to ", "browse to ", "visit ",
    "load ", "fetch ", "open the ", "go to the ", "browse ",
]

# Search action keywords that indicate search intent
_SEARCH_ACTION_KEYWORDS = [
    "search for", "search", "look up", "lookup", "find", "query",
]

# =============================================================================
# WEB SEARCH DETECTION - Direct routing to web_search pipeline
# =============================================================================

# Patterns that should route directly to web_search tool (bypassing Teacher)
_WEB_SEARCH_DIRECT_PATTERNS = [
    # Explicit web search commands
    (r'^web\s+search\s+(.+)$', 1),           # "web search X"
    (r'^search\s+the\s+web\s+for\s+(.+)$', 1),  # "search the web for X"
    (r'^search\s+online\s+for\s+(.+)$', 1),  # "search online for X"
    (r'^look\s+up\s+online\s+(.+)$', 1),     # "look up online X"
    (r'^internet\s+search\s+(.+)$', 1),      # "internet search X"
    # Engine-specific patterns (these go through browser)
    (r'^google\s+search\s+(.+)$', 1),        # "google search X"
    (r'^bing\s+search\s+(.+)$', 1),          # "bing search X"
    (r'^duckduckgo\s+search\s+(.+)$', 1),    # "duckduckgo search X"
]


def _extract_web_search_query(message: str) -> Optional[str]:
    """
    Extract query from a direct web search command.

    Handles patterns like:
    - "web search games"
    - "search the web for physics"
    - "google search latest news"

    Returns:
        The search query if detected, None otherwise
    """
    msg_stripped = message.strip()

    for pattern, group_idx in _WEB_SEARCH_DIRECT_PATTERNS:
        match = re.match(pattern, msg_stripped, re.IGNORECASE)
        if match:
            return match.group(group_idx).strip()

    return None


def _should_force_web_search(message: str, intent: RoutingIntent) -> bool:
    """
    Determine if web_search tool should be forced based on message or intent.

    This ensures "web search X" uses the same pipeline as "google search X".

    Args:
        message: User message text
        intent: Classified intent

    Returns:
        True if web_search tool should be forced
    """
    # Force if intent is WEB_SEARCH with high confidence
    if intent.primary_intent == PrimaryIntent.WEB_SEARCH and intent.confidence >= 0.6:
        return True

    # Force if message matches direct web search patterns
    if _extract_web_search_query(message) is not None:
        return True

    return False

# Mapping of common short hostnames to their full URLs
_HOSTNAME_MAP = {
    "google": "www.google.com",
    "bing": "www.bing.com",
    "yahoo": "www.yahoo.com",
    "duckduckgo": "duckduckgo.com",
    "ddg": "duckduckgo.com",
    "github": "github.com",
    "youtube": "www.youtube.com",
    "twitter": "twitter.com",
    "x": "x.com",
    "facebook": "www.facebook.com",
    "reddit": "www.reddit.com",
    "amazon": "www.amazon.com",
    "wikipedia": "www.wikipedia.org",
    "wiki": "www.wikipedia.org",
    "stackoverflow": "stackoverflow.com",
    "linkedin": "www.linkedin.com",
    "netflix": "www.netflix.com",
    "spotify": "open.spotify.com",
    "twitch": "www.twitch.tv",
}


def normalize_url(url_or_hostname: str) -> str:
    """
    Normalize a URL or hostname to a valid full URL.

    Handles:
    - Full URLs with valid hostnames (returned as-is)
    - Full URLs with short hostnames like 'https://google' -> 'https://www.google.com'
    - Short hostnames like 'google' -> 'https://www.google.com'
    - Hostnames with TLDs like 'example.com' -> 'https://example.com'

    Args:
        url_or_hostname: URL or hostname to normalize

    Returns:
        Normalized full URL
    """
    url = url_or_hostname.strip()
    url_lower = url.lower()

    # If it has a protocol, validate and potentially fix the hostname
    if url_lower.startswith(("http://", "https://")):
        # Extract protocol and the rest
        if url_lower.startswith("https://"):
            protocol = "https://"
            rest = url[8:]  # Preserve original case for path
        else:
            protocol = "http://"
            rest = url[7:]

        # Split into hostname and path
        if "/" in rest:
            hostname, path = rest.split("/", 1)
            path = "/" + path
        else:
            hostname = rest
            path = ""

        hostname_lower = hostname.lower()

        # Check if hostname needs fixing (e.g., "google" -> "www.google.com")
        if hostname_lower in _HOSTNAME_MAP:
            return f"{protocol}{_HOSTNAME_MAP[hostname_lower]}{path}"

        # Check if hostname has no TLD (no dot) - needs fixing
        if "." not in hostname_lower:
            return f"{protocol}www.{hostname_lower}.com{path}"

        # Hostname looks valid, return as-is
        return url

    # No protocol - normalize the hostname
    if url_lower in _HOSTNAME_MAP:
        return f"https://{_HOSTNAME_MAP[url_lower]}"

    # If no dot (no TLD), it's likely a short hostname - add .com
    if "." not in url_lower:
        return f"https://www.{url_lower}.com"

    # Has TLD, just add protocol
    if url_lower.startswith("www."):
        return f"https://{url}"
    return f"https://{url}"


def extract_search_url_from_intent(message: str) -> Optional[str]:
    """
    Extract a search URL from a natural language intent.

    Handles patterns like:
    - "open google and search for games"
    - "go to google and look up python tutorials"
    - "google search for weather"

    Args:
        message: User message text

    Returns:
        Google search URL if search intent detected, None otherwise
    """
    msg_lower = message.lower()

    # Check for "open/go to [site] and search for [query]" pattern
    # Pattern: action_word + site + "and" + search_word + query
    site_search_pattern = re.compile(
        r'(?:open|go to|navigate to|browse to|visit)\s+'
        r'(google|bing|duckduckgo|ddg)\s+'
        r'(?:and\s+)?'
        r'(?:search\s+for|search|look\s+up|lookup|find)\s+'
        r'(.+)',
        re.IGNORECASE
    )

    match = site_search_pattern.search(msg_lower)
    if match:
        site = match.group(1).lower()
        query = match.group(2).strip()

        # URL-encode the query
        from urllib.parse import quote_plus
        encoded_query = quote_plus(query)

        # Build search URL based on site
        if site in ("google",):
            return f"https://www.google.com/search?q={encoded_query}"
        elif site in ("bing",):
            return f"https://www.bing.com/search?q={encoded_query}"
        elif site in ("duckduckgo", "ddg"):
            return f"https://duckduckgo.com/?q={encoded_query}"

    # Check for "google [query]" or "search [query] on google" patterns
    google_search_pattern = re.compile(
        r'(?:google|search\s+(?:for\s+)?|look\s+up\s+)(.+?)(?:\s+on\s+google)?$',
        re.IGNORECASE
    )

    # Only match if there's a clear search intent indicator
    if any(kw in msg_lower for kw in ["search", "look up", "lookup", "find"]):
        if "google" in msg_lower:
            # Extract query: everything after search keywords
            for prefix in ["search for ", "search ", "look up ", "lookup ", "find "]:
                if prefix in msg_lower:
                    idx = msg_lower.index(prefix) + len(prefix)
                    query = message[idx:].strip()
                    # Remove trailing "on google" type suffixes
                    query = re.sub(r'\s+on\s+google\s*$', '', query, flags=re.IGNORECASE)
                    query = re.sub(r'\s+on\s+bing\s*$', '', query, flags=re.IGNORECASE)
                    if query:
                        from urllib.parse import quote_plus
                        return f"https://www.google.com/search?q={quote_plus(query)}"

    return None


def _detect_url_or_domain(message: str) -> Optional[str]:
    """
    Detect if message contains a URL or domain.

    Args:
        message: User message text

    Returns:
        The matched URL/domain string, or None if not found
    """
    match = _URL_PATTERN.search(message)
    if match:
        return match.group(0)
    return None


def _detect_browser_prefix(message: str) -> Optional[str]:
    """
    Detect if message starts with 'browser:' prefix for direct browser tool call.

    Args:
        message: User message text

    Returns:
        The message content after 'browser:' prefix, or None if no prefix
    """
    msg_stripped = message.strip()
    if msg_stripped.lower().startswith("browser:"):
        return msg_stripped[8:].strip()  # Return content after "browser:"
    return None


def _has_browser_action_keyword(message: str) -> bool:
    """
    Check if message contains browser action keywords.

    Args:
        message: User message text

    Returns:
        True if browser action keyword found
    """
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in _BROWSER_ACTION_KEYWORDS)


def _has_known_site_with_action(message: str) -> bool:
    """
    Check if message contains a known site name with an action keyword.

    Handles patterns like:
    - "open google"
    - "go to youtube"
    - "open google and search for games"

    Args:
        message: User message text

    Returns:
        True if a known site with action is detected
    """
    msg_lower = message.lower()

    # Check if message has an action keyword
    if not _has_browser_action_keyword(message):
        return False

    # Check if any known hostname is mentioned
    for hostname in _HOSTNAME_MAP:
        if hostname in msg_lower:
            return True

    return False


def _has_search_intent(message: str) -> bool:
    """
    Check if message indicates a web search intent.

    Handles patterns like:
    - "search for games on google"
    - "look up python tutorials"
    - "open google and search for weather"

    Args:
        message: User message text

    Returns:
        True if search intent detected
    """
    msg_lower = message.lower()

    # Check for search keywords combined with known search engines
    search_engines = ["google", "bing", "duckduckgo", "ddg", "yahoo"]
    has_search_kw = any(kw in msg_lower for kw in _SEARCH_ACTION_KEYWORDS)
    has_search_engine = any(se in msg_lower for se in search_engines)

    return has_search_kw and has_search_engine


def _should_force_browser_tool(message: str, available_tools: List[str]) -> bool:
    """
    Determine if browser tool should be forced based on message content.

    This is the key function that ensures URLs/domains route to browser tool.

    Args:
        message: User message text
        available_tools: List of currently available tools

    Returns:
        True if browser tool should be forced
    """
    # Check if browser tool is available (could be "browser", "browser_runtime", or "browser_tool")
    browser_available = any(
        "browser" in tool.lower()
        for tool in available_tools
    )

    if not browser_available:
        return False

    # Check for browser: prefix (always force)
    if _detect_browser_prefix(message):
        return True

    # Check for search intent (e.g., "open google and search for games")
    if _has_search_intent(message):
        return True

    # Check for known site with action keyword (e.g., "open google")
    if _has_known_site_with_action(message):
        return True

    # Check for URL/domain
    if _detect_url_or_domain(message):
        return True

    # Check for browser action keywords with potential sites
    if _has_browser_action_keyword(message):
        return True

    return False


def _get_browser_tool_name(available_tools: List[str]) -> str:
    """
    Get the actual browser tool name from available tools.

    Args:
        available_tools: List of available tool names

    Returns:
        The browser tool name (e.g., "browser_runtime", "browser", "browser_tool")
    """
    # Priority order: browser_runtime > browser_tool > browser
    for preferred in ["browser_runtime", "browser_tool", "browser"]:
        if preferred in available_tools:
            return preferred

    # Fallback: find any tool with "browser" in name
    for tool in available_tools:
        if "browser" in tool.lower():
            return tool

    return "browser"  # Default fallback


# =============================================================================
# MAIN ROUTING FUNCTIONS
# =============================================================================

def classify_intent(
    user_message: str,
    context_bundle: Optional[Dict[str, Any]] = None,
    use_teacher: bool = True,
) -> RoutingIntent:
    """
    Classify the intent of a user message.

    This is the main classification function that:
    1. First tries fast local classification
    2. For self-intents, returns immediately (no Teacher needed)
    3. For other intents with low confidence, optionally uses Teacher

    Args:
        user_message: The user's message
        context_bundle: Recent conversation context
        use_teacher: Whether to use Teacher for classification (default True)

    Returns:
        RoutingIntent with classification result
    """
    # Step 1: Fast local classification
    local_intent = classify_intent_local(user_message, context_bundle)

    # Step 2: Self-intents return immediately (MUST NOT go to Teacher)
    if local_intent.is_self_intent():
        print(f"[SMART_ROUTING] Self-intent detected: {local_intent.primary_intent.value}")
        return local_intent

    # Step 3: High-confidence local classification - use it
    if local_intent.confidence >= 0.7:
        print(f"[SMART_ROUTING] High-confidence local: {local_intent.primary_intent.value} "
              f"(conf={local_intent.confidence:.2f})")
        return local_intent

    # Step 4: Low-confidence - optionally use Teacher
    if use_teacher and local_intent.confidence < 0.6:
        try:
            from brains.cognitive.teacher.service.teacher_helper import get_routing_suggestion

            capability_snapshot = _get_capability_snapshot()
            suggestion = get_routing_suggestion(
                user_message,
                context_bundle,
                capability_snapshot,
            )

            if suggestion and suggestion.get("confidence", 0) > local_intent.confidence:
                # Teacher has higher confidence, use its classification
                teacher_intent = PrimaryIntent(suggestion.get("primary_intent", "chat_answer"))
                return RoutingIntent(
                    primary_intent=teacher_intent,
                    secondary_tags=suggestion.get("secondary_tags", []),
                    urgency=local_intent.urgency,  # Keep local estimate
                    complexity=local_intent.complexity,  # Keep local estimate
                    confidence=suggestion.get("confidence", 0.5),
                    raw_signals={
                        "source": "teacher",
                        "local_intent": local_intent.to_dict(),
                        "teacher_notes": suggestion.get("notes", ""),
                    },
                )
        except Exception as e:
            print(f"[SMART_ROUTING] Teacher classification failed: {e}")

    # Fallback to local classification
    return local_intent


def compute_routing_plan(
    message: str,
    context_bundle: Optional[Dict[str, Any]] = None,
    use_teacher: bool = True,
) -> RoutingPlan:
    """
    Compute the final routing plan for a message.

    This is the main entry point for smart routing. It:
    1. Classifies intent using classify_intent()
    2. For self-intents, hard-routes to self_model (NO Teacher)
    3. Otherwise, gets Teacher suggestion and validates against capabilities
    4. Falls back to existing patterns if needed
    5. Logs the decision for training

    Args:
        message: The user's message
        context_bundle: Recent conversation context
        use_teacher: Whether to use Teacher for routing suggestions

    Returns:
        RoutingPlan with validated routing decision
    """
    start_time = time.time()
    decision_id = str(uuid.uuid4())
    context_bundle = context_bundle or {}

    # Initialize tracking
    local_intent_dict = None
    teacher_suggestion_dict = None
    validation_notes = []
    reasons = []

    # Step 1: Classify intent
    intent = classify_intent(message, context_bundle, use_teacher=use_teacher)
    local_intent_dict = intent.to_dict()

    # Step 2: SELF-INTENT GATE - capability/self/history questions
    if intent.is_self_intent():
        # Hard-route to self_model + system_capabilities
        # DO NOT call Teacher for routing here
        reasons.append(f"Self-intent detected: {intent.primary_intent.value}")
        reasons.append("Routing to self_model (Teacher bypassed)")

        plan = RoutingPlan(
            final_brains=["self_model", "system_history"] if intent.primary_intent == PrimaryIntent.HISTORY_QUESTION else ["self_model"],
            final_tools=[],  # Self-intents don't need tools
            intent=intent,
            suggestion_source="self_intent_gate",
            reasons=reasons,
            validation_notes=["Self-intent bypasses Teacher for routing"],
        )

        # Log decision
        latency_ms = int((time.time() - start_time) * 1000)
        _log_routing_decision(
            decision_id=decision_id,
            message=message,
            context_bundle=context_bundle,
            local_intent=local_intent_dict,
            teacher_suggestion=None,
            final_plan=plan,
            latency_ms=latency_ms,
        )

        print(f"[SMART_ROUTING] Self-intent plan: brains={plan.final_brains} (latency={latency_ms}ms)")
        return plan

    # Step 3: Get capability snapshot for validation
    registered_brains = _get_registered_brains()
    available_tools = _get_available_tools()
    capability_snapshot = _get_capability_snapshot()

    # Step 3.5: URL/DOMAIN DETECTION - Force browser tool if URL/domain detected
    # This is a hard rule: URLs/domains ALWAYS go to browser tool
    force_browser = _should_force_browser_tool(message, available_tools)
    browser_prefix_content = _detect_browser_prefix(message)
    detected_url = _detect_url_or_domain(message)

    if force_browser:
        browser_tool_name = _get_browser_tool_name(available_tools)
        print(f"[SMART_ROUTING] URL/Browser detection triggered: "
              f"url={detected_url}, prefix={browser_prefix_content is not None}, "
              f"tool={browser_tool_name}")
        reasons.append(f"URL/domain detected: {detected_url or 'browser action keyword'}")
        reasons.append(f"Forcing browser tool: {browser_tool_name}")

    # Step 3.6: WEB SEARCH DETECTION - Force web_search for "web search X" patterns
    # This ensures "web search physics" uses same pipeline as "google search physics"
    force_web_search = _should_force_web_search(message, intent)
    web_search_query = _extract_web_search_query(message)

    if force_web_search:
        print(f"[SMART_ROUTING] Web search detected: query={web_search_query}, "
              f"intent={intent.primary_intent.value}")
        reasons.append(f"Web search query detected: {web_search_query or 'intent-based'}")
        reasons.append("Forcing web_search tool pipeline")

    # Step 4: Get Teacher suggestion (if enabled)
    suggested_brains = []
    suggested_tools = []
    suggestion_source = "local_classification"

    if use_teacher:
        try:
            from brains.cognitive.teacher.service.teacher_helper import get_routing_suggestion

            teacher_suggestion = get_routing_suggestion(
                message,
                context_bundle,
                capability_snapshot,
            )

            if teacher_suggestion:
                teacher_suggestion_dict = teacher_suggestion
                suggested_brains = teacher_suggestion.get("recommended_brains", [])
                suggested_tools = teacher_suggestion.get("recommended_tools", [])
                suggestion_source = "teacher"
                reasons.append(f"Teacher suggestion: brains={suggested_brains}, tools={suggested_tools}")
                reasons.append(f"Teacher notes: {teacher_suggestion.get('notes', '')[:100]}")

        except Exception as e:
            reasons.append(f"Teacher failed: {str(e)[:50]}")
            print(f"[SMART_ROUTING] Teacher suggestion failed: {e}")

    # Step 5: Map intent to default brains (fallback)
    if not suggested_brains:
        suggested_brains = _intent_to_default_brains(intent)
        suggestion_source = "intent_mapping"
        reasons.append(f"Using intent mapping: {intent.primary_intent.value} -> {suggested_brains}")

    # Step 6: Validate brains against registered brains
    valid_brains, filtered_brains = _validate_brains(suggested_brains, registered_brains)
    if filtered_brains:
        validation_notes.append(f"Filtered non-existent brains: {filtered_brains}")

    # Step 7: Validate tools against available tools
    valid_tools, filtered_tools = _validate_tools(suggested_tools, available_tools)
    if filtered_tools:
        validation_notes.append(f"Filtered unavailable tools: {filtered_tools}")

    # Step 8: Check capability constraints
    if not capability_snapshot.get("execution_enabled", True):
        # Execution disabled - filter out action-oriented brains
        action_brains = {"action_engine", "coder"}
        before = valid_brains[:]
        valid_brains = [b for b in valid_brains if b not in action_brains]
        if set(before) != set(valid_brains):
            validation_notes.append("Filtered action brains (execution disabled)")

    if not capability_snapshot.get("web_research_enabled", True):
        # Web disabled - filter web tools
        before = valid_tools[:]
        valid_tools = [t for t in valid_tools if "web" not in t.lower()]
        if before != valid_tools:
            validation_notes.append("Filtered web tools (web disabled)")

    # Step 9: Apply safety enforcement (final check)
    try:
        safe_brains, safe_tools, safety_notes = enforce_capability_boundary(
            query=message,
            plan_brains=valid_brains,
            plan_tools=valid_tools,
            capability_snapshot=capability_snapshot,
        )
        valid_brains = safe_brains
        valid_tools = safe_tools
        validation_notes.extend(safety_notes)
    except Exception as e:
        print(f"[SMART_ROUTING] Safety enforcement failed, using fallback: {e}")
        valid_brains, valid_tools = safe_routing_fallback()
        validation_notes.append(f"Safety enforcement failed: {str(e)[:50]}")

    # Step 9.5: FORCE BROWSER TOOL if URL/domain was detected
    # This happens AFTER safety enforcement to ensure browser is always included
    if force_browser:
        browser_tool_name = _get_browser_tool_name(available_tools)
        if browser_tool_name not in valid_tools:
            valid_tools.append(browser_tool_name)
            validation_notes.append(f"Forced browser tool ({browser_tool_name}) due to URL/domain detection")
        # Also update suggestion source to indicate browser forcing
        if suggestion_source == "local_classification":
            suggestion_source = "url_browser_detection"

    # Step 9.6: FORCE WEB_SEARCH TOOL if web search was detected
    # This ensures "web search X" uses the same pipeline as "google search X"
    if force_web_search:
        # Add web_search to tools if not already present
        if "web_search" not in valid_tools:
            valid_tools.append("web_search")
            validation_notes.append("Forced web_search tool due to web search pattern")
        # Ensure research_manager is in brains for synthesis
        if "research_manager" not in valid_brains:
            valid_brains.insert(0, "research_manager")
            validation_notes.append("Added research_manager for web search synthesis")
        # Update suggestion source
        if suggestion_source in ("local_classification", "intent_mapping"):
            suggestion_source = "web_search_detection"

    # Step 10: Ensure we always have at least one brain
    if not valid_brains:
        valid_brains = ["language"]  # Safe default
        validation_notes.append("Defaulted to language brain (no valid brains)")

    # Step 11: Create final plan
    plan = RoutingPlan(
        final_brains=valid_brains,
        final_tools=valid_tools,
        intent=intent,
        suggestion_source=suggestion_source,
        reasons=reasons,
        validation_notes=validation_notes,
    )

    # Log decision
    latency_ms = int((time.time() - start_time) * 1000)
    _log_routing_decision(
        decision_id=decision_id,
        message=message,
        context_bundle=context_bundle,
        local_intent=local_intent_dict,
        teacher_suggestion=teacher_suggestion_dict,
        final_plan=plan,
        latency_ms=latency_ms,
    )

    print(f"[SMART_ROUTING] Final plan: brains={plan.final_brains}, tools={plan.final_tools}, "
          f"source={suggestion_source} (latency={latency_ms}ms)")

    return plan


def _intent_to_default_brains(intent: RoutingIntent) -> List[str]:
    """
    Map intent to default brains (used when Teacher doesn't provide suggestion).
    """
    mapping = {
        PrimaryIntent.CHAT_ANSWER: ["language", "reasoning"],
        PrimaryIntent.SMALL_TALK: ["language"],
        PrimaryIntent.CODE_TASK: ["coder", "reasoning"],
        PrimaryIntent.RESEARCH_QUESTION: ["research_manager", "reasoning"],
        PrimaryIntent.TOOL_REQUEST: ["action_engine"],
        PrimaryIntent.CAPABILITY_QUESTION: ["self_model"],
        PrimaryIntent.SELF_QUESTION: ["self_model"],
        PrimaryIntent.HISTORY_QUESTION: ["system_history", "memory_librarian"],
        PrimaryIntent.TASK_FOLLOWUP: ["reasoning", "language"],
        PrimaryIntent.CLARIFICATION: ["language"],
        PrimaryIntent.META_INSTRUCTION: ["integrator"],
        PrimaryIntent.FEEDBACK: ["self_review"],
        PrimaryIntent.UNKNOWN: ["language", "reasoning"],
        # WEB_SEARCH routes to research_manager which handles web search synthesis
        PrimaryIntent.WEB_SEARCH: ["research_manager", "memory_librarian"],
    }
    return mapping.get(intent.primary_intent, ["language"])


def _log_routing_decision(
    decision_id: str,
    message: str,
    context_bundle: Dict[str, Any],
    local_intent: Optional[Dict[str, Any]],
    teacher_suggestion: Optional[Dict[str, Any]],
    final_plan: RoutingPlan,
    latency_ms: int,
) -> None:
    """Log a routing decision for training data."""
    try:
        log_entry = RoutingDecisionLog(
            decision_id=decision_id,
            timestamp=_now_iso(),
            user_text=message[:500],  # Truncate
            context_hash=_hash_context(context_bundle),
            local_intent=local_intent,
            teacher_suggestion=teacher_suggestion,
            final_plan=final_plan.to_dict(),
            latency_ms=latency_ms,
        )
        _append_routing_log(log_entry)
    except Exception as e:
        logger.error("Failed to log routing decision: %s", e)


# =============================================================================
# FEEDBACK INTEGRATION
# =============================================================================

_current_decision_id: Optional[str] = None
_current_plan: Optional[RoutingPlan] = None


def set_current_routing_context(decision_id: str, plan: RoutingPlan) -> None:
    """Set the current routing context for feedback tracking."""
    global _current_decision_id, _current_plan
    _current_decision_id = decision_id
    _current_plan = plan


def apply_routing_feedback(
    verdict: str,
    user_correction: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Apply feedback to the current routing decision.

    This updates the routing log with outcome signals for training.

    Args:
        verdict: "ok", "minor_issue", "major_issue"
        user_correction: Optional user correction text (e.g., "no, I meant X")
        metadata: Optional additional metadata
    """
    global _current_decision_id

    if not _current_decision_id:
        logger.warning("No current routing decision to apply feedback to")
        return

    # Calculate reward
    reward_map = {
        "ok": 1.0,
        "minor_issue": 0.0,
        "major_issue": -1.0,
    }
    reward = reward_map.get(verdict, 0.0)

    # Apply user correction penalty
    if user_correction:
        reward -= 1.0  # Strong penalty for explicit corrections

    # Log feedback
    try:
        feedback_entry = {
            "kind": "routing_feedback",
            "decision_id": _current_decision_id,
            "timestamp": _now_iso(),
            "verdict": verdict,
            "reward": reward,
            "user_correction": user_correction,
            "metadata": metadata or {},
        }
        _ensure_dir(SMART_ROUTING_LOG_PATH)
        with SMART_ROUTING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")

        print(f"[SMART_ROUTING] Feedback applied: verdict={verdict}, reward={reward}")

    except Exception as e:
        logger.error("Failed to apply routing feedback: %s", e)


def detect_user_correction(message: str) -> bool:
    """
    Detect if a message is a user correction.

    Corrections like "no, I meant X" indicate routing failure.
    """
    correction_patterns = [
        "no, i meant",
        "no i meant",
        "that's not what i asked",
        "that's not what i meant",
        "i didn't ask",
        "i didn't mean",
        "wrong, i wanted",
        "not what i wanted",
        "let me rephrase",
        "let me clarify",
        "what i actually meant",
    ]
    msg_lower = message.lower()
    return any(p in msg_lower for p in correction_patterns)


# =============================================================================
# INTEGRATION WITH EXISTING ROUTING
# =============================================================================

def get_smart_routing_decision(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get smart routing decision for integration with existing routing.

    This function provides a simple interface for the integrator brain
    to use smart routing alongside existing patterns.

    Args:
        query: The user query
        context: Optional context dict

    Returns:
        Dict with routing decision or None if smart routing is disabled
    """
    try:
        # Build context bundle from context dict
        context_bundle = {}
        if context:
            context_bundle = {
                "recent_turns": context.get("history", [])[-4:],
                "project": context.get("project", ""),
                "episode_id": context.get("episode_id", ""),
            }

        # Compute routing plan
        plan = compute_routing_plan(query, context_bundle)

        return {
            "brains": plan.final_brains,
            "tools": plan.final_tools,
            "intent": plan.intent.primary_intent.value if plan.intent else "unknown",
            "source": plan.suggestion_source,
            "reasons": plan.reasons,
        }

    except Exception as e:
        print(f"[SMART_ROUTING] Error: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "classify_intent",
    "compute_routing_plan",
    "apply_routing_feedback",
    "detect_user_correction",
    "get_smart_routing_decision",
    "RoutingDecisionLog",
    "SMART_ROUTING_LOG_PATH",
]
