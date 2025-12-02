"""
Browser Tool
============

Maven tool for executing browser automation tasks.
Provides high-level interface for Maven's cognitive stack to use the browser.

Supports multiple search engines:
- DuckDuckGo (default, no CAPTCHA)
- Bing
- Google (may require CAPTCHA solving)
"""

from __future__ import annotations

import asyncio
import urllib.parse
import uuid
from typing import Dict, Any, Optional, Literal
from datetime import datetime, timezone

from optional.maven_browser_client.client import BrowserClient
from optional.maven_browser_client.types import (
    BrowserPlan,
    BrowserAction,
    ActionType,
    BrowserTaskResult,
    TaskStatus,
)


# ============================================================================
# Search Engine URL Builders
# ============================================================================

SearchEngine = Literal["ddg", "bing", "google"]

# Default fallback chain: if primary engine fails with CAPTCHA, try these
FALLBACK_CHAIN = {
    "google": "ddg",  # Google CAPTCHA -> fallback to DuckDuckGo
    "bing": "ddg",    # Bing issues -> fallback to DuckDuckGo
    "ddg": None,      # DuckDuckGo is the last resort
}


def build_ddg_url(query: str) -> str:
    """Build DuckDuckGo search URL."""
    return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"


def build_bing_url(query: str) -> str:
    """Build Bing search URL."""
    return f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"


def build_google_url(query: str) -> str:
    """Build Google search URL."""
    return f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"


def build_search_url(query: str, engine: SearchEngine = "ddg") -> str:
    """
    Build search URL for the specified engine.

    Args:
        query: Search query string
        engine: Search engine ("ddg", "bing", "google")

    Returns:
        Full search URL
    """
    engine = engine.lower()
    if engine == "google":
        return build_google_url(query)
    elif engine == "bing":
        return build_bing_url(query)
    else:
        return build_ddg_url(query)


def get_fallback_engine(primary: SearchEngine, error_type: Optional[str] = None) -> Optional[str]:
    """
    Get fallback engine when primary fails.

    Args:
        primary: Primary engine that failed
        error_type: Type of error (captcha_block, timeout, etc.)

    Returns:
        Fallback engine name or None if no fallback available
    """
    # Trigger fallback on CAPTCHA blocks or certain errors
    if error_type in ("captcha_block", "captcha_blocked", "timeout", "navigation_error"):
        return FALLBACK_CHAIN.get(primary.lower())
    return None


async def execute_browser_plan(plan: BrowserPlan) -> BrowserTaskResult:
    """
    Execute a browser plan and return the result.

    Args:
        plan: BrowserPlan with goal and steps

    Returns:
        BrowserTaskResult with execution outcome
    """
    task_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    steps_executed = 0
    current_page_id: Optional[str] = None
    final_url: Optional[str] = None
    final_text: Optional[str] = None
    screenshot_path: Optional[str] = None
    error_message: Optional[str] = None

    try:
        async with BrowserClient() as client:
            # Check browser runtime is healthy
            try:
                health = await client.health_check()
                if health.get("status") != "healthy":
                    raise RuntimeError("Browser runtime is not healthy")
            except Exception as e:
                return BrowserTaskResult(
                    task_id=task_id,
                    goal=plan.goal,
                    status=TaskStatus.FAILED,
                    error_message=f"Browser runtime unavailable: {str(e)}",
                    duration_seconds=0,
                )

            # Execute each step in the plan
            for i, action in enumerate(plan.steps):
                if i >= plan.max_steps:
                    break

                steps_executed += 1

                try:
                    result = await _execute_action(client, action, current_page_id)

                    # Update current page ID if action created/changed it
                    if result.get("page_id"):
                        current_page_id = result["page_id"]

                    # Check for errors
                    if result.get("status") == "error":
                        error_message = result.get("error", {}).get("message", "Unknown error")
                        return BrowserTaskResult(
                            task_id=task_id,
                            goal=plan.goal,
                            status=TaskStatus.FAILED,
                            steps_executed=steps_executed,
                            error_message=error_message,
                            duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                        )

                    # Extract final state
                    if result.get("snapshot"):
                        snapshot = result["snapshot"]
                        final_url = snapshot.get("url")
                        final_text = snapshot.get("text")
                        screenshot_path = snapshot.get("screenshot_path")

                except Exception as e:
                    error_message = f"Error executing step {i + 1}: {str(e)}"
                    return BrowserTaskResult(
                        task_id=task_id,
                        goal=plan.goal,
                        status=TaskStatus.FAILED,
                        steps_executed=steps_executed,
                        error_message=error_message,
                        duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                    )

            # Success!
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return BrowserTaskResult(
                task_id=task_id,
                goal=plan.goal,
                status=TaskStatus.COMPLETED,
                steps_executed=steps_executed,
                duration_seconds=duration,
                final_url=final_url,
                final_text=final_text,
                screenshot_path=screenshot_path,
            )

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        return BrowserTaskResult(
            task_id=task_id,
            goal=plan.goal,
            status=TaskStatus.FAILED,
            steps_executed=steps_executed,
            error_message=f"Unexpected error: {str(e)}",
            duration_seconds=duration,
        )


async def _execute_action(
    client: BrowserClient,
    action: BrowserAction,
    current_page_id: Optional[str]
) -> Dict[str, Any]:
    """
    Execute a single browser action.

    Args:
        client: BrowserClient instance
        action: BrowserAction to execute
        current_page_id: Current page ID (if any)

    Returns:
        Action result dictionary
    """
    params = action.params.copy()

    if action.action == ActionType.OPEN:
        result = await client.open(**params)
        return {
            "status": result.status,
            "page_id": result.page_id,
            "snapshot": result.snapshot.model_dump() if result.snapshot else None,
            "error": result.error.model_dump() if result.error else None,
        }

    elif action.action == ActionType.CLICK:
        if not current_page_id:
            raise ValueError("No active page for click action")
        params["page_id"] = current_page_id
        result = await client.click(**params)
        return {
            "status": result.status,
            "page_id": result.page_id,
            "snapshot": result.snapshot.model_dump() if result.snapshot else None,
            "error": result.error.model_dump() if result.error else None,
        }

    elif action.action == ActionType.TYPE:
        if not current_page_id:
            raise ValueError("No active page for type action")
        params["page_id"] = current_page_id
        result = await client.type_text(**params)
        return {
            "status": result.status,
            "page_id": result.page_id,
            "snapshot": result.snapshot.model_dump() if result.snapshot else None,
            "error": result.error.model_dump() if result.error else None,
        }

    elif action.action == ActionType.WAIT_FOR:
        if not current_page_id:
            raise ValueError("No active page for wait_for action")
        params["page_id"] = current_page_id
        result = await client.wait_for(**params)
        return {
            "status": result.status,
            "page_id": result.page_id,
            "error": result.error.model_dump() if result.error else None,
        }

    elif action.action == ActionType.SCROLL:
        if not current_page_id:
            raise ValueError("No active page for scroll action")
        params["page_id"] = current_page_id
        result = await client.scroll(**params)
        return {
            "status": result.status,
            "page_id": result.page_id,
            "error": result.error.model_dump() if result.error else None,
        }

    elif action.action == ActionType.SCREENSHOT:
        if not current_page_id:
            raise ValueError("No active page for screenshot action")
        params["page_id"] = current_page_id
        result = await client.screenshot(**params)
        return {
            "status": result.status,
            "page_id": result.page_id,
            "snapshot": result.snapshot.model_dump() if result.snapshot else None,
            "error": result.error.model_dump() if result.error else None,
        }

    elif action.action == ActionType.EXTRACT_TEXT:
        if not current_page_id:
            raise ValueError("No active page for extract_text action")
        snapshot = await client.get_page(current_page_id, include_html=False, include_text=True)
        return {
            "status": "success",
            "page_id": current_page_id,
            "snapshot": snapshot.model_dump(),
        }

    else:
        raise ValueError(f"Unknown action type: {action.action}")


# ============================================================================
# Synchronous wrapper for Maven's tool interface
# ============================================================================


def run_browser_task(goal: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a browser task (synchronous wrapper).

    This is the main entry point for Maven's agent executor.

    Args:
        goal: High-level goal description
        plan: Browser plan dictionary

    Returns:
        Task result dictionary
    """
    # Convert plan dict to BrowserPlan
    browser_plan = BrowserPlan(**plan)
    browser_plan.goal = goal

    # Run async execution
    result = asyncio.run(execute_browser_plan(browser_plan))

    # Convert to dict for Maven
    return result.model_dump()


def simple_google_search(query: str) -> Dict[str, Any]:
    """
    Simple Google search (convenience function).

    Args:
        query: Search query

    Returns:
        Task result dictionary
    """
    from maven_browser_client.types import GOOGLE_SEARCH_PATTERN

    # Create plan from template
    plan = GOOGLE_SEARCH_PATTERN.model_copy()

    # Replace {query} placeholder in template
    for step in plan.steps:
        if "text" in step.params and "{query}" in step.params["text"]:
            step.params["text"] = step.params["text"].replace("{query}", query)

    plan.goal = f"Search Google for: {query}"

    # Execute
    result = asyncio.run(execute_browser_plan(plan))
    return result.model_dump()


def open_url(url: str) -> Dict[str, Any]:
    """
    Open a URL and return page content (convenience function).

    Args:
        url: URL to open

    Returns:
        Task result dictionary
    """
    from maven_browser_client.types import OPEN_URL_PATTERN

    # Create plan from template
    plan = OPEN_URL_PATTERN.model_copy()
    plan.steps[0].params["url"] = url
    plan.goal = f"Open URL: {url}"

    # Execute
    result = asyncio.run(execute_browser_plan(plan))
    return result.model_dump()


# ============================================================================
# Multi-Engine Web Search
# ============================================================================


async def _execute_search_with_fallback(
    query: str,
    engine: str = "ddg",
    max_fallbacks: int = 2,
) -> Dict[str, Any]:
    """
    Execute a web search with automatic fallback on CAPTCHA/errors.

    Args:
        query: Search query
        engine: Initial search engine ("ddg", "bing", "google")
        max_fallbacks: Maximum number of fallback attempts

    Returns:
        Task result dictionary with engine used
    """
    current_engine = engine.lower()
    attempts = 0

    async with BrowserClient() as client:
        while attempts <= max_fallbacks:
            url = build_search_url(query, current_engine)

            try:
                result = await client.open(url=url)

                # Check for CAPTCHA block or error
                if result.status == "captcha_blocked":
                    error_type = "captcha_block"
                elif result.status == "error" and result.error:
                    error_type = result.error.error_type
                else:
                    error_type = None

                # Success - return result
                if result.status == "success":
                    return {
                        "status": "success",
                        "engine": current_engine,
                        "url": url,
                        "page_id": result.page_id,
                        "snapshot": result.snapshot.model_dump() if result.snapshot else None,
                    }

                # Check for fallback
                fallback = get_fallback_engine(current_engine, error_type)
                if fallback and attempts < max_fallbacks:
                    attempts += 1
                    current_engine = fallback
                    continue

                # No fallback available or max attempts reached
                return {
                    "status": result.status,
                    "engine": current_engine,
                    "url": url,
                    "page_id": result.page_id,
                    "error": result.error.model_dump() if result.error else None,
                    "snapshot": result.snapshot.model_dump() if result.snapshot else None,
                }

            except Exception as e:
                # Try fallback on exception
                fallback = get_fallback_engine(current_engine, "navigation_error")
                if fallback and attempts < max_fallbacks:
                    attempts += 1
                    current_engine = fallback
                    continue

                return {
                    "status": "error",
                    "engine": current_engine,
                    "url": url,
                    "error": {"error_type": "exception", "message": str(e)},
                }

    return {
        "status": "error",
        "engine": engine,
        "error": {"error_type": "client_error", "message": "Failed to connect to browser runtime"},
    }


def search_web(query: str, engine: str = "ddg") -> Dict[str, Any]:
    """
    Perform a web search using the specified engine.

    This is the main entry point for Maven's search functionality.
    Supports automatic fallback when CAPTCHA is encountered.

    Args:
        query: Search query string
        engine: Search engine to use ("ddg", "bing", "google")
                Default is "ddg" (DuckDuckGo) which doesn't use CAPTCHA

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - engine: Engine that was used (may differ from requested if fallback occurred)
        - url: Search URL
        - page_id: Browser page ID (if successful)
        - snapshot: Page snapshot with text/html (if successful)
        - error: Error details (if failed)

    Example:
        >>> result = search_web("python tutorials", engine="ddg")
        >>> if result["status"] == "success":
        ...     print(result["snapshot"]["text"])
    """
    return asyncio.run(_execute_search_with_fallback(query, engine))


def google_search(query: str) -> Dict[str, Any]:
    """
    Search Google (convenience function).

    Note: May trigger CAPTCHA. Falls back to DuckDuckGo if blocked.
    """
    return search_web(query, engine="google")


def bing_search(query: str) -> Dict[str, Any]:
    """
    Search Bing (convenience function).
    """
    return search_web(query, engine="bing")


def ddg_search(query: str) -> Dict[str, Any]:
    """
    Search DuckDuckGo (convenience function).

    This is the recommended default - no CAPTCHA issues.
    """
    return search_web(query, engine="ddg")
