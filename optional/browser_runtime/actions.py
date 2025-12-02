"""
Browser Actions
===============

Implementation of browser actions: open, click, type, wait, scroll, screenshot, etc.
Includes human-like delays, error handling, and CAPTCHA detection/solving.
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from optional.browser_runtime.config import get_config
from optional.browser_runtime.models import (
    ActionResult,
    PageSnapshot,
    BrowserError,
    OpenRequest,
    ClickRequest,
    TypeRequest,
    WaitForRequest,
    ScrollRequest,
    ScreenshotRequest,
)

logger = logging.getLogger(__name__)


async def human_delay() -> None:
    """Add a random human-like delay."""
    config = get_config()
    delay_ms = random.randint(config.min_delay_ms, config.max_delay_ms)
    await asyncio.sleep(delay_ms / 1000.0)


def create_error_result(error_type: str, message: str, details: Optional[dict] = None) -> ActionResult:
    """Create an error result."""
    return ActionResult(
        status="error",
        error=BrowserError(error_type=error_type, message=message, details=details or {}),
    )


async def create_page_snapshot(page: Page, include_html: bool = True, include_text: bool = True) -> PageSnapshot:
    """Create a snapshot of the current page state."""
    url = page.url
    title = await page.title()

    html = ""
    text = ""

    if include_html:
        html = await page.content()

    if include_text:
        # Extract visible text content
        text = await page.evaluate("() => document.body.innerText")

    return PageSnapshot(url=url, title=title, html=html, text=text)


# ============================================================================
# CAPTCHA Detection and Solving
# ============================================================================


async def detect_captcha(page: Page) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if the current page has a CAPTCHA.

    Returns:
        Tuple of (has_captcha, sitekey, captcha_type)
        - has_captcha: True if CAPTCHA detected
        - sitekey: The data-sitekey if found
        - captcha_type: "recaptcha", "hcaptcha", or None
    """
    try:
        # Check for reCAPTCHA
        recaptcha_count = await page.locator("div.g-recaptcha").count()
        if recaptcha_count > 0:
            sitekey = await page.locator("div.g-recaptcha").first.get_attribute("data-sitekey")
            return True, sitekey, "recaptcha"

        # Check for generic data-sitekey attribute (covers both reCAPTCHA and hCaptcha)
        sitekey_count = await page.locator("[data-sitekey]").count()
        if sitekey_count > 0:
            sitekey = await page.locator("[data-sitekey]").first.get_attribute("data-sitekey")
            # Determine type based on other indicators
            hcaptcha_count = await page.locator(".h-captcha, [data-hcaptcha-widget-id]").count()
            captcha_type = "hcaptcha" if hcaptcha_count > 0 else "recaptcha"
            return True, sitekey, captcha_type

        # Check for hCaptcha specifically
        hcaptcha_count = await page.locator(".h-captcha").count()
        if hcaptcha_count > 0:
            sitekey = await page.locator(".h-captcha").first.get_attribute("data-sitekey")
            return True, sitekey, "hcaptcha"

        # Check for reCAPTCHA iframe (invisible reCAPTCHA)
        recaptcha_iframe_count = await page.locator("iframe[src*='recaptcha']").count()
        if recaptcha_iframe_count > 0:
            # Try to extract sitekey from page source
            content = await page.content()
            import re
            match = re.search(r'data-sitekey=["\']([^"\']+)["\']', content)
            if match:
                return True, match.group(1), "recaptcha"
            # Also check for sitekey in script
            match = re.search(r'sitekey["\']?\s*[:=]\s*["\']([^"\']+)["\']', content)
            if match:
                return True, match.group(1), "recaptcha"

        return False, None, None

    except Exception as e:
        logger.warning(f"Error detecting CAPTCHA: {e}")
        return False, None, None


async def solve_captcha_on_page(page: Page, sitekey: str, captcha_type: str) -> Tuple[bool, Optional[str]]:
    """
    Attempt to solve a CAPTCHA on the page.

    Args:
        page: Playwright page instance
        sitekey: The CAPTCHA sitekey
        captcha_type: "recaptcha" or "hcaptcha"

    Returns:
        Tuple of (success, error_message)
    """
    config = get_config()

    if not config.captcha_enabled:
        return False, "[CAPTCHA BLOCK] CAPTCHA solving is disabled in configuration"

    try:
        # Import captcha tool
        from brains.tools.captcha_tool import (
            solve_captcha_or_request_human,
            HumanInterventionRequired,
        )

        url = page.url
        logger.info(f"[CAPTCHA] Detected {captcha_type} on {url}, attempting to solve...")

        try:
            # Attempt to solve
            token = solve_captcha_or_request_human(sitekey, url)

            # Inject the token into the page
            if captcha_type == "recaptcha":
                # Set the g-recaptcha-response textarea
                await page.evaluate(f"""
                    (function() {{
                        var responseField = document.querySelector('[name="g-recaptcha-response"]');
                        if (responseField) {{
                            responseField.innerHTML = '{token}';
                            responseField.value = '{token}';
                        }}
                        // Also try to find hidden textareas
                        var hiddenFields = document.querySelectorAll('textarea[id*="g-recaptcha-response"]');
                        hiddenFields.forEach(function(field) {{
                            field.innerHTML = '{token}';
                            field.value = '{token}';
                        }});
                    }})();
                """)

                # Try to trigger the callback for invisible reCAPTCHA
                try:
                    await page.evaluate(f"""
                        (function() {{
                            if (typeof ___grecaptcha_cfg !== 'undefined' && ___grecaptcha_cfg.clients) {{
                                for (var i in ___grecaptcha_cfg.clients) {{
                                    var client = ___grecaptcha_cfg.clients[i];
                                    if (client && client.callback) {{
                                        client.callback('{token}');
                                        return;
                                    }}
                                    // Check nested structure
                                    for (var key in client) {{
                                        if (client[key] && typeof client[key].callback === 'function') {{
                                            client[key].callback('{token}');
                                            return;
                                        }}
                                    }}
                                }}
                            }}
                            // Also try grecaptcha.execute callback
                            if (typeof grecaptcha !== 'undefined' && grecaptcha.getResponse) {{
                                // Token is already set, form should work
                            }}
                        }})();
                    """)
                except Exception as e:
                    logger.debug(f"Could not trigger reCAPTCHA callback: {e}")

            elif captcha_type == "hcaptcha":
                # Set hCaptcha response
                await page.evaluate(f"""
                    (function() {{
                        var responseField = document.querySelector('[name="h-captcha-response"]');
                        if (responseField) {{
                            responseField.innerHTML = '{token}';
                            responseField.value = '{token}';
                        }}
                        var gResponseField = document.querySelector('[name="g-recaptcha-response"]');
                        if (gResponseField) {{
                            gResponseField.innerHTML = '{token}';
                            gResponseField.value = '{token}';
                        }}
                    }})();
                """)

            # Wait a moment for any JS to process
            await asyncio.sleep(2)

            logger.info(f"[CAPTCHA] Successfully solved and injected token")
            return True, None

        except HumanInterventionRequired as e:
            logger.warning(f"[CAPTCHA] Human intervention required: {e}")
            return False, str(e)

    except ImportError:
        return False, "[CAPTCHA BLOCK] CAPTCHA tool not available"
    except Exception as e:
        logger.error(f"[CAPTCHA] Error solving CAPTCHA: {e}")
        return False, f"[CAPTCHA BLOCK] Error solving CAPTCHA: {str(e)}"


async def detect_and_solve_captcha(page: Page) -> Tuple[bool, Optional[str]]:
    """
    Detect and attempt to solve any CAPTCHA on the current page.

    This is the main entry point for CAPTCHA handling in browser actions.

    Args:
        page: Playwright page instance

    Returns:
        Tuple of (captcha_was_present, error_message_if_unsolved)
        - If no CAPTCHA: (False, None)
        - If CAPTCHA solved: (True, None)
        - If CAPTCHA unsolved: (True, error_message)
    """
    has_captcha, sitekey, captcha_type = await detect_captcha(page)

    if not has_captcha:
        return False, None

    if not sitekey:
        return True, "[CAPTCHA BLOCK] CAPTCHA detected but could not extract sitekey"

    success, error = await solve_captcha_on_page(page, sitekey, captcha_type)

    if success:
        return True, None
    else:
        return True, error


# ============================================================================
# Action Implementations
# ============================================================================


def _is_search_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if URL is a search engine URL and extract query.

    Returns:
        Tuple of (is_search, engine, query)
    """
    import re
    from urllib.parse import parse_qs

    parsed = urlparse(url)

    # Google search
    if 'google.com' in parsed.netloc and '/search' in parsed.path:
        query_params = parse_qs(parsed.query)
        query = query_params.get('q', [''])[0]
        return True, 'google', query

    # Bing search
    if 'bing.com' in parsed.netloc and '/search' in parsed.path:
        query_params = parse_qs(parsed.query)
        query = query_params.get('q', [''])[0]
        return True, 'bing', query

    # DuckDuckGo
    if 'duckduckgo.com' in parsed.netloc:
        query_params = parse_qs(parsed.query)
        query = query_params.get('q', [''])[0]
        return True, 'ddg', query

    return False, None, None


def _build_fallback_url(query: str, failed_engine: str) -> Optional[str]:
    """Build fallback search URL when primary engine fails."""
    import urllib.parse

    # Fallback chain: Google/Bing -> DuckDuckGo
    if failed_engine in ('google', 'bing'):
        return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"

    return None


async def action_open(page: Page, request: OpenRequest) -> ActionResult:
    """
    Open a URL in the browser.

    Args:
        page: Playwright page instance
        request: OpenRequest with URL and navigation options

    Returns:
        ActionResult with page snapshot or error
    """
    try:
        # Validate domain
        config = get_config()
        parsed = urlparse(request.url)
        domain = parsed.netloc

        if not config.is_domain_allowed(domain):
            return create_error_result(
                error_type="domain_blocked",
                message=f"Domain {domain} is not allowed",
                details={"domain": domain, "url": request.url},
            )

        # Check if this is a search URL (for fallback purposes)
        is_search, search_engine, search_query = _is_search_url(request.url)

        # Navigate to URL
        await page.goto(
            request.url,
            wait_until=request.wait_until,
            timeout=request.timeout_ms,
        )

        # Human-like delay
        await human_delay()

        # Check for and handle CAPTCHA
        captcha_detected, captcha_error = await detect_and_solve_captcha(page)
        if captcha_detected and captcha_error:
            # CAPTCHA was detected but could not be solved
            # Try fallback to DuckDuckGo if this was a search
            if is_search and search_query and search_engine != 'ddg':
                fallback_url = _build_fallback_url(search_query, search_engine)
                if fallback_url:
                    logger.info(f"[FALLBACK] CAPTCHA on {search_engine}, trying DuckDuckGo")
                    await page.goto(
                        fallback_url,
                        wait_until=request.wait_until,
                        timeout=request.timeout_ms,
                    )
                    await human_delay()

                    # Check if fallback also has CAPTCHA (unlikely for DDG)
                    fb_captcha, fb_error = await detect_and_solve_captcha(page)
                    if not fb_captcha or not fb_error:
                        # Fallback succeeded!
                        snapshot = await create_page_snapshot(page)
                        return ActionResult(
                            status="success",
                            snapshot=snapshot,
                            metadata={"fallback_used": "ddg", "original_engine": search_engine},
                        )

            # No fallback available or fallback failed
            snapshot = await create_page_snapshot(page)
            return ActionResult(
                status="captcha_blocked",
                snapshot=snapshot,
                error=BrowserError(
                    error_type="captcha_block",
                    message=captcha_error,
                    details={"url": request.url, "requires_human": True},
                ),
            )

        # Create snapshot
        snapshot = await create_page_snapshot(page)

        return ActionResult(status="success", snapshot=snapshot)

    except PlaywrightTimeout as e:
        return create_error_result(
            error_type="timeout",
            message=f"Navigation to {request.url} timed out",
            details={"timeout_ms": request.timeout_ms},
        )
    except Exception as e:
        return create_error_result(
            error_type="navigation_error",
            message=f"Failed to navigate to {request.url}: {str(e)}",
            details={"exception": type(e).__name__},
        )


async def action_click(page: Page, request: ClickRequest) -> ActionResult:
    """
    Click an element on the page.

    Args:
        page: Playwright page instance
        request: ClickRequest with selector or text to click

    Returns:
        ActionResult with page snapshot or error
    """
    try:
        # Find element by selector or text
        if request.selector:
            locator = page.locator(request.selector)
        elif request.text:
            locator = page.get_by_text(request.text)
        else:
            return create_error_result(
                error_type="invalid_request",
                message="Either selector or text must be provided for click action",
            )

        # Get nth matching element
        if request.nth > 0:
            locator = locator.nth(request.nth)

        # Wait for element and click
        await locator.click(timeout=request.timeout_ms)

        # Human-like delay
        await human_delay()

        # Create snapshot
        snapshot = await create_page_snapshot(page)

        return ActionResult(status="success", snapshot=snapshot)

    except PlaywrightTimeout:
        return create_error_result(
            error_type="selector_not_found",
            message=f"Element not found: {request.selector or request.text}",
            details={"selector": request.selector, "text": request.text, "timeout_ms": request.timeout_ms},
        )
    except Exception as e:
        return create_error_result(
            error_type="click_error",
            message=f"Failed to click element: {str(e)}",
            details={"exception": type(e).__name__},
        )


async def action_type(page: Page, request: TypeRequest) -> ActionResult:
    """
    Type text into an input element.

    Args:
        page: Playwright page instance
        request: TypeRequest with selector, text, and options

    Returns:
        ActionResult with page snapshot or error
    """
    try:
        # Find input element
        locator = page.locator(request.selector)

        # Type text with delay
        await locator.fill("")  # Clear existing content
        await locator.type(request.text, delay=request.delay_ms)

        # Submit if requested
        if request.submit:
            await locator.press("Enter")

        # Human-like delay
        await human_delay()

        # Create snapshot
        snapshot = await create_page_snapshot(page)

        return ActionResult(status="success", snapshot=snapshot)

    except PlaywrightTimeout:
        return create_error_result(
            error_type="selector_not_found",
            message=f"Input element not found: {request.selector}",
            details={"selector": request.selector},
        )
    except Exception as e:
        return create_error_result(
            error_type="type_error",
            message=f"Failed to type text: {str(e)}",
            details={"exception": type(e).__name__},
        )


async def action_wait_for(page: Page, request: WaitForRequest) -> ActionResult:
    """
    Wait for an element or condition.

    Args:
        page: Playwright page instance
        request: WaitForRequest with selector and wait options

    Returns:
        ActionResult indicating success or timeout
    """
    try:
        if request.selector:
            locator = page.locator(request.selector)
            await locator.wait_for(state=request.state, timeout=request.timeout_ms)

        return ActionResult(status="success")

    except PlaywrightTimeout:
        return create_error_result(
            error_type="timeout",
            message=f"Timeout waiting for element: {request.selector}",
            details={"selector": request.selector, "timeout_ms": request.timeout_ms, "state": request.state},
        )
    except Exception as e:
        return create_error_result(
            error_type="wait_error",
            message=f"Error while waiting: {str(e)}",
            details={"exception": type(e).__name__},
        )


async def action_scroll(page: Page, request: ScrollRequest) -> ActionResult:
    """
    Scroll the page.

    Args:
        page: Playwright page instance
        request: ScrollRequest with direction and amount

    Returns:
        ActionResult indicating success
    """
    try:
        if request.direction == "top":
            await page.evaluate("window.scrollTo(0, 0)")
        elif request.direction == "bottom":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif request.direction == "down":
            await page.evaluate(f"window.scrollBy(0, {request.amount})")
        elif request.direction == "up":
            await page.evaluate(f"window.scrollBy(0, -{request.amount})")

        # Human-like delay
        await human_delay()

        return ActionResult(status="success")

    except Exception as e:
        return create_error_result(
            error_type="scroll_error",
            message=f"Failed to scroll: {str(e)}",
            details={"exception": type(e).__name__},
        )


async def action_screenshot(page: Page, request: ScreenshotRequest) -> ActionResult:
    """
    Take a screenshot of the page.

    Args:
        page: Playwright page instance
        request: ScreenshotRequest with options

    Returns:
        ActionResult with screenshot path
    """
    try:
        config = get_config()
        screenshots_dir = config.log_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        import time

        timestamp = int(time.time() * 1000)
        screenshot_path = screenshots_dir / f"screenshot_{timestamp}.png"

        # Take screenshot
        await page.screenshot(path=str(screenshot_path), full_page=request.full_page)

        snapshot = await create_page_snapshot(page, include_html=False)
        snapshot.screenshot_path = str(screenshot_path)

        return ActionResult(status="success", snapshot=snapshot)

    except Exception as e:
        return create_error_result(
            error_type="screenshot_error",
            message=f"Failed to take screenshot: {str(e)}",
            details={"exception": type(e).__name__},
        )
