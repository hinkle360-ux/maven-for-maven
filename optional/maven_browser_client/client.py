"""
Maven Browser Client
===================

HTTP client for interacting with the Browser Runtime service.
Provides high-level Python API for browser automation.
"""

from __future__ import annotations

import httpx
from typing import Optional, Dict, Any

from optional.browser_runtime.models import (
    ActionResult,
    PageSnapshot,
    OpenRequest,
    ClickRequest,
    TypeRequest,
    WaitForRequest,
    ScrollRequest,
    ScreenshotRequest,
    GetPageRequest,
    ClosePageRequest,
)
from optional.browser_runtime.config import get_config


class BrowserClient:
    """Client for browser runtime HTTP API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize browser client.

        Args:
            base_url: Base URL of browser runtime server. If None, uses config.
            timeout: Default timeout for HTTP requests in seconds.
        """
        if base_url is None:
            config = get_config()
            base_url = f"http://{config.host}:{config.port}"

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ========================================================================
    # Low-level API methods
    # ========================================================================

    async def open(
        self,
        url: str,
        wait_until: str = "load",
        timeout_ms: int = 30000
    ) -> ActionResult:
        """
        Open a URL in the browser.

        Args:
            url: URL to navigate to
            wait_until: Wait condition ("load", "domcontentloaded", "networkidle")
            timeout_ms: Navigation timeout in milliseconds

        Returns:
            ActionResult with page_id and snapshot
        """
        request = OpenRequest(url=url, wait_until=wait_until, timeout_ms=timeout_ms)
        response = await self.client.post("/open", json=request.model_dump())
        response.raise_for_status()
        return ActionResult(**response.json())

    async def click(
        self,
        page_id: str,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        nth: int = 0,
        timeout_ms: int = 5000
    ) -> ActionResult:
        """
        Click an element on the page.

        Args:
            page_id: ID of the page to interact with
            selector: CSS selector for element (optional)
            text: Text content to search for (optional)
            nth: Which matching element to click (0-indexed)
            timeout_ms: Timeout for finding element

        Returns:
            ActionResult with updated snapshot
        """
        request = ClickRequest(
            page_id=page_id,
            selector=selector,
            text=text,
            nth=nth,
            timeout_ms=timeout_ms
        )
        response = await self.client.post("/click", json=request.model_dump())
        response.raise_for_status()
        return ActionResult(**response.json())

    async def type_text(
        self,
        page_id: str,
        selector: str,
        text: str,
        submit: bool = False,
        delay_ms: int = 50
    ) -> ActionResult:
        """
        Type text into an input element.

        Args:
            page_id: ID of the page to interact with
            selector: CSS selector for input element
            text: Text to type
            submit: Whether to press Enter after typing
            delay_ms: Delay between keystrokes

        Returns:
            ActionResult with updated snapshot
        """
        request = TypeRequest(
            page_id=page_id,
            selector=selector,
            text=text,
            submit=submit,
            delay_ms=delay_ms
        )
        response = await self.client.post("/type", json=request.model_dump())
        response.raise_for_status()
        return ActionResult(**response.json())

    async def wait_for(
        self,
        page_id: str,
        selector: str,
        timeout_ms: int = 5000,
        state: str = "visible"
    ) -> ActionResult:
        """
        Wait for an element to appear.

        Args:
            page_id: ID of the page to wait on
            selector: CSS selector to wait for
            timeout_ms: Timeout in milliseconds
            state: State to wait for ("attached", "visible", etc.)

        Returns:
            ActionResult indicating success or timeout
        """
        request = WaitForRequest(
            page_id=page_id,
            selector=selector,
            timeout_ms=timeout_ms,
            state=state
        )
        response = await self.client.post("/wait_for", json=request.model_dump())
        response.raise_for_status()
        return ActionResult(**response.json())

    async def scroll(
        self,
        page_id: str,
        direction: str = "down",
        amount: int = 500
    ) -> ActionResult:
        """
        Scroll the page.

        Args:
            page_id: ID of the page to scroll
            direction: Scroll direction ("up", "down", "top", "bottom")
            amount: Pixels to scroll (for up/down)

        Returns:
            ActionResult indicating success
        """
        request = ScrollRequest(page_id=page_id, direction=direction, amount=amount)
        response = await self.client.post("/scroll", json=request.model_dump())
        response.raise_for_status()
        return ActionResult(**response.json())

    async def screenshot(
        self,
        page_id: str,
        full_page: bool = False
    ) -> ActionResult:
        """
        Take a screenshot of the page.

        Args:
            page_id: ID of the page to screenshot
            full_page: Whether to capture full scrollable page

        Returns:
            ActionResult with screenshot path
        """
        request = ScreenshotRequest(page_id=page_id, full_page=full_page)
        response = await self.client.post("/screenshot", json=request.model_dump())
        response.raise_for_status()
        return ActionResult(**response.json())

    async def get_page(
        self,
        page_id: str,
        include_html: bool = True,
        include_text: bool = True
    ) -> PageSnapshot:
        """
        Get current page state.

        Args:
            page_id: ID of the page to retrieve
            include_html: Whether to include full HTML
            include_text: Whether to include extracted text

        Returns:
            PageSnapshot with current page state
        """
        request = GetPageRequest(
            page_id=page_id,
            include_html=include_html,
            include_text=include_text
        )
        response = await self.client.post("/get_page", json=request.model_dump())
        response.raise_for_status()
        return PageSnapshot(**response.json())

    async def close_page(self, page_id: str) -> Dict[str, Any]:
        """
        Close a page.

        Args:
            page_id: ID of the page to close

        Returns:
            Status dictionary
        """
        request = ClosePageRequest(page_id=page_id)
        response = await self.client.post("/close_page", json=request.model_dump())
        response.raise_for_status()
        return response.json()

    # ========================================================================
    # High-level convenience methods
    # ========================================================================

    async def search_google(self, query: str) -> PageSnapshot:
        """
        Perform a Google search.

        Args:
            query: Search query

        Returns:
            PageSnapshot of search results
        """
        # Open Google
        result = await self.open("https://www.google.com")
        if result.status == "error":
            raise RuntimeError(f"Failed to open Google: {result.error.message}")

        page_id = result.page_id

        # Type query and submit
        result = await self.type_text(
            page_id=page_id,
            selector="textarea[name=q]",
            text=query,
            submit=True
        )
        if result.status == "error":
            raise RuntimeError(f"Failed to search: {result.error.message}")

        # Wait for results
        await self.wait_for(page_id, selector="#search", timeout_ms=10000)

        # Get final page state
        return await self.get_page(page_id)

    async def open_url(self, url: str) -> tuple[str, PageSnapshot]:
        """
        Open a URL and return page ID and snapshot.

        Args:
            url: URL to open

        Returns:
            Tuple of (page_id, snapshot)
        """
        result = await self.open(url)
        if result.status == "error":
            raise RuntimeError(f"Failed to open URL: {result.error.message}")

        return result.page_id, result.snapshot

    async def get_text(self, page_id: str, selector: Optional[str] = None) -> str:
        """
        Get text content from the page.

        Args:
            page_id: ID of the page
            selector: Optional CSS selector to get text from specific element

        Returns:
            Text content
        """
        snapshot = await self.get_page(page_id, include_html=False, include_text=True)

        if selector:
            # Would need to parse HTML and extract text from selector
            # For now, return full text
            # TODO: Implement selector-based text extraction
            pass

        return snapshot.text

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if browser runtime is healthy.

        Returns:
            Health status dictionary
        """
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()
