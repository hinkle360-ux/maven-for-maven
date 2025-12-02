"""
Browser Runtime Models
======================

Pydantic models for request/response schemas in the browser runtime API.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Response Models
# ============================================================================


class BrowserError(BaseModel):
    """Error information from browser operations."""

    error_type: str = Field(..., description="Type of error (timeout, selector_not_found, etc.)")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class PageSnapshot(BaseModel):
    """Snapshot of a web page's current state."""

    url: str = Field(..., description="Current URL of the page")
    title: str = Field(default="", description="Page title")
    html: str = Field(default="", description="Full HTML content")
    text: str = Field(default="", description="Extracted text content")
    screenshot_path: Optional[str] = Field(default=None, description="Path to screenshot if captured")


class ActionResult(BaseModel):
    """Result of a browser action."""

    status: Literal["success", "error", "captcha_blocked"] = Field(..., description="Status of the action")
    page_id: Optional[str] = Field(default=None, description="ID of the page this action affected")
    snapshot: Optional[PageSnapshot] = Field(default=None, description="Page snapshot after action")
    error: Optional[BrowserError] = Field(default=None, description="Error information if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Request Models
# ============================================================================


class OpenRequest(BaseModel):
    """Request to open a URL in the browser."""

    url: str = Field(..., description="URL to navigate to")
    wait_until: Literal["load", "domcontentloaded", "networkidle"] = Field(
        default="load", description="Wait condition before considering navigation complete"
    )
    timeout_ms: int = Field(default=30000, description="Navigation timeout in milliseconds")


class ClickRequest(BaseModel):
    """Request to click an element."""

    page_id: str = Field(..., description="ID of the page to interact with")
    selector: Optional[str] = Field(default=None, description="CSS selector for element to click")
    text: Optional[str] = Field(default=None, description="Text content to search for and click")
    nth: int = Field(default=0, description="Which matching element to click (0-indexed)")
    timeout_ms: int = Field(default=5000, description="Timeout for finding element")


class TypeRequest(BaseModel):
    """Request to type text into an element."""

    page_id: str = Field(..., description="ID of the page to interact with")
    selector: str = Field(..., description="CSS selector for input element")
    text: str = Field(..., description="Text to type")
    submit: bool = Field(default=False, description="Whether to press Enter after typing")
    delay_ms: int = Field(default=50, description="Delay between keystrokes in milliseconds")


class WaitForRequest(BaseModel):
    """Request to wait for an element or condition."""

    page_id: str = Field(..., description="ID of the page to wait on")
    selector: Optional[str] = Field(default=None, description="CSS selector to wait for")
    timeout_ms: int = Field(default=5000, description="Timeout in milliseconds")
    state: Literal["attached", "detached", "visible", "hidden"] = Field(
        default="visible", description="State to wait for"
    )


class ScrollRequest(BaseModel):
    """Request to scroll the page."""

    page_id: str = Field(..., description="ID of the page to scroll")
    direction: Literal["up", "down", "top", "bottom"] = Field(default="down", description="Scroll direction")
    amount: int = Field(default=500, description="Pixels to scroll (for up/down)")


class ScreenshotRequest(BaseModel):
    """Request to take a screenshot."""

    page_id: str = Field(..., description="ID of the page to screenshot")
    full_page: bool = Field(default=False, description="Whether to capture full scrollable page")


class GetPageRequest(BaseModel):
    """Request to get current page state."""

    page_id: str = Field(..., description="ID of the page to retrieve")
    include_html: bool = Field(default=True, description="Whether to include full HTML")
    include_text: bool = Field(default=True, description="Whether to include extracted text")


class ClosePageRequest(BaseModel):
    """Request to close a page."""

    page_id: str = Field(..., description="ID of the page to close")


class SearchRequest(BaseModel):
    """Request to perform a web search."""

    query: str = Field(..., description="Search query string")
    engine: Literal["ddg", "bing", "google"] = Field(
        default="ddg",
        description="Search engine to use (ddg=DuckDuckGo, bing=Bing, google=Google)"
    )


# ============================================================================
# Browser Plan Models (for Phase 4)
# ============================================================================


class BrowserAction(BaseModel):
    """A single action in a browser plan."""

    action: Literal["open", "click", "type", "wait_for", "scroll", "screenshot", "extract_text"]
    params: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")


class BrowserPlan(BaseModel):
    """A complete plan for executing a browsing task."""

    goal: str = Field(..., description="High-level description of the browsing goal")
    max_steps: int = Field(default=20, description="Maximum number of steps to execute")
    steps: List[BrowserAction] = Field(..., description="Ordered list of actions to execute")
    allowed_domains: Optional[List[str]] = Field(default=None, description="Domains allowed for this plan")


class TaskLog(BaseModel):
    """Log entry for a browser task."""

    task_id: str = Field(..., description="Unique task identifier")
    goal: str = Field(..., description="Task goal")
    plan: Optional[BrowserPlan] = Field(default=None, description="Executed plan")
    start_time: datetime = Field(..., description="When task started")
    end_time: Optional[datetime] = Field(default=None, description="When task ended")
    status: Literal["running", "completed", "failed", "timeout"] = Field(..., description="Task status")
    steps_executed: int = Field(default=0, description="Number of steps executed")
    error: Optional[BrowserError] = Field(default=None, description="Error if failed")
    result: Optional[PageSnapshot] = Field(default=None, description="Final result snapshot")
