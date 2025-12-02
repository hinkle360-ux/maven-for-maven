"""
Maven Browser Client Types
==========================

Type definitions and schemas for browser plans and task execution.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Available browser action types."""

    OPEN = "open"
    CLICK = "click"
    TYPE = "type"
    WAIT_FOR = "wait_for"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    EXTRACT_TEXT = "extract_text"


class BrowserAction(BaseModel):
    """A single action in a browser plan."""

    action: ActionType = Field(..., description="Type of action to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")


class BrowserPlan(BaseModel):
    """A complete plan for executing a browsing task."""

    goal: str = Field(..., description="High-level description of the browsing goal")
    max_steps: int = Field(default=20, description="Maximum number of steps to execute")
    steps: List[BrowserAction] = Field(..., description="Ordered list of actions to execute")
    allowed_domains: Optional[List[str]] = Field(
        default=None,
        description="Domains allowed for this plan (None = use global config)"
    )
    timeout_seconds: int = Field(default=120, description="Overall timeout for plan execution")


class TaskStatus(str, Enum):
    """Status of a browser task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class BrowserTaskResult(BaseModel):
    """Result of executing a browser task."""

    task_id: str = Field(..., description="Unique task identifier")
    goal: str = Field(..., description="Task goal")
    status: TaskStatus = Field(..., description="Task status")
    steps_executed: int = Field(default=0, description="Number of steps executed")
    duration_seconds: float = Field(default=0.0, description="Task duration in seconds")
    final_url: Optional[str] = Field(default=None, description="Final URL after task completion")
    final_text: Optional[str] = Field(default=None, description="Final page text content")
    screenshot_path: Optional[str] = Field(default=None, description="Path to final screenshot if captured")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PatternMatch(BaseModel):
    """A learned pattern for browser interactions."""

    name: str = Field(..., description="Pattern name (e.g., 'google_search', 'amazon_product_search')")
    description: str = Field(..., description="What this pattern does")
    trigger_keywords: List[str] = Field(..., description="Keywords that suggest this pattern")
    domains: List[str] = Field(..., description="Domains this pattern applies to")
    template_plan: BrowserPlan = Field(..., description="Template plan for this pattern")
    success_count: int = Field(default=0, description="Number of times this pattern succeeded")
    failure_count: int = Field(default=0, description="Number of times this pattern failed")
    last_used: Optional[datetime] = Field(default=None, description="When this pattern was last used")


# Pre-defined pattern templates
GOOGLE_SEARCH_PATTERN = BrowserPlan(
    goal="Search Google for a query",
    max_steps=5,
    steps=[
        BrowserAction(action=ActionType.OPEN, params={"url": "https://www.google.com"}),
        BrowserAction(
            action=ActionType.TYPE,
            params={"selector": "textarea[name=q]", "text": "{query}", "submit": True}
        ),
        BrowserAction(
            action=ActionType.WAIT_FOR,
            params={"selector": "#search", "timeout_ms": 10000}
        ),
    ],
    allowed_domains=["google.com", "www.google.com"]
)


OPEN_URL_PATTERN = BrowserPlan(
    goal="Open a specific URL",
    max_steps=1,
    steps=[
        BrowserAction(action=ActionType.OPEN, params={"url": "{url}"}),
    ]
)
