"""
Browser Brain
=============

Cognitive module for web browsing capabilities.
Integrates browser automation into Maven's cognitive stack.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from optional.maven_browser_client.types import (
    BrowserPlan,
    BrowserAction,
    ActionType,
    BrowserTaskResult,
    TaskStatus,
)
from optional.browser_tools.intent_resolver import IntentResolver, resolve_intent
from optional.browser_tools.browser_tool import execute_browser_plan, run_browser_task
from optional.browser_tools.plan_validator import validate_plan, ValidationError
from optional.browser_tools.pattern_store import get_pattern_store
from optional.browser_tools.reflection import reflect_on_task, TaskReflection


class BrowserBrain:
    """
    Maven cognitive module for web browsing.

    Provides high-level interface for browser automation,
    including intent resolution, plan execution, and learning.
    """

    def __init__(self):
        self.intent_resolver = IntentResolver()
        self.pattern_store = get_pattern_store()
        self._last_result: Optional[BrowserTaskResult] = None
        self._last_reflection: Optional[TaskReflection] = None

    def handle_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a browser operation request.

        Args:
            msg: Message with operation and payload

        Returns:
            Response dictionary
        """
        op = msg.get("op", "").upper()

        handlers = {
            "BROWSE": self._handle_browse,
            "SEARCH": self._handle_search,
            "OPEN_URL": self._handle_open_url,
            "EXECUTE_PLAN": self._handle_execute_plan,
            "VALIDATE_PLAN": self._handle_validate_plan,
            "RESOLVE_INTENT": self._handle_resolve_intent,
            "GET_PATTERNS": self._handle_get_patterns,
            "GET_LAST_RESULT": self._handle_get_last_result,
        }

        handler = handlers.get(op)
        if not handler:
            return {
                "ok": False,
                "error": f"Unknown operation: {op}",
                "valid_ops": list(handlers.keys()),
            }

        try:
            return handler(msg)
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _handle_browse(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a natural language browsing request.

        Resolves intent to plan, validates, executes, and reflects.
        """
        payload = msg.get("payload", {})
        goal = payload.get("goal") or payload.get("query") or payload.get("intent")

        if not goal:
            return {"ok": False, "error": "No goal/query/intent provided"}

        context = payload.get("context", {})

        # Resolve intent to plan
        plan = self.intent_resolver.resolve(goal, context)

        # Track which pattern was used
        pattern_used = None
        matched_pattern = self.pattern_store.find_pattern(goal)
        if matched_pattern:
            pattern_used = matched_pattern.name

        # Validate plan
        try:
            validate_plan(plan)
        except ValidationError as e:
            return {
                "ok": False,
                "error": f"Plan validation failed: {str(e)}",
                "plan": plan.model_dump(),
            }

        # Execute plan
        result = run_browser_task(goal, plan.model_dump())
        self._last_result = BrowserTaskResult(**result)

        # Reflect on execution
        reflection = reflect_on_task(
            plan,
            self._last_result,
            pattern_used=pattern_used,
        )
        self._last_reflection = reflection

        return {
            "ok": True,
            "result": result,
            "reflection": reflection.to_dict(),
            "pattern_used": pattern_used,
        }

    def _handle_search(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a search request (convenience for web search)."""
        payload = msg.get("payload", {})
        query = payload.get("query")

        if not query:
            return {"ok": False, "error": "No query provided"}

        # Use Google search pattern
        from brains.agent.tools.browser.browser_tool import simple_google_search

        result = simple_google_search(query)
        self._last_result = BrowserTaskResult(**result)

        return {"ok": True, "result": result}

    def _handle_open_url(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an open URL request."""
        payload = msg.get("payload", {})
        url = payload.get("url")

        if not url:
            return {"ok": False, "error": "No URL provided"}

        from brains.agent.tools.browser.browser_tool import open_url

        result = open_url(url)
        self._last_result = BrowserTaskResult(**result)

        return {"ok": True, "result": result}

    def _handle_execute_plan(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct plan execution."""
        payload = msg.get("payload", {})
        plan_dict = payload.get("plan")

        if not plan_dict:
            return {"ok": False, "error": "No plan provided"}

        try:
            plan = BrowserPlan(**plan_dict)
        except Exception as e:
            return {"ok": False, "error": f"Invalid plan format: {str(e)}"}

        # Validate
        try:
            validate_plan(plan)
        except ValidationError as e:
            return {"ok": False, "error": f"Plan validation failed: {str(e)}"}

        # Execute
        result = run_browser_task(plan.goal, plan_dict)
        self._last_result = BrowserTaskResult(**result)

        return {"ok": True, "result": result}

    def _handle_validate_plan(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a plan without executing."""
        payload = msg.get("payload", {})
        plan_dict = payload.get("plan")

        if not plan_dict:
            return {"ok": False, "error": "No plan provided"}

        try:
            plan = BrowserPlan(**plan_dict)
            validate_plan(plan)
            return {"ok": True, "valid": True, "plan": plan.model_dump()}
        except ValidationError as e:
            return {"ok": True, "valid": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"Invalid plan: {str(e)}"}

    def _handle_resolve_intent(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve an intent to a plan without executing."""
        payload = msg.get("payload", {})
        intent = payload.get("intent") or payload.get("goal")

        if not intent:
            return {"ok": False, "error": "No intent provided"}

        context = payload.get("context", {})
        plan = self.intent_resolver.resolve(intent, context)

        return {
            "ok": True,
            "plan": plan.model_dump(),
            "pattern_matched": self.pattern_store.find_pattern(intent) is not None,
        }

    def _handle_get_patterns(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Get available browsing patterns."""
        patterns = self.pattern_store.list_patterns()

        return {
            "ok": True,
            "patterns": [
                {
                    "name": p.name,
                    "description": p.description,
                    "keywords": p.trigger_keywords,
                    "domains": p.domains,
                    "success_count": p.success_count,
                    "failure_count": p.failure_count,
                }
                for p in patterns
            ],
        }

    def _handle_get_last_result(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Get the last browser task result."""
        if not self._last_result:
            return {"ok": False, "error": "No previous result available"}

        return {
            "ok": True,
            "result": self._last_result.model_dump(),
            "reflection": self._last_reflection.to_dict() if self._last_reflection else None,
        }


# Global brain instance
_browser_brain: Optional[BrowserBrain] = None


def get_browser_brain() -> BrowserBrain:
    """Get the global browser brain instance."""
    global _browser_brain
    if _browser_brain is None:
        _browser_brain = BrowserBrain()
    return _browser_brain


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Browser brain service API entry point.

    This is the standard Maven brain interface.

    Operations:
        BROWSE: Execute a natural language browsing task
        SEARCH: Perform a web search
        OPEN_URL: Open a specific URL
        EXECUTE_PLAN: Execute a browser plan directly
        VALIDATE_PLAN: Validate a plan without executing
        RESOLVE_INTENT: Convert intent to plan without executing
        GET_PATTERNS: List available browsing patterns
        GET_LAST_RESULT: Get the last task result

    Args:
        msg: Message with "op" and "payload" keys

    Returns:
        Response dictionary with "ok" and result/error
    """
    brain = get_browser_brain()
    return brain.handle_request(msg)


# Convenience functions for direct use

def browse(goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a browsing task from natural language goal.

    Args:
        goal: Natural language description of browsing goal
        context: Optional context information

    Returns:
        Task result dictionary
    """
    return service_api({
        "op": "BROWSE",
        "payload": {"goal": goal, "context": context or {}},
    })


def search(query: str) -> Dict[str, Any]:
    """
    Perform a web search.

    Args:
        query: Search query

    Returns:
        Task result dictionary
    """
    return service_api({
        "op": "SEARCH",
        "payload": {"query": query},
    })


def open_webpage(url: str) -> Dict[str, Any]:
    """
    Open a webpage.

    Args:
        url: URL to open

    Returns:
        Task result dictionary
    """
    return service_api({
        "op": "OPEN_URL",
        "payload": {"url": url},
    })
