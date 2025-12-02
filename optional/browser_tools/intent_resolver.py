"""
Browser Intent Resolver
=======================

Resolves natural language intents into browser plans.
Uses pattern matching and LLM-based plan generation.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

from optional.maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
from optional.browser_tools.pattern_store import get_pattern_store


class IntentResolver:
    """Resolves natural language intents into browser plans."""

    def __init__(self):
        self.pattern_store = get_pattern_store()

    def resolve(self, intent: str, context: Optional[Dict[str, Any]] = None) -> BrowserPlan:
        """
        Resolve an intent into a browser plan.

        Args:
            intent: Natural language intent/goal
            context: Optional context information

        Returns:
            BrowserPlan ready for execution
        """
        # Try to find a matching pattern
        pattern = self.pattern_store.find_pattern(intent)

        if pattern:
            # Use pattern template
            plan = self._instantiate_pattern(pattern, intent, context)
            return plan
        else:
            # Generate a custom plan
            # For now, return a simple plan
            # In a full implementation, this would call an LLM to generate the plan
            return self._generate_simple_plan(intent, context)

    def _instantiate_pattern(
        self,
        pattern: Any,
        intent: str,
        context: Optional[Dict[str, Any]]
    ) -> BrowserPlan:
        """
        Instantiate a pattern template with specific values.

        Args:
            pattern: PatternMatch with template
            intent: Natural language intent
            context: Optional context

        Returns:
            Instantiated BrowserPlan
        """
        # Copy the template
        plan = pattern.template_plan.model_copy(deep=True)
        plan.goal = intent

        # Replace placeholders
        if pattern.name == "google_search":
            # Extract query from intent
            query = self._extract_search_query(intent)
            for step in plan.steps:
                if "text" in step.params and "{query}" in step.params["text"]:
                    step.params["text"] = step.params["text"].replace("{query}", query)

        elif pattern.name == "open_url":
            # Extract URL from intent
            url = self._extract_url(intent, context)
            for step in plan.steps:
                if "url" in step.params and "{url}" in step.params["url"]:
                    step.params["url"] = url

        return plan

    def _extract_search_query(self, intent: str) -> str:
        """
        Extract search query from intent.

        Args:
            intent: Natural language intent

        Returns:
            Search query
        """
        # Simple extraction - look for common patterns
        intent_lower = intent.lower()

        # Remove common prefixes
        for prefix in ["search for ", "search ", "google ", "find ", "lookup ", "look up "]:
            if intent_lower.startswith(prefix):
                return intent[len(prefix):].strip()

        # If no prefix found, use entire intent
        return intent.strip()

    def _extract_url(self, intent: str, context: Optional[Dict[str, Any]]) -> str:
        """
        Extract URL from intent or context.

        Args:
            intent: Natural language intent
            context: Optional context

        Returns:
            URL
        """
        # Check context first
        if context and "url" in context:
            return context["url"]

        # Look for URL in intent
        import re
        url_pattern = r'https?://[^\s]+'
        matches = re.findall(url_pattern, intent)
        if matches:
            return matches[0]

        # Look for domain-like strings
        domain_pattern = r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
        matches = re.findall(domain_pattern, intent)
        if matches:
            return f"https://{matches[0]}"

        # Default to empty (will fail validation)
        return ""

    def _generate_simple_plan(self, intent: str, context: Optional[Dict[str, Any]]) -> BrowserPlan:
        """
        Generate a simple browser plan.

        This is a fallback when no pattern matches.
        In a full implementation, this would use an LLM.

        Args:
            intent: Natural language intent
            context: Optional context

        Returns:
            Simple BrowserPlan
        """
        # Try to extract URL
        url = self._extract_url(intent, context)

        if url:
            # Simple open URL plan
            return BrowserPlan(
                goal=intent,
                max_steps=1,
                steps=[
                    BrowserAction(action=ActionType.OPEN, params={"url": url})
                ]
            )
        else:
            # Default to Google search
            query = self._extract_search_query(intent)
            return BrowserPlan(
                goal=intent,
                max_steps=5,
                steps=[
                    BrowserAction(action=ActionType.OPEN, params={"url": "https://www.google.com"}),
                    BrowserAction(
                        action=ActionType.TYPE,
                        params={"selector": "textarea[name=q]", "text": query, "submit": True}
                    ),
                    BrowserAction(
                        action=ActionType.WAIT_FOR,
                        params={"selector": "#search", "timeout_ms": 10000}
                    ),
                ],
            )


def resolve_intent(intent: str, context: Optional[Dict[str, Any]] = None) -> BrowserPlan:
    """
    Resolve an intent into a browser plan.

    Args:
        intent: Natural language intent/goal
        context: Optional context information

    Returns:
        BrowserPlan ready for execution
    """
    resolver = IntentResolver()
    return resolver.resolve(intent, context)
