"""
Browser Plan Validator
======================

Validates browser plans against safety rules and constraints.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from urllib.parse import urlparse

from optional.maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
from optional.browser_runtime.config import get_config


class ValidationError(Exception):
    """Raised when a browser plan fails validation."""

    pass


class PlanValidator:
    """Validates browser plans for safety and correctness."""

    def __init__(self):
        self.config = get_config()

    def validate(self, plan: BrowserPlan) -> Tuple[bool, Optional[str]]:
        """
        Validate a browser plan.

        Args:
            plan: BrowserPlan to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self._validate_max_steps(plan)
            self._validate_domains(plan)
            self._validate_actions(plan)
            return True, None
        except ValidationError as e:
            return False, str(e)

    def _validate_max_steps(self, plan: BrowserPlan) -> None:
        """Validate max steps constraint."""
        if plan.max_steps > self.config.max_steps_per_task:
            raise ValidationError(
                f"Plan max_steps ({plan.max_steps}) exceeds limit ({self.config.max_steps_per_task})"
            )

        if len(plan.steps) > plan.max_steps:
            raise ValidationError(
                f"Plan has {len(plan.steps)} steps but max_steps is {plan.max_steps}"
            )

    def _validate_domains(self, plan: BrowserPlan) -> None:
        """Validate that plan only accesses allowed domains."""
        # Extract all domains from OPEN actions
        domains = []
        for action in plan.steps:
            if action.action == ActionType.OPEN:
                url = action.params.get("url", "")
                if url:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        domains.append(parsed.netloc)

        # Check each domain
        for domain in domains:
            # Use plan-specific allowed domains if provided, otherwise use config
            if plan.allowed_domains is not None:
                if domain not in plan.allowed_domains:
                    raise ValidationError(
                        f"Domain '{domain}' not in plan's allowed domains: {plan.allowed_domains}"
                    )
            else:
                if not self.config.is_domain_allowed(domain):
                    raise ValidationError(f"Domain '{domain}' is not allowed by global configuration")

    def _validate_actions(self, plan: BrowserPlan) -> None:
        """Validate action sequence and parameters."""
        has_open = False

        for i, action in enumerate(plan.steps):
            # First action should typically be OPEN
            if i == 0 and action.action != ActionType.OPEN:
                # Warning: not an error, but unusual
                pass

            # Track if we've opened a page
            if action.action == ActionType.OPEN:
                has_open = True

            # Actions that need a page should come after OPEN
            if action.action in [ActionType.CLICK, ActionType.TYPE, ActionType.WAIT_FOR, ActionType.SCROLL]:
                if not has_open and i > 0:
                    raise ValidationError(
                        f"Action {action.action} at step {i} requires an OPEN action first"
                    )

            # Validate required parameters
            self._validate_action_params(action, i)

    def _validate_action_params(self, action: BrowserAction, step_num: int) -> None:
        """Validate parameters for a specific action."""
        params = action.params

        if action.action == ActionType.OPEN:
            if "url" not in params:
                raise ValidationError(f"OPEN action at step {step_num} missing 'url' parameter")

        elif action.action == ActionType.CLICK:
            if "selector" not in params and "text" not in params:
                raise ValidationError(
                    f"CLICK action at step {step_num} must have 'selector' or 'text' parameter"
                )

        elif action.action == ActionType.TYPE:
            if "selector" not in params:
                raise ValidationError(f"TYPE action at step {step_num} missing 'selector' parameter")
            if "text" not in params:
                raise ValidationError(f"TYPE action at step {step_num} missing 'text' parameter")

        elif action.action == ActionType.WAIT_FOR:
            if "selector" not in params:
                raise ValidationError(f"WAIT_FOR action at step {step_num} missing 'selector' parameter")


def validate_plan(plan: BrowserPlan) -> None:
    """
    Validate a browser plan and raise exception if invalid.

    Args:
        plan: BrowserPlan to validate

    Raises:
        ValidationError: If plan is invalid
    """
    validator = PlanValidator()
    is_valid, error = validator.validate(plan)
    if not is_valid:
        raise ValidationError(error)
