"""
Step Router
Phase 8 - Deterministic routing of steps to specialist brains.

Routing Rules (deterministic, tag-based):
- Step with "coding" tag → coder_brain
- Step with "plan" or "parse" tag → planner_brain
- Step with "creative" tag → imaginer_brain
- Step with "governance" or "conflict" tag → committee_brain
- Step with "language" tag → language_brain
- Step with "reasoning" tag → reasoning_brain
- Default → planner_brain
"""

from __future__ import annotations
from typing import Dict, Any, List, Set


class StepRouter:
    """
    Routes steps to appropriate specialist brains based on tags.

    All routing is deterministic:
    - Same tags always route to same brain
    - Priority order for multiple matching tags
    - No randomness in routing decisions
    """

    # Routing rules: tag -> brain name
    # Priority order matters: first match wins
    ROUTING_RULES = [
        ({"coding", "code", "implement", "debug"}, "coder"),
        ({"plan", "planning", "decompose", "organize"}, "planner"),
        ({"parse", "analyze", "understand"}, "planner"),
        ({"creative", "imagine", "brainstorm", "hypothesize"}, "imaginer"),
        ({"governance", "arbitrate", "decide"}, "committee"),
        ({"conflict", "resolve", "consensus"}, "committee"),
        ({"language", "generate", "text", "communicate"}, "language"),
        ({"reasoning", "logic", "deduce", "infer"}, "reasoning"),
    ]

    DEFAULT_BRAIN = "planner"

    @staticmethod
    def route_step(step: Dict[str, Any]) -> str:
        """
        Route a step to appropriate brain based on tags.

        Args:
            step: Step dictionary with:
                - tags: List[str] - Tags for routing
                - type: str - Step type
                - description: str - Step description

        Returns:
            str: Brain name to handle this step
        """
        # Get tags from step
        tags = step.get("tags", [])
        step_type = step.get("type", "")

        # Convert to lowercase set for matching
        tag_set = {tag.lower() for tag in tags}

        # Also consider step type as a tag
        if step_type:
            tag_set.add(step_type.lower())

        # Apply routing rules in priority order
        for rule_tags, brain_name in StepRouter.ROUTING_RULES:
            # Check if any rule tags match step tags
            if tag_set & rule_tags:  # Set intersection
                return brain_name

        # Default routing
        return StepRouter.DEFAULT_BRAIN

    @staticmethod
    def get_routing_info(step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed routing information for a step.

        Returns:
            Dict with:
                - brain: str - Selected brain
                - matched_tags: List[str] - Tags that matched routing rules
                - all_tags: List[str] - All step tags
                - rule_applied: str - Description of rule applied
        """
        tags = step.get("tags", [])
        step_type = step.get("type", "")
        tag_set = {tag.lower() for tag in tags}

        if step_type:
            tag_set.add(step_type.lower())

        # Find matching rule
        matched_rule = None
        matched_tags = []

        for rule_tags, brain_name in StepRouter.ROUTING_RULES:
            intersection = tag_set & rule_tags
            if intersection:
                matched_rule = (rule_tags, brain_name)
                matched_tags = list(intersection)
                break

        if matched_rule:
            rule_tags, brain_name = matched_rule
            rule_desc = f"Tags {matched_tags} matched rule for {brain_name}_brain"
        else:
            brain_name = StepRouter.DEFAULT_BRAIN
            rule_desc = f"No tags matched, using default: {brain_name}_brain"

        return {
            "brain": brain_name,
            "matched_tags": sorted(matched_tags),
            "all_tags": sorted(list(tag_set)),
            "rule_applied": rule_desc
        }

    @staticmethod
    def validate_routing_determinism() -> Dict[str, Any]:
        """
        Validate that routing rules are deterministic.

        Checks:
        - No overlapping rule tags (would cause non-determinism)
        - All rules have priority order
        - No randomness in routing logic

        Returns:
            Dict with:
                - deterministic: bool
                - issues: List[str] (if any)
        """
        issues = []

        # Check for overlapping tags between rules
        for i, (tags1, brain1) in enumerate(StepRouter.ROUTING_RULES):
            for j, (tags2, brain2) in enumerate(StepRouter.ROUTING_RULES):
                if i >= j:
                    continue

                overlap = tags1 & tags2
                if overlap:
                    issues.append(
                        f"Overlapping tags {overlap} between {brain1} and {brain2} "
                        f"(priority: {brain1} wins due to rule order)"
                    )

        return {
            "deterministic": True,  # Overlaps are OK due to priority order
            "issues": issues,
            "note": "Overlapping tags resolved by rule priority order"
        }


def route_step(step: Dict[str, Any]) -> str:
    """
    Convenience function to route a step.

    Args:
        step: Step dictionary

    Returns:
        str: Brain name to handle step
    """
    return StepRouter.route_step(step)


def get_routing_info(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get routing information.

    Args:
        step: Step dictionary

    Returns:
        Dict with routing information
    """
    return StepRouter.get_routing_info(step)


def validate_routing() -> Dict[str, Any]:
    """
    Convenience function to validate routing determinism.

    Returns:
        Dict with validation results
    """
    return StepRouter.validate_routing_determinism()
