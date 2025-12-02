"""
Feedback Coordinator for Cognitive Brain Learning
=================================================

This module coordinates the feedback loop between SELF_REVIEW and cognitive brains.

After SELF_REVIEW provides a verdict on an interaction, this coordinator:
1. Identifies which brains participated in the interaction
2. Calls UPDATE_FROM_VERDICT on each brain
3. Extracts brain-specific metadata from the review
4. Logs the learning updates

This creates the closed loop that enables brains to learn from experience.

Usage:
    from brains.cognitive.feedback_coordinator import distribute_feedback

    # After SELF_REVIEW provides verdict
    review_result = {
        "verdict": "ok",  # or "minor_issue", "major_issue"
        "issues": [...],
        "metadata_tags": [...],
        # ... other SELF_REVIEW fields
    }

    interaction_context = {
        "used_brains": ["integrator", "affect_priority", "research_manager"],
        "question": "...",
        "research_timed_out": False,
        # ... other context
    }

    distribute_feedback(review_result, interaction_context)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


def _extract_brain_metadata(
    brain_name: str,
    review_result: Dict[str, Any],
    interaction_context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract brain-specific metadata from review result and interaction context.

    Different brains care about different aspects of feedback:
    - RESEARCH_MANAGER: timeout, budget_exceeded, underkill, overkill
    - CONTEXT_MANAGEMENT: context_loss, too_much_context
    - AFFECT_PRIORITY: (uses verdict only)
    - INTEGRATOR: (uses verdict only)

    Args:
        brain_name: Name of the brain
        review_result: SELF_REVIEW result with verdict and issues
        interaction_context: Context from the interaction

    Returns:
        Brain-specific metadata dict, or None if no metadata needed
    """
    metadata = {}

    if brain_name == "research_manager":
        # Check for research-specific issues
        metadata["timeout"] = interaction_context.get("research_timed_out", False)
        metadata["budget_exceeded"] = interaction_context.get("budget_exceeded", False)

        # Check for depth issues from review
        issues = review_result.get("issues", [])
        for issue in issues:
            # Handle both dict issues (with "code"/"message" keys) and plain string issues
            if isinstance(issue, dict):
                code = issue.get("code", "")
                msg = issue.get("message", "").lower()
            elif isinstance(issue, str):
                code = ""
                msg = issue.lower()
            else:
                continue

            if code == "INCOMPLETE" or "not enough detail" in msg:
                metadata["underkill"] = True
            if "too verbose" in msg:
                metadata["overkill"] = True

    elif brain_name == "context_management":
        # Check for context-related issues
        issues = review_result.get("issues", [])
        for issue in issues:
            # Handle both dict issues (with "message" key) and plain string issues
            if isinstance(issue, dict):
                msg = issue.get("message", "").lower()
            elif isinstance(issue, str):
                msg = issue.lower()
            else:
                continue

            if "incoherent" in msg or "contradictory" in msg or "context" in msg:
                metadata["context_loss"] = True
            if "repetitive" in msg or "too long" in msg:
                metadata["too_much_context"] = True

    return metadata if metadata else None


def distribute_feedback(
    review_result: Dict[str, Any],
    interaction_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Distribute SELF_REVIEW feedback to all participating brains.

    Args:
        review_result: Result from SELF_REVIEW with verdict and issues
        interaction_context: Context with used_brains and other metadata

    Returns:
        Summary dict with update counts and any errors
    """
    verdict = review_result.get("verdict", "ok")
    used_brains = interaction_context.get("used_brains", [])

    print(f"[FEEDBACK] Distributing verdict='{verdict}' to {len(used_brains)} brains")

    updates_sent = 0
    updates_succeeded = 0
    errors = []

    # Map of brain names to module imports
    brain_services = {
        "integrator": "brains.cognitive.integrator.service.integrator_brain",
        "affect_priority": "brains.cognitive.affect_priority.service.affect_priority_brain",
        "research_manager": "brains.cognitive.research_manager.service.research_manager_brain",
        "context_management": "brains.cognitive.context_management.service.context_manager",
    }

    for brain_name in used_brains:
        # Only update brains that have learning capability
        if brain_name not in brain_services:
            continue

        try:
            # Import the brain module
            import importlib
            module_path = brain_services[brain_name]
            brain_module = importlib.import_module(module_path)

            # Check if brain has update_from_verdict function
            if not hasattr(brain_module, "update_from_verdict"):
                print(f"[FEEDBACK] Brain '{brain_name}' doesn't have update_from_verdict, skipping")
                continue

            # Extract brain-specific metadata
            metadata = _extract_brain_metadata(brain_name, review_result, interaction_context)

            # Call update_from_verdict
            brain_module.update_from_verdict(verdict, metadata)

            updates_sent += 1
            updates_succeeded += 1

            print(f"[FEEDBACK] ✓ Updated {brain_name}")

        except Exception as e:
            error_msg = f"Failed to update {brain_name}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"[FEEDBACK] ✗ {error_msg}")

    summary = {
        "verdict": verdict,
        "brains_notified": len(used_brains),
        "updates_sent": updates_sent,
        "updates_succeeded": updates_succeeded,
        "errors": errors
    }

    print(f"[FEEDBACK] Summary: {updates_succeeded}/{updates_sent} updates succeeded")

    return summary


def integrate_feedback_after_review(review_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper that calls distribute_feedback after SELF_REVIEW.

    This can be called directly after getting a review result to trigger learning.

    Args:
        review_result: SELF_REVIEW result
        context: Interaction context with used_brains

    Returns:
        Combined result with original review_result + feedback_summary
    """
    feedback_summary = distribute_feedback(review_result, context)

    return {
        **review_result,
        "feedback_summary": feedback_summary
    }
