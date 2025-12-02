"""
governance_scope.py
~~~~~~~~~~~~~~~~~~~

Governance authority scope definition for Maven.

P1 GOVERNANCE REQUIREMENT:
This module defines and enforces the EXACT scope of governance authority.
It clearly specifies what governance CAN and CANNOT do, preventing scope creep
and maintaining separation of concerns.

GOVERNANCE CAN:
- Inspect context, outputs, and truth tags
- Block or request re-run of pipeline stages
- Audit truth classification compliance
- Log policy violations
- Request explanation/justification

GOVERNANCE CANNOT:
- Directly modify memory tiers (must go through memory API)
- Override Reasoning's fact/guess classification
- Bypass the pipeline executor
- Modify blackboard context directly (except add audit metadata)

STDLIB ONLY, OFFLINE, WINDOWS 10 COMPATIBLE
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from enum import Enum


class GovernanceAction(Enum):
    """Authorized governance actions."""
    INSPECT = "INSPECT"  # Read context/outputs
    BLOCK = "BLOCK"  # Block output/pipeline continuation
    REQUEST_RERUN = "REQUEST_RERUN"  # Request stage re-execution
    AUDIT_LOG = "AUDIT_LOG"  # Log to audit trail
    REQUEST_EXPLANATION = "REQUEST_EXPLANATION"  # Request justification


class GovernanceViolation(Enum):
    """Types of governance violations."""
    MISSING_TRUTH_TAG = "MISSING_TRUTH_TAG"
    INCOMPATIBLE_MEMORY_WRITE = "INCOMPATIBLE_MEMORY_WRITE"
    UNAUTHORIZED_BYPASS = "UNAUTHORIZED_BYPASS"
    TIER_GOVERNANCE_VIOLATION = "TIER_GOVERNANCE_VIOLATION"
    PIPELINE_VIOLATION = "PIPELINE_VIOLATION"


class GovernanceAuthorityError(Exception):
    """Raised when governance exceeds its authority."""
    pass


class GovernanceScope:
    """
    Defines and enforces governance authority boundaries.

    This class ensures governance operates within its defined scope
    and cannot exceed its authority.
    """

    # Actions governance IS authorized to perform
    AUTHORIZED_ACTIONS = [
        GovernanceAction.INSPECT,
        GovernanceAction.BLOCK,
        GovernanceAction.REQUEST_RERUN,
        GovernanceAction.AUDIT_LOG,
        GovernanceAction.REQUEST_EXPLANATION
    ]

    # Actions governance is NOT authorized to perform
    UNAUTHORIZED_ACTIONS = [
        "MODIFY_MEMORY_DIRECTLY",
        "OVERRIDE_TRUTH_CLASSIFICATION",
        "BYPASS_PIPELINE",
        "MODIFY_BLACKBOARD_DIRECTLY",
        "DELETE_RECORDS",
        "CHANGE_TIER_CONFIGURATION"
    ]

    def __init__(self):
        """Initialize governance scope."""
        self._audit_log: List[Dict[str, Any]] = []
        self._violations: List[Dict[str, Any]] = []

    def can_perform(self, action: str) -> bool:
        """
        Check if governance is authorized to perform an action.

        Args:
            action: Action name or GovernanceAction enum

        Returns:
            True if action is authorized
        """
        if isinstance(action, GovernanceAction):
            return action in self.AUTHORIZED_ACTIONS

        # Check unauthorized actions
        if action in self.UNAUTHORIZED_ACTIONS:
            return False

        # Check if it matches an authorized action name
        for auth_action in self.AUTHORIZED_ACTIONS:
            if action == auth_action.value:
                return True

        return False

    def inspect_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect pipeline context (AUTHORIZED).

        Args:
            context: Pipeline context to inspect

        Returns:
            Inspection result
        """
        self._log_action(GovernanceAction.INSPECT, {"inspected": "context"})

        return {
            "action": GovernanceAction.INSPECT.value,
            "has_truth_tag": "truth_classification" in context,
            "has_memory_write_flag": "allow_memory_write" in context,
            "context_keys": list(context.keys())
        }

    def inspect_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect pipeline output (AUTHORIZED).

        Args:
            output: Pipeline output to inspect

        Returns:
            Inspection result
        """
        self._log_action(GovernanceAction.INSPECT, {"inspected": "output"})

        truth_tag = output.get("truth_classification", {})

        return {
            "action": GovernanceAction.INSPECT.value,
            "has_truth_classification": bool(truth_tag),
            "truth_type": truth_tag.get("type"),
            "confidence": truth_tag.get("confidence"),
            "allows_memory_write": truth_tag.get("allow_memory_write")
        }

    def block_output(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Block pipeline output (AUTHORIZED).

        Args:
            reason: Reason for blocking
            context: Pipeline context

        Returns:
            Block action result
        """
        self._log_action(GovernanceAction.BLOCK, {"reason": reason})

        return {
            "action": GovernanceAction.BLOCK.value,
            "blocked": True,
            "reason": reason,
            "timestamp": self._get_timestamp()
        }

    def request_rerun(
        self,
        stage: str,
        reason: str,
        modified_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Request pipeline stage re-run (AUTHORIZED).

        Args:
            stage: Stage to re-run
            reason: Reason for re-run
            modified_input: Optional modified input (governance can add audit metadata)

        Returns:
            Re-run request result
        """
        self._log_action(
            GovernanceAction.REQUEST_RERUN,
            {"stage": stage, "reason": reason}
        )

        return {
            "action": GovernanceAction.REQUEST_RERUN.value,
            "stage": stage,
            "reason": reason,
            "modified_input": modified_input
        }

    def log_violation(
        self,
        violation_type: GovernanceViolation,
        details: Dict[str, Any]
    ) -> None:
        """
        Log a governance violation (AUTHORIZED).

        Args:
            violation_type: Type of violation
            details: Violation details
        """
        violation_record = {
            "violation_type": violation_type.value,
            "details": details,
            "timestamp": self._get_timestamp()
        }

        self._violations.append(violation_record)
        self._log_action(GovernanceAction.AUDIT_LOG, violation_record)

    def check_truth_tag_compliance(self, output: Dict[str, Any]) -> bool:
        """
        Check if output has required truth tag (AUTHORIZED).

        Args:
            output: Pipeline output

        Returns:
            True if compliant

        Raises:
            Logs violation if non-compliant
        """
        truth_tag = output.get("truth_classification")

        if not truth_tag:
            self.log_violation(
                GovernanceViolation.MISSING_TRUTH_TAG,
                {"output": str(output)[:100]}
            )
            return False

        required_fields = ["type", "confidence", "allow_memory_write"]
        missing = [f for f in required_fields if f not in truth_tag]

        if missing:
            self.log_violation(
                GovernanceViolation.MISSING_TRUTH_TAG,
                {"missing_fields": missing}
            )
            return False

        return True

    def check_memory_write_compliance(
        self,
        truth_tag: Dict[str, Any],
        memory_operation: Dict[str, Any]
    ) -> bool:
        """
        Check if memory write is compatible with truth tag (AUTHORIZED).

        Args:
            truth_tag: Truth classification
            memory_operation: Proposed memory operation

        Returns:
            True if compliant
        """
        if not truth_tag.get("allow_memory_write"):
            if memory_operation.get("action") == "WRITE":
                self.log_violation(
                    GovernanceViolation.INCOMPATIBLE_MEMORY_WRITE,
                    {
                        "truth_type": truth_tag.get("type"),
                        "attempted_write": True
                    }
                )
                return False

        # Check tier compatibility
        truth_tier = truth_tag.get("memory_tier")
        write_tier = memory_operation.get("tier")

        if write_tier and truth_tier and write_tier != truth_tier:
            self.log_violation(
                GovernanceViolation.INCOMPATIBLE_MEMORY_WRITE,
                {
                    "truth_tier": truth_tier,
                    "write_tier": write_tier
                }
            )
            return False

        return True

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get logged violations."""
        return self._violations.copy()

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get full audit log."""
        return self._audit_log.copy()

    def _log_action(self, action: GovernanceAction, details: Dict[str, Any]) -> None:
        """Log a governance action."""
        log_entry = {
            "action": action.value,
            "details": details,
            "timestamp": self._get_timestamp()
        }
        self._audit_log.append(log_entry)

    def _get_timestamp(self) -> float:
        """Get current timestamp (simple counter to avoid datetime)."""
        import time
        return time.time()

    # UNAUTHORIZED METHODS - These will raise errors if called

    def modify_memory_directly(self, *args, **kwargs) -> None:
        """
        UNAUTHORIZED: Governance cannot modify memory directly.

        Raises:
            GovernanceAuthorityError: Always
        """
        raise GovernanceAuthorityError(
            "Governance CANNOT modify memory tiers directly. "
            "Memory writes must go through the memory API (BrainMemory.store())"
        )

    def override_truth_classification(self, *args, **kwargs) -> None:
        """
        UNAUTHORIZED: Governance cannot override Reasoning's classification.

        Raises:
            GovernanceAuthorityError: Always
        """
        raise GovernanceAuthorityError(
            "Governance CANNOT override Reasoning's fact/guess classification. "
            "Only Reasoning brain can classify truth types."
        )

    def bypass_pipeline(self, *args, **kwargs) -> None:
        """
        UNAUTHORIZED: Governance cannot bypass pipeline executor.

        Raises:
            GovernanceAuthorityError: Always
        """
        raise GovernanceAuthorityError(
            "Governance CANNOT bypass the pipeline executor. "
            "All pipeline control must go through the CanonicalPipeline."
        )


def create_governance_scope() -> GovernanceScope:
    """
    Factory function to create a governance scope instance.

    Returns:
        GovernanceScope instance
    """
    return GovernanceScope()


# Export key components
__all__ = [
    "GovernanceAction",
    "GovernanceViolation",
    "GovernanceScope",
    "GovernanceAuthorityError",
    "create_governance_scope"
]
