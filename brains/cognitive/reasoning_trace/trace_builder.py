"""
Reasoning Trace Builder
Phase 8 - Build deterministic execution traces with step validation.

This module provides:
1. Step object representation
2. Trace entry validation
3. Determinism marker validation
4. Final trace dictionary production
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime


class Step:
    """
    Represents a single execution step with deterministic properties.

    Each step has:
    - Unique ID (deterministic counter)
    - Type (coding, planning, creative, governance, etc.)
    - Description
    - Input data
    - Tags for routing
    - Determinism markers
    """

    def __init__(
        self,
        step_id: int,
        step_type: str,
        description: str,
        tags: Optional[List[str]] = None,
        input_data: Optional[Any] = None
    ):
        self.step_id = step_id
        self.step_type = step_type
        self.description = description
        self.tags = tags or []
        self.input_data = input_data
        self.output_data: Optional[Any] = None
        self.brain_used: Optional[str] = None
        self.patterns_used: List[str] = []
        self.success: bool = False
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            "step_id": self.step_id,
            "type": self.step_type,
            "description": self.description,
            "tags": sorted(self.tags),  # Deterministic ordering
            "input": self.input_data
        }

    def mark_complete(
        self,
        output: Any,
        brain: str,
        patterns: Optional[List[str]] = None
    ) -> None:
        """Mark step as successfully completed."""
        self.output_data = output
        self.brain_used = brain
        self.patterns_used = patterns or []
        self.success = True

    def mark_failed(self, error_message: str) -> None:
        """Mark step as failed."""
        self.success = False
        self.error = error_message

    def to_trace_entry(self) -> Dict[str, Any]:
        """Convert to trace entry format."""
        entry = {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "description": self.description,
            "tags": sorted(self.tags),
            "input": self.input_data,
            "output": self.output_data,
            "brain": self.brain_used,
            "patterns_used": sorted(self.patterns_used) if self.patterns_used else [],
            "success": self.success
        }

        if self.error:
            entry["error"] = self.error

        return entry


class TraceBuilder:
    """
    Builds and validates deterministic execution traces.

    Responsibilities:
    - Append step entries
    - Validate determinism markers
    - Ensure step ID sequence integrity
    - Produce final trace dictionary
    """

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self._expected_next_step: int = 1
        self._deterministic: bool = True
        self._validation_errors: List[str] = []

    def append_step(self, step: Step) -> None:
        """
        Append a step to the trace.

        Validates:
        - Step ID sequence (must increment by 1)
        - Determinism markers present
        """
        # Validate step ID sequence
        if step.step_id != self._expected_next_step and step.step_id != 0:
            self._validation_errors.append(
                f"Step ID sequence error: expected {self._expected_next_step}, got {step.step_id}"
            )
            self._deterministic = False

        # Add entry
        self._entries.append(step.to_trace_entry())

        # Update expected next step
        if step.step_id > 0:
            self._expected_next_step = step.step_id + 1

    def append_entry(self, entry: Dict[str, Any]) -> None:
        """
        Append a raw dictionary entry to the trace.

        Performs validation on the entry.
        """
        # Validate required fields
        required_fields = ["step_id", "step_type", "description", "success"]
        missing_fields = [f for f in required_fields if f not in entry]

        if missing_fields:
            self._validation_errors.append(
                f"Missing required fields in trace entry: {missing_fields}"
            )
            self._deterministic = False

        # Validate step ID
        step_id = entry.get("step_id", -1)
        if step_id != self._expected_next_step and step_id != 0:
            self._validation_errors.append(
                f"Step ID sequence error: expected {self._expected_next_step}, got {step_id}"
            )
            self._deterministic = False

        # Add entry
        self._entries.append(entry)

        # Update expected next step
        if step_id > 0:
            self._expected_next_step = step_id + 1

    def validate_determinism(self) -> bool:
        """
        Validate that the trace is deterministic.

        Checks:
        - No randomness indicators (random, time, uuid)
        - All step IDs sequential
        - All patterns are deterministic
        """
        # Check for non-deterministic patterns in entries
        non_deterministic_keywords = ["random", "uuid", "time.time", "datetime.now"]

        for entry in self._entries:
            entry_str = str(entry).lower()
            for keyword in non_deterministic_keywords:
                if keyword in entry_str:
                    self._validation_errors.append(
                        f"Non-deterministic keyword '{keyword}' found in trace entry"
                    )
                    self._deterministic = False

        return self._deterministic

    def produce_trace(self) -> Dict[str, Any]:
        """
        Produce final trace dictionary.

        Returns:
            Dict with:
                - entries: List of all trace entries
                - total_steps: Total number of steps
                - deterministic: Whether trace is deterministic
                - validation_errors: List of validation errors (if any)
        """
        trace = {
            "entries": self._entries.copy(),
            "total_steps": len([e for e in self._entries if e.get("step_id", 0) > 0]),
            "deterministic": self._deterministic
        }

        if self._validation_errors:
            trace["validation_errors"] = self._validation_errors.copy()

        return trace

    def reset(self) -> None:
        """Reset trace builder to initial state."""
        self._entries.clear()
        self._expected_next_step = 1
        self._deterministic = True
        self._validation_errors.clear()

    def get_entries(self) -> List[Dict[str, Any]]:
        """Get current trace entries."""
        return self._entries.copy()

    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()


def create_step(
    step_id: int,
    step_type: str,
    description: str,
    tags: Optional[List[str]] = None,
    input_data: Optional[Any] = None
) -> Step:
    """Factory function to create a Step object."""
    return Step(
        step_id=step_id,
        step_type=step_type,
        description=description,
        tags=tags,
        input_data=input_data
    )


def validate_trace_determinism(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a trace dictionary is deterministic.

    Args:
        trace: Trace dictionary to validate

    Returns:
        Dict with:
            - valid: bool
            - errors: List[str] (if any)
    """
    errors = []

    # Check required fields
    if "entries" not in trace:
        errors.append("Missing 'entries' field in trace")
        return {"valid": False, "errors": errors}

    if "deterministic" not in trace:
        errors.append("Missing 'deterministic' field in trace")

    if not trace.get("deterministic", False):
        errors.append("Trace is marked as non-deterministic")

    # Validate step ID sequence
    entries = trace.get("entries", [])
    expected_id = 1

    for i, entry in enumerate(entries):
        step_id = entry.get("step_id")

        if step_id is None:
            errors.append(f"Entry {i} missing step_id")
            continue

        # Skip step_id 0 (decomposition/meta steps)
        if step_id == 0:
            continue

        if step_id != expected_id:
            errors.append(f"Step ID sequence broken: expected {expected_id}, got {step_id}")

        expected_id = step_id + 1

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }
