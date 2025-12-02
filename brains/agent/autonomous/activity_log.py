"""
Activity Log
===========

This module provides a basic activity log for agent actions.  It
records the type of action, details and some contextual information.
The log can be queried for recent activity.  Future implementations
may persist logs to disk or stream them to monitoring systems.
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional


class ActivityLog:
    """Record and retrieve recent agent activities."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log_action(self, action_type: str, details: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Log an agent action with optional context."""
        entry = {
            "timestamp": time.time(),
            "action": action_type,
            "details": details,
            "context": context or {},
        }
        self.entries.append(entry)

    def get_recent_activity(self, hours: float = 1.0) -> List[Dict[str, Any]]:
        """Return activities recorded in the last ``hours`` hours."""
        cutoff = time.time() - (hours * 3600.0)
        return [e for e in self.entries if e["timestamp"] > cutoff]