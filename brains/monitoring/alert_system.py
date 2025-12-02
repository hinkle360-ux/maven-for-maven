"""
alert_system.py
===============

System-wide alerting and notification system for Maven.

This module provides:
1. Alert rule definition and management
2. Alert triggering and routing
3. Alert handlers (console, log, callback)
4. Alert aggregation and deduplication
5. Alert history and tracking
6. Integration with BrainHealthMonitor

Alert Severities:
    - info: Informational notices (pattern learned, upgrade available)
    - warning: Non-critical issues (elevated response time, moderate error rate)
    - critical: Requires immediate attention (brain failure, high error rate)

Alert Categories:
    - health: Brain health issues
    - performance: Performance degradation
    - compliance: Compliance violations
    - learning: Learning and pattern events
    - system: System-level events
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
import json

try:
    from brains.memory.brain_memory import BrainMemory
    _memory = BrainMemory("alert_system")
except Exception:
    _memory = None


# Alert severity levels
SEVERITY_LEVELS = {
    "info": 0,
    "warning": 1,
    "critical": 2
}

# Default alert rules
DEFAULT_ALERT_RULES = {
    "critical": {
        "brain_failure": True,
        "memory_overflow": True,
        "compliance_drop": True,
        "high_error_rate": True,  # >10%
        "extreme_latency": True,  # >2000ms
    },
    "warning": {
        "slow_response": True,  # 500-2000ms
        "elevated_errors": True,  # 2-10%
        "elevated_memory": True,  # 100-500MB
        "coordination_conflict": True,
    },
    "info": {
        "upgrade_available": True,
        "pattern_learned": True,
        "brain_activated": False,  # Disabled by default (too noisy)
        "compliance_improved": True,
    }
}


class AlertSystem:
    """System-wide alerting for Maven."""

    def __init__(self):
        self.alert_rules = DEFAULT_ALERT_RULES.copy()
        self.alert_handlers: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_counts = defaultdict(int)
        self.suppressed_alerts = set()  # For deduplication
        self.suppression_window = 300  # 5 minutes in seconds

    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to handle alerts.

        Args:
            handler: Function that takes alert dict as parameter

        Example:
            def my_handler(alert):
                print(f"Alert: {alert['message']}")

            alert_system.register_alert_handler(my_handler)
        """
        self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable) -> bool:
        """Remove a registered alert handler."""
        try:
            self.alert_handlers.remove(handler)
            return True
        except ValueError:
            return False

    def trigger_alert(
        self,
        severity: str,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger an alert.

        Args:
            severity: "info", "warning", or "critical"
            category: Alert category (health, performance, compliance, etc.)
            message: Human-readable alert message
            details: Optional additional details
            source: Optional source identifier (e.g., brain name)

        Returns:
            Alert dict with metadata
        """
        # Validate severity
        if severity not in SEVERITY_LEVELS:
            severity = "info"

        # Check if this alert type is enabled
        if not self._is_alert_enabled(severity, category):
            return {"suppressed": True, "reason": "alert_type_disabled"}

        # Create alert object
        alert = {
            "severity": severity,
            "category": category,
            "message": message,
            "details": details or {},
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "timestamp_unix": time.time(),
            "alert_id": self._generate_alert_id(severity, category, message)
        }

        # Check for suppression (deduplication)
        if self._is_suppressed(alert):
            return {"suppressed": True, "reason": "duplicate_within_window"}

        # Add to suppression set
        self._add_to_suppression(alert)

        # Add to history
        self.alert_history.append(alert)

        # Update counters
        self.alert_counts[severity] += 1
        self.alert_counts[f"{severity}_{category}"] += 1

        # Store in memory for persistence
        if _memory:
            try:
                _memory.store(
                    content=f"alert:{severity}:{category}",
                    metadata={
                        "kind": "alert",
                        "alert": alert,
                        "confidence": 1.0
                    }
                )
            except Exception:
                pass  # Non-critical

        # Call all registered handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"[ALERT_SYSTEM] Handler error: {e}")

        # Auto-remediation for critical alerts
        if severity == "critical":
            self._attempt_auto_remediation(alert)

        return alert

    def _is_alert_enabled(self, severity: str, category: str) -> bool:
        """Check if alert type is enabled in rules."""
        if severity not in self.alert_rules:
            return True  # Default to enabled if not specified

        rules_for_severity = self.alert_rules[severity]
        return rules_for_severity.get(category, True)  # Default to enabled

    def _generate_alert_id(self, severity: str, category: str, message: str) -> str:
        """Generate a unique-ish ID for alert deduplication."""
        # Simple hash based on severity, category, and first 50 chars of message
        identifier = f"{severity}:{category}:{message[:50]}"
        return str(hash(identifier))

    def _is_suppressed(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed (deduplicated)."""
        alert_id = alert["alert_id"]

        # Check if already in suppression set with recent timestamp
        for suppressed_id, suppressed_time in list(self.suppressed_alerts):
            if suppressed_id == alert_id:
                # Check if still within suppression window
                if time.time() - suppressed_time < self.suppression_window:
                    return True
                else:
                    # Remove expired suppression
                    self.suppressed_alerts.discard((suppressed_id, suppressed_time))

        return False

    def _add_to_suppression(self, alert: Dict[str, Any]) -> None:
        """Add alert to suppression set."""
        alert_id = alert["alert_id"]
        self.suppressed_alerts.add((alert_id, time.time()))

        # Clean up old suppressions (older than window)
        current_time = time.time()
        self.suppressed_alerts = {
            (aid, ts) for aid, ts in self.suppressed_alerts
            if current_time - ts < self.suppression_window
        }

    def _attempt_auto_remediation(self, alert: Dict[str, Any]) -> None:
        """Attempt automatic remediation for critical alerts."""
        category = alert["category"]
        details = alert.get("details", {})

        # Auto-remediation logic based on alert type
        if category == "brain_failure":
            # Could attempt brain restart/reset
            print(f"[ALERT_AUTO_REMEDIATION] Brain failure detected: {alert['source']}")

        elif category == "memory_overflow":
            # Could trigger memory cleanup
            print(f"[ALERT_AUTO_REMEDIATION] Memory overflow detected")

        elif category == "high_error_rate":
            # Could trigger diagnostic scan
            brain = details.get("brain_name")
            if brain:
                print(f"[ALERT_AUTO_REMEDIATION] High error rate in {brain} - triggering diagnostic")

    def get_recent_alerts(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts, optionally filtered.

        Args:
            severity: Filter by severity
            category: Filter by category
            limit: Maximum number of alerts to return

        Returns:
            List of alert dicts (most recent first)
        """
        filtered = []

        # Iterate in reverse (most recent first)
        for alert in reversed(self.alert_history):
            # Apply filters
            if severity and alert["severity"] != severity:
                continue
            if category and alert["category"] != category:
                continue

            filtered.append(alert)

            if len(filtered) >= limit:
                break

        return filtered

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alert counts and recent activity.

        Returns:
            Dict with alert statistics
        """
        # Count by severity in last hour
        one_hour_ago = time.time() - 3600
        recent_by_severity = {"info": 0, "warning": 0, "critical": 0}

        for alert in self.alert_history:
            if alert["timestamp_unix"] > one_hour_ago:
                recent_by_severity[alert["severity"]] += 1

        # Get most recent critical alert
        recent_critical = None
        for alert in reversed(self.alert_history):
            if alert["severity"] == "critical":
                recent_critical = alert
                break

        return {
            "total_alerts": len(self.alert_history),
            "total_by_severity": dict(self.alert_counts),
            "recent_hour_by_severity": recent_by_severity,
            "recent_critical_alert": recent_critical,
            "active_handlers": len(self.alert_handlers),
            "suppression_window_seconds": self.suppression_window,
            "currently_suppressed": len(self.suppressed_alerts)
        }

    def clear_alert_history(self) -> None:
        """Clear all alert history."""
        self.alert_history.clear()
        self.alert_counts.clear()
        self.suppressed_alerts.clear()

    def set_alert_rule(self, severity: str, category: str, enabled: bool) -> None:
        """
        Enable or disable a specific alert rule.

        Args:
            severity: Alert severity level
            category: Alert category
            enabled: True to enable, False to disable
        """
        if severity not in self.alert_rules:
            self.alert_rules[severity] = {}

        self.alert_rules[severity][category] = enabled

    def set_suppression_window(self, seconds: int) -> None:
        """Set the alert deduplication window in seconds."""
        self.suppression_window = max(0, seconds)


# Built-in alert handlers

def console_alert_handler(alert: Dict[str, Any]) -> None:
    """Print alerts to console with formatting."""
    severity = alert["severity"].upper()
    category = alert["category"]
    message = alert["message"]
    source = alert.get("source", "system")

    # Color codes (if terminal supports it)
    colors = {
        "CRITICAL": "\033[91m",  # Red
        "WARNING": "\033[93m",   # Yellow
        "INFO": "\033[94m",      # Blue
        "RESET": "\033[0m"
    }

    color = colors.get(severity, colors["INFO"])
    reset = colors["RESET"]

    print(f"{color}[{severity}]{reset} [{category}] {source}: {message}")


def log_alert_handler(alert: Dict[str, Any]) -> None:
    """Log alerts to file (placeholder)."""
    # In real implementation, would write to log file
    log_entry = {
        "timestamp": alert["timestamp"],
        "severity": alert["severity"],
        "category": alert["category"],
        "message": alert["message"],
        "source": alert.get("source"),
        "details": alert.get("details")
    }
    # Would write: json.dumps(log_entry) to log file
    pass


# Singleton instance
_alert_system = AlertSystem()


def get_alert_system() -> AlertSystem:
    """Get the global alert system instance."""
    return _alert_system


def trigger_alert(
    severity: str,
    category: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to trigger an alert."""
    return _alert_system.trigger_alert(severity, category, message, details, source)


def get_recent_alerts(severity: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience function to get recent alerts."""
    return _alert_system.get_recent_alerts(severity=severity, limit=limit)


def register_console_handler() -> None:
    """Register the console alert handler."""
    _alert_system.register_alert_handler(console_alert_handler)


# Auto-register console handler by default
register_console_handler()
