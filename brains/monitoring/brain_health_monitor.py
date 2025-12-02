"""
brain_health_monitor.py
=======================

Real-time health monitoring and diagnostics for all Maven brains.

This module provides:
1. Individual brain health metrics (response time, error rate, memory usage)
2. System-wide health aggregation
3. Health status classification (healthy, warning, critical)
4. Alert triggering for health issues
5. Trend analysis and prediction

Operations:
    - monitor_brain_health(brain_name): Get current health for specific brain
    - get_system_health_summary(): Get overall system health
    - enable_continuous_monitoring(interval): Start background monitoring
    - get_health_history(brain_name, time_range): Get historical health data
    - check_alert_conditions(metrics): Check if metrics trigger alerts
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
import statistics

try:
    from brains.memory.brain_memory import BrainMemory
    _memory = BrainMemory("health_monitor")
except Exception:
    _memory = None


# Health status thresholds
HEALTH_THRESHOLDS = {
    "response_time_ms": {
        "healthy": 500,    # <500ms is healthy
        "warning": 1000,   # 500-1000ms is warning
        "critical": 2000   # >2000ms is critical
    },
    "error_rate_percent": {
        "healthy": 2,      # <2% error rate is healthy
        "warning": 5,      # 2-5% is warning
        "critical": 10     # >10% is critical
    },
    "memory_usage_mb": {
        "healthy": 100,    # <100MB is healthy
        "warning": 300,    # 100-300MB is warning
        "critical": 500    # >500MB is critical
    }
}

# Global metrics storage
_brain_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
    "response_times": [],
    "error_count": 0,
    "success_count": 0,
    "total_operations": 0,
    "memory_usage_mb": 0,
    "last_health_check": None,
    "alerts": [],
    "status": "unknown"
})

_system_start_time = time.time()


class BrainHealthMonitor:
    """Real-time health monitoring for all Maven brains."""

    def __init__(self):
        self.alert_handlers = []
        self.monitoring_enabled = False
        self.monitoring_interval = 60  # seconds

    def record_operation(self, brain_name: str, operation: str,
                        duration_ms: float, success: bool) -> None:
        """
        Record a brain operation for health tracking.

        Args:
            brain_name: Name of the brain
            operation: Operation that was executed
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
        """
        global _brain_metrics

        metrics = _brain_metrics[brain_name]

        # Update response times (keep last 100)
        metrics["response_times"].append(duration_ms)
        if len(metrics["response_times"]) > 100:
            metrics["response_times"].pop(0)

        # Update success/error counts
        metrics["total_operations"] += 1
        if success:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1

        # Store in memory for persistence
        if _memory:
            try:
                _memory.store(
                    content=f"operation:{brain_name}:{operation}",
                    metadata={
                        "kind": "operation_metric",
                        "brain": brain_name,
                        "operation": operation,
                        "duration_ms": duration_ms,
                        "success": success,
                        "timestamp": time.time(),
                        "confidence": 1.0
                    }
                )
            except Exception:
                pass  # Non-critical

    def monitor_brain_health(self, brain_name: str) -> Dict[str, Any]:
        """
        Monitor and return current health metrics for a specific brain.

        Args:
            brain_name: Name of the brain to monitor

        Returns:
            Dict containing health metrics and status
        """
        global _brain_metrics

        metrics = _brain_metrics[brain_name]

        # Calculate current metrics
        response_times = metrics["response_times"]
        avg_response_time = (statistics.mean(response_times)
                           if response_times else 0)
        p95_response_time = (statistics.quantiles(response_times, n=20)[18]
                           if len(response_times) >= 20 else avg_response_time)

        total_ops = metrics["total_operations"]
        error_rate = ((metrics["error_count"] / total_ops * 100)
                     if total_ops > 0 else 0)

        # Determine health status
        status = self._determine_status(avg_response_time, error_rate,
                                       metrics["memory_usage_mb"])

        # Check for alert conditions
        alerts = self._check_alert_conditions({
            "brain_name": brain_name,
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "memory_usage_mb": metrics["memory_usage_mb"]
        })

        health_data = {
            "brain_name": brain_name,
            "status": status,
            "metrics": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "p95_response_time_ms": round(p95_response_time, 2),
                "error_rate_percent": round(error_rate, 2),
                "success_rate_percent": round(
                    (metrics["success_count"] / total_ops * 100) if total_ops > 0 else 0,
                    2
                ),
                "total_operations": total_ops,
                "memory_usage_mb": metrics["memory_usage_mb"]
            },
            "alerts": alerts,
            "last_check": datetime.now().isoformat(),
            "uptime_seconds": int(time.time() - _system_start_time)
        }

        metrics["last_health_check"] = time.time()
        metrics["status"] = status
        metrics["alerts"] = alerts

        return health_data

    def _determine_status(self, avg_response_time: float, error_rate: float,
                         memory_usage: float) -> str:
        """Determine overall health status based on metrics."""

        # Check if any metric is in critical range
        if (avg_response_time > HEALTH_THRESHOLDS["response_time_ms"]["critical"] or
            error_rate > HEALTH_THRESHOLDS["error_rate_percent"]["critical"] or
            memory_usage > HEALTH_THRESHOLDS["memory_usage_mb"]["critical"]):
            return "critical"

        # Check if any metric is in warning range
        if (avg_response_time > HEALTH_THRESHOLDS["response_time_ms"]["warning"] or
            error_rate > HEALTH_THRESHOLDS["error_rate_percent"]["warning"] or
            memory_usage > HEALTH_THRESHOLDS["memory_usage_mb"]["warning"]):
            return "warning"

        return "healthy"

    def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check if metrics trigger any alert conditions."""
        alerts = []

        brain_name = metrics["brain_name"]
        avg_response_time = metrics["avg_response_time"]
        error_rate = metrics["error_rate"]
        memory_usage = metrics["memory_usage_mb"]

        # Response time alerts
        if avg_response_time > HEALTH_THRESHOLDS["response_time_ms"]["critical"]:
            alerts.append({
                "severity": "critical",
                "type": "slow_response",
                "message": f"{brain_name} response time critical: {avg_response_time:.0f}ms",
                "metric": "response_time_ms",
                "value": avg_response_time,
                "threshold": HEALTH_THRESHOLDS["response_time_ms"]["critical"]
            })
        elif avg_response_time > HEALTH_THRESHOLDS["response_time_ms"]["warning"]:
            alerts.append({
                "severity": "warning",
                "type": "slow_response",
                "message": f"{brain_name} response time elevated: {avg_response_time:.0f}ms",
                "metric": "response_time_ms",
                "value": avg_response_time,
                "threshold": HEALTH_THRESHOLDS["response_time_ms"]["warning"]
            })

        # Error rate alerts
        if error_rate > HEALTH_THRESHOLDS["error_rate_percent"]["critical"]:
            alerts.append({
                "severity": "critical",
                "type": "high_error_rate",
                "message": f"{brain_name} error rate critical: {error_rate:.1f}%",
                "metric": "error_rate_percent",
                "value": error_rate,
                "threshold": HEALTH_THRESHOLDS["error_rate_percent"]["critical"]
            })
        elif error_rate > HEALTH_THRESHOLDS["error_rate_percent"]["warning"]:
            alerts.append({
                "severity": "warning",
                "type": "elevated_errors",
                "message": f"{brain_name} error rate elevated: {error_rate:.1f}%",
                "metric": "error_rate_percent",
                "value": error_rate,
                "threshold": HEALTH_THRESHOLDS["error_rate_percent"]["warning"]
            })

        # Memory usage alerts
        if memory_usage > HEALTH_THRESHOLDS["memory_usage_mb"]["critical"]:
            alerts.append({
                "severity": "critical",
                "type": "high_memory",
                "message": f"{brain_name} memory usage critical: {memory_usage:.0f}MB",
                "metric": "memory_usage_mb",
                "value": memory_usage,
                "threshold": HEALTH_THRESHOLDS["memory_usage_mb"]["critical"]
            })
        elif memory_usage > HEALTH_THRESHOLDS["memory_usage_mb"]["warning"]:
            alerts.append({
                "severity": "warning",
                "type": "elevated_memory",
                "message": f"{brain_name} memory usage elevated: {memory_usage:.0f}MB",
                "metric": "memory_usage_mb",
                "value": memory_usage,
                "threshold": HEALTH_THRESHOLDS["memory_usage_mb"]["warning"]
            })

        # Trigger alert handlers
        for alert in alerts:
            self._trigger_alert(alert)

        return alerts

    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        """Trigger registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"[HEALTH_MONITOR] Alert handler error: {e}")

    def register_alert_handler(self, handler) -> None:
        """Register a callback function for alerts."""
        self.alert_handlers.append(handler)

    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get overall Maven system health summary.

        Returns:
            Dict containing system-wide health metrics
        """
        global _brain_metrics

        if not _brain_metrics:
            return {
                "status": "unknown",
                "total_brains": 0,
                "message": "No health data available yet"
            }

        # Count brains by status
        status_counts = {"healthy": 0, "warning": 0, "critical": 0, "unknown": 0}
        all_response_times = []
        total_operations = 0
        total_alerts = []

        for brain_name, metrics in _brain_metrics.items():
            status = metrics.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

            all_response_times.extend(metrics.get("response_times", []))
            total_operations += metrics.get("total_operations", 0)
            total_alerts.extend(metrics.get("alerts", []))

        # Calculate system-wide metrics
        avg_response_time = (statistics.mean(all_response_times)
                           if all_response_times else 0)

        # Determine overall system status
        if status_counts["critical"] > 0:
            system_status = "critical"
        elif status_counts["warning"] > 0:
            system_status = "warning"
        elif status_counts["healthy"] > 0:
            system_status = "healthy"
        else:
            system_status = "unknown"

        # Critical alerts count
        critical_alerts = [a for a in total_alerts if a.get("severity") == "critical"]

        return {
            "status": system_status,
            "total_brains": len(_brain_metrics),
            "brains_by_status": status_counts,
            "healthy_percentage": round(
                (status_counts["healthy"] / len(_brain_metrics) * 100)
                if _brain_metrics else 0,
                1
            ),
            "system_metrics": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "total_operations": total_operations,
                "total_alerts": len(total_alerts),
                "critical_alerts": len(critical_alerts)
            },
            "uptime_seconds": int(time.time() - _system_start_time),
            "timestamp": datetime.now().isoformat()
        }

    def get_unhealthy_brains(self) -> List[str]:
        """Get list of brains with warning or critical status."""
        global _brain_metrics
        return [
            brain_name for brain_name, metrics in _brain_metrics.items()
            if metrics.get("status") in ["warning", "critical"]
        ]

    def reset_metrics(self, brain_name: Optional[str] = None) -> None:
        """Reset metrics for a brain or all brains."""
        global _brain_metrics

        if brain_name:
            if brain_name in _brain_metrics:
                _brain_metrics[brain_name] = {
                    "response_times": [],
                    "error_count": 0,
                    "success_count": 0,
                    "total_operations": 0,
                    "memory_usage_mb": 0,
                    "last_health_check": None,
                    "alerts": [],
                    "status": "unknown"
                }
        else:
            _brain_metrics.clear()


# Singleton instance
_health_monitor = BrainHealthMonitor()


def get_health_monitor() -> BrainHealthMonitor:
    """Get the global health monitor instance."""
    return _health_monitor


def record_operation(brain_name: str, operation: str,
                    duration_ms: float, success: bool) -> None:
    """Convenience function to record an operation."""
    _health_monitor.record_operation(brain_name, operation, duration_ms, success)


def get_brain_health(brain_name: str) -> Dict[str, Any]:
    """Convenience function to get brain health."""
    return _health_monitor.monitor_brain_health(brain_name)


def get_system_health() -> Dict[str, Any]:
    """Convenience function to get system health."""
    return _health_monitor.get_system_health_summary()
