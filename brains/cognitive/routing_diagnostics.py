"""
Routing Diagnostics Tool
========================

This module provides diagnostic capabilities to trace which path each request
takes through the Maven cognitive pipeline. It helps identify bypass routes,
cache hits, and ensures all requests flow through the cognitive pathway as
intended.

Usage:
    from brains.cognitive.routing_diagnostics import tracer, RouteType

    tracer.start_request(mid, text)
    tracer.record_route(mid, RouteType.FAST_CACHE)
    tracer.end_request(mid, final_answer)

    # Get statistics
    stats = tracer.get_statistics()
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from brains.maven_paths import get_reports_path


class RouteType(Enum):
    """Types of routes a request can take through the system."""
    FULL_PIPELINE = "full_pipeline"           # Goes through complete pipeline
    FAST_CACHE = "fast_cache"                 # Fast cache hit (bypasses pipeline)
    SEMANTIC_CACHE = "semantic_cache"         # Semantic cache hit (bypasses pipeline)
    SELF_QUERY = "self_query"                 # Self/environment hardcoded handler
    COMMAND = "command"                       # Direct command (DMN, status, etc.)
    STAGE6_GENERATE = "stage6_generate"       # Reached stage6_generate function
    TEMPLATE_MATCH = "template_match"         # Template handler in stage6
    HEURISTIC_MATCH = "heuristic_match"       # Heuristic handler in stage6
    LLM_FALLBACK = "llm_fallback"             # LLM fallback in stage6
    BLOCKED_MEMORY = "blocked_memory"         # Blocked by memory gate
    BLOCKED_GOVERNANCE = "blocked_governance" # Blocked by governance gate
    ERROR = "error"                           # Error occurred


@dataclass
class RequestTrace:
    """Trace information for a single request."""
    mid: str
    text: str
    start_time: float
    end_time: Optional[float] = None
    routes: List[RouteType] = None
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.routes is None:
            self.routes = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mid": self.mid,
            "text": self.text[:100] if self.text else "",  # Truncate for privacy
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else None,
            "routes": [r.value for r in self.routes],
            "primary_route": self.routes[0].value if self.routes else None,
            "reached_stage6": RouteType.STAGE6_GENERATE in self.routes,
            "bypassed": any(r in self.routes for r in [
                RouteType.FAST_CACHE,
                RouteType.SEMANTIC_CACHE,
                RouteType.SELF_QUERY
            ]),
            "final_answer_preview": self.final_answer[:50] if self.final_answer else None,
            "metadata": self.metadata
        }


class RoutingTracer:
    """Singleton tracer for tracking request routing through the system."""

    def __init__(self):
        self._active_requests: Dict[str, RequestTrace] = {}
        self._completed_requests: List[RequestTrace] = []
        self._enabled = True
        self._max_history = 1000
        self._report_dir = get_reports_path("routing_diagnostics")
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def enable(self):
        """Enable tracing."""
        self._enabled = True

    def disable(self):
        """Disable tracing."""
        self._enabled = False

    def start_request(self, mid: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Start tracing a new request.

        Args:
            mid: Message ID
            text: Request text
            metadata: Optional metadata dictionary
        """
        if not self._enabled:
            return

        trace = RequestTrace(
            mid=mid,
            text=text,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self._active_requests[mid] = trace

    def record_route(self, mid: str, route_type: RouteType, metadata: Optional[Dict[str, Any]] = None):
        """Record that a request took a specific route.

        Args:
            mid: Message ID
            route_type: Type of route taken
            metadata: Optional metadata about this route
        """
        if not self._enabled:
            return

        trace = self._active_requests.get(mid)
        if trace:
            trace.routes.append(route_type)
            if metadata:
                trace.metadata.update(metadata)

    def end_request(self, mid: str, final_answer: Optional[str] = None):
        """End tracing for a request.

        Args:
            mid: Message ID
            final_answer: The final answer returned
        """
        if not self._enabled:
            return

        trace = self._active_requests.pop(mid, None)
        if trace:
            trace.end_time = time.time()
            trace.final_answer = final_answer
            self._completed_requests.append(trace)

            # Write to JSONL log
            self._write_trace(trace)

            # Limit history size
            if len(self._completed_requests) > self._max_history:
                self._completed_requests = self._completed_requests[-self._max_history:]

    def _write_trace(self, trace: RequestTrace):
        """Write a single trace to the JSONL log file."""
        try:
            log_file = self._report_dir / f"routing_trace_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass  # Silent failure to avoid disrupting the pipeline

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about request routing.

        Returns:
            Dictionary with routing statistics
        """
        if not self._completed_requests:
            return {
                "total_requests": 0,
                "message": "No completed requests to analyze"
            }

        total = len(self._completed_requests)
        route_counts: Dict[str, int] = {}
        reached_stage6 = 0
        bypassed = 0
        avg_duration = 0

        for trace in self._completed_requests:
            # Count primary route (first route taken)
            if trace.routes:
                primary = trace.routes[0].value
                route_counts[primary] = route_counts.get(primary, 0) + 1

            # Check if reached stage6
            if RouteType.STAGE6_GENERATE in trace.routes:
                reached_stage6 += 1

            # Check if bypassed
            if any(r in trace.routes for r in [
                RouteType.FAST_CACHE,
                RouteType.SEMANTIC_CACHE,
                RouteType.SELF_QUERY
            ]):
                bypassed += 1

            # Calculate duration
            if trace.end_time:
                avg_duration += (trace.end_time - trace.start_time)

        avg_duration = (avg_duration / total) * 1000 if total > 0 else 0

        return {
            "total_requests": total,
            "reached_stage6_count": reached_stage6,
            "reached_stage6_percent": (reached_stage6 / total * 100) if total > 0 else 0,
            "bypassed_count": bypassed,
            "bypassed_percent": (bypassed / total * 100) if total > 0 else 0,
            "route_breakdown": route_counts,
            "average_duration_ms": round(avg_duration, 2),
            "report_location": str(self._report_dir)
        }

    def print_statistics(self):
        """Print routing statistics to console."""
        stats = self.get_statistics()

        print("\n" + "="*70)
        print("MAVEN ROUTING DIAGNOSTICS")
        print("="*70)

        if stats["total_requests"] == 0:
            print("No requests traced yet.")
            return

        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"Average Duration: {stats['average_duration_ms']:.2f}ms")
        print()

        print(f"Reached stage6_generate: {stats['reached_stage6_count']} ({stats['reached_stage6_percent']:.1f}%)")
        print(f"Bypassed Pipeline: {stats['bypassed_count']} ({stats['bypassed_percent']:.1f}%)")
        print()

        print("Route Breakdown:")
        for route, count in sorted(stats['route_breakdown'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total_requests'] * 100)
            print(f"  {route:25s}: {count:4d} ({pct:5.1f}%)")

        print()
        print(f"Detailed logs: {stats['report_location']}")
        print("="*70 + "\n")

    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent request traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of trace dictionaries
        """
        recent = self._completed_requests[-limit:]
        return [trace.to_dict() for trace in recent]

    def clear_history(self):
        """Clear all completed request history."""
        self._completed_requests.clear()


# Global singleton instance
tracer = RoutingTracer()


# Convenience functions
def enable_tracing():
    """Enable routing diagnostics."""
    tracer.enable()


def disable_tracing():
    """Disable routing diagnostics."""
    tracer.disable()


def print_stats():
    """Print routing statistics."""
    tracer.print_statistics()
