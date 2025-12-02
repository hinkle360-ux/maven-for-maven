"""
Time Now Tool
=============

This tool provides real-time access to the system clock.
It returns the current local time and timezone information.

CRITICAL: Time questions MUST route to this tool, NOT to Teacher.
The Teacher/LLM cannot provide accurate real-time information.

The service API supports a single operation ``GET_TIME`` with an optional
``timezone`` payload and returns the current time in a structured format.

Example:

>>> service_api({"op": "GET_TIME", "payload": {}})
{"ok": True, "payload": {"local_time": "10:42 AM", "datetime": "2024-01-15T10:42:33", "timezone": "America/New_York", "formatted": "It's 10:42 AM in America/New_York."}}
"""

from __future__ import annotations

import time as time_module
from datetime import datetime
from typing import Dict, Any, Optional

# Try to get timezone info - use stdlib only
try:
    from datetime import timezone
    _has_timezone = True
except ImportError:
    _has_timezone = False


def _get_local_timezone_name() -> str:
    """
    Get the local timezone name from the system.

    Uses stdlib only - no external dependencies.
    Falls back to UTC offset if name unavailable.
    """
    try:
        # Try to get timezone name from time module
        if time_module.daylight and time_module.localtime().tm_isdst:
            tz_name = time_module.tzname[1]  # Daylight saving time
        else:
            tz_name = time_module.tzname[0]  # Standard time

        if tz_name:
            return tz_name
    except Exception:
        pass

    # Fallback: compute UTC offset
    try:
        local_now = datetime.now()
        utc_now = datetime.utcnow()
        offset_seconds = (local_now - utc_now).total_seconds()
        offset_hours = int(offset_seconds // 3600)
        offset_minutes = int((abs(offset_seconds) % 3600) // 60)

        if offset_seconds >= 0:
            return f"UTC+{offset_hours:02d}:{offset_minutes:02d}"
        else:
            return f"UTC{offset_hours:+03d}:{offset_minutes:02d}"
    except Exception:
        return "Local"


def _format_time_12h(dt: datetime) -> str:
    """Format time in 12-hour format with AM/PM."""
    try:
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return dt.strftime("%H:%M")


def _format_time_24h(dt: datetime) -> str:
    """Format time in 24-hour format."""
    return dt.strftime("%H:%M")


def _format_datetime_iso(dt: datetime) -> str:
    """Format datetime in ISO 8601 format."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _format_date_friendly(dt: datetime) -> str:
    """Format date in a friendly format."""
    return dt.strftime("%A, %B %d, %Y")


def get_current_time(format_24h: bool = False) -> Dict[str, Any]:
    """
    Get the current local time and timezone information.

    This is the core function that hits the system clock.

    Args:
        format_24h: If True, use 24-hour format. Default is 12-hour.

    Returns:
        Dict with:
        - local_time: str (e.g., "10:42 AM" or "10:42")
        - datetime: str (ISO 8601 format)
        - date: str (friendly format, e.g., "Monday, January 15, 2024")
        - timezone: str (timezone name or UTC offset)
        - timestamp: float (Unix timestamp)
        - formatted: str (human-readable sentence)
    """
    now = datetime.now()
    tz_name = _get_local_timezone_name()

    if format_24h:
        time_str = _format_time_24h(now)
    else:
        time_str = _format_time_12h(now)

    return {
        "local_time": time_str,
        "datetime": _format_datetime_iso(now),
        "date": _format_date_friendly(now),
        "timezone": tz_name,
        "timestamp": now.timestamp(),
        "formatted": f"It's {time_str} in {tz_name}.",
        "day_of_week": now.strftime("%A"),
        "month": now.strftime("%B"),
        "month_number": now.month,
        "year": now.year,
        "day": now.day,
        "week_number": int(now.strftime("%W")),
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
    }


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for the time_now tool.

    Operations:
    - GET_TIME: Get current time (primary operation)
    - GET_DATE: Get current date (alias for GET_TIME, same data)
    - GET_CALENDAR: Get calendar info (alias for GET_TIME, same data)
    - HEALTH: Health check

    Args:
        msg: Dict with "op" and optional "payload"

    Returns:
        Dict with "ok", "payload" or "error"
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "TIME_NOW"

    # GET_TIME, GET_DATE, GET_CALENDAR all return the same comprehensive data
    if op in ("GET_TIME", "GET_DATE", "GET_CALENDAR"):
        format_24h = payload.get("format_24h", False)

        try:
            result = get_current_time(format_24h=format_24h)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "TIME_ERROR", "message": str(e)}
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service": "time_now",
                "capability": "time",
                "description": "Real-time system clock access (time, date, calendar)"
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"}
    }


# Standard service contract: handle is the entry point
handle = service_api


# =============================================================================
# TOOL METADATA (for registry and capabilities)
# =============================================================================

TOOL_NAME = "time_now"
TOOL_CAPABILITY = "time"
TOOL_DESCRIPTION = "Get current time from system clock"
TOOL_OPERATIONS = ["GET_TIME", "HEALTH"]


def is_available() -> bool:
    """Check if the time tool is available (always True for stdlib-only tool)."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
        "requires_execution": False,  # No sandboxing needed
        "module": "brains.agent.tools.time_now",
    }


# =============================================================================
# CONVENIENCE FUNCTION FOR DIRECT USE
# =============================================================================

def what_time_is_it() -> str:
    """
    Quick convenience function to get a human-readable time string.

    Returns:
        str: e.g., "It's 10:42 AM in America/New_York."
    """
    result = get_current_time()
    return result["formatted"]


# For testing
if __name__ == "__main__":
    print("Testing time_now tool:")
    print("-" * 40)

    # Test GET_TIME
    response = service_api({"op": "GET_TIME"})
    print(f"GET_TIME response: {response}")

    # Test HEALTH
    response = service_api({"op": "HEALTH"})
    print(f"HEALTH response: {response}")

    # Test convenience function
    print(f"what_time_is_it(): {what_time_is_it()}")
