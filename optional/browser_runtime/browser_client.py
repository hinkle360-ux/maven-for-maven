"""
Browser Client Bridge
=====================

Provides an `is_available()` function for the capabilities probe to check
whether the browser runtime server is running and reachable.

This module bridges the capability probe in brains/system_capabilities.py
with the actual browser runtime server.
"""

from __future__ import annotations

import socket
from typing import Optional

from optional.browser_runtime.config import get_config


def is_available(timeout: float = 1.0) -> bool:
    """
    Check if the browser runtime server is available.

    This does a simple TCP connection test to the server's host:port.
    A full HTTP healthcheck would require additional dependencies.

    Args:
        timeout: Connection timeout in seconds.

    Returns:
        True if the server is reachable, False otherwise.
    """
    config = get_config()
    host = config.host
    port = config.port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (socket.error, socket.timeout, OSError):
        return False


def get_server_url() -> str:
    """
    Get the URL of the browser runtime server.

    Returns:
        The base URL for the browser runtime server (e.g., "http://127.0.0.1:8765")
    """
    config = get_config()
    return f"http://{config.host}:{config.port}"


def health_check() -> Optional[dict]:
    """
    Perform an HTTP healthcheck against the browser runtime server.

    Returns:
        The health response dict if server is healthy, None otherwise.
    """
    try:
        import urllib.request
        import json

        url = f"{get_server_url()}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except Exception:
        return None


def open_url(target_url: str, wait_until: str = "domcontentloaded") -> dict:
    """
    Open a URL in the browser runtime server.

    This sends a POST request to the /open endpoint to navigate
    to a URL and return a page snapshot.

    Args:
        target_url: The URL to open.
        wait_until: When to consider navigation done ('domcontentloaded', 'load', 'networkidle').

    Returns:
        A dict with the result including page_id, snapshot, etc.
        On error, returns dict with 'error' key.
    """
    try:
        import urllib.request
        import json

        api_url = f"{get_server_url()}/open"
        payload = json.dumps({
            "url": target_url,
            "wait_until": wait_until,
        }).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30.0) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        return {"error": str(e), "success": False}
