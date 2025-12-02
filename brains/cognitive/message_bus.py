"""
Simple Message Bus
===================

This module implements a minimal in‑memory message bus for cross‑brain
communication.  It allows brains to send structured messages to each
other without introducing any external dependencies.  Messages are
appended to a global list and can be retrieved or cleared by other
components.  This design deliberately avoids complex concurrency or
persistence; it is intended for a single‑threaded pipeline where
messages are consumed synchronously by the memory librarian or other
brains.

Example usage::

    from brains.cognitive.message_bus import send, pop_all

    # Reasoning brain sends a targeted search request
    send({
        "from": "reasoning",
        "to": "memory",
        "type": "SEARCH_REQUEST",
        "domains": ["science"],
        "confidence_threshold": 0.7,
    })

    # Memory librarian retrieves pending messages
    messages = pop_all()
    for msg in messages:
        handle_message(msg)

Note: The message bus is intentionally simple.  In a future phase it
could be replaced by a more robust pub/sub mechanism, but this
implementation suffices for Phase 1 to enable targeted queries
between brains.
"""

from __future__ import annotations

from typing import Dict, Any, List

_messages: List[Dict[str, Any]] = []

def send(message: Dict[str, Any]) -> None:
    """Send a message to the bus.

    Appends the provided message dictionary to the internal message
    queue.  Messages are not validated; callers are responsible for
    using a consistent schema.

    Args:
        message: A dictionary representing the message payload.
    """
    try:
        if message:
            _messages.append(dict(message))
    except Exception:
        # Silently ignore errors to avoid disrupting the caller
        pass

def pop_all() -> List[Dict[str, Any]]:
    """Return all pending messages and clear the bus.

    Returns:
        A list of messages that were queued since the last call.
    """
    msgs = list(_messages)
    _messages.clear()
    return msgs

class MessageBus:
    """Simple wrapper class for the message bus interface."""

    def send(self, message: Dict[str, Any]) -> None:
        """Send a message to the bus."""
        send(message)

    def pop_all(self) -> List[Dict[str, Any]]:
        """Return all pending messages and clear the bus."""
        return pop_all()

_bus_instance = MessageBus()

def get_message_bus() -> MessageBus:
    """Return the singleton message bus instance.

    Returns:
        The global MessageBus instance.
    """
    return _bus_instance