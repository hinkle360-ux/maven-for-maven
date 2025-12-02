"""
Simple shared Blackboard implementation
=====================================

This module implements a lightweight publish/subscribe blackboard that can
be used by disparate parts of the Maven cognitive architecture to share
structured events without introducing new dependencies or folders.  It
provides a ring‐buffer per topic with a fixed maximum length and basic
subscription support.  Consumers may subscribe to one or more topics
and poll for new items since their last poll.  Producers can publish
arbitrary JSON‐serialisable objects to a topic.  All state is kept in
memory only; persistence to disk is optional and left to callers.

Usage:

    from brains.agent.service.blackboard import put, get, subscribe, poll

    # Producer writes an utterance
    put("dialogue", {"type": "utterance", "role": "assistant", "text": "Hello"})

    # Consumer subscribes to a topic
    subscribe("my_sub", "dialogue")
    # Later, poll for new events
    events = poll("my_sub")

The implementation uses only the Python standard library and imposes
no concurrency controls.  It is safe for sequential use within a single
process.  For multi‐process scenarios a more robust IPC mechanism would
be required.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, MutableMapping

__all__ = ["put", "get", "subscribe", "poll"]

# Maximum number of events retained per topic.  When the capacity is
# exceeded, the oldest events are discarded automatically by the deque.
_MAX_LEN = 100

# Internal mapping of topic → deque of events.  Each topic maintains
# its own ring buffer.  Keys are strings, values are deque objects.
_topics: MutableMapping[str, deque] = {}

# Internal mapping of subscriber id → dict(topic → last read index).  Each
# subscriber tracks the index within each topic so that only unseen
# events are returned on the next poll.  Subscriber ids must be unique
# per consumer.
_subscribers: MutableMapping[str, Dict[str, int]] = {}


def put(topic: str, item: Any) -> None:
    """Publish an item to the specified topic.

    If the topic does not yet exist it will be created.  Items beyond
    ``_MAX_LEN`` are automatically dropped.  This function does not
    perform any validation on the item; callers should ensure that
    published objects are JSON‑serialisable if persistence or
    interoperability is required.

    Args:
        topic: Name of the topic to publish to.
        item: Any object to append to the topic's ring buffer.
    """
    # Get or create the deque for this topic
    dq = _topics.get(topic)
    if dq is None:
        dq = deque(maxlen=_MAX_LEN)
        _topics[topic] = dq
    dq.append(item)


def get(topic: str, window: int | None = 10) -> List[Any]:
    """Retrieve the most recent items from a topic.

    Args:
        topic: The topic to read from.
        window: The number of most recent items to return.  If None,
            all available items are returned.

    Returns:
        A list of items from the topic.  An empty list is returned if
        the topic does not exist or contains no events.
    """
    dq = _topics.get(topic)
    if not dq:
        return []
    items = list(dq)
    if window is None:
        return items
    # Limit to the last N items
    return items[-window:]


def subscribe(subscriber: str, topic: str) -> None:
    """Subscribe a consumer to a topic.

    Consumers call this once per topic they wish to observe.  The
    internal cursor for the subscriber will be initialised to the
    current length of the topic, so that only events published after
    subscription are delivered.

    Args:
        subscriber: Unique identifier for the subscriber.  Each consumer
            must use a distinct id to avoid cursor collisions.
        topic: Topic name to subscribe to.
    """
    subs = _subscribers.setdefault(subscriber, {})
    # Start cursor at end of existing events
    subs[topic] = len(_topics.get(topic, []))


def poll(subscriber: str) -> List[Dict[str, Any]]:
    """Retrieve all unseen events for a subscriber.

    This function returns a list of dictionaries, one per event, each
    containing the topic name and the event payload.  After polling,
    the subscriber's cursor is advanced to the end of each topic so
    that subsequent polls return only new events.

    Args:
        subscriber: The identifier used when subscribing.

    Returns:
        A list of dicts with keys ``topic`` and ``item``.  If the
        subscriber is unknown or has no subscriptions, an empty list is
        returned.
    """
    subs = _subscribers.get(subscriber)
    if not subs:
        return []
    events: List[Dict[str, Any]] = []
    for topic, idx in subs.items():
        dq = _topics.get(topic)
        if not dq:
            continue
        items = list(dq)
        # Clamp index to range
        start_idx = idx if idx >= 0 else 0
        # Collect new events since last poll
        new_events = items[start_idx:]
        # Advance cursor to end
        subs[topic] = len(items)
        for it in new_events:
            events.append({"topic": topic, "item": it})
    return events
