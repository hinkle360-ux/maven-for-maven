"""
graph_engine.py
~~~~~~~~~~~~~~~~~

This module implements a very simple cognitive graph engine for Maven.  It is
designed to replace the fixed broadcast pipeline with a flexible graph of
interconnected nodes.  Each cognitive brain registers itself as a node
with declared inputs, outputs and triggers.  Edges between nodes are
established via the ``connect`` API and data flows through the graph when
``emit`` is called.  During execution the engine records a trace of the
activations and edges to ``reports/trace_graph.jsonl`` so that later
analysis can understand the dynamic cognitive flow.

The graph engine intentionally remains minimal and offline-only to comply
with the constraints of the Maven environment: no external dependencies,
no new package roots and no changes to the linear broadcast pipeline
unless explicitly enabled.  It is expected that future upgrades will
evolve this skeleton into a more sophisticated attention‐routed
architecture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .message_bus import get_message_bus  # reuse existing bus

# Path to write traces; this location mirrors other report files.
TRACE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "trace_graph.jsonl")


@dataclass
class GraphNode:
    """Represents a node in the cognitive graph.

    Attributes:
        name: Unique identifier for the node.
        process_func: Callable that performs the node's computation.  It
            accepts a context dict and arbitrary payload, returning a
            possibly updated payload.  If the node declines to process
            (e.g., its triggers do not match), it should return None.
        inputs: Optional list of input event types this node is interested in.
        outputs: Optional list of output event types it emits.
    """

    name: str
    process_func: Callable[[Dict[str, Any], Any], Optional[Any]]
    inputs: Optional[List[str]] = field(default_factory=list)
    outputs: Optional[List[str]] = field(default_factory=list)


class GraphEngine:
    """A flexible cognitive graph engine with bounded execution.

    Nodes can be registered with declared inputs and outputs, and directed
    edges define propagation order.  Events are emitted into the graph
    starting from entry nodes matching the event type.  The engine stops
    propagation once a configurable maximum number of node visits is
    reached (if provided).  Each emission writes a trace of visited
    nodes and their outputs to ``trace_graph.jsonl`` for offline analysis.
    Duplicate node registrations are ignored to provide idempotent setup.
    """

    def __init__(self, max_steps: Optional[int] = None) -> None:
        # mapping of node name to GraphNode
        self.nodes: Dict[str, GraphNode] = {}
        # adjacency list: source node -> list of destination nodes
        self.edges: Dict[str, List[str]] = {}
        # optional maximum number of node visits per event
        self.max_steps: Optional[int] = max_steps

    def register_node(self, node: GraphNode) -> None:
        """Register a node with the graph.

        If a node with the same name is already present, the new
        registration is ignored to avoid accidental replacement.  This
        behaviour simplifies graph construction by making registration
        idempotent.
        """
        if node.name in self.nodes:
            return
        self.nodes[node.name] = node
        # ensure an adjacency list exists for this node
        self.edges.setdefault(node.name, [])

    def connect(self, source: str, dest: str) -> None:
        """Connect two nodes by name.

        Creates a directed edge from ``source`` to ``dest``.  Duplicate
        edges and invalid endpoints are ignored silently.
        """
        if source not in self.nodes or dest not in self.nodes:
            return
        self.edges.setdefault(source, [])
        if dest not in self.edges[source]:
            self.edges[source].append(dest)

    def emit(self, event_type: str, payload: Any, context: Dict[str, Any]) -> Any:
        """Emit an event into the graph.

        Finds entry nodes whose ``inputs`` include the given event type
        and recursively processes each branch.  Propagation halts when
        ``max_steps`` nodes have been visited (if set).  A trace is
        appended summarising the order of visits and outputs.
        Returns the last non-``None`` payload produced.
        """
        result = payload
        visits: List[Tuple[str, Any]] = []
        # determine entry nodes interested in this event type
        entry_nodes = [n for n in self.nodes.values() if event_type in (n.inputs or [])]
        for node in entry_nodes:
            r = self._run_node(node.name, result, context, visits, 0)
            if r is not None:
                result = r
        if visits:
            self._append_trace({
                "event_type": event_type,
                "payload": payload,
                "visits": [(n, o) for n, o in visits],
            })
        return result

    def _run_node(self, node_name: str, payload: Any, context: Dict[str, Any], visits: List[Tuple[str, Any]], step_count: int) -> Optional[Any]:
        """Recursively process a node and its successors.

        The depth-first traversal stops when ``max_steps`` is reached.
        Returns the last non-None output along this branch.
        """
        if self.max_steps is not None and step_count >= self.max_steps:
            return None
        node = self.nodes.get(node_name)
        if node is None:
            return None
        result: Optional[Any] = None
        try:
            out = node.process_func(context, payload)
        except Exception:
            out = None
        if out is not None:
            result = out
        visits.append((node_name, out))
        step_count += 1
        for succ in self.edges.get(node_name, []):
            r = self._run_node(succ, result, context, visits, step_count)
            if r is not None:
                result = r
        return result

    def run(self, context: Dict[str, Any]) -> None:
        """Run a full cognition cycle.

        Drains all events from the message bus and emits them through the
        graph one by one.  Each event respects the ``max_steps`` limit.
        """
        bus = get_message_bus()
        events = bus.pop_all() or []
        for ev in events:
            etype = ev.get("type")
            payload = ev.get("payload")
            self.emit(etype, payload, context)

    def _append_trace(self, record: Dict[str, Any]) -> None:
        """Append a JSON record to the trace file.

        Creates the parent directory if it does not exist.
        """
        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        try:
            with open(TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # ignore file I/O errors
            pass


def default_graph_engine() -> GraphEngine:
    """Instantiate and configure a default graph engine.

    This helper sets up a minimal graph connecting the existing reasoning
    and planning brains via working memory events.  It is not enabled
    by default; consumers must explicitly call ``run`` on the returned
    engine.
    """
    engine = GraphEngine()
    # Register placeholder nodes for existing brains.  These wrappers
    # leverage the existing service APIs.  If a brain does not need to
    # modify the payload it returns None.
    from .reasoning.service.reasoning_brain import service_api as reason_api  # noqa
    from .planner.service.planner_brain import service_api as planner_api  # noqa
    from .memory_librarian.service.memory_librarian import service_api as memory_api  # noqa

    def reasoning_node(context: Dict[str, Any], payload: Any) -> Optional[Any]:
        # only process if payload is a dict with a question
        if not isinstance(payload, dict) or "question" not in payload:
            return None
        # call the reasoning brain; context carries necessary state
        try:
            response = reason_api("ANSWER_QUESTION", {"question": payload["question"]}, context)
            # write the answer into working memory
            memory_api("WM_PUT", {
                "key": payload.get("question_key", "generic"),
                "value": response,
                "tags": ["answer"],
                "confidence": response.get("confidence", 0.0),
                "ttl": 60.0,
            }, context)
            return response
        except Exception:
            return None

    def planner_node(context: Dict[str, Any], payload: Any) -> Optional[Any]:
        # planner reacts to plan events
        if payload is None or not isinstance(payload, dict):
            return None
        if payload.get("plan"):
            try:
                planner_api("CREATE_GOAL", {"plan": payload["plan"]}, context)
            except Exception:
                pass
        return None

    def memory_listener(context: Dict[str, Any], payload: Any) -> Optional[Any]:
        # memory listener triggers when new working memory events occur
        # simply forward the payload downstream
        return payload

    # Register nodes
    engine.register_node(GraphNode("reasoning", reasoning_node, inputs=["question"], outputs=["answer"]))
    engine.register_node(GraphNode("planner", planner_node, inputs=["plan"], outputs=[]))
    engine.register_node(GraphNode("memory", memory_listener, inputs=["WM_EVENT"], outputs=[]))
    # Connect nodes: WM_EVENT -> reasoning and planner
    engine.connect("memory", "reasoning")
    engine.connect("memory", "planner")
    return engine