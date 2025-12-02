"""
Routing Package - Maven's Layered Routing System
=================================================

This package provides the unified routing system for Maven.
All routing decisions should go through routing_engine.build_routing_plan().

Modules:
    - router_schema: Data classes (RoutingDecision, ParsedCommand, etc.)
    - routing_engine: Main entry point (build_routing_plan)
    - normalizer: Text normalization for typo tolerance

Pipeline:
    raw text → normalize() → grammar/router → brains/tools

Usage:
    from brains.routing import build_routing_plan, RoutingDecision

    decision = build_routing_plan(
        query="x grok hello",  # typos are auto-corrected
        capability_snapshot=snapshot,
    )
"""

from brains.routing.router_schema import (
    RoutingDecision,
    ParsedCommand,
    LLMRouterResult,
    RoutingExample,
    RoutingContext,
    RouterToolChoice,
    RouterBrainChoice,
    VALID_BRAINS,
    VALID_TOOLS,
)

from brains.routing.routing_engine import (
    build_routing_plan,
    is_explicit_command,
)

from brains.routing.normalizer import (
    normalize,
    normalize_for_routing,
    NormalizationResult,
    save_typo,
    learn_from_routing_error,
)

__all__ = [
    # Main function
    "build_routing_plan",
    "is_explicit_command",
    # Normalizer
    "normalize",
    "normalize_for_routing",
    "NormalizationResult",
    "save_typo",
    "learn_from_routing_error",
    # Data classes
    "RoutingDecision",
    "ParsedCommand",
    "LLMRouterResult",
    "RoutingExample",
    "RoutingContext",
    "RouterToolChoice",
    "RouterBrainChoice",
    # Constants
    "VALID_BRAINS",
    "VALID_TOOLS",
]
