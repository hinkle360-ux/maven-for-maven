"""
Integrator Brain
================

This module provides a minimal implementation of the proposed
cognitive synchronisation layer outlined in the Stage 2.5 → 3.0
roadmap.  The purpose of the integrator brain is to arbitrate
attention between specialist brains and, in future phases, to
coordinate meta‑cognitive functions such as self‑modelling and
cross‑brain messaging.  For now, it implements a simple attention
manager that resolves competing bids for cognitive focus.

The integrator brain exposes a ``service_api`` entry point with a
single operation:

``RESOLVE``
    Accepts a list of bids from other brains (each encoded as a
    dictionary with ``brain_name``, ``priority``, ``reason`` and
    ``evidence`` keys) and returns the name of the brain that should
    receive focus.  The resolution algorithm implements the rule‑based
    arbiter described in the roadmap: contradictions detected by the
    reasoning brain take precedence, followed by unanswered questions
    handled by the language brain, then the bid with the highest
    priority value.

The module also defines two simple data containers, ``BrainBid`` and
``AttentionState``.  These are plain Python classes rather than
dataclasses because they need to remain lightweight and avoid pulling
in additional dependencies.  ``BrainBid`` encapsulates a single bid
from a brain for attention, while ``AttentionState`` records the
current focus and a history of resolved transitions.  The history is
not persisted across runs and is maintained in memory only for
debugging purposes.

This implementation is deliberately conservative: it does not attempt
to modify the global pipeline directly.  Downstream callers such as
``memory_librarian.service`` may import this module and call
``service_api`` with the ``RESOLVE`` op to determine which brain
should take precedence during the current pipeline invocation.  The
decision and its supporting evidence can then be recorded in the
context for auditing or future self‑review.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import re
from capabilities import get_capabilities

# Pattern store for unified learning across all brains
from brains.cognitive.pattern_store import (
    get_pattern_store,
    Pattern,
    verdict_to_reward
)
from brains.cognitive.integrator.initial_patterns import (
    initialize_integrator_patterns
)

# Unified routing engine (layered routing: Grammar > Self-Intent > LLM Router > Learned)
# Use alias to avoid conflict with routing_brain.RoutingDecision
try:
    from brains.routing import build_routing_plan
    from brains.routing import RoutingDecision as RoutingEngineDecision
    _routing_engine_available = True
    print("[INTEGRATOR] ✓ Routing engine loaded successfully")
except ImportError as e:
    print(f"[INTEGRATOR] Routing engine not available: {e}")
    _routing_engine_available = False
    build_routing_plan = None  # type: ignore
    RoutingEngineDecision = None  # type: ignore

# Learning mode for memory-first, LLM-as-teacher architecture
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"

# Teacher integration for learning cross-brain integration and coordination patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("integrator")
except Exception as e:
    print(f"[INTEGRATOR] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for conversation context awareness
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint,
        get_last_web_search_state,
        followup_refers_to_last_search,
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[INTEGRATOR] Continuation helpers not available: {e}")
    # Provide fallback stubs
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    get_last_web_search_state = lambda: None  # type: ignore
    followup_refers_to_last_search = lambda query: False  # type: ignore
    _continuation_helpers_available = False

# Agency routing patterns for direct tool execution
try:
    from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern
    from brains.cognitive.integrator.agency_executor import execute_agency_tool, format_agency_response
    _agency_routing_available = True
except Exception as e:
    print(f"[INTEGRATOR] Agency routing not available: {e}")
    _agency_routing_available = False

# Reinforcement Learning Routing Brain for learned route selection
try:
    from brains.cognitive.integrator.routing_brain import (
        RoutingBrain,
        RoutingDecision,
        RoutingFeedback,
        get_default_routing_brain,
    )
    _rl_routing_brain = get_default_routing_brain()
    _rl_routing_available = True
except Exception as e:
    print(f"[INTEGRATOR] RL Routing brain not available: {e}")
    _rl_routing_brain = None  # type: ignore
    _rl_routing_available = False

# Smart Routing (LLM-assisted classification with Teacher)
try:
    from brains.cognitive.integrator.smart_routing import (
        classify_intent,
        compute_routing_plan,
        apply_routing_feedback,
        detect_user_correction,
        get_smart_routing_decision,
    )
    from brains.cognitive.integrator.routing_intent import (
        PrimaryIntent,
        RoutingIntent,
        RoutingPlan,
    )
    _smart_routing_available = True
except Exception as e:
    print(f"[INTEGRATOR] Smart routing not available: {e}")
    _smart_routing_available = False
    classify_intent = None  # type: ignore
    compute_routing_plan = None  # type: ignore
    apply_routing_feedback = None  # type: ignore
    detect_user_correction = None  # type: ignore
    get_smart_routing_decision = None  # type: ignore

# Initialize pattern store
_pattern_store = get_pattern_store()

# Load initial patterns if not already present
try:
    initialize_integrator_patterns()
except Exception as e:
    print(f"[INTEGRATOR] Failed to initialize patterns: {e}")


class BrainBid:
    """Simple container representing a bid for attention from a brain.

    Attributes:
        brain_name: The canonical name of the brain submitting the bid.
        priority: A numeric value between 0.0 and 1.0 indicating
            urgency/confidence.  Higher values win when no overriding
            conditions are met.
        reason: A short string describing why the brain is bidding.
        evidence: Arbitrary additional data supporting the bid.
    """

    def __init__(self, brain_name: str, priority: float, reason: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        self.brain_name = str(brain_name)
        # Clamp priority to the [0.0, 1.0] range
        try:
            p = float(priority)
        except Exception:
            p = 0.0
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        self.priority = p
        self.reason = str(reason)
        self.evidence = evidence or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brain_name": self.brain_name,
            "priority": self.priority,
            "reason": self.reason,
            "evidence": self.evidence,
        }


class AttentionTransition:
    """Record of an attention state change.

    Each transition captures the previous and next brain focus, the
    reason for the change and any associated evidence.  These records
    are stored in ``AttentionState.history`` for auditing and may be
    used during self‑review phases to analyse how the integrator is
    allocating cognitive resources.
    """

    def __init__(self, previous: str, current: str, reason: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        self.previous_focus = previous
        self.current_focus = current
        self.reason = reason
        self.evidence = evidence or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_focus": self.previous_focus,
            "current_focus": self.current_focus,
            "reason": self.reason,
            "evidence": self.evidence,
        }


class AttentionState:
    """Encapsulate the current attention focus and its history.

    ``current_focus`` holds the name of the brain that presently has
    attention.  ``focus_strength`` is a float in [0.0, 1.0] that could
    represent how strongly this focus should be maintained – future
    enhancements may adjust this value based on domain confidence or
    urgency.  ``focus_reason`` stores the justification for the
    current focus.  ``competing_bids`` retains the latest set of
    processed bids.  ``history`` accumulates ``AttentionTransition``
    instances describing previous state changes.
    """

    def __init__(self) -> None:
        self.current_focus: str = ""
        self.focus_strength: float = 0.0
        self.focus_reason: str = ""
        self.competing_bids: List[BrainBid] = []
        self.history: List[AttentionTransition] = []

    def update(self, new_focus: str, reason: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        # Record transition if there is an existing focus
        if self.current_focus:
            self.history.append(AttentionTransition(self.current_focus, new_focus, reason, evidence))
        self.current_focus = new_focus
        self.focus_reason = reason
        self.focus_strength = 1.0  # Default to full focus strength for now
        # Preserve the evidence on competing bids for traceability
        if evidence:
            # Store in evidence of the latest transition for introspection
            pass


_STATE = AttentionState()

# Track which pattern was used for the current resolution
_current_pattern: Optional[Pattern] = None

# Direct agency tool info storage (bypasses Pattern when not needed)
# This avoids Pattern construction issues for simple tool routing
_current_agency_tool: Optional[Dict[str, Any]] = None


def _compute_signature(bids: List[BrainBid], context: Optional[Dict[str, Any]] = None) -> str:
    """
    Compute a routing signature from bids and context.

    This determines which pattern to use for routing decisions.
    """
    if not bids:
        return "default"

    # Extract reasons from bids
    reasons = set([b.reason for b in bids])

    # Check for special cases (highest priority)
    if "contradiction_detected" in reasons:
        return "contradiction_detected"
    if "unanswered_question" in reasons:
        return "unanswered_question"

    # Check for multi-brain conflicts
    if len(bids) > 3:
        return "multi_brain_conflict"

    # Check context for hints about input type
    if context:
        query = str(context.get("query", "")).lower()

        # Explicit execution routing: send action requests directly to the tool layer
        execution_patterns = {
            r"^(write|create|save).*file": "fs_tool",
            r"^(list|scan).*files": "fs_tool",
            r"^(run|execute).*python": "python_exec",
            r"^(run|execute).*code": "python_exec",
            r"^(git|commit|push|add).*": "git_tool",
            r"^(reload|hotload|reimport).*": "reload_tool",
        }
        for pattern, tool in execution_patterns.items():
            try:
                if re.match(pattern, query):
                    return f"tool_call:{tool}"
            except re.error:
                continue

        # Check if sensorium classified this as a follow-up question
        norm_type = context.get("norm_type", "")
        if norm_type == "follow_up_question":
            return "follow_up_question"

        # Self-diagnostic commands
        if "diagnose" in query or "self_diag" in query:
            return "self_diag:diagnose_web"

        # Research commands
        if "research" in query or "deep research" in query:
            return "research_command:deep"

        # Tool calls
        tool_phrases = [
            "scan your entire codebase",
            "scan codebase",
            "codebase scan",
            "write module",
            "commit changes",
            "git status",
            "reload your code",
            "make pull request",
        ]
        if "tool_call" in query or any(phrase in query for phrase in tool_phrases):
            return "tool_call:action_engine"

        # Question types
        if query.startswith("what is") or query.startswith("what are"):
            return "direct_question:what_is"
        if query.startswith("how to") or query.startswith("how do"):
            return "direct_question:how_to"

        # Casual statements (short, no question mark)
        if len(query) < 50 and "?" not in query:
            return "casual_statement"

    # Default fallback
    return "default"


def _compute_context_tags(bids: List[BrainBid], context: Optional[Dict[str, Any]] = None) -> List[str]:
    """Extract context tags from bids and context for pattern matching."""
    tags = []

    # Extract from reasons
    reasons = set([b.reason for b in bids])
    if "contradiction_detected" in reasons:
        tags.extend(["contradiction", "safety"])
    if "unanswered_question" in reasons:
        tags.extend(["unanswered", "safety"])

    # Extract from context
    if context:
        query = str(context.get("query", "")).lower()
        norm_type = context.get("norm_type", "")

        # Check for follow-up questions
        if norm_type == "follow_up_question":
            tags.extend(["follow_up", "context_dependent"])

        if "?" in query:
            tags.append("direct_question")
        if any(word in query for word in ["research", "investigate", "study"]):
            tags.append("research")
        if any(word in query for word in ["diagnose", "debug", "self"]):
            tags.append("self_diag")
        tool_words = ["scan", "codebase", "commit", "git", "reload", "tool"]
        if any(word in query for word in tool_words):
            tags.append("requires_tools")

    # Add based on bid count
    if len(bids) > 3:
        tags.append("complex")
    elif len(bids) <= 1:
        tags.append("simple")

    return tags


# Track current RL routing decision for feedback
_current_rl_decision: Optional[RoutingDecision] = None


def _resolve_attention(bids: List[BrainBid], context: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve the winning brain from a list of bids using learned patterns.

    Routing priority:
    -1. ROUTING ENGINE (unified layered routing - Grammar > Self-Intent > LLM Router)
    0. SENSORIUM ROUTING HINTS (system_capability/self_identity -> self_model, bypasses all)
    1. RL Routing Brain (learned from feedback, uses reinforcement learning)
    2. Pattern Store (signature-based patterns)
    3. Agency patterns (tool execution)
    4. Rule-based fallback

    The pattern/decision used is tracked globally so it can be updated after feedback.

    NOW WITH CONVERSATION AWARENESS:
    - Accesses conversation history to detect continuations
    - Adjusts routing based on previous topics and interactions
    """
    global _current_pattern, _current_rl_decision, _current_agency_tool

    # Reset agency tool at start of each resolution
    _current_agency_tool = None

    if not bids:
        return "language"

    # =========================================================================
    # PRIORITY -2: UNIFIED ROUTING ENGINE (NEW - Layered routing with hard floor)
    # =========================================================================
    # The routing engine provides layered routing with strict precedence:
    #   1. Grammar (hard floor) - "x grok hello" -> x tool
    #   2. Self-Intent Gate - "who are you" -> self_model
    #   3. LLM Router - JSON-only routing with confidence
    #   4. Learned patterns
    # Grammar matches are NEVER overridden by any other routing.
    # =========================================================================
    if _routing_engine_available and context:
        query = str(context.get("query", context.get("user_query", ""))).strip()
        if query:
            try:
                # Build capability snapshot from current capabilities
                from capabilities import get_capabilities
                caps = get_capabilities()
                capability_snapshot = {
                    "execution_mode": caps.get("execution_guard", {}).get("mode", "UNKNOWN"),
                    "web_research_enabled": caps.get("web_research", {}).get("enabled", False),
                    "tools_available": list(caps.get("tools", {}).keys()) if caps.get("tools") else [],
                }

                # Call the unified routing engine
                decision = build_routing_plan(
                    query=query,
                    capability_snapshot=capability_snapshot,
                    llm_router_enabled=False,  # Disable LLM router for now (use existing Teacher flow)
                    llm_confidence_threshold=0.75,
                )

                # If routing engine made a high-confidence decision, use it
                if decision and decision.confidence >= 0.9:
                    print(f"[INTEGRATOR] ✓ ROUTING ENGINE: '{query[:40]}...' -> {decision.brains}")
                    print(f"[INTEGRATOR]   Source: {decision.source}, Confidence: {decision.confidence}")

                    # Store routing info in context for downstream use
                    if isinstance(context, dict):
                        context["routing_engine_decision"] = True
                        context["routing_source"] = decision.source
                        context["routing_intent"] = decision.intent
                        context["grammar_tools"] = decision.tools
                        context["grammar_subcommand"] = decision.subcommand
                        context["grammar_args"] = decision.args
                        context["grammar_metadata"] = decision.metadata
                        context["bypass_teacher"] = decision.bypass_teacher

                        # Set agency tool if tools specified
                        if decision.tools:
                            context["agency_tool"] = decision.tools[0]
                            _current_agency_tool = decision.tools[0]

                    # Return the first brain
                    if decision.brains:
                        return decision.brains[0]
                    return "language"

            except Exception as e:
                print(f"[INTEGRATOR] Routing engine error: {e}, continuing to fallback routing")

    # =========================================================================
    # PRIORITY -1: COMMAND PRE-PARSER (HARD GRAMMAR - NEVER OVERRIDDEN)
    # =========================================================================
    # The command pre-parser provides a deterministic "hard floor" that routes
    # explicit tool commands BEFORE any learned patterns or LLM routing.
    # This guarantees correct routing for commands like:
    #   - "x grok hello" -> browser tool x, subcommand grok
    #   - "research: topic" -> research_manager
    #   - "use grok tool: message" -> explicit tool call
    # These MUST NOT be overridden by integrator, even if patterns disagree.
    # =========================================================================
    if context:
        query = str(context.get("query", context.get("user_query", ""))).strip()
        if query:
            try:
                from brains.cognitive.command_pre_parser import (
                    parse_command,
                    get_routing_plan,
                    log_gold_routing,
                )

                parsed = parse_command(query)
                if parsed:
                    # Get routing plan from parsed command
                    plan = get_routing_plan(parsed)
                    print(f"[INTEGRATOR] ✓ COMMAND PRE-PARSER: '{query[:50]}...' -> {parsed.intent}")
                    print(f"[INTEGRATOR]   Pattern: {parsed.matched_pattern}, Tools: {parsed.tools}, Brains: {parsed.brains}")

                    # Store routing info in context for downstream use
                    if isinstance(context, dict):
                        context["grammar_routing"] = True
                        context["grammar_intent"] = parsed.intent
                        context["grammar_tools"] = parsed.tools
                        context["grammar_subcommand"] = parsed.subcommand
                        context["grammar_args"] = parsed.args
                        context["grammar_metadata"] = parsed.metadata
                        context["bypass_teacher"] = True  # Don't let Teacher override

                        # Set agency tool if tools specified
                        if parsed.tools:
                            context["agency_tool"] = parsed.tools[0]
                            _current_agency_tool = parsed.tools[0]

                    # Log as gold routing for learning
                    try:
                        gold_example = log_gold_routing(parsed)
                        # Could store this for routing pattern learning
                    except Exception:
                        pass

                    # Return the first brain from the routing plan
                    if parsed.brains:
                        return parsed.brains[0]
                    return "language"

            except ImportError:
                print("[INTEGRATOR] Command pre-parser not available, continuing to other routing")
            except Exception as e:
                print(f"[INTEGRATOR] Command pre-parser error: {e}, continuing to other routing")

    # =========================================================================
    # PRIORITY 0: SENSORIUM ROUTING HINTS (HIGHEST PRIORITY - BYPASSES ALL)
    # =========================================================================
    # When sensorium detects system_capability or self_identity intent,
    # it sets routing_target = "self_model" in the context. We MUST respect
    # this and route directly to self_model, bypassing Teacher entirely.
    # This prevents Teacher from answering with Apache Maven / Java garbage.
    # =========================================================================
    if context:
        routing_target = context.get("routing_target", "")
        routing_reason = context.get("routing_reason", "")
        intent_kind = context.get("intent_kind", "")

        # Check for explicit routing hint from sensorium
        if routing_target == "self_model":
            if routing_reason in ("system_capability_query", "self_identity_query",
                                  "llm_detected_capability_query", "llm_detected_self_identity"):
                print(f"[INTEGRATOR] ✓ SENSORIUM OVERRIDE: Routing to self_model ({routing_reason})")
                return "self_model"

        # Also check intent_kind directly (fallback if routing_target not set)
        if intent_kind in ("system_capability", "self_identity"):
            print(f"[INTEGRATOR] ✓ INTENT OVERRIDE: Routing to self_model (intent={intent_kind})")
            return "self_model"

        # Check for semantic normalization results
        semantic_info = context.get("semantic", {})
        if isinstance(semantic_info, dict):
            semantic_intent = semantic_info.get("intent_kind", "")
            if semantic_intent in ("system_capability", "self_identity"):
                print(f"[INTEGRATOR] ✓ SEMANTIC OVERRIDE: Routing to self_model (semantic_intent={semantic_intent})")
                return "self_model"

        # =====================================================================
        # PRIORITY 0.25: DIRECT SHELL/GIT COMMAND DETECTION (HIGHEST TOOL PRIORITY)
        # =====================================================================
        # Direct shell commands MUST route to action_engine, not reasoning.
        # This check happens BEFORE brain strategy checks to prevent reasoning
        # from claiming it can handle commands like "git add ." or "git push".
        # =====================================================================
        query = str(context.get("query", context.get("user_query", ""))).strip()
        query_lower = query.lower()

        # Direct shell command prefixes that should ALWAYS route to execution
        shell_prefixes = (
            "git ", "git add", "git commit", "git push", "git pull", "git status",
            "git log", "git diff", "git branch", "git checkout", "git clone",
            "ls ", "ls", "cd ", "mkdir ", "rm ", "cp ", "mv ", "cat ", "echo ",
            "pwd", "pip ", "python ", "npm ", "yarn ", "make ", "cargo ",
            "docker ", "kubectl ", "curl ", "wget ", "dir", "dir ",
        )

        # Check for exact single-word commands (ls, pwd, dir, etc.)
        single_word_commands = {"ls", "pwd", "dir", "whoami", "hostname", "date", "uptime"}

        if query_lower.startswith(shell_prefixes) or query_lower in single_word_commands:
            print(f"[INTEGRATOR] ✓ DIRECT SHELL COMMAND DETECTED: '{query}' -> action_engine")
            # Store the command for execution
            if isinstance(context, dict):
                context["agency_tool"] = "shell_tool"
                context["shell_command"] = query
                context["agency_args"] = {"command": query}  # For agency_executor
                context["bypass_teacher"] = True
            return "action_engine"

        # =====================================================================
        # PRIORITY 0.3: LLM INTENT HINTS (from router-teacher)
        # =====================================================================
        # The router-teacher LLM classifies intent for weird/typo-heavy input.
        # These hints allow hard-routing without more hard-coded patterns.
        # This replaces the typo whack-a-mole with LLM-based normalization.
        # =====================================================================
        llm_intent_hints = context.get("llm_intent_hints", {})
        if llm_intent_hints:
            # Check for web search intent from LLM
            if llm_intent_hints.get("is_web_search", False):
                print(f"[INTEGRATOR] ✓ LLM INTENT HINT: Routing to research_manager (web_search)")
                return "research_manager"

            # Check for capability query from LLM
            if llm_intent_hints.get("is_capability_query", False):
                print(f"[INTEGRATOR] ✓ LLM INTENT HINT: Routing to self_model (capability_query)")
                return "self_model"

            # Check for self-identity from LLM
            if llm_intent_hints.get("is_self_identity", False):
                print(f"[INTEGRATOR] ✓ LLM INTENT HINT: Routing to self_model (self_identity)")
                return "self_model"

            # Check for follow-up from LLM
            if llm_intent_hints.get("is_follow_up", False):
                print(f"[INTEGRATOR] LLM detected follow-up intent")
                # Fall through to normal routing but with context

            # Check for greeting from LLM
            if llm_intent_hints.get("is_greeting", False):
                print(f"[INTEGRATOR] ✓ LLM INTENT HINT: Routing to language (greeting)")
                return "language"

            # NOTE: Time/date/calendar queries are handled in Priority 0.4 below
            # using the existing time_query detection from sensorium

    # HISTORY ACCESS: Get conversation context to inform routing decisions
    conv_context = {}
    if _continuation_helpers_available:
        try:
            conv_context = get_conversation_context()
            if conv_context:
                print(f"[INTEGRATOR] Conversation context available: {list(conv_context.keys())}")
        except Exception as e:
            print(f"[INTEGRATOR] Failed to get conversation context: {e}")

    # =========================================================================
    # MEMORY-FIRST CHECK: Check if any brain already has a strategy for this
    # =========================================================================
    # Before making a routing decision, check if brains report having a
    # high-confidence strategy or cached answer. This implements the
    # "brains that already know" optimization.
    #
    # STRATEGY PREFERENCE RULES:
    # - HIGH_CONFIDENCE_THRESHOLD (0.7): Immediate return to that brain
    # - MODERATE_CONFIDENCE_THRESHOLD (0.5): Track as candidate, boost priority
    # - Below 0.5: Strategy exists but not reliable enough
    #
    # Brains with known strategies are ALWAYS preferred over Language fallback.
    # Language only handles queries when NO brain has a relevant strategy.
    # =========================================================================
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MODERATE_CONFIDENCE_THRESHOLD = 0.5

    # Track brains with strategies for boosted routing
    brains_with_strategies = {}  # brain_name -> (strategy, confidence, problem_type)

    if context:
        query = str(context.get("query", context.get("user_query", "")))
        learning_mode = context.get("learning_mode", LearningMode.TRAINING)

        # Check if reasoning brain has a strategy
        try:
            from brains.cognitive.reasoning.service.reasoning_brain import (
                select_strategy,
                classify_reasoning_problem,
                load_strategies_from_lessons
            )
            # Load strategies if not already loaded
            load_strategies_from_lessons(context)
            # Check if we have a strategy for this query type
            problem_type = classify_reasoning_problem(context)
            strategy = select_strategy(problem_type)
            if strategy:
                confidence = strategy.get("confidence", 0)
                if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                    print(f"[INTEGRATOR] ✓ HIGH-CONFIDENCE reasoning strategy (no teacher call) for: {problem_type}")
                    return "reasoning"
                elif confidence >= MODERATE_CONFIDENCE_THRESHOLD:
                    brains_with_strategies["reasoning"] = (strategy, confidence, problem_type)
                    print(f"[INTEGRATOR] Found moderate-confidence reasoning strategy for: {problem_type} (conf={confidence:.2f})")
        except Exception:
            pass  # Reasoning brain strategies not available

        # Check if planner has a strategy
        try:
            from brains.cognitive.planner.service.planner_brain import (
                select_planner_strategy,
                classify_planning_problem,
                load_planner_strategies_from_lessons
            )
            load_planner_strategies_from_lessons(context)
            problem_type = classify_planning_problem(context)
            strategy = select_planner_strategy(problem_type)
            if strategy:
                confidence = strategy.get("confidence", 0)
                if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                    print(f"[INTEGRATOR] ✓ HIGH-CONFIDENCE planner strategy (no teacher call) for: {problem_type}")
                    return "planner"
                elif confidence >= MODERATE_CONFIDENCE_THRESHOLD:
                    brains_with_strategies["planner"] = (strategy, confidence, problem_type)
                    print(f"[INTEGRATOR] Found moderate-confidence planner strategy for: {problem_type} (conf={confidence:.2f})")
        except Exception:
            pass  # Planner strategies not available

        # Check if Research Manager has learned patterns for this query
        try:
            from brains.memory.brain_memory import BrainMemory as ResearchBrainMemory
            research_memory = ResearchBrainMemory("research_manager")

            # Check for learned research patterns
            research_patterns = research_memory.retrieve(query=query, limit=5)
            for pattern in research_patterns:
                metadata = pattern.get("metadata", {}) or {}
                if metadata.get("kind") == "learned_pattern":
                    confidence = metadata.get("confidence", 0)
                    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                        print(f"[INTEGRATOR] ✓ HIGH-CONFIDENCE research pattern found")
                        return "research_manager"
                    elif confidence >= MODERATE_CONFIDENCE_THRESHOLD:
                        brains_with_strategies["research_manager"] = (pattern, confidence, "research")
                        print(f"[INTEGRATOR] Found moderate-confidence research pattern (conf={confidence:.2f})")
                        break
        except Exception:
            pass  # Research manager memory not available

        # =====================================================================
        # MODERATE CONFIDENCE ROUTING: If any brain has a moderate-confidence
        # strategy, prefer it over Language fallback
        # =====================================================================
        if brains_with_strategies:
            # Choose the brain with highest confidence strategy
            best_brain = max(brains_with_strategies.items(), key=lambda x: x[1][1])
            brain_name, (strategy, confidence, problem_type) = best_brain
            print(f"[INTEGRATOR] ✓ Using best moderate-confidence strategy: {brain_name} for {problem_type} (conf={confidence:.2f})")
            return brain_name

    # Step 1: Compute signature and context tags
    signature = _compute_signature(bids, context)
    context_tags = _compute_context_tags(bids, context)

    # Enhance context tags based on conversation history
    if conv_context:
        last_topic = conv_context.get("last_topic", "")
        if last_topic:
            context_tags.append("has_history")

    print(f"[INTEGRATOR] Routing signature: {signature}, tags: {context_tags}")

    # ===== PRIORITY 0.4: TIME QUERY ROUTING (direct tool, NO Teacher) =====
    # Time questions MUST route to time_now tool, never to Teacher.
    # The LLM cannot provide accurate real-time information.
    #
    # Now also checks LLM intent hints for time/date queries that might have
    # been missed by local pattern matching (e.g., "what is to day" typo).
    try:
        from brains.cognitive.sensorium.semantic_normalizer import is_time_query, get_time_query_type
        from capabilities import is_capability_enabled

        if context:
            query = str(context.get("query", context.get("user_query", "")))

            # Check LLM intent hints first (catches typos like "what is to day")
            llm_hints = context.get("llm_intent_hints", {})
            is_llm_time_query = (
                llm_hints.get("is_time_query", False) or
                llm_hints.get("is_date_or_day_query", False) or  # New preferred flag
                llm_hints.get("is_date_query", False) or  # Legacy flag
                llm_hints.get("is_calendar_query", False)
            )

            if query and (is_time_query(query) or is_llm_time_query):
                # Check if time capability is available
                if is_capability_enabled("time"):
                    # Determine query type (time, date, or calendar)
                    # Priority: sensorium's time_query_type > local detection > LLM hints

                    # First check if sensorium already determined the type
                    query_type = context.get("time_query_type")

                    # Fall back to local detection
                    if not query_type:
                        query_type = get_time_query_type(query)

                    # Fall back to LLM hints
                    if not query_type and is_llm_time_query:
                        # Use LLM hints to determine query type
                        if llm_hints.get("is_calendar_query", False):
                            query_type = "calendar"
                        elif llm_hints.get("is_date_or_day_query", False) or llm_hints.get("is_date_query", False):
                            query_type = "date"  # Date/day queries use GET_DATE
                        elif llm_hints.get("is_time_query", False):
                            query_type = "time"
                        else:
                            query_type = "time"  # Default to time
                        print(f"[INTEGRATOR] Query type from LLM hints: {query_type}")
                    elif not query_type:
                        query_type = "time"  # Default fallback
                    method_map = {
                        "time": "GET_TIME",
                        "date": "GET_DATE",
                        "calendar": "GET_CALENDAR",
                    }
                    method = method_map.get(query_type, "GET_TIME")
                    print(f"[INTEGRATOR] ✓ TIME QUERY DETECTED: '{query}' -> time_now:{method} (bypassing Teacher)")
                    # Store agency tool info directly (no Pattern needed)
                    _current_agency_tool = {
                        "tool": "time_now",
                        "method": method,
                        "args": {"query_type": query_type},  # Pass query type for response formatting
                        "bypass_teacher": True,  # CRITICAL: Never call Teacher for time
                    }
                    # Mark context so downstream knows to use the tool
                    if isinstance(context, dict):
                        context["agency_tool"] = "time_now"
                        context["agency_method"] = method  # CRITICAL: Pass method (GET_TIME/GET_DATE/GET_CALENDAR)
                        context["agency_args"] = {"query_type": query_type}  # Pass args for response formatting
                        context["bypass_teacher"] = True
                        context["time_query_type"] = query_type  # Pass query type to downstream
                    # Route to language brain which will check agency_tool
                    return "language"
                else:
                    print(f"[INTEGRATOR] Time capability not available, routing to language with apology")
                    return "language"
    except Exception as e:
        print(f"[INTEGRATOR] Time query check failed: {e}")

    # ===== PRIORITY 0.5: SMART ROUTING (LLM-assisted classification) =====
    # Smart routing provides robust intent classification that's resilient to phrasing
    if _smart_routing_available and context:
        try:
            query = str(context.get("query", ""))
            if query:
                # Build context bundle for smart routing
                context_bundle = {
                    "recent_turns": context.get("history", [])[-4:],
                    "project": context.get("project", ""),
                    "episode_id": context.get("episode_id", ""),
                }

                # Get smart routing decision
                smart_result = get_smart_routing_decision(query, context)

                if smart_result:
                    intent = smart_result.get("intent", "")
                    brains = smart_result.get("brains", [])
                    source = smart_result.get("source", "")

                    # Self-intent questions MUST route to self_model
                    if intent in ("capability_question", "self_question", "history_question"):
                        print(f"[INTEGRATOR] ✓ SMART ROUTING: Self-intent '{intent}' -> self_model")
                        return "self_model"

                    # Use first brain from smart routing if high confidence
                    if brains and source in ("teacher", "self_intent_gate"):
                        chosen = brains[0]
                        print(f"[INTEGRATOR] ✓ SMART ROUTING: {intent} -> {chosen} (source={source})")
                        return chosen

                    # For other sources, continue to RL routing for potential override
                    print(f"[INTEGRATOR] Smart routing suggests: {brains}, continuing to RL...")

        except Exception as e:
            print(f"[INTEGRATOR] Smart routing failed: {e}")

    # ===== PRIORITY 0.6: WEB SEARCH FOLLOW-UP ROUTING =====
    # When a user asks a follow-up question after a web search (like "tell me more
    # about music" after "web search music"), route to research_manager in followup
    # mode instead of generic reasoning. This ensures follow-ups use the actual
    # SERP data rather than falling back to generic LLM answers.
    if _continuation_helpers_available and context:
        query = str(context.get("query", context.get("user_query", "")))
        if query and is_continuation(query, context):
            # Check if this follow-up refers to a recent web search
            if followup_refers_to_last_search(query):
                last_web = get_last_web_search_state()
                if last_web:
                    print(f"[INTEGRATOR] ✓ WEB SEARCH FOLLOW-UP: '{query}' references last web search '{last_web.get('query', '')}' -> research_manager")
                    # Store context hint for research_manager to use web_followup mode
                    if isinstance(context, dict):
                        context["web_followup_mode"] = True
                        context["last_web_search"] = last_web
                    return "research_manager"

    # ===== PRIORITY 1: RL ROUTING BRAIN (learned from feedback) =====
    # Try reinforcement learning-based routing first
    if _rl_routing_available and _rl_routing_brain and context:
        try:
            query = str(context.get("query", ""))
            if query:
                rl_decision = _rl_routing_brain.route(query, top_n=1)
                _current_rl_decision = rl_decision

                if rl_decision.chosen_routes:
                    chosen_route = rl_decision.chosen_routes[0]
                    score = rl_decision.scores.get(chosen_route, 0)

                    # Map route IDs to brain names
                    route_to_brain = {
                        "language": "language",
                        "technology": "teacher",  # technology questions -> teacher
                        "factual": "teacher",     # factual questions -> teacher
                        "research_reports": "research_manager",  # matches default route ID
                        "fs_tool": "action_engine",
                        "git_tool": "action_engine",
                        "self_diag": "self_model",
                        "pattern_coder": "coder",
                    }

                    brain_name = route_to_brain.get(chosen_route, chosen_route)

                    # Only use RL routing if score is reasonably high
                    if score > 0.3:
                        print(f"[INTEGRATOR] RL routing: {chosen_route} -> {brain_name} (score={score:.3f})")
                        return brain_name
                    else:
                        print(f"[INTEGRATOR] RL routing score too low ({score:.3f}), trying other methods")
        except Exception as e:
            print(f"[INTEGRATOR] RL routing failed: {e}")

    if isinstance(signature, str) and signature.startswith("tool_call:"):
        tool_name = signature.split("tool_call:", 1)[1]
        capability_map = {
            "fs_tool": "filesystem_agency",
            "git_tool": "git_agency",
            "exec_tool": "execution_agency",
        }
        cap_key = capability_map.get(tool_name)
        if cap_key:
            cap_state = get_capabilities().get(cap_key, {})
            if cap_state.get("available") and cap_state.get("enabled"):
                print(f"[INTEGRATOR] Capability {cap_key} enabled, routing to action_engine")
                return "action_engine"
            if cap_state.get("available") and not cap_state.get("enabled"):
                reason = cap_state.get("reason") or "disabled"
                print(f"[INTEGRATOR] Capability {cap_key} disabled: {reason}")
                return "language"
            print(f"[INTEGRATOR] Capability {cap_key} unavailable, routing to teacher as documentation-only fallback")
            return "teacher"

    # Step 1.5: Check for agency tool patterns (bypass Teacher for direct tool execution)
    if _agency_routing_available and context:
        query = str(context.get("query", ""))
        if query:
            agency_match = match_agency_pattern(query, threshold=0.7)
            if agency_match:
                print(f"[INTEGRATOR] Agency pattern matched: {agency_match['tool']} (confidence: {agency_match['confidence']})")
                # Store agency match in global pattern for later execution
                # Note: _current_pattern is already declared global at function start (line 398)
                _current_pattern = Pattern(
                    brain="integrator",
                    signature=f"agency_tool:{agency_match['tool']}",
                    action={
                        "agency_tool": agency_match['tool'],
                        "agency_method": agency_match.get('method'),
                        "agency_args": agency_match.get('args'),
                        "bypass_teacher": agency_match.get('bypass_teacher', True)
                    },
                    score=agency_match['confidence']
                )
                # Route to action_engine which should execute the agency tool
                return "action_engine"

    # Step 2: Try to find a learned pattern (only if no agency pattern matched)
    pattern = _pattern_store.get_best_pattern(
        brain="integrator",
        signature=signature,
        context_tags=context_tags,
        score_threshold=0.0  # Accept any pattern above 0
    )

    # Track which pattern we're using for later updates (if not already set by agency routing)
    if not _current_pattern:
        _current_pattern = pattern

    # Step 3: If we have a good pattern, use it
    if pattern and pattern.score > 0.0:
        # For now, we just use the pattern to guide brain selection
        # The actual pipeline execution happens elsewhere
        # So we return the first brain from the pattern's pipeline
        pipeline = pattern.action.get("pipeline", [])
        if pipeline:
            # Map pipeline names to brain names (simplified)
            brain_map = {
                "REASONING": "reasoning",
                "LANGUAGE": "language",
                "PLANNER": "planner",
                "SELF_MODEL": "self_model",
                "ACTION_ENGINE": "action_engine"
            }
            for brain_name in pipeline:
                if brain_name in brain_map:
                    result = brain_map[brain_name]
                    print(f"[INTEGRATOR] Using learned pattern: {signature} -> {result}")
                    return result

    # Step 4: Fall back to rule-based logic if no pattern or pattern failed
    print(f"[INTEGRATOR] No good pattern found, using rule-based fallback")

    # Rule 1: contradictions override everything
    for b in bids:
        if b.reason == "contradiction_detected":
            return b.brain_name

    # Rule 2: unanswered questions override remaining
    for b in bids:
        if b.reason == "unanswered_question":
            return b.brain_name

    # Rule 3: highest priority wins
    winner = max(bids, key=lambda x: x.priority)
    return winner.brain_name


def get_current_pattern() -> Optional[Pattern]:
    """
    Get the current routing pattern (for downstream execution).

    Returns:
        The current pattern if one was used, None otherwise.
    """
    return _current_pattern


def get_agency_tool_info() -> Optional[Dict[str, Any]]:
    """
    Get agency tool execution info if an agency tool was matched.

    Returns:
        Dict with tool, method, args, and bypass_teacher flag if matched, None otherwise.
    """
    # Check direct agency tool storage first (used for time_now and other simple tools)
    if _current_agency_tool:
        return _current_agency_tool

    # Fall back to pattern-based agency tools
    if _current_pattern and _current_pattern.action.get("agency_tool"):
        return {
            "tool": _current_pattern.action.get("agency_tool"),
            "method": _current_pattern.action.get("agency_method"),
            "args": _current_pattern.action.get("agency_args"),
            "bypass_teacher": _current_pattern.action.get("bypass_teacher", True)
        }
    return None


def review_routing_decision(
    user_query: str,
    actual_route: str,
    response_used_tool: Optional[str] = None,
    llm_intent_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Review a routing decision after the response is produced.

    This implements the "router-review" path for self-correction:
    - Check if the query looked like a time/date query but time tool wasn't used
    - Check if the answer contradicts tool capabilities
    - Flag potential routing mistakes for learning

    Args:
        user_query: The original user query
        actual_route: The brain that actually handled the query
        response_used_tool: The tool that was used (if any)
        llm_intent_hints: Intent hints from router-teacher (if available)

    Returns:
        Dict with:
        - is_correct: bool - whether routing was likely correct
        - suggested_route: str - what route should have been used (if different)
        - reason: str - explanation
        - should_learn: bool - whether to store a new routing pattern
    """
    result = {
        "is_correct": True,
        "suggested_route": None,
        "reason": "routing_ok",
        "should_learn": False,
    }

    if not user_query:
        return result

    # Check if LLM hints suggest a different route
    if llm_intent_hints:
        # Time/date queries should use time_now tool
        if (llm_intent_hints.get("is_time_query", False) or
            llm_intent_hints.get("is_date_query", False) or
            llm_intent_hints.get("is_calendar_query", False)):
            if response_used_tool != "time_now":
                result["is_correct"] = False
                result["suggested_route"] = "time_now_tool"
                result["reason"] = "time_query_should_use_time_tool"
                result["should_learn"] = True
                print(f"[ROUTER_REVIEW] ⚠ Query '{user_query[:30]}...' looks like time query "
                      f"but used {response_used_tool or actual_route} instead of time_now")

        # Capability queries should route to self_model
        if llm_intent_hints.get("is_capability_query", False):
            if actual_route not in ("self_model", "self_dmn"):
                result["is_correct"] = False
                result["suggested_route"] = "self_model"
                result["reason"] = "capability_query_should_use_self_model"
                result["should_learn"] = True
                print(f"[ROUTER_REVIEW] ⚠ Query '{user_query[:30]}...' looks like capability query "
                      f"but routed to {actual_route} instead of self_model")

        # Web search queries should route to research_manager
        if llm_intent_hints.get("is_web_search", False):
            if actual_route != "research_manager" and response_used_tool != "web_search":
                result["is_correct"] = False
                result["suggested_route"] = "research_manager"
                result["reason"] = "web_search_should_use_research_manager"
                result["should_learn"] = True
                print(f"[ROUTER_REVIEW] ⚠ Query '{user_query[:30]}...' looks like web search "
                      f"but routed to {actual_route} instead of research_manager")

    # Also check using local pattern matching
    try:
        from brains.cognitive.sensorium.semantic_normalizer import is_time_query
        if is_time_query(user_query) and response_used_tool != "time_now":
            result["is_correct"] = False
            result["suggested_route"] = "time_now_tool"
            result["reason"] = "pattern_detected_time_query"
            result["should_learn"] = True
            print(f"[ROUTER_REVIEW] ⚠ Pattern detected time query '{user_query[:30]}...' "
                  f"but used {response_used_tool or actual_route}")
    except Exception:
        pass

    if result["is_correct"]:
        print(f"[ROUTER_REVIEW] ✓ Routing appears correct: {user_query[:30]}... -> {actual_route}")

    return result


def update_from_verdict(verdict: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Update the current pattern's score based on SELF_REVIEW/Teacher verdict.

    This is the learning mechanism: after each interaction, we get feedback
    and use it to adjust pattern scores.

    Args:
        verdict: One of 'ok', 'minor_issue', 'major_issue'
        metadata: Optional metadata (not used by INTEGRATOR, but kept for API consistency)
    """
    global _current_pattern, _current_rl_decision

    updated_something = False

    # Update smart routing (LLM-assisted) with feedback
    if _smart_routing_available and apply_routing_feedback:
        try:
            # Check for user corrections in metadata
            user_correction = None
            if metadata:
                user_text = metadata.get("user_text", "")
                if user_text and detect_user_correction and detect_user_correction(user_text):
                    user_correction = user_text
                    print(f"[INTEGRATOR] Detected user correction in feedback")

            apply_routing_feedback(
                verdict=verdict,
                user_correction=user_correction,
                metadata=metadata,
            )
            updated_something = True
        except Exception as e:
            print(f"[INTEGRATOR] Failed to update smart routing: {e}")

    # Update RL routing brain if a decision was made
    if _rl_routing_available and _rl_routing_brain and _current_rl_decision:
        try:
            for route_id in _current_rl_decision.chosen_routes:
                feedback = RoutingFeedback(
                    decision_id=_current_rl_decision.decision_id,
                    route_id=route_id,
                    verdict=verdict,
                    metadata=metadata or {}
                )
                _rl_routing_brain.apply_feedback(feedback)
                print(f"[INTEGRATOR] Updated RL route {route_id} with verdict={verdict}")
                updated_something = True
        except Exception as e:
            print(f"[INTEGRATOR] Failed to update RL routing: {e}")

    # Update pattern store if a pattern was used
    if _current_pattern:
        # Convert verdict to reward signal
        reward = verdict_to_reward(verdict)

        # Update pattern score
        _pattern_store.update_pattern_score(
            pattern=_current_pattern,
            reward=reward,
            alpha=0.85  # Learning rate: 0.85 = slower, more stable
        )

        print(f"[INTEGRATOR] Updated pattern {_current_pattern.signature} "
              f"based on verdict={verdict} (reward={reward:+.1f})")
        updated_something = True

    if not updated_something:
        print("[INTEGRATOR] No routing decision to update")


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the integrator brain.

    The ``msg`` must contain an ``op`` field specifying the operation
    and may include a ``payload`` field with additional data.  Only
    the ``RESOLVE`` operation is currently supported.  The payload
    should include a ``bids`` key containing a list of bid dictionaries.

    Returns a dictionary with ``ok`` status and, when resolving,
    the ``focus`` key indicating the chosen brain.  The current
    attention state (including competing bids and history) is also
    returned for traceability.
    """
    try:
        op = (msg.get("op") or "").upper()
    except Exception:
        op = ""
    payload: Dict[str, Any] = msg.get("payload") or {}
    mid = msg.get("mid") or "UNKNOWN"
    if op == "RESOLVE":
        # Build bid objects from raw payload entries
        raw_bids = payload.get("bids") or []
        bids: List[BrainBid] = []
        for rb in raw_bids:
            try:
                bids.append(BrainBid(
                    brain_name=rb.get("brain_name"),
                    priority=rb.get("priority", 0.0),
                    reason=rb.get("reason", ""),
                    evidence=rb.get("evidence", {}),
                ))
            except Exception:
                continue
        # Step‑3 & 4 enhancements: incorporate motivation weights into attention
        motivation_weights: Dict[str, float] = {}
        try:
            from brains.cognitive.motivation.service.motivation_brain import service_api as motivation_api  # type: ignore
            query = str(payload.get("query", ""))
            context = payload.get("context", {})

            resp = motivation_api({
                "op": "EVALUATE_QUERY",
                "payload": {"query": query, "context": context}
            })
            if resp.get("ok"):
                motivation_weights = (resp.get("payload") or {}).get("weights", {})
        except Exception:
            pass

        # Apply motivation-weighted priority adjustments
        if motivation_weights:
            for b in bids:
                brain_name = str(b.brain_name).lower()
                try:
                    weight_boost = 0.0

                    if brain_name == "language":
                        weight_boost = motivation_weights.get("helpfulness", 0.8) * 0.1
                    elif brain_name == "reasoning":
                        weight_boost = (
                            motivation_weights.get("truthfulness", 0.9) * 0.1 +
                            motivation_weights.get("curiosity", 0.5) * 0.05
                        )
                    elif brain_name == "planner":
                        weight_boost = motivation_weights.get("self_improvement", 0.5) * 0.1

                    if weight_boost > 0:
                        b.priority = min(1.0, float(b.priority) + weight_boost)
                        b.evidence = b.evidence or {}
                        b.evidence["motivation_boost"] = weight_boost
                        b.evidence["motivation_weights"] = motivation_weights
                except Exception:
                    pass

        # Apply attention nudge if enabled
        nudge_enabled = False
        try:
            from api.utils import CFG  # type: ignore
            nudge_enabled = bool((CFG.get("wm", {}) or {}).get("nudge", False))
        except Exception:
            pass
        if nudge_enabled:
            for b in bids:
                if b.brain_name in ("reasoning", "planner"):
                    try:
                        b.priority = min(1.0, float(b.priority) + 0.05)
                        b.evidence = b.evidence or {}
                        b.evidence["nudge"] = "wm_overlap"
                    except Exception:
                        pass

        # Query the motivation brain for overall drive to scale all priorities
        drive = 0.0
        try:
            from brains.cognitive.motivation.service.motivation_brain import service_api as motivation_api  # type: ignore
            resp = motivation_api({"op": "SCORE_DRIVE", "payload": {"context": context or {}}})
            drive = float((resp.get("payload") or {}).get("drive", 0.0))
        except Exception:
            drive = 0.0
        if drive:
            for b in bids:
                try:
                    b.priority = min(1.0, float(b.priority) * (1.0 + 0.2 * drive))
                    b.evidence = b.evidence or {}
                    b.evidence["drive_scaling"] = drive
                except Exception:
                    pass
        # Resolve the winning brain after adjustments
        _STATE.competing_bids = bids

        # Pass context to resolution for signature computation
        # Include norm_type from sensorium if available
        resolution_context = {
            "query": payload.get("query", ""),
            "context": context,
            "norm_type": context.get("norm_type", "") if context else ""
        }
        focus = _resolve_attention(bids, context=resolution_context)
        _STATE.update(focus, reason="resolved_attention", evidence={"bids": [b.to_dict() for b in bids]})

        # Extract web_followup context if set during routing.
        # Note: _resolve_attention sets web_followup_mode directly on resolution_context,
        # not inside resolution_context["context"], so we check at the top level.
        web_followup_mode = resolution_context.get("web_followup_mode", False)
        last_web_search = resolution_context.get("last_web_search") if web_followup_mode else None

        # Extract agency_tool context if set during routing (for time_now, etc.)
        agency_tool = resolution_context.get("agency_tool")
        agency_method = resolution_context.get("agency_method")
        agency_args = resolution_context.get("agency_args", {})
        bypass_teacher = resolution_context.get("bypass_teacher", False)
        grammar_metadata = resolution_context.get("grammar_metadata", {})
        routing_source = resolution_context.get("routing_source", "")

        return {
            "ok": True,
            "mid": mid,
            "payload": {
                "focus": focus,
                "web_followup_mode": web_followup_mode,
                "last_web_search": last_web_search,
                "agency_tool": agency_tool,
                "agency_method": agency_method,
                "agency_args": agency_args,
                "bypass_teacher": bypass_teacher,
                "grammar_metadata": grammar_metadata,
                "routing_source": routing_source,
                "state": {
                    "current_focus": _STATE.current_focus,
                    "focus_strength": _STATE.focus_strength,
                    "focus_reason": _STATE.focus_reason,
                    "competing_bids": [b.to_dict() for b in _STATE.competing_bids],
                    "history": [t.to_dict() for t in _STATE.history],
                },
            },
        }
    elif op == "STATE":
        # Return the current attention state without modifications
        return {
            "ok": True,
            "mid": mid,
            "payload": {
                "state": {
                    "current_focus": _STATE.current_focus,
                    "focus_strength": _STATE.focus_strength,
                    "focus_reason": _STATE.focus_reason,
                    "competing_bids": [b.to_dict() for b in _STATE.competing_bids],
                    "history": [t.to_dict() for t in _STATE.history],
                }
            },
        }
    elif op == "REVIEW_ROUTING":
        # Review a routing decision after response is produced (router-review path)
        # This enables self-correction and learning from routing mistakes
        user_query = payload.get("user_query", "")
        actual_route = payload.get("actual_route", "")
        response_used_tool = payload.get("response_used_tool")
        llm_intent_hints = payload.get("llm_intent_hints", {})

        review_result = review_routing_decision(
            user_query=user_query,
            actual_route=actual_route,
            response_used_tool=response_used_tool,
            llm_intent_hints=llm_intent_hints,
        )

        # If routing was wrong and we should learn, store a pattern
        if review_result.get("should_learn") and review_result.get("suggested_route"):
            try:
                from brains.memory.brain_memory import BrainMemory
                integrator_memory = BrainMemory("integrator")
                integrator_memory.store(
                    content={
                        "query_pattern": user_query[:100],
                        "wrong_route": actual_route,
                        "correct_route": review_result["suggested_route"],
                        "reason": review_result["reason"],
                    },
                    metadata={
                        "kind": "routing_correction",
                        "source": "router_review",
                        "confidence": 0.8,
                        "fact_type": "non_fact",
                        "purpose": "routing_learning",
                        "writable_to_domain": False,
                    }
                )
                print(f"[INTEGRATOR] Stored routing correction: {user_query[:20]}... -> {review_result['suggested_route']}")
            except Exception as e:
                print(f"[INTEGRATOR] Failed to store routing correction: {str(e)[:50]}")

        return {
            "ok": True,
            "mid": mid,
            "payload": review_result,
        }
    else:
        return {"ok": False, "mid": mid, "error": {"code": "UNSUPPORTED_OP", "message": f"Unsupported op: {op}"}}

# Standard service contract: handle is the entry point
service_api = handle