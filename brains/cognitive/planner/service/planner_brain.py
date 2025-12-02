from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import re
import sys

# Deferred import for affect modulation.  Importing inside the PLAN
# operation avoids circular dependencies when the affect brain itself
# imports the planner.  We guard failures gracefully.
import importlib

# BrainMemory tier API for persistent state
from brains.memory.brain_memory import BrainMemory

# Learning mode for memory-first, LLM-as-teacher architecture
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"

# Lesson utilities for strategy-based learning
try:
    from brains.learning.lesson_utils import (
        create_lesson_record,
        store_lesson,
        retrieve_lessons,
        planning_concept_key,
        canonical_concept_key,
    )
except Exception:
    create_lesson_record = None  # type: ignore
    store_lesson = None  # type: ignore
    retrieve_lessons = None  # type: ignore
    planning_concept_key = None  # type: ignore
    canonical_concept_key = None  # type: ignore

# Teacher integration for learning new planning patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("planner")
except Exception as e:
    print(f"[PLANNER] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for follow-up and conversation context tracking
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint,
        extract_continuation_intent
    )
except Exception as e:
    print(f"[PLANNER] Continuation helpers not available: {e}")
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore

# PHASE 2: Context manager for follow-up action tracking
try:
    from brains.cognitive.context_management.service.context_manager import (
        is_confirmation_message,
        extract_action_request,
        store_action_request,
        get_last_action_request,
        get_follow_up_context,
        mark_action_executed,
        clear_action_request,
        ActionRequest,
    )
    _action_tracking_available = True
except Exception as e:
    print(f"[PLANNER] Action tracking not available: {e}")
    _action_tracking_available = False
    is_confirmation_message = lambda text: False  # type: ignore
    extract_action_request = lambda text: None  # type: ignore
    store_action_request = lambda req: None  # type: ignore
    get_last_action_request = lambda: None  # type: ignore
    get_follow_up_context = lambda: None  # type: ignore
    mark_action_executed = lambda: None  # type: ignore
    clear_action_request = lambda: None  # type: ignore

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

# Import domain lookup for accessing planning patterns
MAVEN_ROOT = HERE.parents[3]
sys.path.insert(0, str(MAVEN_ROOT / "brains" / "domain_banks"))
try:
    from domain_lookup import lookup_by_bank_and_kind, lookup_by_tag
except Exception:
    lookup_by_bank_and_kind = None  # type: ignore
    lookup_by_tag = None  # type: ignore

# Initialize BrainMemory for planner's persistent state
memory = BrainMemory("planner")

def _get_planning_patterns() -> Dict[str, Any]:
    """
    Get planning patterns from domain bank.

    Returns:
        Dict mapping pattern types to patterns
    """
    patterns = {}
    if lookup_by_bank_and_kind:
        try:
            # Get all planning patterns (strategies, constraints, patterns, heuristics)
            for kind in ["strategy", "constraint", "pattern", "heuristic"]:
                entries = lookup_by_bank_and_kind("planning_patterns", kind)
                for entry in entries:
                    entry_id = entry.get("id", "")
                    patterns[entry_id] = entry
        except Exception:
            pass  # Return empty dict if lookup fails
    return patterns


# =============================================================================
# MEMORY-FIRST LEARNING ROUTE (following Reasoning brain pattern)
# =============================================================================
#
# Problem Types for Planning:
# - goal_decomposition: Breaking goals into subtasks
# - schedule_planning: Time-based planning
# - repair_plan: Fixing failed or incomplete plans
# - resource_allocation: Managing resources across steps
# - implementation_task: Code/build planning
# - creative_brainstorm: Ideation and design
# - decision_resolution: Conflicts and choices
# - analysis_parsing: Understanding/explanation tasks
# - generic_planning: Fallback for unmapped tasks
#
# Strategy Table: Maps (problem_type, domain) -> strategy dict
# =============================================================================

PLANNER_STRATEGY_TABLE: Dict[tuple, Dict[str, Any]] = {}


# =============================================================================
# CONCEPT-KEY BASED PLANNING (Step 1: Memory-First Planning Strategies)
# =============================================================================
# This enables the Planner to become a "life assistant" that remembers YOUR
# planning style, not generic methods. Similar planning queries map to the
# same concept_key for consistent memory lookups.

def get_planning_strategy_from_memory(query: str) -> Dict[str, Any] | None:
    """
    Look up a learned planning strategy using concept-key matching.

    This is the memory-first approach for planning:
    1. Extract the planning concept key (e.g., "planning_week")
    2. Check memory for strategies learned for this concept
    3. Return the best matching strategy if found

    Args:
        query: User's planning-related query

    Returns:
        Strategy dict if found in memory, None otherwise
    """
    if not planning_concept_key:
        return None

    # Get the canonical planning concept key
    concept_key = planning_concept_key(query)
    if not concept_key:
        return None

    print(f"[PLANNER] Detected planning concept: {concept_key}")

    # Look up strategies in memory with this concept key
    try:
        # Search for learned strategies with this concept key
        results = memory.retrieve(
            query=f"planning_strategy:{concept_key}",
            limit=5,
            tiers=["stm", "mtm", "ltm"]
        )

        for rec in results:
            metadata = rec.get("metadata", {})
            # Check if this is a planning strategy with matching concept key
            if metadata.get("kind") == "planning_strategy":
                stored_concept = metadata.get("concept_key", "")
                if stored_concept == concept_key:
                    content = rec.get("content", {})
                    if isinstance(content, dict) and content.get("steps"):
                        print(f"[PLANNER] ✓ Found learned strategy for concept: {concept_key}")
                        return content

        # Also check lesson records with matching concept key
        results = memory.retrieve(
            query=f"concept_key:{concept_key}",
            limit=5,
            tiers=["stm", "mtm", "ltm"]
        )

        for rec in results:
            metadata = rec.get("metadata", {})
            if metadata.get("concept_key") == concept_key:
                content = rec.get("content", {})
                if isinstance(content, dict):
                    # Convert lesson to strategy format
                    rule = content.get("distilled_rule", "")
                    if rule:
                        print(f"[PLANNER] ✓ Using learned lesson for concept: {concept_key}")
                        return {
                            "name": f"learned_{concept_key}",
                            "concept_key": concept_key,
                            "steps": rule,
                            "confidence": content.get("confidence", 0.7),
                            "source": "lesson_memory"
                        }

    except Exception as e:
        print(f"[PLANNER] Memory lookup error: {e}")

    return None


def store_planning_strategy(
    query: str,
    strategy: Dict[str, Any],
    confidence: float = 0.8
) -> bool:
    """
    Store a learned planning strategy with concept-key metadata.

    This enables future queries with the same concept key to retrieve
    the learned strategy without calling Teacher again.

    Args:
        query: The original query that triggered the strategy
        strategy: The strategy dict (with steps, name, etc.)
        confidence: Confidence score for this strategy

    Returns:
        True if stored successfully
    """
    if not planning_concept_key:
        return False

    concept_key = planning_concept_key(query)
    if not concept_key:
        # Fall back to canonical_concept_key for non-planning queries
        if canonical_concept_key:
            concept_key = canonical_concept_key(query)
        else:
            return False

    try:
        # Store with concept_key in metadata for lookup
        memory.store(
            content={
                "name": strategy.get("name", f"strategy_{concept_key}"),
                "concept_key": concept_key,
                "steps": strategy.get("steps", ""),
                "confidence": confidence,
                "original_query": query[:200],
                "source": "planner_learning"
            },
            metadata={
                "kind": "planning_strategy",
                "concept_key": concept_key,
                "confidence": confidence,
                "source": "planner"
            }
        )
        print(f"[PLANNER] ✓ Stored strategy for concept: {concept_key}")
        return True
    except Exception as e:
        print(f"[PLANNER] Failed to store strategy: {e}")
        return False


def classify_planning_problem(context: Dict[str, Any]) -> str:
    """
    Classify the type of planning problem from context.

    Args:
        context: Pipeline context containing query and metadata

    Returns:
        Problem type string for strategy selection
    """
    try:
        # Extract query text
        query = ""
        if "user_query" in context:
            query = str(context["user_query"])
        elif "payload" in context and isinstance(context["payload"], dict):
            query = str(context["payload"].get("user_intent", ""))
            if not query:
                query = str(context["payload"].get("task", ""))

        q_lower = (query or "").lower()

        # Goal decomposition patterns
        if any(w in q_lower for w in ["break down", "decompose", "subtasks", "steps to"]):
            return "goal_decomposition"

        # Schedule/time planning
        if any(w in q_lower for w in ["schedule", "timeline", "when should", "deadline"]):
            return "schedule_planning"

        # Repair/fix patterns
        if any(w in q_lower for w in ["fix", "repair", "failed", "didn't work", "try again"]):
            return "repair_plan"

        # Implementation/coding
        if any(w in q_lower for w in ["implement", "code", "build", "create", "develop", "write"]):
            return "implementation_task"

        # Creative/brainstorm
        if any(w in q_lower for w in ["brainstorm", "ideas", "creative", "imagine", "design"]):
            return "creative_brainstorm"

        # Decision/choice
        if any(w in q_lower for w in ["decide", "choose", "which", "compare", "vs", "better"]):
            return "decision_resolution"

        # Analysis/explanation
        if any(w in q_lower for w in ["analyze", "understand", "explain", "why", "how does"]):
            return "analysis_parsing"

        return "generic_planning"

    except Exception:
        return "generic_planning"


def load_planner_strategies_from_lessons(context: Dict[str, Any]) -> None:
    """
    Load planning strategies from stored lessons into PLANNER_STRATEGY_TABLE.

    Args:
        context: Pipeline context
    """
    global PLANNER_STRATEGY_TABLE

    if not retrieve_lessons:
        return

    try:
        lessons = retrieve_lessons("planner", memory)

        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue

            # Extract problem type from topic or input_signature
            topic = lesson.get("topic", "")
            input_sig = lesson.get("input_signature", {})
            problem_type = input_sig.get("problem_type", topic) or "generic_planning"
            domain = input_sig.get("domain")

            # Build strategy from lesson
            strategy = {
                "name": f"learned_{problem_type}",
                "problem_type": problem_type,
                "domain": domain,
                "steps": lesson.get("distilled_rule", ""),
                "confidence": lesson.get("confidence", 0.5),
                "examples": lesson.get("examples", []),
                "source_topic": topic,
            }

            strategy_key = (problem_type, domain)

            # Only update if newer/better
            existing = PLANNER_STRATEGY_TABLE.get(strategy_key)
            if not existing or strategy["confidence"] > existing.get("confidence", 0):
                PLANNER_STRATEGY_TABLE[strategy_key] = strategy

        if PLANNER_STRATEGY_TABLE:
            print(f"[PLANNER] Loaded {len(PLANNER_STRATEGY_TABLE)} strategies from lessons")

    except Exception as e:
        print(f"[PLANNER] Error loading strategies: {e}")


def select_planner_strategy(problem_type: str, domain: str = None) -> Dict[str, Any] | None:
    """
    Select the best planning strategy for a problem type.

    Args:
        problem_type: The classified problem type
        domain: Optional domain for domain-specific strategies

    Returns:
        Strategy dict or None if no suitable strategy found
    """
    # Try exact match first
    strategy = PLANNER_STRATEGY_TABLE.get((problem_type, domain))
    if strategy:
        return strategy

    # Try domain-agnostic
    strategy = PLANNER_STRATEGY_TABLE.get((problem_type, None))
    if strategy:
        return strategy

    # Try generic fallback
    if problem_type != "generic_planning":
        strategy = PLANNER_STRATEGY_TABLE.get(("generic_planning", None))
        return strategy

    return None


def apply_planner_strategy(
    strategy: Dict[str, Any],
    context: Dict[str, Any]
) -> tuple:
    """
    Apply a planning strategy to generate a plan.

    Memory-first approach: Use learned strategies before calling LLM.

    Args:
        strategy: Strategy dict from PLANNER_STRATEGY_TABLE
        context: Pipeline context

    Returns:
        Tuple of (result_dict, confidence_score)
    """
    try:
        strategy_name = strategy.get("name", "unknown")
        steps = strategy.get("steps", "")
        base_confidence = strategy.get("confidence", 0.5)
        problem_type = strategy.get("problem_type", "generic_planning")

        print(f"[PLANNER] Applying strategy '{strategy_name}' for problem_type={problem_type}")

        result = {
            "strategy_used": strategy_name,
            "problem_type": problem_type,
            "planning_steps": steps,
        }

        # Check if we have a learned plan template
        if steps and len(steps) > 20:
            result["plan_template"] = steps
            result["source"] = "learned_strategy"
            result["confidence"] = base_confidence
            print(f"[PLANNER] ✓ Strategy path: Using learned plan template (no LLM needed)")
            return result, base_confidence

        # No usable strategy template
        result["verdict"] = "NO_STRATEGY_PLAN"
        result["confidence"] = 0.0
        print(f"[PLANNER] Strategy path: No learned plan, may need LLM teacher")
        return result, 0.0

    except Exception as e:
        print(f"[PLANNER] Strategy error: {e}")
        return {"error": "strategy_failed"}, 0.0


def planner_llm_lesson(
    context: Dict[str, Any],
    task: str,
    learning_mode: LearningMode
) -> Dict[str, Any] | None:
    """
    Generate a lesson from LLM interaction for the planner brain.

    Args:
        context: Pipeline context
        task: The planning task
        learning_mode: Current learning mode

    Returns:
        Lesson record dict, or None if failed
    """
    if not create_lesson_record or not store_lesson:
        return None

    problem_type = classify_planning_problem(context)
    domain = context.get("domain")

    input_signature = {
        "problem_type": problem_type,
        "domain": domain,
    }

    # OFFLINE mode - no LLM call
    if learning_mode == LearningMode.OFFLINE:
        return create_lesson_record(
            brain="planner",
            topic=problem_type,
            input_signature=input_signature,
            llm_prompt="",
            llm_response="",
            distilled_rule="",
            examples=[],
            confidence=0.0,
            mode="offline",
            status="new"
        )

    # TRAINING or SHADOW mode - call TeacherHelper
    if not _teacher_helper:
        return None

    try:
        teacher_result = _teacher_helper.maybe_call_teacher(
            question=task,
            context=context,
            learning_mode=learning_mode
        )

        if not teacher_result or teacher_result.get("verdict") in ("LLM_DISABLED", "NO_MEMORY"):
            return None

        llm_response = teacher_result.get("answer", "")
        verdict = teacher_result.get("verdict", "UNKNOWN")

        confidence = 0.8 if verdict == "LEARNED" else (0.9 if verdict == "KNOWN" else 0.5)

        lesson = create_lesson_record(
            brain="planner",
            topic=problem_type,
            input_signature=input_signature,
            llm_prompt=f"Planning task: {task}",
            llm_response=llm_response or "",
            distilled_rule=llm_response or "",
            examples=[],
            confidence=confidence,
            mode=str(learning_mode.value) if hasattr(learning_mode, 'value') else str(learning_mode),
            status="new"
        )

        stored = store_lesson("planner", lesson, memory)
        if stored:
            print(f"[PLANNER] Stored lesson: {problem_type} (confidence={confidence})")

        lesson["plan"] = llm_response
        return lesson

    except Exception as e:
        print(f"[PLANNER] Lesson error: {e}")
        return None


def _guess_intents_targets(text: str):
    text_l = (text or "").lower()
    intents = []
    targets = []
    # intents (very light heuristics)
    if any(w in text_l for w in ["show", "display", "find", "search", "retrieve"]):
        intents.append("retrieve_relevant_memories")
    if any(w in text_l for w in ["explain", "why", "how"]):
        intents.append("compose_explanation")
    if not intents:
        intents.append("compose_response")
    # targets/entities
    toks = [t.strip(",.!?") for t in (text or "").split()]
    for t in toks:
        if t and (t[0].isupper() or t.lower() in ("paris","eiffel","tower","photos")):
            targets.append(t)
    # dedupe preserving order
    seen=set(); targets = [x for x in targets if not (x in seen or seen.add(x))]
    return intents, targets

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import generate_mid, success_response, error_response  # type: ignore
    from api.memory import compute_success_average  # type: ignore
    op = (msg or {}).get("op"," ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    # ------------------------------------------------------------------
    # Step‑4: handshake from working memory
    # PLAN_FROM_WM creates a single goal entry based on a WM event.  When
    # an entry contains tags=["plan"], a goal identifier prefixed with
    # WM_PLAN: is returned and a governance audit entry may be recorded.  The
    # implementation is deliberately simple: only one goal is created per
    # invocation and duplicate calls with the same key do not create
    # multiple goals.  The ledger location is reused from existing plan
    # storage to maintain consistency.
    if op == "PLAN_FROM_WM":
        entry = payload.get("entry") or {}
        key = str(entry.get("key", ""))
        # Create a deterministic goal ID for this WM entry
        goal_id = f"WM_PLAN:{key}"
        # Check if goal already exists using BrainMemory
        try:
            # Retrieve existing goals with this goal_id
            existing_goals = memory.retrieve(query=goal_id, limit=1)
            exists = len(existing_goals) > 0
            if not exists:
                # Persist new goal using BrainMemory tier system
                memory.store(
                    content={"goal_id": goal_id, "source": "PLAN_FROM_WM"},
                    metadata={
                        "kind": "goal",
                        "source": "planner",
                        "confidence": 0.9,
                        "goal_id": goal_id
                    }
                )
        except Exception:
            pass
        return success_response(op, mid, {"goal": goal_id})
    # Health check
    if op == "HEALTH":
        # Store health check using BrainMemory tier system
        try:
            memory.store(
                content={"op": "HEALTH"},
                metadata={
                    "kind": "health_check",
                    "source": "planner",
                    "confidence": 1.0
                }
            )
        except Exception:
            pass
        # Get memory stats from BrainMemory
        memory_health = memory.get_stats()
        return success_response(op, mid, {"status": "operational", "memory_health": memory_health})

    # =========================================================================
    # PHASE 2: HANDLE_FOLLOWUP - Process "do it please" follow-up messages
    # =========================================================================
    # When a user says "do it please" after an action request like
    # "can you write me a story about birds", this operation:
    # 1. Checks if the message is a confirmation
    # 2. Retrieves the pending action request
    # 3. Creates a plan step to execute the original request
    if op == "HANDLE_FOLLOWUP":
        text = str(payload.get("text", ""))

        if not text:
            return error_response(op, mid, "MISSING_TEXT", "Text is required")

        # Check if this is a confirmation message
        if not is_confirmation_message(text):
            return success_response(op, mid, {
                "is_followup": False,
                "reason": "not_confirmation",
            })

        # Check if there's a pending action
        follow_up_context = get_follow_up_context()
        if not follow_up_context:
            return success_response(op, mid, {
                "is_followup": False,
                "reason": "no_pending_action",
            })

        # We have a follow-up! Create a plan step to execute the original request
        original_request = follow_up_context.get("original_request", "")
        request_type = follow_up_context.get("request_type", "unknown")
        action_payload = follow_up_context.get("payload", {})

        # Extract detailed info from payload
        content_type = action_payload.get("content_type", "")
        topic = action_payload.get("topic", "")
        intent = action_payload.get("intent", request_type)
        arguments = action_payload.get("arguments", {})

        print(f"[PLANNER] Handling follow-up: executing '{request_type}' from '{original_request[:50]}...'")
        print(f"[PLANNER] Follow-up details: intent={intent}, content_type={content_type}, topic={topic}")

        # Create execution plan based on request type
        steps = []
        if request_type == "write_content":
            # Build detailed task description
            task_description = original_request
            if content_type and topic:
                task_description = f"Write a {content_type} about {topic}"
            elif content_type:
                task_description = f"Write a {content_type}"

            steps = [
                {
                    "type": "language",
                    "description": f"Generate content: {task_description}",
                    "tags": ["generate", "content", "follow_up"],
                    "input": {
                        "task": original_request,
                        "task_description": task_description,
                        "is_follow_up": True,
                        "confirm_execution": True,
                        # Pass detailed info for language brain
                        "intent": intent,
                        "content_type": content_type,
                        "topic": topic,
                        "arguments": arguments,
                        # This is CRITICAL - tells language brain to execute, not ask capability
                        "execute_mode": True,
                    }
                }
            ]
        elif request_type == "search":
            steps = [
                {
                    "type": "search",
                    "description": f"Execute search: {original_request}",
                    "tags": ["search", "follow_up"],
                    "input": {
                        "task": original_request,
                        "is_follow_up": True,
                        "topic": topic,
                        "arguments": arguments,
                        "execute_mode": True,
                    }
                }
            ]
        elif request_type == "explain":
            steps = [
                {
                    "type": "reasoning",
                    "description": f"Provide explanation: {original_request}",
                    "tags": ["explain", "follow_up"],
                    "input": {
                        "task": original_request,
                        "is_follow_up": True,
                        "topic": topic,
                        "arguments": arguments,
                        "execute_mode": True,
                    }
                }
            ]
        else:
            # Generic execution - use language brain
            steps = [
                {
                    "type": "language",
                    "description": f"Execute: {original_request}",
                    "tags": ["execute", "follow_up"],
                    "input": {
                        "task": original_request,
                        "is_follow_up": True,
                        "intent": intent,
                        "topic": topic,
                        "arguments": arguments,
                        "execute_mode": True,
                    }
                }
            ]

        # Mark the action as executed
        mark_action_executed()

        return success_response(op, mid, {
            "is_followup": True,
            "original_request": original_request,
            "request_type": request_type,
            "steps": steps,
            "follow_up_context": follow_up_context,
        })

    # =========================================================================
    # PHASE 2: EXTRACT_ACTION - Extract actionable request from user text
    # =========================================================================
    # When processing a message like "can you write me a story about birds",
    # this operation extracts and stores the action request for potential
    # follow-up execution.
    if op == "EXTRACT_ACTION":
        text = str(payload.get("text", ""))

        if not text:
            return error_response(op, mid, "MISSING_TEXT", "Text is required")

        # Try to extract an actionable request
        action_request = extract_action_request(text)

        if action_request:
            # Store for potential follow-up
            store_action_request(action_request)
            print(f"[PLANNER] Stored action request: {action_request.request_type}")

            return success_response(op, mid, {
                "is_action_request": True,
                "request_type": action_request.request_type,
                "original_text": action_request.original_text,
                "payload": action_request.payload,
            })
        else:
            return success_response(op, mid, {
                "is_action_request": False,
            })

    # DECOMPOSE_TASK - Phase 8: Deterministic task decomposition
    if op == "DECOMPOSE_TASK":
        task = str(payload.get("task", ""))
        context = payload.get("context") or {}

        if not task:
            return error_response(op, mid, "MISSING_TASK", "Task description is required")

        # Get planning patterns from domain bank
        planning_patterns = _get_planning_patterns()
        patterns_used = []

        # Decompose task using planning patterns
        steps = []
        learned_from_teacher = False

        # First, try to find learned patterns in memory
        try:
            learned_patterns = memory.retrieve(
                query=f"task decomposition: {task}",
                limit=5,
                tiers=["stm", "mtm", "ltm"]
            )

            # Look for high-confidence learned patterns
            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    # Found a learned pattern from Teacher
                    content = pattern_rec.get("content", "")
                    if isinstance(content, str) and "PATTERN:" in content:
                        # Use this learned pattern
                        print(f"[PLANNER] Using learned pattern from Teacher for: {task[:50]}...")
                        learned_from_teacher = True

                        # Parse pattern and create steps
                        # For now, create a generic step based on the learned answer
                        steps = [
                            {
                                "type": "planning",
                                "description": f"Apply learned pattern: {content[:100]}",
                                "tags": ["plan", "learned"],
                                "input": {"task": task, "pattern": content}
                            }
                        ]
                        patterns_used.append("learned_from_teacher")
                        break
        except Exception:
            pass

        # Analyze task to determine decomposition strategy
        task_lower = task.lower()

        # Check for coding/implementation tasks
        if any(keyword in task_lower for keyword in ["implement", "code", "write", "build", "create", "fix", "debug"]):
            # Use divide_and_conquer strategy for implementation
            patterns_used.append("planning_patterns:strategy:divide_and_conquer")
            patterns_used.append("planning_patterns:constraint:determinism_first")

            steps = [
                {
                    "type": "planning",
                    "description": f"Analyze requirements for: {task}",
                    "tags": ["plan", "analyze"],
                    "input": {"task": task, "context": context}
                },
                {
                    "type": "coding",
                    "description": f"Implement solution for: {task}",
                    "tags": ["coding", "implement"],
                    "input": {"task": task, "context": context}
                },
                {
                    "type": "reasoning",
                    "description": f"Verify implementation correctness",
                    "tags": ["reasoning", "verify"],
                    "input": {"task": task, "context": context}
                }
            ]

        # Check for creative/brainstorming tasks
        elif any(keyword in task_lower for keyword in ["brainstorm", "imagine", "creative", "design", "ideate"]):
            patterns_used.append("planning_patterns:strategy:divide_and_conquer")

            steps = [
                {
                    "type": "creative",
                    "description": f"Generate ideas for: {task}",
                    "tags": ["creative", "brainstorm"],
                    "input": {"task": task, "context": context}
                },
                {
                    "type": "reasoning",
                    "description": f"Evaluate and rank ideas",
                    "tags": ["reasoning", "evaluate"],
                    "input": {"task": task, "context": context}
                }
            ]

        # Check for conflict/decision tasks
        elif any(keyword in task_lower for keyword in ["decide", "choose", "resolve", "conflict", "arbitrate"]):
            patterns_used.append("planning_patterns:strategy:divide_and_conquer")

            steps = [
                {
                    "type": "governance",
                    "description": f"Arbitrate decision for: {task}",
                    "tags": ["governance", "decide"],
                    "input": {"task": task, "context": context}
                }
            ]

        # Check for analysis/understanding tasks
        elif any(keyword in task_lower for keyword in ["analyze", "understand", "explain", "parse"]):
            steps = [
                {
                    "type": "planning",
                    "description": f"Parse and understand: {task}",
                    "tags": ["parse", "analyze"],
                    "input": {"task": task, "context": context}
                },
                {
                    "type": "reasoning",
                    "description": f"Reason about: {task}",
                    "tags": ["reasoning", "logic"],
                    "input": {"task": task, "context": context}
                },
                {
                    "type": "language",
                    "description": f"Compose explanation",
                    "tags": ["language", "generate"],
                    "input": {"task": task, "context": context}
                }
            ]

        # Default: generic multi-step plan OR call Teacher to learn
        else:
            # If no pattern matched and we haven't learned from Teacher yet, try Teacher
            if not learned_from_teacher and not steps and _teacher_helper:
                try:
                    print(f"[PLANNER] No pattern found for '{task[:50]}...', calling Teacher to learn...")

                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"How should I decompose this task into steps: {task}",
                        context=context,
                        check_memory_first=True  # Check memory to avoid redundant calls
                    )

                    if teacher_result and teacher_result.get("answer"):
                        # Teacher provided a planning pattern!
                        answer = teacher_result["answer"]
                        patterns_stored = teacher_result.get("patterns_stored", 0)

                        print(f"[PLANNER] Learned from Teacher: {patterns_stored} patterns stored")
                        print(f"[PLANNER] Answer: {answer[:100]}...")

                        # Create steps based on Teacher's guidance
                        steps = [
                            {
                                "type": "planning",
                                "description": f"Apply Teacher pattern: {answer[:100]}",
                                "tags": ["plan", "teacher_learned"],
                                "input": {"task": task, "context": context, "teacher_pattern": answer}
                            },
                            {
                                "type": "reasoning",
                                "description": f"Execute using learned pattern",
                                "tags": ["reasoning", "execute"],
                                "input": {"task": task, "context": context}
                            }
                        ]
                        patterns_used.append("teacher_learned_pattern")
                        learned_from_teacher = True

                except Exception as e:
                    print(f"[PLANNER] Teacher call failed: {str(e)[:100]}")

            # Fallback: use generic pattern if Teacher didn't help
            if not steps:
                patterns_used.append("planning_patterns:heuristic:smallest_testable_unit")

                steps = [
                    {
                        "type": "planning",
                        "description": f"Plan approach for: {task}",
                        "tags": ["plan"],
                        "input": {"task": task, "context": context}
                    },
                    {
                        "type": "reasoning",
                        "description": f"Execute and reason about: {task}",
                        "tags": ["reasoning"],
                        "input": {"task": task, "context": context}
                    }
                ]

        result = {
            "steps": steps,
            "patterns_used": patterns_used,
            "task": task
        }

        # Persist decomposition using BrainMemory tier system
        try:
            memory.store(
                content={
                    "op": "DECOMPOSE_TASK",
                    "task": task,
                    "steps": steps,
                    "patterns_used": patterns_used
                },
                metadata={
                    "kind": "task_decomposition",
                    "source": "planner",
                    "confidence": 0.85,
                    "task": task
                }
            )
        except Exception:
            pass

        return success_response(op, mid, result)

    # EXECUTE_STEP: Phase 8 - Execute a planning/parsing step
    if op == "EXECUTE_STEP":
        step = payload.get("step") or {}
        step_id = payload.get("step_id", 0)
        context = payload.get("context") or {}

        # Get planning patterns
        planning_patterns = _get_planning_patterns()
        patterns_used = []

        # Extract step details
        description = step.get("description", "")
        step_input = step.get("input") or {}
        task = step_input.get("task", description)

        # Execute planning step
        # For parse/analyze tasks, decompose into sub-components
        # For plan tasks, create a structured plan

        intents, targets = _guess_intents_targets(task)

        output = {
            "analysis": description,
            "intents": intents,
            "targets": targets,
            "task": task
        }

        # Use relevant planning patterns
        if planning_patterns:
            patterns_used = ["planning_patterns:strategy:divide_and_conquer"]

        return success_response(op, mid, {
            "output": output,
            "patterns_used": patterns_used
        })

    # Generate a basic plan for user requests and track long‑term goals.
    if op == "PLAN":
        text = str(payload.get("text", ""))
        intent = str(payload.get("intent", "")).upper()
        context = payload.get("context") or {}
        motivation = payload.get("motivation")

        # =================================================================
        # STEP 1 ENHANCEMENT: Concept-Key Based Strategy Lookup
        # =================================================================
        # Check for learned planning strategies using concept-key matching
        # This enables "plan my week" and "help me schedule my week" to
        # use the same learned strategy.
        concept_strategy = get_planning_strategy_from_memory(text)
        if concept_strategy:
            plan_id = f"plan_{int(importlib.import_module('time').time() * 1000)}"
            concept_key = concept_strategy.get("concept_key", "unknown")
            print(f"[PLANNER] ✓ Using concept-key strategy: {concept_key}")

            # Build plan from learned strategy
            plan: Dict[str, Any] = {
                "plan_id": plan_id,
                "goal": f"Satisfy user request: {text}",
                "steps": [
                    {"id": "s1", "kind": "apply_learned_strategy", "strategy": concept_key, "status": "pending"},
                    {"id": "s2", "kind": "execute", "status": "pending"},
                    {"id": "s3", "kind": "compose_response", "status": "pending"},
                ],
                "priority": concept_strategy.get("confidence", 0.8),
                "can_parallelize": False,
                "intent": intent,
                "concept_key": concept_key,
                "learned_strategy": concept_strategy,
                "notes": f"Using learned strategy for concept: {concept_key}"
            }

            # Store plan in memory
            try:
                memory.store(
                    content={"op": "PLAN", "input": text, "plan": plan, "success": None},
                    metadata={
                        "kind": "plan",
                        "source": "planner",
                        "confidence": concept_strategy.get("confidence", 0.8),
                        "plan_id": plan_id,
                        "concept_key": concept_key
                    }
                )
            except Exception:
                pass

            return success_response(op, mid, plan)

        # FOLLOW-UP DETECTION: Check if this is a continuation
        is_cont = False
        try:
            is_cont = is_continuation(text, context)
            if is_cont:
                print("[PLANNER] Detected follow-up, adjusting plan complexity")
                # For continuations, plans may be simpler (expanding on existing plan)
        except Exception as e:
            print(f"[PLANNER] Could not check continuation: {e}")

        # Affect modulation: score the input text to derive valence,
        # arousal and priority delta.  Defer import to runtime to avoid
        # circular dependencies.
        affect: Dict[str, Any] = {}
        try:
            ap_mod = importlib.import_module(
                "brains.cognitive.affect_priority.service.affect_priority_brain"
            )
            aff_res = ap_mod.service_api({"op": "SCORE", "payload": {"text": text}})
            affect = aff_res.get("payload") or {}
        except Exception:
            affect = {}

        # Step 2: Determine plan structure based on intent
        plan_id = f"plan_{int(importlib.import_module('time').time() * 1000)}"
        steps: list[Dict[str, Any]] = []
        priority = 0.5
        can_parallelize = False

        # Simple Q&A (simple_fact_query, question)
        if intent in ("SIMPLE_FACT_QUERY", "QUESTION", "QUERY"):
            steps = [
                {"id": "s1", "kind": "retrieve", "target": "personal_memory", "status": "pending"},
                {"id": "s2", "kind": "reason", "status": "pending"},
                {"id": "s3", "kind": "compose_answer", "status": "pending"},
            ]
            priority = 0.7

        # Explain / Why / How
        elif intent in ("EXPLAIN", "WHY", "HOW"):
            steps = [
                {"id": "s1", "kind": "retrieve", "target": "all_banks", "status": "pending"},
                {"id": "s2", "kind": "reason", "target": "chain_facts", "status": "pending"},
                {"id": "s3", "kind": "compose_explanation", "status": "pending"},
            ]
            priority = 0.8

        # Compare ("compare A and B")
        elif intent == "COMPARE":
            # Extract comparison targets from text if possible
            text_lower = text.lower()
            targets_a = []
            targets_b = []
            if " and " in text_lower:
                parts = text_lower.split(" and ", 1)
                # Extract nouns/entities from each part
                import re
                targets_a = re.findall(r'\b[A-Za-z]+\b', parts[0])[-3:] if parts[0] else []
                targets_b = re.findall(r'\b[A-Za-z]+\b', parts[1])[:3] if len(parts) > 1 else []

            steps = [
                {"id": "s1", "kind": "retrieve", "target": "subject_a", "subjects": targets_a, "status": "pending"},
                {"id": "s2", "kind": "retrieve", "target": "subject_b", "subjects": targets_b, "status": "pending"},
                {"id": "s3", "kind": "reason", "target": "highlight_differences", "status": "pending"},
                {"id": "s4", "kind": "compose_comparison", "status": "pending"},
            ]
            priority = 0.85

        # User command ("do X", "summarize", "analyze this")
        elif intent in ("COMMAND", "REQUEST", "ANALYZE"):
            steps = [
                {"id": "s1", "kind": "interpret_command", "status": "pending"},
                {"id": "s2", "kind": "retrieve", "target": "relevant_context", "status": "pending"},
                {"id": "s3", "kind": "act", "target": "action_engine", "status": "pending"},
                {"id": "s4", "kind": "compose_result", "status": "pending"},
            ]
            priority = 0.9
            can_parallelize = False

        # Preference/identity queries
        elif intent in ("PREFERENCE_QUERY", "IDENTITY_QUERY", "RELATIONSHIP_QUERY"):
            steps = [
                {"id": "s1", "kind": "retrieve", "target": "personal_memory", "status": "pending"},
                {"id": "s2", "kind": "reason", "status": "pending"},
                {"id": "s3", "kind": "compose_answer", "status": "pending"},
            ]
            priority = 0.75

        # Unknown or unsupported intent
        else:
            if intent:
                steps = [
                    {"id": "s1", "kind": "fail", "reason": "unsupported_intent", "intent": intent, "status": "pending"}
                ]
            else:
                # No intent provided - create a generic plan
                intents, targets = _guess_intents_targets(text)
                steps = [
                    {"id": "s1", "kind": "retrieve", "target": "general", "status": "pending"},
                    {"id": "s2", "kind": "reason", "status": "pending"},
                    {"id": "s3", "kind": "compose_response", "status": "pending"},
                ]
            priority = 0.5

        # Add backward compatibility fields (intents/targets)
        intents, targets = _guess_intents_targets(text)

        plan: Dict[str, Any] = {
            "plan_id": plan_id,
            "goal": f"Satisfy user request: {text}",
            "steps": steps,
            "priority": priority,
            "can_parallelize": can_parallelize,
            "intent": intent,
            "intents": intents,  # Backward compatibility
            "targets": targets,  # Backward compatibility
            "notes": "Step 2 planner: real multi-step plan"
        }
        # Record affect metrics in the plan
        if affect:
            plan["affect"] = affect
            # Derive a high‑level mood indicator from valence
            try:
                val = float(affect.get("valence", 0.0))
            except Exception:
                val = 0.0
            if val > 0.2:
                mood = "upbeat"
            elif val < -0.2:
                mood = "cautious"
            else:
                mood = "neutral"
            plan["mood"] = mood
        # Incorporate a learned bias into the plan based on recent success history.
        try:
            learned_bias = compute_success_average(BRAIN_ROOT)
        except Exception:
            learned_bias = 0.0
        # Adjust learned bias using the affect‑derived priority delta if available
        try:
            delta = float(affect.get("priority_delta", 0.0))
        except Exception:
            delta = 0.0
        plan["learned_bias"] = learned_bias + delta
        # Persist the plan using BrainMemory tier system
        try:
            # Store full plan details
            memory.store(
                content={"op": "PLAN", "input": text, "plan": plan, "success": None},
                metadata={
                    "kind": "plan",
                    "source": "planner",
                    "confidence": priority,
                    "plan_id": plan_id,
                    "intent": intent
                }
            )
            # Store plan summary for long-term tracking
            memory.store(
                content={"op": "PLAN", "intents": intents, "targets": targets},
                metadata={
                    "kind": "plan_summary",
                    "source": "planner",
                    "confidence": priority,
                    "plan_id": plan_id
                }
            )
            # Store as a goal for long-term tracking
            memory.store(
                content={"plan": plan},
                metadata={
                    "kind": "goal",
                    "source": "planner",
                    "confidence": priority,
                    "plan_id": plan_id
                }
            )
        except Exception:
            pass

        # Split the request into sub‑tasks and record each as a persistent goal.
        # STEP 2: Only do this if steps weren't already set by intent-based planning
        if not steps or (len(steps) == 1 and steps[0].get("kind") == "fail"):
            try:
                # First, detect simple conditional patterns of the form
                # Detect conditional patterns and sequence instructions into sub‑goals.  We
                # handle three forms:
                #   1) "if X then Y" → second goal runs on success of first.
                #   2) "if not X then Y" or "if X fails then Y" → second goal runs on
                #      failure of first.
                #   3) "unless X, Y" (or "unless X then Y") → equivalent to
                #      "if not X then Y".
                segments: list[str] = []
                conditions: list[Optional[str]] = []
                # Pattern for "unless" conditions
                unless_match = re.search(r"\bunless\s+(.+?)\s*(?:,\s*|\s+then\s+)(.+)", text, flags=re.IGNORECASE)
                # Pattern for generic "if X then Y"
                cond_match = re.search(r"\bif\s+(.+?)\s+then\s+(.+)", text, flags=re.IGNORECASE)
                if unless_match:
                    cond_part = unless_match.group(1).strip()
                    action_part = unless_match.group(2).strip()
                    if cond_part:
                        segments.append(cond_part)
                        conditions.append(None)
                    if action_part:
                        segments.append(action_part)
                        # For "unless", the action triggers on failure of the condition
                        conditions.append("failure")
                elif cond_match:
                    cond_part = cond_match.group(1).strip()
                    action_part = cond_match.group(2).strip()
                    trigger = "success"
                    # If the condition includes negation or mentions failure, set trigger to failure
                    if re.search(r"\bnot\b", cond_part, flags=re.IGNORECASE) or re.search(r"fail", cond_part, flags=re.IGNORECASE):
                        trigger = "failure"
                    if cond_part:
                        segments.append(cond_part)
                        conditions.append(None)
                    if action_part:
                        segments.append(action_part)
                        conditions.append(trigger)
                else:
                    # Split by sequencing conjunctions for simple lists of tasks
                    raw_segments = [s.strip() for s in re.split(
                        r"\b(?:and|then|after|before|once\s+you\s+have|once\s+you\'ve|once)\b|,",
                        text,
                        flags=re.IGNORECASE,
                    ) if s and s.strip()]
                    segments = raw_segments
                    conditions = [None for _ in segments]
                # Only record goals when more than one segment is detected
                # NOTE: This sets plan["steps"] to a list of strings for backward compatibility
                # In Step 2, we prefer structured step dicts, so only use this fallback
                # when no structured steps were created above
                if len(segments) > 1:
                    plan["steps"] = segments
                    # Also update the "steps" variable for goal memory below
                    steps = segments
                try:
                    from brains.personal.memory import goal_memory  # type: ignore
                    prev_id: str | None = None
                    # The first segment becomes the parent goal; all
                    # subsequent segments are children of the first to
                    # construct a hierarchical plan.  The previous
                    # segment ID still determines the linear dependency.
                    root_id: str | None = None
                    for seg, cond in zip(segments, conditions):
                        try:
                            parent_arg = None
                            depends_arg = None
                            # The first segment has no dependencies; we
                            # treat it as the root.  Subsequent segments
                            # depend on the immediate previous goal.  We no
                            # longer assign a parent_id for sub‑goals to
                            # avoid conflicting hierarchy and dependency
                            # specifications.
                            if prev_id:
                                depends_arg = [prev_id]
                            # Create the goal.  Only depends_on is used for
                            # sequencing; parent_id is not specified for
                            # sub‑goals to simplify the structure.
                            rec = goal_memory.add_goal(
                                seg,
                                depends_on=depends_arg,
                                condition=cond,
                                parent_id=None,
                            )
                            if isinstance(rec, dict) and rec.get("goal_id"):
                                if root_id is None:
                                    root_id = rec["goal_id"]
                                prev_id = rec["goal_id"]
                        except Exception:
                            continue
                except Exception:
                    pass
            except Exception:
                pass
        return success_response(op, mid, plan)
    # Unsupported operations
    return error_response(op, mid, "UNSUPPORTED_OP", op)

def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bid for attention based on planning needs.

    Provides routing hints with conversation context for continuations.
    """
    try:
        lang_info = ctx.get("stage_3_language", {}) or {}
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")
        conv_context = lang_info.get("conversation_context", {})

        # Plans bid moderately when needed
        routing_hint = create_routing_hint(
            brain_name="planner",
            action="create_plan",
            confidence=0.25,
            context_tags=["planning", "orchestration"],
            metadata={
                "is_continuation": is_cont,
                "continuation_type": continuation_intent,
                "last_topic": conv_context.get("last_topic", "")
            }
        )

        return {
            "brain_name": "planner",
            "priority": 0.25,
            "reason": "planning_needed",
            "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
        }
    except Exception:
        return {
            "brain_name": "planner",
            "priority": 0.25,
            "reason": "default",
            "evidence": {},
        }

# ---------------------------------------------------------------------------
# Handle wrapper for planner entry point
# ---------------------------------------------------------------------------

# Save reference to original service_api implementation
_service_api_impl = service_api

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle function that calls the planner service implementation.

    This wrapper provides a consistent entry point name across all
    cognitive service modules.

    Args:
        msg: Request dictionary with 'op' and optional 'payload'

    Returns:
        Response dictionary from planner service
    """
    return _service_api_impl(msg)

# Service API entry point
service_api = handle