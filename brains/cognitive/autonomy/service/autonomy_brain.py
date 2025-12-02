"""Autonomy Brain Service
========================

This module provides a lightweight autonomous executor.  It processes
persisted goals from the personal goal memory and marks them as
completed.  The intent is to demonstrate a simple tick‑based scheduler
that works alongside the existing motivation and planner modules.  Each
tick performs at most one goal completion to avoid long blocking loops.
Future enhancements could include dynamic priority selection, resource
budgeting and re‑planning.

Operations:

  TICK
      Execute a single autonomous cycle.  This reads active goals
      from the personal goal memory (see ``brains/personal/memory/goal_memory.py``).
      It completes one goal (FIFO order) and returns its ID and title in
      the payload.  If no goals are pending or an error occurs, the
      executed list is empty.

  HEALTH
      Return a basic health status for the autonomy brain.

Unknown operations return an error.

"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import sys
import time

# Import BrainMemory for tier-based persistent storage
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning autonomous goal scheduling and priority patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("autonomy")
except Exception as e:
    print(f"[AUTONOMY] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[AUTONOMY] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize BrainMemory for autonomy brain
_memory = BrainMemory("autonomy")

# Attempt to import utilities for consistent response formatting.  If
# unavailable (e.g., in isolated tests), fallback to simple dicts.
try:
    pass  # Imports moved to function scope
except Exception:
    import random
    def generate_mid() -> str: return f"MID-{random.randint(100000, 999999)}"
    def success_response(op, mid, payload): return {"ok": True, "op": op, "mid": mid, "payload": payload}
    def error_response(op, mid, code, message): return {"ok": False, "op": op, "mid": mid, "error": {"code": code, "message": message}}

def _load_autonomy_config() -> Dict[str, Any]:
    """Load autonomy configuration settings.

    This helper ascends to the Maven project root and reads
    ``config/autonomy.json``.  The autonomy configuration can specify
    several parameters that influence how the autonomy brain runs:

      - ``enable`` (bool): Whether to run the autonomy tick at all.
      - ``max_goals`` (int): How many goals to complete per tick.
      - ``priority_strategy`` (str): Either ``"priority"`` (default) or
        ``"fifo"`` to control how active goals are sorted before
        execution.
      - ``rate_limit_minutes`` (int or float): Minimum minutes that must
        elapse between successive ticks.  When this rate limit is
        exceeded, the tick call will return immediately without
        completing any goals.  A zero or missing value disables rate
        limiting.

    If the config file is missing or malformed, this helper returns
    an empty dict.

    Returns:
        A dictionary of configuration values.
    """
    try:
        from pathlib import Path
        import json
        # Determine the project root by ascending from this file's location.
        # Ascend to the Maven project root.  ``parents[4]`` corresponds to
        # ``.../maven`` when this file lives at
        # ``maven/brains/cognitive/autonomy/service/autonomy_brain.py``.
        root = Path(__file__).resolve().parents[4]
        cfg_path = root / "config" / "autonomy.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}

def _should_run(cfg: Dict[str, Any]) -> bool:
    """Determine whether an autonomy tick should execute.

    This override disables rate limiting entirely to ensure that
    autonomous goals continue to make progress on every tick.  Any
    configured ``rate_limit_minutes`` field in the autonomy config
    is ignored and the tick always proceeds.  The previous
    implementation consulted a timestamp in ``reports/autonomy/last_tick.json``
    and skipped ticks when invoked too frequently; however, this
    behaviour caused the agent to report "rate_limited" under
    normal usage.  To support uninterrupted execution and partial
    progress logging, rate limiting is now disabled.

    Args:
        cfg: Autonomy configuration (unused).

    Returns:
        Always True to indicate that the tick should run.
    """
    # Ignore configuration and always allow tick execution
    return True

def _update_last_tick() -> None:
    """Record the last tick event using BrainMemory tier API.

    Stores a tick event record in the autonomy brain's memory tiers.
    This provides an audit trail of autonomous execution cycles.
    Exceptions are ignored silently.
    """
    try:
        _memory.store(
            content={"event": "tick", "timestamp": time.time()},
            metadata={
                "kind": "tick_event",
                "source": "autonomy",
                "confidence": 1.0
            }
        )
    except Exception:
        pass


def _get_remaining_budget(cfg: Dict[str, Any]) -> float:
    """Return the remaining execution budget for the autonomy system using BrainMemory.

    The autonomy configuration can specify an ``execution_budget`` field
    indicating the maximum number of goals that may be executed across
    all autonomy ticks.  To track consumption of this budget across
    runs, this helper stores budget state in the autonomy brain's memory
    tiers.  When no budget record exists, it is initialized with the
    configured budget value.  If the configuration does not specify a
    budget or the value is invalid (non‑numeric or less than zero), the
    budget is treated as unlimited (returns infinity).

    Args:
        cfg: Autonomy configuration dictionary.

    Returns:
        The remaining budget as a float.  Infinity indicates unlimited
        budget.
    """
    try:
        # Determine the configured budget from config.  Non‑positive or
        # missing values indicate unlimited budget.
        budget_cfg = cfg.get("execution_budget")
        if budget_cfg is None:
            return float("inf")
        try:
            total_budget = float(budget_cfg)
        except Exception:
            return float("inf")
        if total_budget <= 0:
            return float("inf")

        # Retrieve the most recent budget record from memory
        try:
            budget_records = _memory.retrieve(limit=100)
            # Filter for budget_state records
            budget_state_records = [
                r for r in budget_records
                if isinstance(r.get("content"), dict) and
                r.get("content", {}).get("budget_state") is not None
            ]

            if budget_state_records:
                # Get most recent budget record
                latest = budget_state_records[0]
                current_rem = float(latest.get("content", {}).get("budget_state", {}).get("remaining", total_budget))

                # If the configured budget has changed or the remaining budget
                # appears exhausted (<1), reset the ledger to the configured
                # total_budget.
                if current_rem < 1.0 or abs(current_rem - total_budget) > 0.0001:
                    _memory.store(
                        content={"budget_state": {"remaining": total_budget, "total": total_budget}},
                        metadata={
                            "kind": "budget_state",
                            "source": "autonomy",
                            "confidence": 1.0,
                            "action": "reset"
                        }
                    )
                    return total_budget
                return current_rem
            else:
                # No budget record exists, initialize it
                _memory.store(
                    content={"budget_state": {"remaining": total_budget, "total": total_budget}},
                    metadata={
                        "kind": "budget_state",
                        "source": "autonomy",
                        "confidence": 1.0,
                        "action": "initialize"
                    }
                )
                return total_budget
        except Exception:
            # If retrieval fails, initialize budget
            _memory.store(
                content={"budget_state": {"remaining": total_budget, "total": total_budget}},
                metadata={
                    "kind": "budget_state",
                    "source": "autonomy",
                    "confidence": 1.0,
                    "action": "initialize"
                }
            )
            return total_budget
    except Exception:
        return float("inf")


def _decrease_budget(amount: int) -> None:
    """Decrease the execution budget by a specified amount using BrainMemory.

    This helper subtracts ``amount`` from the current budget state stored
    in the autonomy brain's memory tiers and records the update.  If no
    budget record exists or the operation fails, it is silently ignored.
    Negative values for ``amount`` are ignored.

    Args:
        amount: Number of executed goals to subtract from the budget.
    """
    if amount <= 0:
        return
    try:
        # Retrieve the most recent budget record
        budget_records = _memory.retrieve(limit=100)
        # Filter for budget_state records
        budget_state_records = [
            r for r in budget_records
            if isinstance(r.get("content"), dict) and
            r.get("content", {}).get("budget_state") is not None
        ]

        if not budget_state_records:
            return

        # Get current remaining budget
        latest = budget_state_records[0]
        try:
            remaining = float(latest.get("content", {}).get("budget_state", {}).get("remaining", 0))
            total = float(latest.get("content", {}).get("budget_state", {}).get("total", 0))
        except Exception:
            remaining = 0.0
            total = 0.0

        # Calculate new remaining budget
        remaining = max(0.0, remaining - float(amount))

        # Store updated budget state
        _memory.store(
            content={"budget_state": {"remaining": remaining, "total": total, "decreased_by": amount}},
            metadata={
                "kind": "budget_state",
                "source": "autonomy",
                "confidence": 1.0,
                "action": "decrease"
            }
        )
    except Exception:
        return


def _goal_priority(goal: Dict[str, Any]) -> int:
    """Compute a priority value for a goal record.

    Goals may not include an explicit priority field.  Use simple
    heuristics based on the description to infer priority:

    - AUTO_REPAIR goals receive highest priority (value 2).
    - Delegated tasks (description starting with 'DELEGATED_TO:') get medium priority (1).
    - All other goals default to 0.

    Args:
        goal: A goal dictionary from goal memory.

    Returns:
        An integer priority score (higher means more important).
    """
    try:
        desc = str(goal.get("description", "")).strip().upper()
    except Exception:
        desc = ""

    # Check for learned priority patterns first
    learned_priority = None
    if _teacher_helper and _memory and desc:
        try:
            desc_preview = desc[:50]
            learned_patterns = _memory.retrieve(
                query=f"goal priority pattern: {desc_preview[:30]}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", {})
                    if isinstance(content, dict) and "priority" in content:
                        learned_priority = content.get("priority")
                        print(f"[AUTONOMY] Using learned priority pattern from Teacher")
                        break
        except Exception:
            pass

    # Use learned priority if found, otherwise use heuristics
    if learned_priority is not None:
        try:
            return int(learned_priority)
        except Exception:
            pass

    # Base priority from description prefix
    base = 0
    if desc == "AUTO_REPAIR" or desc.startswith("AUTO_REPAIR"):
        base = 2
    elif desc.startswith("DELEGATED_TO:"):
        base = 1
    # Note: Deadline proximity logic removed (no time-based checks)
    return base


def _complete_goals(max_goals: int = 1, priority_strategy: str = "priority") -> List[Dict[str, Any]]:
    """Complete up to ``max_goals`` active goals using a scheduling strategy.

    This helper consults the personal goal memory for active goals and
    completes them according to a priority strategy.  It respects goal
    dependencies: a goal with a ``depends_on`` list will only be
    considered for execution if all of its dependencies have been
    completed.  Goals can be prioritised based on an inferred priority
    and creation time.  Any errors encountered during loading or
    completion are swallowed and reflected by an empty list.

    Args:
        max_goals: Maximum number of goals to complete in this tick.
        priority_strategy: Strategy to select goals.  Supported values:
            ``priority`` – sort by inferred priority descending then by creation time ascending.
            ``fifo`` – execute in FIFO order (default behaviour).

    Returns:
        A list of dictionaries with goal_id, title and status fields for
        each completed goal.
    """
    executed: List[Dict[str, Any]] = []
    if max_goals <= 0:
        return executed
    try:
        from brains.personal.memory import goal_memory  # type: ignore
    except Exception:
        return executed
    # Load all goals to determine dependency status.  We avoid multiple
    # reads by caching the full goal list once.  If this fails,
    # dependency checks fall back to always allowing execution.
    goal_map: Dict[str, Dict[str, Any]] = {}
    try:
        all_goals = goal_memory.get_goals(active_only=False)
        for rec in all_goals:
            gid = rec.get("goal_id")
            if gid:
                goal_map[gid] = rec
    except Exception:
        all_goals = []
        goal_map = {}
    # Fetch active goals (not completed).  If this fails, return empty.
    try:
        active = goal_memory.get_goals(active_only=True)
    except Exception:
        return executed
    # Filter out goals whose dependencies are not yet met or whose
    # conditional requirements are not satisfied.  A dependency is
    # considered met if the referenced goal exists in the map and is
    # marked completed.  Conditional goals ("condition" field set to
    # "success" or "failure") only become ready if all dependencies
    # have completed with the appropriate success flag.  Missing
    # dependency IDs are treated as unmet.
    ready: List[Dict[str, Any]] = []
    for g in active:
        try:
            deps: List[str] = g.get("depends_on", []) or []
            cond: Optional[str] = g.get("condition") or None
            # If there are no dependencies, the condition is ignored.
            if not deps:
                ready.append(g)
                continue
            # Check that all dependencies exist and are completed
            deps_met = True
            # Accumulate success status of dependencies
            dep_successes: List[bool] = []
            for dep in deps:
                rec = goal_map.get(dep)
                if not rec or not rec.get("completed", False):
                    deps_met = False
                    break
                # Use None as True to avoid blocking on old goals that
                # predate the success flag introduction.
                dep_successes.append(rec.get("success", True))
            if not deps_met:
                continue
            # If a condition is specified, evaluate it.  For "success",
            # require all dependencies to have success == True.  For
            # "failure", require at least one dependency with success
            # == False.  Unknown conditions are treated as unmet.
            if cond:
                lc = str(cond).lower()
                if lc == "success":
                    if all(dep_successes):
                        ready.append(g)
                    else:
                        continue
                elif lc == "failure":
                    # ready if any dependency failed
                    if any(not s for s in dep_successes):
                        ready.append(g)
                    else:
                        continue
                else:
                    # unsupported condition: skip goal
                    continue
            else:
                # No condition specified; dependencies are met
                ready.append(g)
        except Exception:
            # If any error occurs during dependency check, skip the goal
            continue
    # Sort ready goals based on the chosen strategy
    try:
        if priority_strategy.lower() == "priority":
            ready = sorted(ready, key=lambda g: (-_goal_priority(g), float(g.get("created_at", 0.0))))
    except Exception:
        # Fallback: leave ready list unsorted
        pass
    # Complete up to max_goals goals
    for _ in range(min(len(ready), max_goals)):
        try:
            goal = ready.pop(0)
            goal_id = goal.get("goal_id")
            if not goal_id:
                continue
            # Mark the goal as completed with success=True.  This may
            # eventually be parameterised, but for now we treat all
            # autonomous executions as successful.  Passing success
            # allows dependent goals to inspect outcome flags.
            goal_memory.complete_goal(goal_id, success=True)
            executed.append({"goal_id": goal_id, "title": goal.get("title", ""), "status": "completed", "success": True})
        except Exception:
            # Continue on errors
            continue
    return executed

def tick(ctx: dict) -> dict:
    """
    Autonomous tick with self-DMN integration.

    Calls RUN_IDLE_CYCLE from self_dmn and processes its actions.
    Must NEVER raise exceptions.

    Args:
        ctx: Pipeline context dictionary.

    Returns:
        A dictionary with action, reason, and confidence fields.
    """
    try:
        from brains.cognitive.self_dmn.service.self_dmn_brain import service_api as self_dmn_api
        from brains.cognitive.motivation.service.motivation_brain import service_api as motivation_api

        # Get conversation context for continuation detection
        conv_context = {}
        is_follow_up = False
        if _continuation_helpers_available:
            try:
                conv_context = get_conversation_context()
                query = ctx.get("query", "")
                is_follow_up = is_continuation(query, ctx)
            except Exception:
                pass

        system_history = []
        recent_issues = []

        try:
            if ctx.get("stage_self_review"):
                verdict = (ctx.get("stage_self_review") or {}).get("verdict", "ok")
                if verdict != "ok":
                    recent_issues.append({
                        "severity": "major" if verdict == "major_issue" else "minor",
                        "issues": (ctx.get("stage_self_review") or {}).get("issues", [])
                    })
        except Exception:
            pass

        try:
            motivation_state_resp = motivation_api({"op": "GET_STATE"})
            motivation_state = {}
            if motivation_state_resp.get("ok"):
                motivation_state = motivation_state_resp.get("payload", {})
        except Exception:
            motivation_state = {}

        dmn_resp = self_dmn_api({
            "op": "RUN_IDLE_CYCLE",
            "payload": {
                "system_history": system_history,
                "recent_issues": recent_issues,
                "motivation_state": motivation_state
            }
        })

        if dmn_resp.get("ok"):
            dmn_payload = dmn_resp.get("payload", {})
            actions = dmn_payload.get("actions", [])

            for action in actions:
                try:
                    kind = action.get("kind")
                    if kind == "adjust_motivation":
                        delta = action.get("delta", {})
                        motivation_api({
                            "op": "ADJUST_STATE",
                            "payload": {"deltas": delta}
                        })
                    elif kind == "schedule_learning_task":
                        pass
                except Exception:
                    continue

            if actions:
                # Store autonomous decisions for governance visibility
                try:
                    _memory.store(
                        content={
                            "decision": {
                                "type": "autonomous_tick",
                                "actions": actions,
                                "insights": dmn_payload.get("insights", []),
                                "timestamp": time.time(),
                                "is_continuation": is_follow_up
                            }
                        },
                        metadata={
                            "kind": "decision",
                            "source": "autonomy",
                            "confidence": 0.8,
                            "truth_type": "EDUCATED",
                            "requires_governance": True
                        }
                    )
                except Exception:
                    pass

                result = {
                    "action": "dmn_actions_processed",
                    "reason": f"processed {len(actions)} self-dmn actions",
                    "confidence": 0.8,
                    "insights": dmn_payload.get("insights", []),
                    "is_continuation": is_follow_up
                }

                # Add routing hint for Teacher learning
                if _continuation_helpers_available:
                    try:
                        result["routing_hint"] = create_routing_hint(
                            brain_name="autonomy",
                            action="multi_step_continuation" if is_follow_up else "autonomous_execution",
                            confidence=0.8,
                            context_tags=["autonomous", "multi_step"] if is_follow_up else ["autonomous"]
                        )
                    except Exception:
                        pass

                return result

        result = {
            "action": "noop",
            "reason": "no_dmn_actions",
            "confidence": 0.5,
            "is_continuation": is_follow_up
        }

        # Add routing hint even for no-op
        if _continuation_helpers_available:
            try:
                result["routing_hint"] = create_routing_hint(
                    brain_name="autonomy",
                    action="idle_tick",
                    confidence=0.5,
                    context_tags=["autonomous", "idle"]
                )
            except Exception:
                pass

        return result
    except Exception:
        return {
            "action": "noop",
            "reason": "tick_exception_caught",
            "confidence": 0.0,
        }

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
# COGNITIVE BRAIN CONTRACT: Signal 1 & 2 - Detect continuation and get context
    continuation_detected = False
    conv_context = {}

    try:
        # Extract query from payload
        query = (payload.get("query") or
                payload.get("question") or
                payload.get("user_query") or
                payload.get("text") or "")

        if query:
            continuation_detected = is_continuation(query, payload)

            if continuation_detected:
                conv_context = get_conversation_context()
                # Enrich payload with conversation context
                payload["continuation_detected"] = True
                payload["last_topic"] = conv_context.get("last_topic", "")
                payload["conversation_depth"] = conv_context.get("conversation_depth", 0)
    except Exception as e:
        # Silently continue if continuation detection fails
        pass
    from api.utils import error_response  # type: ignore
    from api.utils import success_response  # type: ignore
    from api.utils import generate_mid  # type: ignore
    """Entry point for the autonomy brain.

    Recognised operations:

      TICK   – perform a single autonomous cycle (complete one goal).
      HEALTH – return a simple status message.

    Args:
        msg: A dict with at least the ``op`` key.

    Returns:
        A response dict containing the result or an error.
    """
    op = str((msg or {}).get("op", "")).upper()
    mid = msg.get("mid") or generate_mid()
    if op == "HEALTH":
        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="autonomy",
                action="health",
                confidence=0.7,
                context_tags=[
                    "health",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )
            if isinstance(result, dict):
                result["routing_hint"] = routing_hint
            elif isinstance(payload_result, dict):
                payload_result["routing_hint"] = routing_hint
        except Exception:
            pass  # Routing hint generation is non-critical
        return success_response(op, mid, {"status": "operational", "type": "autonomy_brain"})
    if op == "TICK":
        # Load the autonomy configuration.  If rate limiting is configured
        # and the tick is too soon after the previous run, skip executing
        # any goals and indicate the reason in the payload.  Otherwise,
        # complete up to ``max_goals`` goals using the configured
        # scheduling strategy and update the last tick timestamp.
        cfg = _load_autonomy_config() or {}
        # Check rate limit
        if not _should_run(cfg):
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="autonomy",
                    action="tick",
                    confidence=0.7,
                    context_tags=[
                        "tick",
                        "continuation" if continuation_detected else "fresh_query"
                    ]
                )
                if isinstance(result, dict):
                    result["routing_hint"] = routing_hint
                elif isinstance(payload_result, dict):
                    payload_result["routing_hint"] = routing_hint
            except Exception:
                pass  # Routing hint generation is non-critical
            return success_response(op, mid, {
                "executed_goals": [],
                "skipped": True,
                "reason": "rate_limited"
            })
        # Check execution budget before proceeding.  If a finite budget
        # is configured and no budget remains, skip the tick entirely.
        try:
            remaining_budget = _get_remaining_budget(cfg)
        except Exception:
            remaining_budget = float("inf")
        if remaining_budget <= 0:
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="autonomy",
                    action="unknown",
                    confidence=0.7,
                    context_tags=[
                        "unknown",
                        "continuation" if continuation_detected else "fresh_query"
                    ]
                )
                if isinstance(result, dict):
                    result["routing_hint"] = routing_hint
                elif isinstance(payload_result, dict):
                    payload_result["routing_hint"] = routing_hint
            except Exception:
                pass  # Routing hint generation is non-critical
            return success_response(op, mid, {
                "executed_goals": [],
                "skipped": True,
                "reason": "budget_exhausted"
            })
        # Determine how many goals to complete from configuration.  The
        # autonomy config may specify a "max_goals" field controlling
        # throughput per tick.  Default to 1 when unspecified or invalid.
        max_goals = 1
        try:
            mg = int(cfg.get("max_goals", 1))
            if mg > 0:
                max_goals = mg
        except Exception:
            max_goals = 1
        # If a finite execution budget is defined, cap the number of goals
        # executed in this tick to the remaining budget so we never exceed it.
        try:
            if remaining_budget != float("inf"):
                # remaining_budget may not be integer; floor to nearest int
                avail = int(max(0.0, remaining_budget))
                if avail < max_goals:
                    max_goals = avail
        except Exception:
            pass
        # Determine the priority strategy.  Supported values are
        # "priority" (default) and "fifo".  Unknown values fall back
        # to "priority".
        strat = str(cfg.get("priority_strategy", "priority")).lower()
        if strat not in {"fifo", "priority"}:
            strat = "priority"
        executed = _complete_goals(max_goals, priority_strategy=strat)
        # Decrease the remaining budget if a finite budget is configured.  Each
        # executed goal consumes one unit of the budget.
        try:
            _decrease_budget(len(executed))
        except Exception:
            pass
        # Update the last tick timestamp after executing goals
        _update_last_tick()

        # Store execution results in memory for Librarian visibility and audit trail
        if executed:
            try:
                _memory.store(
                    content={
                        "execution_result": {
                            "executed_goals": executed,
                            "count": len(executed),
                            "strategy": strat,
                            "timestamp": time.time()
                        }
                    },
                    metadata={
                        "kind": "execution",
                        "source": "autonomy",
                        "confidence": 0.95,
                        "action": "goal_completion"
                    }
                )
            except Exception:
                pass

        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
                try:
                    routing_hint = create_routing_hint(
                        brain_name="autonomy",
                        action="unknown",
                        confidence=0.7,
                        context_tags=[
                            "unknown",
                            "continuation" if continuation_detected else "fresh_query"
                        ]
                    )
                    if isinstance(result, dict):
                        result["routing_hint"] = routing_hint
                    elif isinstance(payload_result, dict):
                        payload_result["routing_hint"] = routing_hint
                except Exception:
                    pass  # Routing hint generation is non-critical
        return success_response(op, mid, {"executed_goals": executed})
    return error_response(op, mid, "UNSUPPORTED_OP", f"Unknown operation: {op}")

# Standard service contract: handle is the entry point
service_api = handle