
from __future__ import annotations
import json
from typing import Dict, Any, List
from pathlib import Path
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning abstraction and generalization patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("abstraction")
except Exception as e:
    print(f"[ABSTRACTION] Teacher helper not available: {e}")
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
    print(f"[ABSTRACTION] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("abstraction")

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

# Concept ID counter (deterministic, monotonic)
_CONCEPT_ID_COUNTER: int = 0

def _next_concept_id() -> int:
    """Generate next concept ID deterministically."""
    global _CONCEPT_ID_COUNTER
    _CONCEPT_ID_COUNTER += 1
    return _CONCEPT_ID_COUNTER

def _load_concepts() -> List[Dict[str, Any]]:
    """Load all concepts from BrainMemory."""
    try:
        results = _memory.retrieve()

        concepts = []
        for record in results:
            concept = record.get("content", {})
            if isinstance(concept, dict) and "concept_id" in concept:
                concepts.append(concept)
                # Update counter to max
                cid = concept.get("concept_id", 0)
                if isinstance(cid, int):
                    global _CONCEPT_ID_COUNTER
                    _CONCEPT_ID_COUNTER = max(_CONCEPT_ID_COUNTER, cid)

        return concepts
    except Exception:
        return []

def _save_concept(concept: Dict[str, Any]) -> None:
    """Save a concept to BrainMemory."""
    try:
        importance = concept.get("importance", 0.7)

        _memory.store(
            content=concept,
            metadata={
                "kind": "concept",
                "source": "abstraction",
                "confidence": importance
            }
        )
    except Exception:
        pass

def _create_concept(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a concept from a pattern.

    Args:
        pattern: Pattern dict with pattern_type, subject/intent/topic, etc.

    Returns:
        Concept record with structured attributes.
    """
    pattern_type = pattern.get("pattern_type", "")

    # Check for learned abstraction patterns first
    learned_concept = None
    if _teacher_helper and _memory and pattern_type:
        try:
            learned_patterns = _memory.retrieve(
                query=f"abstraction pattern: {pattern_type}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, dict) and "name" in content:
                        learned_concept = content
                        print(f"[ABSTRACTION] Using learned abstraction pattern from Teacher")
                        break
        except Exception:
            pass

    # Use learned pattern if found, otherwise use built-in heuristics
    if learned_concept:
        concept = {
            "concept_id": _next_concept_id(),
            "name": learned_concept.get("name", f"concept_{pattern_type}"),
            "attributes": learned_concept.get("attributes", []),
            "derived_from_pattern": pattern,
            "tier": learned_concept.get("tier", "LONG"),
            "importance": min(1.0, learned_concept.get("importance", 0.7))
        }
    else:
        # Determine concept name and attributes based on pattern type
        if pattern_type == "preference_cluster":
            name = f"preference_{pattern.get('subject', 'unknown')}"
            attributes = [f"likes_{pattern.get('subject', 'unknown')}"]
            importance = 0.8 + (pattern.get("consistency", 0.0) * 0.2)
        elif pattern_type == "recurring_intent":
            name = f"intent_pattern_{pattern.get('intent', 'unknown').lower()}"
            attributes = [f"frequent_{pattern.get('intent', 'unknown').lower()}_queries"]
            importance = 0.7 + (pattern.get("consistency", 0.0) * 0.2)
        elif pattern_type == "domain_focus":
            name = f"domain_{pattern.get('topic', 'unknown')}"
            attributes = [f"focus_on_{pattern.get('topic', 'unknown')}"]
            importance = 0.75 + (pattern.get("consistency", 0.0) * 0.15)
        elif pattern_type == "relation_structure":
            name = f"relation_{pattern.get('relation_type', 'unknown')}"
            attributes = [f"has_{pattern.get('relation_type', 'unknown')}_relationships"]
            importance = 0.7 + (pattern.get("consistency", 0.0) * 0.2)
        else:
            name = f"concept_{pattern_type}"
            attributes = []
            importance = 0.6

        concept = {
            "concept_id": _next_concept_id(),
            "name": name,
            "attributes": attributes,
            "derived_from_pattern": pattern,
            "tier": "LONG",
            "importance": min(1.0, importance)
        }

        # If no learned pattern and Teacher available, try to learn
        if _teacher_helper and pattern_type:
            try:
                print(f"[ABSTRACTION] No learned pattern for {pattern_type}, calling Teacher...")
                teacher_result = _teacher_helper.maybe_call_teacher(
                    question=f"How should I abstract pattern_type={pattern_type} into a concept?",
                    context={
                        "pattern_type": pattern_type,
                        "pattern": pattern,
                        "current_concept": concept
                    },
                    check_memory_first=True
                )

                if teacher_result and teacher_result.get("answer"):
                    patterns_stored = teacher_result.get("patterns_stored", 0)
                    print(f"[ABSTRACTION] Learned from Teacher: {patterns_stored} abstraction patterns stored")
                    # Learned pattern now in memory for future use
            except Exception as e:
                print(f"[ABSTRACTION] Teacher call failed: {str(e)[:100]}")

    return concept

def _update_concept(concept_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing concept.

    Args:
        concept_id: ID of concept to update
        updates: Dict of fields to update

    Returns:
        Updated concept or error dict.
    """
    concepts = _load_concepts()

    for i, concept in enumerate(concepts):
        if concept.get("concept_id") == concept_id:
            # Apply updates
            for key, value in updates.items():
                if key != "concept_id":  # Don't allow ID changes
                    concept[key] = value

            # Store updated concept with BrainMemory
            try:
                memory = BrainMemory("abstraction")
                importance = concept.get("importance", 0.7)

                memory.store(
                    content=concept,
                    metadata={
                        "kind": "concept_update",
                        "source": "abstraction",
                        "confidence": importance
                    }
                )
            except Exception:
                return {"error": "Failed to update concept in memory"}

            return concept

    return {"error": f"Concept {concept_id} not found"}

def _query_concepts(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Query concepts by filters.

    Args:
        filters: Dict with optional keys: name, tier, min_importance

    Returns:
        List of matching concepts.
    """
    concepts = _load_concepts()
    results = []

    name_filter = filters.get("name", "")
    tier_filter = filters.get("tier", "")
    min_importance = filters.get("min_importance", 0.0)

    for concept in concepts:
        # Apply filters
        if name_filter and name_filter not in concept.get("name", ""):
            continue
        if tier_filter and concept.get("tier", "") != tier_filter:
            continue
        if concept.get("importance", 0.0) < min_importance:
            continue

        results.append(concept)

    return results

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Abstraction brain service API.

    Supported operations:
    - HEALTH: Health check
    - CREATE_CONCEPT: Create concept from pattern
    - UPDATE_CONCEPT: Update existing concept
    - QUERY_CONCEPT: Query concepts by filters
    """
    from api.utils import generate_mid, success_response, error_response  # type: ignore

    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}
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

    if op == "HEALTH":
        concepts = _load_concepts()
        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="abstraction",
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
        return success_response(op, mid, {
            "status": "operational",
            "concept_count": len(concepts)
        })

    if op == "CREATE_CONCEPT":
        pattern = payload.get("pattern", {})
        if not pattern:
            return error_response(op, mid, "MISSING_PATTERN", "Pattern required")

        # CONTINUATION AWARENESS: Detect abstraction ladder movement
        is_abstraction_refinement = False
        abstraction_direction = None
        base_concept = None
        conv_context = {}

        if _continuation_helpers_available:
            try:
                pattern_text = str(pattern.get("subject", "") or pattern.get("topic", "") or pattern.get("intent", ""))
                is_abstraction_refinement = is_continuation(pattern_text, {"pattern": pattern})

                if is_abstraction_refinement:
                    conv_context = get_conversation_context()
                    last_topic = conv_context.get("last_topic", "")

                    # Detect abstraction direction from language cues
                    if any(phrase in pattern_text.lower() for phrase in ["more general", "abstract", "generalize", "broader", "higher level"]):
                        abstraction_direction = "up"
                        print(f"[ABSTRACTION] ✓ Moving UP abstraction ladder from: {last_topic}")
                    elif any(phrase in pattern_text.lower() for phrase in ["more specific", "concrete", "example", "instance", "detail"]):
                        abstraction_direction = "down"
                        print(f"[ABSTRACTION] ✓ Moving DOWN abstraction ladder from: {last_topic}")
                    else:
                        abstraction_direction = "lateral"
                        print(f"[ABSTRACTION] ✓ Lateral abstraction movement from: {last_topic}")

                    base_concept = last_topic
            except Exception as e:
                print(f"[ABSTRACTION] Warning: Continuation detection failed: {str(e)[:100]}")
                is_abstraction_refinement = False

        concept = _create_concept(pattern)

        # Add abstraction metadata
        if is_abstraction_refinement:
            concept["abstraction_metadata"] = {
                "is_refinement": True,
                "direction": abstraction_direction,
                "base_concept": base_concept
            }

        _save_concept(concept)

        # Create routing hint
        routing_hint = None
        if _continuation_helpers_available:
            try:
                if is_abstraction_refinement:
                    action = f"abstract_{abstraction_direction}"
                    context_tags = ["abstraction", "ladder", abstraction_direction]
                else:
                    action = "create_concept"
                    context_tags = ["abstraction", "fresh"]

                routing_hint = create_routing_hint(
                    brain_name="abstraction",
                    action=action,
                    confidence=concept.get("importance", 0.7),
                    context_tags=context_tags
                )
            except Exception as e:
                print(f"[ABSTRACTION] Warning: Failed to create routing hint: {str(e)[:100]}")

        result = {"concept": concept}
        if routing_hint:
            result["routing_hint"] = routing_hint

        return success_response(op, mid, result)

    if op == "UPDATE_CONCEPT":
        concept_id = payload.get("concept_id")
        updates = payload.get("updates", {})

        if concept_id is None:
            return error_response(op, mid, "MISSING_ID", "Concept ID required")

        result = _update_concept(int(concept_id), updates)

        if "error" in result:
            return error_response(op, mid, "UPDATE_FAILED", result["error"])

        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="abstraction",
                action="update_concept",
                confidence=0.7,
                context_tags=[
                    "update_concept",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )
            if isinstance(result, dict):
                result["routing_hint"] = routing_hint
            elif isinstance(payload_result, dict):
                payload_result["routing_hint"] = routing_hint
        except Exception:
            pass  # Routing hint generation is non-critical
        return success_response(op, mid, {"concept": result})

    if op == "QUERY_CONCEPT":
        filters = payload.get("filters", {})
        concepts = _query_concepts(filters)

        return success_response(op, mid, {
            "concepts": concepts,
            "count": len(concepts)
        })

    return error_response(op, mid, "UNSUPPORTED_OP", op)

# Standard service contract: handle is the entry point
service_api = handle
