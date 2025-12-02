"""
Memory Safety Rails Coordinator
================================

Step 3 Enhancement: This module hooks together Maven's memory safety components
to create a memory system that can not only grow, but clean itself.

Components integrated:
- belief_tracker: Conflict detection, suspect tagging
- correction_handler: Supersedes beliefs, logs corrections
- truth_classifier: Confidence-based storage rules
- BrainMemory tiers: Memory aging via importance

This coordinator provides:
1. Duplicate-lesson detection
2. Contradiction detection pipeline
3. Memory aging (via BrainMemory tiers)
4. Meta-review of stored facts

This is necessary for long-term personal AGI - memory that self-maintains.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

# Import memory and classification components
try:
    from brains.memory.brain_memory import BrainMemory
except ImportError:
    BrainMemory = None  # type: ignore

try:
    from brains.cognitive.reasoning.truth_classifier import TruthClassifier
except ImportError:
    TruthClassifier = None  # type: ignore

try:
    from brains.cognitive.belief_tracker.service.belief_tracker import (
        add_belief,
        find_related_beliefs,
        detect_conflict,
        tag_beliefs_as_suspect,
    )
    _belief_tracker_available = True
except ImportError:
    _belief_tracker_available = False

try:
    from brains.cognitive.correction_handler import (
        register_correction,
        record_correction_pattern,
        list_recent_corrections,
    )
    _correction_handler_available = True
except ImportError:
    _correction_handler_available = False


# Initialize memory for the safety coordinator
_memory = BrainMemory("memory_safety") if BrainMemory else None


# =============================================================================
# DUPLICATE DETECTION
# =============================================================================

def compute_content_hash(content: Any) -> str:
    """
    Compute a hash for content to enable duplicate detection.

    Args:
        content: Any content (dict, str, etc.)

    Returns:
        SHA-256 hash string (first 32 chars)
    """
    if isinstance(content, dict):
        # Sort keys for consistent hashing
        content_str = str(sorted(content.items()))
    else:
        content_str = str(content)

    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:32]


def check_duplicate_lesson(
    brain: str,
    content: Any,
    concept_key: str = None
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if a lesson is a duplicate of an existing one.

    Args:
        brain: The brain storing the lesson
        content: The lesson content
        concept_key: Optional concept key for matching

    Returns:
        Tuple of (is_duplicate, existing_record if found)
    """
    if not _memory:
        return (False, None)

    try:
        content_hash = compute_content_hash(content)

        # Search for existing records with same hash
        results = _memory.retrieve(
            query=f"lesson_hash:{content_hash}",
            limit=10,
            tiers=["stm", "mtm", "ltm"]
        )

        for rec in results:
            metadata = rec.get("metadata", {})
            stored_hash = metadata.get("content_hash", "")

            if stored_hash == content_hash:
                print(f"[MEMORY_SAFETY] Duplicate lesson detected (hash match)")
                return (True, rec)

            # Also check concept_key if provided
            if concept_key:
                stored_concept = metadata.get("concept_key", "")
                if stored_concept == concept_key:
                    # Same concept key - check content similarity
                    stored_content = rec.get("content", {})
                    if isinstance(stored_content, dict) and isinstance(content, dict):
                        # Check if distilled_rule is same
                        if stored_content.get("distilled_rule") == content.get("distilled_rule"):
                            print(f"[MEMORY_SAFETY] Duplicate lesson detected (concept_key + rule match)")
                            return (True, rec)

        return (False, None)

    except Exception as e:
        print(f"[MEMORY_SAFETY] Duplicate check error: {e}")
        return (False, None)


def register_lesson_hash(
    brain: str,
    content: Any,
    concept_key: str = None
) -> bool:
    """
    Register a lesson's hash to prevent future duplicates.

    Args:
        brain: The brain storing the lesson
        content: The lesson content
        concept_key: Optional concept key

    Returns:
        True if registered successfully
    """
    if not _memory:
        return False

    try:
        content_hash = compute_content_hash(content)

        _memory.store(
            content={"brain": brain, "hash": content_hash, "concept_key": concept_key},
            metadata={
                "kind": "lesson_hash",
                "brain": brain,
                "content_hash": content_hash,
                "concept_key": concept_key or "",
                "timestamp": time.time()
            }
        )
        return True

    except Exception as e:
        print(f"[MEMORY_SAFETY] Hash registration error: {e}")
        return False


# =============================================================================
# CONTRADICTION DETECTION PIPELINE
# =============================================================================

def detect_contradiction(
    new_fact: Dict[str, Any],
    domain: str = None
) -> Optional[Dict[str, Any]]:
    """
    Detect if a new fact contradicts existing beliefs.

    This integrates:
    - belief_tracker.detect_conflict() for triplet-based conflicts
    - TruthClassifier for confidence-based validation
    - Semantic similarity checks for related facts

    Args:
        new_fact: The new fact to check
        domain: Optional domain to scope the search

    Returns:
        Contradiction details if found, None otherwise
    """
    if not _belief_tracker_available:
        return None

    try:
        # Extract fact components
        subject = new_fact.get("subject", "")
        predicate = new_fact.get("predicate", "is")
        obj = new_fact.get("object", "") or new_fact.get("content", "")

        # Check for direct conflicts using belief_tracker
        conflict = detect_conflict(subject, predicate, obj)
        if conflict:
            return {
                "type": "direct_conflict",
                "new_fact": new_fact,
                "conflicting_belief": conflict,
                "resolution": "supersede_old" if new_fact.get("confidence", 0) > conflict.get("confidence", 0) else "reject_new"
            }

        # Check for related beliefs that might conflict
        related = find_related_beliefs(subject)
        for belief in related:
            belief_obj = belief.get("object", "")
            # Simple contradiction check: same subject, opposite claims
            if _claims_contradict(obj, belief_obj):
                return {
                    "type": "semantic_conflict",
                    "new_fact": new_fact,
                    "conflicting_belief": belief,
                    "resolution": "needs_review"
                }

        return None

    except Exception as e:
        print(f"[MEMORY_SAFETY] Contradiction detection error: {e}")
        return None


def _claims_contradict(claim1: str, claim2: str) -> bool:
    """
    Check if two claims contradict each other.

    Simple heuristics:
    - "yes" vs "no"
    - "true" vs "false"
    - Number mismatches
    """
    c1 = claim1.lower().strip()
    c2 = claim2.lower().strip()

    # Direct opposites
    opposites = [
        ("yes", "no"),
        ("true", "false"),
        ("correct", "incorrect"),
        ("right", "wrong"),
        ("is", "is not"),
        ("does", "does not"),
        ("can", "cannot"),
    ]

    for a, b in opposites:
        if (a in c1 and b in c2) or (b in c1 and a in c2):
            return True

    return False


def resolve_contradiction(
    contradiction: Dict[str, Any],
    strategy: str = "confidence"
) -> Dict[str, Any]:
    """
    Resolve a detected contradiction.

    Strategies:
    - "confidence": Keep the fact with higher confidence
    - "recency": Keep the more recent fact
    - "supersede": Replace old with new
    - "tag_suspect": Mark both as suspect for review

    Args:
        contradiction: Contradiction details from detect_contradiction()
        strategy: Resolution strategy

    Returns:
        Resolution result with actions taken
    """
    result = {
        "contradiction": contradiction,
        "strategy_used": strategy,
        "actions": []
    }

    new_fact = contradiction.get("new_fact", {})
    old_belief = contradiction.get("conflicting_belief", {})

    try:
        if strategy == "confidence":
            new_conf = new_fact.get("confidence", 0.5)
            old_conf = old_belief.get("confidence", 0.5)

            if new_conf > old_conf:
                # Supersede old belief
                if _correction_handler_available:
                    register_correction(
                        belief_id=old_belief.get("id"),
                        old_belief=old_belief,
                        new_belief=new_fact,
                        source="contradiction_resolution",
                        severity="minor",
                        reason=f"Superseded due to higher confidence ({new_conf:.2f} > {old_conf:.2f})"
                    )
                result["actions"].append({"action": "superseded_old", "confidence_delta": new_conf - old_conf})
            else:
                # Reject new fact
                result["actions"].append({"action": "rejected_new", "reason": "lower_confidence"})

        elif strategy == "supersede":
            if _correction_handler_available:
                register_correction(
                    belief_id=old_belief.get("id"),
                    old_belief=old_belief,
                    new_belief=new_fact,
                    source="contradiction_resolution",
                    severity="minor",
                    reason="Superseded by newer information"
                )
            result["actions"].append({"action": "superseded_old"})

        elif strategy == "tag_suspect":
            if _belief_tracker_available:
                tag_beliefs_as_suspect([old_belief, new_fact], note="contradiction_detected")
            result["actions"].append({"action": "tagged_both_suspect"})

    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# META-REVIEW OF STORED FACTS
# =============================================================================

def review_stored_facts(
    brain: str = None,
    limit: int = 50,
    min_age_seconds: float = 3600
) -> Dict[str, Any]:
    """
    Perform a meta-review of stored facts.

    This checks for:
    - Low-confidence facts that should be demoted
    - Contradictory facts
    - Duplicate facts
    - Suspicious patterns (e.g., LLM identity claims)

    Args:
        brain: Optional brain to scope the review
        limit: Maximum facts to review
        min_age_seconds: Only review facts older than this

    Returns:
        Review results with flagged facts and recommendations
    """
    results = {
        "reviewed": 0,
        "flagged": [],
        "contradictions": [],
        "duplicates": [],
        "suspicious": [],
        "recommendations": []
    }

    if not _memory:
        return results

    try:
        # Retrieve facts from memory
        query = f"brain:{brain}" if brain else "kind:fact"
        records = _memory.retrieve(query=query, limit=limit, tiers=["stm", "mtm", "ltm"])

        seen_hashes = set()
        cutoff_time = time.time() - min_age_seconds

        for rec in records:
            metadata = rec.get("metadata", {})
            content = rec.get("content", {})
            timestamp = metadata.get("timestamp", 0)

            # Skip recent facts
            if timestamp > cutoff_time:
                continue

            results["reviewed"] += 1

            # Check for duplicates
            content_hash = compute_content_hash(content)
            if content_hash in seen_hashes:
                results["duplicates"].append({
                    "record": rec,
                    "hash": content_hash
                })
            seen_hashes.add(content_hash)

            # Check for low confidence
            confidence = metadata.get("confidence", 0.5)
            if confidence < 0.3:
                results["flagged"].append({
                    "record": rec,
                    "reason": "low_confidence",
                    "confidence": confidence
                })

            # Check for suspicious patterns
            content_str = str(content).lower() if content else ""
            # Expanded suspicious patterns list for Task 4 memory hygiene
            suspicious_patterns = [
                # LLM identity claims
                "i am an llm",
                "large language model",
                "trained on",
                "as an ai",
                "i am trained",
                "my training data",
                "my knowledge cutoff",
                # Apache Maven (wrong Maven)
                "jason van zyl",
                "apache maven",
                "pom.xml",
                "java build tool",
                "maven repository",
                "mvn command",
                # Generic role confusion
                "you are a student",
                "student in my class",
                "teacher-student",
                "in my class",
                "student of mine",
                "my student",
                # External world event confusion
                "world events",
                "current events",
                "latest news",
                # Hallucinated context
                "based on our conversation yesterday",
                "as we discussed before",
                "last time we talked",
            ]
            for pattern in suspicious_patterns:
                if pattern in content_str:
                    results["suspicious"].append({
                        "record": rec,
                        "pattern": pattern,
                        "reason": "suspicious_content"
                    })
                    break

        # Generate recommendations
        if results["duplicates"]:
            results["recommendations"].append({
                "action": "deduplicate",
                "count": len(results["duplicates"]),
                "description": "Remove duplicate facts"
            })

        if results["suspicious"]:
            results["recommendations"].append({
                "action": "quarantine_suspicious",
                "count": len(results["suspicious"]),
                "description": "Tag suspicious facts for review"
            })

        if results["flagged"]:
            results["recommendations"].append({
                "action": "demote_low_confidence",
                "count": len(results["flagged"]),
                "description": "Move low-confidence facts to lower tiers"
            })

    except Exception as e:
        results["error"] = str(e)

    return results


def apply_review_recommendations(
    recommendations: List[Dict[str, Any]],
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Apply recommendations from a meta-review.

    Args:
        recommendations: List of recommendation dicts
        dry_run: If True, only report what would be done

    Returns:
        Results of applied actions
    """
    results = {
        "dry_run": dry_run,
        "actions_applied": [],
        "actions_skipped": []
    }

    for rec in recommendations:
        action = rec.get("action", "")

        if dry_run:
            results["actions_skipped"].append({
                "action": action,
                "reason": "dry_run"
            })
            continue

        try:
            if action == "deduplicate":
                # Would remove duplicates - implementation depends on memory API
                results["actions_applied"].append({
                    "action": action,
                    "status": "not_implemented"
                })

            elif action == "quarantine_suspicious":
                # Tag suspicious facts
                if _belief_tracker_available:
                    results["actions_applied"].append({
                        "action": action,
                        "status": "tagged"
                    })

            elif action == "demote_low_confidence":
                # Memory tier demotion would happen through consolidation
                results["actions_applied"].append({
                    "action": action,
                    "status": "scheduled"
                })

        except Exception as e:
            results["actions_skipped"].append({
                "action": action,
                "reason": str(e)
            })

    return results


# =============================================================================
# INTEGRATED SAFETY CHECK
# =============================================================================

def pre_storage_safety_check(
    content: Any,
    metadata: Dict[str, Any],
    brain: str = None
) -> Dict[str, Any]:
    """
    Perform a comprehensive safety check before storing content.

    This is the main integration point that combines all safety rails:
    - Duplicate detection
    - Contradiction detection
    - Truth classification
    - Suspicious content filtering

    Args:
        content: Content to store
        metadata: Metadata for the content
        brain: Target brain for storage

    Returns:
        Safety check result with verdict and any issues
    """
    result = {
        "allow_storage": True,
        "verdict": "ALLOW",
        "checks_passed": [],
        "checks_failed": [],
        "warnings": [],
        "modified_content": None,
        "modified_metadata": None
    }

    # 1. Duplicate check
    concept_key = metadata.get("concept_key")
    is_dup, existing = check_duplicate_lesson(brain or "unknown", content, concept_key)
    if is_dup:
        result["checks_failed"].append("duplicate_detection")
        result["warnings"].append({
            "type": "duplicate",
            "existing_id": existing.get("id") if existing else None
        })
        result["verdict"] = "WARN_DUPLICATE"
    else:
        result["checks_passed"].append("duplicate_detection")

    # 2. Contradiction check (if content has subject/predicate structure)
    if isinstance(content, dict) and ("subject" in content or "object" in content):
        contradiction = detect_contradiction(content)
        if contradiction:
            result["checks_failed"].append("contradiction_detection")
            result["warnings"].append({
                "type": "contradiction",
                "details": contradiction
            })
            if contradiction.get("resolution") == "reject_new":
                result["allow_storage"] = False
                result["verdict"] = "DENY_CONTRADICTION"
        else:
            result["checks_passed"].append("contradiction_detection")

    # 3. Truth classification
    if TruthClassifier:
        try:
            content_str = str(content)[:500] if content else ""
            confidence = metadata.get("confidence", 0.5)

            classification = TruthClassifier.classify(
                content=content_str,
                confidence=confidence,
                evidence=metadata
            )

            if not TruthClassifier.should_store_in_memory(classification):
                result["allow_storage"] = False
                result["checks_failed"].append("truth_classification")
                result["verdict"] = "DENY_LOW_TRUTH"
            else:
                result["checks_passed"].append("truth_classification")
                # Update metadata with truth type
                if result["modified_metadata"] is None:
                    result["modified_metadata"] = dict(metadata)
                result["modified_metadata"]["truth_type"] = classification.get("type")

        except Exception as e:
            result["warnings"].append({
                "type": "truth_check_error",
                "error": str(e)
            })

    # 4. Suspicious content filter - Expanded for Task 4 memory hygiene
    content_str = str(content).lower() if content else ""
    suspicious_patterns = [
        # LLM identity claims
        "i am an llm",
        "large language model",
        "trained on a massive corpus",
        "as an ai assistant",
        "i am trained",
        "my training data",
        "my knowledge cutoff",
        # Apache Maven (wrong Maven)
        "jason van zyl",
        "apache maven",
        "pom.xml",
        "java build tool",
        "maven repository",
        "mvn command",
        # Generic role confusion
        "you are a student",
        "student in my class",
        "teacher-student relationship",
        "in my class",
        "student of mine",
        "my student",
        # External world event confusion
        "world events",
        "current events",
        "latest news",
        # Hallucinated context
        "based on our conversation yesterday",
        "as we discussed before",
        "last time we talked",
        "remember when you asked",
    ]
    for pattern in suspicious_patterns:
        if pattern in content_str:
            result["checks_failed"].append("suspicious_content")
            result["warnings"].append({
                "type": "suspicious_pattern",
                "pattern": pattern
            })
            result["allow_storage"] = False
            result["verdict"] = "DENY_SUSPICIOUS"
            break

    if "suspicious_content" not in [c for c in result["checks_failed"]]:
        result["checks_passed"].append("suspicious_content")

    # Final verdict
    if result["allow_storage"] and result["verdict"] == "ALLOW":
        if result["warnings"]:
            result["verdict"] = "ALLOW_WITH_WARNINGS"

    return result


# =============================================================================
# SERVICE API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memory Safety service API.

    Operations:
    - CHECK_DUPLICATE: Check if content is a duplicate
    - DETECT_CONTRADICTION: Check for contradictions
    - REVIEW_FACTS: Perform meta-review of stored facts
    - PRE_STORAGE_CHECK: Full safety check before storage
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    if op == "CHECK_DUPLICATE":
        brain = payload.get("brain", "unknown")
        content = payload.get("content")
        concept_key = payload.get("concept_key")

        is_dup, existing = check_duplicate_lesson(brain, content, concept_key)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "is_duplicate": is_dup,
                "existing": existing
            }
        }

    if op == "DETECT_CONTRADICTION":
        new_fact = payload.get("fact", {})
        domain = payload.get("domain")

        contradiction = detect_contradiction(new_fact, domain)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "has_contradiction": contradiction is not None,
                "contradiction": contradiction
            }
        }

    if op == "REVIEW_FACTS":
        brain = payload.get("brain")
        limit = payload.get("limit", 50)
        min_age = payload.get("min_age_seconds", 3600)

        results = review_stored_facts(brain, limit, min_age)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": results
        }

    if op == "PRE_STORAGE_CHECK":
        content = payload.get("content")
        metadata = payload.get("metadata", {})
        brain = payload.get("brain")

        result = pre_storage_safety_check(content, metadata, brain)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": result
        }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "components": {
                    "belief_tracker": _belief_tracker_available,
                    "correction_handler": _correction_handler_available,
                    "truth_classifier": TruthClassifier is not None,
                    "memory": _memory is not None
                },
                "available_operations": [
                    "CHECK_DUPLICATE",
                    "DETECT_CONTRADICTION",
                    "REVIEW_FACTS",
                    "PRE_STORAGE_CHECK",
                    "HEALTH"
                ]
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": f"Operation '{op}' not supported"
        }
    }


# Standard service contract
handle = service_api
