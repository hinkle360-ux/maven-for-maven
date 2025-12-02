"""
brain_coordinator.py
====================

Cross-brain coordination and multi-brain response integration for Maven.

This module provides:
1. Multi-brain response coordination and merging
2. Conflict detection and resolution
3. Parallel brain processing
4. Brain dependency management
5. Context sharing between brains

Key Features:
    - Detects when multiple brains should work together
    - Resolves conflicts between competing brain responses
    - Merges complementary responses into cohesive output
    - Manages brain execution order based on dependencies
    - Shares context efficiently across brains
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
import time


# Brain dependency graph (defines execution order)
BRAIN_DEPENDENCIES = {
    # Core flow
    "sensorium": [],  # No dependencies (entry point)
    "system_history": ["sensorium"],
    "memory_librarian": ["sensorium", "system_history"],
    "integrator": ["sensorium", "memory_librarian"],

    # Cognitive processing
    "language": ["sensorium"],
    "reasoning": ["language", "sensorium"],
    "pattern_recognition": ["sensorium"],
    "attention": ["sensorium"],

    # Meta-cognitive
    "self_model": [],
    "self_dmn": [],
    "self_review": ["reasoning", "language"],

    # Learning
    "teacher": ["reasoning", "memory_librarian"],
    "belief_tracker": ["reasoning"],

    # Planning & execution
    "planner": ["reasoning", "memory_librarian"],
    "replanner": ["planner"],
    "coder": ["reasoning", "planner"],
    "autonomy": ["planner"],

    # Specialized
    "research_manager": ["reasoning"],
    "committee": ["reasoning"],
    "abstraction": ["reasoning", "pattern_recognition"],
    "imaginer": ["reasoning"],

    # Other
    "affect_priority": ["sensorium"],
    "personality": [],
    "motivation": ["reasoning"],
}


class BrainCoordinator:
    """Coordinates multi-brain responses and parallel processing."""

    def __init__(self):
        self.dependency_graph = BRAIN_DEPENDENCIES
        self.active_coordinations = {}

    def coordinate_multi_brain_response(
        self,
        brain_responses: List[Dict[str, Any]],
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate responses from multiple brains into a coherent result.

        Args:
            brain_responses: List of responses from different brains
            query: Original user query
            context: Conversation context

        Returns:
            Coordinated response dict
        """
        if not brain_responses:
            return {
                "success": False,
                "error": "No brain responses to coordinate",
                "coordinated": False
            }

        if len(brain_responses) == 1:
            # Single brain response - no coordination needed
            result = brain_responses[0].copy()
            result["coordinated"] = False
            result["brain_count"] = 1
            return result

        # Detect conflicts and overlaps
        conflicts = self._detect_conflicts(brain_responses)

        if conflicts:
            # Resolve conflicts via weighted voting or integration
            resolved = self._resolve_conflicts(conflicts, brain_responses)
            resolved["conflicts_detected"] = len(conflicts)
            resolved["resolution_method"] = "weighted_voting"
            return resolved

        # Merge complementary responses
        merged = self._merge_responses(brain_responses, query, context)
        merged["coordinated"] = True
        merged["brain_count"] = len(brain_responses)
        merged["conflicts_detected"] = 0

        return merged

    def _detect_conflicts(
        self,
        brain_responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between brain responses.

        Conflicts occur when:
        - Brains provide contradictory information
        - Brains suggest different actions
        - Confidence levels disagree significantly
        """
        conflicts = []

        # Check for contradictory answers
        answers = [(r.get("brain_name"), r.get("answer") or r.get("result"))
                  for r in brain_responses if r.get("answer") or r.get("result")]

        if len(answers) > 1:
            # Simple conflict detection: different non-empty answers
            unique_answers = set(str(a[1])[:100] for a in answers if a[1])
            if len(unique_answers) > 1:
                conflicts.append({
                    "type": "contradictory_answers",
                    "brains": [a[0] for a in answers],
                    "details": "Brains provided different answers"
                })

        # Check for confidence disagreements
        confidences = [(r.get("brain_name"), r.get("confidence", 0))
                      for r in brain_responses]

        if len(confidences) > 1:
            conf_values = [c[1] for c in confidences]
            if max(conf_values) - min(conf_values) > 0.4:
                conflicts.append({
                    "type": "confidence_disagreement",
                    "brains": [c[0] for c in confidences],
                    "details": f"Confidence range: {min(conf_values):.2f}-{max(conf_values):.2f}"
                })

        return conflicts

    def _resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        brain_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between brain responses using weighted voting.

        Priority order:
        1. Self-model/self-DMN (for self-questions)
        2. Reasoning brain
        3. Teacher brain (for world questions)
        4. Highest confidence
        5. Most recent/active
        """
        # Brain priority weights
        priority_weights = {
            "self_model": 100,
            "self_dmn": 100,
            "reasoning": 80,
            "teacher": 70,
            "memory_librarian": 60,
            "language": 50,
            "integrator": 90,
        }

        # Calculate weighted scores for each response
        scored_responses = []
        for response in brain_responses:
            brain_name = response.get("brain_name", "unknown")
            confidence = response.get("confidence", 0.5)
            priority = priority_weights.get(brain_name, 40)

            # Weighted score = priority * confidence
            score = priority * confidence

            scored_responses.append({
                "response": response,
                "score": score,
                "brain_name": brain_name
            })

        # Sort by score (highest first)
        scored_responses.sort(key=lambda x: x["score"], reverse=True)

        # Winner is highest scored response
        winner = scored_responses[0]["response"].copy()
        winner["coordination_method"] = "conflict_resolution"
        winner["winner_brain"] = scored_responses[0]["brain_name"]
        winner["winner_score"] = scored_responses[0]["score"]
        winner["competing_brains"] = [s["brain_name"] for s in scored_responses[1:]]

        return winner

    def _merge_responses(
        self,
        brain_responses: List[Dict[str, Any]],
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge complementary responses from multiple brains.

        Combines:
        - Answers/results
        - Confidence (averaged)
        - Evidence (concatenated)
        - Routing hints (merged)
        """
        merged = {
            "brains_involved": [],
            "answers": [],
            "combined_answer": "",
            "confidence": 0.0,
            "evidence": {},
            "routing_hints": [],
            "coordination_method": "merge_complementary",
            "coordinated": True
        }

        total_confidence = 0.0
        answer_parts = []

        for response in brain_responses:
            brain_name = response.get("brain_name", "unknown")
            merged["brains_involved"].append(brain_name)

            # Collect answers
            answer = response.get("answer") or response.get("result") or ""
            if answer:
                merged["answers"].append({
                    "brain": brain_name,
                    "answer": answer
                })
                answer_parts.append(f"[{brain_name}] {answer}")

            # Aggregate confidence
            confidence = response.get("confidence", 0.5)
            total_confidence += confidence

            # Collect evidence
            evidence = response.get("evidence", {})
            if evidence:
                merged["evidence"][brain_name] = evidence

            # Collect routing hints
            routing_hint = response.get("routing_hint", {})
            if routing_hint:
                merged["routing_hints"].append({
                    "brain": brain_name,
                    "hint": routing_hint
                })

        # Calculate average confidence
        if brain_responses:
            merged["confidence"] = total_confidence / len(brain_responses)

        # Combine answers into single text
        if answer_parts:
            merged["combined_answer"] = "\n".join(answer_parts)

        return merged

    def get_execution_order(self, required_brains: List[str]) -> List[str]:
        """
        Determine optimal execution order based on dependencies.

        Args:
            required_brains: List of brains that need to execute

        Returns:
            Ordered list of brains (dependencies first)
        """
        # Topological sort
        visited = set()
        order = []

        def visit(brain: str):
            if brain in visited:
                return
            visited.add(brain)

            # Visit dependencies first
            deps = self.dependency_graph.get(brain, [])
            for dep in deps:
                if dep in required_brains:
                    visit(dep)

            order.append(brain)

        for brain in required_brains:
            visit(brain)

        return order

    def get_brain_dependencies(self, brain_name: str) -> List[str]:
        """Get direct dependencies for a brain."""
        return self.dependency_graph.get(brain_name, [])

    def get_dependent_brains(self, brain_name: str) -> List[str]:
        """Get brains that depend on this brain."""
        dependents = []
        for brain, deps in self.dependency_graph.items():
            if brain_name in deps:
                dependents.append(brain)
        return dependents

    def can_run_parallel(self, brain_names: List[str]) -> bool:
        """
        Check if brains can run in parallel (no dependencies between them).

        Args:
            brain_names: List of brains to check

        Returns:
            True if they can run in parallel
        """
        for brain in brain_names:
            deps = self.dependency_graph.get(brain, [])
            # If any brain depends on another in the list, can't parallelize
            if any(d in brain_names for d in deps):
                return False
        return True

    def create_shared_context(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        norm_type: str = "question",
        last_topic: str = "",
        relevant_memories: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized context shared across all brains.

        Args:
            query: User query
            conversation_history: Previous conversation turns
            norm_type: Query type from Sensorium
            last_topic: Last discussed topic
            relevant_memories: Retrieved memories

        Returns:
            Standardized context dict
        """
        return {
            "query": query,
            "norm_type": norm_type,
            "last_topic": last_topic,
            "conversation_history": conversation_history,
            "conversation_depth": len(conversation_history),
            "relevant_memories": relevant_memories or [],
            "timestamp": time.time(),
            "coordination_enabled": True
        }

    def share_context(
        self,
        context: Dict[str, Any],
        target_brains: List[str]
    ) -> Dict[str, bool]:
        """
        Share context with multiple brains efficiently.

        Args:
            context: Context to share
            target_brains: List of brain names

        Returns:
            Dict mapping brain names to success status
        """
        results = {}

        for brain in target_brains:
            try:
                # In real implementation, would call each brain's context receiver
                # For now, just record that context would be shared
                results[brain] = True
            except Exception as e:
                results[brain] = False
                print(f"[COORDINATOR] Failed to share context with {brain}: {e}")

        return results


# Singleton instance
_coordinator = BrainCoordinator()


def get_coordinator() -> BrainCoordinator:
    """Get the global brain coordinator instance."""
    return _coordinator


def coordinate_responses(
    brain_responses: List[Dict[str, Any]],
    query: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Convenience function to coordinate brain responses."""
    return _coordinator.coordinate_multi_brain_response(
        brain_responses, query, context
    )


def get_execution_order(brain_names: List[str]) -> List[str]:
    """Convenience function to get execution order."""
    return _coordinator.get_execution_order(brain_names)
