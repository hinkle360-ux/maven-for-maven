"""
self_awareness_api.py
=====================

Unified API for all Maven self-awareness operations.

This module provides a comprehensive interface for:
1. Identity & Code Awareness (who am I, what's my code)
2. Health & Monitoring (system health, brain status)
3. Compliance & Upgrades (compliance status, upgrade recommendations)
4. Coordination & Integration (active brains, dependencies)
5. Memory & Learning (memory stats, learned patterns)
6. Meta-Cognitive Operations (self-evaluation, reasoning explanation)

Usage:
    from api.self_awareness_api import SelfAwarenessAPI

    api = SelfAwarenessAPI()

    # Get identity
    identity = api.get_identity()

    # Check system health
    health = api.get_system_health()

    # Get compliance status
    compliance = api.get_compliance_status()

    # Plan self-upgrade
    upgrade_plan = api.plan_self_upgrade()
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import time


class SelfAwarenessAPI:
    """Unified API for all self-awareness operations."""

    def __init__(self):
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all self-awareness components."""
        # Import components lazily to avoid circular dependencies
        try:
            from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api
            self.self_model = self_model_api
        except Exception as e:
            print(f"[SELF_AWARENESS_API] Failed to load self_model: {e}")
            self.self_model = None

        try:
            from brains.cognitive.self_model.service.self_introspection import BrainIntrospector
            self.introspector = BrainIntrospector()
        except Exception as e:
            print(f"[SELF_AWARENESS_API] Failed to load introspector: {e}")
            self.introspector = None

        try:
            from brains.monitoring.brain_health_monitor import get_health_monitor
            self.health_monitor = get_health_monitor()
        except Exception as e:
            print(f"[SELF_AWARENESS_API] Failed to load health_monitor: {e}")
            self.health_monitor = None

        try:
            from brains.coordination.brain_coordinator import get_coordinator
            self.coordinator = get_coordinator()
        except Exception as e:
            print(f"[SELF_AWARENESS_API] Failed to load coordinator: {e}")
            self.coordinator = None

        try:
            from brains.cognitive.teacher.service.teacher_brain import get_learned_patterns
            self.teacher_patterns = get_learned_patterns
        except Exception as e:
            print(f"[SELF_AWARENESS_API] Failed to load teacher patterns: {e}")
            self.teacher_patterns = None

    # ==========================================================================
    # IDENTITY & CODE AWARENESS
    # ==========================================================================

    def get_identity(self) -> Dict[str, Any]:
        """
        Get Maven's core identity.

        Returns:
            Dict with identity information
        """
        if not self.self_model:
            return {"error": "Self-model not available"}

        try:
            response = self.self_model({
                "op": "QUERY_SELF",
                "payload": {"self_kind": "identity"}
            })
            return response.get("payload", {})
        except Exception as e:
            return {"error": str(e)}

    def scan_code_structure(self) -> Dict[str, Any]:
        """
        Scan Maven's code structure using dynamic introspection.

        Returns:
            Dict with code structure information
        """
        if not self.self_model:
            return {"error": "Self-model not available"}

        try:
            response = self.self_model({
                "op": "QUERY_SELF",
                "payload": {"self_kind": "code"}
            })
            return response.get("payload", {})
        except Exception as e:
            return {"error": str(e)}

    def introspect_brain(self, brain_name: str) -> Dict[str, Any]:
        """
        Deep introspection of a specific brain using AST analysis.

        Args:
            brain_name: Name of brain to introspect

        Returns:
            Dict with brain analysis results
        """
        if not self.introspector:
            return {"error": "Introspector not available"}

        try:
            return self.introspector.analyze_brain(brain_name)
        except Exception as e:
            return {"error": str(e)}

    def describe_capabilities(self) -> Dict[str, Any]:
        """
        Describe Maven's current capabilities.

        Returns:
            Dict with capabilities information
        """
        return {
            "cognitive_brains": 25,
            "domain_banks": ["science", "history", "geography", "math", "personal"],
            "capabilities": [
                "Natural language understanding and generation",
                "Multi-turn conversation with context awareness",
                "Fact learning and memory storage",
                "Question answering from world knowledge",
                "Self-introspection and code analysis",
                "Pattern recognition and learning",
                "Task planning and execution",
                "Code generation and analysis",
                "Self-review and quality control"
            ],
            "self_awareness_features": [
                "Dynamic code scanning",
                "Brain compliance monitoring",
                "Health tracking and alerting",
                "Cross-brain coordination",
                "Automated self-upgrade planning"
            ]
        }

    # ==========================================================================
    # HEALTH & MONITORING
    # ==========================================================================

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get complete system health status.

        Returns:
            Dict with system health metrics
        """
        if not self.health_monitor:
            return {"error": "Health monitor not available"}

        try:
            return self.health_monitor.get_system_health_summary()
        except Exception as e:
            return {"error": str(e)}

    def get_brain_health(self, brain_name: str) -> Dict[str, Any]:
        """
        Get health metrics for a specific brain.

        Args:
            brain_name: Name of brain to check

        Returns:
            Dict with brain health metrics
        """
        if not self.health_monitor:
            return {"error": "Health monitor not available"}

        try:
            return self.health_monitor.monitor_brain_health(brain_name)
        except Exception as e:
            return {"error": str(e)}

    def get_unhealthy_brains(self) -> List[str]:
        """
        Get list of brains with health issues.

        Returns:
            List of brain names with warning or critical status
        """
        if not self.health_monitor:
            return []

        try:
            return self.health_monitor.get_unhealthy_brains()
        except Exception as e:
            return []

    # ==========================================================================
    # COMPLIANCE & UPGRADES
    # ==========================================================================

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get brain compliance status (3-signal cognitive contract).

        Returns:
            Dict with compliance information
        """
        if not self.introspector:
            return {"error": "Introspector not available"}

        try:
            results = self.introspector.scan_all_brains()
            return {
                "total_brains": len(results),
                "compliant_count": sum(1 for r in results.values() if r.get("compliant")),
                "compliance_percentage": round(
                    sum(1 for r in results.values() if r.get("compliant")) / len(results) * 100
                    if results else 0,
                    1
                ),
                "detailed_results": results
            }
        except Exception as e:
            return {"error": str(e)}

    def get_upgrade_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommended upgrades for non-compliant brains.

        Returns:
            List of upgrade recommendations
        """
        if not self.introspector:
            return []

        try:
            compliance = self.get_compliance_status()
            non_compliant = [
                brain for brain, result in compliance.get("detailed_results", {}).items()
                if not result.get("compliant")
            ]

            recommendations = []
            for brain in non_compliant:
                recommendations.append({
                    "brain": brain,
                    "issue": "Missing cognitive contract signals",
                    "priority": "high" if brain in ["teacher", "self_review", "self_model"] else "medium",
                    "recommendation": "Add continuation detection, context retrieval, and routing hints"
                })

            return recommendations
        except Exception as e:
            return []

    def plan_self_upgrade(self) -> Dict[str, Any]:
        """
        Generate comprehensive self-upgrade plan using dynamic introspection.

        Returns:
            Dict with upgrade plan
        """
        if not self.self_model:
            return {"error": "Self-model not available"}

        try:
            response = self.self_model({
                "op": "QUERY_SELF",
                "payload": {"self_kind": "upgrade"}
            })
            return response.get("payload", {})
        except Exception as e:
            return {"error": str(e)}

    # ==========================================================================
    # COORDINATION & INTEGRATION
    # ==========================================================================

    def get_active_brains(self) -> List[str]:
        """
        Get list of currently active brains.

        Returns:
            List of active brain names
        """
        if not self.health_monitor:
            return []

        try:
            system_health = self.health_monitor.get_system_health_summary()
            # Return brains with recent activity
            # In real implementation, would track active brains
            return ["sensorium", "memory_librarian", "integrator"]  # Placeholder
        except Exception as e:
            return []

    def get_brain_dependencies(self, brain_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get brain dependency graph.

        Args:
            brain_name: Optional specific brain to get dependencies for

        Returns:
            Dict mapping brains to their dependencies
        """
        if not self.coordinator:
            return {}

        try:
            if brain_name:
                return {brain_name: self.coordinator.get_brain_dependencies(brain_name)}
            else:
                return self.coordinator.dependency_graph
        except Exception as e:
            return {}

    def get_execution_order(self, brain_names: List[str]) -> List[str]:
        """
        Get optimal execution order for a set of brains.

        Args:
            brain_names: List of brains to order

        Returns:
            Ordered list (dependencies first)
        """
        if not self.coordinator:
            return brain_names

        try:
            return self.coordinator.get_execution_order(brain_names)
        except Exception as e:
            return brain_names

    # ==========================================================================
    # MEMORY & LEARNING
    # ==========================================================================

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dict with memory stats
        """
        if not self.self_model:
            return {"error": "Self-model not available"}

        try:
            response = self.self_model({
                "op": "QUERY_MEMORY",
                "payload": {"self_mode": "stats"}
            })
            return response.get("payload", {})
        except Exception as e:
            return {"error": str(e)}

    def get_learning_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get learned patterns from Teacher brain.

        Args:
            pattern_type: Optional filter by pattern type

        Returns:
            List of learned patterns
        """
        if not self.teacher_patterns:
            return []

        try:
            return self.teacher_patterns(pattern_type)
        except Exception as e:
            return []

    # ==========================================================================
    # META-COGNITIVE OPERATIONS
    # ==========================================================================

    def explain_last_response(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Explain why Maven responded the way it did.

        Args:
            context: Optional context about the response to explain

        Returns:
            Dict with explanation
        """
        # Placeholder - would integrate with reasoning trace
        return {
            "explanation": "Response generated based on conversation context and relevant memories",
            "brains_involved": ["sensorium", "memory_librarian", "reasoning", "integrator"],
            "confidence": 0.8
        }

    def evaluate_self_performance(self) -> Dict[str, Any]:
        """
        Self-evaluation of overall performance.

        Returns:
            Dict with performance metrics
        """
        health = self.get_system_health()
        compliance = self.get_compliance_status()

        return {
            "overall_score": 0.75,  # Placeholder calculation
            "health_score": 0.85 if health.get("status") == "healthy" else 0.5,
            "compliance_score": compliance.get("compliance_percentage", 0) / 100,
            "areas_for_improvement": [
                "Increase brain compliance to 100%",
                "Optimize response times",
                "Expand learned pattern coverage"
            ],
            "strengths": [
                "Dynamic self-awareness and code introspection",
                "Multi-turn conversation handling",
                "Cross-brain coordination"
            ]
        }

    # ==========================================================================
    # NATURAL LANGUAGE QUERY INTERFACE
    # ==========================================================================

    def query_self(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Answer natural language questions about Maven's self.

        Args:
            natural_language_query: Question about Maven

        Returns:
            Dict with answer
        """
        query_lower = natural_language_query.lower()

        # Route to appropriate method based on query
        if any(word in query_lower for word in ["who are you", "identity", "what are you"]):
            return self.get_identity()

        elif any(word in query_lower for word in ["code", "implementation", "architecture"]):
            return self.scan_code_structure()

        elif any(word in query_lower for word in ["health", "status", "running"]):
            return self.get_system_health()

        elif any(word in query_lower for word in ["compliance", "signals", "contract"]):
            return self.get_compliance_status()

        elif any(word in query_lower for word in ["capabilities", "can you", "what can"]):
            return self.describe_capabilities()

        elif any(word in query_lower for word in ["memory", "learned", "remember"]):
            return self.get_memory_stats()

        elif any(word in query_lower for word in ["upgrade", "improve", "fix"]):
            return self.plan_self_upgrade()

        else:
            return {
                "error": "Query not recognized",
                "suggestion": "Try asking about: identity, code, health, compliance, capabilities, memory, or upgrades"
            }


# Convenience functions
_api_instance = None


def get_api() -> SelfAwarenessAPI:
    """Get singleton API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = SelfAwarenessAPI()
    return _api_instance


# Quick access functions
def get_identity() -> Dict[str, Any]:
    """Quick access to identity."""
    return get_api().get_identity()


def get_system_health() -> Dict[str, Any]:
    """Quick access to system health."""
    return get_api().get_system_health()


def get_compliance_status() -> Dict[str, Any]:
    """Quick access to compliance status."""
    return get_api().get_compliance_status()


def query_self(question: str) -> Dict[str, Any]:
    """Quick access to natural language queries."""
    return get_api().query_self(question)
