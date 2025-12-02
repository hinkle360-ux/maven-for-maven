"""
pattern_router.py
=================

Pattern-based routing engine for Maven.

This module learns from successful routing patterns and uses them to optimize
future brain selection and coordination.

Key Features:
1. Pattern storage and retrieval from Teacher brain
2. Query pattern matching and similarity scoring
3. Routing prediction based on historical success
4. Continuous learning from routing outcomes
5. Performance metrics tracking

Integration:
    - Learns from Teacher brain's continuation patterns
    - Works with Integrator for brain selection
    - Tracks success/failure of routing decisions
    - Self-improves over time based on outcomes
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import re


class PatternBasedRouter:
    """Route queries using learned patterns from successful interactions."""

    def __init__(self):
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.success_metrics = defaultdict(lambda: {"success": 0, "total": 0})
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.last_cache_refresh = 0

    def route_by_pattern(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route query using learned patterns from Teacher brain.

        Args:
            query: User query to route
            context: Optional context dict

        Returns:
            Routing suggestion with brain sequence and confidence
        """
        # Refresh pattern cache if needed
        if time.time() - self.last_cache_refresh > self.cache_ttl:
            self._refresh_pattern_cache()

        # Extract query pattern
        query_pattern = self._extract_query_pattern(query)

        # Find matching patterns
        matching_patterns = self._find_matching_patterns(query_pattern, query)

        if not matching_patterns:
            # No patterns found - use standard routing
            return {
                "pattern_matched": False,
                "suggested_brains": [],
                "confidence": 0.0,
                "fallback_to_standard": True
            }

        # Sort patterns by success rate and relevance
        best_pattern = self._select_best_pattern(matching_patterns, query)

        # Extract brain sequence from pattern
        brain_sequence = best_pattern.get("successful_pipeline", [])
        success_rate = best_pattern.get("success_score", 0.0)

        return {
            "pattern_matched": True,
            "suggested_brains": brain_sequence,
            "confidence": success_rate,
            "pattern_id": best_pattern.get("pattern_id"),
            "pattern_type": best_pattern.get("user_query_pattern"),
            "fallback_to_standard": False
        }

    def _refresh_pattern_cache(self) -> None:
        """Refresh pattern cache from Teacher brain."""
        try:
            # Import Teacher's pattern retrieval function
            from brains.cognitive.teacher.service.teacher_brain import get_learned_patterns

            # Get all continuation patterns
            patterns = get_learned_patterns()

            # Organize by pattern type
            self.pattern_cache.clear()
            for pattern in patterns:
                pattern_type = pattern.get("user_query_pattern", "general_follow_up")
                if pattern_type not in self.pattern_cache:
                    self.pattern_cache[pattern_type] = []

                # Add pattern ID for tracking
                pattern["pattern_id"] = f"{pattern_type}_{len(self.pattern_cache[pattern_type])}"
                self.pattern_cache[pattern_type].append(pattern)

            self.last_cache_refresh = time.time()

        except Exception as e:
            print(f"[PATTERN_ROUTER] Cache refresh failed: {e}")

    def _extract_query_pattern(self, query: str) -> str:
        """
        Extract pattern type from query.

        Returns pattern type like:
        - expansion_request ("tell me more", "expand on")
        - specific_expansion ("what about X")
        - detail_request ("more details")
        - continuation_request ("continue", "go on")
        - alternative_request ("what else")
        """
        query_lower = query.lower()

        # Pattern matching (same as Teacher brain)
        patterns = {
            "tell me more": "expansion_request",
            "expand on": "expansion_request",
            "more about": "expansion_request",
            "what about": "specific_expansion",
            "how about": "specific_expansion",
            "continue": "continuation_request",
            "go on": "continuation_request",
            "keep going": "continuation_request",
            "more details": "detail_request",
            "elaborate": "detail_request",
            "what else": "alternative_request",
            "any other": "alternative_request",
            "and then": "sequence_continuation",
            "what next": "sequence_continuation"
        }

        for phrase, pattern_type in patterns.items():
            if phrase in query_lower:
                return pattern_type

        return "general_follow_up"

    def _find_matching_patterns(
        self,
        pattern_type: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Find patterns matching the query pattern type."""
        # Direct pattern type match
        direct_matches = self.pattern_cache.get(pattern_type, [])

        if direct_matches:
            return direct_matches

        # Fallback to general follow_up patterns
        fallback_matches = self.pattern_cache.get("general_follow_up", [])

        return fallback_matches

    def _select_best_pattern(
        self,
        patterns: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """Select the best pattern based on success rate and relevance."""
        if not patterns:
            return {}

        # Score each pattern
        scored_patterns = []
        for pattern in patterns:
            success_score = pattern.get("success_score", 0.5)

            # Check if query is similar to pattern's sample query
            similarity = self._calculate_query_similarity(
                query,
                pattern.get("user_query_sample", "")
            )

            # Combined score: 70% success rate + 30% similarity
            combined_score = (0.7 * success_score) + (0.3 * similarity)

            scored_patterns.append({
                "pattern": pattern,
                "score": combined_score
            })

        # Sort by score (highest first)
        scored_patterns.sort(key=lambda x: x["score"], reverse=True)

        return scored_patterns[0]["pattern"]

    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate simple similarity score between queries."""
        # Normalize queries
        q1_words = set(re.findall(r'\w+', query1.lower()))
        q2_words = set(re.findall(r'\w+', query2.lower()))

        if not q1_words or not q2_words:
            return 0.0

        # Jaccard similarity
        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)

        return intersection / union if union > 0 else 0.0

    def update_pattern_success(
        self,
        pattern_id: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update pattern success metrics after routing.

        Args:
            pattern_id: ID of pattern that was used
            success: Whether the routing was successful
            details: Optional details about the outcome
        """
        metrics = self.success_metrics[pattern_id]
        metrics["total"] += 1
        if success:
            metrics["success"] += 1

        # Calculate success rate
        success_rate = metrics["success"] / metrics["total"] if metrics["total"] > 0 else 0.0

        # Store in routing history
        self.routing_history.append({
            "pattern_id": pattern_id,
            "success": success,
            "success_rate": success_rate,
            "timestamp": time.time(),
            "details": details or {}
        })

        # Keep only last 1000 entries
        if len(self.routing_history) > 1000:
            self.routing_history.pop(0)

    def get_pattern_performance(
        self,
        pattern_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for patterns.

        Args:
            pattern_id: Optional specific pattern to query

        Returns:
            Performance metrics dict
        """
        if pattern_id:
            metrics = self.success_metrics.get(pattern_id, {"success": 0, "total": 0})
            success_rate = (metrics["success"] / metrics["total"]
                          if metrics["total"] > 0 else 0.0)

            return {
                "pattern_id": pattern_id,
                "success_count": metrics["success"],
                "total_attempts": metrics["total"],
                "success_rate": round(success_rate, 3)
            }
        else:
            # Return all pattern metrics
            all_metrics = {}
            for pid, metrics in self.success_metrics.items():
                success_rate = (metrics["success"] / metrics["total"]
                              if metrics["total"] > 0 else 0.0)
                all_metrics[pid] = {
                    "success_count": metrics["success"],
                    "total_attempts": metrics["total"],
                    "success_rate": round(success_rate, 3)
                }

            return all_metrics

    def get_routing_recommendations(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top routing recommendations for a query.

        Args:
            query: User query
            context: Optional context

        Returns:
            List of routing recommendations sorted by confidence
        """
        # Get primary recommendation
        primary = self.route_by_pattern(query, context)

        recommendations = []

        if primary["pattern_matched"]:
            recommendations.append({
                "rank": 1,
                "brains": primary["suggested_brains"],
                "confidence": primary["confidence"],
                "source": "learned_pattern",
                "pattern_type": primary.get("pattern_type")
            })

        # Could add alternative recommendations here
        # For now, just return primary

        return recommendations


# Singleton instance
_pattern_router = PatternBasedRouter()


def get_pattern_router() -> PatternBasedRouter:
    """Get the global pattern router instance."""
    return _pattern_router


def route_by_pattern(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to route by pattern."""
    return _pattern_router.route_by_pattern(query, context)


def update_pattern_success(pattern_id: str, success: bool) -> None:
    """Convenience function to update pattern success."""
    _pattern_router.update_pattern_success(pattern_id, success)
