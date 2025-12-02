"""
context_bundle.py
~~~~~~~~~~~~~~~~~

Unified context bundle for memory/context retrieval.

This module defines a single shape for "what memory/context is available for this turn",
allowing consistent access to memories, episodes, vector hits, goals, and projects
across the pipeline.

Usage:
    from brains.cognitive.memory_librarian.service.context_bundle import (
        ContextBundle,
        build_context_bundle,
        merge_bundles,
    )

    # Build a context bundle for the current request
    bundle = build_context_bundle(
        user_id="user123",
        session_id="sess456",
        query="What were we working on yesterday?"
    )

    # Access memories
    for hit in bundle.retrieved_memories.stm_hits:
        print(hit)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Sequence


@dataclass
class RetrievedMemories:
    """Container for retrieved memories from different sources."""
    stm_hits: List[Dict[str, Any]] = field(default_factory=list)
    mtm_hits: List[Dict[str, Any]] = field(default_factory=list)
    ltm_hits: List[Dict[str, Any]] = field(default_factory=list)
    episodic_hits: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph_facts: List[Dict[str, Any]] = field(default_factory=list)
    vector_hits: List[Dict[str, Any]] = field(default_factory=list)
    fast_cache_hit: Optional[Dict[str, Any]] = None
    semantic_cache_hit: Optional[Dict[str, Any]] = None

    def all_hits(self) -> List[Dict[str, Any]]:
        """Return all memory hits as a flat list, deduplicated by id."""
        seen_ids = set()
        all_items = []
        for hits in [self.stm_hits, self.mtm_hits, self.ltm_hits,
                     self.episodic_hits, self.knowledge_graph_facts, self.vector_hits]:
            for item in hits:
                item_id = item.get("id", id(item))
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    all_items.append(item)
        return all_items

    def count(self) -> int:
        """Return total count of all memory hits."""
        return len(self.all_hits())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stm_hits": self.stm_hits,
            "mtm_hits": self.mtm_hits,
            "ltm_hits": self.ltm_hits,
            "episodic_hits": self.episodic_hits,
            "knowledge_graph_facts": self.knowledge_graph_facts,
            "vector_hits": self.vector_hits,
            "fast_cache_hit": self.fast_cache_hit,
            "semantic_cache_hit": self.semantic_cache_hit,
            "total_count": self.count(),
        }


@dataclass
class GoalsAndProjects:
    """Container for active goals and projects."""
    active_goals: List[Dict[str, Any]] = field(default_factory=list)
    active_projects: List[Dict[str, Any]] = field(default_factory=list)
    current_project_id: Optional[str] = None
    current_goal_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "active_goals": self.active_goals,
            "active_projects": self.active_projects,
            "current_project_id": self.current_project_id,
            "current_goal_ids": self.current_goal_ids,
        }


@dataclass
class BundleMetadata:
    """Metadata about the context bundle."""
    timestamp: float = field(default_factory=time.time)
    source: str = "chat_cli"  # chat_cli, browser, tests, api
    retrieval_time_ms: int = 0
    query_used: str = ""
    tiers_queried: List[str] = field(default_factory=list)
    vector_search_enabled: bool = False
    episodic_search_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "retrieval_time_ms": self.retrieval_time_ms,
            "query_used": self.query_used,
            "tiers_queried": self.tiers_queried,
            "vector_search_enabled": self.vector_search_enabled,
            "episodic_search_enabled": self.episodic_search_enabled,
        }


@dataclass
class ContextBundle:
    """
    Unified context bundle for a single turn/request.

    This is the single shape for "what memory/context is available for this turn",
    passed through the pipeline and used by brains to access relevant memories.
    """
    # Identity
    user_id: str = ""
    session_id: str = ""

    # Current episode (optional - populated when episode tracking is active)
    current_episode_id: Optional[str] = None

    # Recent turns from current session
    recent_turns: List[Dict[str, Any]] = field(default_factory=list)

    # Retrieved memories from all sources
    retrieved_memories: RetrievedMemories = field(default_factory=RetrievedMemories)

    # Goals and projects
    goals_and_projects: GoalsAndProjects = field(default_factory=GoalsAndProjects)

    # Metadata about the retrieval
    metadata: BundleMetadata = field(default_factory=BundleMetadata)

    # Bundle ID for tracking
    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def has_memories(self) -> bool:
        """Check if any memories were retrieved."""
        return self.retrieved_memories.count() > 0

    def has_cache_hit(self) -> bool:
        """Check if there was a fast or semantic cache hit."""
        return (self.retrieved_memories.fast_cache_hit is not None or
                self.retrieved_memories.semantic_cache_hit is not None)

    def has_episodic_hits(self) -> bool:
        """Check if there are episodic memory hits."""
        return len(self.retrieved_memories.episodic_hits) > 0

    def has_vector_hits(self) -> bool:
        """Check if there are vector search hits."""
        return len(self.retrieved_memories.vector_hits) > 0

    def get_best_cache_hit(self) -> Optional[Dict[str, Any]]:
        """Get the best cache hit (fast > semantic)."""
        if self.retrieved_memories.fast_cache_hit:
            return self.retrieved_memories.fast_cache_hit
        return self.retrieved_memories.semantic_cache_hit

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire bundle to a dictionary."""
        return {
            "bundle_id": self.bundle_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "current_episode_id": self.current_episode_id,
            "recent_turns": self.recent_turns,
            "retrieved_memories": self.retrieved_memories.to_dict(),
            "goals_and_projects": self.goals_and_projects.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextBundle":
        """Create a ContextBundle from a dictionary."""
        bundle = cls(
            bundle_id=data.get("bundle_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            current_episode_id=data.get("current_episode_id"),
            recent_turns=data.get("recent_turns", []),
        )

        # Populate retrieved memories
        mem_data = data.get("retrieved_memories", {})
        bundle.retrieved_memories = RetrievedMemories(
            stm_hits=mem_data.get("stm_hits", []),
            mtm_hits=mem_data.get("mtm_hits", []),
            ltm_hits=mem_data.get("ltm_hits", []),
            episodic_hits=mem_data.get("episodic_hits", []),
            knowledge_graph_facts=mem_data.get("knowledge_graph_facts", []),
            vector_hits=mem_data.get("vector_hits", []),
            fast_cache_hit=mem_data.get("fast_cache_hit"),
            semantic_cache_hit=mem_data.get("semantic_cache_hit"),
        )

        # Populate goals and projects
        gp_data = data.get("goals_and_projects", {})
        bundle.goals_and_projects = GoalsAndProjects(
            active_goals=gp_data.get("active_goals", []),
            active_projects=gp_data.get("active_projects", []),
            current_project_id=gp_data.get("current_project_id"),
            current_goal_ids=gp_data.get("current_goal_ids", []),
        )

        # Populate metadata
        meta_data = data.get("metadata", {})
        bundle.metadata = BundleMetadata(
            timestamp=meta_data.get("timestamp", time.time()),
            source=meta_data.get("source", "unknown"),
            retrieval_time_ms=meta_data.get("retrieval_time_ms", 0),
            query_used=meta_data.get("query_used", ""),
            tiers_queried=meta_data.get("tiers_queried", []),
            vector_search_enabled=meta_data.get("vector_search_enabled", False),
            episodic_search_enabled=meta_data.get("episodic_search_enabled", True),
        )

        return bundle


def merge_bundles(bundles: Sequence[ContextBundle]) -> ContextBundle:
    """
    Merge multiple context bundles into one.

    Useful when combining results from multiple retrieval sources.
    """
    if not bundles:
        return ContextBundle()

    if len(bundles) == 1:
        return bundles[0]

    # Start with the first bundle as base
    result = ContextBundle(
        user_id=bundles[0].user_id,
        session_id=bundles[0].session_id,
        current_episode_id=bundles[0].current_episode_id,
    )

    # Merge all memories
    seen_ids = set()
    for bundle in bundles:
        for hit in bundle.retrieved_memories.stm_hits:
            hit_id = hit.get("id", id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                result.retrieved_memories.stm_hits.append(hit)

        for hit in bundle.retrieved_memories.mtm_hits:
            hit_id = hit.get("id", id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                result.retrieved_memories.mtm_hits.append(hit)

        for hit in bundle.retrieved_memories.ltm_hits:
            hit_id = hit.get("id", id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                result.retrieved_memories.ltm_hits.append(hit)

        for hit in bundle.retrieved_memories.episodic_hits:
            hit_id = hit.get("id", id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                result.retrieved_memories.episodic_hits.append(hit)

        for hit in bundle.retrieved_memories.knowledge_graph_facts:
            hit_id = hit.get("id", id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                result.retrieved_memories.knowledge_graph_facts.append(hit)

        for hit in bundle.retrieved_memories.vector_hits:
            hit_id = hit.get("id", id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                result.retrieved_memories.vector_hits.append(hit)

        # Take first non-None cache hit
        if result.retrieved_memories.fast_cache_hit is None:
            result.retrieved_memories.fast_cache_hit = bundle.retrieved_memories.fast_cache_hit
        if result.retrieved_memories.semantic_cache_hit is None:
            result.retrieved_memories.semantic_cache_hit = bundle.retrieved_memories.semantic_cache_hit

        # Merge recent turns (dedupe by timestamp)
        for turn in bundle.recent_turns:
            if turn not in result.recent_turns:
                result.recent_turns.append(turn)

        # Merge goals (dedupe by goal_id)
        for goal in bundle.goals_and_projects.active_goals:
            goal_id = goal.get("goal_id", id(goal))
            if not any(g.get("goal_id") == goal_id for g in result.goals_and_projects.active_goals):
                result.goals_and_projects.active_goals.append(goal)

        # Merge projects (dedupe by project_id)
        for project in bundle.goals_and_projects.active_projects:
            project_id = project.get("project_id", id(project))
            if not any(p.get("project_id") == project_id for p in result.goals_and_projects.active_projects):
                result.goals_and_projects.active_projects.append(project)

    return result


def create_empty_bundle(
    user_id: str = "",
    session_id: str = "",
    source: str = "unknown"
) -> ContextBundle:
    """Create an empty context bundle with basic identity."""
    return ContextBundle(
        user_id=user_id,
        session_id=session_id,
        metadata=BundleMetadata(source=source),
    )


# Public API
__all__ = [
    "ContextBundle",
    "RetrievedMemories",
    "GoalsAndProjects",
    "BundleMetadata",
    "merge_bundles",
    "create_empty_bundle",
]
