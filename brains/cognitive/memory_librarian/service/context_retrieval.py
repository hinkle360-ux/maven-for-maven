"""
context_retrieval.py
~~~~~~~~~~~~~~~~~~~~

Unified context retrieval system for the memory librarian.

This module provides the main interface for building ContextBundles by:
- Retrieving from tiered memory (STM/MTM/LTM)
- Querying episodic memory for relevant episodes
- Searching the vector index for semantic matches
- Loading active goals and projects
- Checking caches (fast and semantic)

Usage:
    from brains.cognitive.memory_librarian.service.context_retrieval import (
        build_context_bundle,
        retrieve_for_query,
    )

    # Build a full context bundle
    bundle = build_context_bundle(
        query="What were we working on yesterday?",
        user_id="user123",
        session_id="sess456",
    )

    # Quick retrieval for a query
    memories = retrieve_for_query(query="Python programming")
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional

from brains.cognitive.memory_librarian.service.context_bundle import (
    ContextBundle,
    RetrievedMemories,
    GoalsAndProjects,
    BundleMetadata,
    create_empty_bundle,
)


def build_context_bundle(
    query: str,
    user_id: str = "",
    session_id: str = "",
    source: str = "chat_cli",
    include_episodic: bool = True,
    include_vector: bool = True,
    include_goals: bool = True,
    stm_limit: int = 10,
    mtm_limit: int = 10,
    ltm_limit: int = 5,
    episodic_limit: int = 5,
    vector_limit: int = 5,
) -> ContextBundle:
    """
    Build a complete context bundle for a query.

    This is the main entry point for context retrieval, aggregating
    memories from all sources into a single ContextBundle.

    Args:
        query: The user's query
        user_id: User identifier
        session_id: Session identifier
        source: Source of the request (chat_cli, browser, tests)
        include_episodic: Whether to search episodic memory
        include_vector: Whether to search vector index
        include_goals: Whether to load active goals
        stm_limit: Max STM results
        mtm_limit: Max MTM results
        ltm_limit: Max LTM results
        episodic_limit: Max episodic results
        vector_limit: Max vector results

    Returns:
        Populated ContextBundle
    """
    start_time = time.time()

    bundle = create_empty_bundle(
        user_id=user_id,
        session_id=session_id,
        source=source,
    )
    bundle.metadata.query_used = query

    # 1. Check caches first (fast path)
    _check_caches(bundle, query)

    # 2. Retrieve from tiered memory
    _retrieve_from_tiers(bundle, query, stm_limit, mtm_limit, ltm_limit)

    # 3. Search episodic memory
    if include_episodic:
        _retrieve_from_episodic(bundle, query, user_id, session_id, episodic_limit)

    # 4. Search vector index
    if include_vector:
        _retrieve_from_vector(bundle, query, vector_limit)

    # 5. Load knowledge graph facts
    _retrieve_from_knowledge_graph(bundle, query)

    # 6. Load active goals and projects
    if include_goals:
        _load_goals_and_projects(bundle, user_id)

    # 7. Get current episode info
    _load_current_episode(bundle, session_id)

    # 8. Get recent turns
    _load_recent_turns(bundle, session_id)

    # Update metadata
    bundle.metadata.retrieval_time_ms = int((time.time() - start_time) * 1000)

    return bundle


def _check_caches(bundle: ContextBundle, query: str) -> None:
    """Check fast and semantic caches."""
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import (
            _lookup_fast_cache,
            _lookup_semantic_cache,
        )

        # Check fast cache (exact match)
        fast_hit = _lookup_fast_cache(query)
        if fast_hit:
            bundle.retrieved_memories.fast_cache_hit = fast_hit

        # Check semantic cache (approximate match)
        semantic_hit = _lookup_semantic_cache(query)
        if semantic_hit:
            bundle.retrieved_memories.semantic_cache_hit = semantic_hit

    except Exception as e:
        print(f"[CONTEXT_RETRIEVAL] Cache check failed: {e}")


def _retrieve_from_tiers(
    bundle: ContextBundle,
    query: str,
    stm_limit: int,
    mtm_limit: int,
    ltm_limit: int,
) -> None:
    """Retrieve from tiered memory (STM/MTM/LTM)."""
    tiers_queried = []

    try:
        from brains.memory.brain_memory import BrainMemory

        # Use the librarian's memory as the primary source
        # In practice, we may want to query multiple brains
        memory = BrainMemory("memory_librarian")

        # Retrieve from each tier separately for categorization
        stm_results = memory.retrieve(query=query, limit=stm_limit, tiers=["stm"])
        bundle.retrieved_memories.stm_hits = stm_results
        if stm_results:
            tiers_queried.append("stm")

        mtm_results = memory.retrieve(query=query, limit=mtm_limit, tiers=["mtm"])
        bundle.retrieved_memories.mtm_hits = mtm_results
        if mtm_results:
            tiers_queried.append("mtm")

        ltm_results = memory.retrieve(query=query, limit=ltm_limit, tiers=["ltm"])
        bundle.retrieved_memories.ltm_hits = ltm_results
        if ltm_results:
            tiers_queried.append("ltm")

        bundle.metadata.tiers_queried = tiers_queried

    except Exception as e:
        print(f"[CONTEXT_RETRIEVAL] Tier retrieval failed: {e}")


def _retrieve_from_episodic(
    bundle: ContextBundle,
    query: str,
    user_id: str,
    session_id: str,
    limit: int,
) -> None:
    """Retrieve from episodic memory."""
    try:
        from brains.memory.enhanced_episodic import query_episodes, get_current_episode

        # Detect time-based queries
        time_range = _detect_time_range(query)

        # Query episodes
        episodes = query_episodes(
            user_id=user_id if user_id else None,
            time_range=time_range,
            query=query if not time_range else None,
            limit=limit,
        )

        # Convert episodes to hits
        for episode in episodes:
            hit = {
                "id": episode.episode_id,
                "type": "episode",
                "topic": episode.topic,
                "summary": episode.summary,
                "turn_count": episode.turn_count(),
                "start_time": episode.start_time,
                "end_time": episode.end_time,
                "project_id": episode.project_id,
                "session_id": episode.session_id,
                "turns": [t.to_dict() for t in episode.turns[-3:]],  # Last 3 turns
            }
            bundle.retrieved_memories.episodic_hits.append(hit)

        # Get current episode ID
        current_ep = get_current_episode(session_id)
        if current_ep:
            bundle.current_episode_id = current_ep.episode_id

        bundle.metadata.episodic_search_enabled = True

    except ImportError:
        # Fallback to basic episodic memory
        try:
            from brains.memory.episodic_memory import get_episodes

            episodes = get_episodes(limit=limit)
            for ep in episodes:
                hit = {
                    "id": str(hash(ep.get("question", ""))),
                    "type": "qa_pair",
                    "question": ep.get("question", ""),
                    "answer": ep.get("answer", ""),
                    "timestamp": ep.get("timestamp", 0),
                }
                bundle.retrieved_memories.episodic_hits.append(hit)

        except Exception:
            pass

    except Exception as e:
        print(f"[CONTEXT_RETRIEVAL] Episodic retrieval failed: {e}")


def _detect_time_range(query: str) -> Optional[str]:
    """Detect time-based queries."""
    query_lower = query.lower()

    if "yesterday" in query_lower:
        return "yesterday"
    elif "today" in query_lower:
        return "today"
    elif "last week" in query_lower:
        return "last_week"
    elif "last month" in query_lower:
        return "last_month"

    return None


def _retrieve_from_vector(
    bundle: ContextBundle,
    query: str,
    limit: int,
) -> None:
    """Retrieve from vector index."""
    try:
        from brains.memory.vector_index import (
            search_similar,
            is_vector_index_enabled,
        )

        if not is_vector_index_enabled():
            bundle.metadata.vector_search_enabled = False
            return

        hits = search_similar(query, top_k=limit, min_score=0.3)

        for hit in hits:
            bundle.retrieved_memories.vector_hits.append(hit.to_dict())

        bundle.metadata.vector_search_enabled = True

    except ImportError:
        bundle.metadata.vector_search_enabled = False
    except Exception as e:
        print(f"[CONTEXT_RETRIEVAL] Vector retrieval failed: {e}")
        bundle.metadata.vector_search_enabled = False


def _retrieve_from_knowledge_graph(bundle: ContextBundle, query: str) -> None:
    """Retrieve facts from knowledge graph."""
    try:
        from brains.personal.service.personal_brain import service_api as personal_api

        response = personal_api({
            "op": "SEARCH_KG",
            "payload": {"query": query, "limit": 5}
        })

        if response.get("ok"):
            facts = response.get("payload", {}).get("facts", [])
            bundle.retrieved_memories.knowledge_graph_facts = facts

    except Exception as e:
        print(f"[CONTEXT_RETRIEVAL] KG retrieval failed: {e}")


def _load_goals_and_projects(bundle: ContextBundle, user_id: str) -> None:
    """Load active goals and projects."""
    try:
        from brains.personal.service.personal_brain import service_api as personal_api

        # Get active goals
        response = personal_api({
            "op": "GET_GOALS",
            "payload": {"active_only": True, "limit": 10}
        })

        if response.get("ok"):
            goals = response.get("payload", {}).get("goals", [])
            bundle.goals_and_projects.active_goals = goals

    except Exception as e:
        print(f"[CONTEXT_RETRIEVAL] Goals loading failed: {e}")

    # Projects would be loaded from a project registry if available
    # For now, we infer from episode project_ids
    try:
        project_ids = set()
        for ep_hit in bundle.retrieved_memories.episodic_hits:
            pid = ep_hit.get("project_id")
            if pid:
                project_ids.add(pid)

        bundle.goals_and_projects.active_projects = [
            {"project_id": pid} for pid in project_ids
        ]

    except Exception:
        pass


def _load_current_episode(bundle: ContextBundle, session_id: str) -> None:
    """Load current episode information."""
    if bundle.current_episode_id:
        return  # Already set in episodic retrieval

    try:
        from brains.memory.enhanced_episodic import get_current_episode

        episode = get_current_episode(session_id)
        if episode:
            bundle.current_episode_id = episode.episode_id

    except Exception:
        pass


def _load_recent_turns(bundle: ContextBundle, session_id: str) -> None:
    """Load recent turns from current session."""
    try:
        from brains.memory.enhanced_episodic import get_current_episode

        episode = get_current_episode(session_id)
        if episode:
            # Get last N turns
            for turn in episode.turns[-5:]:
                bundle.recent_turns.append({
                    "question": turn.question,
                    "answer": turn.answer,
                    "timestamp": turn.timestamp,
                })

    except Exception:
        pass

    # Fallback to system_history if enhanced_episodic not available
    if not bundle.recent_turns:
        try:
            from brains.cognitive.system_history.service.system_history_brain import (
                service_api as history_api
            )

            response = history_api({
                "op": "QUERY_HISTORY",
                "payload": {"history_type": "recent", "limit": 5}
            })

            if response.get("ok"):
                questions = response.get("payload", {}).get("questions", [])
                for q in questions:
                    bundle.recent_turns.append({"question": q, "answer": ""})

        except Exception:
            pass


def retrieve_for_query(
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Quick retrieval for a query without full context bundle.

    Returns a flat list of all memory hits.
    """
    bundle = build_context_bundle(
        query=query,
        stm_limit=limit,
        mtm_limit=limit // 2,
        ltm_limit=limit // 4,
        episodic_limit=limit // 2,
        vector_limit=limit // 2,
        include_goals=False,
    )

    return bundle.retrieved_memories.all_hits()


def answer_history_question(
    query: str,
    user_id: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Answer a history-related question using episodic memory.

    This is specifically for questions like:
    - "What did I ask you first today?"
    - "What were we working on yesterday?"
    - "What did we discuss about X?"

    Returns:
        Dict with "answer", "found", "source", and "episodes" fields
    """
    try:
        from brains.memory.enhanced_episodic import query_episodes

        # Detect time range
        time_range = _detect_time_range(query)

        # Extract topic if mentioned
        topic_query = None
        query_lower = query.lower()
        for marker in ["about ", "on ", "regarding ", "with "]:
            if marker in query_lower:
                idx = query_lower.find(marker)
                topic_query = query[idx + len(marker):].strip().rstrip("?")
                break

        # Query episodes
        episodes = query_episodes(
            user_id=user_id if user_id else None,
            time_range=time_range,
            query=topic_query,
            limit=10,
        )

        if not episodes:
            return {
                "answer": "I don't have any recorded episodes for that time period or topic.",
                "found": False,
                "source": "episodic_memory",
                "episodes": [],
            }

        # Build answer
        if "first" in query_lower:
            # First question
            oldest = min(episodes, key=lambda e: e.start_time)
            if oldest.turns:
                first_q = oldest.turns[0].question
                return {
                    "answer": f"Your first question was: '{first_q}'",
                    "found": True,
                    "source": "episodic_memory",
                    "episodes": [oldest.to_dict()],
                }

        # General history
        summaries = []
        for ep in episodes[:3]:
            if ep.summary:
                summaries.append(ep.summary)
            elif ep.turns:
                summaries.append(f"Discussed: {ep.turns[0].question[:50]}")

        answer = "Here's what we discussed: " + "; ".join(summaries)

        return {
            "answer": answer,
            "found": True,
            "source": "episodic_memory",
            "episodes": [ep.to_dict() for ep in episodes[:5]],
        }

    except Exception as e:
        return {
            "answer": f"I couldn't retrieve history: {str(e)}",
            "found": False,
            "source": "error",
            "episodes": [],
        }


# Public API
__all__ = [
    "build_context_bundle",
    "retrieve_for_query",
    "answer_history_question",
]
