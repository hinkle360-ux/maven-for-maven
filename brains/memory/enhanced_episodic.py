"""
enhanced_episodic.py
~~~~~~~~~~~~~~~~~~~~

Enhanced episodic memory with session boundaries, project linking, and summaries.

This module extends the basic episodic memory to support:
- Session-aware episode grouping
- Project/goal linking
- Episode summarization for LTM consolidation
- Cross-session retrieval

The enhanced episodic memory integrates with the ContextBundle for unified access.

Usage:
    from brains.memory.enhanced_episodic import (
        EpisodeManager,
        Episode,
        start_episode,
        append_turn,
        close_episode,
        query_episodes,
    )

    # Start an episode for a session
    episode = start_episode(user_id="user123", session_id="sess456")

    # Append turns
    append_turn(episode.episode_id, question="What is Python?", answer="...")

    # Query episodes by time or project
    episodes = query_episodes(user_id="user123", time_range="yesterday")
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from brains.maven_paths import get_reports_path


# Storage path
EPISODES_PATH = get_reports_path("episodes.jsonl")
EPISODES_INDEX_PATH = get_reports_path("episodes_index.json")


@dataclass
class Turn:
    """A single turn in an episode (question + answer)."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(
            turn_id=data.get("turn_id", str(uuid.uuid4())),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            timestamp=data.get("timestamp", time.time()),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Episode:
    """
    An episode is a cluster of turns around a topic/goal.

    Episodes are the primary unit of episodic memory, grouping related
    interactions for later retrieval.
    """
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    project_id: Optional[str] = None
    goal_ids: List[str] = field(default_factory=list)
    topic: str = ""
    summary: str = ""
    turns: List[Turn] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    is_closed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, question: str, answer: str, confidence: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None) -> Turn:
        """Add a turn to this episode."""
        turn = Turn(
            question=question,
            answer=answer,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        return turn

    def close(self, summary: Optional[str] = None) -> None:
        """Close the episode and optionally set a summary."""
        self.is_closed = True
        self.end_time = time.time()
        if summary:
            self.summary = summary
        elif not self.summary and self.turns:
            # Auto-generate a simple summary
            first_q = self.turns[0].question[:100] if self.turns else ""
            self.summary = f"Discussion about: {first_q}"

    def duration_seconds(self) -> float:
        """Get the duration of the episode in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def turn_count(self) -> int:
        """Get the number of turns in the episode."""
        return len(self.turns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "goal_ids": self.goal_ids,
            "topic": self.topic,
            "summary": self.summary,
            "turns": [t.to_dict() for t in self.turns],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_closed": self.is_closed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        episode = cls(
            episode_id=data.get("episode_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            project_id=data.get("project_id"),
            goal_ids=data.get("goal_ids", []),
            topic=data.get("topic", ""),
            summary=data.get("summary", ""),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            is_closed=data.get("is_closed", False),
            metadata=data.get("metadata", {}),
        )
        episode.turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return episode


class EpisodeManager:
    """
    Manages episode lifecycle: creation, updates, retrieval, and persistence.

    This is the primary interface for episodic memory operations.
    """

    def __init__(self):
        self._ensure_storage()
        self._current_episodes: Dict[str, Episode] = {}  # session_id -> Episode
        self._index: Dict[str, List[str]] = {}  # user_id -> [episode_ids]
        self._load_index()

    def _ensure_storage(self) -> None:
        """Ensure storage files exist."""
        EPISODES_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not EPISODES_PATH.exists():
            EPISODES_PATH.write_text("")
        if not EPISODES_INDEX_PATH.exists():
            EPISODES_INDEX_PATH.write_text("{}")

    def _load_index(self) -> None:
        """Load the episode index from disk."""
        try:
            if EPISODES_INDEX_PATH.exists():
                content = EPISODES_INDEX_PATH.read_text().strip()
                if content:
                    self._index = json.loads(content)
        except Exception:
            self._index = {}

    def _save_index(self) -> None:
        """Save the episode index to disk."""
        try:
            EPISODES_INDEX_PATH.write_text(json.dumps(self._index, indent=2))
        except Exception as e:
            print(f"[EPISODIC] Failed to save index: {e}")

    def _load_episode(self, episode_id: str) -> Optional[Episode]:
        """Load a single episode from storage."""
        try:
            for line in EPISODES_PATH.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("episode_id") == episode_id:
                        return Episode.from_dict(data)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        return None

    def _save_episode(self, episode: Episode) -> None:
        """Append or update an episode in storage."""
        try:
            # Read all episodes
            episodes = []
            if EPISODES_PATH.exists():
                for line in EPISODES_PATH.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("episode_id") != episode.episode_id:
                            episodes.append(data)
                    except json.JSONDecodeError:
                        continue

            # Add/update the episode
            episodes.append(episode.to_dict())

            # Write back
            content = "\n".join(json.dumps(ep) for ep in episodes)
            if content:
                content += "\n"
            EPISODES_PATH.write_text(content)

            # Update index
            user_id = episode.user_id
            if user_id not in self._index:
                self._index[user_id] = []
            if episode.episode_id not in self._index[user_id]:
                self._index[user_id].append(episode.episode_id)
            self._save_index()

        except Exception as e:
            print(f"[EPISODIC] Failed to save episode: {e}")

    def start_episode(
        self,
        user_id: str = "",
        session_id: str = "",
        project_id: Optional[str] = None,
        goal_ids: Optional[List[str]] = None,
        topic: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """
        Start a new episode.

        If there's an existing open episode for this session, close it first.
        """
        # Close existing episode for this session
        if session_id in self._current_episodes:
            self.close_episode(session_id)

        episode = Episode(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            goal_ids=goal_ids or [],
            topic=topic,
            metadata=metadata or {},
        )

        self._current_episodes[session_id] = episode
        self._save_episode(episode)

        print(f"[EPISODIC] Started episode {episode.episode_id[:8]} for session {session_id[:8] if session_id else 'unknown'}")
        return episode

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Turn]:
        """
        Append a turn to the current episode for a session.

        If no episode exists, creates one.
        """
        if session_id not in self._current_episodes:
            # Auto-create episode
            self.start_episode(session_id=session_id)

        episode = self._current_episodes.get(session_id)
        if not episode:
            return None

        turn = episode.add_turn(
            question=question,
            answer=answer,
            confidence=confidence,
            metadata=metadata,
        )

        # Persist
        self._save_episode(episode)

        return turn

    def close_episode(
        self,
        session_id: str,
        summary: Optional[str] = None,
    ) -> Optional[Episode]:
        """
        Close the current episode for a session.

        Returns the closed episode or None if no episode was open.
        """
        episode = self._current_episodes.pop(session_id, None)
        if not episode:
            return None

        episode.close(summary=summary)
        self._save_episode(episode)

        print(f"[EPISODIC] Closed episode {episode.episode_id[:8]} with {episode.turn_count()} turns")
        return episode

    def get_current_episode(self, session_id: str) -> Optional[Episode]:
        """Get the current open episode for a session."""
        return self._current_episodes.get(session_id)

    def query_episodes(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        time_range: Optional[str] = None,  # "today", "yesterday", "last_week"
        query: Optional[str] = None,  # Text search
        limit: int = 10,
        include_open: bool = True,
    ) -> List[Episode]:
        """
        Query episodes with various filters.

        Args:
            user_id: Filter by user
            session_id: Filter by session
            project_id: Filter by project
            time_range: Time-based filter (today, yesterday, last_week)
            query: Text search in topics/summaries/turns
            limit: Maximum number of episodes to return
            include_open: Whether to include currently open episodes

        Returns:
            List of matching episodes, most recent first
        """
        episodes: List[Episode] = []

        # Load all episodes
        try:
            for line in EPISODES_PATH.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    episode = Episode.from_dict(data)

                    # Apply filters
                    if user_id and episode.user_id != user_id:
                        continue
                    if session_id and episode.session_id != session_id:
                        continue
                    if project_id and episode.project_id != project_id:
                        continue
                    if not include_open and not episode.is_closed:
                        continue

                    # Time range filter
                    if time_range:
                        if not self._matches_time_range(episode, time_range):
                            continue

                    # Text search
                    if query:
                        if not self._matches_query(episode, query):
                            continue

                    episodes.append(episode)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"[EPISODIC] Query error: {e}")

        # Include current open episodes
        if include_open:
            for ep in self._current_episodes.values():
                if ep not in episodes:
                    # Apply same filters
                    if user_id and ep.user_id != user_id:
                        continue
                    if session_id and ep.session_id != session_id:
                        continue
                    if project_id and ep.project_id != project_id:
                        continue
                    if time_range and not self._matches_time_range(ep, time_range):
                        continue
                    if query and not self._matches_query(ep, query):
                        continue
                    episodes.append(ep)

        # Sort by start_time descending (most recent first)
        episodes.sort(key=lambda e: e.start_time, reverse=True)

        return episodes[:limit]

    def _matches_time_range(self, episode: Episode, time_range: str) -> bool:
        """Check if episode matches a time range."""
        now = datetime.now()
        episode_time = datetime.fromtimestamp(episode.start_time)

        if time_range == "today":
            return episode_time.date() == now.date()
        elif time_range == "yesterday":
            yesterday = now - timedelta(days=1)
            return episode_time.date() == yesterday.date()
        elif time_range == "last_week":
            week_ago = now - timedelta(days=7)
            return episode_time >= week_ago
        elif time_range == "last_month":
            month_ago = now - timedelta(days=30)
            return episode_time >= month_ago

        return True

    def _matches_query(self, episode: Episode, query: str) -> bool:
        """Check if episode matches a text query."""
        query_lower = query.lower()

        # Check topic and summary
        if query_lower in episode.topic.lower():
            return True
        if query_lower in episode.summary.lower():
            return True

        # Check turns
        for turn in episode.turns:
            if query_lower in turn.question.lower():
                return True
            if query_lower in turn.answer.lower():
                return True

        return False

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        # Check current episodes first
        for ep in self._current_episodes.values():
            if ep.episode_id == episode_id:
                return ep

        # Load from storage
        return self._load_episode(episode_id)

    def summarize_for_ltm(
        self,
        user_id: Optional[str] = None,
        older_than_days: int = 7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get episode summaries suitable for LTM consolidation.

        Returns summaries of closed episodes older than the specified days,
        suitable for embedding and long-term storage.
        """
        summaries = []
        cutoff = time.time() - (older_than_days * 86400)

        episodes = self.query_episodes(
            user_id=user_id,
            include_open=False,
            limit=100,  # Get more, then filter
        )

        for episode in episodes:
            if episode.start_time > cutoff:
                continue  # Too recent

            if not episode.is_closed:
                continue

            # Create LTM summary
            summary = {
                "episode_id": episode.episode_id,
                "user_id": episode.user_id,
                "project_id": episode.project_id,
                "topic": episode.topic,
                "summary": episode.summary or self._generate_summary(episode),
                "turn_count": episode.turn_count(),
                "duration_seconds": episode.duration_seconds(),
                "start_time": episode.start_time,
                "end_time": episode.end_time,
                "key_questions": [t.question for t in episode.turns[:3]],
            }
            summaries.append(summary)

            if len(summaries) >= limit:
                break

        return summaries

    def _generate_summary(self, episode: Episode) -> str:
        """Generate a simple summary for an episode without one."""
        if not episode.turns:
            return "Empty episode"

        topics = []
        for turn in episode.turns[:3]:
            q = turn.question[:50]
            topics.append(q)

        return f"Discussed: {'; '.join(topics)}"

    def link_to_project(self, episode_id: str, project_id: str) -> bool:
        """Link an episode to a project."""
        # Check current episodes
        for ep in self._current_episodes.values():
            if ep.episode_id == episode_id:
                ep.project_id = project_id
                self._save_episode(ep)
                return True

        # Load and update from storage
        episode = self._load_episode(episode_id)
        if episode:
            episode.project_id = project_id
            self._save_episode(episode)
            return True

        return False

    def link_to_goals(self, episode_id: str, goal_ids: List[str]) -> bool:
        """Link an episode to goals."""
        # Check current episodes
        for ep in self._current_episodes.values():
            if ep.episode_id == episode_id:
                ep.goal_ids = list(set(ep.goal_ids + goal_ids))
                self._save_episode(ep)
                return True

        # Load and update from storage
        episode = self._load_episode(episode_id)
        if episode:
            episode.goal_ids = list(set(episode.goal_ids + goal_ids))
            self._save_episode(episode)
            return True

        return False


# Module-level singleton
_episode_manager: Optional[EpisodeManager] = None


def get_episode_manager() -> EpisodeManager:
    """Get the singleton episode manager."""
    global _episode_manager
    if _episode_manager is None:
        _episode_manager = EpisodeManager()
    return _episode_manager


# Convenience functions
def start_episode(
    user_id: str = "",
    session_id: str = "",
    project_id: Optional[str] = None,
    goal_ids: Optional[List[str]] = None,
    topic: str = "",
) -> Episode:
    """Start a new episode."""
    return get_episode_manager().start_episode(
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        goal_ids=goal_ids,
        topic=topic,
    )


def append_turn(
    session_id: str,
    question: str,
    answer: str,
    confidence: float = 1.0,
) -> Optional[Turn]:
    """Append a turn to the current episode."""
    return get_episode_manager().append_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        confidence=confidence,
    )


def close_episode(session_id: str, summary: Optional[str] = None) -> Optional[Episode]:
    """Close the current episode."""
    return get_episode_manager().close_episode(session_id, summary)


def query_episodes(
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    time_range: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 10,
) -> List[Episode]:
    """Query episodes with filters."""
    return get_episode_manager().query_episodes(
        user_id=user_id,
        project_id=project_id,
        time_range=time_range,
        query=query,
        limit=limit,
    )


def get_current_episode(session_id: str) -> Optional[Episode]:
    """Get the current open episode for a session."""
    return get_episode_manager().get_current_episode(session_id)


# Public API
__all__ = [
    "Episode",
    "Turn",
    "EpisodeManager",
    "get_episode_manager",
    "start_episode",
    "append_turn",
    "close_episode",
    "query_episodes",
    "get_current_episode",
]
