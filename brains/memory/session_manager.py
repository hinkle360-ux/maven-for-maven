"""
session_manager.py
~~~~~~~~~~~~~~~~~~

Session and identity management for cross-session continuity.

This module provides:
- Stable user identity across sessions
- Session lifecycle management
- Cross-session context loading
- Session history persistence

Usage:
    from brains.memory.session_manager import (
        SessionManager,
        get_session_manager,
        start_session,
        end_session,
        get_current_session,
    )

    # Start a new session
    session = start_session(user_id="user123")

    # Get session info
    session_id = session.session_id

    # End the session
    end_session(session_id)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from brains.maven_paths import get_reports_path


# Storage paths
SESSIONS_PATH = get_reports_path("sessions.jsonl")
CURRENT_SESSION_PATH = get_reports_path("current_session.json")
USER_IDENTITY_PATH = get_reports_path("user_identity.json")


@dataclass
class Session:
    """Represents a single session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    is_active: bool = True
    episode_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "chat_cli"  # chat_cli, browser, api, tests

    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            is_active=data.get("is_active", True),
            episode_ids=data.get("episode_ids", []),
            metadata=data.get("metadata", {}),
            source=data.get("source", "chat_cli"),
        )


@dataclass
class UserIdentity:
    """Stable user identity across sessions."""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    display_name: str = ""
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    session_count: int = 0
    total_turns: int = 0
    preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserIdentity":
        return cls(
            user_id=data.get("user_id", str(uuid.uuid4())),
            display_name=data.get("display_name", ""),
            created_at=data.get("created_at", time.time()),
            last_seen=data.get("last_seen", time.time()),
            session_count=data.get("session_count", 0),
            total_turns=data.get("total_turns", 0),
            preferences=data.get("preferences", {}),
        )


class SessionManager:
    """
    Manages session lifecycle and cross-session continuity.

    This is the central point for:
    - Starting/ending sessions
    - Tracking user identity
    - Loading cross-session context
    """

    def __init__(self):
        self._ensure_storage()
        self._current_session: Optional[Session] = None
        self._user_identity: Optional[UserIdentity] = None
        self._load_state()

    def _ensure_storage(self) -> None:
        """Ensure storage files exist."""
        SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not SESSIONS_PATH.exists():
            SESSIONS_PATH.write_text("")

    def _load_state(self) -> None:
        """Load current session and user identity."""
        # Load current session
        try:
            if CURRENT_SESSION_PATH.exists():
                data = json.loads(CURRENT_SESSION_PATH.read_text())
                if data.get("is_active"):
                    self._current_session = Session.from_dict(data)
        except Exception:
            pass

        # Load user identity
        try:
            if USER_IDENTITY_PATH.exists():
                data = json.loads(USER_IDENTITY_PATH.read_text())
                self._user_identity = UserIdentity.from_dict(data)
        except Exception:
            pass

    def _save_current_session(self) -> None:
        """Save current session to disk."""
        try:
            if self._current_session:
                CURRENT_SESSION_PATH.write_text(
                    json.dumps(self._current_session.to_dict(), indent=2)
                )
            elif CURRENT_SESSION_PATH.exists():
                CURRENT_SESSION_PATH.unlink()
        except Exception as e:
            print(f"[SESSION] Failed to save current session: {e}")

    def _save_user_identity(self) -> None:
        """Save user identity to disk."""
        try:
            if self._user_identity:
                USER_IDENTITY_PATH.write_text(
                    json.dumps(self._user_identity.to_dict(), indent=2)
                )
        except Exception as e:
            print(f"[SESSION] Failed to save user identity: {e}")

    def _append_session_log(self, session: Session) -> None:
        """Append session to the log file."""
        try:
            with open(SESSIONS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(session.to_dict()) + "\n")
        except Exception as e:
            print(f"[SESSION] Failed to log session: {e}")

    def get_or_create_user_identity(
        self,
        display_name: str = "",
    ) -> UserIdentity:
        """Get existing user identity or create new one."""
        if self._user_identity is None:
            # Try to load from disk
            try:
                if USER_IDENTITY_PATH.exists():
                    data = json.loads(USER_IDENTITY_PATH.read_text())
                    self._user_identity = UserIdentity.from_dict(data)
            except Exception:
                pass

        if self._user_identity is None:
            # Create new identity
            self._user_identity = UserIdentity(
                display_name=display_name or self._get_system_user(),
            )
            self._save_user_identity()
            print(f"[SESSION] Created new user identity: {self._user_identity.user_id[:8]}")

        return self._user_identity

    def _get_system_user(self) -> str:
        """Get system username."""
        try:
            return os.getenv("USER", os.getenv("USERNAME", "local_user"))
        except Exception:
            return "local_user"

    def start_session(
        self,
        user_id: Optional[str] = None,
        source: str = "chat_cli",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Start a new session.

        If there's an active session, end it first.
        """
        # End existing session
        if self._current_session and self._current_session.is_active:
            self.end_session()

        # Get user identity
        identity = self.get_or_create_user_identity()
        if user_id:
            # Override user_id if provided
            identity.user_id = user_id

        # Create new session
        session = Session(
            user_id=identity.user_id,
            source=source,
            metadata=metadata or {},
        )

        self._current_session = session
        self._save_current_session()

        # Update identity stats
        identity.last_seen = time.time()
        identity.session_count += 1
        self._save_user_identity()

        print(f"[SESSION] Started session {session.session_id[:8]} for user {identity.user_id[:8]}")

        # Start a new episode for this session
        try:
            from brains.memory.enhanced_episodic import start_episode
            episode = start_episode(
                user_id=identity.user_id,
                session_id=session.session_id,
            )
            session.episode_ids.append(episode.episode_id)
            self._save_current_session()
        except ImportError:
            pass

        return session

    def end_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[Session]:
        """
        End the current or specified session.

        Returns the ended session.
        """
        if session_id and (not self._current_session or
                          self._current_session.session_id != session_id):
            # Try to find and end a specific session
            return None

        if not self._current_session:
            return None

        session = self._current_session
        session.is_active = False
        session.end_time = time.time()

        # Close the session's episodes
        try:
            from brains.memory.enhanced_episodic import close_episode
            close_episode(session.session_id)
        except ImportError:
            pass

        # Log the session
        self._append_session_log(session)

        # Clear current session
        self._current_session = None
        self._save_current_session()

        print(f"[SESSION] Ended session {session.session_id[:8]} (duration: {session.duration_seconds():.0f}s)")

        return session

    def get_current_session(self) -> Optional[Session]:
        """Get the current active session."""
        if self._current_session and self._current_session.is_active:
            return self._current_session
        return None

    def get_session_id(self) -> str:
        """Get current session ID, starting one if needed."""
        session = self.get_current_session()
        if not session:
            session = self.start_session()
        return session.session_id

    def get_user_id(self) -> str:
        """Get current user ID."""
        identity = self.get_or_create_user_identity()
        return identity.user_id

    def record_turn(self, question: str, answer: str) -> None:
        """Record a turn in the current session."""
        session = self.get_current_session()
        if not session:
            session = self.start_session()

        # Update user stats
        if self._user_identity:
            self._user_identity.total_turns += 1
            self._save_user_identity()

        # Append to episode
        try:
            from brains.memory.enhanced_episodic import append_turn
            append_turn(
                session_id=session.session_id,
                question=question,
                answer=answer,
            )
        except ImportError:
            pass

    def get_recent_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Session]:
        """Get recent sessions for a user."""
        sessions: List[Session] = []

        try:
            for line in SESSIONS_PATH.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    session = Session.from_dict(data)

                    if user_id and session.user_id != user_id:
                        continue

                    sessions.append(session)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        # Sort by start_time descending
        sessions.sort(key=lambda s: s.start_time, reverse=True)

        return sessions[:limit]

    def load_cross_session_context(
        self,
        user_id: Optional[str] = None,
        session_limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Load context from previous sessions.

        This is the main entry point for cross-session continuity.

        Returns:
            Dict with previous sessions, episodes, and context
        """
        user_id = user_id or self.get_user_id()

        # Get recent sessions
        recent_sessions = self.get_recent_sessions(
            user_id=user_id,
            limit=session_limit,
        )

        # Collect episode IDs from sessions
        episode_ids = []
        for session in recent_sessions:
            episode_ids.extend(session.episode_ids)

        # Load episodes
        episodes = []
        try:
            from brains.memory.enhanced_episodic import get_episode_manager
            manager = get_episode_manager()
            for ep_id in episode_ids[:20]:  # Limit to 20 episodes
                episode = manager.get_episode_by_id(ep_id)
                if episode:
                    episodes.append(episode.to_dict())
        except ImportError:
            pass

        return {
            "user_id": user_id,
            "session_count": len(recent_sessions),
            "recent_sessions": [s.to_dict() for s in recent_sessions[:3]],
            "episodes": episodes[:10],
            "total_turns": self._user_identity.total_turns if self._user_identity else 0,
        }


# Module-level singleton
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Convenience functions
def start_session(
    user_id: Optional[str] = None,
    source: str = "chat_cli",
) -> Session:
    """Start a new session."""
    return get_session_manager().start_session(user_id=user_id, source=source)


def end_session(session_id: Optional[str] = None) -> Optional[Session]:
    """End the current or specified session."""
    return get_session_manager().end_session(session_id)


def get_current_session() -> Optional[Session]:
    """Get the current active session."""
    return get_session_manager().get_current_session()


def get_session_id() -> str:
    """Get current session ID, starting one if needed."""
    return get_session_manager().get_session_id()


def get_user_id() -> str:
    """Get current user ID."""
    return get_session_manager().get_user_id()


def record_turn(question: str, answer: str) -> None:
    """Record a turn in the current session."""
    get_session_manager().record_turn(question, answer)


# Public API
__all__ = [
    "Session",
    "UserIdentity",
    "SessionManager",
    "get_session_manager",
    "start_session",
    "end_session",
    "get_current_session",
    "get_session_id",
    "get_user_id",
    "record_turn",
]
