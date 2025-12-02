"""
Browser Pattern Store
====================

Stores and retrieves learned browsing patterns for reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from optional.maven_browser_client.types import PatternMatch, BrowserPlan
from optional.browser_runtime.config import get_config


class PatternStore:
    """Stores and manages browsing patterns."""

    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize pattern store.

        Args:
            store_path: Path to pattern store file. If None, uses default from config.
        """
        if store_path is None:
            config = get_config()
            store_path = config.log_dir / "patterns" / "patterns.json"

        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with default patterns
        self.patterns: List[PatternMatch] = []
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.patterns = [PatternMatch(**p) for p in data]
            except Exception as e:
                print(f"Failed to load patterns: {e}")
                self.patterns = []
        else:
            # Initialize with default patterns
            self._initialize_default_patterns()

    def _save_patterns(self) -> None:
        """Save patterns to disk."""
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                data = [p.model_dump(mode="json") for p in self.patterns]
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save patterns: {e}")

    def _initialize_default_patterns(self) -> None:
        """Initialize with default patterns."""
        from maven_browser_client.types import GOOGLE_SEARCH_PATTERN, OPEN_URL_PATTERN

        default_patterns = [
            PatternMatch(
                name="google_search",
                description="Search Google for a query",
                trigger_keywords=["search", "google", "find", "lookup"],
                domains=["google.com", "www.google.com"],
                template_plan=GOOGLE_SEARCH_PATTERN,
                success_count=0,
                failure_count=0,
            ),
            PatternMatch(
                name="open_url",
                description="Open a specific URL",
                trigger_keywords=["open", "navigate", "goto", "visit"],
                domains=[],
                template_plan=OPEN_URL_PATTERN,
                success_count=0,
                failure_count=0,
            ),
        ]

        self.patterns = default_patterns
        self._save_patterns()

    def find_pattern(self, goal: str, keywords: Optional[List[str]] = None) -> Optional[PatternMatch]:
        """
        Find a matching pattern for a goal.

        Args:
            goal: Goal description
            keywords: Optional list of keywords to match against

        Returns:
            Matching PatternMatch or None
        """
        goal_lower = goal.lower()

        # Score each pattern
        scored_patterns = []
        for pattern in self.patterns:
            score = 0

            # Check trigger keywords in goal
            for keyword in pattern.trigger_keywords:
                if keyword.lower() in goal_lower:
                    score += 1

            # Bonus for successful patterns
            if pattern.success_count > 0:
                success_rate = pattern.success_count / (pattern.success_count + pattern.failure_count)
                score += success_rate

            if score > 0:
                scored_patterns.append((score, pattern))

        # Return best match
        if scored_patterns:
            scored_patterns.sort(key=lambda x: x[0], reverse=True)
            return scored_patterns[0][1]

        return None

    def record_success(self, pattern_name: str) -> None:
        """
        Record a successful pattern execution.

        Args:
            pattern_name: Name of the pattern
        """
        for pattern in self.patterns:
            if pattern.name == pattern_name:
                pattern.success_count += 1
                pattern.last_used = datetime.now(timezone.utc)
                self._save_patterns()
                break

    def record_failure(self, pattern_name: str) -> None:
        """
        Record a failed pattern execution.

        Args:
            pattern_name: Name of the pattern
        """
        for pattern in self.patterns:
            if pattern.name == pattern_name:
                pattern.failure_count += 1
                pattern.last_used = datetime.now(timezone.utc)
                self._save_patterns()
                break

    def add_pattern(self, pattern: PatternMatch) -> None:
        """
        Add a new pattern to the store.

        Args:
            pattern: PatternMatch to add
        """
        # Check if pattern with same name exists
        for i, existing in enumerate(self.patterns):
            if existing.name == pattern.name:
                # Update existing pattern
                self.patterns[i] = pattern
                self._save_patterns()
                return

        # Add new pattern
        self.patterns.append(pattern)
        self._save_patterns()

    def get_pattern(self, name: str) -> Optional[PatternMatch]:
        """
        Get a pattern by name.

        Args:
            name: Pattern name

        Returns:
            PatternMatch or None
        """
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        return None

    def list_patterns(self) -> List[PatternMatch]:
        """
        List all patterns.

        Returns:
            List of PatternMatch
        """
        return self.patterns.copy()


# Global pattern store instance
_pattern_store: Optional[PatternStore] = None


def get_pattern_store() -> PatternStore:
    """Get the global pattern store instance."""
    global _pattern_store
    if _pattern_store is None:
        _pattern_store = PatternStore()
    return _pattern_store
