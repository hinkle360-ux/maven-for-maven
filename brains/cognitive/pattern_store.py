"""
Unified Pattern Store for Cognitive Brains
==========================================

This module provides a single, shared pattern/weight layer for all cognitive
brains to learn from experience. Instead of each brain hacking its own ad-hoc
pattern memory, they all use this common store.

Pattern Schema:
    - id: unique identifier
    - brain: which brain owns this pattern ('integrator', 'affect_priority', etc.)
    - signature: compact description of the situation
    - context_tags: list of tags like ['direct_question', 'high_stress']
    - action: brain-specific config/choice (dict, varies by brain)
    - score: float [-1.0, 1.0] indicating pattern quality
    - success_count: number of times this pattern worked well
    - failure_count: number of times this pattern failed
    - last_updated: timestamp of last score update
    - frozen: bool, if True pattern never auto-updates (hard safety rules)

Learning Flow:
    1. Brain computes signature from input
    2. Brain retrieves best-matching pattern
    3. Brain applies pattern's action
    4. SELF_REVIEW/Teacher provides verdict
    5. Brain updates pattern score based on verdict
"""

from __future__ import annotations
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from brains.maven_paths import get_brains_path

# Pattern store location - stored under brains/cognitive/pattern_memory/
_PATTERN_STORE_DIR = get_brains_path("cognitive", "pattern_memory")
_PATTERN_STORE_FILE = _PATTERN_STORE_DIR / "pattern_store.json"


class Pattern:
    """Single pattern record with learning metadata."""

    def __init__(
        self,
        id: str,
        brain: str,
        signature: str,
        context_tags: List[str],
        action: Dict[str, Any],
        score: float = 0.0,
        success_count: int = 0,
        failure_count: int = 0,
        last_updated: Optional[float] = None,
        frozen: bool = False
    ):
        self.id = id
        self.brain = brain
        self.signature = signature
        self.context_tags = context_tags or []
        self.action = action
        self.score = max(-1.0, min(1.0, score))  # clamp to [-1, 1]
        self.success_count = success_count
        self.failure_count = failure_count
        self.last_updated = last_updated or time.time()
        self.frozen = frozen

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "brain": self.brain,
            "signature": self.signature,
            "context_tags": self.context_tags,
            "action": self.action,
            "score": self.score,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_updated": self.last_updated,
            "frozen": self.frozen
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Pattern:
        """Deserialize from dictionary."""
        return Pattern(
            id=d.get("id", ""),
            brain=d.get("brain", ""),
            signature=d.get("signature", ""),
            context_tags=d.get("context_tags", []),
            action=d.get("action", {}),
            score=d.get("score", 0.0),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            last_updated=d.get("last_updated"),
            frozen=d.get("frozen", False)
        )


class PatternStore:
    """Thread-safe (file-based) pattern storage for all brains."""

    def __init__(self, store_file: Optional[Path] = None):
        self.store_file = store_file or _PATTERN_STORE_FILE
        self._ensure_store_exists()

    def _ensure_store_exists(self):
        """Create pattern store file if it doesn't exist."""
        try:
            if not self.store_file.exists():
                self.store_file.parent.mkdir(parents=True, exist_ok=True)
                self._save_patterns([])
        except Exception as e:
            print(f"[PATTERN_STORE] Warning: couldn't create store file: {e}")

    def _load_patterns(self) -> List[Pattern]:
        """Load all patterns from disk."""
        try:
            if not self.store_file.exists():
                return []
            with open(self.store_file, 'r') as f:
                data = json.load(f)
                return [Pattern.from_dict(p) for p in data.get("patterns", [])]
        except Exception as e:
            print(f"[PATTERN_STORE] Error loading patterns: {e}")
            return []

    def _save_patterns(self, patterns: List[Pattern]):
        """Save all patterns to disk."""
        try:
            self.store_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_file, 'w') as f:
                json.dump(
                    {"patterns": [p.to_dict() for p in patterns]},
                    f,
                    indent=2
                )
        except Exception as e:
            print(f"[PATTERN_STORE] Error saving patterns: {e}")

    def store_pattern(self, pattern: Pattern):
        """Store or update a pattern."""
        patterns = self._load_patterns()

        # Remove existing pattern with same id
        patterns = [p for p in patterns if p.id != pattern.id]

        # Add new pattern
        patterns.append(pattern)

        self._save_patterns(patterns)
        print(f"[PATTERN_STORED] brain={pattern.brain} sig={pattern.signature} score={pattern.score:.3f}")

    def get_best_pattern(
        self,
        brain: str,
        signature: str,
        context_tags: Optional[List[str]] = None,
        score_threshold: float = 0.0
    ) -> Optional[Pattern]:
        """
        Retrieve the best-matching pattern for a brain/signature.

        Args:
            brain: Brain name (e.g., 'integrator')
            signature: Pattern signature to match
            context_tags: Optional context tags to filter by
            score_threshold: Minimum score to consider (default 0.0)

        Returns:
            Best matching pattern, or None if no good match found
        """
        patterns = self._load_patterns()

        # Filter by brain and signature
        matches = [p for p in patterns if p.brain == brain and p.signature == signature]

        # Further filter by context tags if provided
        if context_tags:
            # Pattern must have at least one matching context tag
            matches = [
                p for p in matches
                if any(tag in p.context_tags for tag in context_tags)
            ]

        # Filter by score threshold
        matches = [p for p in matches if p.score >= score_threshold]

        if not matches:
            return None

        # Return pattern with highest score
        best = max(matches, key=lambda p: p.score)
        print(f"[PATTERN_MATCH] brain={brain} sig={signature} score={best.score:.3f}")
        return best

    def get_patterns_by_brain(self, brain: str) -> List[Pattern]:
        """Get all patterns for a specific brain."""
        patterns = self._load_patterns()
        return [p for p in patterns if p.brain == brain]

    def update_pattern_score(
        self,
        pattern: Pattern,
        reward: float,
        alpha: float = 0.85
    ) -> Pattern:
        """
        Update a pattern's score using exponential moving average.

        Args:
            pattern: Pattern to update
            reward: Feedback signal (+1 good, 0 neutral, -1 bad)
            alpha: Learning rate (0.8-0.95, higher = slower learning)

        Returns:
            Updated pattern
        """
        # Don't update frozen patterns (hard safety rules)
        if pattern.frozen:
            print(f"[PATTERN_UPDATE] Skipping frozen pattern: {pattern.id}")
            return pattern

        # Update score using exponential moving average
        old_score = pattern.score
        new_score = old_score * alpha + reward * (1 - alpha)
        pattern.score = max(-1.0, min(1.0, new_score))  # clamp

        # Update counts
        if reward > 0:
            pattern.success_count += 1
        elif reward < 0:
            pattern.failure_count += 1

        pattern.last_updated = time.time()

        print(f"[PATTERN_UPDATE] brain={pattern.brain} sig={pattern.signature} "
              f"old={old_score:.3f} new={pattern.score:.3f} reward={reward:+.1f}")

        # Save updated pattern
        self.store_pattern(pattern)

        return pattern


# Global pattern store instance
_store = PatternStore()


def get_pattern_store() -> PatternStore:
    """Get the global pattern store instance."""
    return _store


def verdict_to_reward(verdict: str) -> float:
    """
    Convert SELF_REVIEW verdict to reward signal.

    Args:
        verdict: One of 'ok', 'minor_issue', 'major_issue'

    Returns:
        Reward: +1 (good), 0 (neutral), -1 (bad)
    """
    mapping = {
        "ok": 1.0,
        "minor_issue": 0.0,
        "major_issue": -1.0
    }
    return mapping.get(verdict, 0.0)


# Utility functions for brains
def create_pattern_id(brain: str, signature: str) -> str:
    """Create a unique pattern ID from brain and signature."""
    import hashlib
    key = f"{brain}:{signature}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
