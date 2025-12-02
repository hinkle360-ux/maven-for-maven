"""
Coder Pattern Store
===================

Specialized pattern storage for the Coder Brain that enables pattern-learned
code generation, refinement, and correction.

Pattern Schema:
- id: unique identifier
- problem_description: natural language description of the coding task
- context: dict with language, framework, tags
- code_before: code state before transformation (if any)
- code_after: code state after transformation (the solution)
- test_code: test code that validates the solution
- diff: optional unified diff format of the transformation
- verification_outcome: dict with tests_passed, lint_passed, etc.
- score: float [0, 1] indicating pattern quality
- usage_count: number of times pattern was used
- success_count: number of successful applications
- failure_count: number of failed applications
- created_at: timestamp
- last_used: timestamp

Pattern Types:
- GENERATION: Pattern for generating new code
- CORRECTION: Pattern for fixing failing code
- REFACTORING: Pattern for improving code structure
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from difflib import unified_diff, SequenceMatcher

from brains.maven_paths import get_brains_path


# Pattern store location
_PATTERN_STORE_DIR = get_brains_path("cognitive", "coder", "memory")
_PATTERN_STORE_FILE = _PATTERN_STORE_DIR / "coder_patterns.json"


@dataclass
class PatternContext:
    """Context information for a coding pattern."""
    language: str = "python"
    framework: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    complexity: str = "simple"  # simple, medium, complex
    domain: Optional[str] = None


@dataclass
class VerificationOutcome:
    """Outcome of verifying a code pattern."""
    tests_passed: bool = False
    lint_passed: bool = True
    test_output: str = ""
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class Pattern:
    """A coding pattern with learning metadata."""
    id: str
    problem_description: str
    context: PatternContext
    code_before: Optional[str] = None
    code_after: str = ""
    test_code: str = ""
    diff: Optional[str] = None
    verification_outcome: Optional[VerificationOutcome] = None
    pattern_type: str = "GENERATION"  # GENERATION, CORRECTION, REFACTORING
    score: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    frozen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "problem_description": self.problem_description,
            "context": asdict(self.context) if isinstance(self.context, PatternContext) else self.context,
            "code_before": self.code_before,
            "code_after": self.code_after,
            "test_code": self.test_code,
            "diff": self.diff,
            "verification_outcome": asdict(self.verification_outcome) if self.verification_outcome else None,
            "pattern_type": self.pattern_type,
            "score": self.score,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "frozen": self.frozen,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Pattern":
        """Deserialize from dictionary."""
        context_data = d.get("context", {})
        context = PatternContext(
            language=context_data.get("language", "python"),
            framework=context_data.get("framework"),
            tags=context_data.get("tags", []),
            complexity=context_data.get("complexity", "simple"),
            domain=context_data.get("domain"),
        )

        verification_data = d.get("verification_outcome")
        verification = None
        if verification_data:
            verification = VerificationOutcome(
                tests_passed=verification_data.get("tests_passed", False),
                lint_passed=verification_data.get("lint_passed", True),
                test_output=verification_data.get("test_output", ""),
                error_message=verification_data.get("error_message"),
                execution_time_ms=verification_data.get("execution_time_ms", 0.0),
            )

        return Pattern(
            id=d.get("id", ""),
            problem_description=d.get("problem_description", ""),
            context=context,
            code_before=d.get("code_before"),
            code_after=d.get("code_after", ""),
            test_code=d.get("test_code", ""),
            diff=d.get("diff"),
            verification_outcome=verification,
            pattern_type=d.get("pattern_type", "GENERATION"),
            score=d.get("score", 0.5),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            created_at=d.get("created_at", time.time()),
            last_used=d.get("last_used", time.time()),
            frozen=d.get("frozen", False),
        )


@dataclass
class PatternQuery:
    """Query for finding similar patterns."""
    problem_description: str
    context: Optional[PatternContext] = None
    pattern_type: Optional[str] = None
    min_score: float = 0.0
    min_success_rate: float = 0.0
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class CoderPatternStore:
    """Pattern store for the Coder Brain with similarity search."""

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
            print(f"[CODER_PATTERN_STORE] Warning: couldn't create store file: {e}")

    def _load_patterns(self) -> List[Pattern]:
        """Load all patterns from disk."""
        try:
            if not self.store_file.exists():
                return []
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Pattern.from_dict(p) for p in data.get("patterns", [])]
        except Exception as e:
            print(f"[CODER_PATTERN_STORE] Error loading patterns: {e}")
            return []

    def _save_patterns(self, patterns: List[Pattern]):
        """Save all patterns to disk."""
        try:
            self.store_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "patterns": [p.to_dict() for p in patterns],
                        "last_updated": time.time(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[CODER_PATTERN_STORE] Error saving patterns: {e}")

    def _compute_similarity(self, query_text: str, pattern: Pattern) -> float:
        """
        Compute similarity between a query and a pattern.

        Uses multiple signals:
        - Text similarity of problem descriptions
        - Keyword overlap
        - Context matching (language, tags)

        Returns:
            Float between 0 and 1
        """
        if not query_text or not pattern.problem_description:
            return 0.0

        query_lower = query_text.lower()
        pattern_desc_lower = pattern.problem_description.lower()

        # 1. Sequence matcher for text similarity (0.4 weight)
        seq_ratio = SequenceMatcher(None, query_lower, pattern_desc_lower).ratio()

        # 2. Keyword overlap (0.4 weight)
        query_words = set(query_lower.split())
        pattern_words = set(pattern_desc_lower.split())

        common_words = query_words & pattern_words
        total_words = query_words | pattern_words

        keyword_ratio = len(common_words) / len(total_words) if total_words else 0.0

        # 3. Code content similarity (0.2 weight)
        code_ratio = 0.0
        if pattern.code_after:
            code_lower = pattern.code_after.lower()
            # Check if query terms appear in code
            matches = sum(1 for word in query_words if word in code_lower)
            code_ratio = min(1.0, matches / max(1, len(query_words)))

        # Weighted combination
        similarity = 0.4 * seq_ratio + 0.4 * keyword_ratio + 0.2 * code_ratio

        return similarity

    def _matches_context(self, query: PatternQuery, pattern: Pattern) -> bool:
        """Check if pattern matches query context constraints."""
        # Check language
        if query.language and pattern.context.language != query.language:
            return False

        # Check pattern type
        if query.pattern_type and pattern.pattern_type != query.pattern_type:
            return False

        # Check tags (at least one tag should match if query has tags)
        if query.tags:
            pattern_tags = set(pattern.context.tags)
            query_tags = set(query.tags)
            if not pattern_tags & query_tags:
                return False

        return True

    def _compute_success_rate(self, pattern: Pattern) -> float:
        """Compute success rate of a pattern."""
        total = pattern.success_count + pattern.failure_count
        if total == 0:
            return 0.5  # Neutral for unused patterns
        return pattern.success_count / total

    def store_pattern(self, pattern: Pattern) -> str:
        """
        Store or update a pattern.

        Args:
            pattern: Pattern to store

        Returns:
            Pattern ID
        """
        patterns = self._load_patterns()

        # Generate ID if not present
        if not pattern.id:
            pattern.id = self._generate_pattern_id(pattern)

        # Remove existing pattern with same ID
        patterns = [p for p in patterns if p.id != pattern.id]

        # Add new pattern
        patterns.append(pattern)

        self._save_patterns(patterns)
        print(f"[CODER_PATTERN_STORE] Stored pattern: id={pattern.id}, "
              f"type={pattern.pattern_type}, score={pattern.score:.3f}")

        return pattern.id

    def find_similar_patterns(
        self,
        query: PatternQuery,
        k: int = 5
    ) -> List[Tuple[Pattern, float]]:
        """
        Find patterns similar to the query.

        Args:
            query: Pattern query with search criteria
            k: Maximum number of patterns to return

        Returns:
            List of (pattern, similarity_score) tuples, sorted by similarity (descending)
        """
        patterns = self._load_patterns()

        scored_patterns: List[Tuple[Pattern, float]] = []

        for pattern in patterns:
            # Check minimum score threshold
            if pattern.score < query.min_score:
                continue

            # Check success rate threshold
            success_rate = self._compute_success_rate(pattern)
            if success_rate < query.min_success_rate:
                continue

            # Check context constraints
            if not self._matches_context(query, pattern):
                continue

            # Compute similarity
            similarity = self._compute_similarity(query.problem_description, pattern)

            # Boost score based on pattern quality
            adjusted_score = similarity * 0.6 + pattern.score * 0.2 + success_rate * 0.2

            scored_patterns.append((pattern, adjusted_score))

        # Sort by score (descending)
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        results = scored_patterns[:k]

        if results:
            print(f"[CODER_PATTERN_STORE] Found {len(results)} similar patterns "
                  f"for query: '{query.problem_description[:50]}...'")

        return results

    def record_pattern_usage(
        self,
        pattern_id: str,
        success: bool,
        verification_outcome: Optional[VerificationOutcome] = None
    ) -> Optional[Pattern]:
        """
        Record that a pattern was used and update its statistics.

        Args:
            pattern_id: ID of the pattern that was used
            success: Whether the usage was successful
            verification_outcome: Optional verification details

        Returns:
            Updated pattern, or None if not found
        """
        patterns = self._load_patterns()

        for pattern in patterns:
            if pattern.id == pattern_id:
                if pattern.frozen:
                    print(f"[CODER_PATTERN_STORE] Pattern {pattern_id} is frozen, not updating")
                    return pattern

                pattern.usage_count += 1
                pattern.last_used = time.time()

                if success:
                    pattern.success_count += 1
                    # Increase score on success
                    pattern.score = min(1.0, pattern.score + 0.05)
                else:
                    pattern.failure_count += 1
                    # Decrease score on failure
                    pattern.score = max(0.0, pattern.score - 0.1)

                if verification_outcome:
                    pattern.verification_outcome = verification_outcome

                self._save_patterns(patterns)
                print(f"[CODER_PATTERN_STORE] Updated pattern {pattern_id}: "
                      f"success={success}, score={pattern.score:.3f}")
                return pattern

        return None

    def extract_and_store_pattern(
        self,
        problem_description: str,
        code: str,
        test_code: str,
        verification_outcome: VerificationOutcome,
        context: Optional[PatternContext] = None,
        code_before: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract a normalized pattern from a successful coding task and store it.

        This is called during pattern reinforcement when a coding task succeeds.

        Args:
            problem_description: Natural language description of the task
            code: The generated/fixed code
            test_code: Tests that validate the code
            verification_outcome: Outcome of verification
            context: Optional context information
            code_before: Optional original code (for correction patterns)

        Returns:
            Pattern ID if stored, None if skipped
        """
        # Only store patterns that pass tests
        if not verification_outcome.tests_passed:
            print("[CODER_PATTERN_STORE] Skipping pattern storage - tests did not pass")
            return None

        # Create pattern
        pattern_type = "GENERATION"
        diff = None

        if code_before:
            pattern_type = "CORRECTION"
            # Compute diff
            diff_lines = list(unified_diff(
                code_before.splitlines(keepends=True),
                code.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            ))
            diff = "".join(diff_lines) if diff_lines else None

        context = context or PatternContext()

        pattern = Pattern(
            id="",  # Will be generated
            problem_description=problem_description,
            context=context,
            code_before=code_before,
            code_after=code,
            test_code=test_code,
            diff=diff,
            verification_outcome=verification_outcome,
            pattern_type=pattern_type,
            score=0.7,  # Start with a decent score for verified patterns
            success_count=1,  # Already succeeded once
        )

        return self.store_pattern(pattern)

    def find_correction_patterns(
        self,
        failing_code: str,
        error_message: str,
        k: int = 3
    ) -> List[Tuple[Pattern, float]]:
        """
        Find patterns where similar failures were fixed.

        Used during pattern-based correction when code fails tests.

        Args:
            failing_code: The code that is failing
            error_message: The error message from test failure
            k: Maximum patterns to return

        Returns:
            List of (pattern, similarity) tuples
        """
        patterns = self._load_patterns()

        correction_patterns: List[Tuple[Pattern, float]] = []

        for pattern in patterns:
            # Only look at correction patterns
            if pattern.pattern_type != "CORRECTION":
                continue

            # Check if pattern had similar error
            if pattern.verification_outcome and pattern.verification_outcome.error_message:
                error_sim = SequenceMatcher(
                    None,
                    error_message.lower()[:200],
                    pattern.verification_outcome.error_message.lower()[:200]
                ).ratio()
            else:
                error_sim = 0.0

            # Check code similarity with code_before
            code_sim = 0.0
            if pattern.code_before:
                code_sim = SequenceMatcher(
                    None,
                    failing_code[:500],
                    pattern.code_before[:500]
                ).ratio()

            # Combined score
            if error_sim > 0.3 or code_sim > 0.3:
                combined = 0.5 * error_sim + 0.5 * code_sim
                correction_patterns.append((pattern, combined))

        correction_patterns.sort(key=lambda x: x[1], reverse=True)

        results = correction_patterns[:k]

        if results:
            print(f"[CODER_PATTERN_STORE] Found {len(results)} correction patterns "
                  f"for error: '{error_message[:50]}...'")

        return results

    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by its ID."""
        patterns = self._load_patterns()
        for pattern in patterns:
            if pattern.id == pattern_id:
                return pattern
        return None

    def get_all_patterns(self) -> List[Pattern]:
        """Get all patterns."""
        return self._load_patterns()

    def get_patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        """Get all patterns of a specific type."""
        patterns = self._load_patterns()
        return [p for p in patterns if p.pattern_type == pattern_type]

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern by ID."""
        patterns = self._load_patterns()
        original_count = len(patterns)
        patterns = [p for p in patterns if p.id != pattern_id]

        if len(patterns) < original_count:
            self._save_patterns(patterns)
            print(f"[CODER_PATTERN_STORE] Deleted pattern: {pattern_id}")
            return True
        return False

    def _generate_pattern_id(self, pattern: Pattern) -> str:
        """Generate a unique ID for a pattern."""
        key = f"{pattern.problem_description}:{pattern.pattern_type}:{pattern.code_after[:100]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pattern store."""
        patterns = self._load_patterns()

        if not patterns:
            return {
                "total_patterns": 0,
                "by_type": {},
                "avg_score": 0.0,
                "avg_success_rate": 0.0,
            }

        by_type: Dict[str, int] = {}
        total_score = 0.0
        total_success_rate = 0.0

        for pattern in patterns:
            by_type[pattern.pattern_type] = by_type.get(pattern.pattern_type, 0) + 1
            total_score += pattern.score
            total_success_rate += self._compute_success_rate(pattern)

        return {
            "total_patterns": len(patterns),
            "by_type": by_type,
            "avg_score": total_score / len(patterns),
            "avg_success_rate": total_success_rate / len(patterns),
        }


# Global instance
_store: Optional[CoderPatternStore] = None


def get_coder_pattern_store() -> CoderPatternStore:
    """Get the global coder pattern store instance."""
    global _store
    if _store is None:
        _store = CoderPatternStore()
    return _store


# Convenience functions
def store_pattern(pattern: Pattern) -> str:
    """Convenience function to store a pattern."""
    return get_coder_pattern_store().store_pattern(pattern)


def find_similar_patterns(query: PatternQuery, k: int = 5) -> List[Tuple[Pattern, float]]:
    """Convenience function to find similar patterns."""
    return get_coder_pattern_store().find_similar_patterns(query, k)


def record_usage(pattern_id: str, success: bool) -> Optional[Pattern]:
    """Convenience function to record pattern usage."""
    return get_coder_pattern_store().record_pattern_usage(pattern_id, success)
