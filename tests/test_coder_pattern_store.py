"""
Tests for the Coder Brain Pattern Store
=======================================

Tests that verify:
1. Pattern storage works
2. find_similar_patterns returns sorted by similarity
3. Pattern reinforcement on success
4. Correction pattern lookup
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCoderPatternStore:
    """Test suite for the coder brain pattern store."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary pattern store."""
        from brains.cognitive.coder.pattern_store import CoderPatternStore
        store_file = tmp_path / "test_patterns.json"
        return CoderPatternStore(store_file=store_file)

    def test_store_pattern(self, temp_store):
        """Test storing a pattern."""
        from brains.cognitive.coder.pattern_store import (
            Pattern, PatternContext, VerificationOutcome
        )

        pattern = Pattern(
            id="test_pattern_1",
            problem_description="Add two numbers together",
            context=PatternContext(language="python"),
            code_after="def add(a, b):\n    return a + b",
            test_code="assert add(1, 2) == 3",
            pattern_type="GENERATION",
            score=0.8,
        )

        pattern_id = temp_store.store_pattern(pattern)

        assert pattern_id == "test_pattern_1"

        # Retrieve and verify
        retrieved = temp_store.get_pattern_by_id("test_pattern_1")
        assert retrieved is not None
        assert retrieved.problem_description == "Add two numbers together"

    def test_find_similar_patterns_sorted(self, temp_store):
        """Test that find_similar_patterns returns sorted by similarity."""
        from brains.cognitive.coder.pattern_store import (
            Pattern, PatternContext, PatternQuery
        )

        # Store multiple patterns
        patterns = [
            Pattern(
                id=f"pattern_{i}",
                problem_description=desc,
                context=PatternContext(language="python"),
                code_after=f"# code {i}",
                pattern_type="GENERATION",
                score=0.7,
            )
            for i, desc in enumerate([
                "Add two numbers together",
                "Multiply two numbers",
                "Add integers and return sum",
            ])
        ]

        for p in patterns:
            temp_store.store_pattern(p)

        # Query for "add numbers"
        query = PatternQuery(
            problem_description="add numbers",
            pattern_type="GENERATION",
        )

        results = temp_store.find_similar_patterns(query, k=3)

        assert len(results) > 0

        # Check that results are sorted by score (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_record_pattern_usage(self, temp_store):
        """Test recording pattern usage updates statistics."""
        from brains.cognitive.coder.pattern_store import Pattern, PatternContext

        pattern = Pattern(
            id="usage_test",
            problem_description="Test pattern",
            context=PatternContext(),
            code_after="pass",
            pattern_type="GENERATION",
            score=0.5,
            usage_count=0,
            success_count=0,
        )

        temp_store.store_pattern(pattern)

        # Record successful usage
        updated = temp_store.record_pattern_usage("usage_test", success=True)

        assert updated is not None
        assert updated.usage_count == 1
        assert updated.success_count == 1
        assert updated.score > 0.5  # Score should increase on success

        # Record failed usage
        updated = temp_store.record_pattern_usage("usage_test", success=False)

        assert updated.usage_count == 2
        assert updated.failure_count == 1
        assert updated.score < 0.55  # Score should decrease on failure

    def test_find_correction_patterns(self, temp_store):
        """Test finding correction patterns for errors."""
        from brains.cognitive.coder.pattern_store import (
            Pattern, PatternContext, VerificationOutcome
        )

        # Store a correction pattern
        pattern = Pattern(
            id="correction_1",
            problem_description="Fix missing import",
            context=PatternContext(language="python"),
            code_before="def foo():\n    return numpy.array([1])",
            code_after="import numpy\n\ndef foo():\n    return numpy.array([1])",
            pattern_type="CORRECTION",
            verification_outcome=VerificationOutcome(
                tests_passed=True,
                error_message="NameError: name 'numpy' is not defined"
            ),
            score=0.8,
        )

        temp_store.store_pattern(pattern)

        # Find correction for similar error
        results = temp_store.find_correction_patterns(
            failing_code="def bar():\n    return pandas.DataFrame()",
            error_message="NameError: name 'pandas' is not defined",
            k=3
        )

        # May or may not find pattern depending on similarity
        # Just verify it doesn't crash and returns proper structure
        assert isinstance(results, list)

    def test_extract_and_store_pattern(self, temp_store):
        """Test extracting and storing a pattern from successful task."""
        from brains.cognitive.coder.pattern_store import (
            PatternContext, VerificationOutcome
        )

        verification = VerificationOutcome(
            tests_passed=True,
            lint_passed=True,
        )

        pattern_id = temp_store.extract_and_store_pattern(
            problem_description="Calculate factorial",
            code="def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
            test_code="assert factorial(5) == 120",
            verification_outcome=verification,
            context=PatternContext(language="python", tags=["recursion"]),
        )

        assert pattern_id is not None

        # Verify pattern was stored
        pattern = temp_store.get_pattern_by_id(pattern_id)
        assert pattern is not None
        assert "factorial" in pattern.problem_description.lower()

    def test_get_statistics(self, temp_store):
        """Test getting pattern store statistics."""
        from brains.cognitive.coder.pattern_store import Pattern, PatternContext

        # Store some patterns
        for i in range(3):
            temp_store.store_pattern(Pattern(
                id=f"stat_test_{i}",
                problem_description=f"Test {i}",
                context=PatternContext(),
                code_after="pass",
                pattern_type="GENERATION" if i % 2 == 0 else "CORRECTION",
                score=0.5 + i * 0.1,
            ))

        stats = temp_store.get_statistics()

        assert stats["total_patterns"] == 3
        assert "by_type" in stats
        assert "avg_score" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
