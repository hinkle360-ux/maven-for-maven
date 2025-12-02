"""
Tests for the Synonym System
============================

Tests that verify:
1. JSON loads cleanly
2. Known pairs map to same canonical
3. Reverse mapping returns expected synonyms
4. Failure path returns identity
"""

import pytest
import sys
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSynonyms:
    """Test suite for the synonym system."""

    def test_json_loads_cleanly(self):
        """Test that synonyms.json loads without errors."""
        from brains.personal.memory.synonyms import reload_synonyms, get_synonym_count

        # Force reload
        result = reload_synonyms()
        assert result is True, "Synonyms should load successfully"

        # Should have mappings
        count = get_synonym_count()
        assert count > 0, "Should have at least some synonym mappings"

    def test_known_pairs_canonical(self):
        """Test that known synonym pairs map to the same canonical form."""
        from brains.personal.memory.synonyms import canonicalize, are_equivalent

        # Test abbreviations
        assert canonicalize("js") == "javascript"
        assert canonicalize("py") == "python"
        assert canonicalize("nyc") == "new york city"

        # Test equivalence
        assert are_equivalent("js", "javascript") is True
        assert are_equivalent("the red planet", "mars") is True
        assert are_equivalent("big apple", "new york city") is True

    def test_reverse_mapping(self):
        """Test that reverse mapping returns expected synonyms."""
        from brains.personal.memory.synonyms import get_synonyms

        # Get synonyms for canonical forms
        python_synonyms = get_synonyms("python")
        assert "py" in python_synonyms or "phyton" in python_synonyms

        mars_synonyms = get_synonyms("mars")
        assert len(mars_synonyms) >= 1  # Should have at least "the red planet"

    def test_failure_returns_identity(self):
        """Test that unknown terms return identity (no crash)."""
        from brains.personal.memory.synonyms import canonicalize, get_synonyms

        # Unknown term should return itself
        result = canonicalize("xyzunknownterm123")
        assert result == "xyzunknownterm123"

        # Unknown term should return empty synonyms list
        synonyms = get_synonyms("xyzunknownterm123")
        assert synonyms == []

    def test_empty_input(self):
        """Test handling of empty input."""
        from brains.personal.memory.synonyms import canonicalize, are_equivalent, get_synonyms

        assert canonicalize("") == ""
        assert are_equivalent("", "") is True
        assert get_synonyms("") == []

    def test_case_insensitivity(self):
        """Test that canonicalization is case-insensitive."""
        from brains.personal.memory.synonyms import canonicalize

        assert canonicalize("JS") == "javascript"
        assert canonicalize("Js") == "javascript"
        assert canonicalize("NYC") == "new york city"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
