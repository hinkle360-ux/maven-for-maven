"""
Tests for Context Management
============================

Tests that verify:
1. apply_decay reduces numeric fields
2. reconstruct_context merges history
3. _ensure_pattern_object handles various types
4. Service API operations work correctly
"""

import pytest
import sys
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestApplyDecay:
    """Test suite for apply_decay function."""

    def test_decay_numeric_fields(self):
        """Test that numeric fields are decayed."""
        from brains.cognitive.context_management.service.context_manager import apply_decay

        ctx = {
            "score": 1.0,
            "count": 100,
            "name": "test"
        }

        result = apply_decay(ctx, decay=0.5)

        assert result["score"] == 0.5  # 1.0 * 0.5
        assert result["count"] == 50   # 100 * 0.5
        assert result["name"] == "test"  # Non-numeric unchanged

    def test_decay_nested_dict(self):
        """Test that nested dicts are decayed recursively."""
        from brains.cognitive.context_management.service.context_manager import apply_decay

        ctx = {
            "outer": 10,
            "nested": {
                "inner": 20
            }
        }

        result = apply_decay(ctx, decay=0.5)

        assert result["outer"] == 5
        assert result["nested"]["inner"] == 10

    def test_decay_preserves_strings(self):
        """Test that string fields are preserved."""
        from brains.cognitive.context_management.service.context_manager import apply_decay

        ctx = {
            "text": "hello world",
            "value": 100
        }

        result = apply_decay(ctx, decay=0.9)

        assert result["text"] == "hello world"
        assert result["value"] == 90

    def test_decay_empty_dict(self):
        """Test decay of empty dict."""
        from brains.cognitive.context_management.service.context_manager import apply_decay

        result = apply_decay({}, decay=0.5)

        assert result == {}

    def test_decay_non_dict_input(self):
        """Test decay with non-dict input."""
        from brains.cognitive.context_management.service.context_manager import apply_decay

        result = apply_decay("not a dict", decay=0.5)

        assert result == {}


class TestReconstructContext:
    """Test suite for reconstruct_context function."""

    def test_merge_simple_contexts(self):
        """Test merging simple contexts."""
        from brains.cognitive.context_management.service.context_manager import reconstruct_context

        history = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4}
        ]

        result = reconstruct_context(history)

        assert result["a"] == 1
        assert result["b"] == 3  # Later value wins
        assert result["c"] == 4

    def test_merge_lists(self):
        """Test that lists are concatenated."""
        from brains.cognitive.context_management.service.context_manager import reconstruct_context

        history = [
            {"items": [1, 2]},
            {"items": [3, 4]}
        ]

        result = reconstruct_context(history)

        assert result["items"] == [1, 2, 3, 4]

    def test_merge_nested_dicts(self):
        """Test that nested dicts are merged recursively."""
        from brains.cognitive.context_management.service.context_manager import reconstruct_context

        history = [
            {"config": {"a": 1}},
            {"config": {"b": 2}}
        ]

        result = reconstruct_context(history)

        assert result["config"]["a"] == 1
        assert result["config"]["b"] == 2

    def test_empty_history(self):
        """Test with empty history."""
        from brains.cognitive.context_management.service.context_manager import reconstruct_context

        result = reconstruct_context([])

        assert result == {}

    def test_skip_non_dict_items(self):
        """Test that non-dict items in history are skipped."""
        from brains.cognitive.context_management.service.context_manager import reconstruct_context

        history = [
            {"a": 1},
            "not a dict",
            {"b": 2}
        ]

        result = reconstruct_context(history)

        assert result["a"] == 1
        assert result["b"] == 2


class TestEnsurePatternObject:
    """Test suite for _ensure_pattern_object helper."""

    def test_handle_none(self):
        """Test handling None input."""
        from brains.cognitive.context_management.service.context_manager import _ensure_pattern_object

        result = _ensure_pattern_object(None)

        assert result is None

    def test_handle_pattern_object(self):
        """Test handling actual Pattern object."""
        from brains.cognitive.context_management.service.context_manager import _ensure_pattern_object
        from brains.cognitive.pattern_store import Pattern

        pattern = Pattern(
            brain="test",
            signature="test:sig",
            action={"key": "value"},
            score=0.8
        )

        result = _ensure_pattern_object(pattern)

        assert result is pattern  # Should return same object

    def test_handle_string(self):
        """Test converting string to Pattern."""
        from brains.cognitive.context_management.service.context_manager import _ensure_pattern_object
        from brains.cognitive.pattern_store import Pattern

        result = _ensure_pattern_object("some_signature")

        assert isinstance(result, Pattern)
        assert result.signature == "some_signature"
        assert result.brain == "context_management"

    def test_handle_dict(self):
        """Test converting dict to Pattern."""
        from brains.cognitive.context_management.service.context_manager import _ensure_pattern_object
        from brains.cognitive.pattern_store import Pattern

        result = _ensure_pattern_object({
            "signature": "dict_sig",
            "action": {"decay_factor": 0.8},
            "score": 0.7
        })

        assert isinstance(result, Pattern)
        assert result.signature == "dict_sig"


class TestServiceAPI:
    """Test suite for service API."""

    def test_apply_decay_operation(self):
        """Test APPLY_DECAY operation."""
        from brains.cognitive.context_management.service.context_manager import service_api

        result = service_api({
            "op": "APPLY_DECAY",
            "payload": {
                "context": {"value": 100},
                "decay": 0.5
            }
        })

        assert result["ok"] is True
        assert result["payload"]["context"]["value"] == 50

    def test_reconstruct_operation(self):
        """Test RECONSTRUCT operation."""
        from brains.cognitive.context_management.service.context_manager import service_api

        result = service_api({
            "op": "RECONSTRUCT",
            "payload": {
                "history": [
                    {"a": 1},
                    {"b": 2}
                ]
            }
        })

        assert result["ok"] is True
        assert result["payload"]["context"]["a"] == 1
        assert result["payload"]["context"]["b"] == 2

    def test_health_operation(self):
        """Test HEALTH operation."""
        from brains.cognitive.context_management.service.context_manager import service_api

        result = service_api({
            "op": "HEALTH",
            "payload": {}
        })

        assert result["ok"] is True
        assert "status" in result["payload"]

    def test_update_from_verdict_operation(self):
        """Test UPDATE_FROM_VERDICT operation."""
        from brains.cognitive.context_management.service.context_manager import service_api

        result = service_api({
            "op": "UPDATE_FROM_VERDICT",
            "payload": {
                "verdict": "ok",
                "metadata": {}
            }
        })

        assert result["ok"] is True

    def test_unsupported_operation(self):
        """Test unsupported operation."""
        from brains.cognitive.context_management.service.context_manager import service_api

        result = service_api({
            "op": "INVALID_OP",
            "payload": {}
        })

        assert result["ok"] is False
        assert result["error"]["code"] == "UNSUPPORTED_OP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
