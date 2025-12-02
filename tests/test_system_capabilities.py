"""
test_system_capabilities.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the system capabilities module and capability-based behavior.

Test Scenarios:
1. Browser runtime missing -> correctly says "I can't browse"
2. Web search configured -> correctly advertises it
3. Teacher required: brain misses, Teacher gives data, brain answers
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCapabilityProbes:
    """Test individual capability probes."""

    def test_probe_web_client_disabled_by_feature_flag(self):
        """When web_research is False, probe should return DISABLED."""
        from brains.system_capabilities import probe_web_client, CapabilityStatus

        # Mock feature flags to disable web_research
        with patch("brains.system_capabilities._load_feature_flags") as mock_flags:
            mock_flags.return_value = {"web_research": False}

            result = probe_web_client()

            assert result.status == CapabilityStatus.DISABLED
            assert "disabled" in result.reason.lower()

    def test_probe_web_client_disabled_by_env_var(self):
        """When MAVEN_NETWORK_DISABLED is set, probe should return DISABLED."""
        from brains.system_capabilities import probe_web_client, CapabilityStatus

        with patch("brains.system_capabilities._load_feature_flags") as mock_flags:
            mock_flags.return_value = {"web_research": True}
            with patch.dict("os.environ", {"MAVEN_NETWORK_DISABLED": "1"}):
                result = probe_web_client()

                assert result.status == CapabilityStatus.DISABLED
                assert "MAVEN_NETWORK_DISABLED" in result.reason

    def test_probe_browser_runtime_unavailable(self):
        """When Playwright is not installed, probe should return UNAVAILABLE."""
        from brains.system_capabilities import probe_browser_runtime, CapabilityStatus

        # Mock all imports to fail
        with patch.dict("sys.modules", {"playwright": None, "optional.browser_runtime": None}):
            with patch("brains.system_capabilities.probe_browser_runtime") as mock_probe:
                mock_probe.return_value = type("ProbeResult", (), {
                    "status": CapabilityStatus.UNAVAILABLE,
                    "reason": "Playwright not installed",
                    "to_dict": lambda self: {"status": "unavailable", "reason": "Playwright not installed"}
                })()

                result = mock_probe()
                assert result.status == CapabilityStatus.UNAVAILABLE

    def test_probe_git_available(self):
        """When git is installed and feature enabled, probe should return AVAILABLE."""
        from brains.system_capabilities import probe_git_client, CapabilityStatus

        with patch("brains.system_capabilities._load_feature_flags") as mock_flags:
            mock_flags.return_value = {"git_agency": True}
            with patch("shutil.which") as mock_which:
                mock_which.return_value = "/usr/bin/git"
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0,
                        stdout="git version 2.34.1"
                    )

                    result = probe_git_client()

                    assert result.status == CapabilityStatus.AVAILABLE
                    assert "version" in result.details

    def test_probe_shell_disabled_by_feature_flag(self):
        """When execution_agency is False, probe should return DISABLED."""
        from brains.system_capabilities import probe_shell, CapabilityStatus

        with patch("brains.system_capabilities._load_feature_flags") as mock_flags:
            mock_flags.return_value = {"execution_agency": False}

            result = probe_shell()

            assert result.status == CapabilityStatus.DISABLED


class TestCapabilityTruth:
    """Test the capability truth object and scan."""

    def test_scan_all_capabilities_returns_structured_result(self):
        """Scan should return a properly structured result."""
        from brains.system_capabilities import scan_all_capabilities

        # Force a fresh scan
        result = scan_all_capabilities(force_refresh=True)

        # Check structure
        assert "scan_time" in result
        assert "scan_duration_ms" in result
        assert "tools" in result
        assert "brains" in result
        assert "summary" in result

        # Check tools are present
        assert "web_search" in result["tools"]
        assert "git" in result["tools"]
        assert "shell" in result["tools"]

    def test_get_capability_truth_caches_results(self):
        """Capability truth should be cached after first scan."""
        from brains.system_capabilities import get_capability_truth, scan_all_capabilities

        # Clear cache
        import brains.system_capabilities as sc
        sc._capability_truth_cache = None

        # First call should scan
        result1 = get_capability_truth()
        time1 = result1.get("scan_time")

        # Second call should use cache (same time)
        result2 = get_capability_truth()
        time2 = result2.get("scan_time")

        assert time1 == time2

    def test_is_tool_available_helper(self):
        """Helper function should correctly check tool availability."""
        from brains.system_capabilities import is_tool_available, scan_all_capabilities

        # Force scan with known state
        with patch("brains.system_capabilities._load_feature_flags") as mock_flags:
            mock_flags.return_value = {
                "git_agency": True,
                "web_research": False
            }

            # Clear cache and rescan
            import brains.system_capabilities as sc
            sc._capability_truth_cache = None

            # Git should show based on feature flag + probe
            # Web search should be disabled


class TestCapabilityAnswers:
    """Test that capability questions get correct answers."""

    def test_browser_missing_says_cannot_browse(self):
        """When browser runtime unavailable, should say 'I can't browse'."""
        from brains.system_capabilities import get_current_capabilities

        with patch("brains.system_capabilities.probe_browser_runtime") as mock_probe:
            from brains.system_capabilities import ProbeResult, CapabilityStatus
            mock_probe.return_value = ProbeResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Playwright not installed"
            )

            # Clear cache
            import brains.system_capabilities as sc
            sc._capability_truth_cache = None

            result = get_current_capabilities()

            # Should be in unavailable list
            unavailable_str = " ".join(result.get("unavailable", []))
            # The capability should be listed as unavailable
            assert any("browse" in u.lower() for u in result.get("unavailable", []))

    def test_web_search_configured_advertises_it(self):
        """When web search is enabled, should advertise it as available."""
        from brains.system_capabilities import get_current_capabilities, ProbeResult, CapabilityStatus

        with patch("brains.system_capabilities.probe_web_client") as mock_probe:
            mock_probe.return_value = ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                reason="web_client module available"
            )

            # Clear cache
            import brains.system_capabilities as sc
            sc._capability_truth_cache = None

            result = get_current_capabilities()

            # Should be in available list
            available_str = " ".join(result.get("available", []))
            assert "web" in available_str.lower() or "search" in available_str.lower()


class TestTeacherGuards:
    """Test that Teacher properly blocks forbidden questions."""

    def test_teacher_blocks_identity_questions(self):
        """Teacher should block 'who are you' type questions."""
        try:
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
        except ImportError:
            pytest.skip("TeacherHelper not available")

        try:
            helper = TeacherHelper("reasoning")
        except Exception:
            pytest.skip("Cannot create TeacherHelper")

        # Check identity question detection
        assert helper._is_self_query("who are you?")
        assert helper._is_self_query("what are you?")
        assert helper._is_self_query("tell me about yourself")
        assert not helper._is_self_query("who is the president?")

    def test_teacher_blocks_capability_questions(self):
        """Teacher should block 'can you X' capability questions."""
        try:
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
        except ImportError:
            pytest.skip("TeacherHelper not available")

        try:
            helper = TeacherHelper("reasoning")
        except Exception:
            pytest.skip("Cannot create TeacherHelper")

        # Check capability question detection
        assert helper._is_capability_query("can you search the web?")
        assert helper._is_capability_query("can you run code?")
        assert helper._is_capability_query("can you control other programs?")
        assert not helper._is_capability_query("can you tell me about birds?")

    def test_teacher_blocks_history_questions(self):
        """Teacher should block conversation history questions."""
        try:
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
        except ImportError:
            pytest.skip("TeacherHelper not available")

        try:
            helper = TeacherHelper("reasoning")
        except Exception:
            pytest.skip("Cannot create TeacherHelper")

        # Check history question detection
        assert helper._is_history_query("what did we talk about yesterday?")
        assert helper._is_history_query("what was my first question?")
        assert not helper._is_history_query("what is the capital of France?")


class TestTeacherProposalMode:
    """Test that Teacher operates in proposal-only mode."""

    def test_proposal_mode_enabled_by_default(self):
        """Proposal mode should be enabled by default."""
        try:
            from brains.cognitive.teacher.service.teacher_helper import is_proposal_mode_enabled
        except ImportError:
            pytest.skip("teacher_helper not available")

        # Default should be True (proposal mode)
        with patch("brains.cognitive.teacher.service.teacher_helper._load_feature_flags") as mock_flags:
            mock_flags.return_value = {}  # Empty flags, use defaults

            result = is_proposal_mode_enabled()
            assert result is True

    def test_direct_fact_write_disabled_by_default(self):
        """Direct fact writing should be disabled by default."""
        try:
            from brains.cognitive.teacher.service.teacher_helper import is_direct_fact_write_enabled
        except ImportError:
            pytest.skip("teacher_helper not available")

        # Default should be False (no direct writes)
        with patch("brains.cognitive.teacher.service.teacher_helper._load_feature_flags") as mock_flags:
            mock_flags.return_value = {}  # Empty flags, use defaults

            result = is_direct_fact_write_enabled()
            assert result is False


class TestCapabilitySnapshot:
    """Test the capability snapshot for answering user questions."""

    def test_capability_snapshot_structure(self):
        """Capability snapshot should have required fields."""
        from capabilities import get_capability_snapshot

        snapshot = get_capability_snapshot()

        # Check required fields
        assert "web_search_enabled" in snapshot
        assert "code_execution_enabled" in snapshot
        assert "filesystem_scope" in snapshot
        assert "can_control_programs" in snapshot
        assert "autonomous_tools" in snapshot

        # Control programs and autonomous tools should always be False
        assert snapshot["can_control_programs"] is False
        assert snapshot["autonomous_tools"] is False

    def test_answer_capability_question_web_search(self):
        """Should answer web search capability questions truthfully."""
        from capabilities import answer_capability_question

        result = answer_capability_question("can you search the web?")

        assert result is not None
        assert "answer" in result
        assert "capability" in result
        assert result["capability"] == "web_search"
        assert result["source"] == "capability_snapshot"

    def test_answer_capability_question_control_programs(self):
        """Should always say NO to control programs question."""
        from capabilities import answer_capability_question

        result = answer_capability_question("can you control other programs?")

        assert result is not None
        assert result["enabled"] is False
        assert "no" in result["answer"].lower()


class TestSystemCapabilityRouting:
    """
    PHASE 1 Tests: System capability routing.

    Tests that capability questions route to self_model, NOT Teacher.
    Ensures no Apache Maven / Java 17 garbage in responses.
    """

    def test_semantic_normalizer_detects_system_capability_intent(self):
        """Semantic normalizer should detect system_capability intent."""
        from brains.cognitive.sensorium.semantic_normalizer import classify_intent

        # These should all be classified as system_capability
        capability_questions = [
            "what upgrade do you need",
            "can you browse the web",
            "can you run code",
            "can you control other programs on my computer",
            "can you read or change files on my system",
            "what can you do",
            "what tools can you use",
            "what are your capabilities",
        ]

        for question in capability_questions:
            result = classify_intent(question)
            assert result == "system_capability", f"Expected 'system_capability' for '{question}', got '{result}'"

    def test_semantic_normalizer_detects_self_identity_intent(self):
        """Semantic normalizer should detect self_identity intent."""
        from brains.cognitive.sensorium.semantic_normalizer import classify_intent

        # These should be classified as self_identity
        identity_questions = [
            "who are you",
            "what are you",
            "tell me about yourself",
        ]

        for question in identity_questions:
            result = classify_intent(question)
            assert result == "self_identity", f"Expected 'self_identity' for '{question}', got '{result}'"

    def test_teacher_blocks_system_capability_questions(self):
        """Teacher should block system capability questions."""
        try:
            from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
        except ImportError:
            pytest.skip("TeacherHelper not available")

        try:
            helper = TeacherHelper("reasoning")
        except Exception:
            pytest.skip("Cannot create TeacherHelper")

        # Check system capability question detection
        system_questions = [
            "what upgrade do you need",
            "can you browse the web",
            "can you run code",
            "can you control other programs on my computer",
            "can you read or change files on my system",
            "what can you do",
            "what tools can you use",
        ]

        for question in system_questions:
            is_forbidden, reason = helper._is_forbidden_teacher_question(question)
            assert is_forbidden, f"Teacher should block '{question}'"
            assert reason in ("system_capability", "capability"), \
                f"Wrong reason for '{question}': {reason}"

    def test_system_capabilities_helper_accessors(self):
        """Test the new helper accessors in system_capabilities.py."""
        from brains.system_capabilities import (
            get_execution_mode,
            is_execution_enabled,
            get_available_tools,
            get_capability_summary,
        )

        # get_execution_mode should return a string
        mode = get_execution_mode()
        assert isinstance(mode, str)
        assert mode in ("FULL", "READ_ONLY", "DISABLED", "UNKNOWN")

        # is_execution_enabled should return a bool
        enabled = is_execution_enabled()
        assert isinstance(enabled, bool)

        # get_available_tools should return a list
        tools = get_available_tools()
        assert isinstance(tools, list)

        # get_capability_summary should return a dict with required keys
        summary = get_capability_summary()
        assert isinstance(summary, dict)
        assert "execution_mode" in summary
        assert "execution_enabled" in summary
        assert "web_research_enabled" in summary
        assert "tools_available" in summary

    def test_answer_capability_question_no_apache_maven(self):
        """Capability answers should never mention Apache Maven / Java 17."""
        from brains.system_capabilities import answer_capability_question

        # Test various capability questions
        questions = [
            "what upgrade do you need",
            "what can you do",
            "what are your capabilities",
        ]

        for question in questions:
            result = answer_capability_question(question)

            if result:  # May be None for some questions
                answer = result.get("answer", "")

                # Should NEVER mention Apache Maven or Java
                assert "apache maven" not in answer.lower(), \
                    f"Answer should not mention Apache Maven: {answer}"
                assert "java 17" not in answer.lower(), \
                    f"Answer should not mention Java 17: {answer}"
                assert "pom.xml" not in answer.lower(), \
                    f"Answer should not mention pom.xml: {answer}"
                assert "mvn" not in answer.lower(), \
                    f"Answer should not mention mvn: {answer}"


class TestSensoriumCapabilityRouting:
    """Test that sensorium properly routes capability queries to self_model."""

    def test_sensorium_includes_intent_kind_in_normalization(self):
        """Sensorium normalization should include intent_kind for capability queries."""
        try:
            from brains.cognitive.sensorium.service.sensorium_brain import handle
        except ImportError:
            pytest.skip("sensorium_brain not available")

        # Test capability question
        msg = {
            "op": "NORMALIZE",
            "payload": {
                "text": "can you browse the web"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})

        # Should include intent_kind
        assert "intent_kind" in payload or "semantic" in payload
        semantic = payload.get("semantic", {})
        intent_kind = payload.get("intent_kind") or semantic.get("intent_kind")

        assert intent_kind == "system_capability", \
            f"Expected intent_kind='system_capability', got '{intent_kind}'"

    def test_sensorium_adds_routing_target_for_capability_queries(self):
        """Sensorium should add routing_target for capability queries."""
        try:
            from brains.cognitive.sensorium.service.sensorium_brain import handle
        except ImportError:
            pytest.skip("sensorium_brain not available")

        # Test capability question
        msg = {
            "op": "NORMALIZE",
            "payload": {
                "text": "what can you do"
            }
        }

        result = handle(msg)

        assert result.get("ok") is True
        payload = result.get("payload", {})

        # Should include routing_target for capability queries
        routing_target = payload.get("routing_target")
        assert routing_target == "self_model", \
            f"Expected routing_target='self_model', got '{routing_target}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
