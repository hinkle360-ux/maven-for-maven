"""
system_capabilities.py
~~~~~~~~~~~~~~~~~~~~~~

Centralized capability registry with runtime probes.

This module provides:
1. Runtime capability probes that test actual functionality
2. A single truth object for "what can Maven do right now?"
3. Clear separation between "available" (probed) vs "planned" (documented)

CRITICAL: All "can you X" capability questions MUST use this module.
Never use Teacher/LLM to answer capability questions - that leads to
hallucination. This module reads from actual probes and runtime state.

Usage:
    from brains.system_capabilities import (
        scan_all_capabilities,
        get_capability_truth,
        CapabilityStatus,
        probe_web_client,
        probe_browser_runtime,
        probe_git_client,
        probe_shell,
    )

    # At startup
    truth = scan_all_capabilities()

    # Answer "can you X" questions
    if get_capability_truth()["tools"]["web_search"] == "available":
        # Actually available
        pass
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List


class CapabilityStatus(str, Enum):
    """Status of a capability probe."""
    AVAILABLE = "available"        # Probe succeeded, capability works
    UNAVAILABLE = "unavailable"    # Probe failed or not installed
    UNCONFIGURED = "unconfigured"  # Module exists but not configured
    MISCONFIGURED = "misconfigured"  # Configuration error
    DISABLED = "disabled"          # Explicitly disabled by config


@dataclass
class ProbeResult:
    """Result of a capability probe."""
    status: CapabilityStatus
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    probe_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "details": self.details,
            "probe_time_ms": self.probe_time_ms
        }


# Cached capability truth (computed once at startup, refreshed on demand)
_capability_truth_cache: Optional[Dict[str, Any]] = None


# =============================================================================
# TOOL PROBES
# =============================================================================

def probe_web_client() -> ProbeResult:
    """
    Probe web client capability.

    Checks:
    1. web_research feature flag is enabled
    2. network access is not disabled
    3. web_search_tool or web_client module is importable
    """
    import time
    start = time.time()

    try:
        # Check feature flag
        feature_flags = _load_feature_flags()
        if not feature_flags.get("web_research", False):
            return ProbeResult(
                status=CapabilityStatus.DISABLED,
                reason="web_research feature disabled in config/features.json",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Check network override
        network_disabled = os.getenv("MAVEN_NETWORK_DISABLED", "").lower() in ("1", "true", "yes")
        if network_disabled:
            return ProbeResult(
                status=CapabilityStatus.DISABLED,
                reason="MAVEN_NETWORK_DISABLED environment variable set",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Check new service_api-based web_search_tool first
        try:
            from brains.agent.tools.web_search_tool import service_api, is_available
            result = service_api({"op": "HEALTH"})
            if result.get("ok"):
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="web_search_tool service_api available",
                    details={"module": "brains.agent.tools.web_search_tool"},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except ImportError:
            pass

        # Fallback: Check legacy host_tools.web_client
        try:
            from host_tools.web_client import web_client
            if hasattr(web_client, "fetch") or hasattr(web_client, "get"):
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="web_client module available and configured",
                    details={"module": "host_tools.web_client"},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
            else:
                return ProbeResult(
                    status=CapabilityStatus.MISCONFIGURED,
                    reason="web_client module missing required methods",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except ImportError:
            return ProbeResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="web_search module not found",
                probe_time_ms=int((time.time() - start) * 1000)
            )

    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def probe_browser_runtime() -> ProbeResult:
    """
    Probe browser runtime (Playwright) capability.

    Checks:
    1. browser_runtime module is importable
    2. Playwright is installed
    3. Browser server is running (if applicable)
    """
    import time
    start = time.time()

    try:
        # Check if browser runtime module exists
        try:
            from optional.browser_runtime import browser_client
            if hasattr(browser_client, "is_available"):
                available = browser_client.is_available()
                if available:
                    return ProbeResult(
                        status=CapabilityStatus.AVAILABLE,
                        reason="browser runtime available and running",
                        details={"module": "optional.browser_runtime"},
                        probe_time_ms=int((time.time() - start) * 1000)
                    )
                else:
                    return ProbeResult(
                        status=CapabilityStatus.UNCONFIGURED,
                        reason="browser runtime module present but not running",
                        details={"module": "optional.browser_runtime"},
                        probe_time_ms=int((time.time() - start) * 1000)
                    )
            else:
                return ProbeResult(
                    status=CapabilityStatus.MISCONFIGURED,
                    reason="browser_client missing is_available method",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except ImportError:
            pass

        # Check if Playwright is installed at all
        try:
            import playwright
            return ProbeResult(
                status=CapabilityStatus.UNCONFIGURED,
                reason="Playwright installed but browser_runtime module not available",
                details={"playwright_installed": True},
                probe_time_ms=int((time.time() - start) * 1000)
            )
        except ImportError:
            return ProbeResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Playwright not installed",
                probe_time_ms=int((time.time() - start) * 1000)
            )

    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def probe_git_client() -> ProbeResult:
    """
    Probe git capability.

    Checks:
    1. git_agency feature flag is enabled
    2. git command is available in PATH
    3. We're in a git repository (optional)
    """
    import time
    start = time.time()

    try:
        # Check feature flag
        feature_flags = _load_feature_flags()
        if not feature_flags.get("git_agency", False):
            return ProbeResult(
                status=CapabilityStatus.DISABLED,
                reason="git_agency feature disabled in config/features.json",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Check git command exists
        git_path = shutil.which("git")
        if not git_path:
            return ProbeResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="git command not found in PATH",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Try to get git version
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="git available",
                    details={"version": version, "path": git_path},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
            else:
                return ProbeResult(
                    status=CapabilityStatus.MISCONFIGURED,
                    reason=f"git command failed: {result.stderr[:100]}",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except subprocess.TimeoutExpired:
            return ProbeResult(
                status=CapabilityStatus.MISCONFIGURED,
                reason="git command timed out",
                probe_time_ms=int((time.time() - start) * 1000)
            )

    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def probe_shell() -> ProbeResult:
    """
    Probe shell/execution capability.

    Checks:
    1. execution_agency feature flag is enabled
    2. execution mode is not DISABLED
    3. Shell tool service_api is available
    """
    import time
    start = time.time()

    try:
        # Check feature flag
        feature_flags = _load_feature_flags()
        if not feature_flags.get("execution_agency", False):
            return ProbeResult(
                status=CapabilityStatus.DISABLED,
                reason="execution_agency feature disabled in config/features.json",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Check execution mode
        try:
            from brains.tools.execution_guard import get_execution_status, ExecMode
            exec_status = get_execution_status()
            if exec_status.mode == ExecMode.DISABLED:
                return ProbeResult(
                    status=CapabilityStatus.DISABLED,
                    reason=f"execution mode is DISABLED: {exec_status.reason}",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
            elif not exec_status.effective:
                return ProbeResult(
                    status=CapabilityStatus.DISABLED,
                    reason=f"execution not effective: {exec_status.reason}",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except ImportError:
            pass

        # Check service_api based shell tool
        try:
            from brains.agent.tools.shell_tool import service_api
            result = service_api({"op": "HEALTH"})
            if result.get("ok"):
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="shell_tool service_api available",
                    details={"module": "brains.agent.tools.shell_tool"},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except ImportError:
            pass

        # Fallback: Try a simple shell command
        try:
            result = subprocess.run(
                ["echo", "probe_test"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="shell execution available",
                    details={"shell_test": "passed"},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
            else:
                return ProbeResult(
                    status=CapabilityStatus.MISCONFIGURED,
                    reason=f"shell test failed: {result.stderr[:100]}",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except subprocess.TimeoutExpired:
            return ProbeResult(
                status=CapabilityStatus.MISCONFIGURED,
                reason="shell command timed out",
                probe_time_ms=int((time.time() - start) * 1000)
            )

    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def probe_filesystem() -> ProbeResult:
    """
    Probe filesystem access capability.

    Checks:
    1. filesystem_agency feature flag is enabled
    2. Execution mode allows file access
    3. filesystem_agency service_api is available
    """
    import time
    start = time.time()

    try:
        # Check feature flag
        feature_flags = _load_feature_flags()
        if not feature_flags.get("filesystem_agency", False):
            return ProbeResult(
                status=CapabilityStatus.DISABLED,
                reason="filesystem_agency feature disabled in config/features.json",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Check service_api based filesystem_agency first
        try:
            from brains.tools.filesystem_agency import service_api
            result = service_api({"op": "HEALTH"})
            if result.get("ok"):
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="filesystem_agency service_api available",
                    details={"module": "brains.tools.filesystem_agency", "scope": "read_write"},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        except ImportError:
            pass

        # Fallback: Check execution mode
        try:
            from brains.tools.execution_guard import get_execution_status, ExecMode
            exec_status = get_execution_status()
            if exec_status.mode == ExecMode.DISABLED:
                return ProbeResult(
                    status=CapabilityStatus.DISABLED,
                    reason="execution mode DISABLED",
                    probe_time_ms=int((time.time() - start) * 1000)
                )

            # Determine scope based on mode
            if exec_status.mode == ExecMode.FULL:
                scope = "read_write"
            else:
                scope = "read_only"

            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                reason=f"filesystem access available ({scope})",
                details={"scope": scope, "mode": exec_status.mode.value},
                probe_time_ms=int((time.time() - start) * 1000)
            )
        except ImportError:
            # No execution guard, assume full access
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                reason="filesystem access available (no execution guard)",
                details={"scope": "read_write"},
                probe_time_ms=int((time.time() - start) * 1000)
            )

    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def probe_llm_service() -> ProbeResult:
    """
    Probe LLM service capability.

    Checks:
    1. LLM service module is importable
    2. LLM service is enabled and configured
    """
    import time
    start = time.time()

    try:
        from brains.tools.llm_service import llm_service
        if llm_service and hasattr(llm_service, "enabled"):
            if llm_service.enabled:
                return ProbeResult(
                    status=CapabilityStatus.AVAILABLE,
                    reason="LLM service available and enabled",
                    details={"module": "brains.tools.llm_service"},
                    probe_time_ms=int((time.time() - start) * 1000)
                )
            else:
                return ProbeResult(
                    status=CapabilityStatus.DISABLED,
                    reason="LLM service disabled",
                    probe_time_ms=int((time.time() - start) * 1000)
                )
        else:
            return ProbeResult(
                status=CapabilityStatus.MISCONFIGURED,
                reason="LLM service not properly initialized",
                probe_time_ms=int((time.time() - start) * 1000)
            )
    except ImportError:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason="llm_service module not found",
            probe_time_ms=int((time.time() - start) * 1000)
        )
    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


# =============================================================================
# BRAIN PROBES
# =============================================================================

def probe_brain(brain_name: str) -> ProbeResult:
    """
    Probe if a cognitive brain is available.

    Args:
        brain_name: Name of the brain to probe (e.g., "reasoning", "self_model")
    """
    import time
    start = time.time()

    brain_paths = {
        "reasoning": "brains.cognitive.reasoning.service.reasoning_brain",
        "self_model": "brains.cognitive.self_model.service.self_model_brain",
        "teacher": "brains.cognitive.teacher.service.teacher_brain",
        "memory_librarian": "brains.cognitive.memory_librarian.service.memory_librarian",
        "language": "brains.cognitive.language.service.language_brain",
        "integrator": "brains.cognitive.integrator.service.integrator_brain",
        "browser_brain": "brains.cognitive.browser.service.browser_brain",
    }

    module_path = brain_paths.get(brain_name)
    if not module_path:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"unknown brain: {brain_name}",
            probe_time_ms=int((time.time() - start) * 1000)
        )

    try:
        import importlib
        module = importlib.import_module(module_path)
        if hasattr(module, "handle") or hasattr(module, "service_api"):
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                reason=f"{brain_name} brain loaded",
                details={"module": module_path},
                probe_time_ms=int((time.time() - start) * 1000)
            )
        else:
            return ProbeResult(
                status=CapabilityStatus.MISCONFIGURED,
                reason=f"{brain_name} brain missing handle/service_api",
                probe_time_ms=int((time.time() - start) * 1000)
            )
    except ImportError as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"{brain_name} brain import failed: {str(e)[:50]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )
    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


# =============================================================================
# CAPABILITY TRUTH OBJECT
# =============================================================================

def scan_all_capabilities(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Run all capability probes and return the single truth object.

    This is the main function to call at startup. Results are cached
    and reused unless force_refresh=True.

    Returns:
        Dict with structure:
        {
            "scan_time": "2024-01-15T10:30:00",
            "scan_duration_ms": 123,
            "tools": {
                "web_search": "available",
                "browser_runtime": "unavailable",
                "git": "available",
                "shell": "available",
                "filesystem": "available",
                "llm_service": "available"
            },
            "brains": {
                "reasoning": "available",
                "self_model": "available",
                "teacher": "available",
                "browser_brain": "unavailable"
            },
            "summary": "5/6 tools, 3/4 brains available"
        }
    """
    global _capability_truth_cache

    if _capability_truth_cache is not None and not force_refresh:
        return _capability_truth_cache

    import time
    scan_start = time.time()

    # Probe all tools
    tool_probes = {
        "web_search": probe_web_client(),
        "browser_runtime": probe_browser_runtime(),
        "git": probe_git_client(),
        "shell": probe_shell(),
        "filesystem": probe_filesystem(),
        "llm_service": probe_llm_service(),
    }

    # Probe core brains
    brain_probes = {
        "reasoning": probe_brain("reasoning"),
        "self_model": probe_brain("self_model"),
        "teacher": probe_brain("teacher"),
        "browser_brain": probe_brain("browser_brain"),
    }

    scan_duration = time.time() - scan_start

    # Build simplified status maps
    tools = {name: probe.status.value for name, probe in tool_probes.items()}
    brains = {name: probe.status.value for name, probe in brain_probes.items()}

    # Count available
    tools_available = sum(1 for s in tools.values() if s == "available")
    brains_available = sum(1 for s in brains.values() if s == "available")

    result = {
        "scan_time": datetime.now().isoformat(),
        "scan_duration_ms": int(scan_duration * 1000),
        "tools": tools,
        "brains": brains,
        "tool_details": {name: probe.to_dict() for name, probe in tool_probes.items()},
        "brain_details": {name: probe.to_dict() for name, probe in brain_probes.items()},
        "summary": f"{tools_available}/{len(tools)} tools, {brains_available}/{len(brains)} brains available"
    }

    _capability_truth_cache = result
    print(f"[CAPABILITY_SCAN] {result['summary']} in {result['scan_duration_ms']}ms")

    return result


def get_capability_truth() -> Dict[str, Any]:
    """
    Get the cached capability truth object.

    If no scan has been run yet, runs one automatically.
    """
    global _capability_truth_cache
    if _capability_truth_cache is None:
        return scan_all_capabilities()
    return _capability_truth_cache


def is_tool_available(tool_name: str) -> bool:
    """Check if a specific tool is available."""
    truth = get_capability_truth()
    return truth.get("tools", {}).get(tool_name) == "available"


def is_brain_available(brain_name: str) -> bool:
    """Check if a specific brain is available."""
    truth = get_capability_truth()
    return truth.get("brains", {}).get(brain_name) == "available"


# =============================================================================
# CURRENT VS PLANNED CAPABILITIES
# =============================================================================

def get_current_capabilities() -> Dict[str, Any]:
    """
    Get capabilities that are ACTUALLY available right now.

    This reads from the capability probes, not from docs or specs.
    Use this for answering "can you X?" questions.
    """
    truth = get_capability_truth()

    # Build human-readable capability list
    available = []
    unavailable = []

    for tool, status in truth.get("tools", {}).items():
        human_name = {
            "web_search": "search the web",
            "browser_runtime": "browse websites visually",
            "git": "work with git repositories",
            "shell": "run shell commands",
            "filesystem": "read/write files",
            "llm_service": "use language model reasoning",
            "pc": "control your PC (monitor system, clean files, manage processes)",
            "human": "control desktop with mouse and keyboard",
        }.get(tool, tool)

        if status == "available":
            available.append(human_name)
        else:
            reason = truth.get("tool_details", {}).get(tool, {}).get("reason", status)
            unavailable.append(f"{human_name} ({reason})")

    return {
        "available": available,
        "unavailable": unavailable,
        "raw": truth
    }


def get_planned_capabilities() -> Dict[str, Any]:
    """
    Get capabilities that are PLANNED but not yet active.

    This reads from docs/specs, clearly labeled as "planned, not active".
    Use this for answering "what will you be able to do?" questions.
    """
    # Load from specs domain bank or docs
    specs_path = Path(__file__).resolve().parent / "domain_banks" / "specs" / "maven_design.md"

    planned = []
    if specs_path.exists():
        try:
            content = specs_path.read_text(encoding="utf-8")
            # Extract planned features (simplified parsing)
            if "## Planned Features" in content or "## Future" in content:
                planned.append("autonomous web research agents")
                planned.append("multi-modal image understanding")
                planned.append("voice interaction")
        except Exception:
            pass

    return {
        "planned": planned,
        "source": "docs/specs",
        "warning": "These are PLANNED capabilities, not currently active"
    }


# =============================================================================
# INTROSPECTION COMMAND
# =============================================================================

def print_capability_dump() -> str:
    """
    Print a human-readable capability dump.

    Can be used as a /capabilities command or startup log.
    """
    truth = scan_all_capabilities(force_refresh=True)

    lines = []
    lines.append("=" * 60)
    lines.append("MAVEN CAPABILITY SCAN")
    lines.append(f"Time: {truth['scan_time']}")
    lines.append(f"Duration: {truth['scan_duration_ms']}ms")
    lines.append("=" * 60)

    lines.append("\nTOOLS:")
    for tool, status in truth.get("tools", {}).items():
        icon = "+" if status == "available" else "-"
        reason = truth.get("tool_details", {}).get(tool, {}).get("reason", "")
        lines.append(f"  [{icon}] {tool}: {status}")
        if reason and status != "available":
            lines.append(f"      Reason: {reason}")

    lines.append("\nBRAINS:")
    for brain, status in truth.get("brains", {}).items():
        icon = "+" if status == "available" else "-"
        lines.append(f"  [{icon}] {brain}: {status}")

    lines.append(f"\nSUMMARY: {truth['summary']}")
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)
    return output


# =============================================================================
# HELPERS
# =============================================================================

def _load_feature_flags() -> Dict[str, bool]:
    """Load static feature flags from the configuration file."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "features.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: bool(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


# =============================================================================
# HELPER ACCESSORS
# =============================================================================
# These provide easy access to common capability queries without
# needing to parse the full capability truth object.

def get_execution_mode() -> str:
    """
    Get the current execution mode.

    Returns:
        One of: "FULL", "FULL_AGENCY", "READ_ONLY", "DISABLED", "UNKNOWN"
    """
    try:
        from brains.tools.execution_guard import get_execution_status, ExecMode
        status = get_execution_status()
        return status.mode.value if hasattr(status, 'mode') else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def is_execution_enabled() -> bool:
    """Check if any form of execution is currently enabled."""
    mode = get_execution_mode()
    return mode in ("FULL", "FULL_AGENCY", "READ_ONLY")


def is_full_agency_mode() -> bool:
    """Check if the system is in FULL_AGENCY mode (unrestricted access)."""
    return get_execution_mode() == "FULL_AGENCY"


def is_web_research_enabled() -> bool:
    """
    Check if web research capability is enabled.

    This checks both the feature flag and the actual module availability.
    """
    truth = get_capability_truth()
    web_status = truth.get("tools", {}).get("web_search", "unavailable")
    return web_status == "available"


def is_browser_runtime_configured() -> bool:
    """
    Check if browser runtime (Playwright) is available and configured.

    This is different from web_research - browser_runtime means visual browsing
    capability with screenshot/interaction support.
    """
    truth = get_capability_truth()
    browser_status = truth.get("tools", {}).get("browser_runtime", "unavailable")
    return browser_status == "available"


def get_available_tools() -> List[str]:
    """
    Get list of currently available tools.

    Returns:
        List of tool names that are currently available.
    """
    truth = get_capability_truth()
    return [name for name, status in truth.get("tools", {}).items() if status == "available"]


def get_unavailable_tools() -> Dict[str, str]:
    """
    Get dict of unavailable tools with their reasons.

    Returns:
        Dict mapping tool name -> reason string
    """
    truth = get_capability_truth()
    result = {}
    for name, status in truth.get("tools", {}).items():
        if status != "available":
            reason = truth.get("tool_details", {}).get(name, {}).get("reason", status)
            result[name] = reason
    return result


def get_capability_summary() -> Dict[str, Any]:
    """
    Get a simplified capability summary for quick access.

    Returns:
        Dict with:
        - execution_mode: current execution mode
        - execution_enabled: bool
        - full_agency_mode: bool (True if FULL_AGENCY mode)
        - web_research_enabled: bool
        - browser_runtime_configured: bool
        - tools_available: list of available tool names
        - tools_unavailable: dict of unavailable tools with reasons
    """
    return {
        "execution_mode": get_execution_mode(),
        "execution_enabled": is_execution_enabled(),
        "full_agency_mode": is_full_agency_mode(),
        "web_research_enabled": is_web_research_enabled(),
        "browser_runtime_configured": is_browser_runtime_configured(),
        "tools_available": get_available_tools(),
        "tools_unavailable": get_unavailable_tools(),
    }


def answer_capability_question(question: str) -> Optional[Dict[str, Any]]:
    """
    Answer a specific capability question.

    This is a convenience function that maps common capability questions
    to truthful answers based on runtime probes.

    Args:
        question: The user's capability question

    Returns:
        Dict with 'answer', 'enabled', 'capability', 'source' or None if not a capability question
    """
    ql = question.lower().strip()
    truth = get_capability_truth()

    # Web search / browsing
    if any(word in ql for word in ["browse", "web", "internet", "search online"]):
        web_available = is_web_research_enabled()
        browser_available = is_browser_runtime_configured()

        if web_available and browser_available:
            answer = "Yes, I can browse the web. Both web search and visual browsing are available."
        elif web_available:
            answer = "I can search the web (API-based), but visual browser automation is not configured."
        else:
            reason = truth.get("tool_details", {}).get("web_search", {}).get("reason", "disabled")
            answer = f"No, web access is not currently available. Reason: {reason}"

        return {
            "answer": answer,
            "enabled": web_available,
            "capability": "web_search",
            "source": "capability_snapshot",
        }

    # Code execution
    if any(word in ql for word in ["run code", "execute", "run script", "shell"]):
        shell_available = is_tool_available("shell")
        mode = get_execution_mode()

        if shell_available:
            answer = f"Yes, I can execute code. Execution mode: {mode}."
        else:
            reason = truth.get("tool_details", {}).get("shell", {}).get("reason", "disabled")
            answer = f"No, code execution is not available. Reason: {reason}"

        return {
            "answer": answer,
            "enabled": shell_available,
            "capability": "code_execution",
            "source": "capability_snapshot",
        }

    # File access
    if any(word in ql for word in ["read file", "write file", "access file", "file system"]):
        fs_available = is_tool_available("filesystem")
        mode = get_execution_mode()

        if fs_available:
            scope = "read/write" if mode == "FULL" else "read-only" if mode == "READ_ONLY" else "limited"
            answer = f"Yes, I can access files with {scope} permissions."
        else:
            reason = truth.get("tool_details", {}).get("filesystem", {}).get("reason", "disabled")
            answer = f"No, file access is not available. Reason: {reason}"

        return {
            "answer": answer,
            "enabled": fs_available,
            "capability": "filesystem",
            "source": "capability_snapshot",
        }

    # Control other programs
    if any(word in ql for word in ["control program", "control other", "autonomous"]):
        answer = "No, I cannot autonomously control other programs. I operate through a structured pipeline and require explicit user instructions for actions."
        return {
            "answer": answer,
            "enabled": False,
            "capability": "program_control",
            "source": "capability_snapshot",
        }

    # General capabilities question
    if any(word in ql for word in ["what can you", "capabilities", "what do you"]):
        available = get_available_tools()
        answer = f"I currently have access to: {', '.join(available) if available else 'no tools (execution disabled)'}."
        return {
            "answer": answer,
            "enabled": bool(available),
            "capability": "general",
            "source": "capability_snapshot",
        }

    return None


# =============================================================================
# ENHANCED CAPABILITY SNAPSHOT
# =============================================================================

@dataclass
class CapabilitySnapshot:
    """
    Complete snapshot of Maven's current capabilities.

    This is built at startup and provides truthful answers to capability questions.
    """
    # Execution capabilities
    can_read_files: bool = False
    can_write_files: bool = False
    can_run_shell: bool = False
    execution_mode: str = "UNKNOWN"

    # Web capabilities
    can_browse_web: bool = False
    browser_runtime_available: bool = False

    # Memory capabilities
    vector_index_enabled: bool = False
    episodic_memory_enabled: bool = True
    cross_session_memory: bool = True

    # Tool capabilities
    git_available: bool = False
    llm_service_available: bool = False

    # Brain capabilities
    available_brains: List[str] = field(default_factory=list)

    # Metadata
    scan_time: str = ""
    scan_duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_read_files": self.can_read_files,
            "can_write_files": self.can_write_files,
            "can_run_shell": self.can_run_shell,
            "execution_mode": self.execution_mode,
            "can_browse_web": self.can_browse_web,
            "browser_runtime_available": self.browser_runtime_available,
            "vector_index_enabled": self.vector_index_enabled,
            "episodic_memory_enabled": self.episodic_memory_enabled,
            "cross_session_memory": self.cross_session_memory,
            "git_available": self.git_available,
            "llm_service_available": self.llm_service_available,
            "available_brains": self.available_brains,
            "scan_time": self.scan_time,
            "scan_duration_ms": self.scan_duration_ms,
        }


# Cached capability snapshot
_capability_snapshot: Optional[CapabilitySnapshot] = None


def probe_vector_index() -> ProbeResult:
    """
    Probe vector index capability.

    Checks if the vector index is enabled and functional.
    """
    import time
    start = time.time()

    try:
        from brains.memory.vector_index import (
            is_vector_index_enabled,
            get_vector_index,
        )

        if not is_vector_index_enabled():
            return ProbeResult(
                status=CapabilityStatus.DISABLED,
                reason="Vector index is disabled",
                probe_time_ms=int((time.time() - start) * 1000)
            )

        # Try to get the index
        index = get_vector_index()
        stats = index.get_stats()

        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            reason="Vector index available",
            details=stats,
            probe_time_ms=int((time.time() - start) * 1000)
        )

    except ImportError:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason="Vector index module not found",
            probe_time_ms=int((time.time() - start) * 1000)
        )
    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def probe_episodic_memory() -> ProbeResult:
    """
    Probe episodic memory capability.

    Checks if enhanced episodic memory is available.
    """
    import time
    start = time.time()

    try:
        from brains.memory.enhanced_episodic import get_episode_manager

        manager = get_episode_manager()
        # Basic functionality check
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            reason="Enhanced episodic memory available",
            details={"manager": "EpisodeManager"},
            probe_time_ms=int((time.time() - start) * 1000)
        )

    except ImportError:
        # Try basic episodic memory
        try:
            from brains.memory.episodic_memory import get_episodes
            return ProbeResult(
                status=CapabilityStatus.AVAILABLE,
                reason="Basic episodic memory available",
                details={"manager": "basic"},
                probe_time_ms=int((time.time() - start) * 1000)
            )
        except ImportError:
            return ProbeResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Episodic memory module not found",
                probe_time_ms=int((time.time() - start) * 1000)
            )
    except Exception as e:
        return ProbeResult(
            status=CapabilityStatus.UNAVAILABLE,
            reason=f"probe error: {str(e)[:100]}",
            probe_time_ms=int((time.time() - start) * 1000)
        )


def self_scan() -> CapabilitySnapshot:
    """
    Run a comprehensive self-scan of all capabilities.

    This should be called at startup to build the capability snapshot.

    Returns:
        CapabilitySnapshot with all current capabilities
    """
    global _capability_snapshot

    import time
    start = time.time()

    # Run all probes
    truth = scan_all_capabilities(force_refresh=True)

    # Build snapshot
    snapshot = CapabilitySnapshot()
    snapshot.scan_time = truth.get("scan_time", datetime.now().isoformat())

    # Execution capabilities
    exec_mode = get_execution_mode()
    snapshot.execution_mode = exec_mode
    snapshot.can_run_shell = is_tool_available("shell")
    snapshot.can_read_files = is_tool_available("filesystem")
    snapshot.can_write_files = exec_mode in ("FULL", "FULL_AGENCY")

    # Web capabilities
    snapshot.can_browse_web = is_web_research_enabled()
    snapshot.browser_runtime_available = is_browser_runtime_configured()

    # Git and LLM
    snapshot.git_available = is_tool_available("git")
    snapshot.llm_service_available = is_tool_available("llm_service")

    # Available brains
    snapshot.available_brains = [
        name for name, status in truth.get("brains", {}).items()
        if status == "available"
    ]

    # Memory capabilities
    vector_probe = probe_vector_index()
    snapshot.vector_index_enabled = vector_probe.status == CapabilityStatus.AVAILABLE

    episodic_probe = probe_episodic_memory()
    snapshot.episodic_memory_enabled = episodic_probe.status == CapabilityStatus.AVAILABLE

    # Cross-session memory is enabled if we have persistent storage
    snapshot.cross_session_memory = snapshot.episodic_memory_enabled

    # Timing
    snapshot.scan_duration_ms = int((time.time() - start) * 1000)

    # Cache the snapshot
    _capability_snapshot = snapshot

    print(f"[SELF_SCAN] Completed in {snapshot.scan_duration_ms}ms")
    print(f"[SELF_SCAN] Execution: {snapshot.execution_mode}, "
          f"Files: r={snapshot.can_read_files}/w={snapshot.can_write_files}, "
          f"Web: {snapshot.can_browse_web}, "
          f"Vector: {snapshot.vector_index_enabled}")

    return snapshot


def get_capability_snapshot() -> CapabilitySnapshot:
    """
    Get the cached capability snapshot.

    If no scan has been run, runs one automatically.
    """
    global _capability_snapshot
    if _capability_snapshot is None:
        return self_scan()
    return _capability_snapshot


def can_remember_across_sessions() -> bool:
    """Check if Maven can remember across sessions."""
    snapshot = get_capability_snapshot()
    return snapshot.cross_session_memory and snapshot.episodic_memory_enabled


def answer_memory_question(question: str) -> Optional[Dict[str, Any]]:
    """
    Answer a memory-related question truthfully.

    Args:
        question: The user's question about memory

    Returns:
        Dict with answer, or None if not a memory question
    """
    ql = question.lower().strip()

    # Questions about remembering
    if any(phrase in ql for phrase in [
        "do you remember",
        "can you remember",
        "will you remember",
        "remember me",
        "remember this",
        "remember what",
    ]):
        snapshot = get_capability_snapshot()

        if snapshot.cross_session_memory:
            answer = ("Yes, I can remember information across sessions. "
                     "I use episodic memory to store our conversations.")
        else:
            answer = ("I have session-based memory, but I may not remember "
                     "everything across sessions without persistent storage.")

        return {
            "answer": answer,
            "enabled": snapshot.cross_session_memory,
            "capability": "memory",
            "source": "capability_snapshot",
            "details": {
                "episodic_enabled": snapshot.episodic_memory_enabled,
                "vector_enabled": snapshot.vector_index_enabled,
            }
        }

    # Questions about yesterday/past sessions
    if any(phrase in ql for phrase in [
        "yesterday",
        "last session",
        "last time",
        "previous conversation",
    ]):
        snapshot = get_capability_snapshot()

        if snapshot.episodic_memory_enabled:
            return {
                "answer": "Let me check my episodic memory for that...",
                "enabled": True,
                "capability": "episodic_memory",
                "source": "capability_snapshot",
                "should_query_episodic": True,
            }
        else:
            return {
                "answer": "I don't have persistent memory of previous sessions enabled.",
                "enabled": False,
                "capability": "episodic_memory",
                "source": "capability_snapshot",
            }

    return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CapabilityStatus",
    "ProbeResult",
    "CapabilitySnapshot",
    "scan_all_capabilities",
    "get_capability_truth",
    "is_tool_available",
    "is_brain_available",
    "get_current_capabilities",
    "get_planned_capabilities",
    "print_capability_dump",
    "probe_web_client",
    "probe_browser_runtime",
    "probe_git_client",
    "probe_shell",
    "probe_filesystem",
    "probe_llm_service",
    "probe_brain",
    "probe_vector_index",
    "probe_episodic_memory",
    # Startup scan
    "self_scan",
    "get_capability_snapshot",
    "can_remember_across_sessions",
    "answer_memory_question",
    # Helper accessors
    "get_execution_mode",
    "is_execution_enabled",
    "is_full_agency_mode",
    "is_web_research_enabled",
    "is_browser_runtime_configured",
    "get_available_tools",
    "get_unavailable_tools",
    "get_capability_summary",
    "answer_capability_question",
]
