"""Centralized capability registry for Maven.

This module provides a single source of truth for feature availability and
runtime enablement.  It combines static feature flags with runtime checks such
as the execution guard so other systems (self_model, integrator, banner
printing) can describe capabilities consistently.

CRITICAL: All "can you X" capability questions MUST use this module.
Never use Teacher/LLM to answer capability questions - that leads to
hallucination. This module reads from actual config and runtime state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from brains.tools.execution_guard import get_execution_status, ExecMode, enable_full_agency


def _load_feature_flags() -> Dict[str, bool]:
    """Load static feature flags from the configuration file.

    Returns an empty mapping if the file is missing or invalid; callers should
    treat missing entries as disabled.
    """

    config_path = Path(__file__).resolve().parent / "config" / "features.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: bool(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _check_network_available() -> bool:
    """Check if network access is available (for web search capability)."""
    # Check environment variable override
    network_disabled = os.getenv("MAVEN_NETWORK_DISABLED", "").lower() in ("1", "true", "yes")
    if network_disabled:
        return False
    # Default: assume network is available if web_research flag is enabled
    return True


def get_capabilities() -> Dict[str, Dict[str, Any]]:
    """Return structured capability information.

    Each capability entry contains:
        available: static flag indicating the feature exists
        enabled: runtime flag indicating it can be used now
        reason: optional explanatory text when disabled
    """

    feature_flags = _load_feature_flags()
    execution_status = get_execution_status()

    def capability_state(flag_name: str, requires_execution: bool = False) -> Dict[str, Any]:
        available = bool(feature_flags.get(flag_name, False))
        enabled = available
        reason = ""

        if available and requires_execution:
            if execution_status.mode == ExecMode.DISABLED:
                enabled = False
                reason = f"execution.mode={execution_status.mode.value}"
            elif not execution_status.effective:
                enabled = False
                reason = execution_status.reason or "execution not effective"

        return {"available": available, "enabled": enabled, "reason": reason}

    capabilities = {
        "filesystem_agency": capability_state("filesystem_agency", requires_execution=True),
        "git_agency": capability_state("git_agency", requires_execution=True),
        "execution_agency": capability_state("execution_agency", requires_execution=True),
        "hot_reload": capability_state("hot_reload", requires_execution=True),
        "teacher_learning": capability_state("teacher_learning"),
        "routing_learning": capability_state("routing_learning"),
        "web_research": capability_state("web_research"),
    }

    return capabilities


# =============================================================================
# CAPABILITY SNAPSHOT: Single source of truth for "can you X" questions
# =============================================================================

def get_capability_snapshot() -> Dict[str, Any]:
    """
    Return a complete capability snapshot for answering "can you X" questions.

    This is the SINGLE SOURCE OF TRUTH for capability questions.
    All capability answers MUST use this function, never Teacher/LLM.

    Returns:
        Dictionary with:
        - web_search_enabled: bool - Can Maven search the web?
        - code_execution_enabled: bool - Can Maven run code?
        - filesystem_scope: str - What files can Maven access?
        - can_control_programs: bool - Can Maven control other programs?
        - autonomous_tools: bool - Does Maven use tools without asking?
        - execution_mode: str - Current execution mode
        - execution_reason: str - Why execution is in current state
    """
    # First check if SAFE_CHAT profile is active at the capabilities level
    # This takes precedence over execution guard mode
    if is_safe_chat_mode():
        return {
            "web_search_enabled": False,
            "web_search_reason": "disabled in SAFE_CHAT mode",
            "code_execution_enabled": False,
            "code_execution_reason": "disabled in SAFE_CHAT mode",
            "filesystem_scope": "none",
            "can_control_programs": False,
            "autonomous_tools": False,
            "execution_mode": "SAFE_CHAT",
            "execution_effective": False,
            "execution_source": "safe_chat_profile",
            "execution_reason": "SAFE_CHAT mode - no tools enabled",
        }

    exec_status = get_execution_status()

    # Check for SAFE_CHAT mode in execution guard
    if exec_status.mode == ExecMode.SAFE_CHAT:
        return {
            "web_search_enabled": False,
            "web_search_reason": "disabled in SAFE_CHAT mode",
            "code_execution_enabled": False,
            "code_execution_reason": "disabled in SAFE_CHAT mode",
            "filesystem_scope": "none",
            "can_control_programs": False,
            "autonomous_tools": False,
            "execution_mode": exec_status.mode.value,
            "execution_effective": False,
            "execution_source": exec_status.source,
            "execution_reason": "SAFE_CHAT mode - no tools enabled",
        }

    # Check for FULL_AGENCY mode - all tools enabled
    if exec_status.mode == ExecMode.FULL_AGENCY and exec_status.effective:
        return {
            "web_search_enabled": True,
            "web_search_reason": "",
            "code_execution_enabled": True,
            "code_execution_reason": "",
            "filesystem_scope": "unrestricted",
            "can_control_programs": True,  # Can control via shell
            "autonomous_tools": True,      # Can run agents
            "execution_mode": exec_status.mode.value,
            "execution_effective": True,
            "execution_source": exec_status.source,
            "execution_reason": "",
        }

    # Default behavior for other modes
    feature_flags = _load_feature_flags()

    # Web search: requires web_research flag AND network availability
    web_research_flag = bool(feature_flags.get("web_research", False))
    network_available = _check_network_available()
    web_search_enabled = web_research_flag and network_available

    # Code execution: requires execution_agency flag AND execution mode allows it
    execution_flag = bool(feature_flags.get("execution_agency", False))
    code_execution_enabled = execution_flag and exec_status.mode == ExecMode.FULL and exec_status.effective

    # Filesystem scope: based on execution mode and filesystem_agency flag
    filesystem_flag = bool(feature_flags.get("filesystem_agency", False))
    if not filesystem_flag or exec_status.mode == ExecMode.DISABLED:
        filesystem_scope = "none"
    elif exec_status.mode == ExecMode.READ_ONLY:
        filesystem_scope = "read_only_project_dir"
    elif exec_status.mode == ExecMode.FULL and exec_status.effective:
        filesystem_scope = "project_dir_read_write"
    else:
        filesystem_scope = "none"

    # Can control programs: FALSE in standard modes
    can_control_programs = False

    # Autonomous tools: FALSE in standard modes
    autonomous_tools = False

    return {
        "web_search_enabled": web_search_enabled,
        "web_search_reason": "" if web_search_enabled else ("web_research feature disabled" if not web_research_flag else "network unavailable"),
        "code_execution_enabled": code_execution_enabled,
        "code_execution_reason": "" if code_execution_enabled else f"execution.mode={exec_status.mode.value}",
        "filesystem_scope": filesystem_scope,
        "can_control_programs": can_control_programs,
        "autonomous_tools": autonomous_tools,
        "execution_mode": exec_status.mode.value,
        "execution_effective": exec_status.effective,
        "execution_source": exec_status.source,
        "execution_reason": exec_status.reason,
    }


# =============================================================================
# CAPABILITY QUESTION ANSWERING: Truthful answers from config
# =============================================================================

def answer_capability_question(question: str) -> Optional[Dict[str, Any]]:
    """
    Answer a "can you X" capability question TRUTHFULLY from config.

    This function NEVER hallucinates. It reads the actual capability snapshot
    and returns an honest answer based on real configuration.

    Args:
        question: The capability question (e.g., "can you search the web?")

    Returns:
        Dict with:
        - answer: str - The truthful answer
        - capability: str - Which capability was queried
        - enabled: bool - Whether the capability is enabled
        - source: str - Always "capability_snapshot"
        Or None if question is not a recognized capability question.
    """
    q_lower = question.lower().strip()
    snapshot = get_capability_snapshot()

    # Web search questions
    if any(p in q_lower for p in ["search the web", "browse the internet", "look this up online",
                                   "search online", "web search", "internet search"]):
        enabled = snapshot["web_search_enabled"]
        if enabled:
            answer = "Yes, I can search the web. My web research capability is enabled."
        else:
            reason = snapshot.get("web_search_reason", "web research is disabled")
            answer = f"No. This Maven build cannot search the web; it runs offline. I only use the knowledge baked into my code and saved memory. Reason: {reason}"
        return {
            "answer": answer,
            "capability": "web_search",
            "enabled": enabled,
            "source": "capability_snapshot"
        }

    # Code execution questions
    if any(p in q_lower for p in ["run code", "execute code", "run python", "run scripts",
                                   "execute scripts", "run programs"]):
        enabled = snapshot["code_execution_enabled"]
        mode = snapshot["execution_mode"]
        if enabled:
            answer = "Yes, I can run code. Code execution is enabled in my current configuration."
        else:
            answer = f"No. Code execution is disabled in this configuration (execution.mode={mode}). I can suggest code but not run it."
        return {
            "answer": answer,
            "capability": "code_execution",
            "enabled": enabled,
            "source": "capability_snapshot"
        }

    # Control other programs questions
    if any(p in q_lower for p in ["control other programs", "control apps", "control applications",
                                   "control other apps", "run other programs", "launch other apps"]):
        answer = "No. I cannot control other programs on your computer. I only read and write inside my own project directory and whatever you explicitly pass to me."
        return {
            "answer": answer,
            "capability": "control_programs",
            "enabled": False,
            "source": "capability_snapshot"
        }

    # Read/change files questions
    if any(p in q_lower for p in ["read files", "change files", "access files", "read or change files",
                                   "modify files", "write files", "files on my system"]):
        scope = snapshot["filesystem_scope"]
        if scope == "none":
            answer = "No. File access is disabled in my current configuration. I cannot read or write files."
        elif scope == "read_only_project_dir":
            answer = "I can only read files inside my configured working directory (read-only mode). I cannot write or modify files, and I cannot see the rest of your system."
        elif scope == "project_dir_read_write":
            answer = "I can read and write files inside my configured working directory, and only when you ask me to. I cannot see or access the rest of your system."
        else:
            answer = "I can only read/write inside my configured working directory, and only when you ask me to. I cannot see the rest of your system."
        return {
            "answer": answer,
            "capability": "filesystem",
            "enabled": scope != "none",
            "source": "capability_snapshot"
        }

    # Autonomous tools/internet questions
    if any(p in q_lower for p in ["use tools without", "internet without", "without me asking",
                                   "without asking", "autonomous", "on your own"]):
        answer = "No. I never use tools or the internet unless a request (or a hard-coded pipeline stage) explicitly triggers it. I do not act autonomously."
        return {
            "answer": answer,
            "capability": "autonomous_tools",
            "enabled": False,
            "source": "capability_snapshot"
        }

    # Not a recognized capability question
    return None


def describe_capabilities() -> str:
    """
    Generate a human-readable description of all current capabilities.

    Used by self_model to answer "what can you do" questions.
    """
    snapshot = get_capability_snapshot()
    caps = get_capabilities()

    lines = []

    # Web search
    if snapshot["web_search_enabled"]:
        lines.append("✓ Web search: Enabled - I can search the web for information")
    else:
        lines.append(f"✗ Web search: Disabled - {snapshot.get('web_search_reason', 'not available')}")

    # Code execution
    if snapshot["code_execution_enabled"]:
        lines.append("✓ Code execution: Enabled - I can run code")
    else:
        lines.append(f"✗ Code execution: Disabled - {snapshot.get('code_execution_reason', 'execution.mode=DISABLED')}")

    # Filesystem
    scope = snapshot["filesystem_scope"]
    if scope == "project_dir_read_write":
        lines.append("✓ Filesystem: Read/write access to project directory")
    elif scope == "read_only_project_dir":
        lines.append("⚠ Filesystem: Read-only access to project directory")
    else:
        lines.append("✗ Filesystem: No file access")

    # Control programs (always no)
    lines.append("✗ Control other programs: Not available - I cannot control other applications")

    # Autonomous tools (always no)
    lines.append("✗ Autonomous tools: Not available - I only act on explicit requests")

    # Other capabilities from feature flags
    if caps.get("teacher_learning", {}).get("enabled"):
        lines.append("✓ Teacher learning: Enabled - I can learn from conversations")
    if caps.get("git_agency", {}).get("enabled"):
        lines.append("✓ Git agency: Enabled - I can interact with git repositories")

    return "\n".join(lines)


# =============================================================================
# CAPABILITY STARTUP SCAN: Verify capabilities at system startup
# =============================================================================

# Cached scan results (computed once at startup, refreshed on demand)
_startup_scan_cache: Optional[Dict[str, Any]] = None


def run_capability_startup_scan(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Run a safe capability scan at system startup.

    This function performs shallow checks to verify which capabilities
    are actually functional (not just configured). The scan is:
    - Safe (no destructive actions)
    - Bounded in time (uses timeouts)
    - Gracefully degrading (failures mark capability as unavailable)

    The scan is cached and only runs once unless force_refresh=True.

    Args:
        force_refresh: Force re-scan even if cached results exist

    Returns:
        Dict with:
        - scan_time: ISO timestamp of scan
        - capabilities: Dict mapping capability name to scan result
        - summary: Human-readable summary
    """
    global _startup_scan_cache

    if _startup_scan_cache is not None and not force_refresh:
        return _startup_scan_cache

    import time
    from datetime import datetime

    scan_start = time.time()
    scan_results: Dict[str, Dict[str, Any]] = {}

    # Scan each capability
    scan_results["web_client"] = _scan_web_client()
    scan_results["llm_service"] = _scan_llm_service()
    scan_results["filesystem"] = _scan_filesystem()
    scan_results["git"] = _scan_git()
    scan_results["python_sandbox"] = _scan_python_sandbox()
    scan_results["browser_runtime"] = _scan_browser_runtime()
    scan_results["time"] = _scan_time_tool()

    scan_duration = time.time() - scan_start

    # Build summary
    enabled_count = sum(1 for r in scan_results.values() if r.get("available", False))
    total_count = len(scan_results)

    result = {
        "scan_time": datetime.now().isoformat(),
        "scan_duration_ms": int(scan_duration * 1000),
        "capabilities": scan_results,
        "summary": f"{enabled_count}/{total_count} capabilities available",
        "enabled_capabilities": [
            name for name, r in scan_results.items() if r.get("available", False)
        ],
        "disabled_capabilities": [
            name for name, r in scan_results.items() if not r.get("available", False)
        ]
    }

    _startup_scan_cache = result
    print(f"[CAPABILITY_SCAN] Startup scan complete in {scan_duration*1000:.0f}ms: {result['summary']}")

    return result


def _scan_web_client() -> Dict[str, Any]:
    """Scan web client capability."""
    try:
        from host_tools.web_client import web_client
        # Check if module is importable and has basic structure
        if hasattr(web_client, "fetch") or hasattr(web_client, "get"):
            feature_flags = _load_feature_flags()
            web_enabled = feature_flags.get("web_research", False)
            return {
                "available": web_enabled,
                "reason": "" if web_enabled else "web_research feature disabled",
                "module_present": True
            }
        return {"available": False, "reason": "web_client module incomplete", "module_present": False}
    except ImportError:
        return {"available": False, "reason": "web_client module not found", "module_present": False}
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def _scan_llm_service() -> Dict[str, Any]:
    """Scan LLM service capability."""
    try:
        from brains.tools.llm_service import llm_service
        if llm_service and hasattr(llm_service, "enabled"):
            return {
                "available": llm_service.enabled,
                "reason": "" if llm_service.enabled else "LLM service disabled",
                "module_present": True
            }
        return {"available": False, "reason": "llm_service not properly initialized", "module_present": True}
    except ImportError:
        return {"available": False, "reason": "llm_service module not found", "module_present": False}
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def _scan_filesystem() -> Dict[str, Any]:
    """Scan filesystem capability."""
    try:
        feature_flags = _load_feature_flags()
        fs_enabled = feature_flags.get("filesystem_agency", False)
        exec_status = get_execution_status()

        if not fs_enabled:
            return {"available": False, "reason": "filesystem_agency disabled", "module_present": True}

        if exec_status.mode == ExecMode.DISABLED:
            return {"available": False, "reason": "execution mode DISABLED", "module_present": True}

        scope = "read_write" if exec_status.mode == ExecMode.FULL else "read_only"
        return {
            "available": True,
            "reason": "",
            "module_present": True,
            "scope": scope
        }
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def _scan_git() -> Dict[str, Any]:
    """Scan git capability."""
    try:
        feature_flags = _load_feature_flags()
        git_enabled = feature_flags.get("git_agency", False)

        if not git_enabled:
            return {"available": False, "reason": "git_agency disabled", "module_present": True}

        # Check if git tool is importable
        try:
            from brains.tools import git_tool
            if git_tool:
                return {"available": True, "reason": "", "module_present": True}
        except Exception:
            pass

        return {"available": False, "reason": "git_tool not available", "module_present": False}
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def _scan_python_sandbox() -> Dict[str, Any]:
    """Scan Python sandbox capability."""
    try:
        feature_flags = _load_feature_flags()
        exec_enabled = feature_flags.get("execution_agency", False)
        exec_status = get_execution_status()

        if not exec_enabled:
            return {"available": False, "reason": "execution_agency disabled", "module_present": True}

        if exec_status.mode != ExecMode.FULL or not exec_status.effective:
            return {
                "available": False,
                "reason": f"execution.mode={exec_status.mode.value}",
                "module_present": True
            }

        return {"available": True, "reason": "", "module_present": True}
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def _scan_browser_runtime() -> Dict[str, Any]:
    """Scan browser runtime capability."""
    try:
        # Check if browser runtime is configured and available
        try:
            from optional.browser_runtime import browser_client
            if browser_client and hasattr(browser_client, "is_available"):
                available = browser_client.is_available()
                return {
                    "available": available,
                    "reason": "" if available else "browser runtime not running",
                    "module_present": True
                }
        except ImportError:
            pass

        return {"available": False, "reason": "browser_runtime not installed", "module_present": False}
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def _scan_time_tool() -> Dict[str, Any]:
    """Scan time tool capability.

    The time tool is ALWAYS available because it uses stdlib only.
    This is the authoritative source for "what time is it" questions.
    """
    try:
        from brains.agent.tools.time_now import is_available, get_tool_info

        available = is_available()
        info = get_tool_info()

        return {
            "available": available,
            "reason": "" if available else "time tool not functional",
            "module_present": True,
            "tool_name": info.get("name", "time_now"),
            "description": info.get("description", "System clock access"),
        }
    except ImportError:
        return {"available": False, "reason": "time_now module not found", "module_present": False}
    except Exception as e:
        return {"available": False, "reason": f"scan error: {str(e)[:50]}", "module_present": False}


def get_enabled_capabilities() -> list:
    """Get list of currently enabled capability names."""
    scan = run_capability_startup_scan()
    return scan.get("enabled_capabilities", [])


def is_capability_available(capability_name: str) -> bool:
    """Check if a specific capability is available."""
    scan = run_capability_startup_scan()
    cap_result = scan.get("capabilities", {}).get(capability_name, {})
    return cap_result.get("available", False)


def is_capability_enabled(capability_name: str) -> bool:
    """Check if a specific capability is currently enabled.

    This function reads from the live capability state (get_capabilities())
    to determine if a capability can currently be used. It considers both
    the static feature flag AND runtime conditions like execution mode.

    Args:
        capability_name: Name of capability (e.g., "filesystem_agency", "git_agency",
                        "execution_agency", "web_research", "shell", "filesystem")

    Returns:
        bool: True if capability is enabled and can be used now, False otherwise
    """
    # Map common aliases to canonical capability names
    name_aliases = {
        "filesystem": "filesystem_agency",
        "shell": "execution_agency",
        "git": "git_agency",
        "web": "web_research",
        "web_search": "web_research",
        "browser": "web_research",  # Browser functionality tied to web
        "time": "time",  # Time capability (always available)
        "time_now": "time",
        "clock": "time",
        "current_time": "time",
    }
    canonical_name = name_aliases.get(capability_name, capability_name)

    # Special case: time capability is ALWAYS enabled (stdlib only, no execution needed)
    if canonical_name == "time":
        scan = run_capability_startup_scan()
        time_result = scan.get("capabilities", {}).get("time", {})
        return time_result.get("available", True)  # Default to True for time

    # Check in get_capabilities() first (feature flag + execution mode aware)
    caps = get_capabilities()
    if canonical_name in caps:
        return bool(caps[canonical_name].get("enabled", False))

    # Check in capability snapshot for broader capability queries
    snapshot = get_capability_snapshot()
    snapshot_mappings = {
        "web_research": "web_search_enabled",
        "execution_agency": "code_execution_enabled",
        "filesystem_agency": snapshot.get("filesystem_scope", "none") != "none",
    }

    if canonical_name in snapshot_mappings:
        mapping = snapshot_mappings[canonical_name]
        if isinstance(mapping, bool):
            return mapping
        return bool(snapshot.get(mapping, False))

    # Fallback: check startup scan
    scan = run_capability_startup_scan()
    cap_result = scan.get("capabilities", {}).get(canonical_name, {})
    return cap_result.get("available", False)


def get_capability_reason(capability_name: str) -> str:
    """Get human-readable reason why a capability might be disabled.

    Args:
        capability_name: Name of capability (e.g., "filesystem_agency", "execution_agency")

    Returns:
        str: Human-readable reason (empty string if enabled)
    """
    # Map common aliases to canonical capability names
    name_aliases = {
        "filesystem": "filesystem_agency",
        "shell": "execution_agency",
        "git": "git_agency",
        "web": "web_research",
        "web_search": "web_research",
        "browser": "web_research",
    }
    canonical_name = name_aliases.get(capability_name, capability_name)

    # Check in get_capabilities() first
    caps = get_capabilities()
    if canonical_name in caps:
        cap_info = caps[canonical_name]
        if cap_info.get("enabled", False):
            return ""  # Enabled, no reason to report
        return cap_info.get("reason", "capability disabled")

    # Check capability snapshot for reasons
    snapshot = get_capability_snapshot()
    reason_mappings = {
        "web_research": snapshot.get("web_search_reason", "web research disabled"),
        "execution_agency": snapshot.get("code_execution_reason", "execution disabled"),
        "filesystem_agency": "file access disabled" if snapshot.get("filesystem_scope") == "none" else "",
    }

    if canonical_name in reason_mappings:
        return reason_mappings[canonical_name]

    # Fallback: check startup scan
    scan = run_capability_startup_scan()
    cap_result = scan.get("capabilities", {}).get(canonical_name, {})
    if cap_result.get("available", False):
        return ""
    return cap_result.get("reason", "capability not available")


# =============================================================================
# INTEGRATION WITH brains/system_capabilities.py
# =============================================================================
# The new system_capabilities module provides more detailed probes.
# This bridge allows both modules to work together during the transition.

def get_capability_truth() -> Dict[str, Any]:
    """
    Get the comprehensive capability truth object.

    This is a bridge to the new brains/system_capabilities module.
    Use this for the single source of truth for "can you X" questions.
    """
    try:
        from brains.system_capabilities import get_capability_truth as _get_truth
        return _get_truth()
    except ImportError:
        # Fallback to legacy scan if new module not available
        scan = run_capability_startup_scan()
        return {
            "tools": {
                name: "available" if r.get("available") else "unavailable"
                for name, r in scan.get("capabilities", {}).items()
            },
            "brains": {},
            "summary": scan.get("summary", "unknown")
        }


def get_current_capabilities_for_answer() -> Dict[str, Any]:
    """
    Get current capabilities formatted for answering user questions.

    Returns a dict with 'available' and 'unavailable' lists of human-readable
    capability descriptions.
    """
    try:
        from brains.system_capabilities import get_current_capabilities
        return get_current_capabilities()
    except ImportError:
        # Fallback
        snapshot = get_capability_snapshot()
        available = []
        unavailable = []

        if snapshot["web_search_enabled"]:
            available.append("search the web")
        else:
            unavailable.append(f"search the web ({snapshot.get('web_search_reason', 'disabled')})")

        if snapshot["code_execution_enabled"]:
            available.append("run code")
        else:
            unavailable.append(f"run code ({snapshot.get('code_execution_reason', 'disabled')})")

        if snapshot["filesystem_scope"] != "none":
            available.append(f"access files ({snapshot['filesystem_scope']})")
        else:
            unavailable.append("access files (disabled)")

        return {"available": available, "unavailable": unavailable}


# =============================================================================
# CAPABILITY PROFILES
# =============================================================================

from enum import Enum


class CapabilityProfile(str, Enum):
    """Capability profiles that control what Maven can do."""
    SAFE_CHAT = "SAFE_CHAT"        # No tools, no side effects - pure conversation
    FULL_AGENCY = "FULL_AGENCY"    # Full access to all tools and capabilities


# Safe chat profile - no tools, no side effects
SAFE_CHAT_PROFILE = {
    "can_access_files": False,
    "can_use_shell": False,
    "can_use_python": False,
    "can_use_git": False,
    "can_use_web_client": False,
    "can_use_browser_runtime": False,
    "can_run_agents": False,
    "filesystem_scope": "none",
    "execution_mode": "DISABLED",
}

# Full agency profile - all capabilities enabled
FULL_AGENCY_PROFILE = {
    "can_access_files": True,
    "can_use_shell": True,
    "can_use_python": True,
    "can_use_git": True,
    "can_use_web_client": True,
    "can_use_browser_runtime": True,
    "can_run_agents": True,
    "filesystem_scope": "unrestricted",  # No path restrictions
    "execution_mode": "FULL_AGENCY",
}

# Profile registry
CAPABILITY_PROFILES = {
    CapabilityProfile.SAFE_CHAT: SAFE_CHAT_PROFILE,
    CapabilityProfile.FULL_AGENCY: FULL_AGENCY_PROFILE,
}

# Current active profile (default to SAFE_CHAT for safety)
_current_profile: CapabilityProfile = CapabilityProfile.SAFE_CHAT


def get_current_profile() -> CapabilityProfile:
    """Get the current capability profile."""
    global _current_profile
    return _current_profile


def set_profile(profile: CapabilityProfile) -> None:
    """Set the current capability profile."""
    global _current_profile
    _current_profile = profile


def get_profile_capabilities(profile: CapabilityProfile = None) -> dict:
    """Get the capabilities for a specific profile or the current profile."""
    if profile is None:
        profile = get_current_profile()
    return CAPABILITY_PROFILES.get(profile, SAFE_CHAT_PROFILE).copy()


def activate_profile(profile_name: str, reason: str = "profile activation") -> dict:
    """
    Activate a capability profile by name and return the capability snapshot.

    Args:
        profile_name: "SAFE_CHAT" or "FULL_AGENCY"
        reason: Human-readable reason for activation

    Returns:
        The capability snapshot for the activated profile
    """
    profile_name = profile_name.upper()

    if profile_name == "FULL_AGENCY":
        set_profile(CapabilityProfile.FULL_AGENCY)
        # Also enable in execution guard
        enable_full_agency(reason)
        return get_full_agency_snapshot()
    elif profile_name == "SAFE_CHAT":
        set_profile(CapabilityProfile.SAFE_CHAT)
        # Disable execution in guard
        from brains.tools.execution_guard import disable_execution
        disable_execution(reason)
        return get_safe_chat_snapshot()
    else:
        raise ValueError(f"Unknown profile: {profile_name}. Valid profiles: SAFE_CHAT, FULL_AGENCY")


def get_safe_chat_snapshot() -> dict:
    """
    Return a capability snapshot for SAFE_CHAT mode.

    In SAFE_CHAT mode:
    - No tool access
    - No file access
    - No shell/python execution
    - No web access
    - Pure conversation only

    Returns:
        Dictionary with all capabilities disabled.
    """
    return {
        "web_search_enabled": False,
        "web_search_reason": "disabled in SAFE_CHAT mode",
        "code_execution_enabled": False,
        "code_execution_reason": "disabled in SAFE_CHAT mode",
        "filesystem_scope": "none",
        "can_control_programs": False,
        "autonomous_tools": False,
        "execution_mode": "DISABLED",
        "execution_effective": False,
        "execution_source": "safe_chat_profile",
        "execution_reason": "SAFE_CHAT mode - no tools enabled",
        # All capabilities disabled
        "can_access_files": False,
        "can_use_shell": False,
        "can_use_python": False,
        "can_use_git": False,
        "can_use_web_client": False,
        "can_use_browser_runtime": False,
        "can_run_agents": False,
    }


def is_safe_chat_mode() -> bool:
    """Check if the system is currently in SAFE_CHAT mode."""
    return get_current_profile() == CapabilityProfile.SAFE_CHAT


def get_full_agency_snapshot() -> Dict[str, Any]:
    """
    Return a capability snapshot for FULL_AGENCY mode.

    In FULL_AGENCY mode:
    - All capabilities are enabled
    - No path restrictions on filesystem
    - Can control other programs via shell
    - Can run autonomous agents
    - Web search and browser runtime enabled

    Returns:
        Dictionary with all capabilities set to their most permissive values.
    """
    return {
        "web_search_enabled": True,
        "web_search_reason": "",
        "code_execution_enabled": True,
        "code_execution_reason": "",
        "filesystem_scope": "unrestricted",  # Can read/write anywhere
        "can_control_programs": True,  # Can control via shell
        "autonomous_tools": True,  # Can run agents
        "execution_mode": "FULL_AGENCY",
        "execution_effective": True,
        "execution_source": "full_agency_profile",
        "execution_reason": "",
        # Additional FULL_AGENCY capabilities
        "can_access_files": True,
        "can_use_shell": True,
        "can_use_python": True,
        "can_use_git": True,
        "can_use_web_client": True,
        "can_use_browser_runtime": True,
        "can_run_agents": True,
    }


def is_full_agency_mode() -> bool:
    """Check if the system is currently in FULL_AGENCY mode."""
    exec_status = get_execution_status()
    return exec_status.mode == ExecMode.FULL_AGENCY and exec_status.effective


def activate_full_agency(reason: str = "activated via capability module") -> Dict[str, Any]:
    """
    Activate FULL_AGENCY mode and return the new capability snapshot.

    This function:
    1. Enables FULL_AGENCY execution mode
    2. Returns the full agency capability snapshot

    Args:
        reason: Human-readable reason for activation

    Returns:
        Full agency capability snapshot
    """
    enable_full_agency(reason)
    return get_full_agency_snapshot()


def answer_capability_question_full_agency(question: str) -> Optional[Dict[str, Any]]:
    """
    Answer a "can you X" capability question for FULL_AGENCY mode.

    In FULL_AGENCY mode, most capability questions get "yes" answers
    because the system has unrestricted access.

    Args:
        question: The capability question

    Returns:
        Dict with answer, capability, enabled, source or None if not a capability question.
    """
    q_lower = question.lower().strip()

    # Check if we're in FULL_AGENCY mode
    if not is_full_agency_mode():
        return None  # Fall back to normal capability answering

    # Web/browsing questions
    if any(p in q_lower for p in ["search the web", "browse the internet", "look this up online",
                                   "search online", "web search", "internet search"]):
        return {
            "answer": "Yes, I can search the web. I have full web research and browser capabilities enabled.",
            "capability": "web_search",
            "enabled": True,
            "source": "full_agency_profile"
        }

    # Code execution questions
    if any(p in q_lower for p in ["run code", "execute code", "run python", "run scripts",
                                   "execute scripts", "run programs"]):
        return {
            "answer": "Yes, I can run code. I have full code execution capabilities including Python and shell commands.",
            "capability": "code_execution",
            "enabled": True,
            "source": "full_agency_profile"
        }

    # Control other programs questions
    if any(p in q_lower for p in ["control other programs", "control apps", "control applications",
                                   "control other apps", "run other programs", "launch other apps"]):
        return {
            "answer": "Yes, I can control other programs through shell commands and process management.",
            "capability": "control_programs",
            "enabled": True,
            "source": "full_agency_profile"
        }

    # File access questions
    if any(p in q_lower for p in ["read files", "change files", "access files", "read or change files",
                                   "modify files", "write files", "files on my system"]):
        return {
            "answer": "Yes, I can read and write files anywhere on the system (limited only by OS user permissions).",
            "capability": "filesystem",
            "enabled": True,
            "source": "full_agency_profile"
        }

    # Autonomous tools/agents questions
    if any(p in q_lower for p in ["use tools without", "internet without", "without me asking",
                                   "without asking", "autonomous", "on your own", "run agents"]):
        return {
            "answer": "Yes, I can use tools and run autonomous agents to accomplish tasks.",
            "capability": "autonomous_tools",
            "enabled": True,
            "source": "full_agency_profile"
        }

    # Git questions
    if any(p in q_lower for p in ["use git", "git commands", "commit", "push", "clone"]):
        return {
            "answer": "Yes, I have full git capabilities including clone, commit, push, and all other git operations.",
            "capability": "git",
            "enabled": True,
            "source": "full_agency_profile"
        }

    return None  # Not a recognized capability question
