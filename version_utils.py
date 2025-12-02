"""
Dynamic version information from Git, avoiding stale hardcoded values.

This module provides version info (commit, branch, features) by:
1. Running git commands to get current commit/branch
2. Reading feature flags from config/features.json
3. Falling back to maven_version.txt if git is unavailable

Usage:
    from version_utils import get_version_info, get_version_banner

    info = get_version_info()
    # Returns: {"commit": "abc123", "branch": "main", "features": ["fs", "git", ...]}

    banner = get_version_banner()
    # Returns: "commit=abc123, branch=main, features=fs+git+introspection+..."
"""

from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import subprocess
import json


def _get_maven_root() -> Path:
    """Get Maven root directory."""
    return Path(__file__).parent


def _run_git_command(args: List[str], timeout: int = 2) -> str | None:
    """
    Run a git command and return output, or None if it fails.

    Args:
        args: Git command arguments (e.g., ["rev-parse", "--short", "HEAD"])
        timeout: Command timeout in seconds

    Returns:
        Command output stripped, or None if failed
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=_get_maven_root(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def _get_git_commit() -> str:
    """Get current git commit (short SHA), or 'unknown' if not available."""
    commit = _run_git_command(["rev-parse", "--short", "HEAD"])
    return commit if commit else "unknown"


def _get_git_branch() -> str:
    """Get current git branch, or 'unknown' if not available."""
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    return branch if branch else "unknown"


def _get_enabled_features() -> List[str]:
    """
    Get list of enabled feature names from config/features.json.

    Returns:
        List of feature short names (e.g., ["fs", "git", "introspection"])
    """
    maven_root = _get_maven_root()
    features_file = maven_root / "config" / "features.json"

    if not features_file.exists():
        return []

    try:
        with open(features_file, 'r') as f:
            features = json.load(f)

        # Map full feature names to short abbreviations
        feature_abbrev = {
            "filesystem_agency": "fs",
            "git_agency": "git",
            "execution_guard": "exec",
            "hot_reload": "reload",
            "self_introspection": "introspection",
            "teacher_learning": "teacher",
            "routing_learning": "routing",
            "web_research": "web",
            "pattern_recognition": "patterns",
            "memory_consolidation": "memory",
            "learning_mode": "lmode",
            "strategy_reasoning": "strategy",
        }

        enabled = []
        for full_name, abbrev in feature_abbrev.items():
            if features.get(full_name, False):
                enabled.append(abbrev)

        return enabled

    except Exception:
        return []


def _read_version_file() -> Dict[str, str] | None:
    """
    Read maven_version.txt as fallback when git is unavailable.

    Expected format: "commit=abc123, branch=main, features=fs+git+..."

    Returns:
        Dict with commit/branch/features, or None if file doesn't exist or can't be parsed
    """
    maven_root = _get_maven_root()
    version_file = maven_root / "maven_version.txt"

    if not version_file.exists():
        return None

    try:
        content = version_file.read_text().strip()

        # Parse format: "commit=X, branch=Y, features=Z"
        parts = {}
        for part in content.split(", "):
            if "=" in part:
                key, value = part.split("=", 1)
                parts[key.strip()] = value.strip()

        return {
            "commit": parts.get("commit", "unknown"),
            "branch": parts.get("branch", "unknown"),
            "features": parts.get("features", "").split("+") if "features" in parts else []
        }

    except Exception:
        return None


def get_version_info() -> Dict[str, Any]:
    """
    Get version information for this Maven runtime.

    Returns:
        Dict with:
            commit: str (git short SHA or "unknown")
            branch: str (git branch name or "unknown")
            features: List[str] (enabled feature abbreviations)
            source: str ("git" | "file" | "default")
    """
    # Try git first
    commit = _get_git_commit()
    branch = _get_git_branch()

    if commit != "unknown" and branch != "unknown":
        # Git is available, get features from config
        features = _get_enabled_features()
        return {
            "commit": commit,
            "branch": branch,
            "features": features,
            "source": "git"
        }

    # Fall back to version file
    file_info = _read_version_file()
    if file_info:
        return {
            "commit": file_info["commit"],
            "branch": file_info["branch"],
            "features": file_info["features"],
            "source": "file"
        }

    # Final fallback
    return {
        "commit": "unknown",
        "branch": "unknown",
        "features": [],
        "source": "default"
    }


def get_version_banner() -> str:
    """
    Get a one-line version banner string suitable for startup display.

    Format: "commit=abc123, branch=main, features=fs+git+introspection+..."

    Returns:
        Formatted version string
    """
    info = get_version_info()

    features_str = "+".join(info["features"]) if info["features"] else "none"

    return f"commit={info['commit']}, branch={info['branch']}, features={features_str}"


# Export key functions
__all__ = [
    "get_version_info",
    "get_version_banner",
]

