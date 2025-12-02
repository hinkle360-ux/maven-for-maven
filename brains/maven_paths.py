"""
maven_paths.py
~~~~~~~~~~~~~~

Central path management for all Maven operations.

This module ensures ALL memory operations stay within the maven2_fix directory.

HARD CONSTRAINTS:
- All memory files MUST live under maven_root (C:/Users/hinkl/Desktop/maven2_fix on Windows,
  or /home/user/maven/maven2_fix in current environment)
- NO writes to APPDATA, temp directories, Documents, or any other external locations
- ALL paths must be validated to start with MAVEN_ROOT after resolution

Usage:
    from brains.maven_paths import MAVEN_ROOT, validate_path_confinement

    # Get base path
    memory_path = MAVEN_ROOT / "memory" / "brains" / brain_name

    # Validate before writing
    validate_path_confinement(memory_path)
"""

from __future__ import annotations
from pathlib import Path
from typing import Union


def get_maven_root() -> Path:
    """
    Get the Maven root directory (maven2_fix).

    This is the ONLY source of truth for the root path.
    All other paths must be derived from this.

    Returns:
        Absolute path to maven2_fix directory
    """
    # This file is in brains/, so maven_root is parent
    return Path(__file__).resolve().parent.parent


# Global constant - the single source of truth
MAVEN_ROOT = get_maven_root()
REPORTS_ROOT = MAVEN_ROOT / "reports"
BRAINS_ROOT = MAVEN_ROOT / "brains"


class PathConfinementError(Exception):
    """Raised when a path attempts to escape maven_root."""
    pass


def validate_path_confinement(path: Union[str, Path], operation: str = "write") -> Path:
    """
    Validate that a path stays within MAVEN_ROOT.

    This function MUST be called before any file write operation
    to ensure we never write outside maven2_fix.

    Args:
        path: Path to validate (relative or absolute)
        operation: Description of operation (for error messages)

    Returns:
        Resolved absolute path (guaranteed to be under MAVEN_ROOT)

    Raises:
        PathConfinementError: If path escapes MAVEN_ROOT

    Example:
        >>> safe_path = validate_path_confinement(memory_path, "memory write")
        >>> # Now safe to write to safe_path
    """
    path_obj = Path(path).resolve()

    # Check if path is under MAVEN_ROOT
    try:
        path_obj.relative_to(MAVEN_ROOT)
    except ValueError:
        # Path is not under MAVEN_ROOT
        raise PathConfinementError(
            f"SECURITY: {operation} attempted outside MAVEN_ROOT!\n"
            f"  Attempted path: {path_obj}\n"
            f"  MAVEN_ROOT: {MAVEN_ROOT}\n"
            f"  This violates the hard constraint that all files must live under maven2_fix."
        )

    return path_obj


def get_memory_root() -> Path:
    """
    Get the root directory for all memory storage.

    Returns:
        Path to memory/ directory (MAVEN_ROOT/memory)
    """
    return MAVEN_ROOT / "memory"


def get_runtime_memory_root() -> Path:
    """
    Get the root directory for runtime memory storage.

    Runtime memory must live under brains/ to avoid creating new
    top-level folders alongside the Maven source tree.

    Returns:
        Path to brains/runtime_memory directory
    """
    return MAVEN_ROOT / "brains" / "runtime_memory"


def get_reports_path(*parts: str) -> Path:
    """Build a path under the reports directory, enforcing confinement."""

    path = REPORTS_ROOT.joinpath(*parts)
    return validate_path_confinement(path, "reports write")


def get_brains_path(*parts: str) -> Path:
    """Build a path under the brains directory, enforcing confinement."""

    path = BRAINS_ROOT.joinpath(*parts)
    return validate_path_confinement(path, "brains path")


def get_runtime_domain_banks_root() -> Path:
    """
    Get the runtime storage root for domain banks.

    Returns:
        Path to brains/runtime_memory/domain_banks
    """
    return get_runtime_memory_root() / "domain_banks"


def get_brain_memory_path(brain_name: str, brain_category: str) -> Path:
    """
    Get the memory storage path for a specific brain.

    CORRECTED: Each brain has its own local memory/ folder.
    Memory is NOT centralized.

    Args:
        brain_name: Name of the brain (e.g., "reasoning", "factual")
        brain_category: Category of brain ("cognitive", "domain", or "other")

    Returns:
        Path where this brain's memory should be stored

    Example:
        >>> get_brain_memory_path("reasoning", "cognitive")
        Path("/home/user/maven/maven2_fix/brains/cognitive/reasoning/memory")

        >>> get_brain_memory_path("factual", "domain")
        Path("/home/user/maven/maven2_fix/brains/domain_banks/factual/memory")
    """
    brains_root = MAVEN_ROOT / "brains"

    if brain_category == "cognitive":
        # Cognitive brains: brains/cognitive/{brain_name}/memory
        return brains_root / "cognitive" / brain_name / "memory"
    elif brain_category == "domain":
        # Domain banks: brains/domain_banks/{brain_name}/memory
        return brains_root / "domain_banks" / brain_name / "memory"
    elif brain_category == "librarian":
        # Memory Librarian is a cognitive brain
        return brains_root / "cognitive" / brain_name / "memory"
    else:
        # For "other" or unknown categories (agent, governance, etc.)
        # Put under brains/{brain_name}/memory
        return brains_root / brain_name / "memory"


def get_librarian_memory_path() -> Path:
    """
    Get the memory path for the Memory Librarian.

    The Librarian has its own memory for:
    - Routing rules
    - Cross-bank indexes
    - Telemetry

    Returns:
        Path to librarian memory storage
    """
    # Memory Librarian is a cognitive brain with its own local memory
    return MAVEN_ROOT / "brains" / "cognitive" / "memory_librarian" / "memory"


# Ensure MAVEN_ROOT exists (should always be true, but check anyway)
if not MAVEN_ROOT.exists():
    raise RuntimeError(
        f"MAVEN_ROOT does not exist: {MAVEN_ROOT}\n"
        f"This should never happen. Check your installation."
    )


def ensure_runtime_memory_structure() -> None:
    """
    Ensure runtime_memory directory structure exists.

    Creates the runtime memory hierarchy if it doesn't exist:
    - runtime_memory/STM/  (Short-Term Memory)
    - runtime_memory/MTM/  (Medium-Term Memory)
    - runtime_memory/LTM/  (Long-Term Memory)
    - runtime_memory/Archive/

    This directory is in .gitignore, so it must be created at runtime.
    """
    runtime_memory = validate_path_confinement(
        get_runtime_memory_root(), "runtime memory root creation"
    )

    # Create base directory
    runtime_memory.mkdir(parents=True, exist_ok=True)

    # Create memory tier subdirectories
    for tier in ["STM", "MTM", "LTM", "Archive"]:
        tier_path = validate_path_confinement(
            runtime_memory / tier, f"runtime memory tier creation: {tier}"
        )
        tier_path.mkdir(exist_ok=True)


# Auto-create runtime memory structure on module import
ensure_runtime_memory_structure()


# Export public API
__all__ = [
    "MAVEN_ROOT",
    "REPORTS_ROOT",
    "BRAINS_ROOT",
    "PathConfinementError",
    "validate_path_confinement",
    "get_memory_root",
    "get_brain_memory_path",
    "get_librarian_memory_path",
    "get_runtime_memory_root",
    "get_runtime_domain_banks_root",
    "get_reports_path",
    "get_brains_path",
]
