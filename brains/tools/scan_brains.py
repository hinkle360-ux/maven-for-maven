"""
Filesystem scan helpers for brain role detection.

IMPORTANT: This module only scans the filesystem. It does NOT determine
which folders are actually brains. Use brain_roles.py for the canonical
whitelist of cognitive brains.

The scan results should be filtered against CANONICAL_COGNITIVE_BRAINS
to get the actual list of recognized brains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

from brains.maven_paths import MAVEN_ROOT


def _list_dirs(path: Path) -> List[str]:
    """List subdirectories, excluding hidden/private ones."""
    if not path.exists():
        return []
    return sorted(
        [p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("_")]
    )


def scan_all_brains() -> Dict[str, List[str] | str | None]:
    """
    Scan filesystem for brain-like directories.

    NOTE: This returns ALL folders, not just canonical brains.
    Use brain_roles.get_cognitive_brains() for the authoritative list.

    Returns:
        Dict with:
        - cognitive: folders under brains/cognitive/
        - domain: domain bank names + self brain
        - other: top-level folders under brains/
        - self_brain: name of self/user brain if exists
    """
    brains_root = MAVEN_ROOT / "brains"

    cognitive = _list_dirs(brains_root / "cognitive")
    domain_banks = _list_dirs(brains_root / "domain_banks")

    other: List[str] = []
    for entry in _list_dirs(brains_root):
        if entry in {"cognitive", "domain_banks"}:
            continue
        other.append(entry)

    self_brain = "personal" if (brains_root / "personal").exists() else None

    domain: List[str] = []
    domain.extend(domain_banks)
    if self_brain:
        domain.append(f"_top_level/{self_brain}")

    return {
        "cognitive": cognitive,
        "domain": domain,
        "other": other,
        "self_brain": self_brain,
    }


def get_cognitive_brain_folders() -> List[str]:
    """
    Get list of folders under brains/cognitive/.

    NOTE: Not all of these are actual brains. Use brain_roles.py
    to get the canonical whitelist.
    """
    brains_root = MAVEN_ROOT / "brains"
    return _list_dirs(brains_root / "cognitive")


def validate_brain_folders(canonical_brains: Set[str]) -> Dict[str, List[str]]:
    """
    Validate filesystem against canonical brain list.

    Args:
        canonical_brains: Set of brain names that should exist

    Returns:
        Dict with:
        - missing: brains in whitelist but no folder exists
        - extra: folders that exist but aren't in whitelist
    """
    brains_root = MAVEN_ROOT / "brains"
    cognitive_folders = set(_list_dirs(brains_root / "cognitive"))
    cognitive_folders.discard("__pycache__")

    missing = canonical_brains - cognitive_folders
    extra = cognitive_folders - canonical_brains

    return {
        "missing": sorted(missing),
        "extra": sorted(extra),
    }


__all__ = [
    "scan_all_brains",
    "get_cognitive_brain_folders",
    "validate_brain_folders",
]
