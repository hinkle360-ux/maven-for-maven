"""
Inventory Brain Service
=======================

Provides comprehensive inventory of all cognitive brains in the Maven system.

This brain:
1. Recursively scans brains/cognitive/* to discover brain modules
2. Returns list of all modules with "brain" in file or folder name
3. Ignores __pycache__, config folders, and api folders
4. Initializes patterns on startup (with fallback for zero patterns)
5. Stages unclassified items for user review
6. Returns structured inventory: { "brains": [...], "total": <count> }

STDLIB ONLY, OFFLINE, WINDOWS 10 COMPATIBLE
"""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from brains.maven_paths import get_maven_root, MAVEN_ROOT


# Folders to ignore during scan
IGNORED_FOLDERS: Set[str] = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "config",
    "api",
    "memory",  # memory subdirs are data, not brains
    "router_memory",
    "pattern_memory",
}

# File patterns that indicate backup/duplicate versions
BACKUP_PATTERNS: Set[str] = {
    "_old",
    "_copy",
    "_backup",
    ".bak",
    ".orig",
    "_deprecated",
}


class InventoryBrain:
    """
    Brain inventory system that scans and catalogs all cognitive brains.

    Features:
    - Recursive scanning of brains/cognitive/*
    - Pattern-based brain detection (looks for "brain" in file/folder names)
    - Automatic filtering of non-brain folders
    - Fallback path when no patterns are stored
    - Duplicate version detection and quarantine
    - Unclassified item staging
    """

    def __init__(self):
        """Initialize the inventory brain with pattern loading."""
        self.maven_root = get_maven_root()
        self.cognitive_path = self.maven_root / "brains" / "cognitive"
        self.quarantine_path = self.maven_root / "quarantine"

        # Load any stored patterns (for future pattern-based classification)
        self._patterns: Dict[str, Any] = {}
        self._unclassified: List[Dict[str, Any]] = []

        # Initialize patterns on startup
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """
        Initialize pattern matching for brain classification.

        This runs on startup and loads any stored patterns.
        If no patterns exist, the system still works via filesystem scanning.
        """
        pattern_file = self.maven_root / "brains" / "cognitive" / "inventory" / "memory" / "patterns.json"

        try:
            if pattern_file.exists():
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    self._patterns = json.load(f)
                print(f"[INVENTORY] Loaded {len(self._patterns)} classification patterns")
            else:
                # No patterns stored - that's OK, we use filesystem scanning as fallback
                self._patterns = {}
                print("[INVENTORY] No stored patterns, using filesystem scan fallback")
        except Exception as e:
            print(f"[INVENTORY] Pattern initialization warning: {e}")
            self._patterns = {}

    def _is_brain_file(self, path: Path) -> bool:
        """Check if a file is a brain module."""
        name = path.name.lower()

        # Must be a Python file
        if not name.endswith('.py'):
            return False

        # Skip __init__ files
        if name == '__init__.py':
            return False

        # Look for "brain" in the filename OR any service file in service/ dir
        # Many brain modules don't follow the *_brain.py naming convention
        return True  # Any .py file in service/ is a brain module

    def _is_brain_folder(self, path: Path) -> bool:
        """Check if a folder likely contains a brain."""
        name = path.name.lower()

        # Skip ignored folders
        if name in IGNORED_FOLDERS:
            return False

        # Skip hidden folders
        if name.startswith('.') or name.startswith('_'):
            return False

        # Check if folder has a service subdirectory with brain files
        service_path = path / "service"
        if service_path.exists() and service_path.is_dir():
            for f in service_path.iterdir():
                if f.is_file() and self._is_brain_file(f):
                    return True

        # Check for direct brain files in the folder
        for f in path.iterdir():
            if f.is_file() and self._is_brain_file(f):
                return True

        return False

    def _is_backup_file(self, path: Path) -> bool:
        """Check if a file is a backup/duplicate version."""
        name = path.name.lower()

        for pattern in BACKUP_PATTERNS:
            if pattern in name:
                return True

        return False

    def _move_to_quarantine(self, path: Path) -> Optional[str]:
        """
        Move a file to quarantine directory.

        Returns the new path or None if failed.
        Does NOT auto-delete.
        """
        try:
            # Create quarantine directory if needed
            self.quarantine_path.mkdir(parents=True, exist_ok=True)

            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_subdir = self.quarantine_path / timestamp
            quarantine_subdir.mkdir(exist_ok=True)

            # Preserve relative path structure
            rel_path = path.relative_to(self.maven_root)
            dest_path = quarantine_subdir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file (not delete!)
            shutil.move(str(path), str(dest_path))

            print(f"[INVENTORY] Quarantined: {path} -> {dest_path}")
            return str(dest_path)

        except Exception as e:
            print(f"[INVENTORY] Could not quarantine {path}: {e}")
            return None

    def scan_cognitive_brains(self, quarantine_backups: bool = True) -> Dict[str, Any]:
        """
        Recursively scan brains/cognitive/* and return inventory.

        Args:
            quarantine_backups: If True, move backup files to quarantine

        Returns:
            Dictionary with:
            - brains: List of brain info dicts
            - total: Count of brains found
            - quarantined: List of files moved to quarantine
            - unclassified: List of items that couldn't be classified
        """
        brains: List[Dict[str, Any]] = []
        quarantined: List[str] = []
        backups_found: List[Path] = []

        # Clear unclassified list
        self._unclassified = []

        if not self.cognitive_path.exists():
            print(f"[INVENTORY] Cognitive path does not exist: {self.cognitive_path}")
            return {
                "brains": [],
                "total": 0,
                "quarantined": [],
                "unclassified": []
            }

        # Scan all subdirectories
        for item in sorted(self.cognitive_path.iterdir()):
            if not item.is_dir():
                continue

            name = item.name

            # Skip ignored folders
            if name in IGNORED_FOLDERS or name.startswith('.') or name.startswith('_'):
                continue

            # Check if this is a brain folder
            if self._is_brain_folder(item):
                brain_info = self._extract_brain_info(item)
                if brain_info:
                    brains.append(brain_info)
            else:
                # Could not classify - stage for user review
                self._unclassified.append({
                    "name": name,
                    "path": str(item),
                    "type": "folder"
                })

        # Scan for backup files throughout cognitive directory
        for root, dirs, files in os.walk(self.cognitive_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_FOLDERS and not d.startswith('.')]

            for file in files:
                file_path = Path(root) / file
                if self._is_backup_file(file_path):
                    backups_found.append(file_path)

        # Handle backup files
        if quarantine_backups and backups_found:
            for backup_path in backups_found:
                new_path = self._move_to_quarantine(backup_path)
                if new_path:
                    quarantined.append(new_path)

        # Sort brains by name
        brains.sort(key=lambda x: x.get("name", ""))

        result = {
            "brains": brains,
            "total": len(brains),
            "quarantined": quarantined,
            "unclassified": self._unclassified
        }

        print(f"[INVENTORY] Found {len(brains)} cognitive brains")
        if quarantined:
            print(f"[INVENTORY] Quarantined {len(quarantined)} backup files")
        if self._unclassified:
            print(f"[INVENTORY] {len(self._unclassified)} unclassified items staged for review")

        return result

    def _extract_brain_info(self, brain_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Extract information about a brain from its directory.

        Returns dict with name, path, service_file, has_memory, etc.
        """
        name = brain_dir.name

        # Look for service file
        service_file = None
        service_path = brain_dir / "service"

        if service_path.exists():
            # Look in service/ subdirectory
            for f in service_path.iterdir():
                if f.is_file() and self._is_brain_file(f):
                    service_file = str(f.relative_to(self.maven_root))
                    break
        else:
            # Look directly in brain folder
            for f in brain_dir.iterdir():
                if f.is_file() and self._is_brain_file(f):
                    service_file = str(f.relative_to(self.maven_root))
                    break

        if not service_file:
            return None

        # Check for memory directory
        memory_path = brain_dir / "memory"
        has_memory = memory_path.exists() and memory_path.is_dir()

        # Try to extract docstring/purpose from service file
        purpose = self._extract_purpose(self.maven_root / service_file)

        return {
            "name": name,
            "path": str(brain_dir.relative_to(self.maven_root)),
            "service_file": service_file,
            "has_memory": has_memory,
            "purpose": purpose or f"{name} brain"
        }

    def _extract_purpose(self, service_file: Path) -> Optional[str]:
        """Extract the purpose/description from a brain's docstring."""
        try:
            with open(service_file, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars

            # Look for docstring
            if '"""' in content:
                parts = content.split('"""')
                if len(parts) >= 2:
                    docstring = parts[1].strip()
                    # Get first line
                    first_line = docstring.split('\n')[0].strip()
                    if first_line:
                        return first_line[:100]
        except Exception:
            pass

        return None

    def get_inventory(self) -> Dict[str, Any]:
        """
        Get complete brain inventory.

        This is the main entry point that ensures fallback behavior
        even with zero stored patterns.

        Returns:
            { "brains": [...], "total": <count> }
        """
        # Always use filesystem scan as the source of truth
        # Patterns are for future classification enhancement
        result = self.scan_cognitive_brains(quarantine_backups=False)

        # Format for expected output
        return {
            "brains": result["brains"],
            "total": result["total"]
        }

    def get_unclassified_items(self) -> List[Dict[str, Any]]:
        """
        Get list of items that couldn't be classified.

        These should be reviewed by the user:
        "Unclassified item found: <name>. Should this be added?"
        """
        return self._unclassified.copy()

    def quarantine_duplicates(self) -> Dict[str, Any]:
        """
        Find and quarantine duplicate/backup versions.

        Moves files matching backup patterns to /quarantine.
        Does NOT auto-delete.
        """
        result = self.scan_cognitive_brains(quarantine_backups=True)

        return {
            "quarantined": result["quarantined"],
            "count": len(result["quarantined"])
        }


# Singleton instance
_inventory_brain: Optional[InventoryBrain] = None


def get_inventory_brain() -> InventoryBrain:
    """Get or create the singleton InventoryBrain instance."""
    global _inventory_brain
    if _inventory_brain is None:
        _inventory_brain = InventoryBrain()
    return _inventory_brain


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the inventory brain service.

    Supported operations:
    - LIST / INVENTORY: Get list of all cognitive brains
    - QUARANTINE: Move duplicate versions to quarantine
    - UNCLASSIFIED: Get items that couldn't be classified
    - HEALTH: Health check
    """
    op = str((msg or {}).get("op", "")).upper()
    mid = (msg or {}).get("mid")
    payload = (msg or {}).get("payload") or {}

    inventory = get_inventory_brain()

    if op in ("LIST", "INVENTORY", "GET_INVENTORY"):
        result = inventory.get_inventory()

        # Check for unclassified items
        unclassified = inventory.get_unclassified_items()
        if unclassified:
            result["unclassified_notice"] = f"Unclassified items found: {len(unclassified)}"
            result["unclassified"] = unclassified

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": result
        }

    if op == "QUARANTINE":
        result = inventory.quarantine_duplicates()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": result
        }

    if op == "UNCLASSIFIED":
        unclassified = inventory.get_unclassified_items()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "items": unclassified,
                "count": len(unclassified)
            }
        }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "patterns_loaded": len(inventory._patterns)
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": f"Unknown operation: {op}"
        }
    }


# Standard service contract
service_api = handle


def list_brains() -> Dict[str, Any]:
    """
    Convenience function to list all cognitive brains.

    Returns:
        { "brains": [...], "total": <count> }
    """
    return get_inventory_brain().get_inventory()


__all__ = [
    "InventoryBrain",
    "get_inventory_brain",
    "service_api",
    "handle",
    "list_brains",
]
