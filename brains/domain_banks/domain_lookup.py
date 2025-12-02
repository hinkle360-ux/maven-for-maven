"""
Domain Bank Lookup Module

Provides read-only, deterministic access to domain bank knowledge.
All lookups are based on exact matches (ID, tag, bank+kind).
No fuzzy matching, no heuristics, no randomness.

This module enables brains to query seeded domain knowledge.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from brains.maven_paths import (
    get_runtime_domain_banks_root,
    validate_path_confinement,
)

# Default runtime directory - must be inside maven2_fix
_DEFAULT_RUNTIME_DIR = str(get_runtime_domain_banks_root())


class DomainLookup:
    """
    Read-only interface for querying domain bank knowledge.

    All operations are deterministic and based on exact matches.
    """

    def __init__(self, runtime_dir: str):
        """
        Initialize domain lookup.

        Args:
            runtime_dir: Path to runtime domain banks directory
        """
        self.runtime_dir = validate_path_confinement(
            Path(runtime_dir), "domain lookup runtime"
        )
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._id_index: Dict[str, Dict[str, Any]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._loaded = False

    def _load_bank(self, bank_name: str) -> List[Dict[str, Any]]:
        """
        Load entries from a specific bank's LTM storage.

        Args:
            bank_name: Name of the bank to load

        Returns:
            List of entries from the bank
        """
        if bank_name in self._cache:
            return self._cache[bank_name]

        entries = []
        bank_file = self.runtime_dir / bank_name / "memory" / "ltm" / "facts.jsonl"

        if not bank_file.exists():
            self._cache[bank_name] = []
            return []

        try:
            with open(bank_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)

                        # Build indices
                        entry_id = entry.get("id")
                        if entry_id:
                            self._id_index[entry_id] = entry

                        tags = entry.get("content", {}).get("tags", [])
                        for tag in tags:
                            if tag not in self._tag_index:
                                self._tag_index[tag] = set()
                            self._tag_index[tag].add(entry_id)

                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        except Exception:
            pass  # Return empty list if file cannot be read

        self._cache[bank_name] = entries
        return entries

    def _ensure_loaded(self, bank_name: str) -> None:
        """
        Ensure a bank is loaded.

        Args:
            bank_name: Bank to load
        """
        if bank_name not in self._cache:
            self._load_bank(bank_name)

    def get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entry by its unique ID.

        Args:
            entry_id: Entry ID in format {bank}:{kind}:{slug}

        Returns:
            Entry dict if found, None otherwise
        """
        # Extract bank from ID
        parts = entry_id.split(":")
        if len(parts) < 1:
            return None

        bank_name = parts[0]
        self._ensure_loaded(bank_name)

        return self._id_index.get(entry_id)

    def get_by_tag(self, tag: str, bank: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all entries with a specific tag.

        Args:
            tag: Tag to search for
            bank: Optional bank name to restrict search

        Returns:
            List of matching entries (sorted by ID for determinism)
        """
        # If bank specified, only load that bank
        if bank:
            self._ensure_loaded(bank)
        else:
            # Load all known banks (deterministic order)
            known_banks = [
                "science", "technology", "language_arts", "working_theories",
                "personal", "governance_rules", "coding_patterns",
                "planning_patterns", "creative_templates", "environment_rules",
                "conflict_resolution_patterns"
            ]
            for b in known_banks:
                self._ensure_loaded(b)

        # Get entry IDs for this tag
        entry_ids = self._tag_index.get(tag, set())

        # Filter by bank if specified
        if bank:
            entry_ids = {eid for eid in entry_ids if eid.startswith(bank + ":")}

        # Get entries and sort by ID
        entries = [self._id_index[eid] for eid in entry_ids if eid in self._id_index]
        return sorted(entries, key=lambda e: e.get("id", ""))

    def get_by_bank_and_kind(self, bank: str, kind: str) -> List[Dict[str, Any]]:
        """
        Get all entries of a specific kind from a specific bank.

        Args:
            bank: Bank name
            kind: Entry kind (e.g., "rule", "pattern", "concept")

        Returns:
            List of matching entries (sorted by ID for determinism)
        """
        self._ensure_loaded(bank)

        entries = self._cache.get(bank, [])
        matches = [e for e in entries if e.get("kind") == kind]
        return sorted(matches, key=lambda e: e.get("id", ""))

    def get_all_from_bank(self, bank: str) -> List[Dict[str, Any]]:
        """
        Get all entries from a specific bank.

        Args:
            bank: Bank name

        Returns:
            List of all entries (sorted by ID for determinism)
        """
        self._ensure_loaded(bank)
        entries = self._cache.get(bank, [])
        return sorted(entries, key=lambda e: e.get("id", ""))

    def search_by_title(self, title_substring: str, bank: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search entries by title substring (case-insensitive exact substring match).

        Args:
            title_substring: Substring to search for in titles
            bank: Optional bank name to restrict search

        Returns:
            List of matching entries (sorted by ID for determinism)
        """
        title_lower = title_substring.lower()

        if bank:
            self._ensure_loaded(bank)
            entries = self._cache.get(bank, [])
        else:
            # Load all banks
            known_banks = [
                "science", "technology", "language_arts", "working_theories",
                "personal", "governance_rules", "coding_patterns",
                "planning_patterns", "creative_templates", "environment_rules",
                "conflict_resolution_patterns"
            ]
            entries = []
            for b in known_banks:
                self._ensure_loaded(b)
                entries.extend(self._cache.get(b, []))

        matches = []
        for entry in entries:
            title = entry.get("content", {}).get("title", "")
            if title_lower in title.lower():
                matches.append(entry)

        return sorted(matches, key=lambda e: e.get("id", ""))

    def get_related_entries(self, entry_id: str) -> List[Dict[str, Any]]:
        """
        Get entries that are related to the given entry.

        Args:
            entry_id: Entry ID to find relations for

        Returns:
            List of related entries (sorted by ID for determinism)
        """
        entry = self.get_by_id(entry_id)
        if not entry:
            return []

        related_ids = entry.get("content", {}).get("related_ids", [])
        related = []

        for rid in related_ids:
            related_entry = self.get_by_id(rid)
            if related_entry:
                related.append(related_entry)

        return sorted(related, key=lambda e: e.get("id", ""))

    def clear_cache(self) -> None:
        """Clear the internal cache (useful after seeding)."""
        self._cache.clear()
        self._id_index.clear()
        self._tag_index.clear()


# Global instance for convenient access
_global_lookup: Optional[DomainLookup] = None


def get_global_lookup(runtime_dir: str = _DEFAULT_RUNTIME_DIR) -> DomainLookup:
    """
    Get the global DomainLookup instance.

    Args:
        runtime_dir: Runtime directory path

    Returns:
        Global DomainLookup instance
    """
    global _global_lookup
    if _global_lookup is None:
        _global_lookup = DomainLookup(runtime_dir)
    return _global_lookup


def lookup_by_id(entry_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to lookup by ID using global instance."""
    return get_global_lookup().get_by_id(entry_id)


def lookup_by_tag(tag: str, bank: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to lookup by tag using global instance."""
    return get_global_lookup().get_by_tag(tag, bank)


def lookup_by_bank_and_kind(bank: str, kind: str) -> List[Dict[str, Any]]:
    """Convenience function to lookup by bank and kind using global instance."""
    return get_global_lookup().get_by_bank_and_kind(bank, kind)


if __name__ == "__main__":
    # Simple test
    import sys

    lookup = DomainLookup(_DEFAULT_RUNTIME_DIR)

    # Test getting governance rules
    rules = lookup.get_by_bank_and_kind("governance_rules", "rule")
    print(f"Found {len(rules)} governance rules")
    for rule in rules:
        print(f"  - {rule.get('id')}: {rule.get('content', {}).get('title')}")

    print()

    # Test tag lookup
    determinism_entries = lookup.get_by_tag("determinism")
    print(f"Found {len(determinism_entries)} entries tagged 'determinism'")
    for entry in determinism_entries[:5]:  # Show first 5
        print(f"  - {entry.get('id')}")
