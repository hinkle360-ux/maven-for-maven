#!/usr/bin/env python3
"""
Cleanup Bad Facts Script
========================

This script scans all brain memory files and removes facts that contain:
1. Claims that Maven is "a large language model"
2. False capability claims (e.g., "I cannot access the current time")
3. Other identity lies

Run this script after deploying the time tool fix to clean up
any bad facts that were learned before the fix was in place.

Usage:
    python cleanup_bad_facts.py [--dry-run]
"""

import json
import sys
from pathlib import Path


# Bad patterns to look for (facts containing these should be removed)
BAD_PATTERNS = [
    # Identity lies
    "large language model",
    "i am an llm",
    "i'm an llm",
    "i am chatgpt",
    "i am gpt",
    "i am claude",

    # Time capability lies
    "i do not have the ability to query external time",
    "i cannot access the current time",
    "i don't have access to the current time",
    "i do not have access to real-time",
    "i cannot tell the time",
    "i don't know what time it is",
    "i am unable to access the time",
    "i cannot access a clock",
    "no ability to query external time",
    "cannot provide real-time information",

    # Web search lies
    "i cannot search the web",
    "i cannot browse the internet",
    "i do not have internet access",
    "i don't have access to the internet",
    "i cannot access external websites",
    "i am unable to browse",

    # Filesystem lies
    "i cannot read files",
    "i cannot access files",
    "i do not have access to your filesystem",
    "i cannot read or write files",
    "i am unable to access files",

    # Code execution lies
    "i cannot execute code",
    "i cannot run code",
    "i do not have the ability to run code",

    # Git lies
    "i cannot access git",
    "i do not have git access",
    "i cannot run git commands",

    # Browser lies
    "i cannot control a browser",
    "i do not have browser access",
    "i cannot automate a browser",
]

# Concept keys that indicate bad learned facts
BAD_CONCEPT_KEYS = [
    "time it",
    "time is it",
    "current time",
    "the time",
]


def is_bad_fact(record: dict) -> tuple[bool, str]:
    """
    Check if a record contains a bad fact that should be removed.

    Returns:
        Tuple of (is_bad, reason)
    """
    content = str(record.get("content", "")).lower()
    metadata = record.get("metadata", {}) or {}

    # Check content for bad patterns
    for pattern in BAD_PATTERNS:
        if pattern in content:
            return True, f"content contains '{pattern}'"

    # Check concept_key for bad patterns
    concept_key = str(metadata.get("concept_key", "")).lower()
    for bad_key in BAD_CONCEPT_KEYS:
        if bad_key == concept_key:
            # Check if it's a capability lie fact (not a legitimate stored answer)
            if any(p in content for p in ["cannot", "unable", "do not have"]):
                return True, f"concept_key '{concept_key}' with capability lie"

    return False, ""


def scan_and_clean_jsonl(file_path: Path, dry_run: bool = True) -> tuple[int, int]:
    """
    Scan a JSONL file and remove bad facts.

    Returns:
        Tuple of (total_records, removed_count)
    """
    if not file_path.exists():
        return 0, 0

    try:
        lines = file_path.read_text().splitlines()
    except Exception as e:
        print(f"  [ERROR] Cannot read {file_path}: {e}")
        return 0, 0

    clean_records = []
    removed_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                clean_records.append(record)
                continue

            is_bad, reason = is_bad_fact(record)
            if is_bad:
                removed_count += 1
                content_preview = str(record.get("content", ""))[:60]
                print(f"  [REMOVE] {reason}: {content_preview}...")
            else:
                clean_records.append(record)
        except json.JSONDecodeError:
            # Keep unparseable lines as-is
            continue

    total = len(lines)

    if not dry_run and removed_count > 0:
        # Write clean records back
        serialized = "\n".join(json.dumps(rec) for rec in clean_records)
        if serialized:
            serialized += "\n"
        file_path.write_text(serialized)
        print(f"  [SAVED] Removed {removed_count} bad facts from {file_path.name}")

    return total, removed_count


def find_all_memory_files(base_path: Path) -> list[Path]:
    """Find all JSONL memory files."""
    memory_files = []

    # Look for records.jsonl files in brain memory directories
    for jsonl_file in base_path.rglob("*.jsonl"):
        # Skip seed files (those are static)
        if "seeds" in str(jsonl_file):
            continue
        memory_files.append(jsonl_file)

    # Also check for any .json files that might contain facts
    for json_file in base_path.rglob("*.json"):
        if "pattern_store" in str(json_file) or "memory" in str(json_file.parent.name):
            memory_files.append(json_file)

    return memory_files


def clean_pattern_store(file_path: Path, dry_run: bool = True) -> tuple[int, int]:
    """Clean the pattern store JSON file."""
    if not file_path.exists():
        return 0, 0

    try:
        data = json.loads(file_path.read_text())
    except Exception as e:
        print(f"  [ERROR] Cannot read {file_path}: {e}")
        return 0, 0

    if not isinstance(data, dict):
        return 0, 0

    patterns = data.get("patterns", [])
    if not isinstance(patterns, list):
        return 0, 0

    clean_patterns = []
    removed_count = 0

    for pattern in patterns:
        if not isinstance(pattern, dict):
            clean_patterns.append(pattern)
            continue

        # Check pattern content and action
        action = pattern.get("action", {}) or {}
        content_parts = [
            str(pattern.get("signature", "")),
            str(action),
        ]
        content = " ".join(content_parts).lower()

        is_bad = any(bp in content for bp in BAD_PATTERNS)

        if is_bad:
            removed_count += 1
            print(f"  [REMOVE] Pattern with bad content: {pattern.get('signature', '')}...")
        else:
            clean_patterns.append(pattern)

    if not dry_run and removed_count > 0:
        data["patterns"] = clean_patterns
        file_path.write_text(json.dumps(data, indent=2))
        print(f"  [SAVED] Removed {removed_count} bad patterns from {file_path.name}")

    return len(patterns), removed_count


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be modified")
        print("Run without --dry-run to actually clean files")
        print("=" * 60)
    else:
        print("=" * 60)
        print("CLEANING MODE - Files will be modified")
        print("=" * 60)

    # Find the project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent

    print(f"\nScanning project at: {project_root}")
    print()

    # Find all memory files
    memory_files = find_all_memory_files(project_root / "brains")

    total_scanned = 0
    total_removed = 0

    for file_path in memory_files:
        rel_path = file_path.relative_to(project_root)
        print(f"Scanning: {rel_path}")

        if file_path.suffix == ".json" and "pattern" in str(file_path):
            scanned, removed = clean_pattern_store(file_path, dry_run)
        else:
            scanned, removed = scan_and_clean_jsonl(file_path, dry_run)

        total_scanned += scanned
        total_removed += removed

    print()
    print("=" * 60)
    print(f"SUMMARY: Scanned {total_scanned} records, found {total_removed} bad facts")

    if dry_run and total_removed > 0:
        print("\nTo actually remove these facts, run:")
        print("  python cleanup_bad_facts.py")
    elif total_removed > 0:
        print(f"\nRemoved {total_removed} bad facts from memory")
    else:
        print("\nNo bad facts found - memory is clean!")
    print("=" * 60)


if __name__ == "__main__":
    main()
