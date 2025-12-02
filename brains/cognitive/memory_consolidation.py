"""
Memory Consolidation
====================

This module implements a basic mechanism for consolidating short‑term
memories (STM) into mid‑term (MTM) or long‑term (LTM) stores.  It is
designed to prevent unbounded growth of STM by periodically moving
records based on their importance.  The current implementation
assumes a simple JSONL file structure for each memory tier and uses
importance scores attached to each record.  Records without an
explicit importance field are assumed to be of low importance and
archived to cold storage.

The consolidation process can be invoked manually or scheduled via
the agent's budget manager.  It scans all cognitive brains with
memory directories under ``brains/cognitive`` and processes their
``stm/records.jsonl`` files.  Errors are swallowed to avoid
disruptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from brains.maven_paths import get_brains_path

# -----------------------------------------------------------------------------
# Importance computation
#
# The consolidation logic relies on an ``importance`` score attached to
# each record to decide whether it should be promoted to mid‑ or long‑term
# memory.  Earlier versions of Maven left this helper undefined, leading
# to errors when the consolidator attempted to call ``compute_importance``.
#
# The function below provides a default implementation.  Records may
# optionally include an explicit ``importance`` field; when present, it
# is respected.  Otherwise, a simple heuristic assigns high importance
# to self‑definitional facts about Maven (e.g. core identity, purpose
# or notes from the creators) and medium importance to all other
# memories.  Additional logic can be added here to weight other types
# of knowledge differently.

def compute_importance(item: Dict[str, Any]) -> float:
    """Return an importance score for a memory record.

    Args:
        item: The memory record under consideration.
    Returns:
        A float between 0 and 1 indicating importance.  Values above
        0.8 cause promotion directly to long‑term memory; values above
        0.5 promote to mid‑term.  A default of 0.6 ensures that most
        facts eventually graduate from STM.
    """
    try:
        # Honour an explicit importance field if provided
        if item and isinstance(item, dict) and "importance" in item:
            try:
                val = float(item.get("importance") or 0.0)
                # Clamp to [0,1]
                if val < 0.0:
                    return 0.0
                if val > 1.0:
                    return 1.0
                return val
            except Exception:
                pass
        # Inspect the content and source for self‑identity markers
        content = str(item.get("content", "") or "").lower()
        source = str(item.get("source", "") or "").lower()
        # Keywords associated with Maven's identity and creator notes
        if (
            "maven" in content and (
                "purpose" in content
                or "cognitive" in content
                or "living" in content
                or "core" in content
                or "creator" in content
                or "architecture" in content
            )
        ) or (
            "personal_maven_card" in source
            or "core_identity" in source
            or "maven card" in source
        ):
            return 1.0
        # Fallback default importance for general facts
        return 0.6
    except Exception:
        # Conservative default on error
        return 0.6


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from a file.

    Args:
        path: The path to the JSONL file.
    Returns:
        A list of record dictionaries.  Returns an empty list on error.
    """
    recs: List[Dict[str, Any]] = []
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip() or "{}")
                        if isinstance(obj, dict):
                            recs.append(obj)
                    except Exception:
                        continue
    except Exception:
        return []
    return recs


def _write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write a list of records to a JSONL file, overwriting existing content.

    This helper uses the ``atomic_jsonl_write`` utility from ``api.utils``
    when available to ensure that writes are crash‑safe.  When the helper
    cannot be imported, it falls back to a naive write which may be
    susceptible to partial writes.  The function silently ignores
    serialization errors on individual records to avoid corrupting
    existing files.

    Args:
        path: The file path to write to.
        records: The list of record dictionaries.
    """
    # Import atomic write helper to ensure crash‑safe writes
    try:
        from api.utils import atomic_jsonl_write  # type: ignore
    except Exception:
        atomic_jsonl_write = None  # type: ignore

    try:
        # Use atomic JSONL writer if available.  This performs the
        # entire rewrite in a temporary file and swaps it into place.
        if atomic_jsonl_write:
            # Filter out non‑dict items to avoid serialization errors
            filtered = [rec for rec in records if isinstance(rec, dict)]
            atomic_jsonl_write(path, filtered)  # type: ignore
            return
    except Exception:
        # Fall back to naive write when atomic write fails
        pass
    # Naive write: overwrite the file line by line
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                try:
                    fh.write(json.dumps(rec) + "\n")
                except Exception:
                    continue
    except Exception:
        pass


def should_consolidate(item: Dict[str, Any]) -> bool:
    """Determine whether a memory record should be consolidated.

    Note: Time-based TTL checks have been removed. This function now
    always returns False to prevent automatic consolidation.
    Consolidation should be triggered manually or through other
    mechanisms that don't rely on time-based logic.

    Args:
        item: The memory record under consideration.

    Returns:
        Always False (no automatic consolidation).
    """
    # Time-based consolidation disabled
    return False


def consolidate_memories() -> None:
    from api.utils import CFG  # type: ignore
    """Perform memory consolidation across cognitive brains.

    Iterates through each brain under ``brains/cognitive`` that
    contains a memory directory.  For each STM record, decide
    whether to move it to MTM, LTM or cold storage based on its
    importance score.  STM files are truncated after
    consolidation.  After moving records, enforce per‑tier record
    limits defined in the memory configuration by moving the
    oldest entries to the next tier when necessary.

    Note: Time-based TTL logic has been removed from this function.
    """
    # Determine the cognitive brains directory under the Maven root.
    cognitive_dir = get_brains_path("cognitive")
    for brain_path in cognitive_dir.iterdir():
        mem_dir = brain_path / "memory"
        if not mem_dir.exists():
            continue
        stm_path = mem_dir / "stm" / "records.jsonl"
        mtm_path = mem_dir / "mtm" / "records.jsonl"
        ltm_path = mem_dir / "ltm" / "records.jsonl"
        cold_path = mem_dir / "cold" / "records.jsonl"
        stm_records = _load_records(stm_path)
        if not stm_records:
            continue
        # Load existing MTM and LTM records
        mtm_records = _load_records(mtm_path)
        ltm_records = _load_records(ltm_path)
        cold_records = _load_records(cold_path)
        for item in stm_records:
            # Evaluate importance early.  High‑importance records are
            # promoted regardless of age.  This allows significant
            # knowledge to mature into MTM/LTM immediately.
            try:
                importance = compute_importance(item)
            except Exception:
                importance = 0.0
            # Default to moderate importance when no value is available
            if importance is None or importance <= 0.0:
                importance = 0.6
            # Promote to LTM when importance is very high
            if importance > 0.8:
                ltm_records.append(item)
                continue
            # Promote to MTM when importance is moderate (>0.5)
            if importance > 0.5:
                mtm_records.append(item)
                continue
            # For lower importance, fall back to TTL‑based consolidation
            # Note: should_consolidate now always returns False
            if not should_consolidate(item):
                continue
            # At this point the item is aged and of low importance
            cold_records.append(item)
        # Rewrite STM with only non‑consolidated records
        try:
            remaining = []
            for item in stm_records:
                if not should_consolidate(item):
                    remaining.append(item)
            _write_records(stm_path, remaining)
        except Exception:
            pass

        # Note: Age-based promotion from MTM → LTM and LTM → cold has been
        # disabled (time-based TTL logic removed). Records remain in their
        # current tier unless promoted via importance scoring.

        # Persist updated MTM, LTM and cold storage
        _write_records(mtm_path, mtm_records)
        _write_records(ltm_path, ltm_records)
        _write_records(cold_path, cold_records)
        # Enforce per‑tier record limits when configured.  Oldest
        # entries are moved to the next tier until limits are met.
        try:

            mem_cfg = (CFG.get("memory") or {})
            limits = {
                "stm": int(mem_cfg.get("stm_max_records", 0) or 0),
                "mtm": int(mem_cfg.get("mtm_max_records", 0) or 0),
                "ltm": int(mem_cfg.get("ltm_max_records", 0) or 0),
            }
            # Helper to trim a tier by moving excess oldest entries to next tier
            def _trim_tier(from_tier: str, to_tier: str, path_from: Path, path_to: Path, limit: int) -> None:
                if limit <= 0:
                    return
                recs = _load_records(path_from)
                if len(recs) <= limit:
                    return
                # Determine how many to move
                excess = len(recs) - limit
                move = recs[:excess]
                remain = recs[excess:]
                # Append moved records to destination tier
                to_recs = _load_records(path_to)
                to_recs.extend(move)
                _write_records(path_to, to_recs)
                # Rewrite from tier with remaining
                _write_records(path_from, remain)
            # Trim STM → MTM, MTM → LTM, LTM → cold
            stm_limit = limits.get("stm", 0)
            if stm_limit > 0:
                _trim_tier("stm", "mtm", stm_path, mtm_path, stm_limit)
            mtm_limit = limits.get("mtm", 0)
            if mtm_limit > 0:
                _trim_tier("mtm", "ltm", mtm_path, ltm_path, mtm_limit)
            ltm_limit = limits.get("ltm", 0)
            if ltm_limit > 0:
                # Use cold storage for overflow beyond LTM
                cold_limit = limits.get("cold", 0)
                _trim_tier("ltm", "cold", ltm_path, cold_path, ltm_limit)
        except Exception:
            pass
