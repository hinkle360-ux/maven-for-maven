"""Lightweight BrainMemory implementation.

This module provides a minimal, local JSONL-backed memory store that mirrors
the interface used throughout the Maven brains.  It intentionally keeps the
logic simple: writes go to STM by default, spill to deeper tiers when capacity
is exceeded, and reads aggregate records across tiers in reverse
chronological order.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from brains.brain_roles import get_brain_role
from brains.maven_paths import get_brain_memory_path, validate_path_confinement


class BrainMemory:
    """Simple tiered memory backed by JSONL files."""

    def __init__(
        self,
        brain_id: str,
        brain_category: Optional[str] = None,
        stm_capacity: int = 100,
        mtm_capacity: int = 500,
        ltm_capacity: int = 2000,
        enforce_tier_governance: bool = True,
    ) -> None:
        self.brain_id = brain_id
        self.brain_category = brain_category or get_brain_role(brain_id)
        self.enforce_tier_governance = enforce_tier_governance

        self._capacities = {
            "stm": max(0, stm_capacity),
            "mtm": max(0, mtm_capacity),
            "ltm": max(0, ltm_capacity),
        }

        base_path = get_brain_memory_path(self.brain_id, self.brain_category)
        self._memory_root = validate_path_confinement(base_path, "brain memory root")
        self._memory_dir = self._memory_root / "memory"

        # Ensure tier directories exist
        for tier in ("stm", "mtm", "ltm", "archive"):
            self._ensure_tier_file(tier)

    def _tier_file(self, tier: str) -> Path:
        return self._memory_dir / tier / "records.jsonl"

    def _ensure_tier_file(self, tier: str) -> Path:
        path = self._tier_file(tier)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")
        return path

    def _load_records(self, tier: str) -> List[Dict[str, Any]]:
        path = self._ensure_tier_file(tier)
        records: List[Dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
        return records

    def _write_records(self, tier: str, records: Sequence[Dict[str, Any]]) -> None:
        path = self._ensure_tier_file(tier)
        serialized = "\n".join(json.dumps(rec) for rec in records)
        if serialized:
            serialized += "\n"
        path.write_text(serialized)

    def _matches(self, record: Dict[str, Any], query: Optional[str]) -> bool:
        if not query:
            return True

        query_lower = str(query).lower()
        content = str(record.get("content", "")).lower()
        metadata = record.get("metadata", {}) or {}

        if query_lower in content:
            return True

        for key, value in metadata.items():
            try:
                key_str = str(key).lower()
                val_str = str(value).lower()
            except Exception:
                continue
            if query_lower in key_str or query_lower in val_str:
                return True

        # Simple "field:value" matching
        if ":" in query_lower:
            q_field, _, q_val = query_lower.partition(":")
            if q_field and q_val:
                for key, value in metadata.items():
                    if str(key).lower() == q_field and q_val in str(value).lower():
                        return True
        return False

    def _spill_overflow(self, tier: str, records: List[Dict[str, Any]]) -> None:
        capacity = self._capacities.get(tier)
        if capacity is None or capacity <= 0:
            self._write_records(tier, records)
            return

        next_tier = {"stm": "mtm", "mtm": "ltm", "ltm": "archive"}.get(tier)
        while len(records) > capacity:
            spill = records.pop(0)
            if next_tier:
                spill["tier"] = next_tier
                next_records = self._load_records(next_tier)
                next_records.append(spill)
                self._write_records(next_tier, next_records)
                records = records  # keep reference for loop condition
            else:
                break
        self._write_records(tier, records)

    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store content in STM (the only public write entry point).

        Per spec: STM is the only write entry. Records spill FIFO to MTM->LTM->Archive
        as tiers exceed capacity. No direct writes to MTM/LTM/Archive are allowed.
        """
        metadata = metadata or {}
        record = {
            "id": metadata.get("id", str(uuid.uuid4())),
            "content": content,
            "metadata": metadata,
            "tier": "stm",
            "timestamp": time.time(),
        }

        records = self._load_records("stm")
        records.append(record)
        self._spill_overflow("stm", records)

        return {"ok": True, "tier": "stm", "record": record}

    def retrieve(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = 10,
        tiers: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        tiers = list(tiers) if tiers else ["stm", "mtm", "ltm", "archive"]
        collected: List[Dict[str, Any]] = []
        for tier in tiers:
            collected.extend(self._load_records(tier))

        collected.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        filtered = [rec for rec in collected if self._matches(rec, query)]

        if limit is not None:
            return filtered[:limit]
        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about stored memory records."""

        tier_counts: Dict[str, int] = {}
        total_records = 0

        for tier in ("stm", "mtm", "ltm", "archive"):
            count = len(self._load_records(tier))
            tier_counts[tier] = count
            total_records += count

        return {"total": total_records, "by_tier": tier_counts}


__all__ = ["BrainMemory"]
