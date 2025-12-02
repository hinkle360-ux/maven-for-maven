"""Minimal TierManager for memory introspection."""

from __future__ import annotations

from typing import Dict

from brains.memory.brain_memory import BrainMemory


class TierManager:
    """Expose simple tier counts for a brain."""

    def __init__(self, brain_id: str, brain_path=None, enforce_tier_governance: bool = True) -> None:
        self._memory = BrainMemory(brain_id, enforce_tier_governance=enforce_tier_governance)

    def get_tier_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for tier in ["stm", "mtm", "ltm", "archive"]:
            records = self._memory._load_records(tier)
            counts[tier] = len(records)
        return counts


__all__ = ["TierManager"]
