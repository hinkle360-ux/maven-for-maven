"""Shared helpers for domain bank services."""

from __future__ import annotations

from typing import Any, Dict

from brains.memory.brain_memory import BrainMemory
from brains.memory.tier_manager import TierManager


def bank_service_factory(bank_name: str):
    memory = BrainMemory(bank_name, brain_category="domain")
    tier_manager = TierManager(bank_name)

    def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
        op = msg.get("op")
        payload = msg.get("payload", {}) or {}

        try:
            if op == "STORE":
                # Per spec: all writes go to STM only, tier spill is automatic
                record = memory.store(
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                )
                return {"ok": True, "op": op, "payload": record}

            if op == "RETRIEVE":
                results = memory.retrieve(
                    query=payload.get("query"),
                    limit=payload.get("limit", 10),
                    tiers=payload.get("tiers"),
                )
                return {"ok": True, "op": op, "payload": {"results": results}}

            if op == "COUNT":
                counts = tier_manager.get_tier_counts()
                return {"ok": True, "op": op, "payload": counts}

            if op in {"REBUILD_INDEX", "COMPACT_ARCHIVE"}:  # stubs for compatibility
                return {"ok": True, "op": op, "payload": {"status": "noop"}}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": str(exc), "op": op}

        return {"ok": False, "error": "Unsupported operation", "op": op}

    return handle


__all__ = ["bank_service_factory"]
