"""Local episodic memory utilities.

Episodes are stored in ``reports/episodic_memory.jsonl`` with basic metadata
and optional TTL handling.  This implementation mirrors the public API that
callers expect from the previous memory_system module.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from brains.maven_paths import get_reports_path

EPISODE_PATH = get_reports_path("episodic_memory.jsonl")


def _read_all() -> List[Dict[str, Any]]:
    if not EPISODE_PATH.exists():
        EPISODE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EPISODE_PATH.write_text("")
        return []

    records: List[Dict[str, Any]] = []
    for line in EPISODE_PATH.read_text().splitlines():
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


def _write_all(records: List[Dict[str, Any]]) -> None:
    serialized = "\n".join(json.dumps(rec) for rec in records)
    if serialized:
        serialized += "\n"
    EPISODE_PATH.write_text(serialized)


def store_episode(question: str, answer: str, confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
    metadata = metadata or {}
    record = {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "metadata": metadata,
        "timestamp": time.time(),
    }
    if ttl_seconds is not None:
        record["ttl_seconds"] = ttl_seconds

    records = _read_all()
    records.append(record)
    _write_all(records)
    return record


def get_episodes(limit: Optional[int] = None, include_expired: bool = False) -> List[Dict[str, Any]]:
    now = time.time()
    records = _read_all()
    records.sort(key=lambda r: r.get("timestamp", 0), reverse=True)

    result: List[Dict[str, Any]] = []
    for rec in records:
        ttl = rec.get("ttl_seconds")
        if not include_expired and ttl is not None:
            if rec.get("timestamp", 0) + ttl < now:
                continue
        result.append(rec)
        if limit is not None and len(result) >= limit:
            break
    return result


def summarize_episodes(limit: int = 5) -> str:
    episodes = get_episodes(limit=limit)
    lines = []
    for ep in episodes:
        question = ep.get("question", "")
        answer = ep.get("answer", "")
        confidence = ep.get("confidence", 0.0)
        lines.append(f"Q: {question} | A: {answer} | conf={confidence:.2f}")
    return "\n".join(lines)


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = msg.get("op")
    payload = msg.get("payload", {}) or {}

    try:
        if op == "EPISODE_STORE":
            record = store_episode(
                question=payload.get("question", ""),
                answer=payload.get("answer", ""),
                confidence=float(payload.get("confidence", 1.0)),
                metadata=payload.get("metadata", {}),
                ttl_seconds=payload.get("ttl_seconds"),
            )
            return {"ok": True, "op": op, "payload": record}

        if op == "EPISODE_GET":
            records = get_episodes(limit=payload.get("limit"))
            return {"ok": True, "op": op, "payload": {"episodes": records}}

        if op == "EPISODE_SUMMARY":
            summary = summarize_episodes(limit=payload.get("limit", 5))
            return {"ok": True, "op": op, "payload": {"summary": summary}}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc), "op": op}

    return {"ok": False, "error": "Unsupported operation", "op": op}


__all__ = [
    "EPISODE_PATH",
    "get_episodes",
    "service_api",
    "store_episode",
    "summarize_episodes",
]
