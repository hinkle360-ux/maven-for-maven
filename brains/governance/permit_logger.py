"""
Permit Logger
==============

This module defines a lightweight logging helper for governance
permits.  Whenever a permit is requested and issued (or denied), a
record should be appended to a JSONL ledger under
``reports/governance/permit_ledger.jsonl``.  Each record contains
the timestamp, action name, permit ID, whether the request was
allowed and an optional reason.  Logging failures are silently
ignored so as not to disrupt the main control flow.

The logger is deliberately simple and uses no external dependencies.
The log file grows linearly; downstream tools may truncate or
archive it as needed.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from brains.maven_paths import get_reports_path


def log_permit(action: str, permit_id: Optional[str], allowed: bool, reason: Optional[str] = None) -> None:
    """Append a permit record to the governance ledger.

    This helper writes permit events to a persistent JSONL file named
    ``permits.jsonl`` under ``reports/governance``.  Each record
    contains the timestamp, action, permit ID (which may be None), an
    allowed flag, and an optional reason.  If writing fails, the error
    is silently ignored to avoid impacting normal operation.

    Args:
        action: The name of the governance action for which the permit
            was requested (e.g. "CRITIQUE", "OPINION").
        permit_id: The unique identifier issued by the governance
            engine.  May be None when a request fails.
        allowed: True if the permit request was granted; False if it was
            denied.
        reason: Optional descriptive reason returned by the permit
            engine explaining why the request was allowed or denied.
    """
    try:
        log_dir = get_reports_path("governance")
        log_dir.mkdir(parents=True, exist_ok=True)
        # Write to the new permits ledger instead of the legacy permit_ledger
        ledger_path = log_dir / "permits.jsonl"
        entry = {
            "ts": int(time.time()),
            "action": str(action),
            "permit_id": permit_id,
            "allowed": bool(allowed),
            "reason": reason,
        }
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        # Silently ignore any logging failures
        pass