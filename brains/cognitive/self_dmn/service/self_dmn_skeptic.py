"""
Self‑DMN Skeptic submodule.

This submodule implements the dissent logic previously handled by
the Self‑Default brain.  It evaluates claims based on consensus and
skeptic scores and assigns a status of ``undisputed``, ``disputed``
or ``recompute``.  Successful claim registrations are persisted both
to the STM memory tier and to a global claims log under
``reports/self_dmn/claims.jsonl``.

The thresholds ``TAU1`` and ``TAU2`` determine when a claim is deemed
to require recomputation or is marked as disputed:

    if (skeptic_score − consensus_score) ≥ TAU1 → "recompute"
    elif skeptic_score ≥ TAU2 → "disputed"
    else → "undisputed"

This module exposes a ``service_api`` entry point compatible with
other cognitive brains in Maven.  Supported operations:

    - HEALTH: return operational status and memory tier counts
    - REGISTER: register a new claim with consensus and skeptic scores

"""

from __future__ import annotations

import json
from typing import Dict, Any
from pathlib import Path


# Directory references
HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

# Status thresholds
#
# TAU1 governs when a claim should be recomputed.  When the skeptic score
# exceeds the consensus score by at least this amount, the claim is marked
# as ``recompute``.  A larger TAU1 reduces the number of recompute events,
# which helps prevent runaway repair loops.  Empirically a value of 0.50
# provides a good balance between catching real disagreements and avoiding
# excessive recompute actions.  See also Self‑Default brain for the same
# constants.
TAU1 = 0.50  # threshold for recompute: skeptic minus consensus

# TAU2 determines when a claim is simply disputed without recomputation.  The
# default of 0.60 remains appropriate and is intentionally left unchanged.
TAU2 = 0.60  # threshold for disputed

def _counts() -> Dict[str, int]:
    from api.memory import count_lines  # type: ignore
    from api.memory import ensure_dirs  # type: ignore
    """Return record counts per memory tier for this submodule."""
    try:
        t = ensure_dirs(BRAIN_ROOT)
        return {
            "stm": count_lines(t.get("stm")),
            "mtm": count_lines(t.get("mtm")),
            "ltm": count_lines(t.get("ltm")),
            "cold": count_lines(t.get("cold")),
        }
    except Exception:
        return {"stm": 0, "mtm": 0, "ltm": 0, "cold": 0}

def _status(consensus: float, skeptic: float) -> str:
    """Compute claim status based on consensus and skeptic scores."""
    if (skeptic - consensus) >= TAU1:
        return "recompute"
    elif skeptic >= TAU2:
        return "disputed"
    return "undisputed"

def _write_claim(record: Dict[str, Any]) -> None:
    from api.memory import append_jsonl  # type: ignore
    """
    Persist the claim record to the STM tier and to the global claims log.

    Args:
        record: Claim record dict to write.
    """
    # Write to brain STM memory
    try:
        tiers = ensure_dirs(BRAIN_ROOT)
        append_jsonl(tiers.get("stm"), record)
    except Exception:
        pass
    # Write to global claims log under self_dmn
    try:
        proj_root = Path(__file__).resolve().parents[4]
        rpt_dir = proj_root / "reports" / "self_dmn"
        rpt_dir.mkdir(parents=True, exist_ok=True)
        path = rpt_dir / "claims.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import error_response  # type: ignore
    from api.utils import success_response  # type: ignore
    from api.utils import generate_mid  # type: ignore
    """
    Entry point for Self‑DMN Skeptic operations.

    Supported operations:
        - HEALTH: return operational status and memory health
        - REGISTER: register a new claim with consensus and skeptic scores

    Args:
        msg: A dict containing at least an 'op' key and optional 'payload'.

    Returns:
        A success or error response dict.
    """
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    # Health check
    if op == "HEALTH":
        return success_response(op, mid, {"status": "operational", "memory_health": _counts()})

    # Register a new claim
    if op == "REGISTER":
        # Gather claim info
        import random
        cid = payload.get("claim_id") or f"CL-{random.randint(100000, 999999)}"
        proposition = str(payload.get("proposition") or "").strip()
        try:
            consensus = float(payload.get("consensus_score", 0.0))
        except Exception:
            consensus = 0.0
        try:
            skeptic = float(payload.get("skeptic_score", 0.0))
        except Exception:
            skeptic = 0.0
        status = _status(consensus, skeptic)
        expiry = payload.get("expiry")
        record = {
            "claim_id": cid,
            "proposition": proposition,
            "consensus_score": consensus,
            "skeptic_score": skeptic,
            "status": status,
            "expiry": expiry,
            # Track the number of times this claim has been automatically recomputed
            "recompute_count": 0,
        }
        _write_claim(record)
        return success_response(op, mid, {"claim": record})

    # Unsupported operations
    return error_response(op, mid, "UNSUPPORTED_OP", op)


def handle(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Brain contract entry point.

    Args:
        context: Standard brain context dict

    Returns:
        Result dict
    """
    return _handle_impl(context)


# Brain contract alias
service_api = handle