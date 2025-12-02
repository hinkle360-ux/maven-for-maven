from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

"""
Self‑DMN Judge
==============

The Self‑DMN judge examines flagged claim records produced by DISSENT_SCAN and
determines whether each claim requires recomputation.  A claim warrants
recomputation when its status is 'recompute' or 'disputed' or when the
difference between the skeptic and consensus scores exceeds TAU1.  Otherwise,
the claim is ignored.  Results are returned as a list of actions, and may
optionally be persisted to a repairs log for governance tracking.
"""

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import error_response  # type: ignore
    from api.utils import success_response  # type: ignore
    from api.utils import generate_mid  # type: ignore
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}
    # Health check simply reports operational status
    if op == "HEALTH":
        return success_response(op, mid, {"status": "operational"})
    # Adjudicate flagged claims for recomputation
    if op == "ADJUDICATE":
        # Gather claims either from the provided payload or from the audit log
        claims_input = payload.get("claims")
        claims: List[Dict[str, Any]] = []
        if claims_input:
            # ensure we only process dicts
            claims = [rec for rec in claims_input if isinstance(rec, dict)]
        else:
            # fallback to reading audit.jsonl for the most recent entries
            try:
                project_root = Path(__file__).resolve().parents[4]
                audit_path = project_root / "reports" / "self_dmn" / "audit.jsonl"
                if audit_path.exists():
                    with open(audit_path, "r", encoding="utf-8") as fh:
                        for ln in fh:
                            try:
                                obj = json.loads(ln.strip())
                            except Exception:
                                continue
                            if isinstance(obj, dict):
                                claims.append(obj)
            except Exception:
                claims = []
        decisions: List[Dict[str, Any]] = []
        # Determine threshold from skeptic module (TAU1) or default
        try:
            import importlib
            _skeptic_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_skeptic")
            TAU1 = getattr(_skeptic_mod, "TAU1", 0.25)
        except Exception:
            TAU1 = 0.25
        for rec in claims[-50:]:
            try:
                cid = rec.get("claim_id")
                status = str(rec.get("status") or "")
                consensus = float(rec.get("consensus_score", 0.0) or 0.0)
                skeptic = float(rec.get("skeptic_score", 0.0) or 0.0)
            except Exception:
                continue
            # default: ignore
            action = "IGNORE"
            reason = f"status={status or 'undisputed'}"
            if status in {"recompute", "disputed"} or (skeptic - consensus) >= TAU1:
                action = "RECOMPUTE"
                if (skeptic - consensus) >= TAU1:
                    reason = f"skeptic {skeptic:.2f} - consensus {consensus:.2f} >= TAU1 {TAU1}"
                else:
                    reason = f"status={status}"
            decisions.append({"claim_id": cid, "action": action, "reason": reason, "target": "reasoning"})
        # Persist decisions to a repairs log for governance tracking
        if decisions:
            try:
                import random
                project_root = Path(__file__).resolve().parents[4]
                repairs_dir = project_root / "reports" / "governance" / "repairs"
                repairs_dir.mkdir(parents=True, exist_ok=True)
                fname = repairs_dir / f"repairs_{random.randint(100000, 999999)}.jsonl"
                with open(fname, "a", encoding="utf-8") as f:
                    for dec in decisions:
                        f.write(json.dumps(dec) + "\n")
            except Exception:
                pass
        return success_response(op, mid, {"decisions": decisions})
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