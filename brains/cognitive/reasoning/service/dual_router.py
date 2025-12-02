"""
Dual‑process Router
===================

This module exposes a wrapper around the existing learned router that
implements a simple dual‑process strategy.  For most inputs the fast
path (System 1) will suffice and the underlying ``learned_router``
service is invoked directly.  When the confidence of the top routing
choice is close to the next best candidate (indicating low margin),
the router flags the request as a slow‑path (System 2) operation.  A
caller can then choose to perform additional analysis or re‑routing.

Only stdlib modules are used and no external dependencies are added.
The interface mirrors ``learned_router.service_api`` and forwards
unsupported operations unchanged.
"""

from __future__ import annotations

from typing import Dict, Any

# Import the existing learned router.  If the import fails the wrapper
# will gracefully report an error for unsupported operations.
try:
    from .learned_router import service_api as _learned_router_api  # type: ignore
except Exception:
    _learned_router_api = None


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch dual‑router operations.  All operations supported by the
    underlying learned router (``LEARN``, ``LEARN_DEFINITION``, ``ROUTE``
    and ``HEALTH``) are forwarded directly.  For ``ROUTE`` the result
    payload is augmented with a ``slow_path`` boolean indicating whether
    the score margin between the best and second best route is below
    a threshold (currently 0.2).  A True value suggests that a more
    deliberate evaluation may yield a better routing decision.

    Unsupported operations result in a simple error response.
    """
    op = (msg or {}).get("op", "").upper()
    # Passthrough when the learned router is unavailable
    if _learned_router_api is None:
        return {"ok": False, "error": {"code": "MISSING_BACKEND", "message": "learned_router unavailable"}}
    # Directly forward non‑ROUTE operations
    if op in {"LEARN", "LEARN_DEFINITION", "HEALTH"}:
        return _learned_router_api(msg)
    if op == "ROUTE":
        res = _learned_router_api(msg)
        try:
            # Inspect score margin between top two banks
            scores = (res.get("payload") or {}).get("scores") or {}
            if isinstance(scores, dict) and scores:
                values = sorted(scores.values(), reverse=True)
                top = float(values[0])
                second = float(values[1]) if len(values) > 1 else 0.0
                margin = top - second
                # Load the margin threshold from the configuration.  This
                # value determines when the router should favour a slow
                # deliberate path (System 2) over the fast path (System 1).
                threshold = 0.2
                try:
                    from pathlib import Path
                    import json as _json
                    # Derive project root relative to this file
                    root = Path(__file__).resolve().parents[4]
                    cfg_path = root / "config" / "self_dmn_thresholds.json"
                    if cfg_path.exists():
                        with open(cfg_path, "r", encoding="utf-8") as fh:
                            cfg = _json.load(fh) or {}
                        t = cfg.get("margin_threshold")
                        # Validate the loaded threshold; fallback to default on error
                        try:
                            ft = float(t)
                            if 0.0 <= ft <= 1.0:
                                threshold = ft
                        except Exception:
                            pass
                except Exception:
                    # Ignore any configuration read errors and use default
                    pass
                slow = margin < threshold
                res.setdefault("payload", {})["slow_path"] = slow
        except Exception:
            # On any error just leave slow_path unset
            pass
        return res
    # Unsupported op
    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}

# Service API entry point
service_api = handle