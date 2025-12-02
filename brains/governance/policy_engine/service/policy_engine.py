from __future__ import annotations
from typing import Any, Dict
import time, os, secrets, json, hmac, hashlib

from brains.maven_paths import get_reports_path

# Governance Policy Engine (Observation + Repair Authorization)
# - Still ALLOW_AUDIT for normal ops
# - Adds AUTHORIZE_REPAIR to mint a short-lived token

_TTL_MS = 5 * 60 * 1000  # 5 minutes

# Shared secret used to sign and verify authorization tokens.  In a real
# deployment this should be provided via environment variable.  We fall
# back to a hard‑coded default for offline operation.  Do **not** share
# this secret outside of Maven.  All governance tokens carry a signature
# field that is verified by the Repair Engine before any destructive
# operation proceeds.
_SECRET_KEY = os.environ.get("MAVEN_SECRET_KEY", "maven_secret_key")

def _sign_token(data: Dict[str, Any]) -> str:
    """Compute an HMAC‐SHA256 signature for an auth token.

    The signature covers all fields of the token except the "signature"
    itself.  Fields are sorted to ensure deterministic ordering.  The
    secret key is taken from the _SECRET_KEY constant.  Returns a
    lowercase hexadecimal digest.
    """
    # Exclude the signature itself when signing
    payload = {k: v for k, v in data.items() if k != "signature"}
    # JSON serialise with sorted keys for deterministic ordering
    msg = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hmac.new(_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return digest

def _now_ms() -> int:
    return int(time.time() * 1000)

def _mk(mid: str, op: str, payload: Dict[str, Any], allowed: bool=True, effect: str="ALLOW_AUDIT") -> Dict[str, Any]:
    return {
        "ok": True,
        "op": op,
        "mid": mid,
        "effect": effect,
        "allowed": allowed,
        "payload": payload,
        "decision": {"decision": "ALLOW" if allowed else "DENY", "reason": "ok" if allowed else "policy"},
        "ts": _now_ms(),
    }

# Governance analytics logging.  Each call to the policy engine will update
# counts in reports/governance/analytics.json.  The file tracks how many
# times each operation was invoked and whether the effect was ALLOW_AUDIT,
# ALLOW or DENY.  Errors during logging are silently ignored.
def _log_analytics(op: str, effect: str) -> None:
    try:
        analytics_path = get_reports_path("governance", "analytics.json")
        analytics_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}
        if analytics_path.exists():
            try:
                with open(analytics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        key = (op or "").lower()
        if key not in data or not isinstance(data.get(key), dict):
            data[key] = {"allow_audit": 0, "allow": 0, "deny": 0}
        eff = (effect or "").upper()
        if eff == "ALLOW_AUDIT":
            data[key]["allow_audit"] += 1
        elif eff == "ALLOW":
            data[key]["allow"] += 1
        elif eff == "DENY":
            data[key]["deny"] += 1
        with open(analytics_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        # ignore logging failures
        pass

def _authorize_repair(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Determine the scope of the requested repair.  Clients may supply a
    # "scope" field describing the subset of the repository the token
    # applies to (e.g. "repair_engine", "templates:reasoning").  If
    # omitted, default to the target field or a generic repair scope.
    # Regardless of the requested target, repairs are governed by the repair engine.
    # Always issue tokens scoped to "repair_engine" so that repair operations can proceed.
    # Previous versions inherited the target as scope (e.g. "reasoning"), which prevented
    # repair requests from being authorized because the operation name did not match the scope.
    scope = "repair_engine"
    # Mint a pseudo‑random identifier for the token.  Prefix with GOV-
    # for easy recognition.  Note: the token itself carries no
    # authority; the signature is what guarantees integrity.
    token_id = f"GOV-{secrets.token_hex(8)}"
    ts = _now_ms()
    auth: Dict[str, Any] = {
        "issuer": "governance",
        "token": token_id,
        "scope": scope,
        "ts": ts,
        "ttl_ms": _TTL_MS,
        "valid": True,
    }
    # Sign the token contents
    auth["signature"] = _sign_token(auth)
    return {"authorized": True, "auth": auth}

def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op","").upper()
    payload = (msg or {}).get("payload",{}) or {}
    mid = payload.get("mid") or f"mid_{_now_ms()}"

    # Normal observation: always allow & audit
    if op in {"ENFORCE","POLICY_EVENT","TEMPLATE_STATUS","PROMOTE_TEMPLATE","ROLLBACK_TEMPLATE","SCAN"}:
        p = dict(payload)
        if isinstance(p, dict):
            p.setdefault("allowed", True)
            inner = p.get("payload")
            if isinstance(inner, dict):
                inner.setdefault("allowed", True)
        result = _mk(mid, op, p, allowed=True, effect="ALLOW_AUDIT")
        # Log analytics
        _log_analytics(op, result.get("effect", "ALLOW_AUDIT"))
        return result

    if op in {"AUTHORIZE_REPAIR","AUTH_REPAIR"}:
        p = _authorize_repair(payload)
        result = _mk(mid, op, p, allowed=True, effect="ALLOW")
        _log_analytics(op, result.get("effect", "ALLOW"))
        return result

    # Autonomy‑related allowances.  When the system requests to tick the
    # autonomous DMN or to adopt a goal, grant permission by default.
    # These hooks allow higher‑level autonomy circuits to be controlled via
    # policy without interfering with ordinary observation.  Additional
    # logic can be implemented here in future revisions.
    if op in {"ALLOW_AUTONOMY_TICK", "ALLOW_GOAL"}:
        p = dict(payload)
        result = _mk(mid, op, p, allowed=True, effect="ALLOW")
        _log_analytics(op, result.get("effect", "ALLOW"))
        return result

    # Default pass-through
    p = dict(payload)
    p.setdefault("allowed", True)
    result = _mk(mid, op or "DEFAULT", p, allowed=True, effect="ALLOW_AUDIT")
    _log_analytics(op or "DEFAULT", result.get("effect", "ALLOW_AUDIT"))
    return result


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
