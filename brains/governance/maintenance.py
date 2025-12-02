"""
Governance Maintenance Module
=============================

Handles periodic maintenance tasks for Maven governance layer:
- Identity profile updates (behavioral trait analysis)
- Log rotation and cleanup
- Stale permit cleanup
- Health checks

This module should be called:
1. On system startup (via run_startup_maintenance)
2. Periodically during operation (via run_periodic_maintenance)
"""

from __future__ import annotations
from typing import Dict, Any, List
import time
import json
from pathlib import Path

# Paths
MAVEN_ROOT = Path.home() / ".maven"
MAINTENANCE_LOG = MAVEN_ROOT / "logs" / "maintenance.jsonl"


def _log_maintenance_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log a maintenance event to the maintenance log."""
    try:
        MAINTENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "event": event_type,
            "details": details
        }
        with open(MAINTENANCE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Best-effort logging


def update_identity() -> Dict[str, Any]:
    """
    Update the identity profile with latest behavioral metrics.

    Returns:
        Dict with status and any errors
    """
    try:
        from brains.personal.service.identity_inferencer import update_identity_profile
        result = update_identity_profile()
        _log_maintenance_event("identity_update", {
            "success": True,
            "traits_updated": len(result.get("traits", {}))
        })
        return {"success": True, "result": result}
    except ImportError as e:
        _log_maintenance_event("identity_update", {
            "success": False,
            "error": f"import_error: {e}"
        })
        return {"success": False, "error": f"Identity inferencer not available: {e}"}
    except Exception as e:
        _log_maintenance_event("identity_update", {
            "success": False,
            "error": str(e)
        })
        return {"success": False, "error": str(e)}


def cleanup_stale_permits() -> Dict[str, Any]:
    """
    Clean up expired permits from the permit log.

    Returns:
        Dict with cleanup results
    """
    try:
        from brains.governance.permit_logger import cleanup_expired_permits
        removed = cleanup_expired_permits()
        _log_maintenance_event("permit_cleanup", {
            "success": True,
            "permits_removed": removed
        })
        return {"success": True, "permits_removed": removed}
    except ImportError:
        # Permit logger may not have cleanup function
        _log_maintenance_event("permit_cleanup", {
            "success": False,
            "error": "cleanup function not available"
        })
        return {"success": False, "error": "cleanup function not available"}
    except Exception as e:
        _log_maintenance_event("permit_cleanup", {
            "success": False,
            "error": str(e)
        })
        return {"success": False, "error": str(e)}


def check_brain_health() -> Dict[str, Any]:
    """
    Run health checks on key governance brains.

    Returns:
        Dict with health status of each brain
    """
    health_results = {}

    # List of governance brains to check
    brains_to_check = [
        ("council", "brains.governance.council.service.council_brain"),
        ("policy_engine", "brains.governance.policy_engine.service.policy_engine"),
        ("repair_engine", "brains.governance.repair_engine.service.repair_engine"),
        ("upgrade_engine", "brains.governance.upgrade_engine.service.upgrade_engine"),
    ]

    for brain_name, module_path in brains_to_check:
        try:
            import importlib
            module = importlib.import_module(module_path)
            api = getattr(module, "service_api", None)
            if api:
                response = api({"op": "HEALTH", "mid": f"maintenance_{brain_name}"})
                health_results[brain_name] = {
                    "healthy": response.get("ok", False),
                    "details": response.get("payload", {})
                }
            else:
                health_results[brain_name] = {
                    "healthy": False,
                    "error": "no service_api found"
                }
        except Exception as e:
            health_results[brain_name] = {
                "healthy": False,
                "error": str(e)
            }

    _log_maintenance_event("health_check", {
        "results": health_results
    })

    return {"success": True, "health": health_results}


def run_startup_maintenance() -> Dict[str, Any]:
    """
    Run all startup maintenance tasks.

    This should be called when Maven starts up.

    Returns:
        Dict with results of all startup tasks
    """
    results = {
        "timestamp": time.time(),
        "type": "startup",
        "tasks": {}
    }

    # 1. Update identity profile
    results["tasks"]["identity_update"] = update_identity()

    # 2. Run health checks
    results["tasks"]["health_check"] = check_brain_health()

    # Log overall startup
    _log_maintenance_event("startup_maintenance", {
        "success": all(
            t.get("success", False)
            for t in results["tasks"].values()
        ),
        "tasks_run": len(results["tasks"])
    })

    return results


def run_periodic_maintenance() -> Dict[str, Any]:
    """
    Run periodic maintenance tasks.

    This should be called at regular intervals during operation.

    Returns:
        Dict with results of all periodic tasks
    """
    results = {
        "timestamp": time.time(),
        "type": "periodic",
        "tasks": {}
    }

    # 1. Update identity profile (captures latest behavior)
    results["tasks"]["identity_update"] = update_identity()

    # 2. Cleanup stale permits
    results["tasks"]["permit_cleanup"] = cleanup_stale_permits()

    # Log overall periodic run
    _log_maintenance_event("periodic_maintenance", {
        "success": all(
            t.get("success", False)
            for t in results["tasks"].values()
        ),
        "tasks_run": len(results["tasks"])
    })

    return results


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for governance maintenance.

    Supported operations:
    - RUN_STARTUP: Run startup maintenance tasks
    - RUN_PERIODIC: Run periodic maintenance tasks
    - UPDATE_IDENTITY: Update identity profile only
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid", f"maintenance_{int(time.time())}")

    if op == "RUN_STARTUP":
        result = run_startup_maintenance()
        return {"ok": True, "op": op, "mid": mid, "payload": result}

    if op == "RUN_PERIODIC":
        result = run_periodic_maintenance()
        return {"ok": True, "op": op, "mid": mid, "payload": result}

    if op == "UPDATE_IDENTITY":
        result = update_identity()
        return {"ok": result["success"], "op": op, "mid": mid, "payload": result}

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "governance_maintenance",
                "available_operations": ["RUN_STARTUP", "RUN_PERIODIC", "UPDATE_IDENTITY", "HEALTH"]
            }
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": f"Operation '{op}' not supported"
        }
    }
