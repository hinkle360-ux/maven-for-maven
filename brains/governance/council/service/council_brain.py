from __future__ import annotations
from typing import Dict, Any, List
from api.utils import generate_mid, success_response, error_response
import sys
from pathlib import Path
from brains.maven_paths import (
    get_runtime_domain_banks_root,
    validate_path_confinement,
)

# Import seeding engine for admin operations
SEEDS_DIR = Path(__file__).parent.parent.parent.parent / "domain_banks" / "specs" / "data" / "seeds"
# Runtime dir must be inside maven2_fix, not a sibling directory
RUNTIME_DIR = validate_path_confinement(
    get_runtime_domain_banks_root(), "council domain bank runtime"
)

# Add seeds directory to path for imports
if str(SEEDS_DIR) not in sys.path:
    sys.path.insert(0, str(SEEDS_DIR))


def _run_domain_bank_seeding(validate_only: bool) -> Dict[str, Any]:
    """
    Run domain bank seeding operation.

    Args:
        validate_only: If True, validate but don't apply

    Returns:
        Seeding report
    """
    try:
        from seeding_engine import run_seeding
        from seed_validator import ValidationError

        report = run_seeding(
            str(SEEDS_DIR),
            str(RUNTIME_DIR),
            validate_only=validate_only
        )
        return {"ok": True, "report": report}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Council brain service for arbitration and admin operations.

    Operations:
    - ARBITRATE: Coordinate outputs from multiple brains
    - DOMAIN_BANK_SEED_VALIDATE: Validate domain bank seeds without applying
    - DOMAIN_BANK_SEED_APPLY: Validate and apply domain bank seeds

    Unsupported operations return an error.
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    if op == "ARBITRATE":
        cands: List[Dict[str, Any]] = payload.get("candidates") or []
        if not isinstance(cands, list) or not cands:
            return success_response(op, mid, {"decision": None})
        best: Dict[str, Any] | None = None
        best_conf: float = float("-inf")
        for cand in cands:
            try:
                conf = float(cand.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            if conf > best_conf:
                best_conf = conf
                best = cand
        return success_response(op, mid, {"decision": best})

    elif op == "DOMAIN_BANK_SEED_VALIDATE":
        result = _run_domain_bank_seeding(validate_only=True)
        if result["ok"]:
            return success_response(op, mid, result["report"])
        else:
            return error_response(op, mid, "SEEDING_FAILED", result.get("error", "Unknown error"))

    elif op == "DOMAIN_BANK_SEED_APPLY":
        result = _run_domain_bank_seeding(validate_only=False)
        if result["ok"]:
            return success_response(op, mid, result["report"])
        else:
            return error_response(op, mid, "SEEDING_FAILED", result.get("error", "Unknown error"))

    # Phase 8: Task execution operations
    elif op == "TASK_EXECUTE":
        task = payload.get("task", "")
        context = payload.get("context") or {}

        if not task:
            return error_response(op, mid, "MISSING_TASK", "Task description is required")

        try:
            # Import and use Task Execution Engine
            from brains.governance.task_execution_engine.engine import get_engine

            engine = get_engine()
            result = engine.execute_task(task, context, with_trace=False)

            if result.get("success"):
                return success_response(op, mid, {
                    "output": result.get("output"),
                    "steps_executed": result.get("steps_executed", 0)
                })
            else:
                return error_response(
                    op, mid,
                    result.get("error_code", "EXECUTION_FAILED"),
                    result.get("error", "Task execution failed")
                )

        except Exception as e:
            return error_response(op, mid, "TASK_EXECUTION_ERROR", str(e))

    elif op == "TASK_EXECUTE_WITH_TRACE":
        task = payload.get("task", "")
        context = payload.get("context") or {}

        if not task:
            return error_response(op, mid, "MISSING_TASK", "Task description is required")

        try:
            # Import and use Task Execution Engine
            from brains.governance.task_execution_engine.engine import get_engine

            engine = get_engine()
            result = engine.execute_task(task, context, with_trace=True)

            if result.get("success"):
                return success_response(op, mid, {
                    "output": result.get("output"),
                    "steps_executed": result.get("steps_executed", 0),
                    "trace": result.get("trace", {})
                })
            else:
                # Include trace even on failure
                return error_response(
                    op, mid,
                    result.get("error_code", "EXECUTION_FAILED"),
                    result.get("error", "Task execution failed"),
                    extra_data={"trace": result.get("trace", {})}
                )

        except Exception as e:
            return error_response(op, mid, "TASK_EXECUTION_ERROR", str(e))

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