# SAFE restore-all tool (governance-gated). Place under:
# brains/governance/repair_engine/tools/safe_restore_all.py
from __future__ import annotations
import json, sys, time, importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # .../maven
ENGINE_PATH = REPO / "brains" / "governance" / "repair_engine" / "service" / "repair_engine.py"

BRAINS = [
    "sensorium","planner","language","pattern_recognition","reasoning",
    "affect_priority","personality","self_dmn","system_history",
    "memory_librarian","personal"
]

def _load_engine():
    spec = importlib.util.spec_from_file_location("repair_engine_service", ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Provide golden version, e.g., baseline_1761871530")
        sys.exit(2)
    version = sys.argv[1]
    eng = _load_engine()

    results = []
    ok = True
    for b in BRAINS:
        res = eng.service_api({"op":"ROLLBACK_TO_GOLDEN","payload":{"brain": b, "version": version}})
        payload = res.get("payload") or res
        allowed = (payload.get("authorized") is True) or (payload.get("decision") or {}).get("allowed") is True
        results.append({"brain": b, "result": payload})
        ok = ok and allowed and ("error" not in payload)

    # meta-report
    report = {
        "ts": int(time.time()),
        "op": "RESTORE_ALL_FROM_GOLDEN",
        "version": version,
        "results": results,
        "all_ok": ok
    }
    # Write via engine's writer for consistency
    try:
        from api.utils import write_report
        write_report("repair", f"restore_all_from_golden_{report['ts']}.json", json.dumps(report, indent=2))
    except Exception:
        pass

    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
