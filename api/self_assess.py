
from __future__ import annotations
import time, json
from pathlib import Path
from typing import Dict, Any
from .memory import ensure_dirs, count_lines
from .utils import write_report

def primitive_stats(brain_root: Path) -> Dict[str, Any]:
    t = ensure_dirs(brain_root)
    return {
        "stm_count": count_lines(t["stm"]),
        "mtm_count": count_lines(t["mtm"]),
        "ltm_count": count_lines(t["ltm"]),
    }

def adjust_weights(weights_path: Path, stats: Dict[str, Any]) -> Dict[str, Any]:
    try:
        w = json.loads(weights_path.read_text(encoding="utf-8"))
    except Exception:
        w = {"verbosity_bias": 0.5, "parse_priority": 0.5, "explain_bias": 0.5}
    w["verbosity_bias"] = round(min(1.0, w.get("verbosity_bias",0.5)+0.01), 3)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(w, indent=2), encoding="utf-8")
    return w

def write_activity_report(brain: str, stats: Dict[str, Any], weights: Dict[str, Any], root: Path):
    ts = int(time.time())
    payload = {"brain": brain, "timestamp": ts, "stats": stats, "weights": weights}
    write_report("activity", f"{brain}_self_assess_{ts}.json", json.dumps(payload, indent=2))
    return payload
