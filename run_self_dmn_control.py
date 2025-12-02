#!/usr/bin/env python3
"""
run_self_dmn_control.py
=======================

Command line utility for invoking Self‑DMN operations.

This script loads the Self‑DMN brain dynamically and exposes a simple
interface for running maintenance operations such as TICK, REFLECT,
DISSENT_SCAN and ANALYZE_INTERNAL.  The ``--window`` argument controls
how many recent runs are considered during analysis operations.

Example usage::

    python run_self_dmn_control.py --op REFLECT --window 20

The result of the operation is printed as formatted JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api.utils import generate_mid


def _load_self_dmn():
    """
    Dynamically load the Self‑DMN brain module.

    Returns the module object for ``self_dmn_brain`` so that its
    ``service_api`` function can be invoked.  This dynamic loader avoids
    import errors when this script is executed from outside the Maven package.
    """
    base = Path(__file__).resolve().parent
    svc_path = base / "brains" / "cognitive" / "self_dmn" / "service" / "self_dmn_brain.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("self_dmn_brain", svc_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke Self‑DMN maintenance operations")
    parser.add_argument("--op", required=True, help="Operation name: TICK, REFLECT, DISSENT_SCAN, ANALYZE_INTERNAL")
    parser.add_argument("--window", type=int, default=10, help="Window size for REFLECT or ANALYZE_INTERNAL operations")
    args = parser.parse_args()
    mod = _load_self_dmn()
    op = args.op.upper()
    payload: dict[str, object] = {}
    if op in {"REFLECT", "ANALYZE_INTERNAL", "DISSENT_SCAN"}:
        payload["window"] = args.window
    msg = {"op": op, "mid": generate_mid(), "payload": payload}
    try:
        res = mod.service_api(msg)  # type: ignore[attr-defined]
    except Exception as e:
        res = {"ok": False, "op": op, "mid": msg["mid"], "error": {"code": "ERROR", "message": str(e)}}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()