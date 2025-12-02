"""Hot-reload helper with optional compilation checks."""

from __future__ import annotations

import importlib
import importlib.util
import py_compile
from typing import Dict, List, Union

from brains.tools.execution_guard import require_execution_enabled


def _maybe_compile(module_name: str, spec) -> None:
    try:
        if not spec or not spec.origin or not spec.origin.endswith(".py"):
            return
        py_compile.compile(spec.origin, doraise=True)
    except Exception as e:
        print(f"[RELOAD_WARN] compile failed for {module_name}: {e}")


def reload_modules(module_names: List[str]) -> Dict[str, Dict[str, Union[bool, str]]]:
    """Attempt to reload the provided modules with guardrails."""

    require_execution_enabled("reload_modules")
    print(f"[RELOAD] start modules={module_names}")
    results: Dict[str, Dict[str, Union[bool, str]]] = {}

    for name in module_names:
        entry: Dict[str, Union[bool, str]] = {}
        try:
            spec = importlib.util.find_spec(name)
            _maybe_compile(name, spec)
            module = importlib.import_module(name)
            importlib.reload(module)
            entry["ok"] = True
            print(f"[RELOAD_OK] module={name}")
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
            print(f"[RELOAD_FAIL] module={name} error={e}")
        results[name] = entry

    return results
