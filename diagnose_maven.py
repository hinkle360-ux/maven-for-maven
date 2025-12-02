"""Simple diagnostic script for Maven LLM and identity integration.

This script performs several checks against a Maven installation to help
troubleshoot issues with identity persistence, LLM integration and Stage 6
generation.  To run, execute ``python diagnose_maven.py`` from the root
of your Maven checkout.  The output is divided into sections:

* ``ENV & PATH`` – basic environment information and Python path.
* ``CONFIG`` – status of ``config/llm.json`` and its key fields.
* ``IMPORT`` – whether the LLM service can be imported and its base URL.
* ``DIRECT OLLAMA CALL`` – call the local LLM service to ensure it is reachable.
* ``LANGUAGE_BRAIN WIRING`` – confirm that the language brain exposes the
  helper functions required for LLM prompts.
* ``PIPELINE`` – run a parse and generate cycle through the language brain
  for a simple free‑form query, and show the first portion of any answer.
* ``IDENTITY`` – display the current stored identity and simulate a name
  correction via the chat ``process`` function.

This script is intended for debugging and does not modify any files.  It
relies on standard library modules and will run without additional
dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def kv(key: str, value: Any) -> None:
    print(f"{key:24} {value}")


def try_load_json(path: Path) -> Tuple[bool, Any]:
    """Attempt to load JSON from ``path``.  Returns a tuple of
    ``(success, data_or_error)``.  On failure, the second element is an
    error string.  On success, it is the parsed JSON object.  This helper
    prevents diagnostics from crashing due to malformed JSON files.
    """
    try:
        return True, json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    root = Path(__file__).resolve().parent
    # Ensure the project root is on the path so imports work correctly
    sys.path.insert(0, str(root))

    section("ENV & PATH")
    kv("cwd", os.getcwd())
    kv("root", str(root))
    kv("python", sys.version.replace("\n", " "))
    kv("sys.path[0]", sys.path[0])

    section("CONFIG: config/llm.json")
    llm_cfg_path = root / "config" / "llm.json"
    kv("exists", llm_cfg_path.exists())
    ok, data = try_load_json(llm_cfg_path) if llm_cfg_path.exists() else (False, "missing")
    kv("load_ok", ok)
    if ok:
        kv("enabled", data.get("enabled"))
        kv("provider", data.get("provider"))
        kv("ollama_url", data.get("ollama_url"))
        kv("model", data.get("model"))
    else:
        kv("error", data)

    section("IMPORT: llm_service")
    try:
        from brains.tools.llm_service import llm_service
        kv("import", "OK")
        kv("base_url", getattr(llm_service, "base_url", "?"))
        kv("model", getattr(llm_service, "model", "?"))
    except Exception as e:
        kv("import", f"FAIL: {e}")
        traceback.print_exc()

    section("DIRECT OLLAMA CALL")
    try:
        from brains.tools.llm_service import llm_service
        r = llm_service.call("Say hello in one short sentence.", context={"user": {"name": "Diag"}})
        kv("ok", r.get("ok"))
        kv("source", r.get("source"))
        txt = r.get("text", "")[:120].replace("\n", " ")
        kv("text_head", txt)
    except Exception as e:
        kv("error", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    section("LANGUAGE_BRAIN WIRING")
    try:
        from brains.cognitive.language.service.language_brain import _llm, build_generation_prompt
        kv("_llm is None", _llm is None)
        kv("has build_generation_prompt", callable(build_generation_prompt))
    except Exception as e:
        kv("import", f"FAIL: {e}")
        traceback.print_exc()

    section("PIPELINE: PARSE/GEN (freeform explain)")
    try:
        from brains.cognitive.language.service.language_brain import service_api
        mid = "diag-" + str(int(time.time()))
        parse = service_api({"op": "PARSE", "mid": mid, "payload": {"text": "Explain photosynthesis simply"}})
        kv("PARSE.ok", parse.get("ok"))
        kv("PARSE.action", (parse.get("payload") or {}).get("action"))
        gen = service_api({"op": "GEN", "mid": mid, "payload": {"src": "Explain photosynthesis simply"}})
        kv("GEN.ok", gen.get("ok"))
        ans = (gen.get("payload") or {}).get("final_answer", "")[:220].replace("\n", " ")
        kv("GEN.final_head", ans)
    except Exception as e:
        kv("GEN.error", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    section("IDENTITY: durable & session")
    ident_path = root / "reports" / "personal" / "identity.json"
    kv("identity.json exists", ident_path.exists())
    if ident_path.exists():
        ok2, data2 = try_load_json(ident_path)
        kv("identity.json ok", ok2)
        if ok2:
            kv("stored name", (data2.get("primary_user") or {}).get("name"))
    # Test identity setting via process() API
    try:
        from ui.maven_chat import process
        resp1 = process("no i am Diag")
        resp2 = process("who am i")
        kv("process('no i am Diag')", resp1)
        kv("process('who am i')", resp2)
    except Exception as e:
        kv("chat.process", f"FAIL: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()