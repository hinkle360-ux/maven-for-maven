
from __future__ import annotations
# test run_maven patch inserted
import json, importlib.util, sys
from pathlib import Path
from api.utils import generate_mid

HERE = Path(__file__).resolve().parent
# Ensure the project root (this directory) is on sys.path so that local
# packages like ``api`` and ``ui`` can be imported when there is no nested
# ``maven`` package present.  This mirrors the dynamic import logic used in
# other entry points such as ``maven/ui/maven_chat.py``.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Import the new canonical pipeline runner
from brains.pipeline.pipeline_runner import run_pipeline

# Keep legacy memory_librarian import for backward compatibility
lib = HERE / "brains" / "cognitive" / "memory_librarian" / "service" / "memory_librarian.py"
spec = importlib.util.spec_from_file_location("memory_librarian", lib)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _show_memory_tiers():
    """Show memory tier introspection for debugging."""
    try:
        from brains.memory.tier_manager import TierManager
        from pathlib import Path

        # Get Maven root
        maven_root = Path(__file__).parent

        # Check key brains with their storage locations
        brains_to_check = [
            ("personal", maven_root / "brains" / "personal"),
            ("language", maven_root / "brains" / "cognitive" / "language"),
            ("reasoning", maven_root / "brains" / "cognitive" / "reasoning"),
            ("system_history", maven_root / "brains" / "cognitive" / "system_history"),
            ("pattern_recognition", maven_root / "brains" / "cognitive" / "pattern_recognition"),
        ]

        print("\n  Brain                    STM    MTM    LTM    Archive")
        print("  " + "-" * 60)

        for brain_id, brain_path in brains_to_check:
            # Check if the brain's memory directory exists
            memory_path = brain_path / "memory"
            if memory_path.exists() and any(memory_path.iterdir()):
                tm = TierManager(brain_id, brain_path, enforce_tier_governance=False)
                counts = tm.get_tier_counts()
                print(f"  {brain_id:24s} {counts.get('stm', 0):5d}  {counts.get('mtm', 0):5d}  {counts.get('ltm', 0):5d}  {counts.get('archive', 0):7d}")
            else:
                print(f"  {brain_id:24s}     0      0      0        0")

        print("")
    except Exception as e:
        print(f"  Error inspecting memory tiers: {e}\n")


if __name__ == "__main__":
    """
    When invoked as a script, this entry point will launch the natural language
    chat interface if no command line arguments are provided.  Otherwise, it
    treats the first argument as a query and an optional second argument as a
    confidence value and executes the full Maven pipeline on that input.  This
    dual behaviour makes `run_maven.py` both a convenient button to start
    interactive conversation and a quick way to test the pipeline with a
    single query.
    """
    # ==========================================================================
    # CAPABILITY STARTUP SCAN
    # ==========================================================================
    # Run capability probes at startup to build the truth object.
    # This ensures self_model can answer "can you X" questions accurately.
    try:
        from brains.system_capabilities import scan_all_capabilities
        _capability_scan = scan_all_capabilities()
        # Log summary to console
        print(f"[STARTUP] Capabilities: {_capability_scan.get('summary', 'unknown')}")
    except Exception as e:
        print(f"[STARTUP] Capability scan failed: {e}")
        _capability_scan = None

    # Start the learning daemon for pattern analysis.  The daemon runs in
    # a background thread and periodically calls the LLM service to learn
    # new templates from logged interactions.  If the import fails or
    # scheduling is disabled, the daemon is skipped silently.
    try:
        from brains.agent.learning_daemon import LearningDaemon  # type: ignore
        import threading
        _learning_daemon = LearningDaemon()
        # Only start the daemon if learning is enabled
        if getattr(_learning_daemon, "learning_enabled", True):
            _ld_thread = threading.Thread(
                target=_learning_daemon.start_scheduled, daemon=True
            )
            _ld_thread.start()
    except Exception:
        # If anything goes wrong, do not block startup; the pipeline
        # continues without background learning.
        _learning_daemon = None

    # Parse optional arguments
    # --mode: "architect" (default) or "execution"
    # --pipeline: "canonical" (new pipeline) or "legacy" (old memory_librarian)
    # --debug-memory: show memory tier introspection
    # --full-agency: enable FULL_AGENCY mode for unrestricted access
    # --profile: "safe_chat" or "full_agency" - capability profile selection
    mode = "architect"
    use_pipeline = "canonical"  # Default to new canonical pipeline
    debug_memory = False
    full_agency = False
    profile = None  # Will be set by --profile or environment
    args = sys.argv[1:]

    if "--mode" in args:
        try:
            idx = args.index("--mode")
            if idx + 1 < len(args):
                mode = args[idx + 1].strip().lower() or mode
                del args[idx:idx + 2]
        except Exception:
            pass

    if "--pipeline" in args:
        try:
            idx = args.index("--pipeline")
            if idx + 1 < len(args):
                use_pipeline = args[idx + 1].strip().lower() or use_pipeline
                del args[idx:idx + 2]
                # Warn if using legacy pipeline
                if use_pipeline == "legacy":
                    print("\n" + "⚠" * 30)
                    print("⚠ WARNING: Using LEGACY pipeline mode")
                    print("⚠ This mode is DEPRECATED and may be removed in future versions")
                    print("⚠ The canonical pipeline is now the default and recommended mode")
                    print("⚠ If you encounter issues, please report them instead of using legacy mode")
                    print("⚠" * 30 + "\n")
        except Exception:
            pass

    if "--debug-memory" in args:
        debug_memory = True
        args.remove("--debug-memory")

    # Handle --profile argument (new preferred way)
    if "--profile" in args:
        try:
            idx = args.index("--profile")
            if idx + 1 < len(args):
                profile = args[idx + 1].strip().lower()
                del args[idx:idx + 2]
        except Exception:
            pass

    if "--full-agency" in args:
        full_agency = True
        profile = "full_agency"
        args.remove("--full-agency")

    # Check environment variables for profile/mode
    import os

    # MAVEN_CAPABILITIES_PROFILE takes precedence
    env_profile = os.getenv("MAVEN_CAPABILITIES_PROFILE", "").upper()
    if env_profile in ("SAFE_CHAT", "FULL_AGENCY"):
        profile = env_profile.lower()

    # MAVEN_EXECUTION_MODE as fallback
    env_mode = os.getenv("MAVEN_EXECUTION_MODE", "").upper()
    if env_mode == "FULL_AGENCY" and profile is None:
        profile = "full_agency"

    # Activate the selected profile
    if profile == "full_agency":
        try:
            from capabilities import activate_profile
            activate_profile("FULL_AGENCY", "enabled via --profile or environment")
            print("[STARTUP] Profile: FULL_AGENCY - unrestricted access to all tools")
            full_agency = True
        except Exception as e:
            # Fallback to old method
            try:
                from brains.tools.execution_guard import enable_full_agency
                enable_full_agency("enabled via --profile or environment")
                print("[STARTUP] FULL_AGENCY mode enabled - unrestricted access to all tools")
                full_agency = True
            except Exception as e2:
                print(f"[STARTUP] Warning: Failed to enable FULL_AGENCY mode: {e2}")
    elif profile == "safe_chat":
        try:
            from capabilities import activate_profile
            activate_profile("SAFE_CHAT", "enabled via --profile or environment")
            print("[STARTUP] Profile: SAFE_CHAT - pure conversation, no tools")
        except Exception as e:
            # Fallback to execution guard
            try:
                from brains.tools.execution_guard import enable_safe_chat
                enable_safe_chat("enabled via --profile or environment")
                print("[STARTUP] SAFE_CHAT mode enabled - no tool access")
            except Exception as e2:
                print(f"[STARTUP] Warning: Failed to enable SAFE_CHAT mode: {e2}")
    elif profile is None:
        # No profile specified - use default behavior (check old env var)
        if env_mode == "FULL_AGENCY":
            try:
                from brains.tools.execution_guard import enable_full_agency
                enable_full_agency("enabled via MAVEN_EXECUTION_MODE environment variable")
                print("[STARTUP] FULL_AGENCY mode enabled via environment variable")
                full_agency = True
            except Exception as e:
                print(f"[STARTUP] Warning: Failed to enable FULL_AGENCY mode: {e}")

    if "--self-test-routing" in args:
        # Run routing self-test and exit
        print("\n" + "=" * 60)
        print("MAVEN ROUTING + MEMORY SELF-TEST")
        print("=" * 60)
        try:
            from brains.cognitive.memory_librarian.service import memory_librarian
            question = "what sound does a duck make"

            print(f"\nTesting with question: '{question}'")
            print("\n[1/2] First query (should call teacher)...")

            resp1 = memory_librarian.service_api({
                "op": "RUN_PIPELINE",
                "mid": generate_mid(),
                "payload": {"text": question, "confidence": 1.0}
            })

            if not resp1.get("ok"):
                print(f"✗ First query failed: {resp1.get('error')}")
                sys.exit(1)

            ctx1 = resp1.get("payload", {}).get("context", {})
            verdict1 = ctx1.get("verdict")
            teacher_facts_stored = ctx1.get("teacher_facts_stored", 0)

            teacher_called_first = verdict1 == "LEARNED"
            print(f"  Verdict: {verdict1}")
            print(f"  Teacher facts stored: {teacher_facts_stored}")
            print(f"  [SELF-TEST] Teacher called on first query: {'YES' if teacher_called_first else 'NO'}")

            print("\n[2/2] Second query (should use memory)...")

            resp2 = memory_librarian.service_api({
                "op": "RUN_PIPELINE",
                "mid": generate_mid(),
                "payload": {"text": question, "confidence": 1.0}
            })

            if not resp2.get("ok"):
                print(f"✗ Second query failed: {resp2.get('error')}")
                sys.exit(1)

            ctx2 = resp2.get("payload", {}).get("context", {})
            verdict2 = ctx2.get("verdict")
            banks_used = ctx2.get("stage_2R_top_banks", [])

            teacher_called_second = verdict2 == "LEARNED"
            print(f"  Verdict: {verdict2}")
            print(f"  Banks used: {banks_used}")
            print(f"  [SELF-TEST] Teacher called on second query: {'YES' if teacher_called_second else 'NO'}")
            print(f"  [SELF-TEST] Learned routing rule applied: {'YES' if len(banks_used) > 0 else 'UNKNOWN'}")
            print(f"  [SELF-TEST] Banks used on second query: {', '.join(banks_used) if banks_used else 'none'}")

            # Determine pass/fail
            if teacher_called_first and not teacher_called_second and len(banks_used) > 0:
                print("\n" + "=" * 60)
                print("[SELF-TEST] RESULT: PASS")
                print("=" * 60)
                print("✓ Teacher called on first query")
                print("✓ Memory used on second query (teacher not called)")
                print(f"✓ Routing selected banks: {', '.join(banks_used)}")
                sys.exit(0)
            else:
                print("\n" + "=" * 60)
                print("[SELF-TEST] RESULT: FAIL")
                print("=" * 60)
                if not teacher_called_first:
                    print("✗ Teacher was not called on first query")
                if teacher_called_second:
                    print("✗ Teacher was called on second query (should use memory)")
                if len(banks_used) == 0:
                    print("✗ No banks were used on second query")
                sys.exit(1)

        except Exception as e:
            print(f"\n✗ Self-test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # If no additional arguments remain, start the interactive chat
    if not args:
        try:
            # Import the chat interface from the UI package lazily.
            from ui.maven_chat import repl as chat_repl  # type: ignore
            chat_repl()
        except Exception:
            # Fallback: run a default pipeline example in the selected mode
            text = "The cell divides by mitosis."
            conf = 0.8

            # Choose pipeline based on --pipeline flag
            if use_pipeline == "canonical":
                print("Using CANONICAL PIPELINE (PipelineExecutor with 9 stages)")
                result = run_pipeline(text, conf)
                if mode == "execution":
                    print(json.dumps({"ok": result.get("ok"), "response": result.get("response", "")}, indent=2))
                else:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                # Legacy path
                print("Using LEGACY PIPELINE (memory_librarian RUN_PIPELINE)")
                resp = mod.service_api({"op": "RUN_PIPELINE", "mid": generate_mid(), "payload": {"text": text, "confidence": conf}})
                if mode == "execution":
                    try:
                        ctx = (resp.get("payload") or {}).get("context") or {}
                        final_ans = ctx.get("final_answer")
                        final_conf = ctx.get("final_confidence")
                        if final_ans is not None:
                            print(json.dumps({"final_answer": final_ans, "final_confidence": final_conf}, indent=2, ensure_ascii=False))
                        else:
                            print(json.dumps(resp, indent=2))
                    except Exception:
                        print(json.dumps(resp, indent=2))
                else:
                    # Architect mode: show final answer if available for brevity
                    try:
                        ctx = (resp.get("payload") or {}).get("context") or {}
                        final_ans = ctx.get("final_answer")
                        final_conf = ctx.get("final_confidence")
                        if final_ans is not None:
                            print(json.dumps({"final_answer": final_ans, "final_confidence": final_conf}, indent=2, ensure_ascii=False))
                        else:
                            print(json.dumps(resp, indent=2))
                    except Exception:
                        print(json.dumps(resp, indent=2))
    else:
        # Use provided arguments to run the pipeline on a single input
        # Remaining args correspond to query and optional confidence
        text = args[0] if len(args) >= 1 else "The cell divides by mitosis."
        try:
            conf = float(args[1]) if len(args) >= 2 else 0.8
        except Exception:
            conf = 0.8

        # Before running the pipeline, consolidate memory tiers.  This ensures
        # that high‑importance facts from previous sessions are promoted into
        # mid‑ and long‑term stores and available for retrieval.  Errors are
        # swallowed to avoid disrupting normal pipeline execution.
        try:
            from brains.cognitive.memory_consolidation import consolidate_memories  # type: ignore
            consolidate_memories()
        except Exception:
            pass

        # Choose pipeline based on --pipeline flag
        if use_pipeline == "canonical":
            print(f"\n🔧 Using CANONICAL PIPELINE (PipelineExecutor with 9 mandatory stages)")
            print(f"📝 Query: {text}\n")
            result = run_pipeline(text, conf)

            # Show execution log if in architect mode
            if mode == "architect" and result.get("execution_log"):
                print("\n📊 PIPELINE EXECUTION LOG:")
                for entry in result.get("execution_log", []):
                    status_icon = "✓" if entry.get("success") else "✗"
                    print(f"  {status_icon} {entry.get('stage'):20s} {entry.get('duration_ms', 0):6.1f}ms")

            # Show memory tier introspection if requested
            if debug_memory:
                print("\n🧠 MEMORY TIER INTROSPECTION:")
                _show_memory_tiers()

            if mode == "execution":
                print(json.dumps({"ok": result.get("ok"), "response": result.get("response", "")}, indent=2))
            else:
                print(f"\n💬 RESPONSE:\n{result.get('response', '')}\n")
                if result.get("ok"):
                    print(f"✓ Pipeline completed successfully")
                else:
                    print(f"✗ Pipeline failed: {result.get('error', 'Unknown error')}")
        else:
            # Legacy path
            print("\n⚠️  Using LEGACY PIPELINE (memory_librarian RUN_PIPELINE)")
            resp = mod.service_api({"op": "RUN_PIPELINE", "mid": generate_mid(), "payload": {"text": text, "confidence": conf}})
            if mode == "execution":
                try:
                    ctx = (resp.get("payload") or {}).get("context") or {}
                    final_ans = ctx.get("final_answer")
                    final_conf = ctx.get("final_confidence")
                    if final_ans is not None:
                        print(json.dumps({"final_answer": final_ans, "final_confidence": final_conf}, indent=2, ensure_ascii=False))
                    else:
                        print(json.dumps(resp, indent=2))
                except Exception:
                    print(json.dumps(resp, indent=2))
            else:
                # Architect mode: show final answer if available for brevity
                try:
                    ctx = (resp.get("payload") or {}).get("context") or {}
                    final_ans = ctx.get("final_answer")
                    final_conf = ctx.get("final_confidence")
                    if final_ans is not None:
                        print(json.dumps({"final_answer": final_ans, "final_confidence": final_conf}, indent=2, ensure_ascii=False))
                    else:
                        print(json.dumps(resp, indent=2))
                except Exception:
                    print(json.dumps(resp, indent=2))
