#!/usr/bin/env python3
"""
Maven Command‑Line Interface
============================

This script provides a simple interactive shell for users to interact with
Maven's cognitive brains.  It wraps the existing service APIs and exposes
common operations such as asking questions, evaluating proposed facts,
registering claims with the Self‑DMN, and performing maintenance tasks
like hum ticks, reflections and dissent scans.

Running the CLI
---------------

To start the interface, execute this script directly from the Maven root:

    python -m maven.ui.maven_cli

It will present a numbered menu.  Select an option and follow the prompts.
Results are printed in JSON for clarity.
"""

from __future__ import annotations
import sys, json, time, os
from pathlib import Path

# Adjust sys.path to ensure Maven modules are importable when run as a script
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brains.maven_paths import get_maven_root
MAVEN_ROOT = get_maven_root()

from api.utils import generate_mid

# -----------------------------------------------------------------------------
# Runtime mode selection
#
# The CLI supports an optional `--mode` flag which controls the level of
# verbosity for pipeline execution.  In ``architect`` mode (default), the full
# JSON response from the Memory Librarian is displayed, including all
# intermediate stages.  In ``execution`` mode, verbose reasoning context is
# suppressed and only the final answer and confidence are printed if available.
# The flag should appear before any interactive input begins.  Example:
#
#     python -m maven.ui.maven_cli --mode execution
#
# When running interactively via ``run_maven.py`` the mode is forwarded
# automatically.
MODE: str = "architect"

def _parse_mode_flag(argv: list[str]) -> list[str]:
    """Parse and remove --mode flag from argv.

    Returns the list of remaining arguments.
    """
    global MODE
    args = list(argv)
    if "--mode" in args:
        try:
            idx = args.index("--mode")
            if idx + 1 < len(args):
                MODE = args[idx + 1].strip().lower() or MODE
                del args[idx:idx + 2]
        except Exception:
            pass
    return args

# Import the agent executor if available.  The agent lives under
# ``brains.agent.service`` rather than inside the cognitive package.  If the
# import fails (e.g. agent module missing), set ``agent_executor`` to ``None``.
try:
    from brains.agent.service import agent_executor  # type: ignore
except Exception:
    agent_executor = None
from brains.cognitive.reasoning.service import reasoning_brain
from brains.cognitive.self_dmn.service import self_dmn_brain
# Memory Librarian for full pipeline execution
try:
    from brains.cognitive.memory_librarian.service import memory_librarian
except Exception:
    memory_librarian = None


def _print_json(data: object) -> None:
    """Pretty‑print a dict or other JSON‑serializable object."""
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(data)


def _ask_question() -> None:
    """Prompt the user for a question and send it to the Reasoning brain."""
    q = input("\nEnter your question: ").strip()
    if not q:
        print("No question provided.")
        return
    msg = {
        "op": "EVALUATE_FACT",
        "mid": generate_mid(),
        "payload": {
            "proposed_fact": {
                "original_query": q,
                "content": q,
                "storable_type": "QUESTION",
            }
        },
    }
    try:
        resp = reasoning_brain.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _evaluate_fact() -> None:
    """Prompt the user for a fact statement and evaluate it via the Reasoning brain."""
    content = input("\nEnter the fact content: ").strip()
    if not content:
        print("No content provided.")
        return
    orig = input("Original query (optional): ").strip()
    storable = input("Storable type (FACT/QUESTION/SPECULATION/COMMAND/REQUEST) [FACT]: ").strip() or "FACT"
    msg = {
        "op": "EVALUATE_FACT",
        "mid": generate_mid(),
        "payload": {
            "proposed_fact": {
                "original_query": orig,
                "content": content,
                "storable_type": storable.upper(),
            }
        },
    }
    try:
        resp = reasoning_brain.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _register_claim() -> None:
    """Prompt the user to register a claim with the Self‑DMN skeptic."""
    prop = input("\nEnter the claim proposition: ").strip()
    try:
        consensus = float(input("Consensus score (0..1): ").strip() or "0")
    except Exception:
        consensus = 0.0
    try:
        skeptic = float(input("Skeptic score (0..1): ").strip() or "0")
    except Exception:
        skeptic = 0.0
    expiry_input = input("Expiry timestamp (seconds since epoch) [24h from now]: ").strip()
    expiry = None
    if expiry_input:
        try:
            expiry = float(expiry_input)
        except Exception:
            expiry = None
    if expiry is None:
        expiry = time.time() + 24 * 3600
    msg = {
        "op": "REGISTER",
        "mid": generate_mid(),
        "payload": {
            "proposition": prop,
            "consensus_score": consensus,
            "skeptic_score": skeptic,
            "expiry": expiry,
        },
    }
    try:
        resp = self_dmn_brain.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _dmn_tick() -> None:
    """Advance the hum oscillators and display coherence and memory health."""
    msg = {"op": "TICK", "mid": generate_mid()}
    try:
        resp = self_dmn_brain.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _dmn_reflect() -> None:
    """Perform a reflection analysis via Self‑DMN."""
    try:
        window = int(input("Window (number of recent runs) [10]: ").strip() or "10")
    except Exception:
        window = 10
    msg = {"op": "REFLECT", "mid": generate_mid(), "payload": {"window": window}}
    try:
        resp = self_dmn_brain.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _dmn_dissent_scan() -> None:
    """Rescan recent claims and display their status via Self‑DMN."""
    try:
        window = int(input("Window (number of recent claims) [10]: ").strip() or "10")
    except Exception:
        window = 10
    msg = {"op": "DISSENT_SCAN", "mid": generate_mid(), "payload": {"window": window}}
    try:
        resp = self_dmn_brain.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")

def _run_pipeline() -> None:
    """Prompt the user for a query and execute the full Maven pipeline via the Memory Librarian."""
    if memory_librarian is None:
        print("Memory Librarian module unavailable.")
        return
    query = input("\nEnter your query for Maven: ").strip()
    if not query:
        print("No query provided.")
        return
    try:
        msg = {
            "op": "RUN_PIPELINE",
            "mid": generate_mid(),
            "payload": {"text": query, "confidence": 0.8},
        }
        resp = memory_librarian.service_api(msg)
        # Surface final answer in execution mode; otherwise fall back to
        # context printing.  Even in architect mode, we prefer the concise
        # answer if available.
        try:
            ctx = (resp.get("payload") or {}).get("context") or {}
            final_ans = ctx.get("final_answer")
            final_conf = ctx.get("final_confidence")
        except Exception:
            final_ans = None
            final_conf = None
        # Determine output based on MODE
        if MODE == "execution":
            if final_ans is not None:
                print(json.dumps({"final_answer": final_ans, "final_confidence": final_conf}, indent=2, ensure_ascii=False))
            else:
                # Fallback: print only the top-level payload
                try:
                    payload = resp.get("payload") or {}
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                except Exception:
                    _print_json(resp)
        else:
            # Architect mode: show final answer if present, else full context
            if final_ans is not None:
                print(json.dumps({"final_answer": final_ans, "final_confidence": final_conf}, indent=2, ensure_ascii=False))
            else:
                _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _agent_chat() -> None:
    """Interact with the agent executor via natural language."""
    if agent_executor is None:
        print("Agent module unavailable.")
        return
    text = input("\nEnter a request for the agent: ").strip()
    if not text:
        print("No input provided.")
        return
    msg = {"op": "CHAT", "mid": generate_mid(), "payload": {"text": text}}
    try:
        resp = agent_executor.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _agent_report() -> None:
    """Show the last report from the agent executor."""
    if agent_executor is None:
        print("Agent module unavailable.")
        return
    msg = {"op": "REPORT", "mid": generate_mid()}
    try:
        resp = agent_executor.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")


def _agent_status() -> None:
    """Display status information about the agent executor."""
    if agent_executor is None:
        print("Agent module unavailable.")
        return
    msg = {"op": "STATUS", "mid": generate_mid()}
    try:
        resp = agent_executor.service_api(msg)
        _print_json(resp)
    except Exception as e:
        print(f"Error: {e}")

# -----------------------------------------------------------------------------
# Health and readiness checks
#
# The Maven CLI exposes two special non‑interactive operations: ``health`` and
# ``ready``.  When invoked via ``python -m maven.ui.maven_cli health`` or
# ``ready``, these functions run a series of checks to determine if the
# underlying services and configuration are working correctly.  They can
# also be selected from the interactive menu when the agent is available.

def _check_health_cli() -> None:
    """Run a basic health check against the agent executor and report status."""
    # Health status is provided by the agent executor if available.  Fall back
    # to a simple message when the agent is absent.
    if agent_executor is None:
        print("Agent module unavailable. Health check not applicable.")
        return
    try:
        msg = {"op": "HEALTH", "mid": generate_mid()}
        resp = agent_executor.service_api(msg)
        status = (resp.get("status") if isinstance(resp, dict) else None) or {}
        print(json.dumps({"ok": resp.get("ok", False), "status": status}, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Health check error: {e}")


def _check_ready_cli() -> None:
    """Perform a readiness check of Maven configuration and environment."""
    # First run the health check to ensure the agent layer is operational
    _check_health_cli()
    # Assemble readiness diagnostics
    issues: list[str] = []
    # Verify required config files exist
    required_configs = [
        MAVEN_ROOT / "config" / "io_schema.json",
        MAVEN_ROOT / "config" / "autonomy.json",
        MAVEN_ROOT / "config" / "memory.json",
    ]
    for cfg in required_configs:
        if not cfg.exists():
            issues.append(f"Missing config file: {cfg.relative_to(MAVEN_ROOT)}")
    # Check write permission on reports directory
    reports_dir = MAVEN_ROOT / "reports"
    try:
        test_path = reports_dir / ".write_test"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_path, "w", encoding="utf-8") as fh:
            fh.write("ok")
        test_path.unlink(missing_ok=True)
    except Exception:
        issues.append("Cannot write to reports directory")
    # Check environment variables used by Maven
    # If API keys or secrets are required, they should be present in env.  We
    # check for optional presence but do not reveal them.  Report only missing keys.
    if os.environ.get("OPENAI_API_KEY") in {None, ""}:
        issues.append("OPENAI_API_KEY environment variable not set (may be required for certain tasks)")
    # Collate results
    if issues:
        print(json.dumps({"ready": False, "issues": issues}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"ready": True}, indent=2, ensure_ascii=False))


def main() -> None:
    """Main entry point for the Maven CLI."""
    # Parse any --mode flag and remove it from argv.  Update MODE accordingly.
    args = _parse_mode_flag(sys.argv)
    # Support non‑interactive operations when passed as positional arguments.  If
    # ``health`` or ``ready`` is the first remaining argument, run the
    # corresponding check and exit.  This allows one‑shot diagnostics via
    # ``python -m maven.ui.maven_cli health``.
    if len(args) > 1:
        cmd = args[1].lower()
        if cmd == "health":
            _check_health_cli()
            return
        if cmd == "ready":
            _check_ready_cli()
            return
    # Enter the interactive menu loop.  Options are numbered for clarity.
    while True:
        print("\nMaven CLI – choose an option:")
        print("  1. Ask a question")
        print("  2. Evaluate a fact")
        print("  3. Register a claim (Self‑DMN)")
        print("  4. Self‑DMN TICK (advance oscillators)")
        print("  5. Self‑DMN REFLECT (analysis)")
        print("  6. Self‑DMN DISSENT SCAN")
        print("  7. Run full Maven pipeline (talk to Maven)")
        print("  8. Agent: Chat (natural language)")
        print("  9. Agent: Show last report")
        print(" 10. Agent: Status")
        print(" 11. Chat with Maven (natural language)")
        print(" 12. Health check (agent)")
        print(" 13. Readiness check (system)")
        print(" 14. Exit")
        choice = input("Select option [1‑14]: ").strip()
        if choice == "1":
            _ask_question()
        elif choice == "2":
            _evaluate_fact()
        elif choice == "3":
            _register_claim()
        elif choice == "4":
            _dmn_tick()
        elif choice == "5":
            _dmn_reflect()
        elif choice == "6":
            _dmn_dissent_scan()
        elif choice == "7":
            _run_pipeline()
        elif choice == "8":
            _agent_chat()
        elif choice == "9":
            _agent_report()
        elif choice == "10":
            _agent_status()
        elif choice == "11":
            # Natural language chat with Maven (not the agent)
            try:
                from maven.ui.maven_chat import repl as chat_repl  # type: ignore
            except Exception:
                print("Natural language interface unavailable.")
            else:
                chat_repl()
        elif choice == "12":
            _check_health_cli()
        elif choice == "13":
            _check_ready_cli()
        elif choice == "14":
            print("Exiting Maven CLI.")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Maven CLI.")