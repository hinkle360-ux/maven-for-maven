"""
Command Router
==============

This module provides a lightweight command router for handling
command‑style inputs (e.g. ``--status`` or ``--cache purge``) within
Maven's cognitive pipeline.  When the language brain detects that the
user input is a command (via the ``--`` or ``/`` prefix), the memory
librarian can delegate the query to this router instead of invoking the
full reasoning pipeline.  The router looks up built‑in commands,
executes the corresponding handler functions and returns a short
message describing the result.  Unknown commands yield a structured
error message to avoid generic filler responses.

Each handler returns a dictionary with either a ``message`` key
containing a user‑friendly string or an ``error`` key describing
why the command could not be completed.  The memory librarian is
responsible for formatting this into the final answer and assigning
confidence.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

from brains.maven_paths import get_reports_path

# Import execution guard for enable/disable commands
try:
    from brains.tools.execution_guard import (
        enable_execution_full,
        enable_execution_read_only,
        disable_execution,
        execution_status_snapshot,
        get_execution_status,
    )
    _execution_guard_available = True
except ImportError:
    _execution_guard_available = False

# Import self_model for scan self command
try:
    from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api
    _self_model_available = True
except ImportError:
    _self_model_available = False
    self_model_api = None


def _load_command_registry() -> Dict[str, Any]:
    """Load the command registry from ``config/commands.json``.

    Returns an empty dictionary if the file is missing or malformed.
    The registry is not strictly required by the router; handlers can
    be hardcoded below.  However, the file provides a structured
    reference for available commands and their descriptions.
    """
    try:
        # Determine the Maven project root by walking up from this file
        root = Path(__file__).resolve().parents[2]
        cfg_path = root / "config" / "commands.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _handle_status() -> Dict[str, Any]:
    """Return a summary of the autonomous agent's state.

    The report includes whether the background agent daemon is running,
    the number of active goals and the total number of goals.  If the
    goal queue cannot be loaded, an error is returned instead of
    raising.
    """
    try:
        from brains.agent.autonomous.goal_queue import GoalQueue  # type: ignore
        # Instantiate the goal queue.  This reads goals from disk without
        # spawning or interacting with the agent daemon.
        queue = GoalQueue()
        active_goals = queue.load_goals(active_only=True)
        all_goals = queue.load_goals(active_only=False)
        # Determine the running state of the agent daemon if possible.
        running = False
        try:
            from brains.agent.autonomous.agent_daemon import AgentDaemon  # type: ignore
            # Do not start a new daemon thread; instead instantiate and
            # inspect the ``running`` attribute.  In typical usage the
            # agent daemon will set this flag to True when running.  If
            # instantiation raises, default to False.
            daemon = AgentDaemon()
            running = bool(getattr(daemon, "running", False))
        except Exception:
            running = False
        # Build a compact JSON report for the status.  This can be
        # pretty‑printed by the caller if desired.  Titles are
        # truncated to avoid overly long lists.
        report = {
            "running": running,
            "active_goals": len(active_goals),
            "total_goals": len(all_goals),
            "active_goal_titles": [g.get("title", g.get("goal_id")) for g in active_goals],
        }
        return {"message": json.dumps(report, ensure_ascii=False)}
    except Exception as e:
        return {"error": f"status_failed: {e}"}


def _handle_cache_purge() -> Dict[str, Any]:
    """Remove the fast cache file if it exists.

    Deleting the fast cache forces the pipeline to recompute answers on
    subsequent runs.  Any error during deletion is returned in the
    ``error`` key.
    """
    try:
        cache_path = get_reports_path("fast_cache.jsonl")
        if cache_path.exists():
            cache_path.unlink()
            return {"message": "Fast cache purged."}
        else:
            return {"message": "Fast cache is already empty."}
    except Exception as e:
        return {"error": f"cache_purge_failed: {e}"}


def _handle_input(args: List[str]) -> Dict[str, Any]:
    """Placeholder handler for the ``input`` command.

    In a future implementation this function could ingest external
    knowledge or domain data into Maven.  Until then it simply
    returns an error indicating that the command is unsupported.
    """
    return {"error": "input_not_supported"}


def _handle_scan_self() -> Dict[str, Any]:
    """Handle 'scan self' command - invoke self_model.system_scan."""
    if not _self_model_available or self_model_api is None:
        return {"error": "self_model not available"}

    try:
        result = self_model_api({"op": "SYSTEM_SCAN", "payload": {}})
        if result.get("ok"):
            payload = result.get("payload", {})
            # Include execution status in scan
            if _execution_guard_available:
                exec_status = get_execution_status()
                # Use to_dict() for JSON serialization
                payload["execution_status"] = exec_status.to_dict()
            return {"message": json.dumps(payload, indent=2)}
        else:
            error = result.get("error", {})
            return {"error": f"scan_failed: {error.get('message', 'unknown')}"}
    except Exception as e:
        return {"error": f"scan_self_failed: {e}"}


def _handle_scan_memory() -> Dict[str, Any]:
    """Handle 'scan memory' command - check memory health."""
    if not _self_model_available or self_model_api is None:
        return {"error": "self_model not available"}

    try:
        result = self_model_api({"op": "MEMORY_HEALTH", "payload": {}})
        if result.get("ok"):
            return {"message": json.dumps(result.get("payload", {}), indent=2)}
        else:
            error = result.get("error", {})
            return {"error": f"memory_scan_failed: {error.get('message', 'unknown')}"}
    except Exception as e:
        return {"error": f"scan_memory_failed: {e}"}


def _handle_whoami(detail_level: str = "full") -> Dict[str, Any]:
    """Handle 'whoami' or 'introduce' command - generate dynamic self-introduction.

    This invokes the SELF_INTRODUCTION operation in self_model to generate
    a comprehensive introduction based on real introspection:
    - Scans the codebase structure
    - Reads memory statistics
    - Checks active capabilities
    - Generates natural language introduction

    Args:
        detail_level: "brief", "standard", or "full"
    """
    if not _self_model_available or self_model_api is None:
        return {"error": "self_model not available"}

    try:
        result = self_model_api({
            "op": "SELF_INTRODUCTION",
            "payload": {"detail_level": detail_level}
        })
        if result.get("ok"):
            payload = result.get("payload", {})
            # Return both the natural language introduction and structured data
            intro_text = payload.get("introduction_text", "I am Maven.")
            output = {
                "introduction": intro_text,
                "identity": payload.get("identity", {}),
                "capabilities_summary": payload.get("capabilities", {}).get("summary", "unknown"),
                "architecture": {
                    "cognitive_brains": payload.get("architecture", {}).get("cognitive_brains", 0),
                    "domain_banks": payload.get("architecture", {}).get("domain_banks", 0),
                },
                "memory_records": payload.get("memory_stats", {}).get("total_records", 0),
                "generated_at": payload.get("generated_at", "")
            }
            return {"message": intro_text, "data": output}
        else:
            error = result.get("error", {})
            return {"error": f"whoami_failed: {error.get('message', 'unknown')}"}
    except Exception as e:
        return {"error": f"whoami_failed: {e}"}


def _handle_enable_execution(mode: str = "full") -> Dict[str, Any]:
    """Handle 'enable execution' command - enable execution with confirmation.

    Args:
        mode: Either "full" or "read-only"
    """
    if not _execution_guard_available:
        return {"error": "execution_guard not available"}

    try:
        if mode == "read-only":
            cfg = enable_execution_read_only("user requested READ_ONLY mode via chat")
            return {
                "ok": True,
                "kind": "execution_config",
                "message": "Execution mode set to READ_ONLY.",
                "config": {
                    "mode": cfg.mode.value,
                    "user_confirmed": cfg.user_confirmed,
                    "last_updated": cfg.last_updated,
                    "reason": cfg.updated_reason,
                },
            }
        else:
            cfg = enable_execution_full("user requested FULL mode via chat")
            return {
                "ok": True,
                "kind": "execution_config",
                "message": "Execution mode set to FULL with user confirmation.",
                "config": {
                    "mode": cfg.mode.value,
                    "user_confirmed": cfg.user_confirmed,
                    "last_updated": cfg.last_updated,
                    "reason": cfg.updated_reason,
                },
            }
    except Exception as e:
        return {"error": f"enable_execution_failed: {e}"}


def _handle_disable_execution() -> Dict[str, Any]:
    """Handle 'disable execution' command - disable execution."""
    if not _execution_guard_available:
        return {"error": "execution_guard not available"}

    try:
        cfg = disable_execution("user disabled execution via chat")
        return {
            "ok": True,
            "kind": "execution_config",
            "message": "Execution mode set to DISABLED.",
            "config": {
                "mode": cfg.mode.value,
                "user_confirmed": cfg.user_confirmed,
                "last_updated": cfg.last_updated,
                "reason": cfg.updated_reason,
            },
        }
    except Exception as e:
        return {"error": f"disable_execution_failed: {e}"}


def _handle_execution_status() -> Dict[str, Any]:
    """Handle 'execution status' command - show current execution status."""
    if not _execution_guard_available:
        return {"error": "execution_guard not available"}

    try:
        snap = execution_status_snapshot()
        return {
            "ok": True,
            "kind": "execution_status",
            "message": json.dumps(snap, indent=2),
            "status": snap,
        }
    except Exception as e:
        return {"error": f"execution_status_failed: {e}"}


def _handle_run_tests(args: List[str]) -> Dict[str, Any]:
    """Handle 'run tests' command - run test suite via tool interface."""
    try:
        from brains.tools_api import ShellTool, NullShellTool, ToolRegistry

        # Try to get shell tool from tool registry
        shell_tool = None
        try:
            from brains.agent.tools.shell_tool import get_shell_tool
            shell_tool = get_shell_tool()
        except Exception:
            pass

        if shell_tool is None or isinstance(shell_tool, NullShellTool):
            return {"error": "run_tests_failed: Shell tool not available. Tests cannot be run in offline mode."}

        root = Path(__file__).resolve().parents[2]
        test_path = args[0] if args else "tests/"

        # Run pytest via shell tool
        cmd = f"python -m pytest {test_path} -v --tb=short"
        result = shell_tool.run(cmd, cwd=str(root), timeout=60, check_policy=False)

        output = (result.stdout or "") + (result.stderr or "")
        if result.status == "completed" and result.exit_code == 0:
            return {"message": f"Tests passed:\n{output[:2000]}"}
        elif result.status == "timeout":
            return {"error": "test_timeout: tests took longer than 60 seconds"}
        else:
            return {"message": f"Tests failed (exit code {result.exit_code}):\n{output[:2000]}"}
    except Exception as e:
        return {"error": f"run_tests_failed: {e}"}


def _handle_list_tools() -> Dict[str, Any]:
    """Handle 'list tools' command - enumerate available tools and capabilities."""
    tools_info = {
        "action_engine": {
            "available": True,
            "capabilities": [
                "list_python_files", "read_file", "write_file", "append_file",
                "run_python_sandbox", "git_status", "git_diff", "git_commit",
                "hot_reload_module"
            ]
        },
        "browser_tool": {
            "available": True,
            "capabilities": ["web_search", "navigate", "read_page", "screenshot"]
        },
        "coder_brain": {
            "available": True,
            "capabilities": ["plan", "generate", "verify", "refine"]
        },
        "execution_guard": {
            "available": _execution_guard_available,
            "capabilities": ["check_operation", "grant_permission", "revoke_permission"]
        },
        "self_model": {
            "available": _self_model_available,
            "capabilities": ["system_scan", "memory_health", "identity_check"]
        }
    }

    return {"message": json.dumps(tools_info, indent=2)}


def route_command(command_text: str) -> Dict[str, Any]:
    """Dispatch a command string to the appropriate handler.

    The input should begin with a ``--`` or ``/`` prefix.  Leading
    prefix characters are stripped before lookup.  If a subcommand is
    present (e.g. ``cache purge``), the first and second tokens are
    considered separately.  Unknown commands or subcommands result in
    an ``error`` entry describing the failure.

    Args:
        command_text: The raw command string as entered by the user.

    Returns:
        A dictionary with either a ``message`` or ``error`` key.
    """
    try:
        if not command_text:
            return {"error": "empty_command"}
        # Split tokens by whitespace.  Do not strip trailing whitespace
        # on the entire command so that commands like "--cache purge" are
        # parsed correctly.
        tokens = command_text.strip().split()
        if not tokens:
            return {"error": "empty_command"}
        # Remove leading dashes or slashes from the first token.  Allow
        # multiple leading prefixes (e.g. "--status" or "/status").
        cmd = tokens[0].lstrip("-/")
        cmd_lower = cmd.lower()
        # ------------------------------------------------------------------
        # Natural‑language command pre‑processing.  In addition to CLI
        # commands beginning with "--" or "/", the memory librarian may
        # route spoken imperatives such as "you say hello" or "say hi"
        # through this router.  To support these social actions, detect
        # second‑person prefixes ("you" or "u") and remap the command to
        # the next token.  For example, "you say hello" is treated as
        # "say hello".  The remainder of the tokens are passed as
        # arguments.  This enables simple etiquette triggers without
        # polluting the core command namespace.
        args: List[str] = []
        try:
            if cmd_lower in ("you", "u") and len(tokens) > 1:
                cmd_lower = tokens[1].lower()
                args = tokens[2:]
            else:
                args = tokens[1:]
        except Exception:
            args = tokens[1:]
        # ------------------------------------------------------------------
        # Handle built‑in commands.  Additional commands can be
        # registered in config/commands.json without changing this code.
        if cmd_lower in ("status", "agent_status"):
            return _handle_status()
        if cmd_lower == "cache":
            # Determine subcommand if present.  Accept synonyms for purge.
            sub = args[0].lower() if args else ""
            if sub in ("purge", "clear", "reset"):
                return _handle_cache_purge()
            # Unknown subcommand; provide explicit guidance
            return {"error": f"unknown_cache_command: {sub or 'missing_subcommand'}"}
        if cmd_lower == "input":
            # The input command expects at least one argument (e.g. a file path).
            # If no arguments are provided, return a clarifying message instead
            # of an error to avoid triggering high‑effort fallback later in the
            # pipeline.  When arguments are present, pass them to the
            # placeholder handler.
            if not args:
                return {
                    "message": "The input command requires a file path. Example: --input /path/to/file"
                }
            return _handle_input(args)
        # ------------------------------------------------------------------
        # Scan commands: scan self, scan memory
        if cmd_lower == "scan":
            sub = args[0].lower() if args else ""
            if sub == "self":
                return _handle_scan_self()
            if sub == "memory":
                return _handle_scan_memory()
            return {"error": f"unknown_scan_target: {sub or 'missing_target'}. Use 'scan self' or 'scan memory'"}
        # ------------------------------------------------------------------
        # Execution control: enable execution, disable execution, set read-only, execution status
        if cmd_lower == "enable":
            sub = args[0].lower() if args else ""
            sub2 = args[1].lower() if len(args) > 1 else ""
            # "enable full execution" or "enable execution"
            if sub in ("execution", "full"):
                return _handle_enable_execution(mode="full")
            # "enable read-only execution" or "enable read-only"
            if sub == "read-only":
                return _handle_enable_execution(mode="read-only")
            return {"error": f"unknown_enable_target: {sub or 'missing_target'}. Use 'enable execution', 'enable full execution', or 'enable read-only'"}
        if cmd_lower == "disable":
            sub = args[0].lower() if args else ""
            if sub == "execution":
                return _handle_disable_execution()
            return {"error": f"unknown_disable_target: {sub or 'missing_target'}. Use 'disable execution'"}
        # "set read-only" command
        if cmd_lower == "set":
            sub = args[0].lower() if args else ""
            if sub == "read-only":
                return _handle_enable_execution(mode="read-only")
            return {"error": f"unknown_set_target: {sub or 'missing_target'}. Use 'set read-only'"}
        # "execution status" or "show execution status"
        if cmd_lower == "execution":
            sub = args[0].lower() if args else ""
            if sub == "status":
                return _handle_execution_status()
            return {"error": f"unknown_execution_command: {sub or 'missing_subcommand'}. Use 'execution status'"}
        if cmd_lower == "show":
            sub = " ".join(args[:2]).lower() if len(args) >= 2 else (args[0].lower() if args else "")
            if sub == "execution status" or sub == "execution":
                return _handle_execution_status()
            # Could add more "show X" commands here
            return {"error": f"unknown_show_target: {sub or 'missing_target'}. Use 'show execution status'"}
        # ------------------------------------------------------------------
        # Test runner: run tests
        if cmd_lower == "run":
            sub = args[0].lower() if args else ""
            if sub == "tests":
                return _handle_run_tests(args[1:])
            return {"error": f"unknown_run_target: {sub or 'missing_target'}. Use 'run tests'"}
        # ------------------------------------------------------------------
        # List tools: list tools
        if cmd_lower == "list":
            sub = args[0].lower() if args else ""
            if sub == "tools":
                return _handle_list_tools()
            return {"error": f"unknown_list_target: {sub or 'missing_target'}. Use 'list tools'"}
        # ------------------------------------------------------------------
        # Self-introduction: whoami, introduce, who are you
        # Generates dynamic introduction from real introspection
        if cmd_lower in ("whoami", "introduce", "introduction"):
            # Determine detail level from args
            detail_level = "full"
            if args:
                sub = args[0].lower()
                if sub in ("brief", "short"):
                    detail_level = "brief"
                elif sub in ("standard", "normal"):
                    detail_level = "standard"
            return _handle_whoami(detail_level)
        # Handle "who are you" style questions routed as commands
        if cmd_lower == "who":
            sub = " ".join(args).lower() if args else ""
            if sub.startswith("are you") or sub.startswith("am i") or sub == "":
                return _handle_whoami("full")
        # ------------------------------------------------------------------
        # Social speech act: say/speak.  Respond by echoing the provided
        # phrase, capitalising the first letter and preserving the rest.
        # When the phrase appears to be a common greeting or thanks, add
        # an exclamation mark to convey warmth.  Record the behavioural
        # rule in a JSONL file under reports/behavior_rules.jsonl to
        # enable future learning of social patterns.  Errors during
        # storage are ignored to avoid breaking command routing.
        if cmd_lower in ("say", "speak", "tell"):
            phrase = " ".join(args).strip()
            if not phrase:
                return {"error": "nothing_to_say"}
            # Persist the rule for learning behavioural patterns
            try:
                root = Path(__file__).resolve().parents[2]
                beh_file = root / "reports" / "behavior_rules.jsonl"
                beh_file.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "cmd": cmd_lower,
                    "phrase": phrase
                }
                with open(beh_file, "a", encoding="utf-8") as fh:
                    import json as _json
                    fh.write(_json.dumps(rec) + "\n")
            except Exception:
                pass
            # Compose a friendly response
            # Basic capitalisation
            resp = phrase[0].upper() + phrase[1:] if phrase else phrase
            # Add warmth for common salutations or thanks
            try:
                pl = phrase.lower()
            except Exception:
                pl = phrase
            if pl in ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"):
                # Add an exclamation for enthusiasm and a follow‑up question
                resp = resp.rstrip("!") + "! How can I assist you?"
            elif "thank" in pl:
                # Respond politely to thanks
                resp = resp.rstrip("!") + "! You're welcome."
            return {"message": resp}
        # ------------------------------------------------------------------
        # Unknown command
        return {"error": f"unknown_command: {cmd_lower}"}
    except Exception as exc:
        return {"error": f"command_router_failure: {exc}"}
