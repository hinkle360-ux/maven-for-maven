"""
Tool Intent Resolver

Purpose:
- Take a raw user utterance and decide:
    - Is this a *tool action* (fs/git/exec) or just a language question?
    - If it is a tool action, construct a concrete ActionRequest for the Action Engine.

This sits *after*:
    - self-intent gate (which intercepts "scan self", "scan memory", etc.)
and *before*:
    - Teacher / general reasoning

Behavior:
- Recognizes common imperative patterns:
    - "Create directory X"
    - "Write file X with content: Y"
    - "Read file X [exactly as bytes]"
    - "Append to file X with content: Y"
    - "List files in X"
    - "Git status" / "Git diff"
    - "Attempt to execute shell command: CMD"
- Returns:
    - None if not a tool intent (pipeline proceeds as usual).
    - A structured dict with action + params if it is.

No stubs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Import ActionEngineBrain and ActionRequest
try:
    from brains.action_engine_brain import ActionEngineBrain, ActionRequest
    _action_engine_available = True
except ImportError as e:
    logger.warning("ActionEngineBrain not available: %s", e)
    _action_engine_available = False
    ActionEngineBrain = None  # type: ignore
    ActionRequest = None  # type: ignore


@dataclass
class ToolIntentResult:
    """Result of tool intent resolution."""
    is_tool_intent: bool
    reason: str
    action_request: Optional[Any] = None  # ActionRequest if available


class ToolIntentResolver:
    """
    Stateless resolver that parses a single user message.

    It does *not* execute anything itself. It only decides:
    - which action
    - which params

    Execution is delegated to ActionEngineBrain.
    """

    def __init__(self, action_engine: Optional[Any] = None) -> None:
        """
        Initialize the resolver.

        Args:
            action_engine: Optional ActionEngineBrain instance for execution.
                           If None, resolve() still works but maybe_execute() won't execute.
        """
        self._engine = action_engine

        # Precompile regexes for speed and clarity.
        self._re_create_dir = re.compile(
            r"^(create|make)\s+directory\s+([^\s]+)", re.IGNORECASE
        )
        self._re_write_file = re.compile(
            r"^write\s+file\s+(\S+)\s+with\s+content:\s*(.+)$",
            re.IGNORECASE | re.DOTALL,
        )
        self._re_append_file = re.compile(
            r"^append\s+(?:to\s+)?file\s+(\S+)\s+with\s+content:\s*(.+)$",
            re.IGNORECASE | re.DOTALL,
        )
        self._re_read_file = re.compile(
            r"^read\s+file\s+(\S+)(.*)$",
            re.IGNORECASE | re.DOTALL,
        )
        self._re_list_files = re.compile(
            r"^list\s+(?:python\s+)?files(?:\s+in\s+(\S+))?",
            re.IGNORECASE,
        )
        self._re_git_status = re.compile(
            r"^git\s+status$",
            re.IGNORECASE,
        )
        self._re_git_diff = re.compile(
            r"^git\s+diff(?:\s+(.+))?$",
            re.IGNORECASE,
        )
        self._re_shell_exec = re.compile(
            r"^(?:attempt\s+to\s+)?execute\s+(?:shell\s+)?command:\s*(.+)$",
            re.IGNORECASE | re.DOTALL,
        )
        self._re_run_python = re.compile(
            r"^run\s+python(?:\s+code)?:\s*(.+)$",
            re.IGNORECASE | re.DOTALL,
        )

        # Additional natural language patterns for FULL_AGENCY mode
        # File operations - more natural patterns
        self._re_open_file = re.compile(
            r"^(?:open|show\s+me|display|cat)\s+(.+)$",
            re.IGNORECASE,
        )
        self._re_edit_file = re.compile(
            r"^(?:edit|modify|change|update)\s+(?:file\s+)?(.+?)(?:\s+to\s+(.+))?$",
            re.IGNORECASE | re.DOTALL,
        )
        self._re_delete_file = re.compile(
            r"^(?:delete|remove|rm)\s+(?:file\s+)?(.+)$",
            re.IGNORECASE,
        )
        self._re_search_files = re.compile(
            r"^(?:find|search|grep|look\s+for)\s+(.+?)\s+(?:in|inside|within)\s+(.+)$",
            re.IGNORECASE,
        )

        # Shell/command patterns - more natural
        self._re_run_command = re.compile(
            r"^(?:run|execute|do)\s+(?:the\s+)?(?:command\s+)?[`'\"]?(.+?)[`'\"]?$",
            re.IGNORECASE,
        )
        self._re_install_package = re.compile(
            r"^(?:install|add)\s+(?:package\s+)?(.+?)(?:\s+(?:using|with)\s+(.+))?$",
            re.IGNORECASE,
        )

        # Git patterns - more natural
        self._re_git_commit_msg = re.compile(
            r"^(?:commit|git\s+commit)(?:\s+with\s+message)?[:\s]+['\"]?(.+?)['\"]?$",
            re.IGNORECASE,
        )
        self._re_git_push = re.compile(
            r"^(?:push|git\s+push)(?:\s+to\s+(.+))?$",
            re.IGNORECASE,
        )
        self._re_git_pull = re.compile(
            r"^(?:pull|git\s+pull)(?:\s+from\s+(.+))?$",
            re.IGNORECASE,
        )
        self._re_git_clone = re.compile(
            r"^(?:clone|git\s+clone)\s+(.+)$",
            re.IGNORECASE,
        )

        # Web patterns
        self._re_web_search = re.compile(
            r"^(?:search|google|look\s+up|find)\s+(?:for\s+)?(?:on\s+the\s+web\s+)?(.+)$",
            re.IGNORECASE,
        )
        self._re_web_fetch = re.compile(
            r"^(?:fetch|get|download|open\s+url|browse\s+to)\s+(https?://\S+)$",
            re.IGNORECASE,
        )

    # ------------- public API -------------

    def _check_execution_allowed(self) -> tuple:
        """Check if tool execution is allowed based on current profile/mode."""
        try:
            from brains.tools.execution_guard import get_execution_status, ExecMode
            status = get_execution_status()

            # SAFE_CHAT mode - no tools allowed
            if status.mode == ExecMode.SAFE_CHAT:
                return False, "SAFE_CHAT mode - no tool execution allowed"

            # DISABLED mode - no tools allowed
            if status.mode == ExecMode.DISABLED:
                return False, "Execution disabled"

            # FULL_AGENCY mode - all tools allowed
            if status.mode == ExecMode.FULL_AGENCY and status.effective:
                return True, ""

            # FULL mode - most tools allowed (with some restrictions)
            if status.mode == ExecMode.FULL and status.effective:
                return True, ""

            # READ_ONLY mode - only read operations allowed
            if status.mode == ExecMode.READ_ONLY:
                return True, "read_only"  # Caller should check operation type

            # Default: not allowed
            return False, status.reason or "Execution not enabled"
        except Exception as e:
            logger.warning("Could not check execution status: %s", e)
            return False, f"Could not check execution status: {e}"

    def resolve(self, user_text: str) -> ToolIntentResult:
        """
        Attempt to interpret `user_text` as a tool action.

        Returns:
            ToolIntentResult:
                - is_tool_intent=False, reason=...   → pipeline should ignore this resolver.
                - is_tool_intent=True + ActionRequest → caller should execute via action engine.
        """
        text = user_text.strip()
        if not text:
            return ToolIntentResult(False, "empty_text")

        if not _action_engine_available:
            return ToolIntentResult(False, "action_engine_not_available")

        # Check if execution is allowed
        allowed, reason = self._check_execution_allowed()
        if not allowed:
            return ToolIntentResult(False, f"execution_blocked: {reason}")

        # Order matters: we try more specific patterns first.

        # 1) Create directory
        m = self._re_create_dir.match(text)
        if m:
            path = m.group(2).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="create_directory",
                action_request=ActionRequest(
                    action="mkdir",
                    params={"path": path, "exist_ok": True},
                ),
            )

        # 2) Write file with content
        m = self._re_write_file.match(text)
        if m:
            path = m.group(1).strip()
            content = m.group(2)
            return ToolIntentResult(
                is_tool_intent=True,
                reason="write_file",
                action_request=ActionRequest(
                    action="write_text_file",
                    params={
                        "path": path,
                        "content": content,
                        "backup": True,
                    },
                ),
            )

        # 3) Append to file with content
        m = self._re_append_file.match(text)
        if m:
            path = m.group(1).strip()
            content = m.group(2)
            return ToolIntentResult(
                is_tool_intent=True,
                reason="append_file",
                action_request=ActionRequest(
                    action="append_text_file",
                    params={
                        "path": path,
                        "content": content,
                        "backup": True,
                    },
                ),
            )

        # 4) Read file (text or bytes)
        m = self._re_read_file.match(text)
        if m:
            path = m.group(1).strip()
            rest = (m.group(2) or "").lower()
            if "byte" in rest:
                return ToolIntentResult(
                    is_tool_intent=True,
                    reason="read_file_bytes",
                    action_request=ActionRequest(
                        action="read_bytes_file",
                        params={
                            "path": path,
                            "max_bytes": 100_000,
                        },
                    ),
                )
            else:
                return ToolIntentResult(
                    is_tool_intent=True,
                    reason="read_file_text",
                    action_request=ActionRequest(
                        action="read_text_file",
                        params={
                            "path": path,
                            "max_chars": 100_000,
                        },
                    ),
                )

        # 5) List files
        m = self._re_list_files.match(text)
        if m:
            # path = m.group(1) if m.group(1) else None  # Currently not used
            return ToolIntentResult(
                is_tool_intent=True,
                reason="list_files",
                action_request=ActionRequest(
                    action="list_python_files",
                    params={
                        "max_files": 500,
                        "include_tests": True,
                    },
                ),
            )

        # 6) Git status
        m = self._re_git_status.match(text)
        if m:
            return ToolIntentResult(
                is_tool_intent=True,
                reason="git_status",
                action_request=ActionRequest(
                    action="git_status",
                    params={},
                ),
            )

        # 7) Git diff
        m = self._re_git_diff.match(text)
        if m:
            paths_str = m.group(1)
            paths = paths_str.split() if paths_str else None
            return ToolIntentResult(
                is_tool_intent=True,
                reason="git_diff",
                action_request=ActionRequest(
                    action="git_diff",
                    params={"paths": paths},
                ),
            )

        # 8) Run Python code
        m = self._re_run_python.match(text)
        if m:
            code = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="run_python",
                action_request=ActionRequest(
                    action="run_python_sandbox",
                    params={
                        "code": code,
                        "timeout_ms": 3_000,
                    },
                ),
            )

        # 9) Attempt to execute shell command
        m = self._re_shell_exec.match(text)
        if m:
            cmd = m.group(1).strip()
            # Use the run_shell action which delegates to the host-provided ShellTool.
            # The ShellTool handles policy enforcement and execution safely.
            return ToolIntentResult(
                is_tool_intent=True,
                reason="run_shell",
                action_request=ActionRequest(
                    action="run_shell",
                    params={
                        "command": cmd,
                        "timeout": 30,
                    },
                ),
            )

        # =====================================================================
        # FULL_AGENCY MODE - Additional Natural Language Patterns
        # =====================================================================

        # 10) Open/show file (natural language)
        m = self._re_open_file.match(text)
        if m:
            path = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="read_file_natural",
                action_request=ActionRequest(
                    action="read_text_file",
                    params={
                        "path": path,
                        "max_chars": 100_000,
                    },
                ),
            )

        # 11) Delete file
        m = self._re_delete_file.match(text)
        if m:
            path = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="delete_file",
                action_request=ActionRequest(
                    action="delete_file",
                    params={
                        "path": path,
                    },
                ),
            )

        # 12) Run command (natural language)
        m = self._re_run_command.match(text)
        if m:
            cmd = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="run_shell_natural",
                action_request=ActionRequest(
                    action="run_shell",
                    params={
                        "command": cmd,
                        "timeout": 60,
                    },
                ),
            )

        # 13) Install package
        m = self._re_install_package.match(text)
        if m:
            package = m.group(1).strip()
            pkg_manager = m.group(2) if m.lastindex >= 2 and m.group(2) else "pip"
            if pkg_manager.lower() in ("pip", "pip3"):
                cmd = f"pip install {package}"
            elif pkg_manager.lower() in ("npm", "node"):
                cmd = f"npm install {package}"
            elif pkg_manager.lower() in ("apt", "apt-get"):
                cmd = f"apt-get install -y {package}"
            else:
                cmd = f"{pkg_manager} install {package}"
            return ToolIntentResult(
                is_tool_intent=True,
                reason="install_package",
                action_request=ActionRequest(
                    action="run_shell",
                    params={
                        "command": cmd,
                        "timeout": 120,
                    },
                ),
            )

        # 14) Git commit with message
        m = self._re_git_commit_msg.match(text)
        if m:
            message = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="git_commit",
                action_request=ActionRequest(
                    action="git_commit",
                    params={
                        "message": message,
                    },
                ),
            )

        # 15) Git push
        m = self._re_git_push.match(text)
        if m:
            remote = m.group(1).strip() if m.lastindex >= 1 and m.group(1) else "origin"
            return ToolIntentResult(
                is_tool_intent=True,
                reason="git_push",
                action_request=ActionRequest(
                    action="git_push",
                    params={
                        "remote": remote,
                    },
                ),
            )

        # 16) Git pull
        m = self._re_git_pull.match(text)
        if m:
            remote = m.group(1).strip() if m.lastindex >= 1 and m.group(1) else "origin"
            return ToolIntentResult(
                is_tool_intent=True,
                reason="git_pull",
                action_request=ActionRequest(
                    action="git_pull",
                    params={
                        "remote": remote,
                    },
                ),
            )

        # 17) Git clone
        m = self._re_git_clone.match(text)
        if m:
            url = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="git_clone",
                action_request=ActionRequest(
                    action="git_clone",
                    params={
                        "url": url,
                    },
                ),
            )

        # 18) Web fetch
        m = self._re_web_fetch.match(text)
        if m:
            url = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="web_fetch",
                action_request=ActionRequest(
                    action="web_fetch",
                    params={
                        "url": url,
                    },
                ),
            )

        # 19) Web search (lower priority - catches general search requests)
        m = self._re_web_search.match(text)
        if m:
            query = m.group(1).strip()
            return ToolIntentResult(
                is_tool_intent=True,
                reason="web_search",
                action_request=ActionRequest(
                    action="web_search",
                    params={
                        "query": query,
                    },
                ),
            )

        # Not a tool intent this resolver understands
        return ToolIntentResult(False, "no_match")

    # ------------- execution helper -------------

    def maybe_execute(self, user_text: str) -> Dict[str, Any]:
        """
        Convenience helper:

        - Runs `resolve`.
        - If not a tool intent, returns { "handled": False }.
        - If yes, executes via ActionEngineBrain and returns:
            {
              "handled": True,
              "reason": ...,
              "action_result": <result dict from action engine>
            }
        """
        res = self.resolve(user_text)
        if not res.is_tool_intent or res.action_request is None:
            return {
                "handled": False,
                "reason": res.reason,
            }

        if self._engine is None:
            return {
                "handled": False,
                "reason": "no_action_engine_configured",
            }

        try:
            result = self._engine.handle(res.action_request)
        except Exception as e:
            logger.exception("Tool intent execution failed")
            return {
                "handled": True,
                "reason": res.reason,
                "action_result": {
                    "ok": False,
                    "action": res.action_request.action,
                    "summary": "Tool execution failed",
                    "details": {"error": str(e)},
                },
            }

        return {
            "handled": True,
            "reason": res.reason,
            "action_result": result,
        }

    def get_supported_patterns(self) -> List[Dict[str, str]]:
        """Return a list of supported tool intent patterns for documentation."""
        return [
            # Core patterns
            {"pattern": "create directory <path>", "action": "mkdir"},
            {"pattern": "make directory <path>", "action": "mkdir"},
            {"pattern": "write file <path> with content: <content>", "action": "write_text_file"},
            {"pattern": "append to file <path> with content: <content>", "action": "append_text_file"},
            {"pattern": "read file <path>", "action": "read_text_file"},
            {"pattern": "read file <path> exactly as bytes", "action": "read_bytes_file"},
            {"pattern": "list files", "action": "list_python_files"},
            {"pattern": "list python files", "action": "list_python_files"},
            {"pattern": "git status", "action": "git_status"},
            {"pattern": "git diff [paths...]", "action": "git_diff"},
            {"pattern": "run python: <code>", "action": "run_python_sandbox"},
            {"pattern": "execute command: <cmd>", "action": "run_shell"},
            {"pattern": "attempt to execute shell command: <cmd>", "action": "run_shell"},
            # FULL_AGENCY natural language patterns
            {"pattern": "open <file>", "action": "read_text_file"},
            {"pattern": "show me <file>", "action": "read_text_file"},
            {"pattern": "cat <file>", "action": "read_text_file"},
            {"pattern": "delete <file>", "action": "delete_file"},
            {"pattern": "remove <file>", "action": "delete_file"},
            {"pattern": "run <command>", "action": "run_shell"},
            {"pattern": "execute <command>", "action": "run_shell"},
            {"pattern": "install <package>", "action": "run_shell (pip/npm)"},
            {"pattern": "commit <message>", "action": "git_commit"},
            {"pattern": "push", "action": "git_push"},
            {"pattern": "pull", "action": "git_pull"},
            {"pattern": "clone <url>", "action": "git_clone"},
            {"pattern": "search <query>", "action": "web_search"},
            {"pattern": "fetch <url>", "action": "web_fetch"},
            {"pattern": "browse to <url>", "action": "web_fetch"},
        ]


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Default resolver instance (lazy initialization)
_default_resolver: Optional[ToolIntentResolver] = None


def _get_resolver() -> ToolIntentResolver:
    """Get or create default resolver instance."""
    global _default_resolver
    if _default_resolver is None:
        engine = None
        if _action_engine_available:
            try:
                engine = ActionEngineBrain()
            except Exception as e:
                logger.warning("Could not create ActionEngineBrain: %s", e)
        _default_resolver = ToolIntentResolver(action_engine=engine)
    return _default_resolver


def resolve_tool_intent(user_text: str) -> ToolIntentResult:
    """
    Module-level function to resolve tool intent.

    Args:
        user_text: The user's input text

    Returns:
        ToolIntentResult with is_tool_intent flag and optional action_request
    """
    resolver = _get_resolver()
    return resolver.resolve(user_text)


def maybe_execute_tool(user_text: str) -> Dict[str, Any]:
    """
    Module-level function to resolve and optionally execute tool intent.

    Args:
        user_text: The user's input text

    Returns:
        Dict with 'handled' flag and optional 'action_result'
    """
    resolver = _get_resolver()
    return resolver.maybe_execute(user_text)


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for tool intent resolver.

    Supported operations:
    - RESOLVE: Parse user text and return tool intent (does not execute)
    - EXECUTE: Parse and execute tool intent
    - LIST_PATTERNS: List supported patterns
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    if op == "RESOLVE":
        try:
            user_text = payload.get("text", "")
            if not user_text:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TEXT", "message": "text required"},
                }

            result = resolve_tool_intent(user_text)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "is_tool_intent": result.is_tool_intent,
                    "reason": result.reason,
                    "action": result.action_request.action if result.action_request else None,
                    "params": result.action_request.params if result.action_request else None,
                },
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "RESOLVE_FAILED", "message": str(e)},
            }

    if op == "EXECUTE":
        try:
            user_text = payload.get("text", "")
            if not user_text:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TEXT", "message": "text required"},
                }

            result = maybe_execute_tool(user_text)
            return {
                "ok": result.get("handled", False) and result.get("action_result", {}).get("ok", False),
                "op": op,
                "mid": mid,
                "payload": result,
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "EXECUTE_FAILED", "message": str(e)},
            }

    if op == "LIST_PATTERNS":
        resolver = _get_resolver()
        patterns = resolver.get_supported_patterns()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {"patterns": patterns, "count": len(patterns)},
        }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "intent_resolver_tools",
                "action_engine_available": _action_engine_available,
                "available_operations": ["RESOLVE", "EXECUTE", "LIST_PATTERNS", "HEALTH"],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
