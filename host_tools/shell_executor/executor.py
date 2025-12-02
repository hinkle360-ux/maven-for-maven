"""
Host Shell Executor Implementation
==================================

Concrete implementation of shell command execution and Python sandbox.
This module uses subprocess to execute external commands and should
NOT be imported by core brains.

The host runtime creates instances of these tools and injects them
into the brain context via ToolRegistry.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from brains.tools_api import ShellResult, SandboxResult

# Windows/Unix detection
IS_WINDOWS = sys.platform == "win32"

# Unix to Windows command translation
UNIX_TO_WINDOWS = {
    "ls": "dir",
    "pwd": "cd",
    "cat": "type",
    "clear": "cls",
    "cp": "copy",
    "mv": "move",
    "rm": "del",
    "mkdir": "mkdir",  # Same on both
    "rmdir": "rmdir",  # Same on both
    "touch": "type nul >",  # Creates empty file
    "grep": "findstr",
    "which": "where",
    "whoami": "whoami",  # Same on both (Windows has it)
}


def _translate_unix_to_windows(cmd: str) -> str:
    """
    Translate common Unix commands to Windows equivalents.

    Args:
        cmd: Command string to translate

    Returns:
        Translated command for Windows, or original if no translation needed
    """
    if not IS_WINDOWS:
        return cmd

    cmd_stripped = cmd.strip()

    # Handle simple single commands
    for unix_cmd, win_cmd in UNIX_TO_WINDOWS.items():
        # Exact match (command alone)
        if cmd_stripped == unix_cmd:
            return win_cmd
        # Command with arguments (e.g., "ls -la")
        if cmd_stripped.startswith(unix_cmd + " "):
            args = cmd_stripped[len(unix_cmd):].strip()
            # Special handling for ls flags
            if unix_cmd == "ls":
                # -la, -l, -a are common but don't map directly
                # Just return dir with the path if there's a path arg
                path_args = [a for a in args.split() if not a.startswith("-")]
                if path_args:
                    return f"dir {' '.join(path_args)}"
                return "dir"
            return f"{win_cmd} {args}"

    return cmd


def _is_full_agency_mode() -> bool:
    """Check if FULL_AGENCY mode is active."""
    try:
        from brains.tools.execution_guard import get_execution_status, ExecMode
        status = get_execution_status()
        return status.mode == ExecMode.FULL_AGENCY and status.effective
    except Exception:
        return False


def _load_deny_patterns(root_dir: Optional[str] = None) -> List[str]:
    """Load deny patterns from tool policy configuration.

    In FULL_AGENCY mode, uses minimal deny patterns (only truly destructive).
    Otherwise uses the configured patterns from tool_policy.json.
    """
    # FULL_AGENCY mode - minimal deny list (only catastrophic commands)
    if _is_full_agency_mode():
        return [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero of=/dev/",
            "dd if=/dev/random of=/dev/",
            "> /dev/sda",
            "chmod -R 777 /",
            ":(){:|:&};:",
            "mv / /dev/null",
        ]

    deny_patterns: List[str] = []

    # Try to load from config
    if root_dir:
        policy_path = Path(root_dir) / "config" / "tool_policy.json"
    else:
        try:
            from brains.maven_paths import get_maven_root
            policy_path = get_maven_root() / "config" / "tool_policy.json"
        except Exception:
            policy_path = None

    if policy_path and policy_path.exists():
        try:
            with open(policy_path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
                shell_policy = data.get("shell_policy", {}) or {}

                # Check if mode is full_agency in config
                if shell_policy.get("mode") == "full_agency":
                    # Use minimal deny list from config
                    deny_patterns = shell_policy.get("deny_patterns") or []
                else:
                    deny_patterns = shell_policy.get("deny_patterns") or []
        except Exception:
            pass

    if not deny_patterns:
        # Default deny patterns for safety (non-FULL_AGENCY mode)
        deny_patterns = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
            "chmod -R 777 /", ":(){:|:&};:", "shutdown", "reboot", "init 0"
        ]

    return deny_patterns


class HostShellTool:
    """
    Host implementation of shell command execution.

    Satisfies the ShellTool protocol from brains.tools_api.
    """

    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the shell tool.

        Args:
            root_dir: Root directory for command execution (default: Maven root)
        """
        if root_dir:
            self.root_dir = Path(root_dir).resolve()
        else:
            try:
                from brains.maven_paths import get_maven_root
                self.root_dir = get_maven_root()
            except Exception:
                self.root_dir = Path(__file__).resolve().parents[2]

        # Initial deny patterns (will be checked dynamically)
        self._root_dir_str = str(self.root_dir)

    @property
    def deny_patterns(self) -> List[str]:
        """Get deny patterns dynamically (checks current execution mode)."""
        return _load_deny_patterns(self._root_dir_str)

    def run(
        self,
        cmd: str,
        *,
        cwd: Optional[str] = None,
        timeout: int = 120,
        check_policy: bool = True
    ) -> ShellResult:
        """
        Execute a shell command.

        Args:
            cmd: Command to execute
            cwd: Working directory (default: root_dir)
            timeout: Timeout in seconds
            check_policy: Whether to check against deny patterns

        Returns:
            ShellResult with output and status
        """
        if cwd is None:
            cwd = str(self.root_dir)

        # Check against deny patterns if requested
        if check_policy:
            cmd_lc = str(cmd or "").lower()
            for pattern in self.deny_patterns:
                try:
                    if pattern.strip() and pattern.strip().lower() in cmd_lc:
                        return ShellResult(
                            status="denied",
                            error=f"Command contains banned pattern '{pattern.strip()}'."
                        )
                except Exception:
                    continue

        try:
            # Translate Unix commands to Windows if needed
            translated_cmd = _translate_unix_to_windows(cmd)

            if IS_WINDOWS:
                # On Windows, use shell=True for better compatibility
                # This handles paths with spaces, built-in commands, etc.
                proc = subprocess.run(
                    translated_cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=True
                )
            else:
                # On Unix, use shlex.split for security
                proc = subprocess.run(
                    shlex.split(translated_cmd),
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

            # Check for known Git for Windows issues
            output = proc.stdout + proc.stderr
            if "BUG (fork bomb)" in output or "bug (fork bomb)" in output.lower():
                # This is a known Git for Windows issue, not a Maven bug
                return ShellResult(
                    status="error",
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    error=(
                        "Git for Windows error detected: 'BUG (fork bomb)'\n"
                        "This is a known Git for Windows issue, typically caused by:\n"
                        "1. Corrupted Git installation - try reinstalling Git\n"
                        "2. Anaconda/Conda conflict - try 'conda deactivate'\n"
                        "3. Multiple Git installations on PATH - check your PATH variable"
                    )
                )

            return ShellResult(
                status="completed",
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr
            )
        except subprocess.TimeoutExpired:
            return ShellResult(
                status="timeout",
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ShellResult(
                status="error",
                error=str(e)
            )


class HostPythonSandboxTool:
    """
    Host implementation of sandboxed Python execution.

    Satisfies the PythonSandboxTool protocol from brains.tools_api.
    """

    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the Python sandbox.

        Args:
            root_dir: Root directory for code execution (default: Maven root)
        """
        if root_dir:
            self.root_dir = Path(root_dir).resolve()
        else:
            try:
                from brains.maven_paths import get_maven_root
                self.root_dir = get_maven_root()
            except Exception:
                self.root_dir = Path(__file__).resolve().parents[2]

    def execute(
        self,
        code: str,
        *,
        timeout_ms: int = 3000,
        cwd: Optional[str] = None
    ) -> SandboxResult:
        """
        Execute Python code in an isolated subprocess.

        Args:
            code: Python code to execute
            timeout_ms: Timeout in milliseconds
            cwd: Working directory (default: root_dir)

        Returns:
            SandboxResult with output and status
        """
        if cwd is None:
            cwd = str(self.root_dir)

        cmd = [sys.executable, "-c", code]
        timeout_sec = timeout_ms / 1000.0

        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            return SandboxResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                stdout="",
                stderr="",
                returncode=-1,
                error="Execution timed out",
                timed_out=True
            )
        except Exception as e:
            return SandboxResult(
                ok=False,
                stdout="",
                stderr="",
                returncode=-1,
                error=str(e)
            )
