"""
Git Operations Facade
=====================

This module provides a facade for git operations that delegates to the
host-provided git tool. It maintains backward compatibility with existing
code while ensuring no direct subprocess operations occur in brains.

IMPORTANT: This module should not use subprocess directly.
All git operations are delegated to the tool registry.

For direct git access, use host_tools.git_client.client directly
from the host runtime (not from brains).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime

from brains.maven_paths import get_maven_root
from brains.tools.execution_guard import require_execution_enabled
from brains.tools_api import (
    GitTool,
    GitResult,
    GitStatusResult,
    GitLogEntry,
    NullGitTool,
    ToolRegistry,
)


# Global tool registry - set by host runtime
_tool_registry: Optional[ToolRegistry] = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry (called by host runtime)."""
    global _tool_registry
    _tool_registry = registry


def get_git_tool() -> GitTool:
    """Get the git tool from the registry."""
    if _tool_registry and _tool_registry.git:
        return _tool_registry.git
    return NullGitTool()


class _GitResultWrapper:
    """Wrapper to provide subprocess.CompletedProcess-like interface."""
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _run_git(args: List[str], *, timeout: int = 20, check: bool = True) -> _GitResultWrapper:
    """Run a git command via the host-provided tool.

    This function delegates to the tool registry instead of using subprocess.
    """
    cmd_display = "git " + " ".join(args)
    tool = get_git_tool()

    # Handle common git commands
    if not args:
        return _GitResultWrapper(returncode=1, stderr="No git command specified")

    cmd = args[0]
    sub_args = args[1:]

    try:
        if cmd == "status":
            porcelain = "--porcelain" in sub_args
            short = "-sb" in sub_args or "-s" in sub_args
            result = tool.status(short=short, porcelain=porcelain)
            return _GitResultWrapper(stdout=result, returncode=0)

        elif cmd == "diff":
            cached = "--cached" in sub_args
            commit = None
            file_path = None
            for arg in sub_args:
                if not arg.startswith("-"):
                    if commit is None:
                        commit = arg
                    else:
                        file_path = arg
            result = tool.diff(cached=cached, commit=commit, file_path=file_path)
            return _GitResultWrapper(stdout=result, returncode=0)

        elif cmd == "log":
            max_count = 10
            oneline = "--oneline" in sub_args
            for arg in sub_args:
                if arg.startswith("-") and arg[1:].isdigit():
                    max_count = int(arg[1:])
            entries = tool.log(max_count=max_count, oneline=oneline)
            # Format as original git log format
            lines = []
            for e in entries:
                lines.append(f"{e.hash}|{e.author_name}|{e.author_email}|{e.timestamp}|{e.message}")
            return _GitResultWrapper(stdout="\n".join(lines), returncode=0)

        elif cmd == "add":
            paths = [a for a in sub_args if not a.startswith("-")]
            if "-A" in sub_args or "--all" in sub_args:
                if hasattr(tool, "add_all"):
                    result = tool.add_all()
                else:
                    result = tool.add(["."])
            else:
                result = tool.add(paths)
            return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        elif cmd == "commit":
            message = ""
            for i, arg in enumerate(sub_args):
                if arg == "-m" and i + 1 < len(sub_args):
                    message = sub_args[i + 1]
                    break
            allow_empty = "--allow-empty" in sub_args
            result = tool.commit(message, allow_empty=allow_empty)
            return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        elif cmd == "rev-parse":
            if "--abbrev-ref" in sub_args and "HEAD" in sub_args:
                result = tool.current_branch()
                return _GitResultWrapper(stdout=result, returncode=0)
            elif "HEAD" in sub_args:
                if hasattr(tool, "current_commit"):
                    result = tool.current_commit()
                    return _GitResultWrapper(stdout=result, returncode=0)

        elif cmd == "branch":
            if hasattr(tool, "list_branches"):
                all_branches = "-a" in sub_args or "--all" in sub_args
                remote = "-r" in sub_args
                branches = tool.list_branches(all=all_branches, remote=remote)
                return _GitResultWrapper(stdout="\n".join(branches), returncode=0)

        elif cmd == "checkout":
            create = "-b" in sub_args
            branch = None
            for arg in sub_args:
                if not arg.startswith("-"):
                    branch = arg
                    break
            if branch and hasattr(tool, "checkout"):
                result = tool.checkout(branch, create=create)
                return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        elif cmd == "push":
            remote = "origin"
            branch = None
            set_upstream = "-u" in sub_args or "--set-upstream" in sub_args
            force = "--force" in sub_args or "-f" in sub_args
            for arg in sub_args:
                if not arg.startswith("-"):
                    if remote == "origin":
                        remote = arg
                    else:
                        branch = arg
            if hasattr(tool, "push"):
                result = tool.push(remote=remote, branch=branch, set_upstream=set_upstream, force=force)
                return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        elif cmd == "pull":
            remote = "origin"
            branch = None
            rebase = "--rebase" in sub_args
            for arg in sub_args:
                if not arg.startswith("-"):
                    if remote == "origin":
                        remote = arg
                    else:
                        branch = arg
            if hasattr(tool, "pull"):
                result = tool.pull(remote=remote, branch=branch, rebase=rebase)
                return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        elif cmd == "fetch":
            if hasattr(tool, "fetch"):
                all_remotes = "--all" in sub_args
                remote = "origin"
                for arg in sub_args:
                    if not arg.startswith("-"):
                        remote = arg
                        break
                result = tool.fetch(remote=remote, all_remotes=all_remotes)
                return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        elif cmd == "stash":
            if hasattr(tool, "stash"):
                if "pop" in sub_args:
                    result = tool.stash_pop()
                else:
                    message = None
                    for i, arg in enumerate(sub_args):
                        if arg == "-m" and i + 1 < len(sub_args):
                            message = sub_args[i + 1]
                            break
                    result = tool.stash(message=message)
                return _GitResultWrapper(stdout=result.output, returncode=0 if result.ok else 1, stderr=result.error or "")

        # Fallback for unsupported commands
        print(f"[GIT] Warning: command '{cmd}' not fully supported via tool interface")
        return _GitResultWrapper(stderr=f"Git command '{cmd}' not supported in facade mode", returncode=1)

    except Exception as e:
        print(f"[GIT] Error executing '{cmd_display}': {e}")
        if check:
            raise RuntimeError(f"Git command failed: {e}")
        return _GitResultWrapper(stderr=str(e), returncode=1)


# ==================== STATUS AND INFO OPERATIONS ====================

def git_status(short: bool = True, porcelain: bool = False) -> str:
    """Return git status output."""
    require_execution_enabled("git_status")

    args = ["status"]
    if short:
        args.append("-sb")
    if porcelain:
        args.append("--porcelain")

    result = _run_git(args)
    return result.stdout


def git_status_detailed() -> Dict[str, Any]:
    """Return detailed git status information as a structured dict."""
    require_execution_enabled("git_status_detailed")

    result = _run_git(["status", "--porcelain", "--branch"], check=False)

    status = {
        "branch": None,
        "ahead": 0,
        "behind": 0,
        "staged": [],
        "modified": [],
        "untracked": [],
        "deleted": []
    }

    for line in result.stdout.splitlines():
        if line.startswith("##"):
            # Branch info
            parts = line[3:].split("...")
            status["branch"] = parts[0]
            if len(parts) > 1:
                tracking = parts[1]
                if "[" in tracking:
                    track_info = tracking.split("[")[1].strip("]")
                    if "ahead" in track_info:
                        status["ahead"] = int(track_info.split("ahead ")[1].split(",")[0].split("]")[0])
                    if "behind" in track_info:
                        status["behind"] = int(track_info.split("behind ")[1].split("]")[0])
        else:
            code = line[:2]
            filepath = line[3:]
            if code[0] in ("A", "M", "D", "R", "C"):
                status["staged"].append(filepath)
            if code[1] == "M":
                status["modified"].append(filepath)
            if code[1] == "D":
                status["deleted"].append(filepath)
            if code == "??":
                status["untracked"].append(filepath)

    return status


def git_current_branch() -> str:
    """Get the current branch name."""
    require_execution_enabled("git_current_branch")
    result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def git_current_commit() -> str:
    """Get the current commit hash."""
    require_execution_enabled("git_current_commit")
    result = _run_git(["rev-parse", "HEAD"])
    return result.stdout.strip()


def git_list_branches(remote: bool = False, all: bool = False) -> List[str]:
    """List all branches."""
    require_execution_enabled("git_list_branches")

    args = ["branch"]
    if all:
        args.append("-a")
    elif remote:
        args.append("-r")

    result = _run_git(args)
    branches = []
    for line in result.stdout.splitlines():
        branch = line.strip().lstrip("* ")
        if branch:
            branches.append(branch)

    return branches


# ==================== STAGING AND COMMITTING ====================

def git_add(paths: List[str]) -> None:
    """Stage the provided paths."""
    require_execution_enabled("git_add")
    if not paths:
        raise ValueError("paths required for git add")
    result = _run_git(["add", *paths])


def git_add_all() -> None:
    """Stage all changes."""
    require_execution_enabled("git_add_all")
    _run_git(["add", "-A"])


def git_reset(paths: Optional[List[str]] = None) -> None:
    """Unstage changes."""
    require_execution_enabled("git_reset")

    if paths:
        _run_git(["reset", "HEAD", *paths])
    else:
        _run_git(["reset", "HEAD"])


def git_commit(message: str, allow_empty: bool = False) -> str:
    """Commit staged changes and return commit hash."""
    require_execution_enabled("git_commit")

    if not message:
        raise ValueError("commit message is required")

    args = ["commit", "-m", message]
    if allow_empty:
        args.append("--allow-empty")

    commit_result = _run_git(args)
    hash_result = _run_git(["rev-parse", "HEAD"])
    commit_hash = hash_result.stdout.strip()

    print(f"[GIT] commit_hash={commit_hash} msg=\"{message}\"")
    return commit_hash


def git_commit_amend(message: Optional[str] = None) -> str:
    """Amend the last commit."""
    require_execution_enabled("git_commit_amend")

    args = ["commit", "--amend"]
    if message:
        args.extend(["-m", message])
    else:
        args.append("--no-edit")

    _run_git(args)
    hash_result = _run_git(["rev-parse", "HEAD"])
    return hash_result.stdout.strip()


# ==================== BRANCH OPERATIONS ====================

def git_create_branch(branch_name: str, checkout: bool = True) -> str:
    """Create a new branch."""
    require_execution_enabled("git_create_branch")

    if checkout:
        _run_git(["checkout", "-b", branch_name])
    else:
        _run_git(["branch", branch_name])

    print(f"[GIT] Created branch: {branch_name}")
    return branch_name


def git_checkout(branch_name: str, create: bool = False) -> str:
    """Checkout a branch."""
    require_execution_enabled("git_checkout")

    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch_name)

    _run_git(args)
    print(f"[GIT] Checked out: {branch_name}")
    return branch_name


def git_delete_branch(branch_name: str, force: bool = False) -> None:
    """Delete a branch."""
    require_execution_enabled("git_delete_branch")

    flag = "-D" if force else "-d"
    _run_git(["branch", flag, branch_name])
    print(f"[GIT] Deleted branch: {branch_name}")


def git_merge(branch_name: str, no_ff: bool = False, squash: bool = False) -> str:
    """Merge a branch into the current branch."""
    require_execution_enabled("git_merge")

    args = ["merge", branch_name]
    if no_ff:
        args.append("--no-ff")
    if squash:
        args.append("--squash")

    result = _run_git(args, check=False)

    if result.returncode != 0:
        print(f"[GIT] Merge conflict detected")
        return "conflict"

    return "success"


# ==================== REMOTE OPERATIONS ====================

def git_push(remote: str = "origin", branch: Optional[str] = None, set_upstream: bool = False, force: bool = False) -> str:
    """Push the current branch (or specified branch) to the remote."""
    require_execution_enabled("git_push")

    resolved_branch = branch
    if not resolved_branch:
        branch_result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        resolved_branch = branch_result.stdout.strip()

    args = ["push"]
    if set_upstream:
        args.extend(["-u", remote, resolved_branch])
    else:
        args.extend([remote, resolved_branch])

    if force:
        args.append("--force")

    push_result = _run_git(args)
    print(f"[GIT] Pushed {resolved_branch} to {remote}")
    return resolved_branch


def git_pull(remote: str = "origin", branch: Optional[str] = None, rebase: bool = False) -> str:
    """Pull changes from remote."""
    require_execution_enabled("git_pull")

    args = ["pull"]
    if rebase:
        args.append("--rebase")

    args.append(remote)
    if branch:
        args.append(branch)

    _run_git(args)
    return "success"


def git_fetch(remote: str = "origin", all_remotes: bool = False) -> None:
    """Fetch from remote."""
    require_execution_enabled("git_fetch")

    if all_remotes:
        _run_git(["fetch", "--all"])
    else:
        _run_git(["fetch", remote])


def git_list_remotes() -> List[Dict[str, str]]:
    """List all remotes."""
    require_execution_enabled("git_list_remotes")

    result = _run_git(["remote", "-v"])
    remotes = []
    seen = set()

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            url = parts[1]
            if name not in seen:
                remotes.append({"name": name, "url": url})
                seen.add(name)

    return remotes


# ==================== DIFF AND LOG ====================

def git_diff(cached: bool = False, file_path: Optional[str] = None, commit: Optional[str] = None) -> str:
    """Show diff of changes."""
    require_execution_enabled("git_diff")

    args = ["diff"]
    if cached:
        args.append("--cached")
    if commit:
        args.append(commit)
    if file_path:
        args.append(file_path)

    result = _run_git(args, check=False)
    return result.stdout


def git_log(max_count: int = 10, oneline: bool = False, graph: bool = False) -> List[Dict[str, Any]]:
    """Get commit log."""
    require_execution_enabled("git_log")

    args = ["log", f"-{max_count}"]
    if oneline:
        args.append("--oneline")
    if graph:
        args.append("--graph")

    args.append("--format=%H|%an|%ae|%at|%s")

    result = _run_git(args, check=False)
    commits = []

    for line in result.stdout.splitlines():
        if "|" in line:
            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append({
                    "hash": parts[0].strip().lstrip("* "),
                    "author_name": parts[1],
                    "author_email": parts[2],
                    "timestamp": int(parts[3]),
                    "message": parts[4]
                })

    return commits


def git_show(commit: str = "HEAD") -> str:
    """Show commit details."""
    require_execution_enabled("git_show")
    result = _run_git(["show", commit], check=False)
    return result.stdout


# ==================== STASH OPERATIONS ====================

def git_stash(message: Optional[str] = None) -> str:
    """Stash current changes."""
    require_execution_enabled("git_stash")

    args = ["stash"]
    if message:
        args.extend(["push", "-m", message])

    result = _run_git(args)
    return result.stdout


def git_stash_pop() -> str:
    """Pop the most recent stash."""
    require_execution_enabled("git_stash_pop")
    result = _run_git(["stash", "pop"])
    return result.stdout


def git_stash_list() -> List[str]:
    """List all stashes."""
    require_execution_enabled("git_stash_list")
    result = _run_git(["stash", "list"], check=False)
    return result.stdout.splitlines()


# ==================== TAG OPERATIONS ====================

def git_create_tag(tag_name: str, message: Optional[str] = None, commit: str = "HEAD") -> str:
    """Create a tag."""
    require_execution_enabled("git_create_tag")

    args = ["tag"]
    if message:
        args.extend(["-a", tag_name, "-m", message, commit])
    else:
        args.extend([tag_name, commit])

    _run_git(args)
    print(f"[GIT] Created tag: {tag_name}")
    return tag_name


def git_list_tags() -> List[str]:
    """List all tags."""
    require_execution_enabled("git_list_tags")
    result = _run_git(["tag"], check=False)
    return result.stdout.splitlines()


def git_delete_tag(tag_name: str) -> None:
    """Delete a tag."""
    require_execution_enabled("git_delete_tag")
    _run_git(["tag", "-d", tag_name])


# ==================== ADVANCED OPERATIONS ====================

def git_rebase(branch: str, interactive: bool = False) -> str:
    """Rebase current branch onto another branch."""
    require_execution_enabled("git_rebase")

    args = ["rebase"]
    if interactive:
        args.append("-i")
    args.append(branch)

    result = _run_git(args, check=False)

    if result.returncode != 0:
        print(f"[GIT] Rebase conflict detected")
        return "conflict"

    return "success"


def git_cherry_pick(commit: str) -> str:
    """Cherry-pick a commit."""
    require_execution_enabled("git_cherry_pick")

    result = _run_git(["cherry-pick", commit], check=False)

    if result.returncode != 0:
        print(f"[GIT] Cherry-pick conflict detected")
        return "conflict"

    return "success"


def git_blame(file_path: str, line_start: Optional[int] = None, line_end: Optional[int] = None) -> str:
    """Show blame information for a file."""
    require_execution_enabled("git_blame")

    args = ["blame"]
    if line_start and line_end:
        args.append(f"-L{line_start},{line_end}")
    args.append(file_path)

    result = _run_git(args, check=False)
    return result.stdout


def git_clean(dry_run: bool = True, force: bool = False, directories: bool = False) -> str:
    """Remove untracked files."""
    require_execution_enabled("git_clean")

    args = ["clean"]
    if dry_run:
        args.append("-n")
    if force and not dry_run:
        args.append("-f")
    if directories:
        args.append("-d")

    result = _run_git(args, check=False)
    return result.stdout


# ==================== UTILITY FUNCTIONS ====================

def git_config_get(key: str, global_scope: bool = False) -> Optional[str]:
    """Get a git config value."""
    require_execution_enabled("git_config_get")

    args = ["config"]
    if global_scope:
        args.append("--global")
    args.extend(["--get", key])

    result = _run_git(args, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def git_is_clean_working_tree() -> bool:
    """Check if the working tree is clean."""
    require_execution_enabled("git_is_clean")
    result = _run_git(["status", "--porcelain"], check=False)
    return len(result.stdout.strip()) == 0


def git_get_repo_info() -> Dict[str, Any]:
    """Get comprehensive repository information."""
    require_execution_enabled("git_repo_info")

    info = {
        "root": str(get_maven_root()),
        "branch": git_current_branch(),
        "commit": git_current_commit(),
        "is_clean": git_is_clean_working_tree(),
        "remotes": git_list_remotes(),
        "branches": git_list_branches(),
        "tags": git_list_tags()
    }

    status = git_status_detailed()
    info.update({
        "ahead": status["ahead"],
        "behind": status["behind"],
        "staged_files": len(status["staged"]),
        "modified_files": len(status["modified"]),
        "untracked_files": len(status["untracked"])
    })

    return info


# ============================================================================
# Service API (Standard Tool Interface)
# ============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for the git tool.

    Operations:
    - STATUS: Get git status
    - DIFF: Show diff of changes
    - LOG: Get commit log
    - COMMIT: Commit staged changes
    - ADD: Stage files
    - PUSH: Push to remote
    - PULL: Pull from remote
    - BRANCH: List or create branches
    - CHECKOUT: Checkout a branch
    - REPO_INFO: Get comprehensive repo info
    - HEALTH: Health check

    Args:
        msg: Dict with "op" and optional "payload"

    Returns:
        Dict with "ok", "payload" or "error"
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "GIT"

    try:
        if op == "STATUS":
            short = payload.get("short", True)
            porcelain = payload.get("porcelain", False)
            detailed = payload.get("detailed", False)

            if detailed:
                result = git_status_detailed()
            else:
                result = git_status(short=short, porcelain=porcelain)

            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"status": result} if isinstance(result, str) else result
            }

        if op == "DIFF":
            cached = payload.get("cached", False)
            file_path = payload.get("file_path")
            commit = payload.get("commit")

            result = git_diff(cached=cached, file_path=file_path, commit=commit)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"diff": result}
            }

        if op == "LOG":
            max_count = payload.get("max_count", 10)
            oneline = payload.get("oneline", False)

            result = git_log(max_count=max_count, oneline=oneline)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"commits": result}
            }

        if op == "COMMIT":
            message = payload.get("message", "")
            allow_empty = payload.get("allow_empty", False)

            if not message:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_MESSAGE", "message": "Commit message is required"}
                }

            commit_hash = git_commit(message, allow_empty=allow_empty)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"commit_hash": commit_hash, "message": message}
            }

        if op == "ADD":
            paths = payload.get("paths", [])
            all_files = payload.get("all", False)

            if all_files:
                git_add_all()
            elif paths:
                git_add(paths)
            else:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATHS", "message": "Paths or 'all' flag required"}
                }

            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"staged": paths if paths else "all"}
            }

        if op == "PUSH":
            remote = payload.get("remote", "origin")
            branch = payload.get("branch")
            set_upstream = payload.get("set_upstream", False)
            force = payload.get("force", False)

            result = git_push(remote=remote, branch=branch, set_upstream=set_upstream, force=force)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"pushed": result, "remote": remote}
            }

        if op == "PULL":
            remote = payload.get("remote", "origin")
            branch = payload.get("branch")
            rebase = payload.get("rebase", False)

            result = git_pull(remote=remote, branch=branch, rebase=rebase)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"result": result}
            }

        if op == "BRANCH":
            action = payload.get("action", "list")
            name = payload.get("name")
            remote = payload.get("remote", False)
            all_branches = payload.get("all", False)

            if action == "list":
                branches = git_list_branches(remote=remote, all=all_branches)
                current = git_current_branch()
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {"branches": branches, "current": current}
                }
            elif action == "create" and name:
                result = git_create_branch(name)
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {"created": result}
                }
            elif action == "delete" and name:
                git_delete_branch(name)
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {"deleted": name}
                }

        if op == "CHECKOUT":
            branch = payload.get("branch", "")
            create = payload.get("create", False)

            if not branch:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_BRANCH", "message": "Branch name is required"}
                }

            result = git_checkout(branch, create=create)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"checked_out": result}
            }

        if op == "REPO_INFO":
            info = git_get_repo_info()
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": info
            }

        if op == "HEALTH":
            tool = get_git_tool()
            available = not isinstance(tool, NullGitTool)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "status": "operational" if available else "unavailable",
                    "service": "git",
                    "capability": "git",
                    "description": "Git operations via host runtime",
                    "host_provided": available,
                }
            }

        return {
            "ok": False,
            "op": op,
            "mid": mid,
            "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"}
        }

    except Exception as e:
        return {
            "ok": False,
            "op": op,
            "mid": mid,
            "error": {"code": "GIT_ERROR", "message": str(e)}
        }


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "git"
TOOL_CAPABILITY = "git"
TOOL_DESCRIPTION = "Git operations via host runtime"
TOOL_OPERATIONS = ["STATUS", "DIFF", "LOG", "COMMIT", "ADD", "PUSH", "PULL", "BRANCH", "CHECKOUT", "REPO_INFO", "HEALTH"]


def is_available() -> bool:
    """Check if the git tool is available (requires host injection)."""
    tool = get_git_tool()
    return not isinstance(tool, NullGitTool)


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": is_available(),
        "requires_execution": True,
        "module": "brains.tools.git_tool",
    }
