"""
Host Git Client Implementation
==============================

Concrete implementation of git operations using subprocess.
This module executes git commands and should NOT be imported
by core brains.

The host runtime creates instances of this tool and injects them
into the brain context via ToolRegistry.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from brains.tools_api import GitResult, GitStatusResult, GitLogEntry


def _load_github_config() -> Optional[Dict[str, Any]]:
    """Load GitHub configuration from config/github_config.json.

    The token is read from an environment variable specified in the config,
    NOT stored directly in the config file (for security).
    """
    config = None

    try:
        # Try to find config relative to maven root
        from brains.maven_paths import get_maven_root
        config_path = get_maven_root() / "config" / "github_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
    except Exception:
        pass

    # Fallback: try relative to this file
    if not config:
        try:
            config_path = Path(__file__).resolve().parents[2] / "config" / "github_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
        except Exception:
            pass

    if config:
        # Read token from environment variable
        token_env_var = config.get("token_env_var", "MAVEN_GITHUB_TOKEN")
        token = os.environ.get(token_env_var, "")
        if token:
            config["token"] = token

    return config


class HostGitTool:
    """
    Host implementation of git operations.

    Satisfies the GitTool protocol from brains.tools_api.
    """

    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the git tool.

        Args:
            root_dir: Git repository root directory
        """
        if root_dir:
            self.root_dir = Path(root_dir).resolve()
        else:
            try:
                from brains.maven_paths import get_maven_root
                self.root_dir = get_maven_root()
            except Exception:
                self.root_dir = Path(__file__).resolve().parents[2]

    def _run_git(
        self,
        args: List[str],
        *,
        timeout: int = 20,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        cmd_display = "git " + " ".join(args)
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.root_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        print(f"[HOST_GIT] cmd=\"{cmd_display}\" exit={result.returncode}")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())

        if check and result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr.strip()}")

        return result

    def status(self, *, short: bool = True, porcelain: bool = False) -> str:
        """Get git status output."""
        args = ["status"]
        if short:
            args.append("-sb")
        if porcelain:
            args.append("--porcelain")

        result = self._run_git(args, check=False)
        return result.stdout

    def status_detailed(self) -> GitStatusResult:
        """Get detailed git status as structured data."""
        result = self._run_git(["status", "--porcelain", "--branch"], check=False)

        status = GitStatusResult(
            branch=None,
            ahead=0,
            behind=0,
            staged=[],
            modified=[],
            untracked=[],
            deleted=[],
            is_clean=True
        )

        staged = []
        modified = []
        untracked = []
        deleted = []

        for line in result.stdout.splitlines():
            if line.startswith("##"):
                parts = line[3:].split("...")
                status.branch = parts[0]
                if len(parts) > 1:
                    tracking = parts[1]
                    if "[" in tracking:
                        track_info = tracking.split("[")[1].strip("]")
                        if "ahead" in track_info:
                            try:
                                status.ahead = int(
                                    track_info.split("ahead ")[1].split(",")[0].split("]")[0]
                                )
                            except (ValueError, IndexError):
                                pass
                        if "behind" in track_info:
                            try:
                                status.behind = int(
                                    track_info.split("behind ")[1].split("]")[0]
                                )
                            except (ValueError, IndexError):
                                pass
            else:
                code = line[:2]
                filepath = line[3:]
                if code[0] in ("A", "M", "D", "R", "C"):
                    staged.append(filepath)
                if len(code) > 1 and code[1] == "M":
                    modified.append(filepath)
                if len(code) > 1 and code[1] == "D":
                    deleted.append(filepath)
                if code == "??":
                    untracked.append(filepath)

        status.staged = staged
        status.modified = modified
        status.untracked = untracked
        status.deleted = deleted
        status.is_clean = (
            len(staged) == 0 and
            len(modified) == 0 and
            len(untracked) == 0 and
            len(deleted) == 0
        )

        return status

    def diff(
        self,
        *,
        cached: bool = False,
        file_path: Optional[str] = None,
        commit: Optional[str] = None
    ) -> str:
        """Get git diff output."""
        args = ["diff"]
        if cached:
            args.append("--cached")
        if commit:
            args.append(commit)
        if file_path:
            args.append(file_path)

        result = self._run_git(args, check=False)
        return result.stdout

    def log(
        self,
        *,
        max_count: int = 10,
        oneline: bool = False
    ) -> List[GitLogEntry]:
        """Get git log entries."""
        args = ["log", f"-{max_count}"]
        if oneline:
            args.append("--oneline")

        args.append("--format=%H|%an|%ae|%at|%s")

        result = self._run_git(args, check=False)
        commits = []

        for line in result.stdout.splitlines():
            if "|" in line:
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commits.append(GitLogEntry(
                        hash=parts[0].strip().lstrip("* "),
                        author_name=parts[1],
                        author_email=parts[2],
                        timestamp=int(parts[3]) if parts[3].isdigit() else 0,
                        message=parts[4]
                    ))

        return commits

    def add(self, paths: List[str]) -> GitResult:
        """Stage files."""
        if not paths:
            return GitResult(ok=False, output="", error="No paths provided")

        try:
            result = self._run_git(["add", *paths])
            return GitResult(
                ok=True,
                output=result.stdout,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def add_all(self) -> GitResult:
        """Stage all changes."""
        try:
            result = self._run_git(["add", "-A"])
            return GitResult(
                ok=True,
                output=result.stdout,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def commit(self, message: str, *, allow_empty: bool = False) -> GitResult:
        """Create a commit."""
        if not message:
            return GitResult(ok=False, output="", error="Commit message required")

        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")

        try:
            commit_result = self._run_git(args)
            hash_result = self._run_git(["rev-parse", "HEAD"])
            commit_hash = hash_result.stdout.strip()

            print(f"[HOST_GIT] commit_hash={commit_hash} msg=\"{message}\"")
            return GitResult(
                ok=True,
                output=commit_hash,
                returncode=commit_result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def current_branch(self) -> str:
        """Get current branch name."""
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
        return result.stdout.strip()

    def current_commit(self) -> str:
        """Get current commit hash."""
        result = self._run_git(["rev-parse", "HEAD"], check=False)
        return result.stdout.strip()

    def is_clean(self) -> bool:
        """Check if working tree is clean."""
        result = self._run_git(["status", "--porcelain"], check=False)
        return len(result.stdout.strip()) == 0

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        *,
        set_upstream: bool = False,
        force: bool = False
    ) -> GitResult:
        """Push to remote.

        If remote is 'github' or pushing fails on 'origin', will try using
        GitHub credentials from config/github_config.json.
        """
        resolved_branch = branch
        if not resolved_branch:
            resolved_branch = self.current_branch()

        # Check if we should use GitHub credentials
        github_config = _load_github_config()
        use_github = remote == "github" or (github_config and remote == "origin")

        if use_github and github_config:
            # Use GitHub with token authentication
            token = github_config.get("token", "")
            repo_url = github_config.get("repo_url", "")
            target_branch = github_config.get("default_branch", "main")

            if token and repo_url:
                # Build authenticated URL
                if repo_url.startswith("https://"):
                    auth_url = repo_url.replace("https://", f"https://{token}@")
                else:
                    auth_url = f"https://{token}@github.com/{repo_url}"

                print(f"[HOST_GIT] Using GitHub credentials for push")

                # First ensure the github remote exists
                try:
                    self._run_git(["remote", "add", "github", auth_url], check=False)
                except Exception:
                    pass

                # Update the remote URL (in case it changed)
                try:
                    self._run_git(["remote", "set-url", "github", auth_url], check=False)
                except Exception:
                    pass

                args = ["push"]
                if set_upstream:
                    args.extend(["-u", "github", resolved_branch + ":" + target_branch])
                else:
                    args.extend(["github", resolved_branch + ":" + target_branch])

                if force:
                    args.append("--force")

                try:
                    result = self._run_git(args, timeout=120)
                    print(f"[HOST_GIT] Pushed {resolved_branch} to GitHub ({target_branch})")
                    return GitResult(
                        ok=True,
                        output=f"Pushed to GitHub: {target_branch}",
                        returncode=result.returncode
                    )
                except Exception as e:
                    return GitResult(ok=False, output="", error=str(e))

        # Standard push (no GitHub config or different remote)
        args = ["push"]
        if set_upstream:
            args.extend(["-u", remote, resolved_branch])
        else:
            args.extend([remote, resolved_branch])

        if force:
            args.append("--force")

        try:
            result = self._run_git(args)
            print(f"[HOST_GIT] Pushed {resolved_branch} to {remote}")
            return GitResult(
                ok=True,
                output=resolved_branch,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def pull(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        *,
        rebase: bool = False
    ) -> GitResult:
        """Pull from remote."""
        args = ["pull"]
        if rebase:
            args.append("--rebase")

        args.append(remote)
        if branch:
            args.append(branch)

        try:
            result = self._run_git(args)
            return GitResult(
                ok=True,
                output=result.stdout,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def fetch(self, remote: str = "origin", *, all_remotes: bool = False) -> GitResult:
        """Fetch from remote."""
        try:
            if all_remotes:
                result = self._run_git(["fetch", "--all"])
            else:
                result = self._run_git(["fetch", remote])
            return GitResult(
                ok=True,
                output=result.stdout,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def checkout(
        self,
        branch_name: str,
        *,
        create: bool = False
    ) -> GitResult:
        """Checkout a branch."""
        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(branch_name)

        try:
            result = self._run_git(args)
            print(f"[HOST_GIT] Checked out: {branch_name}")
            return GitResult(
                ok=True,
                output=branch_name,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def list_branches(
        self,
        *,
        remote: bool = False,
        all: bool = False
    ) -> List[str]:
        """List branches."""
        args = ["branch"]
        if all:
            args.append("-a")
        elif remote:
            args.append("-r")

        result = self._run_git(args, check=False)
        branches = []
        for line in result.stdout.splitlines():
            branch = line.strip().lstrip("* ")
            if branch:
                branches.append(branch)

        return branches

    def stash(self, message: Optional[str] = None) -> GitResult:
        """Stash current changes."""
        args = ["stash"]
        if message:
            args.extend(["push", "-m", message])

        try:
            result = self._run_git(args)
            return GitResult(
                ok=True,
                output=result.stdout,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))

    def stash_pop(self) -> GitResult:
        """Pop the most recent stash."""
        try:
            result = self._run_git(["stash", "pop"])
            return GitResult(
                ok=True,
                output=result.stdout,
                returncode=result.returncode
            )
        except Exception as e:
            return GitResult(ok=False, output="", error=str(e))
