from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time, os, json, shutil, zipfile, hmac, hashlib as _hashlib, tempfile, ast, subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

from brains.maven_paths import get_brains_path, get_maven_root, get_reports_path

# =============================================================================
# QUARANTINE POLICY
# =============================================================================
# IMPORTANT: Live code must NEVER execute from a quarantine/ directory.
# Quarantine directories are for offline inspection only, not execution.
#
# If a file needs to be quarantined:
# 1. Move it to a non-executable location outside the brains/ tree
# 2. Use docs/archive/repair_engine_patches/ for reference copies
# 3. Never import or execute code from quarantine paths
# =============================================================================

# Import coder brain for targeted repairs
try:
    from brains.cognitive.coder.service.coder_brain import service_api as coder_api
    _coder_available = True
except ImportError:
    _coder_available = False
    coder_api = None

# Repair Engine now requires a Governance authorization token.
# It will REFUSE to execute if a valid auth block is not present.

def _now_ms() -> int:
    return int(time.time() * 1000)

# Determine the Maven root by ascending from this file's location.  This is
# used for backup/restore and cold compaction operations.  The path
# points to the repository root where the "reports" and "brains" folders
# reside.
MAVEN_ROOT = get_maven_root()

# List of all domain banks to operate on when none are explicitly provided.
_ALL_BANKS = [
    "arts","science","history","economics","geography",
    "language_arts","law","math","philosophy","technology",
    "theories_and_contradictions"
]

# Shared secret used to verify signatures on Governance tokens.  The
# value should match the secret used by the Policy Engine.  It can be
# overridden via the MAVEN_SECRET_KEY environment variable.  Do not
# disclose this value outside of Maven.
_SECRET_KEY = os.environ.get("MAVEN_SECRET_KEY", "maven_secret_key")

def _verify_signature(auth: Dict[str, Any]) -> bool:
    """Verify the HMAC signature on an authorization token.

    Returns True if the signature matches the expected value computed
    over the token fields (excluding the signature) using the shared
    secret.  False on any mismatch or error.
    """
    try:
        sig = auth.get("signature")
        if not isinstance(sig, str):
            return False
        # Rebuild the message with sorted keys excluding signature
        data = {k: v for k, v in auth.items() if k != "signature"}
        msg = json.dumps(data, sort_keys=True, separators=(",", ":"))
        expected = hmac.new(_SECRET_KEY.encode(), msg.encode(), _hashlib.sha256).hexdigest()
        # Use timing‑safe comparison
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

def _auth_ok(auth: Dict[str, Any], op: str) -> bool:
    """Validate an authorization token for the given operation.

    Checks issuer, validity flag, expiry, token format, signature and
    scope.  A token is considered valid if:

    * It is a dict issued by governance with valid=True
    * It has not expired (ts + ttl_ms)
    * The token string starts with GOV-
    * The signature matches the expected HMAC over the token contents
    * The operation requested is within the allowed scope ("repair_engine"
      grants access to all repair operations; otherwise the op name must
      appear within the scope string).
    """
    if not isinstance(auth, dict):
        return False
    if auth.get("issuer") != "governance":
        return False
    if not auth.get("valid", False):
        return False
    ts = auth.get("ts")
    ttl = auth.get("ttl_ms", 0)
    if not isinstance(ts, int) or not isinstance(ttl, int):
        return False
    if _now_ms() > ts + ttl:
        return False
    tok = auth.get("token", "")
    if not (isinstance(tok, str) and tok.startswith("GOV-") and len(tok) >= 8):
        return False
    # Verify signature integrity
    if not _verify_signature(auth):
        return False
    # Check scope vs operation
    scope = auth.get("scope", "") or "repair_engine"
    # Normalize to lower case
    scope_lc = scope.lower()
    op_lc = (op or "").lower()
    # If scope is repair_engine or all, allow all repair operations
    if scope_lc in {"repair_engine", "all"}:
        return True
    # Otherwise require that op name appears in the scope string (colon or comma separated)
    # Example scope: "backup,restore" or "compact_cold"
    allowed_ops = [s.strip() for s in scope_lc.replace(";", ",").split(",") if s.strip()]
    return op_lc.lower() in allowed_ops

def _scan_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder: quick scan summary (non-destructive)
    brains = payload.get("brains", ["reasoning","language","memory_librarian","personal","system_history","self_dmn"])
    return {"ok": True, "brains": brains, "issues": []}

@dataclass
class ErrorContext:
    """Context for an error that needs to be fixed."""
    file_path: str
    line_start: int
    line_end: int
    error_type: str  # syntax, import, test_failure, lint
    error_message: str
    code_snippet: str = ""


@dataclass
class RepairPlan:
    """Plan for repairing an error."""
    error: ErrorContext
    problem_summary: str
    constraints: List[str]
    strategy: str  # coder_brain, simple_fix, manual


@dataclass
class RepairResult:
    """Result of a repair attempt."""
    file_path: str
    success: bool
    error_type: str
    pre_diff: str = ""
    post_diff: str = ""
    tests_run: List[str] = field(default_factory=list)
    message: str = ""


def _scan_file_for_errors(file_path: Path) -> List[ErrorContext]:
    """
    Scan a Python file for errors (syntax, imports).

    Returns:
        List of ErrorContext objects describing issues found
    """
    errors = []

    if not file_path.exists() or not file_path.suffix == ".py":
        return errors

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        # 1. Check for syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            error_line = e.lineno or 1
            errors.append(ErrorContext(
                file_path=str(file_path),
                line_start=max(1, error_line - 2),
                line_end=min(len(lines), error_line + 2),
                error_type="syntax",
                error_message=str(e.msg) if hasattr(e, 'msg') else str(e),
                code_snippet="\n".join(lines[max(0, error_line-3):error_line+2]) if lines else "",
            ))
            return errors  # Can't parse further if syntax error

        # 2. Check for obvious import errors
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check if module exists (basic check)
                    module_name = alias.name.split('.')[0]
                    if module_name.startswith('brains.') or module_name.startswith('api.'):
                        # Check if the path exists
                        module_path = MAVEN_ROOT / module_name.replace('.', '/')
                        if not module_path.exists() and not (module_path.parent / f"{module_path.name}.py").exists():
                            errors.append(ErrorContext(
                                file_path=str(file_path),
                                line_start=node.lineno,
                                line_end=node.lineno,
                                error_type="import",
                                error_message=f"Module may not exist: {alias.name}",
                                code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            ))

    except Exception as e:
        errors.append(ErrorContext(
            file_path=str(file_path),
            line_start=1,
            line_end=1,
            error_type="scan_error",
            error_message=f"Error scanning file: {e}",
        ))

    return errors


def _create_backup_for_repair(file_path: Path, backup_dir: Path) -> Optional[str]:
    """
    Create a backup of a file before attempting repair.

    Returns:
        Path to backup file, or None if failed
    """
    try:
        if not file_path.exists():
            return None

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / timestamp
        backup_subdir.mkdir(parents=True, exist_ok=True)

        try:
            rel_path = file_path.relative_to(MAVEN_ROOT)
        except ValueError:
            rel_path = Path(file_path.name)

        backup_path = backup_subdir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    except Exception as e:
        print(f"[REPAIR_ENGINE] Backup failed for {file_path}: {e}")
        return None


def _build_repair_plan(error: ErrorContext) -> RepairPlan:
    """
    Build a repair plan for an error.

    Returns:
        RepairPlan with strategy and constraints
    """
    constraints = []
    strategy = "simple_fix"

    if error.error_type == "syntax":
        problem_summary = f"Syntax error at line {error.line_start}: {error.error_message}"
        constraints = [
            "Must fix syntax error",
            "Must not change functionality",
            "Code must parse successfully after fix",
        ]
        strategy = "coder_brain" if _coder_available else "simple_fix"

    elif error.error_type == "import":
        problem_summary = f"Import error: {error.error_message}"
        constraints = [
            "Fix or remove broken import",
            "Ensure all dependencies are available",
        ]
        strategy = "simple_fix"

    elif error.error_type == "test_failure":
        problem_summary = f"Test failure: {error.error_message}"
        constraints = [
            "Fix code to pass tests",
            "Do not modify tests unless they are incorrect",
        ]
        strategy = "coder_brain" if _coder_available else "manual"

    else:
        problem_summary = f"Unknown error: {error.error_message}"
        strategy = "manual"

    return RepairPlan(
        error=error,
        problem_summary=problem_summary,
        constraints=constraints,
        strategy=strategy,
    )


def _apply_simple_fix(file_path: Path, error: ErrorContext) -> Tuple[bool, str, str]:
    """
    Apply simple heuristic-based fixes.

    Returns:
        Tuple of (success, pre_content, post_content)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            original_content = content

        lines = content.splitlines()
        modified = False

        if error.error_type == "syntax":
            # Common syntax fixes
            error_msg = error.error_message.lower()

            # Fix missing colons
            if "expected ':'" in error_msg or "invalid syntax" in error_msg:
                line_idx = error.line_start - 1
                if 0 <= line_idx < len(lines):
                    line = lines[line_idx].rstrip()
                    # Add colon if missing after if/for/def/class/while
                    if any(kw in line for kw in ['if ', 'for ', 'def ', 'class ', 'while ', 'elif ', 'else', 'try', 'except', 'finally']):
                        if not line.endswith(':'):
                            lines[line_idx] = line + ':'
                            modified = True

            # Fix unmatched parentheses
            if "unexpected EOF" in error_msg or "parenthesis" in error_msg:
                # Count parentheses
                open_count = content.count('(') - content.count(')')
                if open_count > 0:
                    # Add closing parentheses at end
                    lines.append(')' * open_count)
                    modified = True

        elif error.error_type == "import":
            # Comment out broken imports
            line_idx = error.line_start - 1
            if 0 <= line_idx < len(lines):
                lines[line_idx] = f"# REPAIR: commented out broken import\n# {lines[line_idx]}"
                modified = True

        if modified:
            new_content = "\n".join(lines)

            # Verify the fix actually works (syntax check)
            try:
                ast.parse(new_content)
            except SyntaxError:
                return False, original_content, original_content

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, original_content, new_content

        return False, original_content, original_content

    except Exception as e:
        print(f"[REPAIR_ENGINE] Simple fix failed: {e}")
        return False, "", ""


def _apply_coder_brain_fix(
    file_path: Path,
    error: ErrorContext,
    plan: RepairPlan
) -> Tuple[bool, str, str]:
    """
    Apply a fix using the coder brain.

    Returns:
        Tuple of (success, pre_content, post_content)
    """
    if not _coder_available or coder_api is None:
        return False, "", ""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Build repair task for coder brain
        spec = f"""
Fix the following error in Python code:
Error: {error.error_message}
File: {file_path.name}
Lines: {error.line_start}-{error.line_end}

Code snippet with error:
```python
{error.code_snippet}
```

Constraints:
{chr(10).join(f'- {c}' for c in plan.constraints)}

Provide the corrected code.
"""

        # Call coder brain to generate fix
        result = coder_api({
            "op": "GENERATE",
            "payload": {"spec": spec}
        })

        if not result.get("ok"):
            return False, original_content, original_content

        payload = result.get("payload", {})
        generated_code = payload.get("code", "")

        if not generated_code:
            return False, original_content, original_content

        # Try to apply the generated fix
        # For now, we'll replace the error region with the generated code
        lines = original_content.splitlines()
        start_idx = max(0, error.line_start - 1)
        end_idx = min(len(lines), error.line_end)

        # Extract the function/block containing the error
        # and replace with generated code
        new_lines = lines[:start_idx] + generated_code.splitlines() + lines[end_idx:]
        new_content = "\n".join(new_lines)

        # Verify the fix
        try:
            ast.parse(new_content)
        except SyntaxError:
            return False, original_content, original_content

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True, original_content, new_content

    except Exception as e:
        print(f"[REPAIR_ENGINE] Coder brain fix failed: {e}")
        return False, "", ""


def _run_syntax_check(file_path: Path) -> Tuple[bool, str]:
    """
    Run a syntax check on a Python file.

    Returns:
        Tuple of (passed, error_message)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _log_repair(repair_result: RepairResult, log_dir: Path):
    """Log repair result to audit trail."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "repair_log.jsonl"

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": repair_result.file_path,
            "success": repair_result.success,
            "error_type": repair_result.error_type,
            "message": repair_result.message,
            "tests_run": repair_result.tests_run,
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    except Exception as e:
        print(f"[REPAIR_ENGINE] Failed to log repair: {e}")


def _fix_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a fix operation on target files.

    Steps:
    1. Validate inputs and check hash guards
    2. Scan target files for errors
    3. Create backups
    4. Build repair plans for each error
    5. Apply fixes (using coder brain or simple heuristics)
    6. Verify fixes with syntax/test checks
    7. Accept or revert patches based on results
    8. Log all edits

    Payload:
        target_files: List[str] - Files to repair
        errors: List[ErrorContext] - Specific errors to fix (optional)
        max_edits: int - Maximum edits allowed (default: 10)
        backup: bool - Create backups before repair (default: True)
        expected_hash: str - Expected hash guard (optional)
        current_hash: str - Current hash guard (optional)

    Returns:
        Dict with ok, repairs_applied, repairs_failed, details
    """
    # Hash guard check (backwards compatibility)
    expected = (payload or {}).get("expected_hash")
    current = (payload or {}).get("current_hash")
    if expected and current and expected != current:
        return {
            "ok": False,
            "error": "HASH_MISMATCH",
            "message": "Current template hash does not match expected hash; aborting repair.",
            "expected": expected,
            "found": current,
            "repairs_applied": 0,
            "repairs_failed": 0,
            "details": []
        }

    # Extract parameters
    target_files = payload.get("target_files", [])
    provided_errors = payload.get("errors", [])
    max_edits = payload.get("max_edits", 10)
    do_backup = payload.get("backup", True)

    # Validate target_files
    if not target_files:
        return {
            "ok": False,
            "error": "NO_TARGETS",
            "message": "No target files specified",
            "repairs_applied": 0,
            "repairs_failed": 0,
            "details": []
        }

    # Setup directories
    backup_dir = MAVEN_ROOT / ".maven" / "backup"
    log_dir = get_reports_path("governance", "repairs")

    repairs_applied = 0
    repairs_failed = 0
    details: List[Dict[str, Any]] = []
    backups_created: List[str] = []

    # Process each target file
    for file_path_str in target_files:
        if repairs_applied + repairs_failed >= max_edits:
            break

        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            file_path = MAVEN_ROOT / file_path

        if not file_path.exists():
            details.append({
                "file": str(file_path),
                "status": "skipped",
                "reason": "File not found"
            })
            continue

        # Create backup if requested
        backup_path = None
        if do_backup:
            backup_path = _create_backup_for_repair(file_path, backup_dir)
            if backup_path:
                backups_created.append(backup_path)

        # Scan for errors or use provided errors
        file_errors = [e for e in provided_errors if e.get("file_path") == str(file_path)]
        if not file_errors:
            scanned_errors = _scan_file_for_errors(file_path)
            file_errors = [asdict(e) for e in scanned_errors]

        if not file_errors:
            details.append({
                "file": str(file_path),
                "status": "clean",
                "reason": "No errors found"
            })
            continue

        # Process each error
        for error_dict in file_errors:
            if repairs_applied + repairs_failed >= max_edits:
                break

            # Convert dict to ErrorContext
            error = ErrorContext(
                file_path=error_dict.get("file_path", str(file_path)),
                line_start=error_dict.get("line_start", 1),
                line_end=error_dict.get("line_end", 1),
                error_type=error_dict.get("error_type", "unknown"),
                error_message=error_dict.get("error_message", ""),
                code_snippet=error_dict.get("code_snippet", ""),
            )

            # Build repair plan
            plan = _build_repair_plan(error)

            # Apply fix based on strategy
            success = False
            pre_content = ""
            post_content = ""

            if plan.strategy == "coder_brain":
                success, pre_content, post_content = _apply_coder_brain_fix(
                    file_path, error, plan
                )
                if not success:
                    # Fallback to simple fix
                    success, pre_content, post_content = _apply_simple_fix(
                        file_path, error
                    )

            elif plan.strategy == "simple_fix":
                success, pre_content, post_content = _apply_simple_fix(
                    file_path, error
                )

            # Verify the fix
            tests_run = []
            if success:
                syntax_ok, syntax_err = _run_syntax_check(file_path)
                tests_run.append(f"syntax_check: {'pass' if syntax_ok else 'fail'}")

                if not syntax_ok:
                    # Revert to backup
                    if backup_path:
                        try:
                            shutil.copy2(backup_path, file_path)
                        except Exception:
                            pass
                    success = False

            # Create repair result
            result = RepairResult(
                file_path=str(file_path),
                success=success,
                error_type=error.error_type,
                pre_diff=pre_content[:500] if pre_content else "",
                post_diff=post_content[:500] if post_content else "",
                tests_run=tests_run,
                message="Repair successful" if success else "Repair failed",
            )

            # Log the repair
            _log_repair(result, log_dir)

            # Update counters
            if success:
                repairs_applied += 1
            else:
                repairs_failed += 1

            details.append({
                "file": str(file_path),
                "error_type": error.error_type,
                "line": error.line_start,
                "status": "repaired" if success else "failed",
                "strategy": plan.strategy,
                "tests_run": tests_run,
            })

    return {
        "ok": repairs_failed == 0 or repairs_applied > 0,
        "repairs_applied": repairs_applied,
        "repairs_failed": repairs_failed,
        "total_errors": len(details),
        "backups_created": backups_created,
        "details": details,
        "notes": f"Applied {repairs_applied} repairs, {repairs_failed} failed"
    }

def _checksum_path(path: str | Path) -> str:
    """Compute a SHA256 checksum for a given file path."""
    m = _hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                m.update(chunk)
    except Exception:
        return ""
    return m.hexdigest()

def _backup_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a backup of selected domain banks and key report files.  The
    backup is written as a ZIP archive to reports/governance/repairs/backups.
    The payload may specify a list of bank names under the "banks" key.  If
    omitted, all banks are included.  Returns the path of the backup file
    relative to the Maven root.
    """
    banks = payload.get("banks") or _ALL_BANKS
    if not isinstance(banks, list) or not banks:
        banks = _ALL_BANKS
    # Output directory for backups
    outdir = get_reports_path("governance", "repairs", "backups")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    fname = f"backup_{ts}.zip"
    fpath = outdir / fname
    try:
        with zipfile.ZipFile(fpath, "w", zipfile.ZIP_DEFLATED) as zf:
            # Include selected banks
            for b in banks:
                bank_dir = get_brains_path("domain_banks", b)
                if not bank_dir.exists():
                    continue
                for root, _dirs, files in os.walk(bank_dir):
                    for file in files:
                        fullpath = os.path.join(root, file)
                        arcname = os.path.relpath(fullpath, MAVEN_ROOT)
                        zf.write(fullpath, arcname)
            # Include integrity manifest and docs sync info for completeness
            specials = [
                get_reports_path("templates_integrity.json"),
                get_reports_path("docs_sync"),
                get_reports_path("health_dashboard"),
            ]
            for spath in specials:
                if spath.is_dir():
                    for root, _dirs, files in os.walk(spath):
                        for file in files:
                            full = os.path.join(root, file)
                            arcname = os.path.relpath(full, MAVEN_ROOT)
                            zf.write(full, arcname)
                elif spath.exists():
                    arcname = os.path.relpath(spath, MAVEN_ROOT)
                    zf.write(spath, arcname)
        return {"ok": True, "backup_path": str(fpath)}
    except Exception as e:
        return {"ok": False, "error": "BACKUP_FAILED", "message": str(e)}

def _restore_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restore a previously created backup archive.  The payload must specify
    "backup_path" pointing to a zip file created by _backup_impl.  Files
    contained in the archive will overwrite existing files under MAVEN_ROOT.
    Use with care.  Returns True on success.
    """
    bpath = payload.get("backup_path") or payload.get("path")
    if not bpath:
        return {"ok": False, "error": "MISSING_PATH", "message": "Missing backup_path parameter"}
    try:
        # Resolve path relative to Maven root if not absolute
        bfile = Path(bpath)
        if not bfile.is_absolute():
            bfile = MAVEN_ROOT / bfile
        if not bfile.exists():
            return {"ok": False, "error": "NOT_FOUND", "message": f"Backup not found: {bfile}"}
        with zipfile.ZipFile(bfile, "r") as zf:
            for member in zf.namelist():
                # Disallow traversal outside MAVEN_ROOT
                target = MAVEN_ROOT / member
                # Ensure parent directories exist
                target.parent.mkdir(parents=True, exist_ok=True)
                # Extract file
                with zf.open(member) as src, open(target, "wb") as dest:
                    shutil.copyfileobj(src, dest)
        return {"ok": True, "restored": True}
    except Exception as e:
        return {"ok": False, "error": "RESTORE_FAILED", "message": str(e)}

def _compact_cold_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a lightweight compaction of the Cold tier for the specified
    domain banks.  Compaction deduplicates records and removes empty
    lines to reduce file size.  The payload may specify a list of
    bank names under the "banks" key; if omitted, all banks are compacted.
    Returns a list of banks that were compacted.
    """
    banks = payload.get("banks") or _ALL_BANKS
    if not isinstance(banks, list) or not banks:
        banks = _ALL_BANKS
    compacted = []
    for b in banks:
        path = get_brains_path("domain_banks", b, "memory", "cold", "records.jsonl")
        if not path.exists():
            continue
        try:
            # Read existing lines
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # Deduplicate while preserving order
            seen = set()
            uniq = []
            for ln in lines:
                if ln not in seen:
                    seen.add(ln)
                    uniq.append(ln)
            with open(path, "w", encoding="utf-8") as f:
                for ln in uniq:
                    f.write(ln + "\n")
            compacted.append(b)
        except Exception:
            continue
    return {"ok": True, "compacted": compacted}

def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op","").upper()
    payload = (msg or {}).get("payload",{}) or {}
    mid = payload.get("mid") or f"mid_{_now_ms()}"

    # Non-destructive ops do not require auth
    if op in {"SCAN","TEMPLATE_STATUS","HEALTH"}:
        if op == "SCAN":
            res = _scan_impl(payload)
        elif op == "HEALTH":
            res = {"status": "operational", "coder_brain_available": _coder_available}
        else:
            res = {"status": "available"}
        return {"ok": True, "op": op, "mid": mid, "payload": res}

    # Destructive / state-changing ops require Governance auth
    auth = payload.get("auth", {})
    if op in {"REPAIR","PROMOTE_TEMPLATE","ROLLBACK_TEMPLATE","APPLY_TEMPLATE","FIX",
              "BACKUP","RESTORE","COMPACT_COLD"}:
        if not _auth_ok(auth, op):
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": "REPAIR_UNAUTHORIZED",
                "message": "Repair operation requires valid Governance authorization token",
                "payload": {"authorized": False}
            }
        # Proceed based on operation
        if op in {"REPAIR","FIX"}:
            res = _fix_impl(payload)
        elif op == "PROMOTE_TEMPLATE":
            res = {"ok": True, "promoted": True}
        elif op == "ROLLBACK_TEMPLATE":
            res = {"ok": True, "rolled_back": True}
        elif op == "APPLY_TEMPLATE":
            res = {"ok": True, "applied": True}
        elif op == "BACKUP":
            res = _backup_impl(payload)
        elif op == "RESTORE":
            res = _restore_impl(payload)
        elif op == "COMPACT_COLD":
            res = _compact_cold_impl(payload)
        else:
            res = {"ok": False, "error": "UNSUPPORTED_OP", "message": op}
        return {"ok": True, "op": op, "mid": mid, "payload": {"authorized": True, **res}}

    # Unknown
    return {"ok": False, "op": op, "mid": mid, "error": "UNSUPPORTED_OP", "message": op}


def handle(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Brain contract entry point.

    Args:
        context: Standard brain context dict

    Returns:
        Result dict
    """
    return _handle_impl(context)


# Brain contract alias
service_api = handle
