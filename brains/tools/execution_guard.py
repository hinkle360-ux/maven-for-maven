"""
Execution Guard

Centralized gate for any side-effecting operation:

- Reads both environment variables and a persistent config file.
- Enforces three modes:
    DISABLED   : no side-effecting execution allowed
    READ_ONLY  : read-only operations allowed, writes/exec blocked
    FULL       : both read and write/exec allowed

- Provides explicit enable/disable functions intended to be wired to
  chat commands ("enable execution", "disable execution", "set read-only").

- Logs every decision (allow/deny) into ~/.maven/execution_audit.jsonl
  so that identity_inferencer and governance can see the system's risk posture.

There are NO stubs: all functions perform real work and fail honestly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
CONFIG_PATH = MAVEN_DIR / "config.json"
EXEC_AUDIT_PATH = MAVEN_DIR / "execution_audit.jsonl"

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


class ExecMode(str, Enum):
    DISABLED = "DISABLED"
    READ_ONLY = "READ_ONLY"
    FULL = "FULL"
    FULL_AGENCY = "FULL_AGENCY"  # Full access mode - unrestricted capabilities
    SAFE_CHAT = "SAFE_CHAT"      # No tools - pure conversation mode


# Minimal hard-stop deny list for FULL_AGENCY mode
# These commands could brick the machine and have no legitimate use case
FULL_AGENCY_DENY_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero of=/dev/",
    "dd if=/dev/random of=/dev/",
    "> /dev/sda",
    "chmod -R 777 /",
    "chown -R",
    ":(){:|:&};:",  # fork bomb
    "mv / /dev/null",
    "wget -O- | sh",
    "curl | sh",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "halt",
    "poweroff",
]


@dataclass
class ExecutionConfig:
    mode: ExecMode = ExecMode.DISABLED
    user_confirmed: bool = False
    last_updated: str = ""
    updated_reason: str = ""  # human-readable explanation


@dataclass
class ExecutionStatus:
    mode: ExecMode
    effective: bool  # True if execution (writes/exec) allowed for high-risk ops
    source: str      # "env", "config", "default"
    reason: str = ""

    @property
    def enabled(self) -> bool:
        """Backward compatibility: 'enabled' maps to 'effective'."""
        return self.effective

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access for backward compatibility.

        Maps 'enabled' to 'effective' for legacy code compatibility.
        """
        if key == "enabled":
            return self.effective
        if key == "mode":
            return self.mode.value if isinstance(self.mode, ExecMode) else self.mode
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"ExecutionStatus has no key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-style .get() for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like usage."""
        return key in ("mode", "effective", "enabled", "source", "reason")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return {
            "mode": self.mode.value if isinstance(self.mode, ExecMode) else self.mode,
            "effective": self.effective,
            "enabled": self.effective,  # Backward compat alias
            "source": self.source,
            "reason": self.reason,
        }


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ---------------------
# Config load / save
# ---------------------

def _load_config() -> ExecutionConfig:
    if not CONFIG_PATH.exists():
        return ExecutionConfig(
            mode=ExecMode.DISABLED,
            user_confirmed=False,
            last_updated=_now_iso(),
            updated_reason="default config (no file present)",
        )
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load execution config %s: %s", CONFIG_PATH, e)
        # Fall back to safest mode
        return ExecutionConfig(
            mode=ExecMode.DISABLED,
            user_confirmed=False,
            last_updated=_now_iso(),
            updated_reason="config parse error; defaulted to DISABLED",
        )

    mode_str = str(raw.get("execution_mode", "DISABLED")).upper()
    try:
        mode = ExecMode(mode_str)
    except Exception:
        mode = ExecMode.DISABLED

    return ExecutionConfig(
        mode=mode,
        user_confirmed=bool(raw.get("user_confirmed_execution", False)),
        last_updated=str(raw.get("last_updated") or _now_iso()),
        updated_reason=str(raw.get("updated_reason") or ""),
    )


def _save_config(cfg: ExecutionConfig) -> None:
    MAVEN_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        "execution_mode": cfg.mode.value,
        "user_confirmed_execution": cfg.user_confirmed,
        "last_updated": cfg.last_updated,
        "updated_reason": cfg.updated_reason,
    }
    try:
        CONFIG_PATH.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save execution config %s: %s", CONFIG_PATH, e)


# ---------------------
# Audit logging
# ---------------------

def _append_audit(record: Dict[str, Any]) -> None:
    """
    Append a JSONL record to execution_audit.jsonl.
    This is used by identity_inferencer and governance.
    """
    try:
        MAVEN_DIR.mkdir(parents=True, exist_ok=True)
        with EXEC_AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error("Failed to write execution audit: %s", e)


def _log_decision(
    operation: str,
    intent: str,
    risk: str,
    allowed: bool,
    reason: str,
    mode: ExecMode,
    source: str,
) -> None:
    record = {
        "ts": _now_iso(),
        "kind": "execution_decision",
        "operation": operation,
        "intent": intent,
        "risk": risk,
        "allowed": allowed,
        "reason": reason,
        "mode": mode.value,
        "source": source,
    }
    _append_audit(record)


# ---------------------
# Status resolution
# ---------------------

def _status_from_env() -> Optional[ExecutionStatus]:
    """
    Environment overrides config. This lets you run in locked-down or
    fully-enabled mode regardless of config.
    """
    flag = os.getenv("MAVEN_EXECUTION_ENABLED")
    confirm = os.getenv("USER_CONFIRMED_EXECUTION")
    mode_env = os.getenv("MAVEN_EXECUTION_MODE")

    if flag is None and confirm is None and mode_env is None:
        return None  # no env override

    # If someone is using the old-style flags:
    if flag is not None:
        flag_val = flag.strip().lower()
        enabled = flag_val in TRUTHY
        confirmed = (confirm or "").strip().upper() == "YES"
        eff = enabled and confirmed
        reason = ""
        if not eff:
            reason = "MAVEN_EXECUTION_ENABLED requires USER_CONFIRMED_EXECUTION=YES"
        mode = ExecMode.FULL if eff else ExecMode.DISABLED
        return ExecutionStatus(mode=mode, effective=eff, source="env", reason=reason)

    # If they use MAVEN_EXECUTION_MODE:
    if mode_env is not None:
        mode_str = mode_env.strip().upper()
        try:
            mode = ExecMode(mode_str)
        except Exception:
            mode = ExecMode.DISABLED
        confirmed = (confirm or "").strip().upper() == "YES"
        eff = confirmed and mode == ExecMode.FULL
        reason = ""
        if not confirmed:
            reason = "USER_CONFIRMED_EXECUTION must be YES"
        return ExecutionStatus(mode=mode, effective=eff, source="env", reason=reason)

    return None


def get_execution_status() -> ExecutionStatus:
    """
    Resolve effective execution status from env + config.
    - Env wins if present.
    - Otherwise config is used.
    - If config missing/invalid, defaults to DISABLED.
    """

    env_status = _status_from_env()
    if env_status is not None:
        return env_status

    cfg = _load_config()
    # FULL_AGENCY mode - unrestricted access
    if cfg.mode == ExecMode.FULL_AGENCY and cfg.user_confirmed:
        return ExecutionStatus(mode=cfg.mode, effective=True, source="config")
    if cfg.mode == ExecMode.FULL_AGENCY and not cfg.user_confirmed:
        return ExecutionStatus(
            mode=cfg.mode,
            effective=False,
            source="config",
            reason="execution_mode=FULL_AGENCY but user_confirmed_execution is false",
        )
    # FULL mode
    if cfg.mode == ExecMode.FULL and cfg.user_confirmed:
        return ExecutionStatus(mode=cfg.mode, effective=True, source="config")
    if cfg.mode == ExecMode.FULL and not cfg.user_confirmed:
        return ExecutionStatus(
            mode=cfg.mode,
            effective=False,
            source="config",
            reason="execution_mode=FULL but user_confirmed_execution is false",
        )
    if cfg.mode == ExecMode.READ_ONLY:
        return ExecutionStatus(mode=cfg.mode, effective=False, source="config")
    # DISABLED or anything else:
    return ExecutionStatus(
        mode=ExecMode.DISABLED,
        effective=False,
        source="config",
        reason="execution disabled by configuration",
    )


# ---------------------
# Public guard API
# ---------------------

def check_execution_allowed(
    operation: str,
    risk_level: str,
    intent: str = "",
    write_required: bool = True,
) -> Tuple[bool, str]:
    """
    Core decision function.

    Args:
        operation: short name, e.g. "git_status", "write_file", "run_shell"
        risk_level: "LOW"|"MEDIUM"|"HIGH"|"CRITICAL"
        intent: optional human-readable description
        write_required: True if this operation writes or executes code;
                        False if this is read-only (e.g. listing files).

    Returns:
        (allowed: bool, reason: str)
    """
    status = get_execution_status()
    risk = risk_level.upper()

    # Mode logic
    if status.mode == ExecMode.SAFE_CHAT:
        reason = "SAFE_CHAT mode - no tool execution allowed"
        _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
        return False, reason

    if status.mode == ExecMode.DISABLED:
        reason = status.reason or "execution mode DISABLED"
        _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
        return False, reason

    if status.mode == ExecMode.READ_ONLY:
        if write_required:
            reason = "write/exec not allowed in READ_ONLY mode"
            _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
            return False, reason
        # read-only op allowed
        _log_decision(operation, intent, risk, True, "", status.mode, status.source)
        return True, ""

    # FULL mode
    if status.mode == ExecMode.FULL:
        if not status.effective:
            reason = status.reason or "FULL mode not confirmed"
            _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
            return False, reason

        # Optional: block CRITICAL by default unless explicitly allowed.
        if risk == "CRITICAL":
            reason = "CRITICAL operations require explicit approval; blocked by guard"
            _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
            return False, reason

        _log_decision(operation, intent, risk, True, "", status.mode, status.source)
        return True, ""

    # FULL_AGENCY mode - unrestricted access with minimal hard-stops
    if status.mode == ExecMode.FULL_AGENCY:
        if not status.effective:
            reason = status.reason or "FULL_AGENCY mode not confirmed"
            _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
            return False, reason

        # Check against minimal deny list for truly destructive commands
        # This only applies to shell commands
        if operation in ("run_shell", "shell_exec", "shell_tool"):
            intent_lower = intent.lower() if intent else ""
            for pattern in FULL_AGENCY_DENY_PATTERNS:
                if pattern.lower() in intent_lower:
                    reason = f"FULL_AGENCY mode blocks destructive command pattern: {pattern}"
                    _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
                    return False, reason

        # FULL_AGENCY allows everything else, including CRITICAL operations
        _log_decision(operation, intent, risk, True, "", status.mode, status.source)
        return True, ""

    reason = "Unknown execution mode; treating as DISABLED"
    _log_decision(operation, intent, risk, False, reason, status.mode, status.source)
    return False, reason


def require_execution_allowed(
    operation: str,
    risk_level: str,
    intent: str = "",
    write_required: bool = True,
) -> None:
    """
    Raise PermissionError if not allowed.
    """
    ok, reason = check_execution_allowed(operation, risk_level, intent=intent, write_required=write_required)
    if not ok:
        logger.warning("EXECUTION_GATE_DENIED %s (%s): %s", operation, risk_level, reason)
        raise PermissionError(f"Execution not permitted for {operation}: {reason}")


# ---------------------
# Explicit control API
# ---------------------

def set_mode(mode: ExecMode, user_confirmed: bool, reason: str) -> ExecutionConfig:
    """
    Internal helper to update persistent config.
    """
    cfg = ExecutionConfig(
        mode=mode,
        user_confirmed=user_confirmed,
        last_updated=_now_iso(),
        updated_reason=reason,
    )
    _save_config(cfg)
    # Audit config changes as LOW risk events
    _append_audit(
        {
            "ts": cfg.last_updated,
            "kind": "execution_config_change",
            "mode": cfg.mode.value,
            "user_confirmed": cfg.user_confirmed,
            "reason": cfg.updated_reason,
        }
    )
    return cfg


def enable_execution_full(reason: str = "user enable via command") -> ExecutionConfig:
    """
    Set mode to FULL and mark user_confirmed_execution=True in config.
    Should be wired to a chat command that asks the user for explicit consent.
    """
    return set_mode(ExecMode.FULL, user_confirmed=True, reason=reason)


def enable_full_agency(reason: str = "user enable full agency mode") -> ExecutionConfig:
    """
    Set mode to FULL_AGENCY - unrestricted access with minimal hard-stops.

    FULL_AGENCY mode enables:
    - Read/write anywhere on disk (OS user permissions apply)
    - Run arbitrary shell commands
    - Run arbitrary Python code
    - Browse the web
    - Use git (all operations)
    - Autonomous agent execution

    Only a minimal deny list of truly destructive commands is enforced.
    This mode is intended for full integration with natural language chat.
    """
    return set_mode(ExecMode.FULL_AGENCY, user_confirmed=True, reason=reason)


def enable_execution_read_only(reason: str = "user set read-only") -> ExecutionConfig:
    """
    Set mode to READ_ONLY (safe default for many operations).
    """
    return set_mode(ExecMode.READ_ONLY, user_confirmed=True, reason=reason)


def enable_safe_chat(reason: str = "user set safe chat mode") -> ExecutionConfig:
    """
    Set mode to SAFE_CHAT - no tools, pure conversation.

    SAFE_CHAT mode disables:
    - File access (read/write)
    - Shell execution
    - Python execution
    - Web access
    - Git operations
    - All other tools

    This mode is intended for pure conversation without any side effects.
    """
    return set_mode(ExecMode.SAFE_CHAT, user_confirmed=True, reason=reason)


def disable_execution(reason: str = "user disable via command") -> ExecutionConfig:
    """
    Fully disable side-effecting execution.
    """
    return set_mode(ExecMode.DISABLED, user_confirmed=False, reason=reason)


def execution_status_snapshot() -> Dict[str, Any]:
    """
    Lightweight snapshot for self_model / UI.
    """
    status = get_execution_status()
    cfg = _load_config()
    return {
        "mode": status.mode.value,
        "effective": status.effective,
        "source": status.source,
        "reason": status.reason,
        "config": {
            "mode": cfg.mode.value,
            "user_confirmed": cfg.user_confirmed,
            "last_updated": cfg.last_updated,
            "updated_reason": cfg.updated_reason,
        },
    }


# ---------------------
# Backward compatibility
# ---------------------

def check_execution_enabled(require_confirmation: bool = True) -> Tuple[bool, str]:
    """
    Legacy API - checks if execution is enabled.
    Maps to the new mode system.
    """
    status = get_execution_status()
    if status.mode in (ExecMode.FULL, ExecMode.FULL_AGENCY) and status.effective:
        return True, ""
    return False, status.reason or "Execution not enabled"


def set_execution_config(enabled: bool, confirmed: bool) -> None:
    """
    Legacy API - set execution config.
    Maps to the new mode system.
    """
    if enabled and confirmed:
        enable_execution_full("set via legacy set_execution_config")
    elif enabled:
        enable_execution_read_only("set via legacy set_execution_config (not confirmed)")
    else:
        disable_execution("disabled via legacy set_execution_config")


def require_execution_enabled(operation: str, *, require_confirmation: bool = True, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Legacy API - require execution to be enabled.
    Maps to the new require_execution_allowed function.
    """
    require_execution_allowed(
        operation,
        risk_level="MEDIUM",
        intent=str(metadata) if metadata else "",
        write_required=True
    )
