"""Safe file writer with confinement and backups."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from brains.maven_paths import MAVEN_ROOT, validate_path_confinement
from brains.tools.execution_guard import require_execution_enabled

_ALLOWED_ROOTS = [
    MAVEN_ROOT / "brains",
    MAVEN_ROOT / "brains" / "experimental",
    MAVEN_ROOT / "self_tools",
]


def _ensure_allowed(target: Path) -> Path:
    target = validate_path_confinement(target, "fs_write")
    for allowed_root in _ALLOWED_ROOTS:
        try:
            target.relative_to(allowed_root)
            return target
        except ValueError:
            continue
    raise PermissionError(f"Write blocked: {target} is outside allowed directories")


def write_file(relative_path: str, content: Any, *, mode: str = "create_or_backup") -> Path:
    """Write a file under a whitelisted directory with backup support."""

    require_execution_enabled("fs_write")

    if not relative_path:
        raise ValueError("relative_path is required")

    target = MAVEN_ROOT / relative_path
    target = _ensure_allowed(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Path | None = None
    if target.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        backup_path = target.with_suffix(target.suffix + f".bak-{timestamp}")
        shutil.copy2(target, backup_path)
        print(f"[FS_WRITE_BACKUP] original={target} backup={backup_path}")

    data = "" if content is None else str(content)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(data)

    print(f"[FS_WRITE] path={target} bytes={len(data)} mode={mode}")
    return target
