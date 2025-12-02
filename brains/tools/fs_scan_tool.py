"""Read-only filesystem scan helper for Maven's codebase."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from brains.maven_paths import get_maven_root, validate_path_confinement


def scan_codebase(root: Path | str | None = None, pattern: str = "*.py") -> List[Dict[str, Any]]:
    """List Python files under the Maven root (clamped to repo root)."""

    base_root = get_maven_root()
    target_root = Path(root) if root is not None else base_root
    try:
        target_root = validate_path_confinement(target_root, "fs_scan")
    except Exception:
        target_root = base_root

    entries: List[Dict[str, Any]] = []
    files = sorted(target_root.rglob(pattern))
    for path in files:
        if not path.is_file():
            continue
        try:
            path = validate_path_confinement(path, "fs_scan:file")
        except Exception:
            continue
        try:
            stat = path.stat()
        except Exception:
            continue
        rel_path = path.relative_to(base_root)
        entry = {
            "path": str(rel_path),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        }
        print(f"[FS_SCAN_FILE] path={entry['path']} size={entry['size']}")
        entries.append(entry)

    print(f"[FS_SCAN] root={target_root} files_found={len(entries)}")
    return entries
