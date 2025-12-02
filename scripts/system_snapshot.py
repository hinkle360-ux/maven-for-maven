import os
import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # maven2_fix/
SNAPSHOT_DIR = PROJECT_ROOT / "reports" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def scan_codebase():
    py_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # skip virtualenvs, git, etc.
        parts = Path(root).parts
        if any(p in {".git", ".venv", "__pycache__"} for p in parts):
            continue
        for name in files:
            if name.endswith(".py"):
                full = Path(root) / name
                rel = full.relative_to(PROJECT_ROOT)
                py_files.append({"rel": str(rel), "abs": str(full)})
    return py_files


def main():
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    data = {
        "timestamp": now,
        "project_root": str(PROJECT_ROOT),
        "python_files": scan_codebase(),
    }
    out_json = SNAPSHOT_DIR / f"code_snapshot_{now}.json"
    out_txt = SNAPSHOT_DIR / f"code_snapshot_{now}.txt"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Snapshot at {now}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Total Python files: {len(data['python_files'])}\n\n")
        for item in sorted(data["python_files"], key=lambda x: x["rel"]):
            f.write(item["rel"] + "\n")

    print(f"Snapshot written to {out_json} and {out_txt}")


if __name__ == "__main__":
    main()
