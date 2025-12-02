
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Iterable

from .utils import CFG
from brains.maven_paths import get_reports_path

##############################################################################
# Learning helpers
##############################################################################

def compute_success_average(root: Path, n: int = 50) -> float:
    """
    Compute a rolling average of the recent ``success`` flags recorded in the
    short‑term memory (STM) of a given brain or domain bank.  This function
    looks at the last ``n`` records in the STM and returns the fraction of
    those records that were marked as successful.  If no successes are
    recorded or the STM does not exist, it returns 0.0.

    Args:
        root: The root directory of the brain or bank whose STM should be
            inspected.  The STM file is expected under ``memory/<stm>``
            according to ``CFG['paths']``.
        n: The maximum number of recent records to inspect.

    Returns:
        A float in the range [0.0, 1.0] representing the average success
        rate across the inspected records.  When no records are found, the
        result defaults to 0.0.
    """
    try:
        stm_path = root / CFG["paths"].get("stm", "stm") / "records.jsonl"
    except Exception:
        return 0.0
    if not stm_path.exists():
        return 0.0
    successes: list[float] = []
    try:
        with open(stm_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        # iterate backwards over the last n lines
        for line in reversed(lines):
            if len(successes) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            val = obj.get("success")
            if isinstance(val, bool):
                successes.append(1.0 if val else 0.0)
    except Exception:
        return 0.0
    if not successes:
        return 0.0
    try:
        return sum(successes) / float(len(successes))
    except Exception:
        return 0.0

def update_last_record_success(root: Path, success: bool) -> None:
    """
    Update the most recently appended short‑term memory record for the given
    root with a ``success`` flag.  If the STM file does not exist or no
    records are available, the call silently returns.  When a record is
    present, it is parsed as JSON, the ``success`` field is set to the
    provided boolean, and the record is rewritten back into the STM.

    Args:
        root: Brain or bank root whose STM should be updated.
        success: True if the last operation was successful, False otherwise.
    """
    try:
        stm_path = root / CFG["paths"].get("stm", "stm") / "records.jsonl"
    except Exception:
        return
    if not stm_path.exists():
        return
    try:
        with open(stm_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if not lines:
            return
        # parse the last record
        last = lines[-1].strip()
        obj = json.loads(last)
        obj["success"] = bool(success)
        lines[-1] = json.dumps(obj) + "\n"
        with open(stm_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    except Exception:
        # fail silently to avoid crashes during memory updates
        return

def _move_success_records(root: Path) -> None:
    """
    Promote successful short‑term memory records into the medium‑term tier
    immediately.  This helper scans the STM for any entries with
    ``success == True`` and moves them en masse into the MTM.  It also
    removes the moved records from the STM to prevent reprocessing.

    Args:
        root: Root directory whose STM and MTM will be manipulated.
    """
    try:
        stm_path = root / CFG["paths"].get("stm", "stm") / "records.jsonl"
        mtm_path = root / CFG["paths"].get("mtm", "mtm") / "records.jsonl"
    except Exception:
        return
    if not stm_path.exists():
        return
    try:
        with open(stm_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return
    moved: list[str] = []
    remain: list[str] = []
    for line in lines:
        st = line.strip()
        if not st:
            remain.append(line)
            continue
        try:
            obj = json.loads(st)
        except Exception:
            remain.append(line)
            continue
        if obj.get("success") is True:
            moved.append(line)
        else:
            remain.append(line)
    if not moved:
        return
    try:
        # ensure MTM directory exists
        mtm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mtm_path, "a", encoding="utf-8") as fh:
            for l in moved:
                fh.write(l)
        # rewrite STM with remaining records
        with open(stm_path, "w", encoding="utf-8") as fh:
            for l in remain:
                fh.write(l)
    except Exception:
        # ignore failures
        return

def tiers_for(root: Path) -> Dict[str, Path]:
    return {
        "stm": root / CFG["paths"]["stm"] / "records.jsonl",
        "mtm": root / CFG["paths"]["mtm"] / "records.jsonl",
        "ltm": root / CFG["paths"]["ltm"] / "records.jsonl",
        "cold": root / CFG["paths"]["archive"] / "records.jsonl",
    }

def ensure_dirs(root: Path) -> Dict[str, Path]:
    t = tiers_for(root)
    for p in t.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()
    return t

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a record to a JSONL file with duplicate suppression and atomic writes.

    This wrapper delegates to ``append_jsonl_atomic`` in ``api.utils`` to
    ensure that duplicate records (based on a SHA1 digest of the full
    dictionary) are not appended and that the file is rewritten
    atomically to avoid corruption.  On any failure, a best effort
    append is attempted as a fallback.

    Args:
        path: Destination JSONL file to append to.
        obj: Record to append.
    """
    try:
        from .utils import append_jsonl_atomic  # type: ignore
        append_jsonl_atomic(path, obj)
    except Exception:
        # Fallback to naive append if atomic write fails
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")
        except Exception:
            pass

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def iterate_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

# -----------------------------------------------------------------------------
# Memory rotation helpers
# -----------------------------------------------------------------------------
#
# The cognitive brains and domain banks within Maven accumulate short‑term
# memories (STM) rapidly during operation.  Without periodic rotation of
# these logs into medium‑term (MTM) and long‑term (LTM) tiers, the STM files
# will continue to grow without bound, eventually triggering memory overflow
# warnings during health checks.  To address this, the functions below
# implement a simple record rotation mechanism based on configurable
# thresholds defined in ``api.utils.CFG``.  When the number of records in
# a tier exceeds its limit, the oldest entries are moved into the next tier.
# Once records reach the LTM threshold they are moved into cold storage.

def _move_records(root: Path, from_tier: str, to_tier: str, n: int) -> None:
    """Move the oldest ``n`` records from ``from_tier`` to ``to_tier`` for a given root.

    Args:
        root: The brain or bank root directory containing ``memory/<tier>`` subdirectories.
        from_tier: Source tier name: ``"stm"``, ``"mtm"`` or ``"ltm"``.
        to_tier: Destination tier name: ``"mtm"``, ``"ltm"`` or ``"cold"``.
        n: Number of records to move (0 or less means no operation).

    This helper reads the source ``records.jsonl`` file, moves the oldest
    ``n`` entries to the destination file, and rewrites the source file with
    the remaining lines.  Destination directories are created as needed.  If the
    source file does not exist or no records need to be moved, the function
    returns silently.
    """
    if n <= 0:
        return
    # Map tier names to CFG keys (cold storage is named 'cold_storage' in CFG)
    tier_map = {"stm": "stm", "mtm": "mtm", "ltm": "ltm", "cold": "cold_storage"}
    src_path = root / CFG["paths"][tier_map[from_tier]] / "records.jsonl"
    dst_path = root / CFG["paths"][tier_map[to_tier]] / "records.jsonl"
    if not src_path.exists():
        return
    try:
        with open(src_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return
    m = min(n, len(lines))
    if m <= 0:
        return
    move_lines = lines[:m]
    remain_lines = lines[m:]
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Append moved records to destination
    try:
        with open(dst_path, "a", encoding="utf-8") as fh:
            for line in move_lines:
                fh.write(line)
    except Exception:
        pass
    # Rewrite remaining records back to source
    try:
        with open(src_path, "w", encoding="utf-8") as fh:
            for line in remain_lines:
                fh.write(line)
    except Exception:
        pass

def rotate_if_needed(root: Path, thresholds: Dict[str, Any] | None = None) -> None:
    """Rotate memory records across STM→MTM→LTM→Cold tiers based on thresholds.

    Inspects the counts of records in each tier under ``root`` and moves
    the oldest records to the next tier when a configured limit is exceeded.
    If ``thresholds`` is not provided, global defaults from ``CFG['rotation']``
    and optional per‑bank overrides from ``CFG['rotation_per_bank']`` are used.

    Args:
        root: Path of the brain or bank root directory.
        thresholds: Optional dictionary specifying maximum counts for
            ``stm_records``, ``mtm_records`` and ``ltm_records``.  A value of
            0 or ``None`` disables rotation for that tier.
    """
    # Promote successful records into MTM before enforcing record limits.  This
    # ensures high quality operations rise through memory tiers early based on
    # performance rather than just overflow.
    try:
        _move_success_records(root)
    except Exception:
        pass

    if thresholds is None:
        root_name = root.name
        per_bank = (CFG.get("rotation_per_bank", {}) or {}).get(root_name, {})
        global_rot = CFG.get("rotation", {}) or {}
        thresholds = {
            "stm_records": int(per_bank.get("stm_records", global_rot.get("stm_records", 0) or 0)),
            "mtm_records": int(per_bank.get("mtm_records", global_rot.get("mtm_records", 0) or 0)),
            "ltm_records": int(per_bank.get("ltm_records", global_rot.get("ltm_records", 0) or 0)),
        }
    tiers = ensure_dirs(root)
    try:
        stm_count = count_lines(tiers["stm"])
        mtm_count = count_lines(tiers["mtm"])
        ltm_count = count_lines(tiers["ltm"])
    except Exception:
        return
    stm_limit = thresholds.get("stm_records", 0)
    if stm_limit and stm_count > stm_limit:
        to_move = stm_count - stm_limit
        _move_records(root, "stm", "mtm", to_move)
        mtm_count += to_move
        stm_count = stm_limit
    mtm_limit = thresholds.get("mtm_records", 0)
    if mtm_limit and mtm_count > mtm_limit:
        to_move = mtm_count - mtm_limit
        _move_records(root, "mtm", "ltm", to_move)
        ltm_count += to_move
        mtm_count = mtm_limit
    ltm_limit = thresholds.get("ltm_records", 0)
    if ltm_limit and ltm_count > ltm_limit:
        to_move = ltm_count - ltm_limit
        _move_records(root, "ltm", "cold", to_move)

# -----------------------------------------------------------------------------
# Autotune
# -----------------------------------------------------------------------------

def autotune(root: Path, window: int = 50) -> None:
    """
    Dynamically adjust memory rotation thresholds based on recent success rates.

    This helper computes a rolling success average from the STM of the given root
    and derives new limits for the number of STM records to retain.  The
    thresholds are written to a JSON override file under the project's ``config``
    directory and also merged into the in‑memory ``CFG`` object.  This enables
    per‑deployment tuning without manual edits to code.

    Args:
        root: The brain or bank root directory whose STM should be analyzed.
        window: Number of recent records to consider when computing the success
            average.
    """
    try:
        # Import here to avoid circular dependencies at import time
        from .utils import CFG, _update_dict  # type: ignore
    except Exception:
        return
    # Compute the rolling success average for the given root
    try:
        avg = compute_success_average(root, n=window)
        # Derive a new STM limit: scale between 50 and 500 based on average
        new_stm = 50 + int(max(0.0, min(1.0, avg)) * 450)
    except Exception:
        new_stm = 100
    overrides: Dict[str, Any] = {"rotation": {"stm_records": new_stm}}
    # Update global CFG immediately
    try:
        _update_dict(CFG, overrides)
    except Exception:
        pass
    # Persist override to config/autotune.json so future imports apply it
    try:
        cfg_base = Path(__file__).resolve().parents[1] / "config"
        cfg_base.mkdir(parents=True, exist_ok=True)
        override_path = cfg_base / "autotune.json"
        with open(override_path, "w", encoding="utf-8") as fh:
            json.dump(overrides, fh, indent=2)
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Session persistence helpers
# ----------------------------------------------------------------------------

import time

def save_session_summary(summary: Dict[str, Any]) -> None:
    """
    Persist a session summary to the system reports directory.

    A session summary captures high‑level context about a chat or pipeline
    interaction.  When called, this helper writes the provided summary
    dictionary to a timestamped JSON file under ``reports/system``.  If the
    destination directory does not exist, it is created.  Failures to
    write the file are silently ignored to avoid interrupting the caller.

    Args:
        summary: A JSON‑serializable dictionary describing the session.
    """
    try:
        # Determine project root and ensure reports/system exists
        out_dir = get_reports_path("system")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f"session_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    except Exception:
        # Silent failure to avoid disrupting pipeline
        return

def load_recent_sessions(k: int = 1) -> list[Dict[str, Any]]:
    """
    Load the most recent session summaries from the system reports directory.

    This helper returns up to ``k`` of the newest session summary files
    stored under ``reports/system``.  If no session files are found or they
    cannot be parsed, an empty list is returned.  The returned list is
    ordered from oldest to newest of the selected files.

    Args:
        k: Number of recent session summaries to load.

    Returns:
        A list of dictionaries representing session summaries, in ascending
        chronological order.
    """
    summaries: list[Dict[str, Any]] = []
    try:
        out_dir = get_reports_path("system")
        if not out_dir.exists():
            return []
        files = sorted(out_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime)
        # Take the last k files
        selected = files[-k:] if k > 0 else files
        for fp in selected:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    summaries.append(json.load(fh))
            except Exception:
                continue
        return summaries
    except Exception:
        return summaries
