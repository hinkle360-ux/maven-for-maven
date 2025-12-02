
from __future__ import annotations
import time, json, sys
from pathlib import Path

DOMAIN_CATEGORIES = ["math","science","history","technology","arts","language_arts","geography","philosophy","economics","law"]

CFG = {
    "paths": {
        "stm": "memory/stm",
        "mtm": "memory/mtm",
        "ltm": "memory/ltm",
        "archive": "memory/archive"
    },
    "rotation": {
        "stm_records": 500,
        "mtm_records": 2000,
        "ltm_records": 8000
    },
    # Optional per-bank rotation thresholds.  If a bank name exists in this mapping,
    # it overrides the global rotation thresholds defined above.  Each entry may
    # contain any subset of {"stm_records", "mtm_records", "ltm_records"}.
    # Example:
    # "arts": {"stm_records": 300, "mtm_records": 1000, "ltm_records": 6000}
    "rotation_per_bank": {},
    # Parallel bank access configuration.  When enabled, the memory
    # librarian will retrieve from domain banks concurrently using
    # ThreadPoolExecutor.  "enabled" toggles the feature on or off and
    # "max_workers" determines the number of concurrent worker threads.
    "parallel_bank_access": {
        "enabled": False,
        "max_workers": 5
    },
    # Pipeline tracer configuration.  When enabled, the Memory Librarian will
    # emit a JSONL trace file for each RUN_PIPELINE invocation.  max_files
    # limits the number of trace files retained in reports/pipeline_trace.
    "pipeline_tracer": {
        "enabled": False,
        "max_files": 25
    },
    "governance": {
        "strict_mode": True,
        "auto_repair": True,
        "allow_apply": True,
        "require_human_ack": False,
        "ruleset_path": "brains/governance/repair_engine/config/repair_rules.json",
        "audit_bias": True
    },
    "weights_defaults": {"verbosity_bias": 0.5, "parse_priority": 0.5, "explain_bias": 0.5}
    ,
    # Domain banks configuration mapping bank names to their settings
    "domain_banks": {
        "arts": {},
        "science": {},
        "history": {},
        "technology": {},
        "language_arts": {},
        "geography": {},
        "philosophy": {},
        "economics": {},
        "law": {},
        "math": {},
        "factual": {},
        "personal": {},
        "procedural": {},
        "creative": {},
        "working_theories": {},
        "theories_and_contradictions": {},
        "stm_only": {},
        "research_reports": {}
    },
    # Deep research configuration.  Web access is enabled by default and may be
    # governed via config overrides or environment variables.
    "web_research": {
        "enabled": True,
        "max_results": 5,
        "max_chars": 8000
    },
    # Global toggle for web research. Enabled by default to keep Maven usable
    # without manual flags.
    "ENABLE_WEB_RESEARCH": True,
    "WEB_RESEARCH_MAX_SECONDS": 1200,
    "WEB_RESEARCH_MAX_REQUESTS": 20,
    # Configuration for the internal hum oscillators.  Each brain has a
    # logical oscillator with its own natural frequency (in arbitrary units).
    # The coupling constant K controls how strongly oscillators influence
    # one another (0 = independent, higher values = more synchrony).  dt_sec
    # defines the default time step for hum updates.  You can disable the
    # hum system entirely by setting "enabled" to False.
    "hum": {
        "enabled": True,
        "K": 0.02,
        "dt_sec": 0.25,
        "freq": {
            "sensorium": 5.2,
            "planner": 5.8,
            "language": 6.0,
            "pattern_recognition": 6.6,
            "memory_librarian": 7.3,
            "reasoning": 5.1,
            "affect_priority": 4.9,
            "personality": 6.2,
            "self_dmn": 4.7,
            "system_history": 4.2
        }
    }
}

# -----------------------------------------------------------------------------
# Configuration overrides
#
# Load optional configuration overrides from JSON files in the project
# ``config`` directory.  Files ending in ``_thresholds.json`` are reserved for
# individual brain modules (e.g. ``self_dmn_thresholds.json``) and are not
# merged into the global CFG.  Overrides found in other JSON files are
# recursively applied to the CFG dictionary at import time.  Invalid JSON
# content is ignored silently.

def _update_dict(d: dict, u: dict) -> dict:
    """Recursively update dict ``d`` with values from ``u`` and return ``d``."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _update_dict(d[k], v)
        else:
            d[k] = v
    return d


def _load_cfg_overrides() -> None:
    """
    Apply configuration overrides from JSON files in the project's top-level
    ``config`` directory.  Any file whose name ends with ``_thresholds.json``
    (used by specific brains) is skipped.  Overrides are merged into the global
    ``CFG`` dictionary recursively.  All errors during reading or parsing are
    suppressed so that missing or malformed override files do not break the
    application.
    """
    try:
        # Derive the Maven project root (the directory containing this module) and locate
        # the config folder relative to it.  For example, if this file resides in
        # ``maven/api/utils.py``, then ``parents[1]`` yields the ``maven`` directory,
        # and we expect a ``config`` subdirectory under that.
        base = Path(__file__).resolve().parents[1] / "config"
        if base.exists():
            for f in base.glob("*.json"):
                if f.name.endswith("_thresholds.json"):
                    continue
                try:
                    overrides = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(overrides, dict):
                    _update_dict(CFG, overrides)
    except Exception:
        # Silently ignore any failures to keep bootstrap robust
        pass


# Invoke the override loader immediately on import
_load_cfg_overrides()

def generate_mid() -> str:
    return f"MID-{int(time.time() * 1000)}"


def success_response(op: str, mid: str, payload: dict) -> dict:
    return {"ok": True, "op": op, "mid": mid, "payload": payload}

def error_response(op: str, mid: str, code: str, message: str) -> dict:
    return {"ok": False, "op": op, "mid": mid, "error": {"code": code, "message": message}}

def _atomic_write(path: Path, data: str) -> None:
    """Write text to a file atomically.

    This helper writes ``data`` to a temporary file in the same
    directory as ``path`` and then atomically replaces ``path`` with
    the temporary file.  This reduces the likelihood of corrupted
    partially written files on crash.  Errors are propagated to the
    caller.
    """
    import os, tempfile
    # Ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def write_report(subdir: str, name: str, content: str) -> str:
    """
    Persist a report in the given subdirectory under ``reports`` using
    atomic writes.  Returns the path to the written file as a string.
    """
    from brains.maven_paths import get_reports_path

    base = get_reports_path(subdir)
    fpath = base / name
    # Use atomic write to avoid corruption
    _atomic_write(fpath, content)
    return str(fpath)

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

# -----------------------------------------------------------------------------
# Atomic JSONL helpers
# -----------------------------------------------------------------------------
# These helpers provide safer, crash‑resistant JSONL writing semantics.  They
# ensure that writes to memory logs do not corrupt existing data by
# constructing a complete new file in a temporary location before replacing
# the original.  They also support basic duplicate detection by computing
# a SHA1 digest of each record.  When the same content is appended twice,
# the helper will skip the write to avoid unbounded growth of STM logs.

import hashlib
from typing import List, Dict, Any

def atomic_jsonl_write(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file atomically.

    The file at ``path`` is replaced in its entirety by a new file
    containing each record serialized as JSON on its own line.  A
    trailing newline is added.  If an error occurs during serialization
    or the atomic swap, the function falls back silently without
    persisting partial data.

    Args:
        path: Destination JSONL file to overwrite.
        records: Sequence of dictionaries to write.
    """
    try:
        # Serialize all records up front.  This may raise if a record is
        # unserializable; in that case abort.
        data_lines: List[str] = []
        for rec in records:
            data_lines.append(json.dumps(rec))
        data = "\n".join(data_lines)
        if data:
            data += "\n"
        else:
            data = ""
        _atomic_write(path, data)
    except Exception:
        # Ignore any failures to avoid corrupting existing file
        try:
            # Fallback: write each record sequentially using append
            for rec in records:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

def append_jsonl_atomic(path: Path, obj: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file using atomic semantics.

    This helper computes a content‐based digest of the record (excluding
    non‑content fields) to assign a stable identifier.  If another
    record with the same digest is already present in the file, the new
    record is considered a duplicate and the write is skipped.  The
    existing records plus the new record (when unique) are then written
    back using an atomic swap.  On any failure, the function falls
    back to a naive append to preserve data but duplicates may occur.

    Args:
        path: The JSONL file to append to.
        obj: The dictionary representing the record.
    """
    # Ensure the directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Compute a content digest for deduplication.  Exclude fields
    # commonly used for identifiers (e.g. id, digest) so that records
    # with identical content but different IDs hash to the same value.
    try:
        # Create a shallow copy to avoid mutating the original
        tmp = dict(obj)
        # Remove id and digest fields if present
        tmp.pop("id", None)
        tmp.pop("digest", None)
        # Serialise with sorted keys for deterministic ordering
        digest = hashlib.sha1(json.dumps(tmp, sort_keys=True).encode("utf-8")).hexdigest()
        # Assign the digest as the record ID when not already set
        if obj.get("id") is None:
            obj["id"] = digest
        # Also record the digest in a separate field for reference
        obj["digest"] = digest
    except Exception:
        digest = None
    # Read existing records and check for duplicates
    records: List[Dict[str, Any]] = []
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # Determine the digest of the existing record.  It
                    # may already include a digest field; otherwise
                    # compute it on the fly by removing id/digest.
                    try:
                        rec_digest = rec.get("digest")
                        if rec_digest is None:
                            tmp2 = dict(rec)
                            tmp2.pop("id", None)
                            tmp2.pop("digest", None)
                            rec_digest = hashlib.sha1(json.dumps(tmp2, sort_keys=True).encode("utf-8")).hexdigest()
                    except Exception:
                        rec_digest = None
                    # Skip the write if a duplicate digest is detected
                    if digest is not None and rec_digest == digest:
                        return
                    records.append(rec)
    except Exception:
        # Treat as no records on any error
        records = []
    # Append the new record and persist atomically
    records.append(obj)
    try:
        atomic_jsonl_write(path, records)
    except Exception:
        # Fallback: naive append (may duplicate or corrupt on crash)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj) + "\n")
        except Exception:
            pass
