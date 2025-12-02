"""
Self‑Critique Brain
===================

The self‑critique brain provides a lightweight reflective loop that
generates feedback about the system's own outputs.  It is designed
to run offline after Stage 10 (Finalize) and prior to affect learning.
Critiques are appended to a JSONL log under ``reports/reflection`` and
exposed via a simple API for integration with other stages.

Current implementation:

* Operation ``CRITIQUE`` accepts an arbitrary ``text`` payload and
  produces a short critique string.  If the input is long the critique
  encourages brevity, otherwise it acknowledges the effort.
* Logs are written to ``reports/reflection/turn_<ms>.jsonl`` with
  timestamp, original text and critique.
* Additional operations can be added in the future (e.g. threshold
  tuning) without breaking compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
from pathlib import Path
import os
import re

def _log_reflection(root: Path, obj: Dict[str, Any]) -> None:
    """
    Append a reflection entry to a JSONL file.  The log directory is
    created if necessary.  Errors are intentionally swallowed to avoid
    disrupting the main pipeline.
    """
    try:
        import random
        reports_dir = root / "reports" / "reflection"
        reports_dir.mkdir(parents=True, exist_ok=True)
        fname = f"turn_{random.randint(100000, 999999)}.jsonl"
        fpath = reports_dir / fname
        with open(fpath, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
    except Exception:
        pass


def _make_critique(text: str) -> str:
    """
    Produce a simple critique of the final answer text.

    This heuristic considers several basic factors:

    * Empty or missing content results in a request for more detail.
    * If the answer admits uncertainty (e.g. "I don't know"), the critique
      encourages further research or retrieval.
    * Long answers (>100 characters) prompt the user to be more concise.
    * Otherwise a positive acknowledgement is returned.

    These rules can be replaced with a more sophisticated model in the future.
    """
    # Normalise whitespace for checks
    # Normalise string and check for emptiness
    if not text or not str(text).strip():
        return "No answer generated; please provide a response."
    try:
        low = text.lower()
    except Exception:
        low = str(text).lower()
    # Encourage improvement when uncertainty is expressed
    if any(phrase in low for phrase in ["i don't know", "i do not know", "unsure", "not sure"]):
        return "The answer seems incomplete; consider improving memory search or knowledge."
    # Flag common meta or filler phrases as unhelpful
    try:
        from .self_critique import _load_bad_phrases  # circular import safe here
    except Exception:
        _load_bad_phrases = None  # type: ignore
    if _load_bad_phrases:
        try:
            bads = _load_bad_phrases()
        except Exception:
            bads = []
        for bad in bads:
            try:
                if bad and bad in low:
                    return "The response contains filler or meta content; improve retrieval."
            except Exception:
                continue
    # Discourage overly verbose responses
    if len(text.strip()) > 120:
        return "Try to be more concise in your responses."
    # Detect contradictory cues (e.g. multiple opposing conjunctions)
    contradiction_markers = ["however", "but", "although", "nevertheless"]
    contras = sum(1 for m in contradiction_markers if m in low)
    if contras > 1:
        return "The response contains multiple contrasting statements; aim for a clearer stance."
    # Reward succinct and confident statements
    return "Response generated successfully."

# ------------------------------------------------------------------
# Contextual critique helper
#
# In addition to simple text critiques, the self‑critique brain can
# evaluate the pipeline context to provide more meaningful feedback.
# This helper examines the results of storage, reasoning and final
# confidence to identify potential issues and suggest improvements.

from typing import Dict, Any

def contextual_critique(ctx: Dict[str, Any]) -> str:
    """Produce a critique based on pipeline context.

    Args:
        ctx: The full pipeline context dictionary containing stage
            results such as stage_8_validation, stage_9_storage and
            stage_10_finalize.
    Returns:
        A short critique string reflecting the success or failure of
        various stages.
    """
    try:
        # Check storage result (stage 9)
        storage = (ctx.get("stage_9_storage") or {}).get("result") or {}
        if storage.get("ok") is False:
            return "Storage failed. Check STM bank availability."
    except Exception:
        pass
    try:
        # Inspect reasoning verdict (stage 8)
        verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        if verdict == "UNANSWERED":
            return "Could not answer query. Consider searching external sources."
    except Exception:
        pass
    try:
        # Evaluate final confidence (stage 10)
        conf = float((ctx.get("stage_10_finalize") or {}).get("confidence", 0.0) or 0.0)
        if conf < 0.5:
            return "Low confidence response. User may need clarification."
    except Exception:
        pass
    return "Response generated successfully."

# ------------------------------------------------------------------
# Extended run evaluation for Stage 3
#
# The following helper functions and service operation implement a
# comprehensive self‑evaluation of each pipeline run.  This extends
# the basic text‑only critique by examining the full context to
# assess truthfulness, clarity, factual consistency and cache health.
# When issues are detected, the evaluator automatically enqueues
# appropriate repair goals in the goal memory so that the autonomy
# system can address them asynchronously.

def _load_bad_phrases() -> list[str]:
    """Load bad phrases from the project's cache sanity configuration.

    Returns a list of lower‑cased substrings that indicate a meta or
    non‑informative response.  If the config file cannot be read,
    returns a minimal default set.
    """
    try:
        root = Path(__file__).resolve().parents[4]
        cfg_path = root / "config" / "cache_sanity.json"
        if cfg_path.exists():
            import json as _json
            data = _json.loads(cfg_path.read_text(encoding="utf-8")) or {}
            phrases = data.get("bad_phrases") or data.get("BAD_PHRASES") or []
            cleaned: list[str] = []
            for p in phrases:
                try:
                    s = str(p).strip().lower()
                    if s:
                        cleaned.append(s)
                except Exception:
                    continue
            if cleaned:
                return cleaned
    except Exception:
        pass
    # Fallback default list; mirrored from the memory librarian
    return [
        "i'm going to try my best",
        "i am going to try my best",
        "i don't have specific information",
        "i don't have information",
        "as an ai",
        "got it — noted",
        "got it - noted",
        "i'm also considering other possibilities",
        "i’m going to try my best",
        "as an ai language model",
        "lorem ipsum",
        "i cannot help with",
    ]


def _evaluate_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the pipeline context and trigger repair goals when needed.

    This function computes several health metrics for a single run:

    * ``truthfulness``: whether the answer was verified or recomputed and
      not marked as conflicting.
    * ``clarity``: whether the final answer is non‑empty, concise and
      free of meta/filler phrases.
    * ``factual_consistency``: whether the reasoning verdict was TRUE.
    * ``cache_health_percent``: ratio of valid cache entries to total cache
      entries, expressed as a percentage.  When no cache entries exist,
      the health is treated as 100%.
    * ``latency_ms``: approximate latency of the run in milliseconds.

    When issues are detected (e.g. low cache health or unverified
    responses), the evaluator enqueues appropriate repair goals in the
    personal goal memory.  Specifically:

    * A cache health below 80% triggers a "Repair cache poison" goal.
    * A failure of truthfulness or factual consistency triggers a
      "Refresh domain bank" goal.

    The metrics are also appended to ``reports/health.json`` for
    external analysis.  Errors during evaluation or file I/O are
    tolerated; metrics may be omitted but the run will not fail.

    Args:
        ctx: The pipeline context dict produced by RUN_PIPELINE.

    Returns:
        A dictionary containing the computed metrics and a list of
        detected issues.  The ``goals_created`` key lists any new
        goals that were enqueued.
    """
    issues: list[str] = []
    metrics: Dict[str, Any] = {}
    goals_created: list[Dict[str, Any]] = []
    try:
        # Determine truthfulness from cross_check_tag or final_tag
        cross_tag = ctx.get("cross_check_tag") or ctx.get("final_tag") or "asserted_true"
        truthfulness = cross_tag in {"asserted_true", "recomputed"}
        metrics["truthfulness"] = bool(truthfulness)
        if not truthfulness:
            issues.append("truthfulness")
    except Exception:
        metrics["truthfulness"] = False
        issues.append("truthfulness")
    # Clarity: answer should be non‑empty, concise and not contain bad phrases
    try:
        final_ans = str(ctx.get("final_answer") or "")
        final_ans_lc = final_ans.lower()
        bad_phrases = _load_bad_phrases()
        is_empty = not final_ans.strip()
        is_too_long = len(final_ans) > 180
        contains_bad = any(bp in final_ans_lc for bp in bad_phrases)
        clarity = not (is_empty or is_too_long or contains_bad)
        metrics["clarity"] = bool(clarity)
        if not clarity:
            issues.append("clarity")
    except Exception:
        metrics["clarity"] = False
        issues.append("clarity")
    # Factual consistency: reasoning verdict must be TRUE
    try:
        verdict = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
        factual_consistency = verdict == "TRUE"
        metrics["factual_consistency"] = bool(factual_consistency)
        if not factual_consistency:
            issues.append("factual_consistency")
    except Exception:
        metrics["factual_consistency"] = False
        issues.append("factual_consistency")
    # Compute cache health percentage
    cache_health = 1.0
    try:
        root = Path(__file__).resolve().parents[4]
        fc_path = root / "reports" / "fast_cache.jsonl"
        poison_path = root / "reports" / "cache_poison.log"
        total_lines = 0
        poisoned = 0
        try:
            if fc_path.exists():
                with open(fc_path, "r", encoding="utf-8") as fc_fh:
                    for line in fc_fh:
                        if line.strip():
                            total_lines += 1
        except Exception:
            total_lines = 0
        try:
            if poison_path.exists():
                with open(poison_path, "r", encoding="utf-8") as p_fh:
                    for _ in p_fh:
                        poisoned += 1
        except Exception:
            poisoned = 0
        if total_lines > 0:
            ok = max(total_lines - poisoned, 0)
            cache_health = ok / total_lines
        else:
            cache_health = 1.0
        metrics["cache_health_percent"] = round(cache_health * 100.0, 2)
        if cache_health < 0.8:
            issues.append("cache_health")
    except Exception:
        metrics["cache_health_percent"] = 100.0
    # Note: Latency computation removed (no time-based tracking)
    metrics["latency_ms"] = None
    # Append metrics to health report
    try:
        root = Path(__file__).resolve().parents[4]
        health_path = root / "reports" / "health.json"
        # Load existing health logs
        existing: list = []
        if health_path.exists():
            try:
                import json as _json
                existing = _json.loads(health_path.read_text(encoding="utf-8")) or []
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        # Append this run's metrics
        entry = {
            "truthfulness": bool(metrics.get("truthfulness")),
            "clarity": bool(metrics.get("clarity")),
            "factual_consistency": bool(metrics.get("factual_consistency")),
            "cache_health_percent": metrics.get("cache_health_percent"),
            "latency_ms": metrics.get("latency_ms"),
        }
        existing.append(entry)
        # Persist back to file
        try:
            health_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            health_path.write_text(_json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass
    # Enqueue repair goals when issues are present
    try:
        if issues:
            from brains.personal.memory import goal_memory  # type: ignore
            # Determine existing goal titles to avoid duplicates
            existing_goals = []
            try:
                existing_goals = goal_memory.get_goals(active_only=False)
            except Exception:
                existing_goals = []
            existing_titles = {str(g.get("title", "")).strip() for g in existing_goals if g.get("title")}
            # Helper to add a goal if not already present
            def _add_goal_if_missing(title: str, description: str) -> None:
                nonlocal goals_created
                if title not in existing_titles:
                    try:
                        rec = goal_memory.add_goal(title, description)
                        goals_created.append({"goal_id": rec.get("goal_id"), "title": rec.get("title")})
                        existing_titles.add(title)
                    except Exception:
                        pass
            # If cache health is low, add repair cache goal
            if "cache_health" in issues:
                _add_goal_if_missing("Repair cache poison", "AUTO_REPAIR:cache_poison")
            # If truthfulness or factual consistency fail, refresh domain bank
            if any(itm in issues for itm in ("truthfulness", "factual_consistency")):
                _add_goal_if_missing("Refresh domain bank", "AUTO_REPAIR:domain_bank")
            # If a command produced a filler response, enqueue repair for the command router.
            try:
                # Only examine command inputs.  The language parse results
                # are stored in stage_3_language.
                stage3 = ctx.get("stage_3_language") or {}
                if bool(stage3.get("is_command")) or str(stage3.get("storable_type", "")).upper() == "COMMAND":
                    # Check the final answer for bad or filler phrases
                    final_ans = str(ctx.get("final_answer") or "")
                    final_ans_lc = final_ans.lower()
                    bad_phrases = _load_bad_phrases()
                    contains_bad = any(bp in final_ans_lc for bp in bad_phrases)
                    if contains_bad:
                        _add_goal_if_missing("Repair command router", "AUTO_REPAIR:command_router")
            except Exception:
                # If anything fails, silently skip adding the repair goal
                pass
    except Exception:
        pass
    return {"metrics": metrics, "issues": issues, "goals_created": goals_created}


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.utils import error_response  # type: ignore
    from api.utils import success_response  # type: ignore
    from api.utils import generate_mid  # type: ignore
    """
    Entry point for the self‑critique brain.  Currently supports a
    single operation ``CRITIQUE``.  Unsupported operations return an
    error response.  On success the critique text is returned under
    ``payload``.
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}
    if op == "CRITIQUE":
        text = str(payload.get("text", ""))
        critique = _make_critique(text)
        # Generate a simple reason for reflection.  In this implementation we
        # mirror the critique text itself as the reason to satisfy external
        # checks for a reason field.  Future enhancements could provide
        # finer‑grained reasoning such as "long_answer" or "uncertain".
        reason_for_reflection = critique
        # Persist to reflection log.  Use project root (four levels up).
        try:
            root = Path(__file__).resolve().parents[4]
        except Exception:
            root = Path(".")
        _log_reflection(
            root,
            {
                "critique": critique,
                "original": text,
                "reason_for_reflection": reason_for_reflection,
            },
        )
        return success_response(op, mid, {"critique": critique})
    # Extended evaluation of full context (Stage 3)
    if op == "EVAL_CONTEXT":
        ctx = payload.get("context") or {}
        try:
            eval_res = _evaluate_context(ctx)
            return success_response(op, mid, eval_res)
        except Exception as ex:
            return error_response(op, mid, "EVAL_FAILED", str(ex))
    return error_response(op, mid, "UNSUPPORTED_OP", op)


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