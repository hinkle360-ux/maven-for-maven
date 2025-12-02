
from __future__ import annotations
import json, glob
from pathlib import Path
from typing import Dict, Any, List, Optional

# BrainMemory API for tier-compliant storage
from brains.memory.brain_memory import BrainMemory

# TruthClassifier for content classification
from brains.cognitive.reasoning.truth_classifier import TruthClassifier

# Teacher integration for learning self-reflection and meta-cognitive patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("self_dmn")
except Exception as e:
    print(f"[SELF_DMN] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[SELF_DMN] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Optional integration with personal brain and identity journal.  If available,
# `self_dmn_manage` provides higher level reflection and promotion operations
# that unify the personal brain with the self model.
try:
    import brains.cognitive.self_dmn.service.self_dmn_manage as manage_mod
except Exception:
    manage_mod = None
# Dynamically import the skeptic submodule and its thresholds.  Relative
# imports may fail when this module is loaded outside a package context,
# so resolve the absolute path using importlib.  If the import fails, use
# sensible defaults and provide a stubbed skeptic API that returns an error.
import importlib
try:
    _skeptic_mod = importlib.import_module(
        "brains.cognitive.self_dmn.service.self_dmn_skeptic"
    )
    _skeptic_service_api = getattr(_skeptic_mod, "service_api")
    TAU1 = getattr(_skeptic_mod, "TAU1", 0.25)
    TAU2 = getattr(_skeptic_mod, "TAU2", 0.60)
except Exception:
    # Fallback values and a stub API in case the skeptic cannot be imported
    TAU1 = 0.25
    TAU2 = 0.60
    def _skeptic_service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
        from api.utils import error_response  # type: ignore
        from api.utils import generate_mid  # type: ignore
        op = (msg or {}).get("op", " ").upper()
        mid = msg.get("mid") or generate_mid()
        return error_response(op, mid, "ERROR", "Skeptic module unavailable")

HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent
IDENTITY_CARD_PATH = BRAIN_ROOT / "identity_card.json"

# Initialize BrainMemory for self_dmn brain
memory = BrainMemory("self_dmn")


# =============================================================================
# STEP 2: SELF-HEALING META-LESSON MECHANISM
# =============================================================================
# When confusion is detected (via REFLECT_ON_ERROR, reasoning_trace, or
# EXPLAIN_LAST), Maven creates "meta-lessons" that capture:
# - "I misunderstood X because Y"
# - "I should treat phrases like ___ as category Z"
# - "This pattern should be stored as canonical"
#
# This enables:
# - Self-correction
# - Self-stabilization
# - Reduction of future errors
# - Real meta-learning beyond typical LLM behavior

def create_meta_lesson(
    confusion_type: str,
    input_pattern: str,
    misunderstanding: str,
    correct_interpretation: str,
    source_brain: str = "self_dmn",
    confidence: float = 0.8
) -> Dict[str, Any]:
    """
    Create a meta-lesson from a detected confusion or error.

    Meta-lessons are higher-order learning records that capture HOW Maven
    should interpret patterns, not just facts. They enable self-healing
    by preventing the same misunderstanding from recurring.

    Args:
        confusion_type: Type of confusion (e.g., "category_error", "pattern_mismatch",
                       "ambiguous_phrase", "context_loss", "hallucination")
        input_pattern: The input pattern that caused confusion
        misunderstanding: What Maven misunderstood
        correct_interpretation: How Maven should interpret this in the future
        source_brain: Which brain detected the confusion
        confidence: Confidence in the meta-lesson (0.0-1.0)

    Returns:
        Meta-lesson dict with all fields populated

    Example:
        create_meta_lesson(
            confusion_type="category_error",
            input_pattern="plan my week",
            misunderstanding="Treated as generic question instead of planning request",
            correct_interpretation="This is a planning_week concept - use planning strategies",
            source_brain="planner",
            confidence=0.9
        )
    """
    import time

    meta_lesson = {
        "kind": "meta_lesson",
        "confusion_type": confusion_type,
        "input_pattern": input_pattern[:200],  # Limit size
        "misunderstanding": misunderstanding[:500],
        "correct_interpretation": correct_interpretation[:500],
        "source_brain": source_brain,
        "confidence": confidence,
        "timestamp": time.time(),
        "status": "active",  # active, applied, superseded
        "application_count": 0,  # Track how often this lesson is used
    }

    return meta_lesson


def store_meta_lesson(meta_lesson: Dict[str, Any]) -> bool:
    """
    Store a meta-lesson to memory for future self-healing.

    Meta-lessons are stored with high priority metadata to ensure they
    are retrieved during similar confusion scenarios.

    Args:
        meta_lesson: The meta-lesson dict from create_meta_lesson()

    Returns:
        True if stored successfully
    """
    try:
        # Classify using TruthClassifier
        classification = TruthClassifier.classify(
            content=f"Meta-lesson: {meta_lesson.get('confusion_type', 'unknown')}",
            confidence=meta_lesson.get("confidence", 0.8),
            evidence={
                "type": "meta_lesson",
                "confusion_type": meta_lesson.get("confusion_type"),
                "source_brain": meta_lesson.get("source_brain")
            }
        )

        if TruthClassifier.should_store_in_memory(classification):
            memory.store(
                content=meta_lesson,
                metadata={
                    "kind": "meta_lesson",
                    "confusion_type": meta_lesson.get("confusion_type"),
                    "source_brain": meta_lesson.get("source_brain"),
                    "confidence": classification["confidence"],
                    "truth_type": classification["type"],
                    "input_pattern_hash": hash(meta_lesson.get("input_pattern", ""))
                }
            )
            print(f"[SELF_DMN] ✓ Stored meta-lesson: {meta_lesson.get('confusion_type')}")
            return True

    except Exception as e:
        print(f"[SELF_DMN] Failed to store meta-lesson: {e}")

    return False


def lookup_meta_lessons(input_pattern: str, confusion_type: str = None) -> List[Dict[str, Any]]:
    """
    Look up relevant meta-lessons for a given input pattern.

    Used during processing to check if there are existing meta-lessons
    that apply to the current input, enabling self-healing.

    Args:
        input_pattern: The input pattern to check
        confusion_type: Optional filter by confusion type

    Returns:
        List of matching meta-lessons, sorted by relevance
    """
    try:
        # Search for meta-lessons
        query = f"meta_lesson:{input_pattern[:50]}"
        results = memory.retrieve(
            query=query,
            limit=10,
            tiers=["stm", "mtm", "ltm"]
        )

        matches = []
        input_words = set(input_pattern.lower().split())

        for rec in results:
            metadata = rec.get("metadata", {})
            if metadata.get("kind") != "meta_lesson":
                continue

            # Filter by confusion type if specified
            if confusion_type and metadata.get("confusion_type") != confusion_type:
                continue

            content = rec.get("content", {})
            if not isinstance(content, dict):
                continue

            # Check for pattern relevance
            stored_pattern = content.get("input_pattern", "")
            stored_words = set(stored_pattern.lower().split())

            if input_words and stored_words:
                overlap = len(input_words & stored_words)
                if overlap >= 2 or stored_pattern.lower() in input_pattern.lower():
                    # Add relevance score
                    content["_relevance"] = overlap / max(len(input_words), len(stored_words))
                    matches.append(content)

        # Sort by relevance
        matches.sort(key=lambda x: x.get("_relevance", 0), reverse=True)
        return matches

    except Exception as e:
        print(f"[SELF_DMN] Meta-lesson lookup error: {e}")
        return []


def apply_meta_lesson(meta_lesson: Dict[str, Any]) -> None:
    """
    Mark a meta-lesson as applied and increment its usage count.

    This helps track which meta-lessons are effective and should be
    promoted to higher-confidence status.

    Args:
        meta_lesson: The meta-lesson that was applied
    """
    try:
        # Store updated application count
        updated = dict(meta_lesson)
        updated["application_count"] = meta_lesson.get("application_count", 0) + 1
        updated["last_applied"] = __import__("time").time()

        # If applied many times successfully, boost confidence
        if updated["application_count"] >= 3 and updated.get("confidence", 0) < 0.95:
            updated["confidence"] = min(0.95, updated.get("confidence", 0.8) + 0.05)

        store_meta_lesson(updated)

    except Exception:
        pass


def extract_meta_lessons_from_error(
    error_context: Dict[str, Any],
    insights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract meta-lessons from an error reflection.

    This is called during REFLECT_ON_ERROR to automatically create
    meta-lessons from detected issues.

    Args:
        error_context: The error context dict from REFLECT_ON_ERROR
        insights: The insights generated during reflection

    Returns:
        List of created meta-lessons
    """
    meta_lessons = []

    error_type = error_context.get("error_type", "unknown")
    query = error_context.get("query", "")
    issues = error_context.get("issues", [])
    verdict = error_context.get("verdict", "")

    # Create meta-lessons for specific issue types
    for issue in issues:
        issue_code = issue.get("code", "")
        issue_msg = issue.get("message", "")

        if issue_code == "LOW_CONFIDENCE":
            lesson = create_meta_lesson(
                confusion_type="low_confidence_pattern",
                input_pattern=query,
                misunderstanding=f"Low confidence on: {query[:100]}",
                correct_interpretation="This pattern requires more careful reasoning or memory lookup",
                source_brain="self_dmn",
                confidence=0.7
            )
            meta_lessons.append(lesson)

        elif issue_code == "INCOMPLETE":
            lesson = create_meta_lesson(
                confusion_type="incomplete_response",
                input_pattern=query,
                misunderstanding=f"Incomplete response for: {query[:100]}",
                correct_interpretation="This query type requires more complete answers - check for all aspects",
                source_brain="self_dmn",
                confidence=0.7
            )
            meta_lessons.append(lesson)

        elif issue_code == "HALLUCINATION":
            lesson = create_meta_lesson(
                confusion_type="hallucination_pattern",
                input_pattern=query,
                misunderstanding="Generated content not grounded in memory or facts",
                correct_interpretation="For this pattern, always check memory first and avoid speculation",
                source_brain="self_dmn",
                confidence=0.85
            )
            meta_lessons.append(lesson)

    # Create meta-lesson from insights
    for insight in insights:
        insight_type = insight.get("type", "")
        description = insight.get("description", "")

        if insight_type == "pattern":
            lesson = create_meta_lesson(
                confusion_type="recurring_pattern",
                input_pattern=query,
                misunderstanding=description,
                correct_interpretation=f"Pattern detected: {description} - adjust processing",
                source_brain="self_dmn",
                confidence=0.75
            )
            meta_lessons.append(lesson)

        elif insight_type == "weakness":
            lesson = create_meta_lesson(
                confusion_type="weakness_detected",
                input_pattern=query,
                misunderstanding=description,
                correct_interpretation=f"Weakness: {description} - compensate in future",
                source_brain="self_dmn",
                confidence=0.7
            )
            meta_lessons.append(lesson)

    return meta_lessons

def get_core_identity() -> Dict[str, Any]:
    """
    Load and return the canonical identity card.

    This is the ONLY source of truth for Maven's core identity.
    NEVER call Teacher to define or modify identity.

    Returns:
        Dictionary containing core identity fields:
        - name: "Maven"
        - is_llm: false
        - system_type: offline synthetic cognition system
        - creator: Josh / Hink
        - home_directory: maven2_fix
        - architectural_facts: dict of system architecture details
        - core_capabilities: list of capabilities
        - key_principles: list of key principles
    """
    try:
        with open(IDENTITY_CARD_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # Fallback to minimal identity if card cannot be loaded
        print(f"[SELF_DMN] Warning: Could not load identity card: {e}")
        return {
            "name": "Maven",
            "is_llm": False,
            "system_type": "offline synthetic cognition system",
            "creator": "Josh / Hink",
            "home_directory": "maven2_fix",
            "architectural_facts": {},
            "core_capabilities": [],
            "key_principles": []
        }

def _counts():
    from api.memory import count_lines  # type: ignore
    from api.memory import ensure_dirs  # type: ignore
    t = ensure_dirs(BRAIN_ROOT)
    return {"stm": count_lines(t["stm"]), "mtm": count_lines(t["mtm"]), "ltm": count_lines(t["ltm"]), "archive": count_lines(t.get("archive", t.get("cold", t.get("cold_storage", ""))))}

def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _analyze_reports(project_root: Path, window:int=10) -> Dict[str, Any]:
    system_dir = project_root / "reports" / "system"
    bias_dir = project_root / "reports" / "bias_audit"
    repairs_dir = project_root / "reports" / "governance" / "repairs"

    runs = sorted(system_dir.glob("run_*.json"))[-window:]
    quarantines = 0; allows = 0; denies = 0
    stores = 0
    for rp in runs:
        data = _load_json(rp)
        gov = (data.get("stage_8b_governance") or {})
        dec = (gov.get("decision") or {}).get("decision")
        if dec == "QUARANTINE":
            quarantines += 1
        elif dec == "ALLOW":
            allows += 1
            if "stage_9_storage" in data and "stored_id" in (data["stage_9_storage"] or {}):
                stores += 1
        elif dec == "DENY":
            denies += 1

    bias_files = sorted(bias_dir.glob("bias_*.jsonl"))[-window:]
    bias_samples = 0; avg_explain = 0.0; avg_verbosity = 0.0; avg_parse = 0.0
    for bp in bias_files:
        try:
            with open(bp, "r", encoding="utf-8") as f:
                for ln in f:
                    bias_samples += 1
                    try:
                        obj = json.loads(ln.strip())
                    except Exception:
                        continue
                    b = obj.get("bias") or {}
                    avg_explain += float((b.get("planner") or {}).get("explain_bias", 0.5))
                    avg_verbosity += float((b.get("language") or {}).get("verbosity_bias", 0.5))
                    avg_parse += float((b.get("reasoning") or {}).get("parse_priority", 0.5))
        except Exception:
            continue
    if bias_samples:
        avg_explain /= bias_samples; avg_verbosity /= bias_samples; avg_parse /= bias_samples

    repair_files = sorted(repairs_dir.glob("repairs_*.jsonl"))[-window:]
    repair_events = 0
    for rp in repair_files:
        try:
            with open(rp, "r", encoding="utf-8") as f:
                for _ in f:
                    repair_events += 1
        except Exception:
            continue

    # Compile the basic metrics dictionary
    metrics: Dict[str, Any] = {
        "window": window,
        "counts": {
            "runs": len(runs),
            "allows": allows,
            "denies": denies,
            "quarantines": quarantines,
            "stores": stores,
            "repairs": repair_events,
        },
        "averages": {
            "explain_bias": round(avg_explain, 3),
            "verbosity_bias": round(avg_verbosity, 3),
            "parse_priority": round(avg_parse, 3),
        },
    }
    # Compute a simple stress score based on the deny ratio.  Stress reflects the
    # proportion of denied runs and is used to trigger earlier reflections
    # or adjust thresholds in downstream consumers.  Clamp the value between
    # 0 and 1 for sanity.
    try:
        total_runs = int(metrics["counts"].get("runs", 0))
        total_denies = int(metrics["counts"].get("denies", 0))
        stress = (total_denies / total_runs) if total_runs > 0 else 0.0
    except Exception:
        stress = 0.0
    stress = max(0.0, min(1.0, float(stress)))
    metrics["stress"] = round(stress, 3)
    # Derive a basic affect vector using the affect priority brain.  Pass a
    # synthetic text containing the stress value to the affect scorer.  If the
    # affect brain is unavailable, ignore failures silently.
    try:
        import importlib
        ap_mod = importlib.import_module(
            "brains.cognitive.affect_priority.service.affect_priority_brain"
        )
        # Build a minimal text to score; include the stress value to influence
        # arousal and valence heuristics
        text = f"Stress level {stress:.3f}"
        resp = ap_mod.service_api({"op": "SCORE", "payload": {"text": text}})
        aff = (resp.get("payload") or {}) if isinstance(resp, dict) else {}
        # Only record recognised keys to avoid polluting metrics
        metrics["affect"] = {
            k: aff.get(k)
            for k in ("arousal", "valence", "priority_delta", "suggested_tone")
            if k in aff
        }
    except Exception:
        # Fallback: record an empty affect dictionary
        metrics["affect"] = {}
    return metrics

def _load_thresholds() -> Dict[str, Any]:
    """
    Load self‑DMN thresholds from config/self_dmn_thresholds.json.  If the file is missing or
    malformed, fall back to reasonable defaults.
    """
    cfg_path = Path(__file__).resolve().parents[4] / "config" / "self_dmn_thresholds.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        # Default thresholds: high deny ratio, low explain bias, high verbosity/parse weights
        return {
            "denies_ratio": 0.5,
            "min_explain_bias": 0.4,
            "max_verbosity_bias": 0.8,
            "max_parse_priority": 0.8
        }

def _check_upgrade_trigger(metrics: Dict[str, Any]) -> str | None:
    """
    Examine recent run metrics against thresholds.  Returns a reason string if the
    metrics exceed a threshold that warrants emitting an upgrade‑intent preview;
    otherwise returns None.
    """
    thresholds = _load_thresholds()
    counts = metrics.get("counts", {})
    runs = int(counts.get("runs", 0))
    denies = int(counts.get("denies", 0))
    # High deny rate?
    if runs > 0 and (denies / runs) > float(thresholds.get("denies_ratio", 0.5)):
        return f"deny ratio {denies}/{runs} exceeds threshold {thresholds.get('denies_ratio')}"
    avgs = metrics.get("averages", {})
    try:
        explain = float(avgs.get("explain_bias", 0.5))
    except Exception:
        explain = 0.5
    try:
        verb = float(avgs.get("verbosity_bias", 0.5))
    except Exception:
        verb = 0.5
    try:
        parse = float(avgs.get("parse_priority", 0.5))
    except Exception:
        parse = 0.5
    if explain < float(thresholds.get("min_explain_bias", 0.4)):
        return f"explain_bias {explain:.2f} below threshold {thresholds.get('min_explain_bias')}"
    if verb > float(thresholds.get("max_verbosity_bias", 0.8)):
        return f"verbosity_bias {verb:.2f} exceeds threshold {thresholds.get('max_verbosity_bias')}"
    if parse > float(thresholds.get("max_parse_priority", 0.8)):
        return f"parse_priority {parse:.2f} exceeds threshold {thresholds.get('max_parse_priority')}"
    return None

# ----------------------------------------------------------------------------
# Threshold recalibration
# ----------------------------------------------------------------------------

def _recalibrate_thresholds(metrics: Dict[str, Any]) -> None:
    """
    Adjust Self‑DMN threshold values based on recent run metrics.

    This helper derives new threshold values from the observed ratios of
    denies/total runs and average bias metrics.  The computed values are
    persisted to ``config/self_dmn_thresholds.json`` so that subsequent
    invocations of Self‑DMN operations use the updated thresholds.  Failures
    during recalibration are silently ignored.

    Args:
        metrics: Dictionary returned by ``_analyze_reports`` describing recent
            system runs and bias averages.
    """
    try:
        counts: Dict[str, Any] = metrics.get("counts", {}) or {}
        runs = int(counts.get("runs", 0))
        denies = int(counts.get("denies", 0))
        avgs: Dict[str, Any] = metrics.get("averages", {}) or {}
        # Compute deny ratio threshold with a small buffer
        if runs > 0:
            denies_ratio = (denies / runs) + 0.05
        else:
            denies_ratio = 0.5
        try:
            explain = float(avgs.get("explain_bias", 0.5))
        except Exception:
            explain = 0.5
        try:
            verbosity = float(avgs.get("verbosity_bias", 0.5))
        except Exception:
            verbosity = 0.5
        try:
            parse = float(avgs.get("parse_priority", 0.5))
        except Exception:
            parse = 0.5
        new_thresholds = {
            "denies_ratio": round(denies_ratio, 3),
            "min_explain_bias": round(explain * 0.9, 3),
            "max_verbosity_bias": round(verbosity * 1.1, 3),
            "max_parse_priority": round(parse * 1.1, 3),
        }

        # Persist the new thresholds using BrainMemory API
        # Classification: FACT (these are computed values with high confidence)
        classification = TruthClassifier.classify(
            content=str(new_thresholds),
            confidence=0.8,
            evidence={"source": "metrics_analysis", "metrics": metrics}
        )

        if TruthClassifier.should_store_in_memory(classification):
            memory.store(
                content=new_thresholds,
                metadata={
                    "kind": "threshold_calibration",
                    "source": "self_dmn",
                    "confidence": classification["confidence"],
                    "truth_type": classification["type"]
                }
            )
    except Exception:
        # Ignore recalibration failures
        return

def _draft_reflections(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    c = metrics.get("counts", {}); a = metrics.get("averages", {}); w = metrics.get("window", 10)
    items = []

    # Check for learned reflection patterns first
    learned_reflections = None
    if _teacher_helper and memory:
        try:
            metrics_signature = f"{c.get('allows',0)}-{c.get('quarantines',0)}-{c.get('stores',0)}"
            learned_patterns = memory.retrieve(
                query=f"reflection pattern: {metrics_signature[:30]}",
                limit=3,
                tiers=["stm", "mtm", "ltm"]
            )

            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", [])
                    if isinstance(content, list) and len(content) > 0:
                        learned_reflections = content
                        print(f"[SELF_DMN] Using learned reflection pattern from Teacher")
                        break
        except Exception:
            pass

    # Use learned reflections if found, otherwise use built-in heuristics
    if learned_reflections:
        return learned_reflections

    items.append({"content": f"Over the last {w} runs, governance allowed {c.get('allows',0)} and quarantined {c.get('quarantines',0)}.", "confidence": 0.8, "source": "system_internal"})
    items.append({"content": f"{c.get('stores',0)} facts were stored successfully.", "confidence": 0.8, "source": "system_internal"})
    items.append({"content": f"Average explain_bias {a.get('explain_bias',0.5):.2f}, verbosity_bias {a.get('verbosity_bias',0.5):.2f}, parse_priority {a.get('parse_priority',0.5):.2f}.", "confidence": 0.8, "source": "system_internal"})
    if c.get("repairs",0) > 0:
        items.append({"content": f"Repair engine executed {c['repairs']} actions in the last {w} runs.", "confidence": 0.8, "source": "system_internal"})
    return items

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    from api.memory import compute_success_average  # type: ignore
    from api.utils import success_response  # type: ignore
    from api.utils import error_response  # type: ignore
    from api.utils import generate_mid  # type: ignore
    from api.utils import CFG  # type: ignore
    """
    Central entry point for Self‑DMN operations.

    This unified service supports multiple operations related to meta‑cognition
    and dissent handling.  It delegates claim registration to the skeptic
    submodule, performs hum ticks and coherence calculations, produces
    reflections, retrains learned weights, and rescans claims for dissent.

    Supported operations:

      - HEALTH: report operational status and memory tier counts
      - REGISTER: delegate claim registration to the skeptic
      - TICK: advance hum oscillators and return synchrony and memory health
      - REFLECT: compute run metrics and draft reflections
      - DISSENT_SCAN: rescans recent claims and returns statuses
      - ANALYZE_INTERNAL: legacy introspection with retraining and upgrade preview
      - DRAFT_REFLECTIONS: produce reflection drafts from provided metrics

    Args:
        msg: A dict containing at least an 'op' key and optional 'payload'.

    Returns:
        A success or error response dict.
    """
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    # Dispatch personal brain integration operations.  When the op begins with
    # ``PERSONAL_`` the call is delegated to the optional ``self_dmn_manage``
    # module.  If the module is unavailable, return an error response.  Any
    # exceptions raised by the manage module are caught and reported.
    if op.startswith("PERSONAL_"):
        if manage_mod is None:
            return error_response(op, mid, "ERROR", "personal integration not available")
        try:
            return manage_mod.service_api(msg)
        except Exception as e:
            return error_response(op, mid, "ERROR", f"personal op failed: {e}")

    # Health check: always available
    if op == "HEALTH":
        return success_response(op, mid, {"status": "operational", "memory_health": _counts()})

    # Get core identity: return the canonical identity card
    # This operation NEVER calls Teacher and NEVER modifies identity
    if op == "GET_CORE_IDENTITY":
        identity = get_core_identity()
        return success_response(op, mid, {"identity": identity})

    # Delegated claim registration
    if op == "REGISTER":
        # Forward the entire message to the skeptic's API
        try:
            return _skeptic_service_api(msg)
        except Exception:
            return error_response(op, mid, "ERROR", "Failed to register claim")

    # Simple hum tick: advance oscillators and return coherence
    if op == "TICK":
        try:
            hum.tick()
            order = hum.coherence()
        except Exception:
            order = 0.0
        # Log the tick using BrainMemory API
        try:
            tick_data = {"op": "TICK", "hum_order": order}
            classification = TruthClassifier.classify(
                content=f"Hum coherence: {order:.3f}",
                confidence=0.8,
                evidence={"type": "hum_tick"}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=tick_data,
                    metadata={
                        "kind": "hum_tick",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass
        return success_response(op, mid, {"hum_order": order, "memory_health": _counts()})

    # Reflection operation: compute metrics and draft reflections
    if op == "REFLECT":
        # Determine project root and analysis window
        project_root = Path(__file__).resolve().parents[4]
        try:
            window = int(payload.get("window", 10))
        except Exception:
            window = 10
        metrics = _analyze_reports(project_root, window)
        # Advance hum and attach coherence
        try:
            hum.tick()
            metrics["hum_order"] = hum.coherence()
        except Exception:
            pass
        # Draft reflections based on metrics
        drafts = _draft_reflections(metrics)
        # Recalibrate thresholds based on recent metrics (best effort)
        try:
            _recalibrate_thresholds(metrics)
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Adaptive Self‑Model integration (Step 8)
        #
        # After computing run metrics and drafting reflections, invoke the
        # personal reflection and promotion operations provided by
        # ``self_dmn_manage``.  These calls introspect the personal brain
        # for emerging preferences and promote stable preferences into the
        # identity snapshot's personality.  The threshold and limit can
        # optionally be provided in the REFLECT payload.  Failures are
        # ignored to avoid disrupting the core reflection flow.
        try:
            if manage_mod is not None:
                # Extract promotion threshold and candidate limit from payload
                thr = 0.8
                lim = 10
                try:
                    thr = float(payload.get("threshold", thr))
                except Exception:
                    thr = thr
                try:
                    lim = int(payload.get("limit", lim))
                except Exception:
                    lim = lim
                # Perform personal reflection to collect preference candidates
                manage_mod.service_api({"op": "PERSONAL_REFLECT", "mid": mid, "payload": {"limit": lim}})
                # Promote stable preferences into the personality snapshot
                manage_mod.service_api({"op": "PERSONAL_PROMOTE", "mid": mid, "payload": {"threshold": thr, "limit": lim}})
        except Exception:
            # Suppress any errors during personal integration
            pass
        # Persist to memory using BrainMemory API
        try:
            reflect_data = {"op": "REFLECT", "metrics": metrics, "drafts": drafts}
            # Classify based on average confidence from drafts
            avg_confidence = sum(d.get("confidence", 0.5) for d in drafts) / max(len(drafts), 1)
            classification = TruthClassifier.classify(
                content=f"Reflection with {len(drafts)} drafts",
                confidence=avg_confidence,
                evidence={"metrics": metrics, "drafts": drafts}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=reflect_data,
                    metadata={
                        "kind": "reflection",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass
        # --- Adaptive Self‑Model Integration ---
        # When personal integration is available, reflect on the personal brain
        # and promote stable preferences into the personality snapshot.  These
        # calls are best‑effort: they may fail silently if the integration
        # modules are unavailable or raise errors.  We avoid attaching
        # intermediate results to the external response to preserve backward
        # compatibility of the REFLECT payload.
        if manage_mod is not None:
            try:
                # First, surface the top personal likes (no need to examine the result here).
                # A limit of 10 is used unless overridden via the payload.
                try:
                    limit = int(payload.get("limit", 10))
                except Exception:
                    limit = 10
                manage_mod.service_api({"op": "PERSONAL_REFLECT", "mid": mid, "payload": {"limit": limit}})
                # Next, promote candidates whose score meets the threshold.  The
                # threshold defaults to 0.8 but can be overridden via the payload.
                # Choose a lower default threshold.  Empirically the personal brain's
                # boost values are capped at 0.25, so using a threshold near 0.8 would
                # never promote any preferences.  A default of 0.2 ensures that
                # strong likes (boost ≥0.25) are promoted, while weaker likes remain
                # pending.  Developers can override this via the payload.
                try:
                    threshold = float(payload.get("threshold", 0.2))
                except Exception:
                    threshold = 0.2
                manage_mod.service_api({"op": "PERSONAL_PROMOTE", "mid": mid, "payload": {"threshold": threshold, "limit": limit}})
            except Exception:
                # Any error during personal reflection or promotion is ignored to
                # avoid disrupting the REFLECT call.  The underlying modules
                # will log their own errors where appropriate.
                pass
        return success_response(op, mid, {"metrics": metrics, "drafts": drafts})

    # Dissent scanning: rescore recent claims
    if op == "DISSENT_SCAN":
        # How many recent claims to scan
        try:
            window = int(payload.get("window", 10))
        except Exception:
            window = 10
        results: List[Dict[str, Any]] = []
        flagged: List[Dict[str, Any]] = []  # claims requiring recompute/disputed
        # Retrieve existing claims from BrainMemory
        try:
            # Read all existing claim records from memory
            existing_claims = memory.retrieve(limit=None)  # Get all claims
            claim_records = [rec for rec in existing_claims if rec.get("kind") == "claim"]

            new_records: List[Dict[str, Any]] = []
            # Rescore every claim and update its status.
            MAX_RECOMPUTE = 3
            for rec in claim_records:
                try:
                    # Extract content from memory record
                    claim_content = rec.get("content", {})
                    if not isinstance(claim_content, dict):
                        new_records.append(rec)
                        continue

                    consensus = float(claim_content.get("consensus_score", 0.0))
                except Exception:
                    consensus = 0.0
                try:
                    skeptic = float(claim_content.get("skeptic_score", 0.0))
                except Exception:
                    skeptic = 0.0
                # Recompute status using thresholds defined in this module
                if (skeptic - consensus) >= TAU1:
                    rec_status = "recompute"
                elif skeptic >= TAU2:
                    rec_status = "disputed"
                else:
                    rec_status = "undisputed"
                claim_content["status"] = rec_status
                # Ensure recompute metadata fields exist
                rc = claim_content.get("recompute_count")
                if not isinstance(rc, int):
                    rc = 0
                # Determine whether this claim should be flagged
                should_flag = False
                if rec_status == "recompute":
                    # Check recompute throttle (count-based only, no time window)
                    if rc < MAX_RECOMPUTE:
                        should_flag = True
                        # Increment recompute counters for flagged claim
                        claim_content["recompute_count"] = rc + 1
                elif rec_status == "disputed":
                    should_flag = True
                if should_flag:
                    flagged.append(claim_content)

                # Update the claim in memory
                claim_confidence = 1.0 - skeptic  # Higher skepticism = lower confidence
                classification = TruthClassifier.classify(
                    content=str(claim_content.get("proposition", "")),
                    confidence=max(0.0, min(1.0, claim_confidence)),
                    evidence={"consensus": consensus, "skeptic": skeptic}
                )

                if TruthClassifier.should_store_in_memory(classification):
                    memory.store(
                        content=claim_content,
                        metadata={
                            "kind": "claim",
                            "source": "self_dmn",
                            "confidence": classification["confidence"],
                            "truth_type": classification["type"],
                            "status": rec_status
                        }
                    )
                new_records.append(claim_content)
                # Build results from the last 'window' updated records
                for rec in new_records[-window:]:
                    if isinstance(rec, dict):
                        cid = rec.get("claim_id")
                        status = rec.get("status") or "unknown"
                        results.append({"claim_id": cid, "status": status})
        except Exception:
            # ignore any file or parsing errors
            results = []
        # After rescoring, persist audit records using BrainMemory API
        try:
            # If there are any flagged claims from the rescoring loop, store
            # audit entries for each using BrainMemory.
            if flagged:
                for rec in flagged:
                    try:
                        audit_entry = {
                            "claim_id": rec.get("claim_id"),
                            "proposition": rec.get("proposition"),
                            "status": rec.get("status"),
                            "consensus_score": rec.get("consensus_score"),
                            "skeptic_score": rec.get("skeptic_score"),
                            "expiry": rec.get("expiry"),
                        }

                        # Classify audit entry - disputed/recompute = high confidence concern
                        audit_confidence = 0.9 if rec.get("status") == "disputed" else 0.7
                        classification = TruthClassifier.classify(
                            content=str(audit_entry),
                            confidence=audit_confidence,
                            evidence={"flagged_claim": rec}
                        )

                        if TruthClassifier.should_store_in_memory(classification):
                            memory.store(
                                content=audit_entry,
                                metadata={
                                    "kind": "audit",
                                    "source": "self_dmn",
                                    "confidence": classification["confidence"],
                                    "truth_type": classification["type"]
                                }
                            )
                    except Exception:
                        continue
        except Exception:
            pass
        # Log the scan operation using BrainMemory API
        try:
            scan_data = {"op": "DISSENT_SCAN", "results": results}
            classification = TruthClassifier.classify(
                content=f"Dissent scan: {len(results)} claims, {len(flagged)} flagged",
                confidence=0.8,
                evidence={"results": results, "flagged_count": len(flagged)}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=scan_data,
                    metadata={
                        "kind": "dissent_scan",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass
        # Return both the list of recent claims and any flagged claim IDs for downstream consumers
        try:
            flagged_ids = [rec.get("claim_id") for rec in flagged if isinstance(rec, dict)]
        except Exception:
            flagged_ids = []
        # Invoke the Self-DMN judge to produce recomputation directives for flagged claims.
        decisions = []
        try:
            import importlib
            judge_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_judge")
            jres = judge_mod.service_api({"op": "ADJUDICATE", "payload": {"claims": flagged}})
            decisions = (jres.get("payload") or {}).get("decisions") or []
        except Exception:
            decisions = []
        return success_response(op, mid, {"claims": results, "flagged": flagged_ids, "decisions": decisions})

    # Internal analysis: maintain existing behaviour for backwards compatibility
    if op == "ANALYZE_INTERNAL":
        project_root = Path(__file__).resolve().parents[4]
        metrics = _analyze_reports(project_root, int(payload.get("window", 10)))
        # Advance hum and attach coherence
        try:
            hum.tick()
            metrics["hum_order"] = hum.coherence()
        except Exception:
            pass
        # Persist metrics using BrainMemory API
        try:
            analysis_data = {"op": "ANALYZE_INTERNAL", "metrics": metrics}
            classification = TruthClassifier.classify(
                content=f"Internal analysis: stress={metrics.get('stress', 0.0):.3f}",
                confidence=0.8,
                evidence={"metrics": metrics}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=analysis_data,
                    metadata={
                        "kind": "internal_analysis",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass
        # Lightweight retraining of learned weights, scaled by coherence
        try:
            lr_base = 0.2
            try:
                order_val = float(metrics.get("hum_order", 0.0) or 0.0)
            except Exception:
                order_val = 0.0
            lr_eff = lr_base * order_val
            if lr_eff:
                brains = [
                    "sensorium",
                    "planner",
                    "language",
                    "pattern_recognition",
                    "reasoning",
                    "affect_priority",
                    "personality",
                ]
                cognitive_root = BRAIN_ROOT.parent
                for b in brains:
                    try:
                        root = cognitive_root / b
                        avg = compute_success_average(root)
                        # Read weights using BrainMemory API instead of direct file access
                        try:
                            brain_memory = BrainMemory(b)
                            results = brain_memory.retrieve(query="kind:weights", limit=1)
                            if results:
                                w = results[0].get("content", {})
                            else:
                                w = CFG.get("weights_defaults", {}) or {}
                        except Exception:
                            w = CFG.get("weights_defaults", {}) or {}
                        updated = {}
                        for k, v in w.items():
                            try:
                                orig = float(v)
                            except Exception:
                                updated[k] = v
                                continue
                            updated[k] = orig * (1.0 - lr_eff) + avg * lr_eff

                        # Store updated weights using BrainMemory API
                        try:
                            classification = TruthClassifier.classify(
                                content=str(updated),
                                confidence=0.8,
                                evidence={"brain": b, "avg_success": avg, "lr_eff": lr_eff}
                            )

                            if TruthClassifier.should_store_in_memory(classification):
                                # Store to target brain's memory
                                brain_memory.store(
                                    content=updated,
                                    metadata={
                                        "kind": "weights",
                                        "source": "self_dmn",
                                        "confidence": classification["confidence"],
                                        "truth_type": classification["type"]
                                    }
                                )
                        except Exception:
                            pass
                    except Exception:
                        continue
        except Exception:
            pass
        # Emit upgrade preview if thresholds exceeded
        reason = _check_upgrade_trigger(metrics)
        if reason:
            try:
                preview_data = {"metrics": metrics, "reason": reason}
                classification = TruthClassifier.classify(
                    content=f"Upgrade trigger: {reason}",
                    confidence=0.9,
                    evidence={"metrics": metrics, "reason": reason}
                )

                if TruthClassifier.should_store_in_memory(classification):
                    memory.store(
                        content=preview_data,
                        metadata={
                            "kind": "upgrade_preview",
                            "source": "self_dmn",
                            "confidence": classification["confidence"],
                            "truth_type": classification["type"],
                            "reason": reason
                        }
                    )
            except Exception:
                pass
        # Recalibrate thresholds based on recent metrics (best effort)
        try:
            _recalibrate_thresholds(metrics)
        except Exception:
            pass
        return success_response(op, mid, {"metrics": metrics})

    # Draft reflections for given metrics
    if op == "DRAFT_REFLECTIONS":
        metrics = payload.get("metrics") or {}
        drafts = _draft_reflections(metrics)
        try:
            draft_data = {"op": "DRAFT_REFLECTIONS", "drafts": drafts}
            avg_confidence = sum(d.get("confidence", 0.5) for d in drafts) / max(len(drafts), 1)
            classification = TruthClassifier.classify(
                content=f"Draft reflections: {len(drafts)} items",
                confidence=avg_confidence,
                evidence={"metrics": metrics, "drafts": drafts}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=draft_data,
                    metadata={
                        "kind": "draft_reflections",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass
        return success_response(op, mid, {"drafts": drafts})

    # RUN_IDLE_CYCLE: Background reflection when system is idle
    if op == "RUN_IDLE_CYCLE":
        # Get conversation context for continuation detection
        conv_context = {}
        is_follow_up = False
        if _continuation_helpers_available:
            try:
                conv_context = get_conversation_context()
                query = payload.get("query", "")
                is_follow_up = is_continuation(query, payload)
            except Exception:
                pass

        system_history = payload.get("system_history", [])
        recent_issues = payload.get("recent_issues", [])
        motivation_state = payload.get("motivation_state", {})

        insights: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []

        if len(recent_issues) > 3:
            insights.append({
                "type": "weakness",
                "description": f"Observed {len(recent_issues)} issues in recent runs"
            })

        error_count = sum(1 for issue in recent_issues if issue.get("severity") == "major")
        if error_count > 0:
            insights.append({
                "type": "pattern",
                "description": f"Pattern of {error_count} major errors detected"
            })
            actions.append({
                "kind": "adjust_motivation",
                "delta": {"truthfulness": 0.05, "self_improvement": 0.05}
            })

        if len(system_history) > 10:
            recent_denies = sum(1 for h in system_history[-10:] if h.get("decision") == "DENY")
            if recent_denies > 5:
                insights.append({
                    "type": "weakness",
                    "description": f"High denial rate: {recent_denies}/10 recent queries denied"
                })
                actions.append({
                    "kind": "schedule_learning_task",
                    "task": {"type": "analyze_denial_patterns", "priority": 0.8}
                })

        try:
            idle_data = {"op": "RUN_IDLE_CYCLE", "insights": insights, "actions": actions, "is_continuation": is_follow_up}
            # Classify based on insight severity
            avg_confidence = 0.8 if len(insights) > 0 else 0.5
            classification = TruthClassifier.classify(
                content=f"Idle cycle: {len(insights)} insights, {len(actions)} actions",
                confidence=avg_confidence,
                evidence={"insights": insights, "actions": actions}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=idle_data,
                    metadata={
                        "kind": "idle_cycle",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass

        result_payload = {"insights": insights, "actions": actions, "is_continuation": is_follow_up}

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="self_dmn",
                    action="introspective_continuation" if is_follow_up else "idle_reflection",
                    confidence=avg_confidence,
                    context_tags=["introspection", "continuation"] if is_follow_up else ["introspection"]
                )
            except Exception:
                pass

        return success_response(op, mid, result_payload)

    # REFLECT_ON_ERROR: Analyze a specific error and suggest adjustments
    if op == "REFLECT_ON_ERROR":
        # Get conversation context for continuation detection
        conv_context = {}
        is_follow_up = False
        if _continuation_helpers_available:
            try:
                conv_context = get_conversation_context()
                query = payload.get("query", "")
                is_follow_up = is_continuation(query, payload)
            except Exception:
                pass

        error_context = payload.get("error_context", {})
        turn_history = payload.get("turn_history", [])

        insights: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []

        error_type = error_context.get("error_type", "unknown")
        verdict = error_context.get("verdict", "")

        if verdict == "major_issue":
            insights.append({
                "type": "weakness",
                "description": f"Major issue detected: {error_type}"
            })

            issues = error_context.get("issues", [])
            low_conf_issues = [i for i in issues if i.get("code") == "LOW_CONFIDENCE"]
            incomplete_issues = [i for i in issues if i.get("code") == "INCOMPLETE"]

            if low_conf_issues:
                actions.append({
                    "kind": "adjust_motivation",
                    "delta": {"truthfulness": 0.1, "curiosity": -0.05}
                })

            if incomplete_issues:
                actions.append({
                    "kind": "schedule_learning_task",
                    "task": {
                        "type": "improve_completeness",
                        "priority": 0.9,
                        "target": error_context.get("query", "")
                    }
                })

            if len(turn_history) > 5:
                recent_errors = sum(1 for t in turn_history[-5:] if t.get("had_error", False))
                if recent_errors >= 3:
                    insights.append({
                        "type": "pattern",
                        "description": f"Recurring error pattern: {recent_errors}/5 recent turns had errors"
                    })
                    actions.append({
                        "kind": "adjust_motivation",
                        "delta": {"self_improvement": 0.15}
                    })

        # =================================================================
        # STEP 2 ENHANCEMENT: Extract and Store Meta-Lessons
        # =================================================================
        # When confusion/error is detected, create meta-lessons to prevent
        # the same misunderstanding from recurring.
        meta_lessons_created = []
        try:
            meta_lessons = extract_meta_lessons_from_error(error_context, insights)
            for lesson in meta_lessons:
                if store_meta_lesson(lesson):
                    meta_lessons_created.append(lesson.get("confusion_type", "unknown"))

            if meta_lessons_created:
                print(f"[SELF_DMN] ✓ Created {len(meta_lessons_created)} meta-lessons for self-healing")
                actions.append({
                    "kind": "meta_lessons_stored",
                    "count": len(meta_lessons_created),
                    "types": meta_lessons_created
                })
        except Exception as e:
            print(f"[SELF_DMN] Meta-lesson extraction error: {e}")

        try:
            error_data = {"op": "REFLECT_ON_ERROR", "insights": insights, "actions": actions, "is_continuation": is_follow_up, "meta_lessons_created": meta_lessons_created}
            # Classify based on error severity
            error_confidence = 0.9 if verdict == "major_issue" else 0.7
            classification = TruthClassifier.classify(
                content=f"Error reflection: {error_type}, verdict={verdict}",
                confidence=error_confidence,
                evidence={"error_context": error_context, "insights": insights}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=error_data,
                    metadata={
                        "kind": "error_reflection",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"],
                        "error_type": error_type
                    }
                )
        except Exception:
            pass

        result_payload = {"insights": insights, "actions": actions, "is_continuation": is_follow_up, "meta_lessons_created": meta_lessons_created}

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="self_dmn",
                    action="error_analysis_continuation" if is_follow_up else "error_reflection",
                    confidence=error_confidence,
                    context_tags=["error_analysis", "continuation"] if is_follow_up else ["error_analysis"]
                )
            except Exception:
                pass

        return success_response(op, mid, result_payload)

    # RUN_LONG_TERM_REFLECTION: Phase 5 long-term self-improvement cycle
    if op == "RUN_LONG_TERM_REFLECTION":
        """
        Long-term reflection triggered when:
        - System idle ≥ N turns (no time-based logic — N = deterministic constant)
        - Tier imbalance detected
        - Concept drift detected (patterns vs facts)

        Deterministic triggers only:
        - counts
        - seq_ids
        - importance
        - use_count
        No wall-clock time.
        """
        # Get conversation context for continuation detection
        conv_context = {}
        is_follow_up = False
        if _continuation_helpers_available:
            try:
                conv_context = get_conversation_context()
                query = payload.get("query", "")
                is_follow_up = is_continuation(query, payload)
            except Exception:
                pass

        tier_stats = payload.get("tier_stats", {})
        pattern_count = payload.get("pattern_count", 0)
        fact_count = payload.get("fact_count", 0)
        idle_turns = payload.get("idle_turns", 0)

        # Deterministic thresholds
        TIER_WM_THRESHOLD = 100
        TIER_IMBALANCE_RATIO = 3.0
        IDLE_TURN_THRESHOLD = 10
        CONCEPT_DRIFT_THRESHOLD = 20

        insights: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []

        # Check TIER_WM overflow
        wm_count = tier_stats.get("WM", {}).get("count", 0)
        if wm_count > TIER_WM_THRESHOLD:
            insights.append({
                "type": "tier_overflow",
                "description": f"WM tier has {wm_count} records (threshold: {TIER_WM_THRESHOLD})"
            })
            actions.append({
                "kind": "demote_wm_to_short",
                "target_tier": "WM",
                "target_count": wm_count - TIER_WM_THRESHOLD
            })

        # Check tier imbalance
        mid_count = tier_stats.get("MID", {}).get("count", 0)
        short_count = tier_stats.get("SHORT", {}).get("count", 0)
        if short_count > 0 and mid_count / short_count > TIER_IMBALANCE_RATIO:
            insights.append({
                "type": "tier_imbalance",
                "description": f"MID/SHORT ratio is {mid_count / short_count:.2f} (threshold: {TIER_IMBALANCE_RATIO})"
            })
            actions.append({
                "kind": "rebalance_tiers",
                "promote_from": "SHORT",
                "demote_from": "MID"
            })

        # Check concept drift
        if pattern_count > 0 and fact_count > 0:
            drift_ratio = abs(pattern_count - fact_count)
            if drift_ratio > CONCEPT_DRIFT_THRESHOLD:
                insights.append({
                    "type": "concept_drift",
                    "description": f"Pattern/fact imbalance: {pattern_count} patterns vs {fact_count} facts"
                })
                if pattern_count > fact_count:
                    actions.append({
                        "kind": "create_concepts_from_patterns",
                        "pattern_count": pattern_count - fact_count
                    })
                else:
                    actions.append({
                        "kind": "extract_patterns_from_facts",
                        "fact_count": fact_count - pattern_count
                    })

        # Check idle turns
        if idle_turns >= IDLE_TURN_THRESHOLD:
            insights.append({
                "type": "idle_detected",
                "description": f"System idle for {idle_turns} turns (threshold: {IDLE_TURN_THRESHOLD})"
            })
            actions.append({
                "kind": "consolidate_memories",
                "priority": 0.7
            })

        # Concept importance scaling
        concept_count = tier_stats.get("LONG", {}).get("count", 0)
        if concept_count > 50:
            insights.append({
                "type": "concept_growth",
                "description": f"LONG tier has {concept_count} concepts, consider importance scaling"
            })
            actions.append({
                "kind": "scale_concept_importance",
                "scaling_factor": 0.95  # Slight decay
            })

        # Preference consolidation trigger
        preference_count = payload.get("preference_count", 0)
        if preference_count >= 5:
            insights.append({
                "type": "preference_consolidation_ready",
                "description": f"{preference_count} preferences ready for consolidation"
            })
            actions.append({
                "kind": "consolidate_preferences",
                "count": preference_count
            })

        # Skill detection trigger
        query_history_len = payload.get("query_history_len", 0)
        if query_history_len >= 10:
            insights.append({
                "type": "skill_detection_ready",
                "description": f"Query history has {query_history_len} entries, ready for skill detection"
            })
            actions.append({
                "kind": "detect_skills",
                "query_count": query_history_len
            })

        try:
            long_term_data = {"op": "RUN_LONG_TERM_REFLECTION", "insights": insights, "actions": actions, "is_continuation": is_follow_up}
            # Classify based on action count and insight severity
            avg_confidence = 0.85 if len(actions) > 0 else 0.6
            classification = TruthClassifier.classify(
                content=f"Long-term reflection: {len(insights)} insights, {len(actions)} actions",
                confidence=avg_confidence,
                evidence={"tier_stats": tier_stats, "insights": insights, "actions": actions}
            )

            if TruthClassifier.should_store_in_memory(classification):
                memory.store(
                    content=long_term_data,
                    metadata={
                        "kind": "long_term_reflection",
                        "source": "self_dmn",
                        "confidence": classification["confidence"],
                        "truth_type": classification["type"]
                    }
                )
        except Exception:
            pass

        result_payload = {"insights": insights, "actions": actions, "action_count": len(actions), "is_continuation": is_follow_up}

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="self_dmn",
                    action="long_term_continuation" if is_follow_up else "long_term_reflection",
                    confidence=avg_confidence,
                    context_tags=["long_term", "self_improvement", "continuation"] if is_follow_up else ["long_term", "self_improvement"]
                )
            except Exception:
                pass

        return success_response(op, mid, result_payload)

    # Unsupported operations
    return error_response(op, mid, "UNSUPPORTED_OP", op)

# Standard service contract: handle is the entry point
service_api = handle
