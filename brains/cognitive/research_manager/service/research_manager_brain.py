from __future__ import annotations
from typing import Dict, Any, List, Optional
import importlib.util
import os
import re
import time
import uuid
from pathlib import Path

from brains.memory.brain_memory import BrainMemory
from brains.cognitive.reasoning.truth_classifier import TruthClassifier
from brains.cognitive.self_review.service.self_review_brain import run_reflection_engine
from api.utils import CFG
from config.web_config import WEB_RESEARCH_CONFIG
from web_client import WebDocument, search_web

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[RESEARCH_MANAGER] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Import answer style for detail level control
try:
    from brains.cognitive.language.answer_style import (
        DetailLevel,
        infer_detail_level,
        get_web_search_synthesis_instruction,
    )
    _answer_style_available = True
except Exception as e:
    print(f"[RESEARCH_MANAGER] Answer style not available: {e}")
    _answer_style_available = False
    # Provide fallback
    class DetailLevel:  # type: ignore
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"
    def infer_detail_level(question, context=None):  # type: ignore
        return DetailLevel.MEDIUM
    def get_web_search_synthesis_instruction(detail_level, is_followup=False):  # type: ignore
        return "- Synthesize the information into a clear, helpful answer"

# Pattern store for unified learning across all brains
from brains.cognitive.pattern_store import (
    get_pattern_store,
    Pattern,
    verdict_to_reward
)
from brains.cognitive.research_manager.initial_patterns import (
    initialize_research_manager_patterns
)
from brains.cognitive.self_dmn import self_critique_v2

# Research Pattern Learning System for pattern-based research planning
try:
    from brains.cognitive.research_manager.research_patterns import (
        ResearchPatternManager,
        record_research_task,
        suggest_research_plan,
        mark_plan_result,
        list_research_patterns,
        get_default_research_pattern_manager,
    )
    _research_pattern_mgr = get_default_research_pattern_manager()
    _research_patterns_available = True
except Exception as e:
    print(f"[RESEARCH_MANAGER] Research patterns not available: {e}")
    _research_pattern_mgr = None  # type: ignore
    _research_patterns_available = False

# Fact Verification System for verifying research findings
try:
    from brains.cognitive.research_manager.fact_verification import (
        verify_answer,
        VerificationResult,
    )
    _fact_verification_available = True
except Exception as e:
    print(f"[RESEARCH_MANAGER] Fact verification not available: {e}")
    _fact_verification_available = False

# Teacher helper is optional; the brain still functions offline
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("research_manager")
except Exception as e:  # pragma: no cover - optional dependency
    print(f"[RESEARCH_MANAGER] Teacher helper unavailable: {e}")
    _teacher_helper = None  # type: ignore

_MEM = BrainMemory("research_manager")
_JOBS_MEM = BrainMemory("research_jobs")  # Separate memory for job tracking

# Initialize pattern store
_pattern_store = get_pattern_store()

# Load initial patterns if not already present
try:
    initialize_research_manager_patterns()
except Exception as e:
    print(f"[RESEARCH_MANAGER] Failed to initialize patterns: {e}")

# Track which pattern was used for the current research
_current_pattern: Optional[Pattern] = None

HERE = Path(__file__).resolve()
# HERE is .../maven2_fix/brains/cognitive/research_manager/service/research_manager_brain.py
# parents[0] = service/, parents[1] = research_manager/, parents[2] = cognitive/, parents[3] = brains/, parents[4] = maven2_fix/
MAVEN_ROOT = HERE.parents[4]


def _load_service(rel_path_from_root: Path):
    try:
        spec = importlib.util.spec_from_file_location(rel_path_from_root.stem, str(rel_path_from_root))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        return None
    return None


def _bank_module(name: str):
    svc = MAVEN_ROOT / "brains" / "domain_banks" / name / "service" / f"{name}_bank.py"
    if not svc.exists():
        return None
    return _load_service(svc)


def _call_memory_librarian(payload: dict) -> dict:
    try:
        svc = MAVEN_ROOT / "brains" / "cognitive" / "memory_librarian" / "service" / "memory_librarian.py"
        mod = _load_service(svc)
        if mod and hasattr(mod, "service_api"):
            return mod.service_api(payload)
    except Exception:
        pass
    return {"ok": False, "error": "memory_librarian_unavailable"}


def _store_fact_record(content: str, confidence: float, source: str, topic: str, url: str | None = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Store a fact record in the appropriate domain bank. Returns True if stored successfully."""
    fact = self_critique_v2.Fact(
        content=content,
        source=source,
        domain="factual",
        meta=metadata or {},
    )

    decision = self_critique_v2.process_fact(fact)
    print(
        f"[FACT_DECISION] source={source} domain=factual decision={decision.decision} "
        f"conf={decision.confidence:.2f} reason={decision.reason}"
    )

    if decision.reprimand_teacher:
        print("[TEACHER_REPRIMAND] Logged reprimand for unreliable fact")

    if decision.decision != "accept":
        return False

    classification = TruthClassifier.classify(content, confidence, evidence={"source": source})
    if not TruthClassifier.should_store_in_memory(classification):
        print(f"[RESEARCH] Skipping storage (classification rejected): {content[:60]}...")
        return False
    bank_hint = classification.get("memory_tier") or "factual"
    # Map tier hint to actual bank names
    bank = "factual"
    if bank_hint == "working_theories":
        bank = "working_theories"
    elif bank_hint == "stm_only":
        bank = "stm_only"
    mod = _bank_module(bank)
    if not mod or not hasattr(mod, "service_api"):
        print(f"[RESEARCH] Warning: Bank '{bank}' module not available")
        return False
    try:
        meta = {"topic": topic, "url": url or "", "classification": classification.get("classification", "UNKNOWN")}
        if metadata:
            meta.update(metadata)

        fact_payload = {
            "content": content,
            "confidence": confidence,
            "source": source,
            "metadata": meta
        }
        result = mod.service_api({"op": "STORE", "payload": {"fact": fact_payload}})
        if result.get("ok"):
            print(f"[RESEARCH] Stored fact to '{bank}': {content[:60]}...")
            return True
        else:
            print(f"[RESEARCH] Failed to store to '{bank}': {result.get('error', 'unknown')}")
            return False
    except Exception as e:
        print(f"[RESEARCH] Error storing fact: {e}")
        return False


def _web_config_value(key: str, default):
    try:
        if isinstance(WEB_RESEARCH_CONFIG, dict):
            return WEB_RESEARCH_CONFIG.get(key, default)
    except Exception:
        pass
    try:
        return getattr(WEB_RESEARCH_CONFIG, key, default)
    except Exception:
        return default


def _is_web_research_enabled() -> bool:
    config_default = bool(getattr(WEB_RESEARCH_CONFIG, "enabled", False))
    try:
        env_override = os.getenv("MAVEN_ENABLE_WEB_RESEARCH")
        if env_override is not None:
            return env_override.strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass

    try:
        legacy_cfg = bool((CFG.get("web_research") or {}).get("enabled", False))
    except Exception:
        legacy_cfg = False

    try:
        return bool(CFG.get("ENABLE_WEB_RESEARCH", False) or legacy_cfg or config_default)
    except Exception:
        return config_default


def _web_deadline_seconds() -> int:
    config_deadline = int(getattr(WEB_RESEARCH_CONFIG, "max_seconds", 1200))
    try:
        env_override = os.getenv("MAVEN_WEB_RESEARCH_MAX_SECONDS")
        if env_override:
            return max(30, int(env_override))
    except Exception:
        pass
    try:
        return int(CFG.get("WEB_RESEARCH_MAX_SECONDS", config_deadline))
    except Exception:
        return config_deadline


def _max_web_requests_default(depth: int) -> int:
    config_max = int(getattr(WEB_RESEARCH_CONFIG, "max_requests", max(2, depth + 1)))
    try:
        env_override = os.getenv("MAVEN_WEB_RESEARCH_MAX_REQUESTS")
        if env_override:
            return max(1, int(env_override))
    except Exception:
        pass

    try:
        return max(1, int(CFG.get("WEB_RESEARCH_MAX_REQUESTS", config_max)))
    except Exception:
        return max(config_max, 1)


def _deadline_exceeded(deadline: float | None) -> bool:
    if deadline is None:
        return False
    if time.monotonic() >= deadline:
        print("[RESEARCH_WEB] Deadline exceeded — skipping web fetch")
        return True
    return False


def _extract_time_budget_seconds(text: str) -> Optional[int]:
    try:
        match = re.search(r"(\d{1,3})\s*(minute|minutes|min)", text.lower())
        if match:
            return max(1, int(match.group(1))) * 60
    except Exception:
        return None
    return None


def _parse_web_hints(text: str, default_enabled: bool) -> Dict[str, Any]:
    """Parse inline web hints while preserving explicit intent."""

    web_enabled_explicit: Optional[bool] = None
    time_budget_seconds_explicit: Optional[int] = None
    max_web_requests_explicit: Optional[int] = None
    web_enabled = bool(default_enabled)
    max_deadline = _web_deadline_seconds()

    parts: List[str] = []
    for token in str(text).split():
        cleaned = token.strip().strip("\"'“”‘’").rstrip(",")
        lower_cleaned = cleaned.lower()

        if lower_cleaned.startswith("webmax:"):
            try:
                value = int(lower_cleaned.split(":", 1)[1].strip())
                if value > 0:
                    max_web_requests_explicit = value
            except Exception:
                pass
            continue

        if not lower_cleaned.startswith("web:"):
            parts.append(token)
            continue

        hint_value = cleaned.split(":", 1)[1].strip().lower() if ":" in cleaned else ""
        if hint_value.isdigit():
            try:
                time_budget_seconds_explicit = max(1, min(max_deadline, int(hint_value)))
                web_enabled_explicit = True
                web_enabled = True
            except Exception:
                pass
        elif hint_value in {"true", "on", "yes"}:
            web_enabled_explicit = True
            web_enabled = True
        elif hint_value in {"false", "off", "no"}:
            web_enabled_explicit = False
            web_enabled = False
        else:
            parts.append(token)

    cleaned_topic = " ".join(parts).strip().strip("\"'")
    return {
        "topic": cleaned_topic,
        "web_enabled": web_enabled,
        "web_enabled_explicit": web_enabled_explicit,
        "time_budget_seconds": time_budget_seconds_explicit,
        "time_budget_seconds_explicit": time_budget_seconds_explicit,
        "max_web_requests_explicit": max_web_requests_explicit,
    }


def _resolve_web_budget(payload: Dict[str, Any], depth: int) -> Dict[str, Any]:
    max_deadline = _web_deadline_seconds()
    explicit = payload.get("time_budget_seconds") or payload.get("web_time_budget_seconds")
    if explicit is None:
        explicit = _extract_time_budget_seconds(str(payload.get("full_prompt") or payload.get("topic") or ""))
    if explicit is None:
        # Use a smaller default for quick research, larger for deeper tasks
        if str(payload.get("mode") or payload.get("deliverable") or "").lower().startswith("deep") or depth > 2:
            explicit = min(max_deadline, 1200)
        else:
            explicit = min(max_deadline, 300)

    try:
        budget_seconds = max(30, min(max_deadline, int(explicit)))
    except Exception:
        budget_seconds = max_deadline

    max_requests = payload.get("max_requests") or payload.get("web_max_requests")
    try:
        max_requests_int = max(1, int(max_requests)) if max_requests is not None else _max_web_requests_default(depth)
    except Exception:
        max_requests_int = _max_web_requests_default(depth)

    return {
        "time_budget_seconds": budget_seconds,
        "max_requests": max_requests_int,
        "start_time": time.monotonic(),
        "requests_made": 0,
        "time_budget_used": 0.0,
    }


def _budget_exhausted(budget: Dict[str, Any]) -> bool:
    time_budget = float(budget.get("time_budget_seconds") or 0)
    start_time = float(budget.get("start_time") or time.monotonic())
    time_used = float(budget.get("time_budget_used") or 0.0)
    if time_budget and ((time.monotonic() - start_time) > time_budget or time_used >= time_budget):
        return True
    max_requests = int(budget.get("max_requests") or 0)
    requests_made = int(budget.get("requests_made") or 0)
    return max_requests > 0 and requests_made >= max_requests


# ============================================================================
# Research Job Object Management
# ============================================================================

def create_research_job(
    topic: str,
    full_prompt: str,
    owner: str = "user",
    web_enabled: bool = False,
    deadline: float | None = None,
) -> str:
    """Create a new research job and return its ID."""
    job_id = f"research-{int(time.time())}-{str(uuid.uuid4())[:8]}"

    job = {
        "job_id": job_id,
        "topic": topic,
        "full_prompt": full_prompt,
        "status": "pending",
        "subquestions": [],
        "sources": [],
        "facts_stored": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
        "owner": owner,
        "web": web_enabled,
        "web_enabled": web_enabled,
        "time_budget_seconds": None,
        "time_budget_used": 0.0,
        "max_web_requests": None,
        "web_requests_used": 0,
    }

    if web_enabled and _is_web_research_enabled():
        job["deadline"] = deadline or (time.monotonic() + _web_deadline_seconds())

    _JOBS_MEM.store(
        content=job,
        metadata={"kind": "research_job", "job_id": job_id, "topic": topic}
    )

    print(f"[RESEARCH_MODE] New job id={job_id} topic=\"{topic}\"")
    return job_id


def update_research_job(job_id: str, patch_dict: Dict[str, Any]) -> None:
    """Update a research job with new fields."""
    try:
        # Retrieve existing job
        results = _JOBS_MEM.retrieve(query=job_id, limit=1)
        if not results:
            print(f"[RESEARCH] Warning: Job {job_id} not found for update")
            return

        job = results[0].get("content", {})
        if isinstance(job, str):
            print(f"[RESEARCH] Warning: Job stored as string, cannot update")
            return

        # Apply patches
        job.update(patch_dict)
        job["updated_at"] = time.time()

        # Re-store
        _JOBS_MEM.store(
            content=job,
            metadata={"kind": "research_job", "job_id": job_id, "topic": job.get("topic", "")}
        )

        print(f"[RESEARCH] Updated job {job_id} with {list(patch_dict.keys())}")

    except Exception as e:
        print(f"[RESEARCH] Error updating job {job_id}: {e}")


def append_subquestion(job_id: str, question: str, status: str = "pending") -> None:
    """Add a sub-question to a research job."""
    try:
        results = _JOBS_MEM.retrieve(query=job_id, limit=1)
        if not results:
            return

        job = results[0].get("content", {})
        if isinstance(job, str):
            return

        if "subquestions" not in job:
            job["subquestions"] = []

        job["subquestions"].append({
            "q": question,
            "status": status,
            "notes": ""
        })

        job["updated_at"] = time.time()

        _JOBS_MEM.store(
            content=job,
            metadata={"kind": "research_job", "job_id": job_id, "topic": job.get("topic", "")}
        )

    except Exception as e:
        print(f"[RESEARCH] Error appending subquestion: {e}")


def append_source(job_id: str, title: str, url: str, trust_score: float = 0.5) -> None:
    """Add a source to a research job."""
    try:
        results = _JOBS_MEM.retrieve(query=job_id, limit=1)
        if not results:
            return

        job = results[0].get("content", {})
        if isinstance(job, str):
            return

        if "sources" not in job:
            job["sources"] = []

        job["sources"].append({
            "title": title,
            "url": url,
            "trust_score": trust_score
        })

        job["updated_at"] = time.time()

        _JOBS_MEM.store(
            content=job,
            metadata={"kind": "research_job", "job_id": job_id, "topic": job.get("topic", "")}
        )

    except Exception as e:
        print(f"[RESEARCH] Error appending source: {e}")


def increment_facts_count(job_id: str) -> None:
    """Increment the facts stored counter for a job."""
    try:
        results = _JOBS_MEM.retrieve(query=job_id, limit=1)
        if not results:
            return

        job = results[0].get("content", {})
        if isinstance(job, str):
            return

        job["facts_stored"] = job.get("facts_stored", 0) + 1
        job["updated_at"] = time.time()

        _JOBS_MEM.store(
            content=job,
            metadata={"kind": "research_job", "job_id": job_id, "topic": job.get("topic", "")}
        )

    except Exception as e:
        print(f"[RESEARCH] Error incrementing facts count: {e}")


def _store_report(topic: str, summary: str, sources: List[str], facts_count: int, quality: Optional[Dict[str, Any]] = None) -> None:
    mod = _bank_module("research_reports")
    if not mod or not hasattr(mod, "service_api"):
        print(f"[RESEARCH] Warning: research_reports bank module not available")
        return
    try:
        quality = quality or {}
        report_fact = {
            "content": f"Topic: {topic}\nSummary: {summary}",
            "confidence": 0.8 if (quality.get("verdict") in {None, "ok", "skipped"}) else 0.55,
            "source": "research_manager",
            "metadata": {
                "topic": topic,
                "sources": sources,
                "facts_count": facts_count,
                "timestamp": time.time(),
                "quality_verdict": quality.get("verdict"),
                "quality_issues": quality.get("issues", []),
                "quality_low_trust": bool(quality and quality.get("verdict") not in {None, "ok", "skipped"}),
            },
        }
        result = mod.service_api({"op": "STORE", "payload": {"fact": report_fact}})
        if result.get("ok"):
            print(f"[RESEARCH] Stored research report for topic='{topic}' (id={result.get('payload', {}).get('stored_id', 'unknown')})")
        else:
            print(f"[RESEARCH] Warning: Failed to store report: {result.get('error', 'unknown')}")
    except Exception as e:
        print(f"[RESEARCH] Error storing report: {e}")


def _retrieve_report(topic: str) -> str | None:
    mod = _bank_module("research_reports")
    if not mod or not hasattr(mod, "service_api"):
        return None
    try:
        res = mod.service_api({"op": "RETRIEVE", "payload": {"query": topic}})
        payload = res.get("payload") or {}
        results = payload.get("results") or []
        if results:
            # Choose most recent by timestamp metadata when present
            results_sorted = sorted(results, key=lambda r: r.get("metadata", {}).get("timestamp", 0), reverse=True)
            return str(results_sorted[0].get("content", ""))
    except Exception:
        return None
    return None


def _plan_subquestions(topic: str, depth: int) -> List[str]:
    """Use Teacher to break a topic into sub-questions."""
    fallback_subquestions = [
        f"What is {topic}?",
        f"Key facts about {topic}",
    ]
    if not _teacher_helper:
        # Fallback: generate simple sub-questions without Teacher
        return fallback_subquestions

    try:
        max_questions = min(8, max(3, depth * 2))
        prompt = f"""Break down the topic "{topic}" into {max_questions} focused research sub-questions.

Each sub-question should:
- Focus on one specific aspect
- Be answerable with facts
- Build toward a complete understanding

Format: Return ONLY a numbered list like:
1. Question one here
2. Question two here
...

Topic: {topic}"""

        print(f"[RESEARCH_PLAN] Planning sub-questions for topic=\"{topic}\" (depth={depth})")

        # NOTE: check_memory_first is deprecated; memory-first is always enforced
        result = _teacher_helper.maybe_call_teacher(
            question=prompt,
            context={"topic": topic, "task": "plan_subquestions"},
        )

        if not result or not result.get("answer"):
            print(f"[RESEARCH_PLAN] Teacher returned no subquestions, using fallback")
            return fallback_subquestions

        answer = str(result.get("answer", "")).strip()

        # Parse numbered list
        lines = answer.split("\n")
        subquestions = []
        for line in lines:
            line = line.strip()
            # Match "1. Question" or "1) Question" or "- Question"
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering
                cleaned = line.lstrip("0123456789.-) \t")
                if cleaned and len(cleaned) > 10:
                    subquestions.append(cleaned)

        if len(subquestions) < 2:
            # Fallback if parsing failed
            print(f"[RESEARCH_PLAN] Failed to parse subquestions, using fallback")
            return fallback_subquestions

        print(f"[RESEARCH_PLAN] Subquestions count={len(subquestions)}")
        return subquestions[:max_questions]

    except Exception as e:
        print(f"[RESEARCH_PLAN] Error planning subquestions: {e}")
        return fallback_subquestions


def _extract_facts_from_text(text: str, topic: str, source: str) -> List[Dict[str, Any]]:
    """Use Teacher to extract and classify facts from text."""
    if not _teacher_helper or not text or len(text) < 50:
        return []

    try:
        prompt = f"""Extract key factual statements from the following text about "{topic}".

For each fact:
1. State it as a clear, standalone sentence
2. Indicate confidence (high/medium/low)

Text:
{text[:2000]}

Format your response as:
FACT: [statement] | CONFIDENCE: [high/medium/low]
FACT: [statement] | CONFIDENCE: [high/medium/low]
..."""

        print(f"[RESEARCH_FACTS] Extracting facts from {len(text)} chars (source={source})")

        # NOTE: check_memory_first is deprecated; memory-first is always enforced
        result = _teacher_helper.maybe_call_teacher(
            question=prompt,
            context={"topic": topic, "source": source, "task": "extract_facts"},
        )

        if not result or not result.get("answer"):
            # Fallback: split by sentences
            sentences = text.split(". ")
            facts = []
            for sent in sentences[:5]:
                if len(sent) > 30:
                    facts.append({
                        "content": sent.strip(),
                        "confidence": 0.5,
                        "source": source
                    })
            return facts

        answer = str(result.get("answer", "")).strip()

        # Parse FACT: ... | CONFIDENCE: ... format
        facts = []
        lines = answer.split("\n")
        for line in lines:
            if "FACT:" in line.upper():
                try:
                    parts = line.split("|")
                    fact_part = parts[0]
                    fact_text = fact_part.split("FACT:", 1)[1].strip() if "FACT:" in fact_part.upper() else fact_part.strip()

                    confidence = 0.6  # default
                    if len(parts) > 1 and "CONFIDENCE:" in parts[1].upper():
                        conf_str = parts[1].split("CONFIDENCE:", 1)[1].strip().lower()
                        if "high" in conf_str:
                            confidence = 0.8
                        elif "low" in conf_str:
                            confidence = 0.4
                        else:
                            confidence = 0.6

                    if fact_text and len(fact_text) > 15:
                        facts.append({
                            "content": fact_text,
                            "confidence": confidence,
                            "source": source
                        })
                except Exception:
                    pass

        print(f"[RESEARCH_FACTS] Extracted {len(facts)} facts")
        return facts

    except Exception as e:
        print(f"[RESEARCH_FACTS] Error extracting facts: {e}")
        return []


def _summarize_memory(topic: str) -> str:
    print(f"[RESEARCH] Querying existing memory for topic='{topic}'...")
    res = _call_memory_librarian({"op": "UNIFIED_RETRIEVE", "payload": {"query": topic, "k": 5}})
    payload = res.get("payload") or {}
    parts: List[str] = []
    for item in payload.get("results", []):
        content = str(item.get("content", "")).strip()
        if content:
            parts.append(content)
            _store_fact_record(content, float(item.get("confidence", 0.6)), "memory", topic)
    print(f"[RESEARCH] Memory summary: {len(parts)} existing facts found")
    return "; ".join(parts[:5])


def _baseline_teacher(topic: str) -> Dict[str, Any]:
    if not _teacher_helper:
        print(f"[RESEARCH] Teacher helper not available (offline mode)")
        return {}
    try:
        print(f"[RESEARCH] Calling Teacher for baseline understanding of '{topic}'...")
        question = f"Provide a concise briefing on {topic}. Include key factual statements."
        # NOTE: check_memory_first is deprecated; memory-first is always enforced
        result = _teacher_helper.maybe_call_teacher(question=question, context={"topic": topic})
        if not result:
            print(f"[RESEARCH] Teacher returned no result")
            return {}
        answer = str(result.get("answer") or "").strip()
        if answer:
            print(f"[RESEARCH] Teacher response: {len(answer)} chars, extracting facts...")
            fact_count = 0
            for sentence in answer.split(". "):
                sent = sentence.strip()
                if sent:
                    _store_fact_record(sent, 0.7, "teacher", topic)
                    fact_count += 1
            print(f"[RESEARCH] Extracted {fact_count} facts from Teacher response")
        return {"answer": answer}
    except Exception as e:
        print(f"[RESEARCH] Teacher call failed: {e}")
        return {}


def _web_research(topic: str, depth: int, budget: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_web_research_enabled():
        return {"findings": [], "budget_exhausted": False}

    print(f"[RESEARCH_WEB] Starting web research for topic='{topic}', depth={depth}")
    findings: List[str] = []
    base_queries = [topic, f"{topic} overview", f"key facts about {topic}"]
    budget_exhausted = False

    for idx, query in enumerate(base_queries):
        if _budget_exhausted(budget):
            budget_exhausted = True
            break
        if idx >= max(2, depth):
            break
        print(f"[RESEARCH_WEB] Query {idx+1}: '{query}'")
        start = time.monotonic()
        docs: List[WebDocument] = []
        try:
            docs = search_web(
                query=query,
                max_results=3,
                per_request_timeout=min(10.0, float(budget.get("time_budget_seconds") or _web_deadline_seconds())),
            )
        except Exception as e:
            print(f"[RESEARCH_WEB] Error during search: {e!r}")
            docs = []

        elapsed = time.monotonic() - start
        budget["requests_made"] = budget.get("requests_made", 0) + 1
        budget["time_budget_used"] = budget.get("time_budget_used", 0.0) + max(0.0, elapsed)

        if not docs:
            print("[RESEARCH_WEB] No search results parsed")
            continue

        print(f"[RESEARCH_WEB] Parsed {len(docs)} web document(s)")
        for doc in docs:
            combined_text = f"{doc.get('title', '')}\n\n{doc.get('text', '')}".strip()
            facts = _extract_facts_from_text(combined_text, topic, "web")
            if not facts:
                facts = [{"content": combined_text[:500], "confidence": 0.4, "source": "web"}]

            for fact in facts:
                stored = _store_fact_record(
                    fact.get("content", ""),
                    float(fact.get("confidence", 0.4)),
                    "web",
                    topic,
                    doc.get("url", ""),
                    metadata={"origin": "web", "title": doc.get("title", "")},
                )
                if stored:
                    findings.append(fact.get("content", ""))
        if _budget_exhausted(budget):
            budget_exhausted = True
            break
    print(f"[RESEARCH_WEB] Web research complete: {len(findings)} findings")
    return {"findings": findings, "budget_exhausted": budget_exhausted}


def _classify_research_topic(topic: str, payload: Dict[str, Any]) -> str:
    """
    Classify research topic into a signature for pattern matching.

    Returns one of:
    - explicit:deep+web (explicit user request)
    - explicit:quick
    - topic:system_self_diagnostics
    - topic:science
    - topic:current_events
    - topic:technical_docs
    - topic:trivial
    - topic:code
    - default
    """
    global _current_pattern

    lower = topic.lower()

    # Check for explicit depth/web requests from user
    if payload.get("depth") and int(payload.get("depth", 0)) >= 4:
        return "explicit:deep+web" if payload.get("web_enabled") else "explicit:deep"
    if payload.get("depth") and int(payload.get("depth", 0)) == 1:
        return "explicit:quick"

    # Check for system diagnostics
    if any(word in lower for word in ["diagnose", "self_diag", "system health", "maven"]):
        return "topic:system_self_diagnostics"

    # Check for current events
    current_words = ["latest", "recent", "today", "news", "current", "now"]
    if any(word in lower for word in current_words):
        return "topic:current_events"

    # Check for scientific topics
    science_words = ["solar", "quantum", "physics", "biology", "chemistry", "astronomy"]
    if any(word in lower for word in science_words):
        return "topic:science"

    # Check for technical docs
    if any(word in lower for word in ["documentation", "api", "library", "framework", "docs"]):
        return "topic:technical_docs"

    # Check for code topics
    if any(word in lower for word in ["code", "implement", "function", "class", "programming"]):
        return "topic:code"

    # Check for trivial (very short, simple questions)
    if len(topic) < 20:
        return "topic:trivial"

    return "default"


def _build_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _current_pattern

    topic_raw = str(payload.get("topic") or payload.get("text") or payload.get("query") or "").strip()

    # Classify topic for pattern matching
    signature = _classify_research_topic(topic_raw, payload)
    print(f"[RESEARCH_MANAGER] Research signature: {signature}")

    # Try to find a learned pattern
    pattern = _pattern_store.get_best_pattern(
        brain="research_manager",
        signature=signature,
        score_threshold=0.0  # Accept any pattern above 0
    )

    # Track which pattern we're using for later updates
    _current_pattern = pattern

    # Start with defaults
    depth_val = payload.get("depth") or payload.get("depth_level") or 2
    try:
        depth = max(1, min(3, int(depth_val)))
    except Exception:
        depth = 2

    default_web_enabled = _is_web_research_enabled()
    parsed_hints = _parse_web_hints(topic_raw, default_web_enabled)
    text = parsed_hints.get("topic") or topic_raw

    sources_raw = payload.get("sources") or ["memory", "teacher"]
    if isinstance(sources_raw, (list, tuple, set)):
        sources = list(sources_raw)
    else:
        sources = [sources_raw]
    deliverable = payload.get("deliverable") or "detailed_report"

    # Track explicit user intent for web controls
    explicit_web_flag = payload.get("web_enabled")
    if explicit_web_flag is None:
        explicit_web_flag = parsed_hints.get("web_enabled_explicit")

    explicit_time_budget = payload.get("time_budget_seconds")
    if explicit_time_budget is None:
        explicit_time_budget = parsed_hints.get("time_budget_seconds_explicit")

    explicit_max_requests = payload.get("max_requests") or payload.get("web_max_requests")
    if explicit_max_requests is None:
        explicit_max_requests = parsed_hints.get("max_web_requests_explicit")

    user_explicit_web_hint = any(
        val is not None for val in (explicit_web_flag, explicit_time_budget, explicit_max_requests)
    )

    # Base defaults
    web_enabled_flag = parsed_hints.get("web_enabled", default_web_enabled)
    time_budget = None
    max_requests = _max_web_requests_default(depth)

    # Precedence: user hints > learned pattern > defaults
    if pattern and pattern.score > -0.5:
        action = pattern.action

        if not user_explicit_web_hint:
            print(
                f"[RESEARCH_MANAGER] Using learned pattern: depth={action.get('depth')}, "
                f"web={action.get('web_enabled')}, time={action.get('time_budget_seconds')}s"
            )
        else:
            print(
                "[RESEARCH_MANAGER] Learned pattern available but explicit user web hint "
                "takes precedence"
            )

        if "depth" not in payload:
            depth = action.get("depth", depth)
        if not user_explicit_web_hint:
            web_enabled_flag = action.get("web_enabled", web_enabled_flag)
            time_budget = action.get("time_budget_seconds", time_budget)
            if explicit_max_requests is None:
                max_requests = action.get("max_web_requests", max_requests)

    # Finally, overlay explicit user hints
    if explicit_web_flag is not None:
        web_enabled_flag = bool(explicit_web_flag)
    if explicit_time_budget is not None:
        time_budget = explicit_time_budget
    if explicit_max_requests is not None:
        try:
            max_requests = max(1, int(explicit_max_requests))
        except Exception:
            pass

    # Set max_requests default if still None
    if max_requests is None:
        max_requests = _max_web_requests_default(depth)
    try:
        max_requests = max(1, int(max_requests))
    except Exception:
        max_requests = _max_web_requests_default(depth)

    # Update sources based on web_enabled (respect global hard-off only here)
    effective_web_enabled: bool = False
    web_allowed = _is_web_research_enabled()
    effective_web_enabled = bool(web_enabled_flag) and web_allowed
    if effective_web_enabled:
        if "web" not in sources:
            sources.append("web")
    else:
        sources = [s for s in sources if s != "web"]
        web_enabled_flag = False

    print(
        f"[RESEARCH_MANAGER] Final config: topic=\"{text}\", depth={depth}, "
        f"web_enabled={effective_web_enabled}, time_budget={time_budget}s, "
        f"max_web_requests={max_requests}"
    )

    return {
        "topic": text,
        "depth": depth,
        "sources": sources,
        "deliverable": deliverable,
        "web_enabled": bool(effective_web_enabled),
        "time_budget_seconds": time_budget,
        "max_web_requests": max_requests,
    }


def research_diagnostics() -> Dict[str, Any]:
    """Run lightweight diagnostics for research/web pipeline."""

    diagnostics: Dict[str, Any] = {}
    web_enabled = _is_web_research_enabled()
    diagnostics["web_research_enabled"] = web_enabled
    diagnostics["web_backend"] = "SerpAPI"
    diagnostics["max_web_deadline_seconds"] = _web_deadline_seconds()
    diagnostics["default_max_requests"] = _max_web_requests_default(2)

    # Sample final configurations to show precedence behavior
    sample_topic = "diagnostic probe"
    prior_pattern = _current_pattern
    diagnostics["sample_without_web_hint"] = _build_task({
        "topic": sample_topic,
        "sources": ["memory", "teacher"]
    })
    diagnostics["sample_with_web_hint"] = _build_task({
        "topic": sample_topic,
        "web_enabled": True,
        "time_budget_seconds": 60,
        "sources": ["memory", "teacher", "web"],
    })
    _current_pattern = prior_pattern

    # Connectivity check
    connectivity_msg = "skipped (web disabled)"
    if web_enabled:
        start = time.monotonic()
        try:
            docs = search_web(
                query="serpapi connectivity test",
                max_results=1,
                per_request_timeout=5.0,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            if docs:
                connectivity_msg = f"OK ({elapsed_ms:.0f} ms)"
            else:
                connectivity_msg = f"FAILED (no results, {elapsed_ms:.0f} ms)"
        except Exception as e:
            connectivity_msg = f"FAILED (error: {e})"

    diagnostics["serpapi_connectivity"] = connectivity_msg

    lines = [
        f"ENABLE_WEB_RESEARCH: {web_enabled}",
        "Backend: SerpAPI",
        f"Max web deadline: {_web_deadline_seconds()}s",
        f"Default max web requests (depth=2): {_max_web_requests_default(2)}",
        f"SerpAPI connectivity: {connectivity_msg}",
        "Sample config without web hint:",
        str(diagnostics["sample_without_web_hint"]),
        "Sample config with explicit web hint:",
        str(diagnostics["sample_with_web_hint"]),
    ]

    return {
        "ok": True,
        "text": "\n".join(lines),
        "diagnostics": diagnostics,
    }


def run_research(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a full research job with sub-questions, web fetching, and fact extraction.

    NOW WITH CONTINUATION AWARENESS:
    - Detects if this is expansion research (follow-up to previous research)
    - Scopes research to build on previous findings
    - Emits routing hints for Teacher learning
    """
    task = _build_task(payload)
    topic = task.get("topic", "").strip()
    full_prompt = payload.get("full_prompt") or f"research: {topic}"

    if not topic:
        return {"ok": False, "error": "missing_topic"}

    # CONTINUATION AWARENESS: Detect expansion research
    is_expansion = False
    base_research_topic = None
    conv_context = {}

    if _continuation_helpers_available:
        try:
            is_expansion = is_continuation(topic, {"topic": topic, "query": full_prompt})

            if is_expansion:
                conv_context = get_conversation_context()
                base_research_topic = conv_context.get("last_topic", "")
                print(f"[RESEARCH] ✓ Expansion research detected")
                print(f"[RESEARCH] Base topic: {base_research_topic}")
                print(f"[RESEARCH] Expansion query: {topic}")

                # Modify topic to include base context
                if base_research_topic and base_research_topic not in topic.lower():
                    topic = f"{topic} (expanding on: {base_research_topic})"
        except Exception as e:
            print(f"[RESEARCH] Warning: Continuation detection failed: {str(e)[:100]}")
            is_expansion = False

    print(f"[RESEARCH] Starting research task: topic='{topic}', depth={task['depth']}, sources={task['sources']}, expansion={is_expansion}")

    # Create research job
    web_enabled = bool(task.get("web_enabled")) and _is_web_research_enabled()
    budget_payload = dict(payload)
    if task.get("time_budget_seconds") is not None:
        budget_payload["time_budget_seconds"] = task.get("time_budget_seconds")
    if task.get("max_web_requests") is not None:
        budget_payload["max_requests"] = task.get("max_web_requests")
    web_budget: Dict[str, Any] = _resolve_web_budget(budget_payload, task.get("depth", 2)) if web_enabled else {}
    job_deadline = None
    if web_enabled:
        try:
            job_deadline = float(web_budget.get("start_time", time.monotonic())) + float(web_budget.get("time_budget_seconds", _web_deadline_seconds()))
        except Exception:
            job_deadline = time.monotonic() + _web_deadline_seconds()

    job_id = create_research_job(
        topic,
        full_prompt,
        owner="user",
        web_enabled=web_enabled,
        deadline=job_deadline,
    )
    update_research_job(job_id, {
        "status": "in_progress",
        "time_budget_seconds": web_budget.get("time_budget_seconds") if web_enabled else None,
        "max_web_requests": web_budget.get("max_requests") if web_enabled else task.get("max_web_requests"),
        "web_enabled": web_enabled,
        "sources": task.get("sources"),
    })

    # Step 1: Plan sub-questions (try pattern-based first)
    suggested_plan = None
    if _research_patterns_available and _research_pattern_mgr:
        try:
            suggested_plan = suggest_research_plan(topic)
            if suggested_plan and suggested_plan.subquestions:
                print(f"[RESEARCH_PLAN] Using learned pattern: {suggested_plan.pattern_id}")
                # Extract text from ResearchSubquestion objects
                subquestions = [
                    sq.text if hasattr(sq, 'text') else str(sq)
                    for sq in suggested_plan.subquestions
                ]
            else:
                subquestions = _plan_subquestions(topic, task.get("depth", 2))
        except Exception as e:
            print(f"[RESEARCH_PLAN] Pattern suggestion failed: {e}")
            subquestions = _plan_subquestions(topic, task.get("depth", 2))
    else:
        subquestions = _plan_subquestions(topic, task.get("depth", 2))

    for sq in subquestions:
        append_subquestion(job_id, sq, status="pending")
    print(f"[RESEARCH_PLAN] Subquestions count={len(subquestions)}")

    # Step 2: Query existing memory
    memory_notes = _summarize_memory(topic)

    # Step 3: Get Teacher baseline
    teacher_baseline = ""
    reflection_quality: Dict[str, Any] = {"verdict": "skipped"}
    if _teacher_helper:
        teacher_info = _baseline_teacher(topic)
        teacher_baseline = teacher_info.get("answer", "")

    # Step 4: Process each sub-question
    web_enabled = "web" in task.get("sources", [])
    total_facts_stored = 0
    max_sources_per_sq = 3
    web_fact_texts: List[str] = []
    budget_exhausted = False

    for i, sq in enumerate(subquestions):
        print(f"[RESEARCH] Processing sub-question {i+1}/{len(subquestions)}: {sq[:60]}...")

        # Try to answer from Teacher first
        if _teacher_helper:
            try:
                sq_result = _teacher_helper.maybe_call_teacher(
                    question=sq,
                    context={"topic": topic, "subquestion": sq},
                    check_memory_first=True
                )
                if sq_result and sq_result.get("answer"):
                    answer_text = str(sq_result.get("answer", ""))
                    # Teacher already stored facts - use the count it returned
                    teacher_facts_count = sq_result.get("facts_stored", 0)
                    if teacher_facts_count > 0:
                        print(f"[RESEARCH] Teacher stored {teacher_facts_count} facts for sub-question")
                        for _ in range(teacher_facts_count):
                            increment_facts_count(job_id)
                        total_facts_stored += teacher_facts_count
            except Exception as e:
                print(f"[RESEARCH] Error processing sub-question with Teacher: {e}")

        # Fetch from web if enabled
        if web_enabled and not _deadline_exceeded(job_deadline):
            if web_budget and _budget_exhausted(web_budget):
                budget_exhausted = True
                break
            try:
                timeout = 10.0
                if web_budget:
                    try:
                        timeout = min(10.0, float(web_budget.get("time_budget_seconds") or _web_deadline_seconds()))
                    except Exception:
                        timeout = 10.0
                start = time.monotonic()
                docs: List[WebDocument] = search_web(
                    query=f"{topic}: {sq}",
                    max_results=max_sources_per_sq,
                    per_request_timeout=timeout,
                )
                elapsed = time.monotonic() - start
                if web_budget is not None:
                    web_budget["requests_made"] = web_budget.get("requests_made", 0) + 1
                    web_budget["time_budget_used"] = web_budget.get("time_budget_used", 0.0) + max(0.0, elapsed)
                    update_research_job(job_id, {
                        "web_requests_used": web_budget.get("requests_made", 0),
                        "time_budget_used": web_budget.get("time_budget_used", 0.0),
                    })

                if not docs:
                    print("[RESEARCH_WEB] No search results parsed")
                    continue

                print(f"[RESEARCH_WEB] Parsed {len(docs)} web document(s)")
                for doc in docs:
                    if doc.get("url"):
                        append_source(job_id, sq[:80], str(doc.get("url")), trust_score=0.6)

                    combined_text = f"{doc.get('title', '')}\n\n{doc.get('text', '')}".strip()
                    web_confidence = 0.4
                    facts = _extract_facts_from_text(combined_text, topic, "web")
                    if not facts:
                        facts = [{"content": combined_text[:500], "confidence": web_confidence, "source": "web"}]

                    for fact in facts:
                        stored = _store_fact_record(
                            fact.get("content", ""),
                            float(fact.get("confidence", web_confidence)),
                            "web",
                            topic,
                            doc.get("url", ""),
                            metadata={"origin": "web", "title": doc.get("title", "")},
                        )
                        if stored:
                            print(f"[FACT_STORED] research_manager → factual: {fact.get('content', '')[:80]}")
                            increment_facts_count(job_id)
                            total_facts_stored += 1
                            if fact.get("content"):
                                web_fact_texts.append(str(fact.get("content")))

                if web_budget and _budget_exhausted(web_budget):
                    budget_exhausted = True
                    break

            except Exception as e:
                print(f"[RESEARCH_WEB] Error fetching for sub-question: {e}")
        elif web_enabled:
            budget_exhausted = True

    budget_elapsed = 0.0
    if web_enabled and web_budget:
        try:
            budget_elapsed = float(web_budget.get("time_budget_used", 0.0)) or max(0.0, time.monotonic() - float(web_budget.get("start_time", time.monotonic())))
        except Exception:
            budget_elapsed = 0.0

    # Step 5: Generate final summary
    if _teacher_helper:
        try:
            summary_prompt = f"""Based on the research conducted on "{topic}", provide:

1. A concise summary (2-3 paragraphs) suitable for answering the user
2. Key takeaways (3-5 bullet points)

Topic: {topic}
Sub-questions explored: {len(subquestions)}
Facts collected: {total_facts_stored}

Provide a clear, informative summary."""

            summary_result = _teacher_helper.maybe_call_teacher(
                question=summary_prompt,
                context={"topic": topic, "task": "final_summary"},
                check_memory_first=True
            )

            if summary_result and summary_result.get("answer"):
                draft_answer = str(summary_result.get("answer", "")).strip()

                # Step 5b: Self-reflection - critique and improve the draft answer
                print(f"[RESEARCH_REFLECT] Reviewing draft answer for quality and clarity")
                try:
                    reflection_ctx = {
                        "mode": "research_quality",
                        "question": topic,
                        "draft_answer": draft_answer,
                        "final_answer": draft_answer,
                        "confidence": summary_result.get("confidence", 0.72) if isinstance(summary_result, dict) else 0.72,
                        "facts": web_fact_texts or subquestions,
                        "web_facts": web_fact_texts,
                    }
                    reflection_result = run_reflection_engine("research_quality", reflection_ctx)
                    improved_answer = reflection_result.get("improved_answer")
                    verdict = reflection_result.get("verdict", "ok")
                    if reflection_result.get("improvement_needed") and improved_answer:
                        print(f"[RESEARCH_REFLECT] Using improved answer ({len(improved_answer)} chars)")
                        short_answer = improved_answer
                    else:
                        if verdict != "ok":
                            print(f"[RESEARCH_REFLECT] Reflection flagged issues: {verdict}")
                        short_answer = draft_answer
                    reflection_quality = reflection_result
                except Exception as e:
                    print(f"[RESEARCH_REFLECT] Error during reflection: {e}")
                    reflection_quality = {"verdict": "error", "issues": [str(e)[:120]]}
                    short_answer = draft_answer
            else:
                short_answer = f"I researched {topic} and stored {total_facts_stored} facts across {len(subquestions)} sub-questions."
        except Exception as e:
            print(f"[RESEARCH] Error generating summary: {e}")
            short_answer = f"I researched {topic} and stored {total_facts_stored} facts."
    else:
        short_answer = f"Research on {topic} complete. Stored {total_facts_stored} facts across {len(subquestions)} areas."

    # Step 5.5: Verify final answer if fact verification is available
    verification_result = None
    if _fact_verification_available and short_answer:
        try:
            verification_result = verify_answer(
                question=topic,
                answer=short_answer,
                method="heuristic_only"  # Fast verification without external calls
            )
            if verification_result and not verification_result.verified:
                print(f"[RESEARCH_VERIFY] Warning: Answer may have issues: {verification_result.confidence}")
        except Exception as e:
            print(f"[RESEARCH_VERIFY] Verification failed: {e}")

    # Step 5.6: Record research task for pattern learning
    if _research_patterns_available:
        try:
            sources_data = []
            for src in task.get("sources", []):
                if isinstance(src, str):
                    sources_data.append({"domain": src, "useful": True})

            subq_data = []
            for sq in subquestions:
                subq_data.append({"text": sq, "answered": True})

            record_research_task(
                topic=topic,
                subquestions=subq_data,
                sources=sources_data,
                depth=task.get("depth", 2),
                success=total_facts_stored > 0,
                promote_to_pattern=(total_facts_stored >= 3)  # Promote if successful
            )
            print(f"[RESEARCH_PATTERN] Recorded research task for pattern learning")
        except Exception as e:
            print(f"[RESEARCH_PATTERN] Failed to record task: {e}")

    # Step 6: Store research report
    sources_list = task.get("sources", [])
    if budget_exhausted and web_enabled:
        short_answer = f"{short_answer} (Web time budget reached; partial results returned.)"
    _store_report(topic, short_answer, sources_list, total_facts_stored, reflection_quality)

    # Step 7: Mark job complete
    update_research_job(job_id, {
        "status": "complete",
        "facts_stored": total_facts_stored,
        "budget_exhausted": budget_exhausted,
        "time_budget_used": budget_elapsed,
        "web_requests_used": web_budget.get("requests_made") if web_budget else 0,
    })
    print(f"[RESEARCH_DONE] job_id={job_id} status=complete facts={total_facts_stored}")

    # Log telemetry
    _MEM.store(
        content={"op": "research_run", "topic": topic, "depth": task.get("depth"), "facts_stored": total_facts_stored},
        metadata={"source": "research_manager", "kind": "telemetry", "job_id": job_id}
    )

    # Create routing hint for Teacher learning
    routing_hint = None
    if _continuation_helpers_available:
        try:
            action = "expand_research" if is_expansion else "new_research"
            context_tags = ["research", "web"] if web_enabled else ["research"]
            if is_expansion:
                context_tags.append("expansion")

            routing_hint = create_routing_hint(
                brain_name="research_manager",
                action=action,
                confidence=0.72,
                context_tags=context_tags
            )
        except Exception as e:
            print(f"[RESEARCH] Warning: Failed to create routing hint: {str(e)[:100]}")

    result = {
        "ok": True,
        "job_id": job_id,
        "summary": short_answer,
        "topic": topic,
        "sources": sources_list,
        "subquestions_processed": len(subquestions),
        "facts_collected": total_facts_stored,
        "answer": short_answer,
        "confidence": 0.72,
        "budget_exhausted": budget_exhausted,
        "time_budget_seconds": web_budget.get("time_budget_seconds") if web_budget else None,
        "time_budget_used": budget_elapsed,
    }

    # Add continuation awareness fields
    if is_expansion:
        result["is_expansion_research"] = True
        result["base_research_topic"] = base_research_topic

    # Add routing hint
    if routing_hint:
        result["routing_hint"] = routing_hint

    return result


def fetch_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(payload.get("topic") or payload.get("text") or payload.get("query") or "").strip()
    if not topic:
        return {"ok": False, "error": "missing_topic"}
    stored = _retrieve_report(topic)
    if stored:
        return {"ok": True, "summary": stored, "topic": topic, "confidence": 0.72}
    return {"ok": False, "error": "report_not_found"}


def answer_from_search_results(
    results: List[Dict[str, Any]],
    original_query: str,
    store_facts: bool = True,
    detail_level: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synthesize an answer from web search results using the research pipeline.

    This is the main integration point for web_search_tool to get answers
    from search results. It:
    1. Takes structured search results (title, url, snippet)
    2. Uses Teacher to synthesize a coherent answer with appropriate detail level
    3. Optionally stores key facts to memory
    4. Returns answer with sources

    Args:
        results: List of search results, each with {title, url, snippet}
        original_query: The original user query
        store_facts: Whether to store facts to memory (default: True)
        detail_level: DetailLevel enum value for response length (optional)
        context: Additional context for detail level inference

    Returns:
        Dict with:
        - ok: bool - Success status
        - text_answer: str - Synthesized answer
        - sources: List[str] - Source URLs
        - facts_stored: int - Number of facts stored
    """
    print(f"[RESEARCH_MANAGER] answer_from_search_results: query=\"{original_query}\", {len(results)} results")

    if not results:
        return {
            "ok": False,
            "error": "no_results",
            "text_answer": f"I couldn't find information about '{original_query}'.",
            "sources": [],
            "facts_stored": 0,
        }

    # Infer detail level if not provided
    if detail_level is None and _answer_style_available:
        detail_level = infer_detail_level(original_query, context)
        print(f"[RESEARCH_MANAGER] Inferred detail level: {detail_level}")

    # Build evidence bundle from top results
    top_results = results[:5]
    evidence_lines = []
    source_urls = []

    for i, r in enumerate(top_results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")

        evidence_lines.append(f"{i}. {title}")
        if snippet:
            evidence_lines.append(f"   {snippet[:250]}")
        if url:
            evidence_lines.append(f"   Source: {url}")
            source_urls.append(url)
        evidence_lines.append("")

    evidence_text = "\n".join(evidence_lines)

    # Get synthesis instructions based on detail level
    if _answer_style_available and detail_level:
        synthesis_instructions = get_web_search_synthesis_instruction(detail_level, is_followup=False)
    else:
        # Fallback to default instructions
        synthesis_instructions = """- Synthesize the information into a clear, helpful answer
- Provide a clear answer in a short paragraph (3-6 sentences)
- Do NOT make up information beyond what's in the results
- If the results don't fully answer the question, acknowledge the limitations"""

    # Use Teacher to synthesize answer
    text_answer = ""
    if _teacher_helper:
        try:
            prompt = f"""Based on the following web search results, provide an answer to: "{original_query}"

Search Results:
{evidence_text}

Instructions:
{synthesis_instructions}

Answer:"""

            result = _teacher_helper.maybe_call_teacher(
                question=prompt,
                context={"topic": original_query, "task": "search_result_synthesis"},
            )

            if result and result.get("answer"):
                text_answer = str(result.get("answer")).strip()
        except Exception as e:
            print(f"[RESEARCH_MANAGER] Teacher synthesis failed: {e}")

    # Fallback if Teacher unavailable or failed
    if not text_answer:
        answer_parts = [f"Here's what I found about '{original_query}':\n"]
        for i, r in enumerate(top_results[:3], 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if snippet:
                answer_parts.append(f"{i}. **{title}**: {snippet[:150]}...")
            else:
                answer_parts.append(f"{i}. **{title}**")
        text_answer = "\n".join(answer_parts)

    # Store facts if enabled
    facts_stored = 0
    if store_facts:
        for r in top_results[:3]:
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            url = r.get("url", "")

            if snippet:
                fact_content = f"{title}: {snippet[:300]}"
                stored = _store_fact_record(
                    content=fact_content,
                    confidence=0.6,
                    source="web_search",
                    topic=original_query,
                    url=url,
                    metadata={"origin": "web_search_synthesis"}
                )
                if stored:
                    facts_stored += 1
                    print(f"[RESEARCH_MANAGER] Stored fact from web search: {fact_content[:60]}...")

    # Store web search results for follow-up handling
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import store_web_search_result
        store_web_search_result(
            query=original_query,
            results=results,
            engine="",  # Not always known here
            answer=text_answer,
            sources=source_urls,
        )
    except Exception as e:
        print(f"[RESEARCH_MANAGER] Failed to store web search for follow-up: {e}")

    print(f"[RESEARCH_MANAGER] Synthesized answer ({len(text_answer)} chars), stored {facts_stored} facts")

    return {
        "ok": True,
        "text_answer": text_answer,
        "sources": source_urls,
        "facts_stored": facts_stored,
    }


def synthesize_web_followup(
    last_web_search: Dict[str, Any],
    followup_query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synthesize a detailed follow-up answer using cached web search results.

    This function handles "tell me more about X" type follow-ups after a web search.
    Instead of generating a generic LLM answer, it:
    1. Reuses the cached SERP results from the previous search
    2. Generates a DEEPER, more comprehensive answer (not just rephrased)
    3. Explicitly avoids repeating the first summary

    Args:
        last_web_search: Dict with previous search data:
            - query: original search query
            - results: list of search result dicts
            - answer: the first synthesized answer
            - sources: list of source URLs
        followup_query: The user's follow-up question (e.g., "tell me more about music")
        context: Additional context

    Returns:
        Dict with:
        - ok: bool - Success status
        - text_answer: str - Detailed follow-up answer
        - sources: List[str] - Source URLs
        - is_followup: True - Marker that this is a follow-up synthesis
    """
    original_query = last_web_search.get("query", "")
    results = last_web_search.get("results", [])
    previous_answer = last_web_search.get("answer", "")
    sources = last_web_search.get("sources", [])

    print(f"[RESEARCH_MANAGER] synthesize_web_followup: followup=\"{followup_query}\" on original=\"{original_query}\"")

    if not results:
        return {
            "ok": False,
            "error": "no_cached_results",
            "text_answer": f"I don't have previous search results to expand on. Would you like me to search for more information about '{original_query}'?",
            "sources": [],
            "is_followup": True,
        }

    # Build evidence bundle from all cached results
    evidence_lines = []
    for i, r in enumerate(results[:10], 1):  # Use more results for deeper answer
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")

        evidence_lines.append(f"{i}. {title}")
        if snippet:
            evidence_lines.append(f"   {snippet[:300]}")  # More snippet content
        if url:
            evidence_lines.append(f"   Source: {url}")
        evidence_lines.append("")

    evidence_text = "\n".join(evidence_lines)

    # Use Teacher to synthesize a deeper answer
    text_answer = ""
    if _teacher_helper:
        try:
            # Get LONG detail instructions for follow-up
            if _answer_style_available:
                synthesis_instructions = get_web_search_synthesis_instruction(DetailLevel.LONG, is_followup=True)
            else:
                synthesis_instructions = """- Provide a comprehensive, detailed explanation that expands significantly on the previous answer
- Cover multiple aspects: history, types/categories, examples, current relevance
- Use multiple paragraphs and structure with bullet points if helpful
- Do NOT simply rephrase the previous answer - go deeper
- Do NOT make up information beyond what's in the results"""

            # Build explicit topic-aware prompt with all context
            prompt = f"""CONTEXT:
- Base topic: "{original_query}"
- Previous summary given to user: "{previous_answer[:500]}..."
- User's follow-up request: "{followup_query}"

This is a follow-up "tell me more" request. The user wants SIGNIFICANTLY MORE information than the brief summary above.

AVAILABLE SEARCH RESULTS:
{evidence_text}

CRITICAL REQUIREMENTS:
{synthesis_instructions}

Remember:
- The base topic is: "{original_query}"
- Do NOT repeat the previous summary - expand with NEW information
- Cover different facets: definitions, history, categories/types, examples, significance

Comprehensive Answer (at least 2 paragraphs):"""

            result = _teacher_helper.maybe_call_teacher(
                question=prompt,
                context={
                    "topic": original_query,
                    "task": "web_followup_synthesis",
                    "is_followup": True,
                },
            )

            if result and result.get("answer"):
                text_answer = str(result.get("answer")).strip()

                # Length/content sanity check - "tell me more" needs substantive content
                # Minimum 400 chars (2 paragraphs) and 5 sentences for follow-ups
                sentence_count = len([s for s in text_answer.split('.') if s.strip()])
                min_chars = 400
                min_sentences = 5

                if len(text_answer) < min_chars or sentence_count < min_sentences:
                    print(f"[RESEARCH_MANAGER] Follow-up answer too short ({len(text_answer)} chars, {sentence_count} sentences) - requesting expansion")

                    # Re-call with explicit expansion instruction
                    expand_prompt = f"""The previous answer was too brief. Please expand SIGNIFICANTLY.

TOPIC: "{original_query}"
PREVIOUS BRIEF ANSWER: "{text_answer}"

Search Results Available:
{evidence_text}

REQUIREMENTS:
- Write AT LEAST 2-3 full paragraphs
- Include specific examples, facts, or statistics from the search results
- Cover multiple aspects: history/background, different types/categories, real-world applications
- Do NOT just rephrase - add substantial NEW information

Expanded Answer:"""

                    # CRITICAL: Force LLM call for expansion - don't use cached short answers
                    expand_result = _teacher_helper.maybe_call_teacher(
                        question=expand_prompt,
                        context={
                            "topic": original_query,
                            "task": "web_followup_expansion",
                            "is_followup": True,
                            "force_fresh_llm": True,  # Skip memory check for expansions
                        },
                    )

                    if expand_result and expand_result.get("answer"):
                        expanded = str(expand_result.get("answer")).strip()
                        # Use expansion if it's meaningfully longer (at least 50% more)
                        if len(expanded) > len(text_answer) * 1.5:
                            text_answer = expanded
                            print(f"[RESEARCH_MANAGER] Expansion successful ({len(text_answer)} chars)")
                        elif len(expanded) > 400:
                            # Or if it's reasonably long even if not that much longer
                            text_answer = expanded
                            print(f"[RESEARCH_MANAGER] Expansion used (adequate length: {len(text_answer)} chars)")

        except Exception as e:
            print(f"[RESEARCH_MANAGER] Teacher followup synthesis failed: {e}")

    # Fallback if Teacher unavailable or failed
    if not text_answer:
        answer_parts = [f"Here's more detail about '{original_query}':\n"]
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if snippet:
                answer_parts.append(f"**{title}**")
                answer_parts.append(f"{snippet[:300]}\n")
        text_answer = "\n".join(answer_parts)

    print(f"[RESEARCH_MANAGER] Follow-up synthesis complete ({len(text_answer)} chars)")

    return {
        "ok": True,
        "text_answer": text_answer,
        "sources": sources,
        "is_followup": True,
        "original_query": original_query,
    }


def update_from_verdict(verdict: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Update the current pattern's score based on SELF_REVIEW/Teacher verdict.

    For RESEARCH_MANAGER, we also consider:
    - timeout: research took too long → negative reward
    - budget_exceeded: too many web requests → negative reward
    - underkill: didn't go deep enough → negative reward
    - overkill: went too deep for a simple question → negative reward

    Args:
        verdict: One of 'ok', 'minor_issue', 'major_issue'
        metadata: Optional dict with 'timeout', 'budget_exceeded', 'underkill', 'overkill' flags
    """
    global _current_pattern

    if not _current_pattern:
        print("[RESEARCH_MANAGER] No current pattern to update")
        return

    # Start with base reward from verdict
    reward = verdict_to_reward(verdict)

    # Adjust based on research-specific metadata
    if metadata:
        if metadata.get("timeout"):
            reward -= 0.5  # Timeout is bad
        if metadata.get("budget_exceeded"):
            reward -= 0.3  # Going over budget is bad
        if metadata.get("underkill"):
            reward -= 0.2  # Not deep enough
        if metadata.get("overkill"):
            reward -= 0.2  # Too deep for the question

    # Clamp reward to [-1, 1]
    reward = max(-1.0, min(1.0, reward))

    # Update pattern score
    _pattern_store.update_pattern_score(
        pattern=_current_pattern,
        reward=reward,
        alpha=0.85  # Learning rate: 0.85 = slower, more stable
    )

    print(f"[RESEARCH_MANAGER] Updated pattern {_current_pattern.signature} "
          f"based on verdict={verdict} (reward={reward:+.1f})")


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op", "").upper()
    payload = (msg or {}).get("payload") or {}
# COGNITIVE BRAIN CONTRACT: Signal 1 & 2 - Detect continuation and get context
    continuation_detected = False
    conv_context = {}

    try:
        # Extract query from payload
        query = (payload.get("query") or
                payload.get("question") or
                payload.get("user_query") or
                payload.get("text") or "")

        if query:
            continuation_detected = is_continuation(query, payload)

            if continuation_detected:
                conv_context = get_conversation_context()
                # Enrich payload with conversation context
                payload["continuation_detected"] = True
                payload["last_topic"] = conv_context.get("last_topic", "")
                payload["conversation_depth"] = conv_context.get("conversation_depth", 0)
    except Exception as e:
        # Silently continue if continuation detection fails
        pass
    mid = msg.get("mid")
    if op == "RUN_RESEARCH":
        res = run_research(payload)
        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="research_manager",
                action="run_research",
                confidence=0.7,
                context_tags=[
                    "run_research",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )
            if isinstance(res, dict):
                res["routing_hint"] = routing_hint
        except Exception:
            pass  # Routing hint generation is non-critical
        return {"ok": bool(res.get("ok")), "op": op, "mid": mid, "payload": res}
    if op == "RESEARCH_DIAGNOSTICS":
        res = research_diagnostics()
        return {"ok": bool(res.get("ok", True)), "op": op, "mid": mid, "payload": res}
    if op == "FETCH_REPORT":
        res = fetch_report(payload)
        return {"ok": bool(res.get("ok")), "op": op, "mid": mid, "payload": res}
    if op == "UPDATE_FROM_VERDICT":
        verdict = str(payload.get("verdict", "ok"))
        metadata = payload.get("metadata")
        update_from_verdict(verdict, metadata)
        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="research_manager",
                action="fetch_report",
                confidence=0.7,
                context_tags=[
                    "fetch_report",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )
            if isinstance(result, dict):
                result["routing_hint"] = routing_hint
            elif isinstance(payload_result, dict):
                payload_result["routing_hint"] = routing_hint
        except Exception:
            pass  # Routing hint generation is non-critical
        return {"ok": True, "op": op, "mid": mid, "payload": {"updated": True}}

    # Handle web search follow-up synthesis
    # This is called when the integrator routes a follow-up question
    # (like "tell me more about music") to research_manager
    if op == "WEB_FOLLOWUP_SYNTHESIZE":
        followup_query = (payload.get("query") or
                        payload.get("question") or
                        payload.get("user_query") or
                        payload.get("text") or "")
        last_web_search = payload.get("last_web_search", {})

        # If no cached search in payload, try to get from memory_librarian
        if not last_web_search:
            try:
                from brains.cognitive.memory_librarian.service.memory_librarian import get_last_web_search
                last_web_search = get_last_web_search() or {}
            except Exception:
                pass

        res = synthesize_web_followup(last_web_search, followup_query, payload)

        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="research_manager",
                action="web_followup_synthesize",
                confidence=0.8,
                context_tags=["web_followup", "continuation", "expansion"]
            )
            if isinstance(res, dict):
                res["routing_hint"] = routing_hint
        except Exception:
            pass

        return {"ok": bool(res.get("ok")), "op": op, "mid": mid, "payload": res}

    # Handle when integrator sends us a follow-up via web_followup_mode flag
    # This is an alternate path where the context contains the flag
    if payload.get("web_followup_mode"):
        followup_query = (payload.get("query") or
                        payload.get("question") or
                        payload.get("user_query") or
                        payload.get("text") or "")
        last_web_search = payload.get("last_web_search", {})

        # If no cached search in payload, try to get from memory_librarian
        if not last_web_search:
            try:
                from brains.cognitive.memory_librarian.service.memory_librarian import get_last_web_search
                last_web_search = get_last_web_search() or {}
            except Exception:
                pass

        if last_web_search and followup_query:
            res = synthesize_web_followup(last_web_search, followup_query, payload)
            return {"ok": bool(res.get("ok")), "op": "WEB_FOLLOWUP_SYNTHESIZE", "mid": mid, "payload": res}

    return {"ok": False, "op": op, "mid": mid, "error": {"code": "UNSUPPORTED_OP", "message": op}}


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
