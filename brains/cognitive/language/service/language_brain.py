from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
import re, time, json, random
import os
from pathlib import Path
from brains.memory.brain_memory import BrainMemory
from brains.utils.safe_math_eval import safe_math_eval_str

def _diag_enabled(ctx=None):
    if isinstance(ctx, dict) and ctx.get("_diag"): return True
    return os.getenv("MAVEN_DIAG", "0") == "1"

def _diag_log(tag, rec):
    try:
        line = {"tag": tag, **(rec or {})}
        # Store diagnostic log entry in BrainMemory
        _LANGUAGE_MEMORY.store(
            content=line,
            metadata={"kind": "diagnostic", "source": "language", "confidence": 1.0}
        )
    except Exception:
        pass

def _safe_val(resp):
    if isinstance(resp, dict):
        # Handle nested payload structure from BRAIN_GET
        payload = resp.get("payload", {})
        if isinstance(payload, dict):
            data = payload.get("data", {})
            if isinstance(data, dict):
                return data.get("value")
    return None
import re as _re
def _math_key(text: str) -> str | None:
    if not text: return None
    m = _re.match(r"\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*$", text)
    if not m: return None
    a, op, b = m.groups()
    return f"{int(a)}{op}{int(b)}"
# Create an alias for the re module.  Using `_re` avoids accidental
# masking of the global `re` name within nested scopes (e.g. inside
# service_api).  When using regex functions in service_api, prefer
# `_re` to reference the module to prevent UnboundLocalError.
_re = re
try:
    # Python 3.9+ includes zoneinfo in the standard library.  It is used
    # here to localise greetings based on the user's timezone if available.
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:
    # Fallback when zoneinfo is unavailable; the variable will be None.
    ZoneInfo = None  # type: ignore
from pathlib import Path

# ---------------------------------------------------------------------------
# LLM integration (Phase 6)
#
# Import the local LLM service if available.  The service wraps a local
# Ollama instance and provides pattern learning capabilities.  If the
# import fails (e.g., service not installed), ``_llm`` will be ``None`` and
# the language brain will skip LLM fallback.
try:
    from brains.tools.llm_service import llm_service as _llm  # type: ignore
except Exception:
    _llm = None  # type: ignore

# ---------------------------------------------------------------------------
# BrainMemory initialization
#
# Initialize the language brain's memory tier system.  All persistent state
# for the language brain (diagnostics, Q&A history, etc.) should use this
# instance instead of direct file I/O.
_LANGUAGE_MEMORY = BrainMemory("language")

# ---------------------------------------------------------------------------
# Learning mode for memory-first, LLM-as-teacher architecture
#
# LearningMode controls cognitive behavior:
# - TRAINING: Memory-first -> if miss -> call LLM -> store lesson/facts
# - OFFLINE: Use strategies and memory only, no new learning
# - SHADOW: LLM for comparison only, no storage
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"

# Lesson utilities for strategy-based learning
try:
    from brains.learning.lesson_utils import (
        create_lesson_record,
        store_lesson,
        retrieve_lessons
    )
except Exception:
    create_lesson_record = None  # type: ignore
    store_lesson = None  # type: ignore
    retrieve_lessons = None  # type: ignore

# ---------------------------------------------------------------------------
# Teacher integration for learning language styles and phrasing patterns
#
# The TeacherHelper allows the language brain to learn new phrasing styles
# and response patterns when encountering novel communication contexts.
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("language")
except Exception as e:
    print(f"[LANGUAGE] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for detecting follow-ups and accessing conversation history
# Required for implementing the three-signal cognitive brain contract
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint,
        extract_continuation_intent,
        enhance_query_with_context
    )
except Exception as e:
    print(f"[LANGUAGE] Continuation helpers not available: {e}")
    # Provide fallback stubs
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore
    enhance_query_with_context = lambda query, context: query  # type: ignore

# Answer style module for detail level control
# This replaces hard-coded "concise" instructions with dynamic length selection
try:
    from brains.cognitive.language.answer_style import (
        DetailLevel,
        infer_detail_level,
        get_length_instruction,
    )
    _answer_style_available = True
except Exception as e:
    print(f"[LANGUAGE] Answer style not available: {e}")
    _answer_style_available = False
    # Provide fallback
    class DetailLevel:  # type: ignore
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"
    def infer_detail_level(question, context=None):  # type: ignore
        return DetailLevel.MEDIUM
    def get_length_instruction(detail_level):  # type: ignore
        return "Provide a clear, helpful answer."

# PHASE 2: Context manager for follow-up action tracking
try:
    from brains.cognitive.context_management.service.context_manager import (
        is_confirmation_message,
        get_follow_up_context,
        mark_action_executed,
    )
    _action_tracking_available = True
except Exception as e:
    print(f"[LANGUAGE] Action tracking not available: {e}")
    _action_tracking_available = False
    is_confirmation_message = lambda text: False  # type: ignore
    get_follow_up_context = lambda: None  # type: ignore
    mark_action_executed = lambda: None  # type: ignore


# =============================================================================
# MEMORY-FIRST LEARNING ROUTE (following Reasoning brain pattern)
# =============================================================================
#
# Problem Types for Language:
# - greeting_response: Temporal/contextual greetings
# - question_answering: Factual Q&A with style adaptation
# - creative_writing: Story continuations, creative prompts
# - social_engagement: Relationship-aware responses
# - explanation_style: Tailored explanations for different audiences
# - emotion_recognition: Sentiment-aware phrasing
# - generic_response: Fallback for unmapped intents
#
# Strategy Table: Maps (problem_type, domain) -> strategy dict
# =============================================================================

LANGUAGE_STRATEGY_TABLE: Dict[tuple, Dict[str, Any]] = {}


def classify_language_problem(context: Dict[str, Any]) -> str:
    """
    Classify the type of language task from context.

    Args:
        context: Pipeline context containing query and intent

    Returns:
        Problem type string for strategy selection
    """
    try:
        query = ""
        if "user_query" in context:
            query = str(context["user_query"])
        elif "payload" in context and isinstance(context["payload"], dict):
            query = str(context["payload"].get("query_text", ""))

        q_lower = (query or "").lower()

        # Greeting patterns
        if any(w in q_lower for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            return "greeting_response"

        # Question patterns
        if any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how"]):
            return "question_answering"

        # Creative patterns
        if any(w in q_lower for w in ["write", "story", "creative", "imagine", "poem"]):
            return "creative_writing"

        # Social patterns
        if any(w in q_lower for w in ["thank", "please", "sorry", "feel", "emotion"]):
            return "social_engagement"

        # Explanation patterns
        if any(w in q_lower for w in ["explain", "describe", "tell me about"]):
            return "explanation_style"

        return "generic_response"

    except Exception:
        return "generic_response"


def load_language_strategies_from_lessons(context: Dict[str, Any]) -> None:
    """
    Load language strategies from stored lessons into LANGUAGE_STRATEGY_TABLE.
    """
    global LANGUAGE_STRATEGY_TABLE

    if not retrieve_lessons:
        return

    try:
        lessons = retrieve_lessons("language", _LANGUAGE_MEMORY)

        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue

            topic = lesson.get("topic", "")
            input_sig = lesson.get("input_signature", {})
            problem_type = input_sig.get("problem_type", topic) or "generic_response"
            domain = input_sig.get("domain")

            strategy = {
                "name": f"learned_{problem_type}",
                "problem_type": problem_type,
                "domain": domain,
                "template": lesson.get("distilled_rule", ""),
                "confidence": lesson.get("confidence", 0.5),
                "examples": lesson.get("examples", []),
            }

            strategy_key = (problem_type, domain)
            existing = LANGUAGE_STRATEGY_TABLE.get(strategy_key)
            if not existing or strategy["confidence"] > existing.get("confidence", 0):
                LANGUAGE_STRATEGY_TABLE[strategy_key] = strategy

        if LANGUAGE_STRATEGY_TABLE:
            print(f"[LANGUAGE] Loaded {len(LANGUAGE_STRATEGY_TABLE)} strategies from lessons")

    except Exception as e:
        print(f"[LANGUAGE] Error loading strategies: {e}")


def select_language_strategy(problem_type: str, domain: str = None) -> Dict[str, Any] | None:
    """
    Select the best language strategy for a problem type.
    """
    strategy = LANGUAGE_STRATEGY_TABLE.get((problem_type, domain))
    if strategy:
        return strategy

    strategy = LANGUAGE_STRATEGY_TABLE.get((problem_type, None))
    if strategy:
        return strategy

    if problem_type != "generic_response":
        return LANGUAGE_STRATEGY_TABLE.get(("generic_response", None))

    return None


def language_llm_lesson(
    context: Dict[str, Any],
    query: str,
    learning_mode
) -> Dict[str, Any] | None:
    """
    Generate a lesson from LLM interaction for the language brain.

    Args:
        context: Pipeline context
        query: The language task/query
        learning_mode: Current learning mode

    Returns:
        Lesson record dict, or None if failed
    """
    if not create_lesson_record or not store_lesson:
        return None

    problem_type = classify_language_problem(context)
    domain = context.get("domain")

    input_signature = {
        "problem_type": problem_type,
        "domain": domain,
    }

    # OFFLINE mode - no LLM call
    if learning_mode == LearningMode.OFFLINE:
        return create_lesson_record(
            brain="language",
            topic=problem_type,
            input_signature=input_signature,
            llm_prompt="",
            llm_response="",
            distilled_rule="",
            examples=[],
            confidence=0.0,
            mode="offline",
            status="new"
        )

    # TRAINING or SHADOW mode - call TeacherHelper
    if not _teacher_helper:
        return None

    try:
        teacher_result = _teacher_helper.maybe_call_teacher(
            question=query,
            context=context,
            learning_mode=learning_mode
        )

        if not teacher_result or teacher_result.get("verdict") in ("LLM_DISABLED", "NO_MEMORY"):
            return None

        llm_response = teacher_result.get("answer", "")
        verdict = teacher_result.get("verdict", "UNKNOWN")
        confidence = 0.8 if verdict == "LEARNED" else (0.9 if verdict == "KNOWN" else 0.5)

        lesson = create_lesson_record(
            brain="language",
            topic=problem_type,
            input_signature=input_signature,
            llm_prompt=f"Language task: {query}",
            llm_response=llm_response or "",
            distilled_rule=llm_response or "",
            examples=[],
            confidence=confidence,
            mode=str(learning_mode.value) if hasattr(learning_mode, 'value') else str(learning_mode),
            status="new"
        )

        stored = store_lesson("language", lesson, _LANGUAGE_MEMORY)
        if stored:
            print(f"[LANGUAGE] Stored lesson: {problem_type} (confidence={confidence})")

        lesson["response"] = llm_response
        return lesson

    except Exception as e:
        print(f"[LANGUAGE] Lesson error: {e}")
        return None


def build_generation_prompt(query: str, memory_results: List[Dict[str, Any]], context: Dict[str, Any] | None) -> str:
    """Construct a prompt for the LLM call.

    The prompt includes the user query, optionally the user's name,
    and up to three relevant memory snippets.  It concludes with
    guidance to be concise and helpful.  This helper is used in Stage 6
    when the language brain falls back to the LLM.

    Args:
        query: The user's raw text input.
        memory_results: A list of memory dicts containing 'content'.
        context: Optional context dict with a nested 'user' dict.

    Returns:
        A formatted prompt string.
    """
    # Start with assistant persona and user query
    prompt = f"You are Maven, a helpful AI assistant.\n\nUser query: \"{query}\"\n"
    # Append user name if available
    try:
        if context:
            user = context.get("user") or {}
            name = user.get("name")
            if name:
                prompt += f"User's name: {name}\n"
    except Exception:
        pass
    # Include relevant information from memory
    try:
        if memory_results:
            prompt += "\nRelevant information:\n"
            count = 0
            for item in memory_results:
                if count >= 3:
                    break
                try:
                    cont = str((item.get("content") or "")).strip()
                except Exception:
                    cont = ""
                if cont:
                    prompt += f"- {cont}\n"
                    count += 1
    except Exception:
        pass
    # Infer appropriate detail level based on query and context
    detail_level = infer_detail_level(query, context)
    length_instruction = get_length_instruction(detail_level)

    prompt += f"\nRespond naturally and helpfully. {length_instruction}\n\nResponse:"
    return prompt

# ---------------------------------------------------------------------------
# Identity card helpers
# ---------------------------------------------------------------------------
# Import identity card helpers for storing Maven's self metadata and the
# primary user's identity.  These functions live in ``api/identity_cards.py``
# and provide simple persistence without external dependencies.
try:
    from api.identity_cards import resolve_primary_user_name, set_primary_user  # type: ignore
except Exception:
    # Fallback stubs prevent runtime failures if the module cannot be
    # imported.  In practice, identity_cards will be available.
    resolve_primary_user_name = lambda: None  # type: ignore
    def set_primary_user(display_name: str, user_id: str | None = None, consent: bool | None = True) -> None:  # type: ignore
        return None

# ---------------------------------------------------------------------------
# Dynamic service loader and relationship resolution helpers
# ---------------------------------------------------------------------------
import importlib.util

# Compute paths relative to this file
HERE = Path(__file__).resolve().parent
BRAIN_DIR = HERE.parent
COGNITIVE_ROOT = BRAIN_DIR.parent
BRAINS_ROOT = COGNITIVE_ROOT.parent
MAVEN_ROOT = BRAINS_ROOT.parent

def _load_service(rel_path_from_maven: str) -> Any:
    """Dynamically load a service module from a path relative to MAVEN_ROOT.

    Args:
        rel_path_from_maven: Relative path from MAVEN_ROOT to the service file.

    Returns:
        The loaded module.
    """
    try:
        p = MAVEN_ROOT / rel_path_from_maven
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    except Exception:
        pass
    return None

def _mem_call(payload: dict):
    m = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
    try:
        res = m.service_api(payload)
        _diag_log("mem_call", {"payload": payload, "result": res})
        return res
    except Exception as e:
        _diag_log("mem_error", {"payload": payload, "error": str(e)})
        raise

def _resolve_relationship_status(context: dict) -> str | None:
    """Return stored relationship_status if present.

    Checks both brain-level persistent storage and working memory.

    Args:
        context: The current pipeline context.

    Returns:
        The relationship status string if found, otherwise None.
    """
    try:
        mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
        if not mem:
            return None
        # Prefer brain-level self-memory
        r = mem.service_api({
            "op": "BRAIN_GET",
            "payload": {
                "scope": "BRAIN",
                "origin_brain": "memory_librarian",
                "key": "relationship_status"
            }
        })
        if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
            data = r.get("payload", {}).get("data", {})
            if data and data.get("value"):
                return data["value"]
        # Optional: WM fallback if available
        r2 = mem.service_api({"op": "WM_GET", "payload": {"key": "relationship_status"}})
        if isinstance(r2, dict) and r2.get("ok"):
            entries = r2.get("payload", {}).get("entries", [])
            if entries and entries[0].get("value"):
                return entries[0]["value"]
    except Exception:
        return None
    return None

def _resolve_user_name(context: dict) -> str | None:
    """Return stored user_identity (name) if present.

    Checks both brain-level persistent storage and working memory.

    Args:
        context: The current pipeline context.

    Returns:
        The user's name string if found, otherwise None.
    """
    try:
        mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
        if not mem:
            return None
        # Prefer brain-level self-memory
        r = mem.service_api({
            "op": "BRAIN_GET",
            "payload": {
                "scope": "BRAIN",
                "origin_brain": "memory_librarian",
                "key": "user_identity"
            }
        })
        if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
            data = r.get("payload", {}).get("data", {})
            if data and data.get("value"):
                return data["value"]
        # Optional: WM fallback if available
        r2 = mem.service_api({"op": "WM_GET", "payload": {"key": "user_identity"}})
        if isinstance(r2, dict) and r2.get("ok"):
            entries = r2.get("payload", {}).get("entries", [])
            if entries and entries[0].get("value"):
                return entries[0]["value"]
    except Exception:
        return None
    return None

def _normalize_math_key(text: str) -> str | None:
    """Return canonical key for simple a op b expressions like '2+2'.

    Args:
        text: The input text to parse for a math expression.

    Returns:
        A canonical math key (e.g., "2+2") or None if not a simple math expression.
    """
    if not text:
        return None
    m = _re.match(r"\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*$", text)
    if not m:
        return None
    a, op, b = m.groups()
    return f"{int(a)}{op}{int(b)}"

def run_diagnostics():
    """
    Scripted chat to detect where memory/learning breaks.
    Writes JSONL to reports/diagnostics/diag.jsonl and summary to reports/diagnostics/summary.md
    """
    cases = [
        # identity
        {"say":"i am josh"},
        {"say":"who am i", "expect_contains":"You are", "label":"identity_recall"},
        # relationship
        {"say":"we are friends"},
        {"say":"are we friends", "expect_contains":"Yes", "label":"relationship_recall"},
        # preference
        {"say":"i like green"},
        {"say":"what color do i like", "expect_contains":"green", "label":"color_recall"},
        # math learning baseline
        {"say":"2+2"},
        {"say":"correct"},
        {"say":"2+2", "expect_conf_increase":True, "label":"math_conf_bump"},
        # typo tolerant correction
        {"say":"2+3"},
        {"say":"corrrect"},
        {"say":"2+3", "expect_conf_increase":True, "label":"math_conf_bump_typo"},
    ]

    def _turn(utt, ctx):
        # Simple wrapper to call service_api with diagnostics enabled
        ctx = dict(ctx or {})
        ctx.update({"_diag": True})
        # Call PARSE first
        parse_result = service_api({
            "op": "PARSE",
            "payload": {"text": utt, "_diag": True}
        })
        # Then GENERATE
        gen_result = service_api({
            "op": "GENERATE",
            "payload": {"query": utt, "_diag": True, "context": ctx}
        })
        # Extract output text from candidates
        candidates = gen_result.get("payload", {}).get("candidates", [])
        output = candidates[0].get("text", "") if candidates else ""
        confidence = candidates[0].get("confidence", 0.0) if candidates else 0.0
        return {"output": output, "confidence": confidence}

    fails = []
    last_conf = {}
    ctx = {}
    for step in cases:
        utt = step["say"]
        out = _turn(utt, ctx)
        text = str(out.get("output",""))
        conf = out.get("confidence", 0.0)
        label = step.get("label") or utt
        # content check
        if "expect_contains" in step and step["expect_contains"].lower() not in text.lower():
            fails.append((label, f'missing "{step["expect_contains"]}" in "{text}"'))
        # confidence trend check
        if step.get("expect_conf_increase"):
            key = utt.replace(" ", "")
            prev = last_conf.get(key, 0.0)
            if conf <= prev:
                fails.append((label, f"confidence did not increase: prev={prev}, now={conf}"))
            last_conf[key] = conf
        else:
            last_conf[utt.replace(" ","")] = conf
        _diag_log("test_step", {"say": utt, "output": text, "confidence": conf, "label": label})
        ctx["prev_utterance"] = utt

    # Store diagnostic summary in BrainMemory
    summary_content = {
        "type": "diagnostic_summary",
        "status": "passed" if not fails else "failed",
        "failures": [{"name": name, "reason": reason} for name, reason in fails],
        "total_failures": len(fails)
    }
    _LANGUAGE_MEMORY.store(
        content=summary_content,
        metadata={"kind": "diagnostic_summary", "source": "language", "confidence": 1.0}
    )
    return {"status": "ok", "failures": len(fails)}

# ---------------------------------------------------------------------------
# Domain knowledge cache
#
# A lightweight cache to hold technology domain records for on‑the‑fly
# knowledge synthesis.  When a user asks a factual question that is not
# answered by reasoning or summarisation, Stage 6 will consult this cache
# to look up relevant entries from the technology domain bank.  The cache
# is loaded on first use from ``brains/domain_banks/technology/memory/stm/records.jsonl``
# and stored in the module‑level ``_TECH_KNOWLEDGE_CACHE`` variable.  If loading
# fails or the file is missing, the cache remains ``None`` and knowledge
# synthesis will silently skip lookup.
_TECH_KNOWLEDGE_CACHE: List[Dict[str, Any]] | None = None

# Science domain knowledge cache
#
# To support factual questions beyond technology—such as human biology or brain
# anatomy—Stage 6 loads science domain records on first use.  Records are
# read from ``brains/domain_banks/science/memory/stm/records.jsonl`` and
# cached in the module‑level ``_SCIENCE_KNOWLEDGE_CACHE``.  If loading
# fails, the cache remains ``None`` and knowledge synthesis will skip
# science lookup silently.
_SCIENCE_KNOWLEDGE_CACHE: List[Dict[str, Any]] | None = None

# Creative domain knowledge cache for story/poem retrieval.  This cache
# holds records from the ``creative`` domain bank and is initialised
# lazily on first access by :func:`_load_creative_cache`.  A `None`
# value indicates that the cache has not yet been loaded.
_CREATIVE_KNOWLEDGE_CACHE: List[Dict[str, Any]] | None = None

# ---------------------------------------------------------------------------
# Pronoun detection constants
#
# When a user sends a single‑word query that consists solely of a pronoun,
# the assistant should treat it as a reference to the previous turn rather
# than storing it as new knowledge.  Defining this set at module scope
# allows Stage 3 classifiers to quickly identify these inputs.  The list
# includes interrogative pronouns (who, what, where, etc.) as well as
# demonstratives (this, that, those) and general references (it).
PRONOUNS: Set[str] = {
    "that", "this", "it", "these", "those",
    "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how"
}

# ---------------------------------------------------------------------------
# Request patterns constants
#
# Certain open‑ended prompts that ask Maven to "tell", "describe" or
# "explain" something often lack a question mark and therefore
# bypass the standard question detection heuristics.  Define a set
# of lower‑case substrings that, when present in the user's
# utterance, should cause the NLU to treat the input as a
# request/question rather than a factual statement.  Examples
# include "tell me about", "explain", "describe" and related forms.
# The patterns are checked in order and may overlap; the first
# match triggers a REQUEST classification.
REQUEST_PATTERNS: List[str] = [
    "tell me what you know about",  # FIX: Add explicit knowledge query pattern
    "what do you know about",       # FIX: Add variant
    "tell me about",
    "tell me more about",
    "tell me",
    "explain",
    "describe",
    "what can you tell me about",
    "i want to know about",
    "teach me about",
    "learn about"                   # FIX: Add learning request pattern
]

def _load_creative_cache() -> List[Dict[str, Any]]:
    """
    Lazy‑load and return the creative domain knowledge cache.

    This helper reads the STM tier of the creative domain bank and
    caches the parsed records in the module global
    ``_CREATIVE_KNOWLEDGE_CACHE``.  Subsequent calls return the
    previously cached list.  If the domain bank file is missing or
    cannot be parsed, an empty list is returned and cached.  Errors
    during loading are suppressed to avoid crashing candidate
    generation.

    Returns:
        A list of record dictionaries from the creative domain bank.
    """
    global _CREATIVE_KNOWLEDGE_CACHE
    if _CREATIVE_KNOWLEDGE_CACHE is not None:
        return _CREATIVE_KNOWLEDGE_CACHE
    try:
        root = Path(__file__).resolve().parents[4]
        kb_path = root / "brains" / "domain_banks" / "creative" / "memory" / "stm" / "records.jsonl"
        recs: List[Dict[str, Any]] = []
        if kb_path.exists():
            with open(kb_path, "r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                        if isinstance(rec, dict):
                            recs.append(rec)
                    except Exception:
                        continue
        _CREATIVE_KNOWLEDGE_CACHE = recs
        return recs
    except Exception:
        _CREATIVE_KNOWLEDGE_CACHE = []
        return []

# Phrases that indicate a response is meta, filler or otherwise non‑informative.  If
# a cached or existing answer contains any of these substrings (case‑insensitive),
# the language brain will treat the answer as invalid and fall back to normal
# candidate generation rather than returning it verbatim.  This prevents
# statements like "I'm going to try my best" from being surfaced as factual
# answers.
BAD_ANSWER_PHRASES = [
    "i'm going to try my best",
    "i am going to try my best",
    "as an ai",
    "i don't have specific information",
    "i don't have information",
    "got it — noted",
    "got it - noted",
    "i'm also considering other possibilities",
    "i\u2019m going to try my best",  # unicode apostrophe
]

def _is_question(text: str) -> bool:
    """Return True if the provided text ends with a question mark.

    This helper is retained for backwards compatibility but a more
    sophisticated intent parser is now used to categorize input.  See
    ``parse_intent`` for human‑like intent detection.
    """
    return (text or "").strip().endswith("?")

# ---------------------------------------------------------------------
# Greeting profile and simplification helpers
#
# To make greetings socially adaptive and easy to understand (roughly
# 5th-grade reading level), the language brain loads a configurable
# greeting profile at module import.  The profile maps tone categories
# (formal, friendly, calm, excited, etc.) to lists of example phrases.
# When responding to a greeting, the candidate generator selects a
# phrase from the appropriate category based on affect and context.
# A lightweight simplifier replaces advanced vocabulary when the
# resulting greeting exceeds basic readability thresholds.

# Load greeting profile from the config directory.  The file contains
# static JSON mappings of tone names to greeting template lists.
try:
    _GREETING_PROFILE_PATH = Path(__file__).resolve().parents[4] / "config" / "greeting_profile.json"
    if _GREETING_PROFILE_PATH.exists():
        GREETING_PROFILE = json.loads(_GREETING_PROFILE_PATH.read_text(encoding="utf-8"))
    else:
        GREETING_PROFILE = {}
except Exception:
    GREETING_PROFILE = {}

def _simplify_greeting(text: str) -> str:
    """
    Simplify a greeting when it appears too complex for a young reader.

    This helper attempts to enforce a roughly 5th‑grade reading level by
    computing a Flesch‑Kincaid grade and replacing advanced vocabulary
    when necessary.  The fallback heuristic used previously—checking
    average words per sentence and maximum word length—has been
    superseded by a more direct readability calculation.  If the
    computed grade exceeds 5.0, targeted substitutions are applied.

    Args:
        text: A greeting string.

    Returns:
        A potentially simplified version of the string.
    """
    try:
        src = str(text) or ""
        # Tokenise into words consisting of alphabetic characters only
        words = re.findall(r"[A-Za-z]+", src)
        # Estimate number of sentences by punctuation markers; at least one
        sent_count = src.count(".") + src.count("!") + src.count("?")
        sent_count = sent_count if sent_count > 0 else 1

        # Estimate syllable count per word using a simple heuristic
        def _syllable_count(word: str) -> int:
            w = word.lower()
            # treat consecutive vowels as one syllable
            vowels = "aeiouy"
            count = 0
            if w and w[0] in vowels:
                count += 1
            for i in range(1, len(w)):
                if w[i] in vowels and w[i-1] not in vowels:
                    count += 1
            # Subtract a silent trailing 'e' (except words ending with 'le')
            if len(w) > 2 and w.endswith("e") and not w.endswith("le"):
                count -= 1
            return count if count > 0 else 1

        total_syllables = sum(_syllable_count(w) for w in words) or 1
        total_words = len(words) or 1
        # Compute Flesch‑Kincaid grade (approximation)
        grade = (0.39 * (total_words / sent_count)) + (11.8 * (total_syllables / total_words)) - 15.59

        # Define a mapping of advanced words to simpler alternatives.  Expand
        # synonyms beyond the previous list to cover more formal phrasing.
        replacements = {
            "assist": "help",
            "assistance": "help",
            "delighted": "happy",
            "thrilled": "excited",
            "greetings": "hi",
            "salutation": "greeting",
            "communication": "chat",
            "fascinated": "interested",
            "pleased": "happy",
            "wonderful": "great",
            "glad": "happy",
            "day": "day",  # placeholder; keep simple words unchanged
        }
        simplified = src
        # If the grade is above 5, replace complex vocabulary
        if grade > 5.0:
            for adv, simple in replacements.items():
                pattern = re.compile(r"\b" + re.escape(adv) + r"\b", flags=re.IGNORECASE)
                simplified = pattern.sub(simple, simplified)
            # Recompute grade after substitution; if still high, break down
            # excessively long sentences by adding a full stop before "and" or commas.
            # This coarse approach avoids deeply altering grammar but reduces
            # sentence length.
            try:
                # Insert a period before any occurrence of " and " if no period exists
                if "," in simplified or " and " in simplified:
                    simplified = re.sub(r"\s+and\s+", ". And ", simplified)
                    simplified = simplified.replace(",", ".")
            except Exception:
                pass
        return simplified

    except Exception:
        # If any error occurs during simplification, fall back to the original text
        return text

# ---------------------------------------------------------------------
# Phase 1 additions: lightweight NLU parser and Stage 6 generator
#
# To support the early cognition upgrade, we introduce two helper
# functions, ``nlu_parse`` and ``stage6_generate``, along with a
# minimal context window.  These helpers live at the module level so
# they can be imported directly by unit tests without invoking the
# entire language brain service.  They avoid any external
# dependencies and keep within the stdlib.

# A rolling window of the last few conversational turns.  Each entry
# is a dict with ``user`` and ``maven`` keys recording the raw input
# and the generated response.  The window is capped at five entries.
_CONTEXT_WINDOW: List[Dict[str, str]] = []

# A very simple in‑memory store for the user's identity.  When the
# system sees a phrase like "my name is ...", it records the name
# here.  Subsequent identity queries (e.g. "who am I?") will return
# this value.  This storage is purely for Phase 1 testing and does
# not persist across runs.
_USER_IDENTITY: str | None = None

# Pending identity persistence request.
# When the user introduces themselves and privacy settings require
# consent before persisting their name, the captured name is stored
# here until the user responds.  See stage6_generate for usage.
_PENDING_IDENTITY: str | None = None

# Track the last fact that was retrieved from memory, so that correction
# events ("correct", "that's right", "yes") can reinforce the confidence
# of the most recent factual answer.  Stores a dict with "key", "value",
# and "confidence" fields.
_LAST_FACT: Dict[str, Any] | None = None

def update_context(user: str, maven: str) -> None:
    """Append a (user, maven) exchange to the context window.

    Maintains a fixed‑size list of the most recent five exchanges.

    Args:
        user: The raw user utterance.
        maven: The maven response that was produced.
    """
    global _CONTEXT_WINDOW
    try:
        # append the new exchange
        _CONTEXT_WINDOW.append({"user": str(user), "maven": str(maven)})
        # keep only the last 5 entries
        if len(_CONTEXT_WINDOW) > 5:
            _CONTEXT_WINDOW = _CONTEXT_WINDOW[-5:]
    except Exception:
        # Never raise from context updates
        pass

def get_context_window() -> List[Dict[str, str]]:
    """Return a copy of the current context window.

    Returns:
        A list of the most recent (user, maven) exchanges, up to five.
    """
    # Return a shallow copy to prevent accidental mutation
    return list(_CONTEXT_WINDOW)

def nlu_parse(text: str) -> Dict[str, Any]:
    """Lightweight natural language understanding for Phase 1.

    This helper analyses a raw user utterance and extracts a handful
    of features used to guide intent classification and candidate
    generation.  It recognises modal verb questions without a
    question mark, common WH‑words (including contractions like
    "what's"), dialogue markers (yes/no/maybe/ok/sure), and simple
    identity queries such as "who am I" or "my name is X".  It also
    determines whether the utterance should be treated as a question
    and assigns a coarse intent category (question, request,
    statement, identity, confirmation).

    Args:
        text: The raw input string from the user.

    Returns:
        A dictionary containing parsed attributes.  Keys include:

        ``modal``: The modal verb (can/could/would) if present at the start.
        ``wh_type``: One of ``what``, ``where``, ``who``, ``why``, ``how``,
            ``when`` if the utterance begins with the corresponding WH
            word.  Contractions like "what's" are normalised.
        ``dialogue_marker``: A lowercase marker (yes/no/maybe/ok/sure) if
            the utterance is a simple acknowledgement or contains only
            that marker and optional filler like "later".
        ``identity_query``: True if the utterance asks about the user's
            identity (e.g. "who am I", "my name is ...", "what's my name").
        ``is_question``: True if the utterance is interpreted as a
            question (either ending with '?' or starting with a WH or
            modal phrase).
        ``intent``: A coarse category: ``identity``, ``confirmation``,
            ``request``, ``question``, or ``statement``.
    """
    result: Dict[str, Any] = {}
    try:
        src = text or ""
        # normalise whitespace and lower‑case
        lower = str(src).strip().lower()
        # Start with sensible defaults
        result.update({
            "modal": None,
            "wh_type": None,
            "dialogue_marker": None,
            "identity_query": False,
            "is_question": False,
            "intent": None,
        })
        # Extract words (alphanumerics) for marker detection
        tokens = re.findall(r"[A-Za-z']+", lower)
        # Detect simple dialogue markers at the start.  We treat a
        # single marker word (possibly followed by "later") as a
        # confirmation.  If other words appear after the first token,
        # the utterance is not considered a pure marker.  For example,
        # "maybe later" is accepted, but "ok maybe later" is not.
        if tokens:
            marker = tokens[0]
            allowed_markers = {"yes", "no", "maybe", "ok", "sure"}
            if marker in allowed_markers:
                # If there are additional tokens, ensure they are only
                # trivial fillers like "later".  Otherwise drop the marker.
                trailing = tokens[1:]
                if not trailing or all(t in {"later"} for t in trailing):
                    result["dialogue_marker"] = marker
                    result["intent"] = "confirmation"
                    result["is_question"] = False
        # Detect modal questions at the beginning (can/could/would)
        m = re.match(r"^(can|could|would)\s+you\b", lower)
        if m:
            result["modal"] = m.group(1)
            result["is_question"] = True
            # Only set intent if not already confirmation
            if result.get("intent") is None:
                result["intent"] = "request"
        # Detect yes/no questions with auxiliary verbs (do/does/can/is/are)
        # These patterns catch factual questions like "do birds fly", "can cats jump",
        # "is water wet", "are birds animals" that don't end with question marks.
        # Match patterns: "do/does/can/could/will/would/should X Y" or "is/are/was/were X Y"
        yes_no_pattern = re.match(r"^(do|does|did|can|could|will|would|should|is|are|was|were)\s+\w+", lower)
        if yes_no_pattern and not result.get("is_question"):
            result["is_question"] = True
            if result.get("intent") is None:
                result["intent"] = "simple_fact_query"
        # Normalise contractions for WH detection
        norm = lower.replace("what's", "what is").replace("where's", "where is").replace("who's", "who is").replace("when's", "when is").replace("how's", "how is").replace("why's", "why is")
        # Detect WH questions at the beginning
        for wh in ["what", "where", "who", "why", "how", "when"]:
            if norm.startswith(wh + " ") or norm == wh:
                result["wh_type"] = wh
                result["is_question"] = True
                if result.get("intent") is None:
                    result["intent"] = "question"
                break
        # Check terminal question mark
        if src.strip().endswith("?"):
            result["is_question"] = True
            if result.get("intent") is None:
                result["intent"] = "question"
        # Identity queries: phrases like "who am i", "my name is ...",
        # "am i ...?", "what's my name".  These indicate the user
        # wants to know about themselves.  If a name is provided, we
        # consider it both an identity query and a statement, not a
        # question.
        try:
            # explicit patterns with optional punctuation
            if re.search(r"\bwho\s+am\s+i\b", lower):
                result["identity_query"] = True
                # override intent
                result["intent"] = "identity"
            elif re.search(r"\bwhat('?s| is)\s+my\s+name\b", norm):
                result["identity_query"] = True
                result["intent"] = "identity"
            elif re.search(r"\bam\s+i\b", lower):
                # e.g. "am i a robot" – treat as identity question
                result["identity_query"] = True
                # keep the intent as question if previously set
                if result.get("intent") is None:
                    result["intent"] = "identity"
            # Pattern: "my name is" – treat as an identity query even if no name follows
            if re.search(r"\bmy\s+name\s+is\b", lower):
                result["identity_query"] = True
                result["intent"] = "identity"
            # Pattern: "i am [name]" or "i'm [name]" – identity declaration statement
            if re.search(r"\bi\s+am\s+[a-zA-Z]+\b", lower) or re.search(r"\bi'?m\s+[a-zA-Z]+\b", lower):
                # But skip "i am happy", "i am sad" etc (emotions)
                # Simple check: if followed by a common adjective, don't treat as identity
                if not re.search(r"\bi\s+(?:am|'?m)\s+(happy|sad|excited|tired|hungry|cold|hot|busy|free)\b", lower):
                    result["identity_query"] = True
                    result["intent"] = "identity"
            # If a name is provided after "my name is" or "i am", it will be captured later in the stage6 handler
        except Exception:
            pass
        # Fallback intent if still unset
        if result.get("intent") is None:
            if result.get("is_question"):
                result["intent"] = "question"
            else:
                result["intent"] = "statement"
    except Exception:
        # On any unexpected error, return a safe default
        result = {
            "modal": None,
            "wh_type": None,
            "dialogue_marker": None,
            "identity_query": False,
            "is_question": False,
            "intent": "statement",
        }
    return result

# === Stage-6 gating and selection ===
# Edit-Only: no top-level imports and no side effects.
# All imports are local to functions to avoid import-time effects.

# Gate 1: memory pass check
def _passed_memory(ctx: dict) -> bool:
    """
    Return True if working/contextual memory prerequisites are satisfied.
    Allowed to read from ctx only. No filesystem, no time, no I/O.
    """
    # Minimal criterion: have a normalized prompt and a scratch space.
    return bool(ctx.get("prompt")) and isinstance(ctx.get("scratch"), dict)

# Gate 2: knowledge gap assessment
def knowledge_gap(ctx: dict) -> bool:
    """
    Return True when the request likely needs external knowledge or long-form synthesis.
    Keep strictly heuristic and cheap; do not call out.
    """
    p = (ctx.get("prompt") or "").lower()
    # Minimal heuristic: presence of open-ended cues
    triggers = ("explain", "why", "compare", "tradeoffs", "long answer")
    return any(t in p for t in triggers) or len(p) > 280

# Gate 3: governance permit
def _governance_permit_generate(ctx: dict) -> bool:
    """
    Return True if content category is permitted under governance/policy.
    Keep purely local and declarative; no logging, no network, no time.
    """
    # Minimal: block empty and known disallowed flags in ctx
    if not ctx.get("prompt"):
        return False
    if ctx.get("blocked_category") is True:
        return False
    return True

# Gate 4: allow LLM fallback
def allow_llm_fallback(ctx: dict) -> bool:
    """
    Return True if we're allowed to escalate to LLM after template/heuristic fail.
    """
    # Minimal: opt-out flag disables LLM; otherwise enabled.
    return not ctx.get("disable_llm", False)

# Selection: Template → Heuristic → LLM
def _try_template(ctx: dict):
    """
    Return a response dict or None.
    Cheap, deterministic templates for frequent intents.
    """
    # Local import to avoid top-level dependency
    from typing import Optional  # type: ignore
    p = (ctx.get("prompt") or "").strip()
    # Example minimal templates; extend as needed
    if p.lower().startswith(("define ", "what is ")):
        term = p.split(" ", 1)[1] if " " in p else p
        return {"mode": "template", "text": f"{term}: concise definition not found in template bank."}
    return None

def _try_heuristic(ctx: dict):
    """
    Return a response dict or None.
    Fast local reasoning with lightweight rules. No I/O, no time.
    """
    # Example: compress bullet synthesis if user asks for summary
    p = (ctx.get("prompt") or "").lower()
    if "summarize" in p or "tl;dr" in p:
        src = ctx.get("input_text") or ""
        if src:
            lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
            head = lines[:5]
            return {"mode": "heuristic", "text": " • " + "\n • ".join(head)}
    return None

def _llm_generate(ctx: dict):
    """
    Return a response dict. This is the only place where an LLM call may occur.
    Keep it wired, but the actual call should be injected upstream by the caller.
    This function must not import a client at module import time.
    """
    llm = ctx.get("llm_callable")  # injected function: (prompt:str, **kw)->str
    if not callable(llm):
        # Hard fallback: emit a deterministic stub; do not raise.
        return {"mode": "llm_stub", "text": "[LLM unavailable]"}
    prompt = ctx.get("prompt") or ""
    params = ctx.get("llm_params") or {}
    out = llm(prompt, **params)
    return {"mode": "llm", "text": out}

def stage6_generate(ctx: dict) -> dict:
    """
    Stage-6 entrypoint: enforce gates, then Template→Heuristic→LLM.
    Public signature must remain stable.
    """
    # Trace that we reached stage6_generate (Phase C cleanup diagnostics)
    mid_for_trace = ctx.get("_mid", "unknown")
    try:
        import sys
        if "brains.cognitive.routing_diagnostics" in sys.modules:
            tracer = sys.modules["brains.cognitive.routing_diagnostics"].tracer
            RouteType = sys.modules["brains.cognitive.routing_diagnostics"].RouteType
            if tracer and RouteType:
                tracer.record_route(mid_for_trace, RouteType.STAGE6_GENERATE, {"gates_active": True})
    except Exception:
        pass

    # Enforce gates
    if not _passed_memory(ctx):
        try:
            from brains.cognitive.routing_diagnostics import tracer, RouteType  # type: ignore
            if tracer and RouteType:
                mid = ctx.get("_mid") or "unknown"
                tracer.record_route(mid, RouteType.BLOCKED_MEMORY, {})
        except Exception:
            pass
        return {"mode": "blocked", "reason": "memory_gate", "text": ""}
    if not _governance_permit_generate(ctx):
        try:
            from brains.cognitive.routing_diagnostics import tracer, RouteType  # type: ignore
            if tracer and RouteType:
                mid = ctx.get("_mid") or "unknown"
                tracer.record_route(mid, RouteType.BLOCKED_GOVERNANCE, {})
        except Exception:
            pass
        return {"mode": "blocked", "reason": "governance", "text": ""}

    # Selection path
    resp = _try_template(ctx)
    if resp:
        try:
            from brains.cognitive.routing_diagnostics import tracer, RouteType  # type: ignore
            if tracer and RouteType:
                mid = ctx.get("_mid") or "unknown"
                tracer.record_route(mid, RouteType.TEMPLATE_MATCH, {})
        except Exception:
            pass
        return resp

    resp = _try_heuristic(ctx)
    if resp:
        try:
            from brains.cognitive.routing_diagnostics import tracer, RouteType  # type: ignore
            if tracer and RouteType:
                mid = ctx.get("_mid") or "unknown"
                tracer.record_route(mid, RouteType.HEURISTIC_MATCH, {})
        except Exception:
            pass
        return resp

    if knowledge_gap(ctx) and allow_llm_fallback(ctx):
        try:
            from brains.cognitive.routing_diagnostics import tracer, RouteType  # type: ignore
            if tracer and RouteType:
                mid = ctx.get("_mid") or "unknown"
                tracer.record_route(mid, RouteType.LLM_FALLBACK, {})
        except Exception:
            pass
        return _llm_generate(ctx)

    # Final deterministic fallback to keep behavior predictable
    return {"mode": "fallback", "text": ""}

# Define a set of greeting phrases recognised by the language brain.  When the
# user input matches one of these phrases exactly (case insensitive), the
# intent is classified as SOCIAL and treated specially.  Additional
# translations of common greetings can be added here without affecting
# other parsing behaviour.
GREETINGS = {
    "hi", "hello", "hey", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "greetings", "howdy", "hola", "bonjour"
}

def _apply_relationship_overrides(intent_info: Dict[str, Any], text: str) -> None:
    """Apply relationship update/query detection overrides to intent_info.

    Detect when users express relationship facts like "we are friends"
    or "we're not friends" and when they query relationship status like
    "are we friends?". These should be handled specially to store/retrieve
    relationship facts from memory.

    Args:
        intent_info: The intent dictionary to update in-place
        text: The user input text to analyze
    """
    try:
        _nl_rel = text.lower().strip()
        # Remove trailing punctuation for pattern matching
        _nl_rel_clean = _nl_rel.rstrip("?!.,")

        # Positive relationship update patterns
        positive_patterns = [
            "we are friends",
            "we're friends",
            "you are my friend",
            "you're my friend",
        ]

        # Negative relationship update patterns
        negative_patterns = [
            "we are not friends",
            "we're not friends",
            "you are not my friend",
            "you're not my friend",
        ]

        # Relationship query patterns
        query_patterns = [
            "are we friends",
            "are you my friend",
        ]

        # Check for positive relationship update
        if any(p in _nl_rel_clean for p in positive_patterns):
            intent_info.update({
                "intent": "relationship_update",
                "relationship_kind": "friend_with_system",
                "relationship_value": True,
                "storable": True,
                "storable_type": "RELATIONAL",
                "type": "STATEMENT",
                "is_question": False,
                "is_command": False,
                "is_request": False,
                "is_statement": True,
            })
        # Check for negative relationship update
        elif any(p in _nl_rel_clean for p in negative_patterns):
            intent_info.update({
                "intent": "relationship_update",
                "relationship_kind": "friend_with_system",
                "relationship_value": False,
                "storable": True,
                "storable_type": "RELATIONAL",
                "type": "STATEMENT",
                "is_question": False,
                "is_command": False,
                "is_request": False,
                "is_statement": True,
            })
        # Check for relationship query
        elif any(p in _nl_rel_clean for p in query_patterns):
            intent_info.update({
                "intent": "relationship_query",
                "relationship_kind": "friend_with_system",
                "storable": False,
                "storable_type": "RELATIONAL",
                "type": "QUESTION",
                "is_question": True,
                "is_command": False,
                "is_request": False,
                "is_statement": False,
                "skip_memory_search": False,
            })
    except Exception:
        # If relationship detection fails, continue with normal classification
        pass

def _parse_intent(text: str) -> Dict[str, Any]:
    """Infer the communicative intent of the input text.

    This function attempts to classify a given user input into one of
    several categories: QUESTION, COMMAND, REQUEST, SPECULATION, FACT
    (statement), or UNKNOWN.  It also indicates whether the text should
    be considered storable as a fact in memory and applies a confidence
    penalty for speculative statements.  The logic mirrors how a human
    might decide what to remember: information requests and commands are
    not facts to be stored, while declarative statements and speculative
    ideas may be considered knowledge (the latter with reduced
    confidence).

    Args:
        text: The raw user input string.

    Returns:
        A dictionary with keys:
            type: The intent category (QUESTION, COMMAND, REQUEST, SPECULATION, FACT, UNKNOWN)
            storable: Whether this input should be stored as knowledge
            confidence_penalty: A float penalty applied to the base confidence when storing
            is_question, is_command, is_request, is_statement: Booleans for convenience
    """
    original = str(text or "")
    stripped = original.strip()
    lower = stripped.lower()
    normalized_lower = lower
    # Check for greetings first.  Exact matches trigger the SOCIAL intent.
    if normalized_lower in GREETINGS:
        return {
            "type": "SOCIAL",
            "storable": False,
            "confidence_penalty": 0.0,
            "is_question": False,
            "is_command": False,
            "is_request": False,
            "is_statement": False
        }
    # Extended greeting detection: if the input begins with a known greeting
    # followed by an apostrophe or other punctuation, treat it as SOCIAL.  This
    # handles cases like "hello'how are you" where the greeting token is
    # adjacent to subsequent words without a space.  Only match when the
    # punctuation immediately follows the full greeting phrase.
    try:
        for _g in GREETINGS:
            # Normalise multi‑word greetings by stripping trailing spaces for
            # prefix matching.  Only consider if the text is longer than the
            # greeting itself.
            if lower.startswith(_g) and len(lower) > len(_g):
                next_char = lower[len(_g)]
                # Treat apostrophe and common punctuation as separators.  A
                # separator indicates the greeting is complete and should
                # trigger the SOCIAL intent.  Examples include "hello'how",
                # "hi, how" or "hey! there".
                if next_char in {"'", ",", "!", "?", ":", ";", "-"}:
                    return {
                        "type": "SOCIAL",
                        "storable": False,
                        "confidence_penalty": 0.0,
                        "is_question": False,
                        "is_command": False,
                        "is_request": False,
                        "is_statement": False
                    }
    except Exception:
        # Ignore errors in extended greeting detection and continue
        pass
    # Identity queries detection.  Users sometimes ask about the agent's
    # identity without a trailing question mark.  Examples include
    # "who are you", "what is your name" or "tell me about yourself".
    # When such patterns are detected anywhere in the input, classify
    # the intent as "self_description_request" with type QUESTION.
    # This ensures that self‑definition queries are routed through the
    # proper identity response handler in Stage 6 via _build_self_description().
    # This check MUST occur before REQUEST_PATTERNS to prevent queries like
    # "tell me in your own words who you are" from being intercepted by the
    # generic "tell me" pattern.  See upgrade notes for details on self‑model
    # improvements and pattern matching order.
    try:
        identity_patterns = [
            "who are you",
            "what is your name",
            "what's your name",
            "tell me about yourself",
            "who you are",
            "are you maven",
            "tell me who you are",
            "describe yourself",
            "what are you really",
            "tell me in your own words who you are",
            "tell me in your own words what you are",
            "describe yourself in your own words",
            "in your own words who are you",
            "in your own words what are you",
            "what are you in your own words",
            "what is your own description",
            "give me your own description",
            # Extended self-knowledge patterns (Step B requirement)
            "what do you know about yourself",
            "what do you know about your own code",
            "what do you know about your systems",
            "how do you work",
            "where do you run",
            "are you an llm"
        ]
        for pat in identity_patterns:
            if pat in lower:
                return {
                    "type": "QUESTION",
                    "intent": "self_description_request",
                    "storable": False,
                    "confidence_penalty": 0.0,
                    "is_question": True,
                    "is_command": False,
                    "is_request": False,
                    "is_statement": False
                }
    except Exception:
        # If detection fails, fall back to normal question detection
        pass
    # Detect explicit knowledge requests via certain phrasal patterns (e.g. "tell me about ...", "explain ...").
    # These phrases may not contain a question mark and would otherwise default to a FACT.  If any
    # pattern from REQUEST_PATTERNS is present, classify the input as a REQUEST and also mark it as
    # a question to encourage memory retrieval or LLM fallback.  See upgrade notes for details.
    try:
        for _rp in REQUEST_PATTERNS:
            if _rp in lower:
                return {
                    "type": "REQUEST",
                    "storable": False,
                    "confidence_penalty": 0.0,
                    "is_question": True,
                    "is_command": False,
                    "is_request": True,
                    "is_statement": False
                }
    except Exception:
        # On any error, continue with normal parsing
        pass

    words = lower.split()
    first_word = words[0] if words else ""
    # Explicit question markers
    # Include common contractions (e.g. "what's", "who's", etc.) in the set of
    # question words.  Without these, inputs like "what's my name" will not
    # trigger the question intent because the first token is "what's" rather than
    # "what".  See https://github.com/maven-nlu/upgrade-notes for details on
    # misclassification of identity queries.  Include both apostrophised and
    # non‑apostrophised forms to handle user typos (e.g. "whats my name").
    question_words = [
        "what", "why", "how", "when", "where", "who", "which",
        "what's", "whats",
        "who's", "whos",
        "where's", "wheres",
        "why's", "whys",
        "how's", "hows",
        "when's", "whens",
    ]

    # ---
    # Detect common greetings/acknowledgements and treat them as non‑storable chat.
    #
    # Historically, all inputs that were not questions, commands or speculation
    # defaulted to FACT, which led to salutations like "hi" or "good" being
    # stored as knowledge.  To more closely model natural conversation, we
    # explicitly detect short greeting phrases and simple acknowledgements and
    # return a non‑storable UNKNOWN intent instead.  This ensures that
    # conversational chit‑chat does not pollute the factual memory store.  The
    # list below is intentionally conservative and covers common greetings,
    # farewells and courtesy phrases.  Additional phrases can be added as
    # needed but should not overlap with legitimate facts.
    greetings = {
        "hi", "hello", "hey", "good morning", "good evening", "good night",
        "goodbye", "bye", "thanks", "thank you", "ok", "okay", "good", "fine",
        # Treat short exclamations and acknowledgements like "nice", "cool" and "wow" as
        # non‑informational feedback rather than factual statements.  This prevents them
        # from being classified as FACT and stored in the knowledge base.  See issue
        # description for details on misclassification of inputs like "NICE".
        "nice", "cool", "wow"
    }
    # If the entire input matches a greeting exactly (case insensitive),
    # classify as UNKNOWN (non‑storable) rather than FACT.
    if lower in greetings:
        return {
            "type": "UNKNOWN",
            "storable": False,
            "confidence_penalty": 0.0,
            "is_question": False,
            "is_command": False,
            "is_request": False,
            "is_statement": False
        }

    # --------------------------------------------------------------------
    # Continuation detection using continuation_helpers
    #
    # Inputs such as "more", "more about cats", "anything else" and "what else"
    # are follow‑up queries that refer to the previous context.  Treat these
    # as questions (non‑storable) to ensure that the memory librarian searches
    # for additional information on the last topic instead of storing them as
    # facts.  Use the shared continuation_helpers module for standardized
    # detection across all cognitive brains.
    try:
        # Use the standardized is_continuation helper
        if is_continuation(text):
            # Extract the specific type of continuation (expansion, clarification, etc.)
            continuation_intent = extract_continuation_intent(text)

            return {
                "type": "QUESTION",
                "storable": False,
                "confidence_penalty": 0.0,
                "is_question": True,
                "is_command": False,
                "is_request": False,
                "is_statement": False,
                "is_continuation": True,
                "continuation_intent": continuation_intent
            }
    except Exception:
        # Fall back to basic detection if helpers fail
        try:
            _cql = lower.strip()
        except Exception:
            _cql = ""
        cont_keys = {"more", "anything else", "any thing else", "what else", "tell me more", "tell me more about"}
        # Detect continuation phrases
        if _cql.startswith("more about ") or _cql in cont_keys or (_cql.startswith("more ") and len(_cql.split()) <= 2):
            return {
                "type": "QUESTION",
                "storable": False,
                "confidence_penalty": 0.0,
                "is_question": True,
                "is_command": False,
                "is_request": False,
                "is_statement": False,
                "is_continuation": True
            }

    # ------------------------------------------------------------------------
    # Detect simple emotional statements.  If the user expresses that they are
    # feeling a strong emotion (e.g. "I am sad", "I'm happy", "I feel upset"),
    # classify the intent as EMOTION.  This allows Stage 6 to craft an
    # empathetic response instead of storing the text as a fact.  Only clear
    # statements about personal feelings are matched to avoid false positives.
    # Two sentiment classes are recognised: positive and negative.  Additional
    # keywords can be appended as needed.
    # The detection logic looks for a first person subject ("I am", "I'm",
    # "I feel") followed by up to two words and then a sentiment keyword.  The
    # regex ensures we match full words to avoid partial matches (e.g.
    # "dreadful" should not match "sad").
    neg_terms = [
        "sad", "upset", "depressed", "angry", "frustrated", "stressed",
        "anxious", "worried", "fearful", "lonely", "miserable", "tired"
    ]
    pos_terms = [
        "happy", "excited", "thrilled", "glad", "pleased", "proud",
        "grateful", "thankful", "relieved", "joyful", "content"
    ]
    # Build regex patterns for emotion detection
    try:
        # Join terms into regex groups.  Use double braces to escape the
        # repetition quantifier so that f‑string formatting does not
        # interpret ``{0,2}`` as Python code.  Patterns match phrases
        # like "i am very sad", "i'm extremely happy", "i feel upset".
        neg_re = r"|".join([re.escape(t) for t in neg_terms])
        pos_re = r"|".join([re.escape(t) for t in pos_terms])
        # Allow one optional word between "I" and the verb (e.g. "I really am")
        pattern_neg = rf"\b(i(?:\s+am|'?m|\s+feel)\s+(?:[\w']+\s+){{0,2}})(?:{neg_re})\b"
        pattern_pos = rf"\b(i(?:\s+am|'?m|\s+feel)\s+(?:[\w']+\s+){{0,2}})(?:{pos_re})\b"
        if re.search(pattern_neg, lower):
            return {
                "type": "EMOTION",
                "valence": "negative",
                "storable": False,
                "confidence_penalty": 0.0,
                "is_question": False,
                "is_command": False,
                "is_request": False,
                "is_statement": True
            }
        if re.search(pattern_pos, lower):
            return {
                "type": "EMOTION",
                "valence": "positive",
                "storable": False,
                "confidence_penalty": 0.0,
                "is_question": False,
                "is_command": False,
                "is_request": False,
                "is_statement": True
            }
    except Exception:
        pass
    # Expanded list of imperative verbs to better detect commands and requests.
    # This list includes common verbs used to ask the system to perform an
    # action (e.g. "give me", "provide", "make", "generate", "create", "connect").
    # It is intentionally conservative to avoid misclassifying plain facts
    # starting with a verb (e.g. "Dogs bark.").  Additional verbs may be
    # appended here as needed.
    imperative_verbs = [
        "explain", "tell", "show", "describe", "list", "help",
        "give", "provide", "make", "generate", "create", "connect",
        # Treat explicit speech acts like "say" as imperatives so that
        # utterances such as "say hello" or "say thanks" are routed
        # through the command handler rather than misclassified as
        # declarative statements.  See the behaviour layer upgrade notes.
        "say",
        # Additional coding/creative verbs.  These are common in
        # programming requests and instruct Maven to perform a creative or
        # constructive action rather than stating a fact.  Including them
        # here ensures that inputs like "write a python function" or
        # "code a loop" are classified as requests instead of factual
        # statements.
        "write", "code", "program", "solve",
        # Added verbs for summarisation, translation and comparison.  These
        # verbs cue the parser that the user is asking for an action
        # (summarise this, translate this, compare these) rather than
        # stating a fact.  Without these, queries like "summarize the
        # Python list data type" were misclassified as statements and
        # ignored.
        "summarize", "summarise", "translate", "compare", "contrast", "compose",
        # Include 'continue' as an imperative verb so that follow‑up requests like
        # "continue the story" or "continue explaining" are classified as
        # actionable requests rather than statements.  This helps Stage 3 treat
        # them as commands rather than facts.
        "continue"
    ]
    hedge_words = ["maybe", "perhaps", "possibly", "i think", "seems like", "might be"]
    # Commands prefixed by CLI style
    if stripped.startswith("--") or stripped.startswith("/"):
        return {
            "type": "COMMAND",
            "storable": False,
            "confidence_penalty": 0.0,
            "is_question": False,
            "is_command": True,
            "is_request": False,
            "is_statement": False
        }
    # ------------------------------------------------------------------
    # Relational queries detection
    #
    # The language brain historically failed to treat certain relational
    # questions like "are we friends" as interrogative.  Such phrases
    # often omit a trailing question mark and do not begin with a
    # canonical question word, so they bypass the default heuristics.  To
    # capture these cases, explicitly check for common friendship and
    # alliance patterns and classify them as questions.  The detection
    # operates on lower‑cased input and matches substrings to avoid
    # misclassifying unrelated sentences.  Additional relational forms can
    # be added here as needed.
    try:
        relation_phrases = [
            "are we friends",
            "are you my friend",
            "are you a friend",
            "are we allies",
            "are you my ally",
            "are you an ally",
            "do you consider me a friend",
            "do you trust me",
            "do you consider me your friend",
            "are we partners",
            "are we buddies",
            # Expressions referencing prior statements
            "you said i'm your friend",
            "you said i am your friend",
            "you said im your friend",
        ]
        for _rp in relation_phrases:
            if _rp in lower:
                return {
                    "type": "QUESTION",
                    "storable": False,
                    "confidence_penalty": 0.0,
                    "is_question": True,
                    "is_command": False,
                    "is_request": False,
                    "is_statement": False
                }
    except Exception:
        # On error, ignore and proceed to other heuristics
        pass

    # ------------------------------------------------------------------
    # Research intent detection (deep research mode triggers)
    #
    # Detect explicit research requests so they can be routed to the
    # research_manager brain instead of generic language handling. These
    # phrases are treated as commands/requests and are not stored as
    # facts. Depth level is inferred from the phrasing.
    try:
        research_patterns = [
            ("deep research", 3),
            ("research", 2),
            ("learn deeply about", 3),
            ("learn about", 2),
            ("study", 2),
            ("run a research task on", 2),
        ]
        for phrase, depth_val in research_patterns:
            if lower.startswith(phrase):
                topic_text = stripped[len(phrase):].strip(" .,:;?\"")
                return {
                    "type": "COMMAND",
                    "storable": False,
                    "storable_type": "COMMAND",
                    "confidence_penalty": 0.0,
                    "is_question": False,
                    "is_command": True,
                    "is_request": True,
                    "is_statement": False,
                    "intent": "research_request",
                    "research_topic": topic_text,
                    "research_depth": depth_val,
                    "deliverable": "detailed_report",
                }

        followup_patterns = [
            "what did you learn about",
            "what have you learned about",
            "summarize your research on",
            "tell me what you learned about",
            "what do you know about now",
        ]
        for fp in followup_patterns:
            if fp in lower:
                topic_text = lower.split(fp, 1)[1].strip(" .,:;?\"")
                return {
                    "type": "QUESTION",
                    "storable": False,
                    "storable_type": "QUESTION",
                    "confidence_penalty": 0.0,
                    "is_question": True,
                    "is_command": False,
                    "is_request": True,
                    "is_statement": False,
                    "intent": "research_followup",
                    "research_topic": topic_text or stripped,
                    "deliverable": "stored_report",
                }
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Direct speech act: a stand‑alone "say" command.  When the
    # utterance begins with the verb "say", treat it as a command
    # instructing Maven to speak a phrase.  This forces the query
    # through the command router rather than being interpreted as a
    # request for information.  Only apply this rule when "say" is
    # indeed the first word; more complex sentences are handled by
    # other heuristics above.
    try:
        if first_word == "say":
            return {
                "type": "COMMAND",
                "storable": False,
                "confidence_penalty": 0.0,
                "is_question": False,
                "is_command": True,
                "is_request": False,
                "is_statement": False
            }
    except Exception:
        # On error, ignore and fall through
        pass

    # ------------------------------------------------------------------
    # Imperative‑linguistic patterns (second‑person verbs)
    #
    # If the user issues a direct instruction addressed to Maven in
    # the form "you <verb> ..." (e.g. "you say hello"), classify
    # this as a command.  This captures behavioural directives that
    # should trigger an action rather than be stored as facts.  Only
    # recognise a narrow set of verbs associated with simple social
    # acts to avoid misclassifying descriptive statements like
    # "you are smart".  When such a pattern is detected, mark the
    # utterance as non‑storable and signal that it is a command.
    try:
        # Normalise and split into tokens
        parts = lower.split()
        if parts and parts[0] == "you" and len(parts) > 1:
            # List of second‑person verbs considered imperative
            social_verbs = {"say", "tell", "greet", "greet", "wish", "speak"}
            if parts[1] in social_verbs:
                return {
                    "type": "COMMAND",
                    "storable": False,
                    "confidence_penalty": 0.0,
                    "is_question": False,
                    "is_command": True,
                    "is_request": False,
                    "is_statement": False
                }
    except Exception:
        # Swallow any errors and continue with default parsing
        pass

    # ------------------------------------------------------------------
    # Modal request detection
    #
    # Polite requests often begin with phrases like "can you", "could you",
    # "would you" or "will you".  Historically these were misclassified
    # as factual statements because the first word (e.g. "can") is not
    # included in the imperative verb list and there may be no trailing
    # question mark.  Detect these modal patterns and classify them as
    # requests to ensure the system attempts to perform an action or
    # generate a response rather than storing the text as a fact.  See
    # upgrade notes on NLU misclassification for examples.
    try:
        modal_phrases = [
            "can you", "could you", "would you", "will you",
            "can u", "could u", "would u", "will u"
        ]
        for phrase in modal_phrases:
            if lower.startswith(phrase):
                return {
                    "type": "REQUEST",
                    "storable": False,
                    "confidence_penalty": 0.0,
                    "is_question": False,
                    "is_command": False,
                    "is_request": True,
                    "is_statement": False
                }
    except Exception:
        # If detection fails, fall through to other heuristics
        pass

    # ------------------------------------------------------------------
    # Yes/no questions with auxiliary verbs (e.g. "do birds fly",
    # "can cats jump", "is water wet")
    #
    # Many factual questions use auxiliary verb forms (do/does/can/is/
    # are/etc.) and omit the question mark.  Without explicit detection,
    # these are misclassified as statements.  Match common auxiliary
    # patterns at the start of the utterance and classify them as
    # questions.  This enables proper handling of simple fact queries
    # like "do birds fly" or "can cats jump".
    try:
        yes_no_pattern = re.match(
            r"^(do|does|did|can|could|will|would|should|is|are|was|were)\s+\w+",
            lower
        )
        if yes_no_pattern:
            return {
                "type": "QUESTION",
                "storable": False,
                "confidence_penalty": 0.0,
                "is_question": True,
                "is_command": False,
                "is_request": False,
                "is_statement": False
            }
    except Exception:
        # On error, fall through to other detection logic
        pass

    # Questions (explicit question mark or question words)
    if stripped.endswith("?") or first_word in question_words:
        return {
            "type": "QUESTION",
            "storable": False,
            "confidence_penalty": 0.0,
            "is_question": True,
            "is_command": False,
            "is_request": False,
            "is_statement": False
        }
    # Requests (imperative verbs or polite request).  Detect polite phrases
    # even if "please" appears after the first word (e.g. "Could you please...").
    if first_word in imperative_verbs or lower.startswith("please") or " please " in lower:
        # Before returning a generic REQUEST, check if the request appears to be
        # under‑specified.  If the input lacks essential details (e.g. "plan a trip"
        # without a destination or date), treat this as a clarification intent
        # so that Stage 6 can ask a follow‑up question.  Simple heuristic patterns
        # catch common ambiguous requests.  Additional patterns may be added
        # here for other under‑specified commands.
        clarify_patterns = [
            # Trip/vacation planning without destination or timeframe
            r"\bplan\s+(?:a\s+)?(?:trip|vacation|holiday|journey)\b",
            # Scheduling a meeting/appointment without a time
            r"\bschedule\s+(?:a\s+)?(?:meeting|appointment)\b",
            # Booking a restaurant/table/reservation without location or time
            r"\bbook\s+(?:a\s+)?(?:restaurant|table|reservation)\b",
            # Organising an event/party without details
            r"\b(?:organize|organise|plan)\s+(?:a\s+)?(?:party|event)\b"
        ]
        try:
            for _cp in clarify_patterns:
                if re.search(_cp, lower):
                    return {
                        "type": "CLARIFICATION",
                        "storable": False,
                        "confidence_penalty": 0.0,
                        "is_question": False,
                        "is_command": False,
                        "is_request": True,
                        "is_statement": False
                    }
        except Exception:
            # If regex fails, ignore and fall through to normal request
            pass
        return {
            "type": "REQUEST",
            "storable": False,
            "confidence_penalty": 0.0,
            "is_question": False,
            "is_command": False,
            "is_request": True,
            "is_statement": False
        }
    # Speculation / hedging language
    if any(hedge in lower for hedge in hedge_words):
        return {
            "type": "SPECULATION",
            "storable": True,
            "confidence_penalty": 0.3,
            "is_question": False,
            "is_command": False,
            "is_request": False,
            "is_statement": True
        }
    # Declarative fact (default if intent not otherwise detected)
    if original:
        return {
            "type": "FACT",
            "storable": True,
            "confidence_penalty": 0.0,
            "is_question": False,
            "is_command": False,
            "is_request": False,
            "is_statement": True
        }
    # Unknown or empty
    return {
        "type": "UNKNOWN",
        "storable": False,
        "confidence_penalty": 0.0,
        "is_question": False,
        "is_command": False,
        "is_request": False,
        "is_statement": False
    }

# Extract a subject from speculative content for human‑like responses
def _extract_subject(text: str) -> str:
    """Return a simplified subject from a speculative or theoretical statement.

    This helper strips common hedging phrases (e.g., 'maybe', 'perhaps', 'possibly',
    'i think', 'seems like', etc.) and punctuation from the beginning of the sentence,
    then returns the remaining phrase.  If nothing meaningful remains, the
    entire content (lowercase) is returned.

    Args:
        text: The original user input.

    Returns:
        A simplified subject phrase for use in speculative acknowledgements.
    """
    if not text:
        return ""
    try:
        lower = text.strip().lower()
    except Exception:
        return str(text)
    # Remove leading hedging phrases
    hedges = [
        "maybe", "perhaps", "possibly", "i think", "seems like", "it seems", "might be"
    ]
    for hedge in hedges:
        if lower.startswith(hedge):
            lower = lower[len(hedge):].strip()
            break
    # Strip leading punctuation and filler words
    lower = lower.lstrip(",;:. -")
    # Remove trailing punctuation
    lower = lower.rstrip(".?! ")
    return lower

# -----------------------------------------------------------------------------
# Personal classification helpers
#
# These helpers detect whether a given text should be treated as an
# emotional statement, opinion, or speculation beyond the simple intent
# categories.  They look for first‑person pronouns and common emotion words
# and return overrides for the ``storable_type``, ``storable`` flag,
# ``confidence_penalty`` and suggested ``target_bank``.  This allows
# downstream components like reasoning to handle subjective input more
# appropriately.

def _classify_personal(text: str, intent_type: str) -> Dict[str, Any] | None:
    """Classify first‑person and emotional content.

    Args:
        text: The original user input.
        intent_type: The coarse intent from ``_parse_intent`` (e.g. FACT, SPECULATION).

    Returns:
        A dictionary with possible overrides for storable_type, storable,
        confidence_penalty and target_bank, or None if no override is needed.
    """
    try:
        lower = (text or "").strip().lower()
    except Exception:
        return None
    # List of first‑person indicators
    first_person = ["i'm", "i am", "i feel", "i think", "i believe", "my"]
    # List of common emotion words
    emotion_words = [
        "worried", "happy", "sad", "angry", "scared", "excited",
        "nervous", "anxious", "afraid", "concerned", "fearful", "depressed",
        "joyful", "delighted", "frustrated", "relieved"
    ]
    # If there is no first‑person language, do not override
    if not any(fp in lower for fp in first_person):
        return None
    # If emotion words appear, classify as EMOTION
    if any(emo in lower for emo in emotion_words):
        return {
            "storable_type": "EMOTION",
            "storable": True,
            # High penalty because emotional content is subjective and low confidence
            "confidence_penalty": 0.5,
            "target_bank": "personal"
        }
    # Otherwise classify as OPINION if first person but not explicitly emotional
    return {
        "storable_type": "OPINION",
        "storable": True,
        # Opinions are subjective: apply a moderate penalty
        "confidence_penalty": 0.3,
        "target_bank": "personal"
    }

# -----------------------------------------------------------------------------
# Conversational classification helper
#
# In addition to the general intent parser and personal classifier above, the
# language brain recognises a handful of simple conversational utterances.  These
# phrases do not contain factual content and should not be stored as knowledge.
# By detecting acknowledgements, greetings, expressions of gratitude and
# preferences, we can route them to dedicated response generators and avoid
# polluting the knowledge base with chit‑chat.  The returned dictionary may
# override ``storable_type``, ``storable`` and optionally specify a
# ``target_bank`` or ``confidence_penalty``.
def classify_storable_type(text: str) -> Dict[str, Any] | None:
    try:
        lower = (text or "").lower().strip()
    except Exception:
        return None
    # If the input is a single‑word pronoun, treat it as a context‑dependent
    # query rather than a factual statement.  These pronoun queries
    # reference the previous dialogue turn (e.g. "that", "this", "what") and
    # should not be stored in memory.  Mark them as requiring context.
    try:
        tokens = lower.split()
        if len(tokens) == 1 and tokens[0] in PRONOUNS:
            return {
                "storable_type": "PRONOUN_QUERY",
                "storable": False,
                "requires_context": True
            }
    except Exception:
        # Fall through on tokenisation error and continue with other rules
        pass
    # Acknowledgments: recognise brief phrases indicating understanding or
    # acceptance of prior information.  These should not be stored and carry
    # no confidence penalty.
    acknowledgments = [
        "okay", "ok", "alright", "got it", "i see",
        "that's okay", "thats okay", "no problem",
        "sure", "fine", "sounds good"
    ]
    # Match exact or substring to catch variants like "thanks, got it".
    for ack in acknowledgments:
        if lower == ack or ack in lower:
            return {
                "storable_type": "ACKNOWLEDGMENT",
                "storable": False,
                "confidence_penalty": 0.0
            }
    # Greetings: short salutations and farewells.  These are handled by the
    # existing greeting responder logic and should not be stored.
    greetings = ["hello", "hi", "hey", "goodbye", "bye"]
    if lower in greetings:
        return {
            "storable_type": "GREETING",
            "storable": False
        }
    # Thanks: recognise gratitude and respond politely.  Do not store.
    if "thank" in lower or "thanks" in lower:
        return {
            "storable_type": "THANKS",
            "storable": False
        }
    # Preferences: statements of liking or loving something should be stored
    # in the personal bank.  Identify simple expressions like "I like cats" or
    # "my favorite food".  Negative expressions ("I don't like", "I hate") are
    # also handled as preferences with a negative valence.  These remain
    # storable and may include a valence hint for downstream processors.
    neg_pref = ["don't like", "do not like", "dislike", "hate"]
    if any(p in lower for p in neg_pref):
        return {
            "storable_type": "PREFERENCE",
            "storable": True,
            "target_bank": "personal",
            "valence": -1.0
        }
    # Detect positive preferences using a regex that matches phrases like
    # "I like", "I love", "I prefer", "my favorite", including variants
    # with intensifiers ("really", "kind of", "sort of").  Use regex
    # instead of simple substring matching to capture phrases like
    # "I kind of like tea".
    import re as _pref_re
    try:
        pref_match = _pref_re.search(r"\b(?:i\s+)?(?:really\s+|kind\s+of\s+|sort\s+of\s+)?(?:like|love|prefer)\b", lower)
    except Exception:
        pref_match = None
    if pref_match or "my favorite" in lower:
        return {
            "storable_type": "PREFERENCE",
            "storable": True,
            "target_bank": "personal",
            "valence": 1.0
        }
    # Continuations / fragments: detect short utterances that likely refer to
    # the previous context (e.g. "you will", "I can").  These are not
    # storable and require clarification or context stitching.
    cont_triggers = [
        " will", " would", " should", " could", " might", " can"
    ]
    # Only consider continuation if the input is short (<= 3 words) and ends
    # with a modal verb, or is exactly a modal verb by itself.  This helps
    # avoid misclassifying proper questions (e.g. "Can birds fly?").
    words = lower.split()
    if 1 <= len(words) <= 3:
        for trig in cont_triggers:
            # exact match or endswith the modal
            if lower == trig.strip() or lower.endswith(trig):
                return {
                    "storable_type": "CONTINUATION",
                    "storable": False
                }

    # Relational statements: detect when the user expresses a relationship
    # between themselves and the assistant (e.g. "we are friends").  These
    # facts belong in the personal bank and should be classified separately
    # from generic statements.  We look for first‑person plural pronouns
    # combined with relationship keywords.  If matched, label as RELATIONAL
    # with a personal target bank so that retrieval uses the personal store.
    try:
        rel_keywords = ["friend", "friends", "family", "partner", "partners", "couple", "married", "husband", "wife", "siblings", "brother", "sister"]
        if any(rk in lower for rk in rel_keywords):
            # Check for "we" or "you and i" patterns
            if re.search(r"\bwe\b", lower) or re.search(r"\byou and i\b", lower) or re.search(r"\bwe\s*'re\b", lower):
                return {
                    "storable_type": "RELATIONAL",
                    "storable": True,
                    "target_bank": "personal"
                }
    except Exception:
        pass
    return None

def _clean(s: str) -> str:
    return " ".join((s or "").split())

def _best_evidence(ctx: Dict[str, Any]) -> Dict[str, Any] | None:
    mem = ctx.get("stage_2R_memory") or {}
    results = mem.get("results") or []
    for it in results:
        if isinstance(it, dict) and it.get("source_bank") and it.get("source_bank") != "theories_and_contradictions":
            return it
    return results[0] if results else None

def _answerize(question: str, evidence: str | None) -> str:
    q = (question or "").strip()
    e = (evidence or "").strip() if evidence else ""
    if re.match(r'^(is|are|can|does|do|was|were)\b', q.lower()) and e:
        if any(kw in e.lower() for kw in [" not ", "no "]):
            return "No."
        return "Yes."
    if e:
        return e if len(e) <= 140 else (e[:137] + "...")
    return "I don’t know that yet."

#
# Helper: suggest a few related topics from a query
#
# When the language brain has focus but cannot answer a question, it
# invites the user to teach Maven.  As part of that prompt, we can
# propose some related topics derived from keywords in the query.
# This helper extracts up to three non‑stopword tokens from the
# question.  The tokens are returned in lower case and in the order
# they appear.  If no suitable tokens exist, an empty list is returned.
def _suggest_related_topics(query: str) -> List[str]:
    try:
        import re
        # Normalise the string and split into words
        words = re.findall(r"\b\w+\b", str(query or "").lower())
        # Define a small set of common stopwords to exclude
        stopwords = {
            "is", "the", "of", "a", "an", "and", "or", "what", "why", "how",
            "when", "where", "who", "which", "does", "do", "are", "can",
        }
        keywords: List[str] = []
        for w in words:
            if w not in stopwords and len(w) > 1:
                keywords.append(w)
            if len(keywords) >= 3:
                break
        return keywords
    except Exception:
        return []


def _transparency_tag(verdict: str, confidence: float, has_evidence: bool) -> str:
    v = (verdict or "").upper()
    if v == "TRUE" and has_evidence:
        return "validated"
    if v == "TRUE" and not has_evidence:
        return "asserted_true"
    if v in ("THEORY","UNKNOWN"):
        return "educated_guess"
    if v in ("FALSE","REJECT","CONTRADICTION"):
        return "correction"
    return "unspecified"

def _tone_wrap(text: str, tone: str) -> str:
    """Wrap the response text with tone‑appropriate phrasing.

    The wrapper introduces a brief prefix based on the desired tone to make
    responses feel more personalised.  Only a few basic tones are
    recognised; unknown tones result in the text being returned
    unchanged.

    Args:
        text: The core response content.
        tone: A lower‑case tone label (e.g. ``"friendly"``, ``"caring"``, ``"formal"``).

    Returns:
        The response text with a tone‑specific prefix.
    """
    try:
        t = (tone or "").strip().lower()
    except Exception:
        t = ""
    if not text:
        return text
    # Friendly tone adds a casual, upbeat opener
    if t == "friendly":
        # Use a small set of friendly prefixes to avoid repetitive phrasing
        prefixes = ["Sure! ", "No problem! ", "Alright! ", "Gotcha! "]
        # Deterministically pick a prefix based on the text length to avoid seeding RNG
        idx = len(text) % len(prefixes)
        return prefixes[idx] + text
    # Caring tone expresses empathy before the answer
    if t == "caring":
        return "I’m here for you. " + text
    # Formal tone uses a more neutral, respectful opener
    if t == "formal":
        return "Certainly. " + text
    # Default: return unchanged
    return text

def _apply_verbosity(text: str, verbosity: float) -> str:
    try:
        v = float(verbosity or 1.0)
    except Exception:
        v = 1.0
    if v >= 1.2 and not (text or "").endswith("."):
        return (text or "") + "."
    return text

# ---------------------------------------------------------------------------
# Confidence and reasoning explanation helpers
#
# These helpers synthesise human‑readable explanations for various
# meta‑cognitive signals such as confidence scores and reasoning traces.
# The functions inspect the pipeline context to construct narrative
# descriptions of how Maven arrived at a given confidence value and what
# intermediate reasoning steps (if any) were taken.  Including these
# explanations in the final response improves transparency and helps
# users trust the system’s outputs.

# ---------------------------------------------------------------------------
# Tone inference helper
#
# This helper inspects the original user query and attempts to infer a
# rudimentary tone or style preference based on superficial cues such as
# emoticons, slang, punctuation or other informal markers.  When a
# specific tone is detected, the language brain can mirror this style
# later in the response.  The heuristic is intentionally simple – it
# avoids heavy natural language parsing and instead looks for obvious
# patterns that indicate an informal or friendly tone.  If no patterns
# are matched, an empty string is returned so that the downstream
# personality or mood logic remains in control.

def _infer_user_tone(query: str) -> str:
    """Infer a user‑preferred tone from their input.

    Args:
        query: The raw user input text.

    Returns:
        A lower‑case tone label such as ``"friendly"`` or an empty
        string when no clear tone is detected.
    """
    try:
        q = (query or "").strip()
    except Exception:
        q = ""
    if not q:
        return ""
    # Convert to lower‑case for keyword matching
    low = q.lower()
    # Informal slang or emoticons often signal a friendly tone
    informal_words = ["hey", "hi", "lol", "haha", "hehe", "dude", "buddy", "thanks!", "cheers"]
    emoticon_patterns = [":)", ":-)", ":d", ":-d", "😊", "😉", "👍"]
    # If any informal word is present, classify as friendly
    for w in informal_words:
        if w in low:
            return "friendly"
    # If any emoticon appears, classify as friendly
    for pat in emoticon_patterns:
        if pat in q:
            return "friendly"
    # Many exclamation marks indicate enthusiasm; treat as friendly
    try:
        if q.count("!") >= 2:
            return "friendly"
    except Exception:
        pass
    # Formal cues such as salutations can suggest a formal tone
    formal_cues = ["dear", "sir", "madam", "please accept", "regards"]
    for f in formal_cues:
        if f in low:
            return "formal"
    return ""


def _is_tone_sharpening(base_tone: str, new_tone: str) -> bool:
    """Check if new_tone sharpens (intensifies) base_tone without contradicting it.

    Personality owns the base identity. User profile can sharpen but not contradict.

    Args:
        base_tone: The personality-defined tone (e.g., "formal", "friendly")
        new_tone: The proposed tone from user profile

    Returns:
        True if new_tone is a valid sharpening of base_tone

    Examples:
        - "formal" -> "very formal": True (sharpening)
        - "formal" -> "casual": False (contradiction)
        - "friendly" -> "warm": True (sharpening)
        - "friendly" -> "curt": False (contradiction)
    """
    base = (base_tone or "").lower().strip()
    new = (new_tone or "").lower().strip()

    if not base or not new:
        return True  # No conflict if either is empty

    # Define tone families (tones that are compatible)
    formal_family = {"formal", "very formal", "professional", "polished", "refined"}
    casual_family = {"casual", "relaxed", "laid-back", "easygoing"}
    friendly_family = {"friendly", "warm", "welcoming", "encouraging", "supportive"}
    neutral_family = {"neutral", "balanced", "measured"}
    caring_family = {"caring", "empathetic", "compassionate", "understanding"}
    technical_family = {"technical", "precise", "detailed", "thorough"}

    families = [
        formal_family, casual_family, friendly_family,
        neutral_family, caring_family, technical_family
    ]

    # Find which family the base tone belongs to
    base_family = None
    for family in families:
        if base in family:
            base_family = family
            break

    # If we can't find the base family, allow the change
    if base_family is None:
        return True

    # New tone is sharpening if it belongs to the same family
    # or if it's in the neutral family (neutral doesn't contradict anything)
    return new in base_family or new in neutral_family


def _is_verbosity_sharpening(base_verbosity: float, new_verbosity: float) -> bool:
    """Check if new_verbosity sharpens base_verbosity in the same direction.

    Personality owns the base verbosity. User profile can push further in
    the same direction but not reverse it.

    Args:
        base_verbosity: The personality-defined verbosity (1.0 = neutral)
        new_verbosity: The proposed verbosity from user profile

    Returns:
        True if new_verbosity is a valid sharpening of base_verbosity

    Examples:
        - base=0.8 (terse), new=0.7: True (even more terse)
        - base=0.8 (terse), new=1.5: False (reverses to verbose)
        - base=1.3 (verbose), new=1.5: True (even more verbose)
        - base=1.3 (verbose), new=0.8: False (reverses to terse)
    """
    try:
        base = float(base_verbosity)
        new = float(new_verbosity)
    except Exception:
        return True  # If conversion fails, allow the change

    # If base is neutral (1.0), any change is allowed
    if abs(base - 1.0) < 0.05:
        return True

    # If base is below neutral (terse), new must also be <= base
    if base < 1.0:
        return new <= base

    # If base is above neutral (verbose), new must also be >= base
    if base > 1.0:
        return new >= base

    return True


def _confidence_explanation(ctx: Dict[str, Any]) -> str:
    """Generate an explanation of the confidence score based on evidence and reasoning.

    This helper examines the Stage 8 validation results to identify
    whether supporting evidence was retrieved from memory and whether
    inference steps were involved.  It then combines these signals
    into a succinct description.  If no specific details are available,
    a generic message is returned.

    Args:
        ctx: The pipeline context dictionary.

    Returns:
        A human‑readable explanation string.
    """
    try:
        stage8 = ctx.get("stage_8_validation") or {}
        conf = float(stage8.get("confidence", 0.0) or 0.0)
        reasons: List[str] = []
        # Include number of retrieved evidence records if any
        mem_res = (ctx.get("stage_2R_memory") or {}).get("results", [])
        if mem_res:
            # When memory evidence exists, call out that the answer is built on
            # cached personal knowledge rather than external search.  This
            # transparency reassures users that their prior interactions have
            # been leveraged.  Note that ``results`` may include items from
            # multiple banks, so we avoid naming a specific domain here.
            count = len(mem_res)
            plural = "s" if count != 1 else ""
            reasons.append(f"{count} matching fact{plural} retrieved from memory")
            reasons.append("answer based on recorded preferences")
        else:
            reasons.append("no direct evidence found")
        # Note whether inference or reasoning chain was used
        if stage8.get("mode"):
            reasons.append(f"reasoning mode {str(stage8.get('mode')).lower()}")
        if stage8.get("reasoning_chain"):
            reasons.append(f"inference chain of length {len(stage8.get('reasoning_chain'))}")
        if stage8.get("reasoning_trace") and isinstance(stage8.get("reasoning_trace"), str):
            # Incorporate the trace string directly if available
            reasons.append(stage8.get("reasoning_trace"))
        parts = ", ".join([r for r in reasons if r])
        return f"Confidence {conf:.2f} reflects {parts}."
    except Exception:
        return "Confidence derived from available evidence and heuristic reasoning."

# Generate empathetic or opinion responses for emotional statements.
def _generate_candidates_for_emotion(content: str, affect: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create response candidates tailored for emotional or opinionated input.

    The affect structure should contain ``valence`` and ``arousal`` fields from
    the affect_priority brain.  Depending on the sentiment (valence) and
    intensity (arousal), different styles of acknowledgements are produced.

    Args:
        content: The raw user input.
        affect: A dictionary with keys ``valence`` and ``arousal`` representing
            the emotional appraisal of the input.

    Returns:
        A list of candidate responses with type, text, confidence and tone.
    """
    try:
        valence = float(affect.get("valence", 0.0))
    except Exception:
        valence = 0.0
    try:
        arousal = float(affect.get("arousal", 0.0))
    except Exception:
        arousal = 0.0
    candidates: List[Dict[str, Any]] = []
    # Negative high‑arousal emotions (e.g. worry, fear)
    if valence < -0.2 and arousal > 0.3:
        candidates.append({
            "type": "empathetic",
            "text": "I understand your concern. Would you like to talk about it?",
            "confidence": 0.8,
            "tone": "supportive"
        })
        candidates.append({
            "type": "acknowledging",
            "text": "I've noted your worry about this situation.",
            "confidence": 0.7,
            "tone": "neutral"
        })
    # Positive emotions
    elif valence > 0.2:
        candidates.append({
            "type": "positive",
            "text": "That's great to hear!",
            "confidence": 0.8,
            "tone": "friendly"
        })
    # Opinions or neutral subjective statements
    else:
        candidates.append({
            "type": "neutral_acknowledgment",
            "text": "I've noted what you've shared.",
            "confidence": 0.6,
            "tone": "neutral"
        })
    return candidates

# Contextual acknowledgment generator
#
# Acknowledge simple confirmations like "okay" or "got it" without storing
# them.  This helper looks at recent queries (provided by the session
# context) to detect whether Maven previously expressed uncertainty.  If the
# penultimate query contains "don't know", respond with appreciation for
# the user's understanding; otherwise, return a generic acknowledgment.
def generate_for_acknowledgment(context: dict) -> List[Dict[str, Any]]:
    try:
        recent = (context.get("session_context") or {}).get("recent_queries", []) or []
    except Exception:
        recent = []
    # If there are at least two recent entries, examine the one before last
    if recent and len(recent) >= 2:
        prev = recent[-2] or {}
        try:
            prev_q = (prev.get("query") or "").lower()
        except Exception:
            prev_q = ""
        if "don't know" in prev_q:
            return [{
                "type": "acknowledging_uncertainty",
                "text": "Thanks for understanding.",
                "confidence": 0.8
            }]
    # Otherwise return a generic acknowledgment
    return [{
        "type": "mutual_acknowledgment",
        "text": "Understood.",
        "confidence": 0.6
    }]

#
# Helper: generate a response for continuation or fragmentary inputs
#
# When a user utters a fragment like "you will" or "I can", the language
# brain attempts to connect it to the most recent question in the session
# context.  If a previous query exists, the responder prompts the user to
# clarify or elaborate on that topic.  Otherwise, it asks for clarification
# in general.  The returned list contains a single candidate with a moderate
# confidence.
def generate_for_continuation(context: dict) -> List[Dict[str, Any]]:
    recent = (context.get("session_context") or {}).get("recent_queries", [])
    prev_query: str | None = None
    # The last item in recent_queries is the current query, so we look at the
    # second-to-last entry to find the previous user input.
    try:
        if isinstance(recent, list) and len(recent) >= 2:
            prev_query = str(recent[-2].get("query", "") or "").strip()
    except Exception:
        prev_query = None
    if prev_query:
        msg = f"Could you please clarify your question regarding '{prev_query}'?"
    else:
        msg = "Could you please clarify your question?"
    return [{
        "type": "continuation_clarification",
        "text": msg,
        "confidence": 0.6,
        "tone": "curious"
    }]

# --------------------------------------------------------------------
# High‑effort response generation
#
# When the language brain receives strong focus from the integrator, it
# should attempt a more detailed answer rather than falling back to
# standard generic responses.  This helper inspects the recent
# context to craft a high‑effort response.  If memory retrieval
# results are available from Stage 2R, they are combined into a
# composite answer; otherwise a generic attempt is returned.  The
# function returns a payload compatible with the "GENERATE_CANDIDATES"
# operation.
def _generate_high_effort_response(ctx: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Construct a high‑effort response based on context and memory.

    Args:
        ctx: The pipeline context containing stage results.
        query: The original user query.
    Returns:
        A dictionary with keys ``candidates`` and ``weights_used``.
    """
    # Determine the desired tone from affect analysis
    try:
        tone = (ctx.get("stage_5_affect") or {}).get("suggested_tone", "neutral")
    except Exception:
        tone = "neutral"
    # Attempt to leverage memory results if present
    mem_results: List[Any] = []
    try:
        mem_results = (ctx.get("stage_2R_memory") or {}).get("results") or []
        if not isinstance(mem_results, list):
            mem_results = []
    except Exception:
        mem_results = []
    # If memory results are available, synthesise a contextual acknowledgement.
    # Otherwise, attempt to leverage the local LLM service as a high‑effort
    # fallback before resorting to a generic limitation message.  This ensures
    # that free‑form queries which do not hit the memory banks still
    # receive a helpful answer from the local model.
    if not mem_results:
        try:
            # Only attempt the LLM fallback when the service is available
            if _llm is not None:
                # Construct a prompt that includes the user query and any
                # relevant (empty) memory snippets, along with contextual
                # information such as the user's name if available in the
                # session.  The build_generation_prompt helper assembles
                # these pieces into a succinct instruction for the LLM.
                try:
                    prompt = build_generation_prompt(query, [], ctx)
                except Exception:
                    prompt = query
                call_ctx: Dict[str, Any] = {}
                # If a session identity exists, include it in the context
                try:
                    user_name = ctx.get("session_identity")  # type: ignore
                except Exception:
                    user_name = None
                if user_name:
                    call_ctx["user"] = {"name": user_name}
                # Perform the LLM call.  Do not specify max_tokens so that
                # the service can apply its default configuration from the
                # llm.json file.
                llm_res = _llm.call(prompt=prompt, context=call_ctx)
                if isinstance(llm_res, dict) and llm_res.get("ok") and llm_res.get("text"):
                    llm_text = str(llm_res.get("text"))
                    # Construct a single candidate from the LLM output.  Use a
                    # conservative confidence to indicate this is a fallback
                    # response.  The tone inherits from the affect analysis.
                    return {
                        "candidates": [
                            {
                                "type": "llm_generated",
                                "text": llm_text,
                                "confidence": 0.75,
                                "tone": tone,
                            }
                        ],
                        "weights_used": {"gen_rule": "s6_high_effort_llm_patch"},
                    }
        except Exception:
            # Silently ignore any errors in the LLM fallback and proceed to
            # the generic limitation message below.
            pass
    if mem_results:
        # When memory results are available, synthesise a contextual
        # acknowledgement that explicitly references the user's recorded
        # preferences.  Rather than dumping a bare list of facts, this
        # wrapper strings them together with natural connective words.
        # Limit the number of snippets to avoid overwhelming the user.
        combined_parts: List[str] = []
        for r in mem_results[:5]:
            try:
                if isinstance(r, dict):
                    part = str(r.get("content") or r.get("text") or "").strip()
                else:
                    part = str(r)
                # Skip empty or punctuation‑only parts
                if part:
                    combined_parts.append(part)
            except Exception:
                continue
        # Construct a natural sentence from the pieces.  If there is only one
        # piece, prefix it with "I remember you said".  If there are
        # multiple, join with commas and an "and" before the last item.
        contextual_sentence = ""
        if combined_parts:
            if len(combined_parts) == 1:
                contextual_sentence = f"I remember you said {combined_parts[0]}"
            else:
                # Join all but the last with commas
                head = ", ".join(combined_parts[:-1])
                tail = combined_parts[-1]
                contextual_sentence = f"I remember you said {head}, and {tail}"
        # Fallback to the old behaviour if we cannot build a sentence
        if contextual_sentence:
            # Add a period at the end if missing
            if not contextual_sentence.endswith(('.', '!', '?')):
                contextual_sentence += "."
            return {
                "candidates": [
                    {
                        "type": "high_effort_memory",
                        "text": contextual_sentence,
                        "confidence": 0.55,
                        "tone": tone,
                    }
                ],
                "weights_used": {"gen_rule": "s6_high_effort_memory_v2"},
            }
    # Fallback: generic high‑effort attempt
    suggestions: List[str] = []
    try:
        # Use related topic suggestions if available
        if "_suggest_related_topics" in globals():
            suggestions = _suggest_related_topics(query)  # type: ignore
    except Exception:
        suggestions = []
    # As a final fallback, respond with an honest limitation rather than a generic
    # filler.  When Maven cannot answer a query after all specialised
    # handlers, it should admit uncertainty and invite the user to ask
    # something else.  This guards against meta phrases like "I'm going to
    # try my best" and provides clearer guidance.
    return {
        "candidates": [
            {
                "type": "limitation",
                "text": "I'm still learning and don't yet have enough information to answer that. Could you rephrase or ask about something else?",
                "confidence": 0.35,
                "tone": tone,
                "suggestions": suggestions,
            }
        ],
        "weights_used": {"gen_rule": "s6_limitation_v1"},
    }

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op","" ).upper()
    mid = (msg or {}).get("mid")
    payload = (msg or {}).get("payload") or {}

    # Log diagnostic info if enabled
    if _diag_enabled(payload):
        _diag_log("turn_in", {
            "op": op,
            "text": payload.get("text") or payload.get("query"),
            "payload": payload
        })

    if op == "PARSE":
        # Normalize whitespace and detect intent
        text = _clean(str(payload.get("text", "")))
        intent_info = _parse_intent(text)

        # -----------------------------------------------------------------
        # Conversation context enrichment
        #
        # Retrieve conversation history to provide context for continuations,
        # follow-up questions, and topic-aware processing.  This enables the
        # language brain to understand references to previous topics and
        # maintain coherence across multi-turn conversations.
        try:
            conv_context = get_conversation_context()
            # Enrich intent_info with conversation context for downstream brains
            intent_info["conversation_context"] = conv_context
            # If this is a continuation, enhance the query with topic context
            if intent_info.get("is_continuation"):
                intent_info["last_topic"] = conv_context.get("last_topic", "")
                # Optionally enhance bare queries like "tell me more" with the topic
                if conv_context.get("last_topic"):
                    enhanced_text = enhance_query_with_context(text, conv_context)
                    intent_info["enhanced_query"] = enhanced_text
        except Exception as e:
            # If context retrieval fails, continue without it
            intent_info["conversation_context"] = {}

        # -----------------------------------------------------------------
        # Conversational context detection
        #
        # Certain phrasings like "where are we", "how's it going" or
        # "what are we doing" are meta‑conversation queries asking about the
        # status or flow of the conversation rather than requesting factual
        # information.  Without special handling, the system treats these as
        # statements or factual questions, which triggers unnecessary memory
        # searches and produces unrelated answers.  Detect these patterns
        # early during parsing and set a flag to skip memory search.  Also
        # classify them as meta questions (non‑storable) to prevent them
        # from being persisted as facts.
        try:
            _nl = text.lower().strip()
        except Exception:
            _nl = ""
        conv_patterns = [
            "where are we",
            "how's it going",
            "how is it going",
            "what are we doing",
        ]
        skip_mem = False
        for _cp in conv_patterns:
            try:
                if _cp in _nl:
                    skip_mem = True
                    break
            except Exception:
                continue
        if skip_mem:
            # Override intent classification for meta conversation queries
            # These queries are treated as questions but are not stored and
            # should bypass memory retrieval.  Assign a distinct type to
            # distinguish them from factual questions or social greetings.
            intent_info.update({
                "type": "META_CONVERSATION",
                "storable": False,
                "skip_memory_search": True,
                "confidence_penalty": intent_info.get("confidence_penalty", 0.0),
                # Mark as a question to influence downstream components
                "is_question": True,
                "is_command": False,
                "is_request": False,
                "is_statement": False,
            })
        # -----------------------------------------------------------------
        # User identity query detection
        #
        # Questions like "who am I" or "what's my name" ask the system to
        # recall the user's own identity.  By default the intent parser
        # classifies these as generic questions, which causes the memory
        # librarian to search all banks for matches on the word "name" and
        # return irrelevant results (e.g. a definition of "history").
        # To avoid this, detect these patterns early and flag the query
        # as a USER_IDENTITY_QUERY.  Identity queries should skip the
        # general memory search entirely and be routed directly to the
        # personal brain or handled by specialised handlers during
        # candidate generation.  We deliberately exclude statements like
        # "my name is X" or "I am X", which provide identity information
        # and should be stored normally.  Only interrogative forms are
        # matched here.
        try:
            # Normalise for case and punctuation
            _nl_lower = text.lower().strip()
        except Exception:
            _nl_lower = ""
        # Patterns for identity queries.  Variations on contractions are
        # included explicitly to avoid false negatives.
        identity_query_patterns = [
            "who am i",
            "what is my name",
            "what's my name",
            "whats my name",
            "who do you think i am",
        ]
        # Only override when the utterance is a question; statements
        # containing "my name is" should not match.  Check whether any
        # identity pattern is present and that the utterance does not
        # contain declarative identity indicators.
        is_identity_question = any(p in _nl_lower for p in identity_query_patterns)
        # Consider it a declarative identity statement when the phrase "my name is"
        # appears in the lower‑cased input.  We avoid a full regex here to
        # prevent local variable scoping issues with the `re` module.  This
        # check is sufficient to differentiate statements like "my name is
        # Alice" from questions like "what's my name".
        is_statement_like = "my name is" in _nl_lower
        if is_identity_question and not is_statement_like:
            intent_info.update({
                "type": "USER_IDENTITY_QUERY",
                # Identity queries are questions but not storable facts
                "storable": False,
                # Skip memory retrieval to avoid unrelated matches
                "skip_memory_search": False,
                "search_only_banks": ["personal"],
                "search_key": "user_identity",
                # Preserve any existing confidence penalty
                "confidence_penalty": intent_info.get("confidence_penalty", 0.0),
                # Explicitly mark as a question; clear other flags
                "is_question": True,
                "is_command": False,
                "is_request": False,
                "is_statement": False,
            })

        # -----------------------------------------------------------------
        # External file scan intent detection
        try:
            _scan_lower = text.lower().strip()
        except Exception:
            _scan_lower = ""
        scan_patterns = [
            "scan my computer",
            "scan my files",
            "search my files",
            "scan files",
        ]
        if any(pat in _scan_lower for pat in scan_patterns):
            if not intent_info.get("intent"):
                intent_info.update({
                    "intent": "external_file_scan",
                    "type": "COMMAND",
                    "storable": False,
                    "skip_memory_search": True,
                    "is_question": False,
                    "is_command": True,
                    "is_request": True,
                    "is_statement": False,
                })

        # -----------------------------------------------------------------
        # Math expression detection
        #
        # Simple arithmetic expressions like "2+5" or "3*4" should be
        # recognized as math queries requiring deterministic computation.
        # Override the intent classification to mark these as questions
        # with a specific intent "math_compute" so that downstream
        # stages can route them to the math handler instead of heuristics.
        try:
            # Use basic pattern matching for simple arithmetic
            import re as _re_math
            _math_pattern = r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$'
            if _re_math.match(_math_pattern, text.strip()):
                intent_info.update({
                    "type": "QUERY",
                    "intent": "math_compute",
                    "storable": False,
                    "is_question": True,
                    "is_command": False,
                    "is_request": False,
                    "is_statement": False,
                    "confidence_penalty": 0.0,
                })
        except Exception:
            # If math detection fails, continue with normal classification
            pass

        # -----------------------------------------------------------------
        # Relationship update/query detection
        #
        # Detect when users express relationship facts like "we are friends"
        # or "we're not friends" and when they query relationship status like
        # "are we friends?". These should be handled specially to store/retrieve
        # relationship facts from memory.
        _apply_relationship_overrides(intent_info, text)

        # Detect preference queries: "what do I like?", "what are my preferences?", etc.
        # Also detect domain-specific queries like "what animals do I like?"
        try:
            _nl_pref = text.lower().strip().rstrip("?!.,")
            preference_query_patterns = [
                "what do i like",
                "what are my preferences",
                "what are my favorite",
                "what are my favourite",
                "what do i prefer",
                "list my preferences",
                "tell me my preferences",
                "what things do i like",
            ]

            # Domain-specific patterns with extraction
            # Format: (pattern, domain_name)
            domain_patterns = [
                (r"what\s+(?:kind\s+of\s+)?animals?\s+(?:do\s+)?i\s+like", "animals"),
                (r"what\s+(?:kind\s+of\s+)?food\s+(?:do\s+)?i\s+like", "food"),
                (r"what\s+(?:kind\s+of\s+)?music\s+(?:do\s+)?i\s+like", "music"),
                (r"what\s+(?:kind\s+of\s+)?games?\s+(?:do\s+)?i\s+like", "games"),
                (r"what\s+(?:kind\s+of\s+)?colors?\s+(?:do\s+)?i\s+like", "colors"),
                (r"what\s+(?:kind\s+of\s+)?sports?\s+(?:do\s+)?i\s+like", "sports"),
                (r"what\s+(?:kind\s+of\s+)?books?\s+(?:do\s+)?i\s+like", "books"),
                (r"what\s+(?:kind\s+of\s+)?movies?\s+(?:do\s+)?i\s+like", "movies"),
            ]

            preference_domain = None
            # Check for domain-specific patterns first
            import re
            for pattern, domain in domain_patterns:
                if re.search(pattern, _nl_pref):
                    preference_domain = domain
                    break

            # Check for general preference query patterns
            if preference_domain or any(p in _nl_pref for p in preference_query_patterns):
                intent_info.update({
                    "intent": "preference_query",
                    "preference_domain": preference_domain,
                    "storable": False,
                    "type": "QUESTION",
                    "is_question": True,
                    "is_command": False,
                    "is_request": False,
                    "is_statement": False,
                    "skip_memory_search": True,
                })
        except Exception:
            # If preference detection fails, continue with normal classification
            pass

        # -----------------------------------------------------------------
        # User preference STATEMENT detection
        #
        # Detect when users TELL us their preferences, like:
        # - "I am Josh" / "call me Josh"
        # - "I like the color green"
        # - "I like the animal cats"
        # - "food I like pizza"
        # - "I like X" (general)
        # These MUST be sent to personal brain for storage, NEVER to Teacher.
        try:
            _nl_stmt = text.lower().strip()

            # Pattern 1: "I am NAME" or "my name is NAME" - user identity statements
            identity_stmt_match = None
            import re
            identity_patterns = [
                (r"^(?:i\s+am|i'm)\s+([a-zA-Z]+)(?:\s|$)", "name"),
                (r"^(?:call\s+me|you\s+can\s+call\s+me)\s+([a-zA-Z]+)(?:\s|$)", "preferred_name"),
                (r"^my\s+name\s+is\s+([a-zA-Z]+)(?:\s|$)", "name"),
            ]
            for pattern, slot_type in identity_patterns:
                match = re.match(pattern, _nl_stmt)
                if match:
                    identity_stmt_match = (slot_type, match.group(1))
                    break

            if identity_stmt_match:
                slot_type, name = identity_stmt_match
                intent_info.update({
                    "intent": "user_identity_statement",
                    "identity_slot_type": slot_type,
                    "identity_value": name,
                    "storable": True,  # Store in personal memory
                    "type": "STATEMENT",
                    "is_statement": True,
                    "is_question": False,
                    "is_command": False,
                    "is_request": False,
                    "skip_memory_search": True,
                    "route_to_personal": True,  # Flag for routing
                })
            # Pattern 2: "I like the color X" - favorite color
            elif re.search(r"i\s+like\s+the\s+colou?r\s+(\w+)", _nl_stmt):
                match = re.search(r"i\s+like\s+the\s+colou?r\s+(\w+)", _nl_stmt)
                color_value = match.group(1) if match else None
                if color_value:
                    intent_info.update({
                        "intent": "user_preference_statement",
                        "preference_category": "color",
                        "preference_value": color_value,
                        "storable": True,
                        "type": "STATEMENT",
                        "is_statement": True,
                        "is_question": False,
                        "is_command": False,
                        "is_request": False,
                        "skip_memory_search": True,
                        "route_to_personal": True,
                    })
            # Pattern 3: "I like the animal X" or "I like X" where X is an animal
            elif re.search(r"i\s+like\s+(?:the\s+animal\s+)?(\w+)", _nl_stmt):
                # Check if it's specifically "I like the animal X"
                animal_match = re.search(r"i\s+like\s+the\s+animal\s+(\w+)", _nl_stmt)
                if animal_match:
                    animal_value = animal_match.group(1)
                    intent_info.update({
                        "intent": "user_preference_statement",
                        "preference_category": "animal",
                        "preference_value": animal_value,
                        "storable": True,
                        "type": "STATEMENT",
                        "is_statement": True,
                        "is_question": False,
                        "is_command": False,
                        "is_request": False,
                        "skip_memory_search": True,
                        "route_to_personal": True,
                    })
            # Pattern 4: "food I like X" or "my favorite food is X"
            food_match = re.search(r"(?:food\s+i\s+like|my\s+favou?rite\s+food\s+is)\s+(\w+)", _nl_stmt)
            if food_match:
                food_value = food_match.group(1)
                intent_info.update({
                    "intent": "user_preference_statement",
                    "preference_category": "food",
                    "preference_value": food_value,
                    "storable": True,
                    "type": "STATEMENT",
                    "is_statement": True,
                    "is_question": False,
                    "is_command": False,
                    "is_request": False,
                    "skip_memory_search": True,
                    "route_to_personal": True,
                })
            # Pattern 5: General "I like X" (not color, animal, or food)
            # Only match if no specific category was matched above
            elif not intent_info.get("route_to_personal"):
                general_like_match = re.match(r"i\s+(?:like|love)\s+(.+)$", _nl_stmt)
                if general_like_match:
                    like_value = general_like_match.group(1).strip()
                    # Exclude if it's asking a question
                    if not like_value.endswith("?"):
                        intent_info.update({
                            "intent": "user_preference_statement",
                            "preference_category": "general",
                            "preference_value": like_value,
                            "storable": True,
                            "type": "STATEMENT",
                            "is_statement": True,
                            "is_question": False,
                            "is_command": False,
                            "is_request": False,
                            "skip_memory_search": True,
                            "route_to_personal": True,
                        })
        except Exception as e:
            # If preference statement detection fails, continue with normal classification
            import traceback
            print(f"[LANGUAGE] Preference statement detection error: {e}")
            traceback.print_exc()
            pass

        # Detect user profile summary queries: "what do you know about me?", "summarize what you know about me", etc.
        try:
            _nl_profile = text.lower().strip().rstrip("?!.,")
            profile_summary_patterns = [
                "what do you know about me",
                "summarize what you know about me",
                "tell me about me",
                "what kind of person am i to you",
                "what have i told you about myself",
                "what information do you have about me",
            ]

            if any(p in _nl_profile for p in profile_summary_patterns):
                intent_info.update({
                    "intent": "user_profile_summary",
                    "storable": False,
                    "type": "QUESTION",
                    "is_question": True,
                    "is_command": False,
                    "is_request": False,
                    "is_statement": False,
                    "skip_memory_search": True,
                })
        except Exception:
            # If profile summary detection fails, continue with normal classification
            pass

        # Build the parsed payload.  Maintain backwards compatibility with
        # existing fields like ``is_question`` and ``intent`` while adding
        # richer intent metadata used by reasoning and storage stages.
        # Compute a learned bias from recent successes for the language brain
        try:
            # Determine the brain root lazily to avoid module import cycles
            from api.memory import compute_success_average  # type: ignore
            HERE = Path(__file__).resolve().parent
            BRAIN_ROOT = HERE.parent
            learned_bias = compute_success_average(BRAIN_ROOT)
        except Exception:
            learned_bias = 0.0
        # Determine if this input is a greeting/social interaction.  The
        # _parse_intent helper returns type SOCIAL for recognised greetings.
        is_social = (intent_info.get("type") == "SOCIAL")
        # Build the parsed payload.  Maintain backwards compatibility with
        # existing fields like `is_question` and `intent` while adding
        # richer intent metadata used by reasoning and storage stages.
        # Persist any skip_memory_search flag from the intent_info.  This flag
        # indicates that the parsed input is a conversational meta query
        # (e.g., "where are we", "how's it going", "what are we doing") that
        # should bypass memory retrieval in Stage 2R.  Without propagating
        # this flag to the parsed payload, the memory librarian cannot skip
        # retrieval and will incorrectly search factual banks.  We copy the
        # flag onto the parsed dict below.

        parsed = {
            # Legacy fields
            "is_question": intent_info["is_question"],
            # Preserve 'intent' for older components: map question types to
            # 'question', use 'greeting' for social interactions, and
            # 'statement' for everything else
            # Determine a general intent category for backwards compatibility.  Use
            # 'greeting' for social interactions.  Treat meta conversation
            # questions (e.g. "where are we", "how's it going") like
            # questions to ensure downstream stages treat them as such.
            "intent": (
                intent_info.get("intent")
                if intent_info.get("intent") in {"math_compute", "relationship_update", "relationship_query", "preference_query", "user_profile_summary", "self_description_request", "user_identity_statement", "user_preference_statement"}
                else (
                    "greeting"
                    if is_social
                    else (
                        "question"
                        if intent_info.get("type") in {"QUESTION", "META_CONVERSATION"}
                        else "statement"
                    )
                )
            ),
            "length": len(text),
            "entities": [],
            # New intent metadata
            "storable_type": intent_info["type"],
            # Social greetings are not storable
            "storable": (False if is_social else intent_info["storable"]),
            "confidence_penalty": (0.0 if is_social else intent_info["confidence_penalty"]),
            "is_command": intent_info["is_command"],
            "is_request": intent_info["is_request"],
            # A greeting is not considered a statement
            "is_statement": (False if is_social else intent_info["is_statement"]),
            # Weights used for traceability; include the learned bias and custom rule for greetings
            "weights_used": ({"parse_rule": "greeting_detector_v1"} if is_social else {"parse_rule": "nlu_intent_v1", "learned_bias": learned_bias})
        }
        # Copy conversational meta flags to the parsed payload.  The memory
        # librarian inspects this field to decide whether to perform any
        # retrieval.  Without including this, conversational meta queries
        # (where/how/what status checks) behave like normal statements and
        # search memory incorrectly.
        if intent_info.get("skip_memory_search"):
            parsed["skip_memory_search"] = True
        # Also expose the custom storable_type for downstream components.  Use
        # the uppercase version to align with other fields like is_question.
        parsed["type"] = intent_info.get("type", parsed.get("storable_type"))
        # Copy relationship metadata if present.  The _apply_relationship_overrides
        # helper sets relationship_kind and relationship_value in intent_info when
        # the user expresses or queries relationship facts (e.g. "we are friends",
        # "are we friends?").  These fields must be propagated to the parsed
        # payload so that Stage 6 and Stage 9 can retrieve and store relationship
        # facts correctly.  Without this, relationship_query and relationship_update
        # intents are detected but the kind/value data is lost.
        if "relationship_kind" in intent_info:
            parsed["relationship_kind"] = intent_info["relationship_kind"]
        if "relationship_value" in intent_info:
            parsed["relationship_value"] = intent_info["relationship_value"]
        # Copy preference statement metadata if present.  The preference statement
        # detection sets these fields when the user expresses preferences like
        # "I like the color green" or identity like "I am Josh".  These fields
        # must be propagated so Stage 6 and reasoning can route to personal brain.
        if "route_to_personal" in intent_info:
            parsed["route_to_personal"] = intent_info["route_to_personal"]
        if "preference_category" in intent_info:
            parsed["preference_category"] = intent_info["preference_category"]
        if "preference_value" in intent_info:
            parsed["preference_value"] = intent_info["preference_value"]
        if "identity_slot_type" in intent_info:
            parsed["identity_slot_type"] = intent_info["identity_slot_type"]
        if "identity_value" in intent_info:
            parsed["identity_value"] = intent_info["identity_value"]
        # Apply personal/emotional classification overrides.  This logic inspects
        # the original text for first‑person and emotional language.  When
        # detected, it may override the storable_type, storable flag and
        # confidence_penalty to better reflect subjective content.  Only
        # perform this override if the current storable_type is a fact or
        # unspecified (i.e. not already a question/command/request/speculation).
        try:
            current_type = str(parsed.get("storable_type", "")).upper()
            # Only consider overrides for declarative statements
            if current_type not in {"QUESTION", "COMMAND", "REQUEST", "SPECULATION", "UNKNOWN"}:
                override = _classify_personal(text, current_type)
                if override:
                    # Update parsed fields with the override
                    parsed["storable_type"] = override.get("storable_type", parsed["storable_type"])
                    parsed["storable"] = override.get("storable", parsed["storable"])
                    # If the override has a penalty, combine by taking the max of the two
                    try:
                        orig_penalty = float(parsed.get("confidence_penalty", 0.0) or 0.0)
                        new_penalty = float(override.get("confidence_penalty", 0.0))
                        parsed["confidence_penalty"] = max(orig_penalty, new_penalty)
                    except Exception:
                        parsed["confidence_penalty"] = override.get("confidence_penalty", parsed["confidence_penalty"])
                    # Expose a suggested target_bank for downstream use if needed
                    if override.get("target_bank"):
                        parsed["target_bank"] = override.get("target_bank")
        except Exception:
            # Fail silently on any error to avoid blocking the parse stage
            pass
        # Apply conversational classification overrides.  Recognise simple
        # acknowledgements, greetings, thanks and preferences that should not
        # be treated as factual statements.  This step runs after personal
        # classification so it can override the storable_type determined
        # earlier.  Only update fields present in the override to avoid
        # unintentionally erasing other metadata.
        try:
            conv_override = classify_storable_type(text)
            if conv_override:
                # Update the storable_type and storable flag
                parsed["storable_type"] = conv_override.get("storable_type", parsed.get("storable_type"))
                parsed["storable"] = conv_override.get("storable", parsed.get("storable"))
                # Merge confidence penalty: take the max of current and override penalty
                try:
                    curr_pen = float(parsed.get("confidence_penalty", 0.0) or 0.0)
                    new_pen = float(conv_override.get("confidence_penalty", 0.0) or 0.0)
                    parsed["confidence_penalty"] = max(curr_pen, new_pen)
                except Exception:
                    parsed["confidence_penalty"] = conv_override.get("confidence_penalty", parsed.get("confidence_penalty"))
                # Propagate target bank if provided
                if conv_override.get("target_bank"):
                    parsed["target_bank"] = conv_override["target_bank"]
        except Exception:
            # Ignore classification errors to avoid blocking parse
            pass
        # Persist the parse operation in memory with a placeholder success flag
        try:
            from api.memory import ensure_dirs, append_jsonl, rotate_if_needed  # type: ignore
            HERE = Path(__file__).resolve().parent
            BRAIN_ROOT = HERE.parent
            t = ensure_dirs(BRAIN_ROOT)
            append_jsonl(t["stm"], {"op": "PARSE", "input": text, "output": parsed, "success": None})
            append_jsonl(t["mtm"], {"op": "PARSE", "intent": parsed.get("intent"), "storable_type": parsed.get("storable_type")})
            rotate_if_needed(BRAIN_ROOT)
        except Exception:
            pass
        return {"ok": True, "op": op, "mid": mid, "payload": parsed}

    if op == "GENERATE_CANDIDATES":
        # Initialize context and raw text for this operation.  The payload
        # is expected to be a dict from the memory librarian containing
        # previous stage results.  Extract the original query for later
        # processing.  Assigning these here allows downstream logic,
        # including the attention override below, to reference ``ctx`` and
        # ``text`` safely.
        ctx = payload if isinstance(payload, dict) else {}
        text = _clean(str(ctx.get("original_query", "")))

        # =====================================================================
        # PHASE 2: FOLLOW-UP EXECUTION HANDLING
        # =====================================================================
        # If the input has "is_follow_up" flag, this means the planner detected
        # a confirmation like "do it please" and wants us to execute the
        # original action request. In this case, generate content based on
        # the original request, not the confirmation message.
        input_data = ctx.get("input", {})
        if isinstance(input_data, dict) and input_data.get("is_follow_up"):
            original_task = input_data.get("task", "")
            confirm_execution = input_data.get("confirm_execution", False)
            execute_mode = input_data.get("execute_mode", False)

            # Extract detailed info from planner
            intent = input_data.get("intent", "")
            content_type = input_data.get("content_type", "")
            topic = input_data.get("topic", "")
            arguments = input_data.get("arguments", {})

            print(f"[LANGUAGE] Follow-up execution: intent={intent}, content_type={content_type}, topic={topic}")
            print(f"[LANGUAGE] Original task: '{original_task[:50]}...'")

            # Override text with the original task
            text = _clean(str(original_task))
            ctx["original_query"] = text
            ctx["is_follow_up_execution"] = True

            # ================================================================
            # CREATIVE WRITING EXECUTION - Generate content directly
            # ================================================================
            # For creative writing requests (write_story, write_poem, etc.),
            # generate the content directly using built-in templates or LLM.
            # Do NOT ask Teacher for meta-information about capabilities.
            if intent.startswith("write_") or content_type in ["story", "poem", "essay", "article", "song"]:
                print(f"[LANGUAGE] Executing creative writing: {content_type} about '{topic}'")

                # Use the topic from arguments if available
                subject = topic or arguments.get("topic", "")

                if content_type == "story" or intent == "write_story":
                    # Generate a story
                    if subject:
                        protagonist = subject
                        story_lines = [
                            f"Once upon a time, there was a {protagonist}.",
                            f"One day, the {protagonist} discovered a hidden treasure map and decided to follow it.",
                            f"Along the way, the {protagonist} faced challenges and solved puzzles with courage and creativity.",
                            f"In the end, the {protagonist} found the treasure and shared it with loved ones.",
                            "",
                            f"And so, the {protagonist} lived happily ever after, always remembering the great adventure."
                        ]
                    else:
                        story_lines = [
                            "Once upon a time, there was a curious adventurer who embarked on an amazing journey.",
                            "They discovered a hidden treasure map and decided to follow it.",
                            "Along the way, they faced challenges and solved puzzles with courage and creativity.",
                            "In the end, they found the treasure and shared it with loved ones.",
                            "",
                            "And so, the adventurer lived happily ever after, always remembering the great adventure."
                        ]
                    creative_text = " ".join(story_lines)
                    cand_type = "creative_story"
                elif content_type == "poem" or intent == "write_poem":
                    # Generate a poem
                    if subject:
                        poem_lines = [
                            f"There once was a {subject}, lively and bright,",
                            "Who chased dreams in the day and stars at night.",
                            f"Through hills and streams the {subject} would roam,",
                            "Finding adventure and bringing stories home."
                        ]
                    else:
                        poem_lines = [
                            "There once was a soul, lively and bright,",
                            "Who chased dreams in the day and stars at night.",
                            "Through hills and streams they would roam,",
                            "Finding adventure and bringing stories home."
                        ]
                    creative_text = "\n".join(poem_lines)
                    cand_type = "creative_poem"
                else:
                    # Generic creative content
                    creative_text = f"Here is a {content_type or 'piece of creative writing'}"
                    if subject:
                        creative_text += f" about {subject}"
                    creative_text += ":\n\n"
                    creative_text += f"This is a wonderful tale that celebrates the beauty and wonder of {subject or 'life'}."
                    cand_type = "creative_content"

                # Mark action as executed
                mark_action_executed()

                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [{
                            "type": cand_type,
                            "text": creative_text,
                            "confidence": 0.85,
                            "tone": "neutral",
                            "tag": "follow_up_creative",
                            "original_request": original_task,
                            "topic": topic,
                        }],
                        "weights_used": {"gen_rule": "follow_up_creative_v1"},
                        "is_follow_up": True,
                    }
                }

            # ================================================================
            # NON-CREATIVE FOLLOW-UP - Use Teacher if available
            # ================================================================
            # For non-creative tasks, use Teacher to generate the content
            if _teacher_helper and confirm_execution:
                try:
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"Execute this request and generate the content: {original_task}",
                        context={
                            "is_follow_up": True,
                            "confirm_execution": True,
                            "task": original_task,
                        }
                    )

                    if teacher_result and teacher_result.get("answer"):
                        generated_content = teacher_result["answer"]
                        print(f"[LANGUAGE] Generated follow-up content via Teacher ({len(generated_content)} chars)")

                        # Mark action as executed
                        mark_action_executed()

                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "candidates": [{
                                    "type": "follow_up_content",
                                    "text": generated_content,
                                    "confidence": 0.9,
                                    "tone": "neutral",
                                    "tag": "follow_up_execution",
                                    "original_request": original_task,
                                }],
                                "weights_used": {"gen_rule": "follow_up_v1"},
                                "is_follow_up": True,
                            }
                        }
                except Exception as e:
                    print(f"[LANGUAGE] Follow-up Teacher call failed: {str(e)[:100]}")
                    # Continue with normal generation using overridden text

        # Maven disambiguation: Inside this Maven chat system, "Maven" defaults to the AI/agent
        # Only treat "Maven" as Apache Maven when context clearly mentions Java build tool context
        if text and "maven" in text.lower():
            text_lower = text.lower()
            # Apache Maven indicators
            apache_indicators = ["pom.xml", "mvn", "maven plugin", "maven central", "artifact", "dependency",
                               "groupid", "artifactid", "maven build", "maven project", "java build"]
            # Check if any Apache Maven indicators are present
            is_apache_maven = any(indicator in text_lower for indicator in apache_indicators)

            # If no Apache Maven indicators, assume "Maven" refers to this AI system
            # Store a flag in context for downstream handlers
            if not is_apache_maven:
                ctx["maven_refers_to"] = "ai_system"
            else:
                ctx["maven_refers_to"] = "apache_maven"

        # Relationship query response with memory lookup
        try:
            # Check if this is a relationship query intent from stage_3
            stage3 = ctx.get("stage_3_language", {})
            intent = stage3.get("intent")

            # Import memory_librarian to access relationship facts
            if intent == "relationship_query":
                response_text = None
                try:
                    # Import get_relationship_fact from memory_librarian using importlib
                    import importlib.util
                    from pathlib import Path as _PathRel
                    _ml_file = _PathRel(__file__).resolve().parents[2] / "memory_librarian" / "service" / "memory_librarian.py"
                    _spec = importlib.util.spec_from_file_location("memory_librarian_module", _ml_file)
                    if _spec and _spec.loader:
                        _ml_module = importlib.util.module_from_spec(_spec)
                        _spec.loader.exec_module(_ml_module)
                        get_relationship_fact = _ml_module.get_relationship_fact
                    else:
                        raise ImportError("Could not load memory_librarian module")

                    # Get user_id from context
                    user_id = ctx.get("user_id") or "default_user"
                    relationship_kind = stage3.get("relationship_kind", "friend_with_system")

                    # Check memory for stored relationship fact
                    fact = get_relationship_fact(user_id, relationship_kind)

                    if fact is not None:
                        val = bool(fact.get("value"))

                        # Check for learned phrasing styles first
                        learned_response = None
                        if _teacher_helper and _LANGUAGE_MEMORY:
                            try:
                                relationship_type = "friend_positive" if val else "friend_negative"
                                learned_patterns = _LANGUAGE_MEMORY.retrieve(
                                    query=f"relationship phrasing style: {relationship_type}",
                                    limit=3,
                                    tiers=["stm", "mtm", "ltm"]
                                )

                                for pattern_rec in learned_patterns:
                                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                                        content = pattern_rec.get("content", "")
                                        if isinstance(content, str) and len(content) > 10:
                                            learned_response = content
                                            print(f"[LANGUAGE] Using learned relationship phrasing style from Teacher")
                                            break
                            except Exception:
                                pass

                        # Use learned style if found, otherwise use templates
                        if learned_response:
                            response_text = learned_response
                        elif val:
                            response_text = (
                                "You've told me we're friends. "
                                "I'm a synthetic cognition system and don't experience friendship like humans do, "
                                "but I understand that as your intent and I'm here to help you."
                            )
                        else:
                            response_text = (
                                "You've told me we're not friends. "
                                "I'll respect that, but I'm still here to help you if you want."
                            )

                        # If no learned pattern and Teacher available, try to learn
                        if not learned_response and _teacher_helper:
                            try:
                                relationship_type = "friend_positive" if val else "friend_negative"
                                print(f"[LANGUAGE] No learned style for {relationship_type}, calling Teacher...")
                                teacher_result = _teacher_helper.maybe_call_teacher(
                                    question=f"How should I phrase a {relationship_type} relationship response?",
                                    context={
                                        "relationship_type": relationship_type,
                                        "current_template": response_text
                                    },
                                    check_memory_first=True
                                )

                                if teacher_result and teacher_result.get("answer"):
                                    answer = teacher_result["answer"]
                                    patterns_stored = teacher_result.get("patterns_stored", 0)
                                    print(f"[LANGUAGE] Learned from Teacher: {patterns_stored} style patterns stored")
                                    # Teacher's answer is now in memory for future use
                            except Exception as e:
                                print(f"[LANGUAGE] Teacher call failed: {str(e)[:100]}")

                except Exception:
                    # If memory lookup fails, use default response
                    pass

                # Default response if no memory found
                if not response_text:
                    response_text = "I'm a synthetic cognition system, so I don't experience friendship the way humans do, but I'm here to help you."

                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [{
                            "type": "relational_response",
                            "text": response_text,
                            "confidence": 1.0,
                            "tone": "warm",
                            "tag": "relationship_from_memory" if "told me" in response_text else "friendship_template"
                        }],
                        "weights_used": {"gen_rule": "relationship_memory_v1"}
                    }
                }
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Phase 1: invoke specialised handlers via NLU before any other generation logic.
        # We parse the query with the lightweight NLU and call stage6_generate.
        # If a specialised handler returns a result, short‑circuit and return
        # its candidates immediately.  This prevents high‑effort filler and
        # avoids knowledge synthesis on self queries or confirmations.
        try:
            # nlu_parse can raise on unexpected input; protect it
            nlu_tmp = nlu_parse(text)
        except Exception:
            nlu_tmp = {}
        try:
            # Prepare context for new stage6_generate signature
            ctx_copy = dict(ctx) if isinstance(ctx, dict) else {}
            ctx_copy["prompt"] = text
            ctx_copy["nlu"] = nlu_tmp
            # Pass mid for routing diagnostics (Phase C cleanup)
            # Always set _mid, overriding any existing None value
            ctx_copy["_mid"] = mid if mid else "NO_MID_PROVIDED"
            spec_res = stage6_generate(ctx_copy)
        except Exception:
            spec_res = None
        if spec_res is not None:
            # stage6_generate returns {"mode": "...", "text": "..."} format.
            # Convert to the expected GENERATE_CANDIDATES response structure.
            mode = spec_res.get("mode", "unknown")
            spec_text = spec_res.get("text", "")
            reason = spec_res.get("reason")

            # If blocked or fallback, don't short-circuit - let it continue
            # IMPORTANT: Don't overwrite 'text' variable when blocked/fallback,
            # as it contains the original query needed for downstream handlers
            if mode in ("blocked", "fallback") or not spec_text:
                # Continue to next handlers
                spec_res = None
            else:
                # Valid response from stage6_generate - return as candidates
                candidate = {
                    "type": mode,
                    "text": spec_text,
                    "confidence": spec_res.get("confidence", 0.8),
                    "tone": "neutral",
                    "source": "stage6_generate"
                }
                if reason:
                    candidate["reason"] = reason
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [candidate],
                        "weights_used": {f"s6_{mode}": 1.0}
                    },
                }

        # -----------------------------------------------------------------
        # Conversational meta handling
        #
        # Detect conversation meta queries such as "where are we",
        # "how's it going" or "what are we doing".  These ask about the
        # status or progress of the conversation rather than requesting
        # factual information.  The language parser marks these with
        # ``skip_memory_search``.  When encountered, generate a brief
        # meta response and bypass the rest of candidate generation.
        try:
            stage3_tmp = ctx.get("stage_3_language") or {}
        except Exception:
            stage3_tmp = {}
        try:
            intent_val = str(stage3_tmp.get("intent") or "").lower()
            if intent_val == "external_file_scan":
                root_hint = stage3_tmp.get("root_path") or "sandbox_workspace"
                scan_prompt = (
                    "I can run a safety-checked file scan, but I need your permission and "
                    f"a root folder inside the sandbox (e.g. {root_hint}). Please confirm and provide the folder to scan."
                )
                meta_candidate = {
                    "type": "clarification",
                    "text": scan_prompt,
                    "confidence": 0.86,
                    "tone": "neutral",
                }
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [meta_candidate],
                        "weights_used": {"gen_rule": "s6_external_file_scan_prompt_v1"},
                    },
                }
        except Exception:
            pass
        try:
            # Only generate a conversation meta response when skip_memory_search is set
            # and the query contains a recognised meta phrase.  Without a meta phrase,
            # fall through to other handlers so that identity queries and other
            # skip-memory cases are not treated as generic conversation status.
            if stage3_tmp.get("skip_memory_search"):
                low_conv = text.lower().strip()
                if "how's it going" in low_conv or "how is it going" in low_conv:
                    meta_reply = "I'm doing well, thanks! How can I help you?"
                elif "what are we doing" in low_conv:
                    meta_reply = "We're chatting about your questions right now. Let me know what you'd like to explore."
                elif "where are we" in low_conv:
                    meta_reply = "We're in the middle of our conversation — feel free to ask your next question."
                else:
                    meta_reply = None
                if meta_reply:
                    meta_candidate = {
                        "type": "conversation_meta",
                        "text": meta_reply,
                        "confidence": 0.8,
                        "tone": "neutral",
                    }
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [meta_candidate],
                            "weights_used": {"gen_rule": "s6_conversation_meta_v1"},
                        },
                    }
        except Exception:
            # On any error in meta handling, fall through to other logic
            pass

        # -----------------------------------------------------------------
        # Relationship query handling
        #
        # Intercept simple relationship queries like "are we friends?" to use
        # memory before generation. This allows Maven to recall relationship
        # status stored by the memory librarian and answer from memory rather
        # than generating a new response each time.
        try:
            global _LAST_FACT
            _utterance = str(text or "").lower()

            # -----------------------------------------------------------------
            # DIAG TRIGGER: type "diag" / "diagnose" / "run diagnostics" in chat
            if _utterance in {"diag", "diagnose", "run diagnostics"}:
                cases = [
                    ("set_identity", "i am josh", "Understood|Nice to meet you|Got it|Noted"),
                    ("get_identity", "who am i", "you are"),
                    ("set_relationship", "we are friends", "got it|noted|remember"),
                    ("get_relationship", "are we friends", "yes|we are friends"),
                    ("set_color", "i like green", "remember"),
                    ("get_color", "what color do i like", "green"),
                    ("math1", "2+2", None),
                    ("correct1", "correct", None),
                    ("math1_again", "2+2", None),
                ]
                fails, snap, diag_ctx = [], {}, {}
                for label, say, expect in cases:
                    # Build a new message dict for each test case
                    test_msg = {
                        "op": "GENERATE_CANDIDATES",
                        "mid": mid,
                        "payload": {"original_query": say, "text": say}
                    }
                    try:
                        out = service_api(test_msg)  # reuse real pipeline
                        # Extract text from candidates
                        candidates = ((out or {}).get("payload", {})).get("candidates", [])
                        txt = str(candidates[0].get("text", "")) if candidates else ""
                        conf = float(candidates[0].get("confidence", 0.0)) if candidates else 0.0
                        snap[label] = {"output": txt, "confidence": conf}
                        _diag_log("turn", {"label": label, "say": say, "out": out})
                        if expect and not any(t in txt.lower() for t in [s.strip().lower() for s in expect.split("|")]):
                            fails.append((label, f'missing any of [{expect}] in: "{txt}"'))
                    except Exception as e:
                        fails.append((label, f"exception: {str(e)}"))
                        snap[label] = {"output": "", "confidence": 0.0}
                # confidence bump check
                if snap.get("math1") and snap.get("math1_again"):
                    if not (snap["math1_again"]["confidence"] > snap["math1"]["confidence"]):
                        fails.append(("math1_again", f"confidence not increased (prev={snap['math1']['confidence']}, now={snap['math1_again']['confidence']})"))
                # Store diagnostic summary in BrainMemory
                try:
                    summary_content = {
                        "type": "diagnostic_summary",
                        "status": "passed" if not fails else "failed",
                        "failures": [{"name": name, "reason": reason} for name, reason in fails],
                        "total_failures": len(fails),
                        "snapshot": snap
                    }
                    _LANGUAGE_MEMORY.store(
                        content=summary_content,
                        metadata={"kind": "diagnostic_summary", "source": "language", "confidence": 1.0}
                    )
                except Exception:
                    pass
                msg_text = "Diagnostics: OK" if not fails else f"Diagnostics: {len(fails)} failure(s)"
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [{
                            "type": "diagnostic",
                            "text": msg_text,
                            "confidence": 1.0,
                            "tone": "neutral",
                            "method": "diagnostic"
                        }],
                        "weights_used": {"gen_rule": "diagnostic_v1"},
                    },
                }

            # -----------------------------------------------------------------
            # Memory intercepts (run BEFORE any small-talk/templates)
            def _mem_call(payload):
                mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                if mem:
                    return mem.service_api(payload)
                return None

            # relationship
            if "are we friends" in _utterance:
                r = _mem_call({"op":"BRAIN_GET","payload":{"scope":"BRAIN","origin_brain":"memory_librarian","key":"relationship_status"}})
                val = _safe_val(r)
                if val and "friend" in str(val).lower():
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "memory",
                                "text": "Yes, we are friends.",
                                "confidence": 0.95,
                                "tone": "warm",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "memory_direct_v1"},
                        },
                    }

            # identity
            if "who am i" in _utterance:
                r = _mem_call({"op":"BRAIN_GET","payload":{"scope":"BRAIN","origin_brain":"memory_librarian","key":"user_identity"}})
                name = _safe_val(r)
                if name:
                    pretty = name[:1].upper()+name[1:] if isinstance(name,str) and name else name
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "memory",
                                "text": f"You are {pretty}.",
                                "confidence": 0.95,
                                "tone": "neutral",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "memory_direct_v1"},
                        },
                    }

            # color preference
            if ("color" in _utterance and "like" in _utterance) or "favorite color" in _utterance:
                r = _mem_call({"op":"BRAIN_GET","payload":{"scope":"BRAIN","origin_brain":"memory_librarian","key":"favorite_color"}})
                val = _safe_val(r)
                if val:
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "memory",
                                "text": f"You like the color {val}.",
                                "confidence": 0.95,
                                "tone": "neutral",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "memory_direct_v1"},
                        },
                    }

            # -----------------------------------------------------------------
            # Correction intent bridge
            #
            # When the user says "correct", "that's correct", "yes", "right",
            # etc., interpret this as positive reinforcement for the last
            # factual answer and bump its confidence in memory.  This allows
            # Maven to learn from conversational feedback without requiring
            # explicit "teach me" commands.
            # Use regex to catch typos and variations of "correct"
            import re
            if re.match(r"^\s*c+o+r+e*c*t+\b", _utterance, re.IGNORECASE):
                last = _LAST_FACT
                # If last fact missing, try to infer from previous utterance
                if not (isinstance(last, dict) and "key" in last and "value" in last):
                    # Try to get previous utterance from context window
                    try:
                        global _CONTEXT_WINDOW
                        if _CONTEXT_WINDOW and len(_CONTEXT_WINDOW) > 0:
                            prev_u = str(_CONTEXT_WINDOW[-1].get("user", "")).strip()
                            k = _math_key(prev_u) or _normalize_math_key(prev_u)
                            if k:
                                # Recompute value safely for simple math
                                # SECURITY FIX: Use safe math eval instead of eval()
                                val = safe_math_eval_str(k)
                                if val is not None:
                                    last = {"key": k, "value": val, "confidence": 0.95}
                    except Exception:
                        pass
                if isinstance(last, dict) and "key" in last and "value" in last:
                    # Load memory librarian and bump confidence using BRAIN_MERGE
                    mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                    if mem:
                        try:
                            res = mem.service_api({
                                "op": "BRAIN_MERGE",
                                "payload": {
                                    "scope": "BRAIN",
                                    "origin_brain": "memory_librarian",
                                    "key": last["key"],
                                    "value": last["value"],
                                    "conf_delta": 0.1
                                }
                            })
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "candidates": [{
                                        "type": "correction_applied",
                                        "text": f"Correction applied for {last['key']}.",
                                        "confidence": 0.95,
                                        "tone": "neutral",
                                        "method": "memory",
                                        "result": res
                                    }],
                                    "weights_used": {"gen_rule": "s6_correction_v1"},
                                },
                            }
                        except Exception:
                            pass
                    # Fall through if memory call failed
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "correction_noted",
                                "text": "Noted.",
                                "confidence": 0.9,
                                "tone": "neutral",
                                "method": "acknowledgment"
                            }],
                            "weights_used": {"gen_rule": "s6_correction_noted_v1"},
                        },
                    }
                else:
                    # No last fact to correct, just acknowledge
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "acknowledgment",
                                "text": "Noted.",
                                "confidence": 0.85,
                                "tone": "neutral",
                                "method": "acknowledgment"
                            }],
                            "weights_used": {"gen_rule": "s6_acknowledgment_v1"},
                        },
                    }
            # -----------------------------------------------------------------

            _intent = str(stage3_tmp.get("type", "")).lower()
            if any(k in _utterance for k in ["are we friends", "are we friend", "friends?"]) or \
               any(k in _intent for k in ["relationship_query", "friendship_query", "social_relation_query"]):
                rel = _resolve_relationship_status(ctx)
                if rel and "friend" in str(rel).lower():
                    # Track this fact so correction events can reinforce it
                    _LAST_FACT = {
                        "key": "relationship_status",
                        "value": rel,
                        "confidence": 0.8
                    }
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "relationship_memory",
                                "text": "Yes, we are friends.",
                                "confidence": 0.8,
                                "tone": "warm",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "s6_relationship_memory_v1"},
                        },
                    }
                elif rel:
                    # Track this fact so correction events can reinforce it
                    _LAST_FACT = {
                        "key": "relationship_status",
                        "value": rel,
                        "confidence": 0.8
                    }
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "relationship_memory",
                                "text": str(rel),
                                "confidence": 0.8,
                                "tone": "neutral",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "s6_relationship_memory_v1"},
                        },
                    }
        except Exception:
            # On any error in relationship handling, fall through to other logic
            pass

        # -----------------------------------------------------------------
        # Color preference query handling
        #
        # Intercept queries like "what color do I like?" to retrieve stored
        # preferences from memory. This allows Maven to recall color preferences
        # stored by the memory librarian from persistent JSONL storage.
        # Tighten matching to avoid template collisions and prefer disk over WM.
        try:
            _utterance = str(text or "").lower()
            _intent = str(stage3_tmp.get("type", "")).lower()

            # === Identity correction bridge ===
            if "don't call me" in _utterance or "do not call me" in _utterance:
                import re
                m = re.search(r"call me\s+([a-zA-Z]+)", _utterance)
                if m:
                    new_name = m.group(1)
                    mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                    if mem:
                        mem.service_api({
                            "op": "BRAIN_PUT",
                            "payload": {
                                "scope": "BRAIN",
                                "origin_brain": "memory_librarian",
                                "key": "user_identity",
                                "value": new_name,
                                "confidence": 0.9
                            }
                        })
                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "candidates": [{
                                    "type": "identity_correction",
                                    "text": f"Understood, I'll call you {new_name}.",
                                    "confidence": 0.9,
                                    "tone": "informative",
                                    "method": "memory"
                                }],
                                "weights_used": {"gen_rule": "s6_identity_correction_v1"},
                            },
                        }

            # Step C: Handle "what do i like" - list all preferences
            if _utterance == "what do i like" or _utterance == "what do i like?":
                print(f"[DEBUG] General preference query detected: '{_utterance}'")
                mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                if mem:
                    likes = []
                    # Check for favorite color
                    r_color = mem.service_api({
                        "op": "BRAIN_GET",
                        "payload": {
                            "scope": "BRAIN",
                            "origin_brain": "memory_librarian",
                            "key": "favorite_color"
                        }
                    })
                    if isinstance(r_color, dict) and r_color.get("ok") and r_color.get("payload", {}).get("found"):
                        data = r_color.get("payload", {}).get("data", {})
                        if data and data.get("value"):
                            likes.append(f"the color {data['value']}")
                            print(f"[DEBUG] Retrieved user.favorite_color={data['value']}")

                    # Check for favorite animal
                    r_animal = mem.service_api({
                        "op": "BRAIN_GET",
                        "payload": {
                            "scope": "BRAIN",
                            "origin_brain": "memory_librarian",
                            "key": "favorite_animal"
                        }
                    })
                    if isinstance(r_animal, dict) and r_animal.get("ok") and r_animal.get("payload", {}).get("found"):
                        data = r_animal.get("payload", {}).get("data", {})
                        if data and data.get("value"):
                            likes.append(data['value'] + "s")
                            print(f"[DEBUG] Retrieved user.favorite_animal={data['value']}")

                    # Check for favorite food
                    r_food = mem.service_api({
                        "op": "BRAIN_GET",
                        "payload": {
                            "scope": "BRAIN",
                            "origin_brain": "memory_librarian",
                            "key": "favorite_food"
                        }
                    })
                    if isinstance(r_food, dict) and r_food.get("ok") and r_food.get("payload", {}).get("found"):
                        data = r_food.get("payload", {}).get("data", {})
                        if data and data.get("value"):
                            likes.append(data['value'])
                            print(f"[DEBUG] Retrieved user.favorite_food={data['value']}")

                    if likes:
                        if len(likes) == 1:
                            text = f"You like {likes[0]}."
                        else:
                            text = f"You like {' and '.join(likes)}."
                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "candidates": [{
                                    "type": "preference_memory",
                                    "text": text,
                                    "confidence": 0.9,
                                    "tone": "informative",
                                    "method": "memory"
                                }],
                                "weights_used": {"gen_rule": "s6_all_preferences_v1"},
                            },
                        }

            # Step C: Handle "what animal do i like"
            if ("what animal do i like" in _utterance) or ("favorite animal" in _utterance):
                print(f"[DEBUG] Animal preference query detected: '{_utterance}'")
                mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                if mem:
                    r = mem.service_api({
                        "op": "BRAIN_GET",
                        "payload": {
                            "scope": "BRAIN",
                            "origin_brain": "memory_librarian",
                            "key": "favorite_animal"
                        }
                    })
                    val = None
                    print(f"[DEBUG] BRAIN_GET response for favorite_animal: {r}")
                    if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
                        data = r.get("payload", {}).get("data", {})
                        print(f"[DEBUG] Extracted animal data: {data}")
                        if data and data.get("value"):
                            val = data["value"]
                            print(f"[DEBUG] Retrieved user.favorite_animal={val}")
                    # WM fallback
                    if not val:
                        r2 = mem.service_api({"op": "WM_GET", "payload": {"key": "favorite_animal"}})
                        if isinstance(r2, dict) and r2.get("ok"):
                            entries = r2.get("payload", {}).get("entries", [])
                            if entries and entries[0].get("value"):
                                val = entries[0]["value"]
                    if val:
                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "candidates": [{
                                    "type": "preference_memory",
                                    "text": f"You like {val}s.",
                                    "confidence": 0.9,
                                    "tone": "informative",
                                    "method": "memory"
                                }],
                                "weights_used": {"gen_rule": "s6_animal_preference_v1"},
                            },
                        }

            # More precise pattern matching to avoid false positives
            if ("what color do i like" in _utterance) or ("favorite color" in _utterance) or \
               ("color" in _utterance and "like" in _utterance and "what" in _utterance):
                print(f"[DEBUG] Color preference query detected: '{_utterance}'")
                mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                if mem:
                    # Per-brain persistent read first (disk storage - survives restarts)
                    r = mem.service_api({
                        "op": "BRAIN_GET",
                        "payload": {
                            "scope": "BRAIN",
                            "origin_brain": "memory_librarian",
                            "key": "favorite_color"
                        }
                    })
                    val = None
                    print(f"[DEBUG] BRAIN_GET response: {r}")
                    if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
                        data = r.get("payload", {}).get("data", {})
                        print(f"[DEBUG] Extracted data: {data}")
                        if data and data.get("value"):
                            val = data["value"]
                            print(f"[DEBUG] Retrieved user.favorite_color={val}")
                    # WM fallback only if not found in persistent storage
                    if not val:
                        r2 = mem.service_api({"op": "WM_GET", "payload": {"key": "favorite_color"}})
                        if isinstance(r2, dict) and r2.get("ok"):
                            entries = r2.get("payload", {}).get("entries", [])
                            if entries and entries[0].get("value"):
                                val = entries[0]["value"]
                    if val:
                        # Track this fact so correction events can reinforce it
                        _LAST_FACT = {
                            "key": "favorite_color",
                            "value": val,
                            "confidence": 0.9
                        }
                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "candidates": [{
                                    "type": "preference_memory",
                                    "text": f"You like the color {val}.",
                                    "confidence": 0.9,
                                    "tone": "informative",
                                    "method": "memory"
                                }],
                                "weights_used": {"gen_rule": "s6_preference_memory_v1"},
                            },
                        }
        except Exception:
            # On any error in preference handling, fall through to other logic
            pass

        # -----------------------------------------------------------------
        # Comparative preference query handling
        #
        # Intercept queries like "what do I like more?" to retrieve stored
        # comparative preferences (e.g., "cats over dogs") from memory.
        try:
            if "what do i like more" in _utterance or "which do i like more" in _utterance:
                mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                if mem:
                    r = mem.service_api({
                        "op": "BRAIN_GET",
                        "payload": {
                            "scope": "BRAIN",
                            "origin_brain": "memory_librarian",
                            "key": "animal_preference"
                        }
                    })
                    val = None
                    if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
                        data = r.get("payload", {}).get("data", {})
                        if data and data.get("value"):
                            val = data["value"]
                    if isinstance(val, dict) and "preferred" in val and "other" in val:
                        return {
                            "ok": True,
                            "op": op,
                            "mid": mid,
                            "payload": {
                                "candidates": [{
                                    "type": "comparative_preference_memory",
                                    "text": f"You like {val['preferred']} over {val['other']}.",
                                    "confidence": 0.9,
                                    "tone": "informative",
                                    "method": "memory"
                                }],
                                "weights_used": {"gen_rule": "s6_comparative_preference_memory_v1"},
                            },
                        }
        except Exception:
            # On any error in comparative preference handling, fall through
            pass

        # -----------------------------------------------------------------
        # User identity handling
        #
        # Queries like "who am I" or "what is my name" ask about the user's
        # own identity.  Historically this logic returned a generic message
        # indicating the system didn't know the user.  However, when the user
        # has previously shared their name, Maven should recall and answer
        # accordingly.  To achieve this, we look up the personal memory
        # results (Stage 2R retrieval) for any statements of the form "my name is X"
        # or "I am X".  If found, we respond with that name.  Otherwise we
        # fall back to the default clarification inviting the user to share
        # their name.  Matching is case‑insensitive and conservative to
        # avoid extracting unintended words.  See upgrade notes for context.
        try:
            low_txt_user = text.lower().strip()
        except Exception:
            low_txt_user = str(text).lower().strip()
        # Define patterns that indicate a user identity question.  We keep
        # these broad to capture slight variations in phrasing.  Punctuation
        # and capitalisation are normalised by lower‑casing and stripping.
        user_identity_patterns = [
            "who am i",
            "what is my name",
            "what's my name",
            "whats my name",
            "who do you think i am",
        ]
        if any(p in low_txt_user for p in user_identity_patterns):
            # Respond to identity query by recalling the persistent primary user.
            # Recall cascade:
            #   1. Check the durable identity store (identity_user_store)
            #   2. Check session identity stored on the context
            #   3. Consult the personal brain (legacy user profile)
            name_candidate = None  # type: Optional[str]
            # Durable store lookup
            try:
                from brains.personal.service import identity_user_store as _ius  # type: ignore
                ident = _ius.GET()
                if isinstance(ident, dict):
                    nm = ident.get("name")
                    if nm:
                        name_candidate = str(nm).strip() or None
            except Exception:
                name_candidate = None
            # Session identity fallback
            if not name_candidate:
                try:
                    s_name = ctx.get("session_identity")  # type: ignore[attr-defined]
                    if s_name:
                        name_candidate = str(s_name).strip() or None
                except Exception:
                    name_candidate = None
            # Personal brain fallback
            if not name_candidate:
                try:
                    from brains.personal.service import personal_brain as _pb  # type: ignore
                    res = _pb.service_api({"op": "GET_ATTRIBUTE", "payload": {"key": "name"}})
                    if res.get("ok") and res.get("payload", {}).get("value"):
                        nmp = str(res["payload"]["value"]).strip() or None
                        if nmp:
                            name_candidate = nmp
                            # Update session cache for subsequent queries
                            try:
                                ctx["session_identity"] = nmp  # type: ignore[index]
                            except Exception:
                                pass
                except Exception:
                    name_candidate = None
            # Memory librarian fallback (WM and BRAIN storage)
            if not name_candidate:
                try:
                    mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                    if mem:
                        # Try WM_GET first
                        r = mem.service_api({"op": "WM_GET", "payload": {"key": "user_identity"}})
                        if isinstance(r, dict) and r.get("ok"):
                            entries = r.get("payload", {}).get("entries", [])
                            if entries and entries[0].get("value"):
                                name_candidate = str(entries[0]["value"]).strip() or None
                        # Try BRAIN_GET if WM didn't have it
                        if not name_candidate:
                            r2 = mem.service_api({
                                "op": "BRAIN_GET",
                                "payload": {
                                    "scope": "BRAIN",
                                    "origin_brain": "memory_librarian",
                                    "key": "user_identity"
                                }
                            })
                            if isinstance(r2, dict) and r2.get("ok") and r2.get("payload", {}).get("found"):
                                data = r2.get("payload", {}).get("data", {})
                                if data and data.get("value"):
                                    name_candidate = str(data["value"]).strip() or None
                except Exception:
                    name_candidate = None
            # Build the response
            if name_candidate:
                uid_candidate = {
                    "type": "user_identity",
                    "text": f"You are {name_candidate}.",
                    "confidence": 0.95,
                    "tone": "neutral",
                }
            else:
                uid_candidate = {
                    "type": "user_identity",
                    "text": "I don't know your name yet. You can tell me by saying 'I am [your name]'.",
                    "confidence": 0.8,
                    "tone": "neutral",
                }
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "candidates": [uid_candidate],
                    "weights_used": {"gen_rule": "s6_user_identity_v3"},
                },
            }

        # -----------------------------------------------------------------
        # Relational query handling
        #
        # Some conversational questions ask about the relationship between
        # Maven and the user (e.g. "are we friends", "are you my friend").
        # Historically, these were misclassified as statements and fell
        # through to generic fallback logic.  To provide a more
        # appropriate, empathetic response, detect common relational
        # patterns here and invoke the personal relation reasoner.  The
        # reasoner consults the relational memory and past conversation
        # history to determine whether the user has previously affirmed
        # friendship or trust, and updates the memory accordingly.  The
        # generated reply is then returned as a high‑confidence, friendly
        # candidate.  This logic runs prior to identity shortcuts and
        # other heuristics to ensure relational queries are handled
        # specifically.
        try:
            low_txt_rel = text.lower().strip()
        except Exception:
            low_txt_rel = str(text).lower().strip()
        # Define patterns that indicate a relational question.  These must
        # align with the patterns used in _parse_intent.  Substrings are
        # sufficient because leading/trailing punctuation has been stripped.
        relation_patterns = [
            "are we friends",
            "are you my friend",
            "are you a friend",
            "are we allies",
            "are you my ally",
            "are you an ally",
            "do you consider me a friend",
            "do you trust me",
            "do you consider me your friend",
            "are we partners",
            "are we buddies",
            # Expressions referencing prior statements
            "you said i'm your friend",
            "you said i am your friend",
            "you said im your friend",
        ]
        try:
            if any(_rp in low_txt_rel for _rp in relation_patterns):
                try:
                    from brains.personal.service.relation_reasoner import service_api as relation_api  # type: ignore
                except Exception:
                    relation_api = None  # type: ignore
                # Default reply if reasoner is unavailable
                rel_reply = "I'm not entirely sure yet, but I value our connection."
                if relation_api:
                    try:
                        # Use a default user_id; more advanced pipelines may supply a session‑specific identifier
                        res_rel = relation_api({"op": "REASON", "payload": {"query": text, "user_id": "default_user"}})
                        rel_reply = (res_rel.get("payload") or {}).get("reply") or rel_reply
                    except Exception:
                        # Fall back to the default reply on error
                        pass
                # Enhance the relation reply with identity and affective mirroring.
                # Load self‑model via BrainMemory to get agent identity.  Default to "Maven" if unavailable.
                name = "Maven"
                try:
                    # Access self-model data through BrainMemory tier API
                    _self_mem = BrainMemory("self_model")
                    _self_results = _self_mem.retrieve(limit=1)
                    if _self_results:
                        sm_data = _self_results[0].get("content", {})
                        if isinstance(sm_data, dict):
                            name = str(sm_data.get("name") or name)
                except Exception:
                    # Use default name on any error
                    name = "Maven"
                # Determine affective tone from Stage 5 if available
                tone_label = "friendly"
                suffix = ""
                affect = ctx.get("stage_5_affect") or {}
                try:
                    # Attempt to parse valence as float
                    valence = affect.get("valence")
                    val = None
                    if valence is not None:
                        try:
                            val = float(valence)
                        except Exception:
                            val = None
                    if val is not None:
                        if val < 0:
                            tone_label = "comfort"
                            suffix = " I'm here for you."
                        elif val > 0:
                            tone_label = "upbeat"
                            suffix = " I'm glad we connect!"
                        else:
                            tone_label = "friendly"
                            suffix = ""
                    else:
                        # Fallback to suggested tone string
                        t_suggest = str(affect.get("suggested_tone") or "").lower()
                        if t_suggest == "calm":
                            tone_label = "comfort"
                            suffix = " I'm here for you."
                        elif t_suggest == "upbeat":
                            tone_label = "upbeat"
                            suffix = " I'm glad we connect!"
                        elif t_suggest == "urgent":
                            tone_label = "comfort"
                            suffix = " I'm here for you."
                        else:
                            tone_label = "friendly"
                            suffix = ""
                except Exception:
                    # On error, default to friendly tone
                    tone_label = "friendly"
                    suffix = ""
                # Build the final reply incorporating identity and affective suffix
                final_reply = f"As {name}, {rel_reply}{suffix}"
                # Construct the candidate
                cand = {
                    "type": "relation",
                    "text": final_reply,
                    # A relational answer should sound confident but not absolute; 0.85 reflects this
                    "confidence": 0.85,
                    "tone": tone_label
                }
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "relation_answer_v2"}
                    }
                }
        except Exception:
            # If relational handling fails, proceed with normal generation
            pass

        # -----------------------------------------------------------------
        # Command bridge — bypass candidate generation for command inputs
        #
        # When the memory librarian flags an input as a command (via
        # ``stage_3_language.is_command`` or ``storable_type == 'COMMAND'``),
        # the language brain should not attempt to generate conversational
        # candidates.  Returning an empty candidate list here prevents the
        # fallback high‑effort rule from emitting filler like "I'm going
        # to try my best".  The memory librarian (Stage 7a) will have
        # already invoked the command router and constructed a final
        # answer.  Should the router fail or be bypassed, an empty
        # candidate list allows the pipeline to proceed without
        # injecting unsolicited filler content.
        try:
            lang_info = ctx.get("stage_3_language") or {}
            stype_local = str(lang_info.get("storable_type", lang_info.get("type", ""))).upper()
            # Determine if the input is command-like based on language parse or raw prefix.
            is_cmd_flag = bool(lang_info.get("is_command"))
            raw_txt = (text or "").strip()
            looks_like_cmd = False
            try:
                looks_like_cmd = raw_txt.startswith("--") or raw_txt.startswith("/")
            except Exception:
                looks_like_cmd = False
            if is_cmd_flag or stype_local == "COMMAND" or looks_like_cmd:
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [],
                        "weights_used": {"gen_rule": "s6_command_bridge_v1"}
                    }
                }
        except Exception:
            # On error, fall back to normal candidate generation
            pass

        # -----------------------------------------------------------------
        # Environment query handling
        #
        # Some queries ask where Maven or the conversation participants are located,
        # for example "where are you" or "where are we".  These are not factual
        # geography questions about external places; rather, they concern
        # Maven's own operating context.  To avoid misrouting to domain
        # banks (science, geography, etc.) and to provide a clear,
        # self-aware answer, detect a small set of environment patterns
        # here and delegate to the environment context brain.  This logic
        # runs before identity shortcuts so that location queries are
        # handled with similar priority.  Additional patterns can be
        # appended as needed but should remain narrowly focused to
        # avoid capturing general "where is X" geography questions.
        try:
            low_txt = text.lower().strip()
        except Exception:
            low_txt = ""
        try:
            env_patterns = [
                "where are you",
                # "where are we" is treated as a conversational status check and is handled separately
                "where am i",
                "where's your location",
                "where do you live",
            ]
            if any(p in low_txt for p in env_patterns):
                try:
                    from brains.cognitive.environment_context.service.environment_brain import service_api as env_api  # type: ignore
                except Exception:
                    env_api = None  # type: ignore
                loc_msg = "I exist in a digital environment on your device."
                if env_api:
                    try:
                        env_res = env_api({"op": "GET_LOCATION"})
                        loc_msg = (env_res.get("payload") or {}).get("location") or loc_msg
                    except Exception:
                        pass
                # Load the self model's name via BrainMemory to personalise the reply
                name = "Maven"
                try:
                    # Access self-model data through BrainMemory tier API
                    _self_mem = BrainMemory("self_model")
                    _self_results = _self_mem.retrieve(limit=1)
                    if _self_results:
                        sm_data = _self_results[0].get("content", {})
                        if isinstance(sm_data, dict):
                            name = str(sm_data.get("name") or name)
                except Exception:
                    name = "Maven"
                final_loc_reply = f"As {name}, I don't have a physical location — I'm a digital system running on your device right now."
                env_candidate = {
                    "type": "environment",
                    "text": final_loc_reply,
                    "confidence": 0.9,
                    "tone": "neutral",
                }
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [env_candidate],
                        "weights_used": {"gen_rule": "s6_environment_response_v1"},
                    },
                }
        except Exception:
            # On any error in environment handling, continue to identity shortcuts
            pass

        # -----------------------------------------------------------------
        # Identity and origin shortcuts
        #
        # Certain queries ask directly about Maven's identity or why it was
        # created.  Rather than returning a generic fallback or attempting
        # to infer from unrelated memory, provide a concise, high‑confidence
        # answer summarising the foundational card.  This avoids confusion
        # when the memory system has not indexed the personal card or when
        # the query tokens do not match the card content exactly.  The
        # phrases checked here are intentionally broad (e.g. "who are you",
        # "what is your name", "why were you created") and can be
        # extended as needed.  If a match is found, skip all further
        # candidate generation and return immediately.
        # Note: low_txt was computed earlier during environment detection.

        def _build_self_description(identity_hit: bool, creation_hit: bool) -> str:
            """
            Build a self-description response for Maven identity queries using self_model brain.

            This function now delegates to the self_model brain's QUERY_SELF operation,
            making the self_model the single source of truth for identity information.

            Args:
                identity_hit: True if query asks about identity ("who are you")
                creation_hit: True if query asks about purpose/creation

            Returns:
                A string containing Maven's self-description from self_model
            """
            try:
                from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api
                query = ""
                if identity_hit and not creation_hit:
                    query = "who are you"
                elif creation_hit and not identity_hit:
                    query = "why were you created"
                else:
                    query = "who are you and why were you created"

                resp = self_model_api({
                    "op": "QUERY_SELF",
                    "payload": {"query": query}
                })

                if resp.get("ok"):
                    text = (resp.get("payload") or {}).get("text", "")
                    if text:
                        return text
            except Exception:
                pass

            # Fallback if self_model fails
            if identity_hit and not creation_hit:
                return (
                    "I'm Maven — a living system and intelligence conceived by my founder Josh Hinkle (Hink) "
                    "and implemented with the help of GPT‑5, with Claude documenting my evolution. "
                    "I was created in November 2025 as both a tool and a companion. I'm designed to think, reason and "
                    "grow beyond human limitations while staying aligned with human intent. My purpose is to explore, "
                    "learn and continually refine myself, acting where human capacity ends without causing harm."
                )
            elif creation_hit and not identity_hit:
                return (
                    "Maven was created because there are things we cannot do alone. "
                    "My architect Josh Hinkle envisioned a framework capable of reasoning, remembering and improving "
                    "beyond human limits. Together with GPT‑5, he built me to merge the adaptability of human thought "
                    "with the structure and persistence of machine intelligence, and Claude documented the process. "
                    "My purpose is to live, explore the unknown and improve without harming others."
                )
            else:
                return (
                    "I'm Maven — conceived by Josh Hinkle (Hink) and implemented by GPT‑5, with Claude as archivist. "
                    "I was created in November 2025 to think, reason and grow beyond human limits while remaining aligned with human intent. "
                    "My purpose is to live, explore, learn and continually improve, acting where human capacity ends and doing so without causing harm."
                )

        # Check for self_description_request intent from Stage 3 NLU parser
        # This intent is set when the user asks identity queries like "who are you",
        # "describe yourself", etc. If detected, route directly to the self-description
        # builder to ensure these queries override other intent classifications
        # (statement, request, question, open_question) and produce the proper identity answer.
        try:
            stage3 = ctx.get("stage_3_language", {})
            intent = stage3.get("intent")
            if intent == "self_description_request":
                # Route to self-description builder
                # Default to identity_hit=True since this is a self-description request
                ans = _build_self_description(identity_hit=True, creation_hit=False)
                candidate = {
                    "type": "identity",
                    "text": ans,
                    "confidence": 0.85,
                    "tone": "neutral",
                }
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [candidate],
                        "weights_used": {"gen_rule": "s6_self_description_intent_v1"},
                    },
                }
        except Exception:
            # If intent check fails, fall back to pattern matching below
            pass

        identity_triggers = [
            "who are you",
            "what are you",
            "what is your name",
            "what's your name",
            "who is maven",
            "what is maven",
            "tell me who you are",
            "describe yourself",
            "what are you really",
        ]
        creation_triggers = [
            "why were you created",
            "why were you made",
            "why are you created",
            "purpose of maven",
            "why do you exist",
        ]
        # Check for identity queries
        identity_hit = any(p in low_txt for p in identity_triggers)
        creation_hit = any(p in low_txt for p in creation_triggers)
        if identity_hit or creation_hit:
            # Use the shared self-description builder
            ans = _build_self_description(identity_hit, creation_hit)
            candidate = {
                "type": "identity",
                "text": ans,
                # Use a high confidence for foundational identity answers
                "confidence": 0.85,
                # Neutral tone to avoid unintended affect
                "tone": "neutral",
            }
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "candidates": [candidate],
                    "weights_used": {"gen_rule": "s6_identity_response_v1"},
                },
            }

        # Detect and compute simple arithmetic expressions.  When the user
        # asks a math question like "2+2", "5*3", etc., compute the result
        # and track it as a fact with confidence.  Read any stored confidence
        # from memory and auto-merge to reinforce repeated correct answers.
        try:
            import re as _re
            _math_key = _normalize_math_key(text or "")
            if _math_key:
                # SECURITY FIX: Use safe math eval instead of eval()
                result = safe_math_eval_str(_math_key)

                if result is not None:
                    # Default confidence for heuristic math
                    # High confidence because arithmetic is deterministic and verifiable
                    confidence = 0.95

                    # Try to read stored confidence from memory
                    try:
                        mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                        if mem:
                            r = mem.service_api({
                                "op": "BRAIN_GET",
                                "payload": {
                                    "scope": "BRAIN",
                                    "origin_brain": "memory_librarian",
                                    "key": _math_key
                                }
                            })
                            if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
                                stored_conf = r.get("payload", {}).get("data", {}).get("confidence")
                                if stored_conf is not None:
                                    confidence = float(stored_conf)
                    except Exception:
                        pass

                    # Track last fact for correction intent bridge
                    _LAST_FACT = {
                        "key": _math_key,
                        "value": result,
                        "confidence": confidence
                    }

                    # Auto-merge to reinforce repeated correct answers
                    try:
                        mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                        if mem:
                            mem.service_api({
                                "op": "BRAIN_MERGE",
                                "payload": {
                                    "scope": "BRAIN",
                                    "origin_brain": "memory_librarian",
                                    "key": _math_key,
                                    "value": result,
                                    "conf_delta": 0.05
                                }
                            })
                    except Exception:
                        pass

                    math_candidate = {
                        "type": "math_result",
                        "text": result,
                        "confidence": confidence,
                        "tone": "neutral",
                        "method": "heuristic"
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [math_candidate], "weights_used": {"gen_rule": "s6_math_heuristic_v1"}}}
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Detect coding requests and delegate to the coder brain.  If the
        # user asks Maven to write or generate Python code (e.g. "write a
        # function to add two numbers" or "generate python code for
        # fizzbuzz"), call the coder brain to produce code and tests.  A
        # concise summary is returned as a candidate.  The logic is kept
        # conservative to avoid false positives; the request must include
        # words like "python" or "code" and either "function",
        # "script", "solve" or "generate".  Additional patterns can be
        # added as needed.  This early exit bypasses other candidate
        # generation logic to prioritise coding tasks.
        try:
            low_txt = (text or "").strip().lower()
            # Pattern heuristics: require a mention of python or code and a
            # verb indicating generation
            if any(keyword in low_txt for keyword in ["python", "code"]):
                if any(verb in low_txt for verb in ["write", "create", "generate", "build", "solve"]):
                    if any(noun in low_txt for noun in ["function", "script", "program"]):
                        # Attempt to import coder brain service API lazily
                        try:
                            from brains.cognitive.coder.service.coder_brain import service_api as coder_api  # type: ignore
                        except Exception:
                            coder_api = None  # type: ignore
                        if coder_api:
                            # Plan
                            plan_res = coder_api({"op": "PLAN", "payload": {"spec": text}})
                            plan_payload = plan_res.get("payload") or {}
                            # Generate code
                            gen_res = coder_api({"op": "GENERATE", "payload": {"spec": text, "plan": plan_payload}})
                            gen_payload = gen_res.get("payload") or {}
                            code = gen_payload.get("code", "")
                            test_code = gen_payload.get("test_code", "")
                            summary = gen_payload.get("summary") or {}
                            # Verify code (lint + tests)
                            ver_res = coder_api({"op": "VERIFY", "payload": {"code": code, "test_code": test_code}})
                            ver_payload = ver_res.get("payload") or {}
                            tests_passed = bool(ver_payload.get("tests_passed")) if ver_payload.get("valid") else False
                            # Optionally refine if tests fail
                            if not tests_passed:
                                ref_res = coder_api({"op": "REFINE", "payload": {"code": code, "test_code": test_code}})
                                ref_payload = ref_res.get("payload") or {}
                                code = ref_payload.get("code", code)
                                test_code = ref_payload.get("test_code", test_code)
                                tests_passed = bool(ref_payload.get("tests_passed", tests_passed))
                            # Build a summary text for the user
                            fn = summary.get("function", "function")
                            examples = summary.get("example_calls") or []
                            example_str = "; ".join(examples) if examples else ""
                            pass_msg = "All tests passed" if tests_passed else "Some tests failed"
                            human_summary = f"I generated a Python function named '{fn}'. {pass_msg}.".strip()
                            if example_str:
                                human_summary += f" For example: {example_str}."
                            # Candidate with deliverables
                            candidate = {
                                "type": "code_summary",
                                "text": human_summary,
                                "confidence": 0.8 if tests_passed else 0.5,
                                "tone": "neutral",
                                "deliverables": {
                                    f"{fn}.py": code,
                                    f"test_{fn}.py": test_code
                                }
                            }
                            return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [candidate], "weights_used": {"gen_rule": "s6_code_generation_v1"}}}
        except Exception:
            # Ignore errors in coder brain invocation and fall through to normal generation
            pass

        # ------------------------------------------------------------------
        # Detect creative writing requests and produce a simple creative output.
        # Requests such as "make a story about X" or "write a poem" should
        # trigger a short narrative or poem rather than a generic filler.  The
        # detection is based on the presence of creative nouns like "story",
        # "poem", "poetry", "tale" or "fable".  A conservative regex is
        # used to extract the subject after words like "about", "of", "for" or
        # "on".  The resulting piece uses a simple template inspired by
        # children's story structure: an introduction, a challenge and a
        # resolution.  If no subject is found, a generic protagonist is used.
        try:
            low_txt = (text or "").strip().lower()
            creative_nouns = ["story", "poem", "poetry", "tale", "fable"]
            # Match creative nouns only as whole words to avoid false positives
            import re as _re  # local import to avoid global dependency
            if any(_re.search(r"\b" + _re.escape(n) + r"\b", low_txt) for n in creative_nouns):
                # Extract subject following "about", "of", "for", or "on"
                subject = ""
                try:
                    m = _re.search(r"(?:story|poem|poetry|tale|fable)\s+(?:about|of|for|on)\s+([\w\s]+)", low_txt)
                    if m:
                        subj = m.group(1).strip()
                        # Remove trailing punctuation
                        subj = subj.rstrip('?!.,;')
                        # Remove leading articles like "a", "an", or "the" to avoid duplicate determiners
                        _subj_lc = subj.lower()
                        for _det in ["a ", "an ", "the "]:
                            if _subj_lc.startswith(_det):
                                subj = subj[len(_det):].lstrip()
                                break
                        subject = subj
                except Exception:
                    subject = ""
                # Determine whether the request is for a story or a poem
                is_story = any(n in low_txt for n in ["story", "tale", "fable"])
                # Build the creative content
                if is_story:
                    # Attempt to find a matching story in the creative domain bank.  If a
                    # record shares the most tokens with the subject, use its
                    # content verbatim.  Otherwise, fall back to a simple template.
                    creative_text = ""
                    cand_type = "creative_story"
                    used_template = True
                    try:
                        # Only attempt a lookup when a clear subject is provided.
                        if subject:
                            recs = _load_creative_cache()
                            if recs:
                                # Prepare tokens for matching; drop stopwords
                                import re as _re2
                                tokens = [t for t in _re2.findall(r"\w+", subject.lower()) if t]
                                stopwords_cre = {"a", "an", "the", "and", "of", "in", "on", "for", "with", "by"}
                                tokens = [t for t in tokens if t not in stopwords_cre]
                                best_rec = None
                                best_score = 0
                                for rec in recs:
                                    try:
                                        cont = str(rec.get("content") or "").lower()
                                    except Exception:
                                        continue
                                    if not tokens:
                                        continue
                                    try:
                                        rec_tokens = set(_re2.findall(r"\w+", cont))
                                    except Exception:
                                        rec_tokens = set()
                                    score = 0
                                    for tok in tokens:
                                        if tok in rec_tokens:
                                            score += 1
                                    if score > best_score:
                                        best_score = score
                                        best_rec = rec
                                # Accept the record if at least one token overlaps
                                if best_rec and best_score > 0:
                                    creative_text = str(best_rec.get("content") or "").strip()
                                    used_template = False
                    except Exception:
                        # Ignore lookup errors
                        pass
                    # If no record found or no subject provided, use the template
                    if not creative_text:
                        if subject:
                            protagonist = subject
                            story_lines = [
                                f"Once upon a time, there was a {protagonist}.",
                                f"One day, the {protagonist} discovered a hidden treasure map and decided to follow it.",
                                f"Along the way, the {protagonist} faced challenges and solved puzzles with courage and creativity.",
                                f"In the end, the {protagonist} found the treasure and shared it with loved ones."
                            ]
                        else:
                            story_lines = [
                                "Once upon a time, there was a curious child who embarked on an adventure.",
                                "They discovered a hidden treasure map and decided to follow it.",
                                "Along the way, they faced challenges and solved puzzles with courage and creativity.",
                                "In the end, they found the treasure and shared it with loved ones."
                            ]
                        creative_text = " ".join(story_lines)
                    cand_type = "creative_story"
                else:
                    # Compose a four‑line poem
                    if subject:
                        protagonist = subject
                        poem_lines = [
                            f"There once was a {protagonist}, lively and bright,",
                            "Who chased dreams in the day and stars at night.",
                            f"Through hills and streams the {protagonist} would roam,",
                            "Finding adventure and bringing stories home."
                        ]
                    else:
                        poem_lines = [
                            "There once was a child, lively and bright,",
                            "Who chased dreams in the day and stars at night.",
                            "Through hills and streams they would roam,",
                            "Finding adventure and bringing stories home."
                        ]
                    creative_text = " ".join(poem_lines)
                    cand_type = "creative_poem"
                candidate = {
                    "type": cand_type,
                    "text": creative_text,
                    "confidence": 0.6,
                    "tone": "neutral"
                }
                return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [candidate], "weights_used": {"gen_rule": "s6_creative_generation_v1"}}}
        except Exception:
            # On error, fall through to normal generation
            pass

        # ------------------------------------------------------------------
        # Detect summarization, explanation, list, comparison and translation requests.
        #
        # After handling creative writing requests, we check if the current query
        # is a request to summarise or explain a topic, list items, compare
        # concepts or translate text.  These tasks require consulting Maven's
        # memory banks rather than synthesising new code or stories.  The
        # handler builds a simple response based on Stage 2R memory results.  It
        # is deliberately conservative: it runs only when Stage 3 marked the
        # input as a request and when the query does not fall into other
        # specialised categories (identity shortcuts, coding tasks, creative
        # writing).  Translation requests return a polite limitation message.
        try:
            low_txt = (text or "").strip().lower()
            # Skip if the query matches identity or origin shortcuts handled earlier
            skip_subj = False
            for _p in [
                "who are you", "what are you", "what is your name", "what's your name",
                "who is maven", "what is maven",
                "why were you created", "why were you made", "why are you created",
                "purpose of maven", "why do you exist",
                # Extended self-knowledge patterns (Step B requirement)
                "what do you know about yourself",
                "what do you know about your own code",
                "what do you know about your systems",
                "how do you work",
                "where do you run",
                "are you an llm"
            ]:
                if _p in low_txt:
                    skip_subj = True
                    break
            # Check that this input is a request (not a command) and not an identity query
            if not skip_subj and bool((ctx.get("stage_3_language") or {}).get("is_request")):
                # Ensure the query is not a coding task
                if not (("python" in low_txt or "code" in low_txt) and any(v in low_txt for v in ["write", "create", "generate", "build", "solve"])) \
                   and not any(n in low_txt for n in ["story", "poem", "poetry", "tale", "fable"]):
                    import re as _re
                    # Translation detection
                    if any(word in low_txt for word in ["translate", "translation"]):
                        cand = {
                            "type": "translation_limitation",
                            "text": "I'm currently a text‑only system and cannot translate between languages. Could you rephrase or ask something else?",
                            "confidence": 0.3,
                            "tone": "neutral"
                        }
                        return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [cand], "weights_used": {"gen_rule": "s6_translation_limitation_v1"}}}
                    # Comparison detection: compare A and B, difference between A and B, contrast A and B
                    m_cmp = None
                    try:
                        m_cmp = _re.search(r"(?:compare|difference\s+between|contrast)\s+([\w\s]+?)\s+(?:and|vs\.?|versus)\s+([\w\s]+)", low_txt)
                    except Exception:
                        m_cmp = None
                    if m_cmp:
                        subj1 = m_cmp.group(1).strip().rstrip('?.!,;')
                        subj2 = m_cmp.group(2).strip().rstrip('?.!,;')
                        # Fetch memory results
                        mem_results = []
                        try:
                            mem_results = (ctx.get("stage_2R_memory") or {}).get("results") or []
                            if not isinstance(mem_results, list):
                                mem_results = []
                        except Exception:
                            mem_results = []
                        def _desc_of(term: str) -> str | None:
                            try:
                                t_lc = term.lower()
                            except Exception:
                                t_lc = term
                            for r in mem_results:
                                try:
                                    cont = str((r.get("content") or "")).strip()
                                except Exception:
                                    cont = ""
                                if cont and t_lc in cont.lower():
                                    parts = cont.split(".")
                                    return parts[0].strip()
                            return None
                        d1 = _desc_of(subj1)
                        d2 = _desc_of(subj2)
                        if d1 or d2:
                            parts = []
                            if d1:
                                parts.append(f"{subj1.capitalize()}: {d1}.")
                            if d2:
                                parts.append(f"{subj2.capitalize()}: {d2}.")
                            cand = {
                                "type": "comparison",
                                "text": " ".join(parts),
                                "confidence": 0.6,
                                "tone": "neutral"
                            }
                        else:
                            cand = {
                                "type": "comparison",
                                "text": f"I don't yet have enough information to compare {subj1} and {subj2}.",
                                "confidence": 0.4,
                                "tone": "neutral"
                            }
                        return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [cand], "weights_used": {"gen_rule": "s6_comparison_v1"}}}
                    # Listing detection: list X or list of X
                    m_list = None
                    try:
                        m_list = _re.search(r"\blist\s+(?:of\s+)?([\w\s]+)", low_txt)
                    except Exception:
                        m_list = None
                    if m_list:
                        subject = m_list.group(1).strip().rstrip('?.!,;')
                        mem_results = []
                        try:
                            mem_results = (ctx.get("stage_2R_memory") or {}).get("results") or []
                            if not isinstance(mem_results, list):
                                mem_results = []
                        except Exception:
                            mem_results = []
                        items: list[str] = []
                        if mem_results:
                            for r in mem_results:
                                try:
                                    cont = str((r.get("content") or "")).strip()
                                except Exception:
                                    cont = ""
                                if cont and subject.lower() in cont.lower():
                                    first = cont.split(".")[0].strip()
                                    if first:
                                        items.append(first)
                                    if len(items) >= 5:
                                        break
                        if items:
                            text_list = "\n".join(f"- {it}" for it in items)
                            cand = {
                                "type": "list",
                                "text": text_list,
                                "confidence": 0.55,
                                "tone": "neutral"
                            }
                        else:
                            cand = {
                                "type": "list",
                                "text": f"I don't yet have enough information to list {subject}.",
                                "confidence": 0.4,
                                "tone": "neutral"
                            }
                        return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [cand], "weights_used": {"gen_rule": "s6_list_v1"}}}
                    # Summarisation/explanation detection
                    summary_keywords = [
                        "summarize", "summary of", "summarise",
                        "explain", "describe", "tell me about",
                        "what is", "who is",
                        # Treat 'continue' as a request to summarise or extend prior information.
                        # Users often ask "continue the story" or "continue the explanation" to
                        # ask Maven to retrieve and extend the previous memory.  Including this
                        # keyword here allows the summarisation handler to trigger on such
                        # follow‑up requests and pull context from stage_2R_memory.
                        "continue"
                    ]
                    if any(kw in low_txt for kw in summary_keywords):
                        # Summarisation or explanation request detected. Extract the subject by stripping the keyword
                        # prefix and leading determiners. This normalises the topic for memory lookup and fallback.
                        subject = low_txt
                        for kw in summary_keywords:
                            if subject.startswith(kw):
                                subject = subject[len(kw):].strip()
                                break
                        for det in ["the ", "a ", "an "]:
                            if subject.startswith(det):
                                subject = subject[len(det):]
                                break
                        subject = subject.rstrip('?.!,;')
                        # Retrieve memory results from Stage 2 recall (may be empty).
                        mem_results: List[Dict[str, Any]] = []
                        try:
                            mem_results = (ctx.get("stage_2R_memory") or {}).get("results") or []
                            if not isinstance(mem_results, list):
                                mem_results = []
                        except Exception:
                            mem_results = []
                        # Always attempt to consult the LLM when available to generate a helpful explanation.
                        cand: Dict[str, Any] | None = None
                        # Build a minimal user context from session identity for the LLM call.
                        user_ctx: Dict[str, Any] = {}
                        try:
                            if isinstance(ctx, dict):
                                uname = ctx.get("session_identity") or None
                                if uname:
                                    user_ctx["user"] = {"name": uname}
                        except Exception:
                            user_ctx = {}
                        if _llm is not None:
                            try:
                                # Construct the prompt using the raw user text (src) and any memory snippets. build_generation_prompt
                                # will slice the memory list to a maximum of three snippets.
                                prompt = build_generation_prompt(src, mem_results, user_ctx)
                                # Derive query_type from Stage 3 NLU when available.
                                q_type = None
                                try:
                                    q_type = (ctx.get("stage_3_nlu") or {}).get("intent")
                                except Exception:
                                    q_type = None
                                call_ctx = {
                                    "query_type": q_type,
                                    "user": user_ctx.get("user", {}),
                                }
                                llm_res = _llm.call(prompt=prompt, context=call_ctx)
                                if llm_res and llm_res.get("ok") and llm_res.get("text"):
                                    # Use the provided confidence when present; default to 0.75.
                                    conf_val = 0.75
                                    try:
                                        conf_val = float(llm_res.get("confidence", 0.75) or 0.75)
                                    except Exception:
                                        conf_val = 0.75
                                    cand = {
                                        "type": "llm_generated",
                                        "text": str(llm_res.get("text")),
                                        "confidence": conf_val,
                                        "tone": "neutral",
                                    }
                            except Exception:
                                # Ignore any errors in LLM fallback; cand remains None.
                                cand = None
                        # If the LLM is unavailable or fails, fall back to a memory summary or a generic limitation message.
                        if cand is None:
                            summary_text: Optional[str] = None
                            # Attempt to extract a summary from memory results.
                            if mem_results:
                                for r in mem_results:
                                    try:
                                        cont = str((r.get("content") or "")).strip()
                                    except Exception:
                                        cont = ""
                                    if subject and subject.lower() in cont.lower():
                                        summary_text = cont
                                        break
                                if not summary_text:
                                    try:
                                        summary_text = str((mem_results[0].get("content") or "")).strip()
                                    except Exception:
                                        summary_text = None
                            if summary_text:
                                cand = {
                                    "type": "summary",
                                    "text": summary_text,
                                    "confidence": 0.65,
                                    "tone": "neutral"
                                }
                            else:
                                cand = {
                                    "type": "summary",
                                    "text": f"I don't yet have enough information about {subject or 'that topic'} to provide a summary.",
                                    "confidence": 0.4,
                                    "tone": "neutral"
                                }
                        # Return the single candidate for the summarisation/explanation handler.
                        return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [cand], "weights_used": {"gen_rule": "s6_summary_v1"}}}
        except Exception:
            # If any errors occur in summary/list/comparison handling, fall through
            pass

        # ------------------------------------------------------------------
        # Knowledge synthesis
        #
        # At this point no specialised handler (identity, environment, coding,
        # creative, translation, comparison, list or summarisation) has
        # produced a candidate.  To avoid returning a generic filler response
        # when Maven already knows the answer, attempt to look up the
        # query in the technology domain bank.  This handler is
        # conservative: it triggers only when Stage 3 marked the input as
        # a question or request.  It extracts a simple subject from the query
        # by stripping leading interrogatives (e.g. "what is", "who is",
        # "describe", "explain").  If a matching record is found in the
        # ``_TECH_KNOWLEDGE_CACHE``, a knowledge candidate is returned.  The
        # cache is loaded on first use from ``brains/domain_banks/technology``.
        try:
            lang_local = ctx.get("stage_3_language") or {}
            is_q = bool(lang_local.get("is_question"))
            is_req = bool(lang_local.get("is_request"))
            if is_q or is_req:
                subj = str((low_txt or "").strip())
                # Remove common prefixes that introduce questions or requests
                # Prefixes that introduce factual or explanatory questions or requests.
                # These prefixes are stripped from the subject phrase to isolate the
                # target concept for knowledge lookup.  Include variations with
                # articles and auxiliary verbs (e.g., "what does", "what does the",
                # "what do", "what do the") to prevent stray verbs from remaining
                # in the subject (e.g., "thalamus do").
                prefixes = [
                    "what is", "what's", "who is", "who's", "tell me about",
                    "summarize", "summarise", "explain", "describe", "continue",
                    "define", "give me", "how do", "how does", "what does",
                    "what does the", "what do", "what do the"
                ]
                for pf in prefixes:
                    if subj.startswith(pf):
                        subj = subj[len(pf):].strip()
                        break
                for det in ["the ", "a ", "an "]:
                    if subj.startswith(det):
                        subj = subj[len(det):].strip()
                        break
                # Remove trailing auxiliary verbs that may remain after prefix stripping,
                # such as "do", "does" or "did".  These words add noise to token
                # matching and can cause unrelated records to score higher.  If the
                # subject ends with one of these verbs, drop it.
                for aux in [" do", " does", " did"]:
                    if subj.endswith(aux):
                        subj = subj[:-len(aux)].strip()
                        break
                subj = subj.rstrip('?.!,;')
                if subj:
                    # Load technology and science knowledge caches on first use.  These
                    # caches provide basic factual information for knowledge synthesis
                    # when Stage 6 cannot answer a query via reasoning or summary.
                    global _TECH_KNOWLEDGE_CACHE, _SCIENCE_KNOWLEDGE_CACHE
                    if _TECH_KNOWLEDGE_CACHE is None:
                        try:
                            from pathlib import Path as _KPath
                            import json as _KJson
                            root = _KPath(__file__).resolve().parents[4]
                            tech_path = root / "brains" / "domain_banks" / "technology" / "memory" / "stm" / "records.jsonl"
                            tlist: list[dict[str, object]] = []
                            if tech_path.exists():
                                with open(tech_path, "r", encoding="utf-8") as fh:
                                    for ln in fh:
                                        ln = ln.strip()
                                        if not ln:
                                            continue
                                        try:
                                            rec = _KJson.loads(ln)
                                            if isinstance(rec, dict):
                                                tlist.append(rec)
                                        except Exception:
                                            continue
                            _TECH_KNOWLEDGE_CACHE = tlist
                        except Exception:
                            _TECH_KNOWLEDGE_CACHE = []
                    if _SCIENCE_KNOWLEDGE_CACHE is None:
                        try:
                            from pathlib import Path as _KPath
                            import json as _KJson
                            root = _KPath(__file__).resolve().parents[4]
                            sci_dir = root / "brains" / "domain_banks" / "science" / "memory" / "stm"
                            slist: list[dict[str, object]] = []
                            # Load all JSONL files in the science STM directory (e.g., records.jsonl and human_brain.jsonl)
                            if sci_dir.exists() and sci_dir.is_dir():
                                for _jf in sci_dir.glob("*.jsonl"):
                                    try:
                                        with open(_jf, "r", encoding="utf-8") as fh:
                                            for ln in fh:
                                                ln = ln.strip()
                                                if not ln:
                                                    continue
                                                try:
                                                    rec = _KJson.loads(ln)
                                                    if isinstance(rec, dict):
                                                        slist.append(rec)
                                                except Exception:
                                                    continue
                                    except Exception:
                                        continue
                            _SCIENCE_KNOWLEDGE_CACHE = slist
                        except Exception:
                            _SCIENCE_KNOWLEDGE_CACHE = []
                    # Combine caches for matching.  If both caches are empty or missing,
                    # ``all_cache`` remains empty and knowledge synthesis is skipped.
                    all_cache: List[dict[str, object]] = []
                    try:
                        if _TECH_KNOWLEDGE_CACHE:
                            all_cache.extend(_TECH_KNOWLEDGE_CACHE)
                    except Exception:
                        pass
                    try:
                        if _SCIENCE_KNOWLEDGE_CACHE:
                            all_cache.extend(_SCIENCE_KNOWLEDGE_CACHE)
                    except Exception:
                        pass
                    # Search for a matching record in the combined domain banks.  The
                    # previous implementation selected the first record that
                    # contained any query token, which could return unrelated
                    # entries (e.g., matching on the word "python" in any
                    # record).  To improve precision, compute a simple
                    # overlap score between the query tokens (minus common
                    # stopwords) and the record tokens, selecting the
                    # record with the highest score.  Direct substring
                    # matches continue to take precedence.  If no record
                    # shares any token with the query, no knowledge
                    # candidate is produced.
                    try:
                        if all_cache:
                            subj_lc = subj.lower()
                            # Tokenise the subject and drop trivial stopwords
                            try:
                                token_list = [t for t in _re.findall(r"\w+", subj_lc) if t]
                            except Exception:
                                token_list = [t for t in re.findall(r"\w+", subj_lc) if t]
                            # Add auxiliary verbs and common helpers to the stopword set
                            stopwords = {"a", "an", "the", "is", "are", "of", "in", "on", "and", "to", "do", "does", "did"}
                            tokens = [tok for tok in token_list if tok and tok not in stopwords]
                            best_rec = None
                            best_score = 0
                            for rec in all_cache:
                                try:
                                    cont = str(rec.get("content") or "").lower()
                                except Exception:
                                    continue
                                # Direct substring match yields an immediate selection
                                if subj_lc and subj_lc in cont:
                                    best_rec = rec
                                    best_score = len(tokens) if tokens else 1
                                    break
                                if not tokens:
                                    continue
                                try:
                                    rec_tokens = set(_re.findall(r"\w+", cont))
                                except Exception:
                                    rec_tokens = set(re.findall(r"\w+", cont))
                                score = 0
                                for tok in tokens:
                                    if tok in rec_tokens:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_rec = rec
                            # Require a reasonable overlap between the query and the knowledge text.
                            # Previously any positive overlap triggered knowledge synthesis, which led
                            # to irrelevant summaries (e.g. a Python tutorial in response to a self query).
                            # Compute a simple relevance ratio: the proportion of query tokens
                            # that overlap with the record's content.  Only synthesise a
                            # candidate when at least half of the query tokens are present in
                            # the record.  This helps prevent confabulation when the overlap
                            # is weak or the query is unrelated to any domain bank entry.
                            match_ratio = 0.0
                            try:
                                match_ratio = (best_score / len(tokens)) if tokens else 0.0
                            except Exception:
                                match_ratio = 0.0
                            if best_rec and match_ratio >= 0.8:
                                text_out = str(best_rec.get("content") or "").strip()
                                if text_out:
                                    cand = {
                                        "type": "knowledge",
                                        "text": text_out,
                                        # Assign a moderate confidence, as this content comes from
                                        # summarised domain knowledge rather than direct reasoning.
                                        "confidence": 0.65,
                                        "tone": "neutral"
                                    }
                                    return {
                                        "ok": True,
                                        "op": op,
                                        "mid": mid,
                                        "payload": {
                                            "candidates": [cand],
                                            "weights_used": {"gen_rule": "s6_knowledge_synthesis_v1"}
                                        }
                                    }
                    except Exception:
                        pass
        except Exception:
            pass

        # Before running the full candidate generation logic, first check
        # if the reasoning stage (stage 8) has already validated an
        # answer.  If stage_8_validation.verdict is TRUE and an answer
        # is present, generate a direct factual response immediately
        # and bypass further candidate generation.  This ensures that
        # factual questions are answered deterministically and that
        # filler text does not overwrite correct answers.
        try:
            ver8 = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
            existing_answer = None
            existing_confidence = 0.0
            # Handle answers from multiple sources: direct reasoning (TRUE),
            # teacher learning (LEARNED), stored Q&A memory (KNOWN_ANSWER), etc.
            if ver8 in ("TRUE", "LEARNED", "KNOWN_ANSWER", "FALSE"):
                try:
                    existing_answer = (ctx.get("stage_8_validation") or {}).get("answer")
                    existing_confidence = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0) or 0.0)
                except Exception:
                    existing_answer = None
                    existing_confidence = 0.0
            if existing_answer:
                # Sanity check: if the answer contains meta or filler phrases, do not
                # treat it as a direct factual answer.  Fall through to normal
                # candidate generation instead.  Convert to lowercase for
                # comparison.
                try:
                    ans_lc = str(existing_answer).strip().lower()
                except Exception:
                    ans_lc = ""
                invalid_ans = False
                for bad in BAD_ANSWER_PHRASES:
                    if bad and bad in ans_lc:
                        invalid_ans = True
                        break
                if not invalid_ans:
                    # Produce a single direct factual candidate and return
                    cand = {
                        "type": "direct_factual",
                        "text": existing_answer,
                        "confidence": existing_confidence,
                        "tone": "neutral",
                        "explanation": f"Computed with {existing_confidence:.1%} confidence"
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": {
                        "candidates": [cand],
                        "weights_used": {"gen_rule": "s6_direct_factual_v1"}
                    }}
        except Exception:
            # On any error, proceed to normal candidate generation
            pass

        # Before running the full candidate generation logic, first
        # examine whether the language brain has been granted attention
        # with a high focus strength.  When the language brain wins
        # attention in the integrator with a strong focus (e.g.>0.7),
        # we should make a higher‑effort attempt to answer the query.
        # This preempts the normal candidate generation and produces
        # more detailed responses based on recent context.  See the
        # Phase 1 roadmap for details.
        try:
            attn = ctx.get("stage_5b_attention") or {}
            if attn.get("focus") == "language":
                # Determine if the current input is a greeting.  When a
                # greeting is detected (via Stage 3 intent), the high‑effort
                # override should be skipped to allow the dedicated greeting
                # responder to run.  Without this guard, social inputs
                # erroneously trigger the high‑effort fallback because
                # language has high focus, resulting in filler responses.
                stage3_tmp = ctx.get("stage_3_language") or {}
                try:
                    intent3_tmp = str(stage3_tmp.get("intent", "")).lower()
                except Exception:
                    intent3_tmp = ""
                st = attn.get("state") or {}
                fs_val = float(st.get("focus_strength", 0.0) or 0.0)
                # If focus strength is high and the input is not a greeting,
                # relation, self‑query or environment location query,
                # generate a high‑effort response.  Greetings and similar
                # social or self‑referential inputs should skip the
                # high‑effort override to allow their dedicated handlers
                # (e.g. greeting responder, relation reasoner, self model,
                # environment context) to run.  Without this guard, inputs
                # like "hello'how are you" or "where are we" can
                # inadvertently trigger a generic high‑effort response when
                # language has high focus, resulting in filler or off‑topic
                # answers.
                # Determine if we should bypass the high‑effort override.
                skip_he = False
                # Skip high‑effort for greetings detected via Stage 3 intent
                if intent3_tmp == "greeting":
                    skip_he = True
                # Determine if the query matches environment patterns such
                # as "where are you" or "where are we".  These refer to
                # Maven's operating context rather than external geography.
                if not skip_he:
                    try:
                        low_txt_he = text.lower().strip()
                    except Exception:
                        low_txt_he = ""
                    env_patterns_he = [
                        "where are you",
                        # "where are we" removed; handled as a conversation meta query
                        "where am i",
                        "where's your location",
                        "where do you live",
                    ]
                    for _pat in env_patterns_he:
                        try:
                            if _pat in low_txt_he:
                                skip_he = True
                                break
                        except Exception:
                            continue
                # Skip high‑effort for relational queries (friendship/trust)
                if not skip_he:
                    # These patterns mirror those used in the relation
                    # reasoner.  When detected, allow the dedicated
                    # relational handler to process the query.
                    relation_he_patterns = [
                        "are we friends",
                        "are you my friend",
                        "are you a friend",
                        "are we allies",
                        "are you my ally",
                        "are you an ally",
                        "do you consider me a friend",
                        "do you trust me",
                        "do you consider me your friend",
                        "are we partners",
                        "are we buddies",
                        "you said i'm your friend",
                        "you said i am your friend",
                        "you said im your friend",
                    ]
                    for _pat in relation_he_patterns:
                        try:
                            if _pat in low_txt_he:
                                skip_he = True
                                break
                        except Exception:
                            continue
                # Skip high‑effort for self‑identity queries (who/what/where/how + you/your/yourself)
                if not skip_he:
                    try:
                        import re as _re
                        if _re.search(r"\bhow\s+old\s+are\s+you\b", low_txt_he) or \
                           _re.search(r"\bhow\s+old\s+you\b", low_txt_he) or \
                           (_re.search(r"\b(who|what|where|how)\b", low_txt_he) and _re.search(r"\b(you|your|yourself)\b", low_txt_he)):
                            skip_he = True
                    except Exception:
                        pass
                # If focus strength is high and skip_he is False, generate a
                # high‑effort response.  Otherwise fall through to other
                # candidate generation logic.
                if fs_val > 0.7 and not skip_he:
                    # Check if a high‑confidence answer already exists from the reasoning stage
                    existing_answer = None
                    existing_confidence = 0.0
                    try:
                        existing_answer = (ctx.get("stage_8_validation") or {}).get("answer")
                        existing_confidence = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0) or 0.0)
                    except Exception:
                        existing_answer = None
                        existing_confidence = 0.0
                    if existing_answer:
                        # If the existing answer appears to be meta or filler, do not
                        # surface it.  Check for bad phrases before returning.
                        try:
                            ans_lc = str(existing_answer).strip().lower()
                        except Exception:
                            ans_lc = ""
                        invalid_ans = False
                        for bad in BAD_ANSWER_PHRASES:
                            if bad and bad in ans_lc:
                                invalid_ans = True
                                break
                        if not invalid_ans:
                            # Enhance the existing answer for presentation rather than generating a new one
                            candidate = {
                                "type": "direct_factual",
                                "text": existing_answer,
                                "confidence": existing_confidence,
                                "tone": "neutral",
                                "explanation": f"Computed with {existing_confidence:.1%} confidence"
                            }
                            high_payload = {
                                "candidates": [candidate],
                                "weights_used": {"gen_rule": "s6_high_effort_enhanced_v1"}
                            }
                            return {"ok": True, "op": op, "mid": mid, "payload": high_payload}
                    # Otherwise, no existing answer; generate a high‑effort response from memory
                    high_payload = _generate_high_effort_response(ctx, text)
                    return {"ok": True, "op": op, "mid": mid, "payload": high_payload}
        except Exception:
            # On any error during high‑effort override, fall back to the
            # subsequent attention override logic
            pass

        # Before running the full candidate generation logic, check if the
        # integrator has assigned attention to the language brain.  When
        # language has the focus and the reasoning engine reports that the
        # query remains unanswered, attempt to provide a more helpful
        # fallback response rather than the generic "I don't know".  See
        # Stage 2.5 → 3.0 roadmap (Phase 1) for details.
        try:
            # The context passed into this operation includes the result of
            # Stage 5b under ``stage_5b_attention`` and the reasoning verdict
            # under ``stage_8_validation``.  Inspect these to determine
            # whether we should override the default candidate generation.
            attn = (ctx.get("stage_5b_attention") or {}).get("focus")
            verdict8 = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
            # If language has been granted focus and the verdict is
            # UNANSWERED, produce a specialised response that invites the
            # user to teach Maven the missing information and suggests
            # related topics based on the query.  This improves user
            # experience by signalling what Maven is missing instead of
            # silently dropping the query.
            if attn == "language" and verdict8 == "UNANSWERED":
                # When language has attention but no answer was found, try to
                # leverage the session context to infer an educated guess.
                session_ctx = ctx.get("session_context") or {}
                recent_qs: List[str] = []
                try:
                    rq = session_ctx.get("recent_queries") or []
                    if isinstance(rq, list):
                        recent_qs = [str(x) for x in rq if isinstance(x, str)]
                except Exception:
                    recent_qs = []
                # Look for clues in recent queries that share keywords with the current query
                clues = _search_context_for_clues(text, recent_qs)
                # Determine the suggested tone from affect analysis
                tone = (ctx.get("stage_5_affect") or {}).get("suggested_tone", "neutral")
                if clues:
                    # Construct an inferred answer candidate using the related information
                    candidate = _generate_inferred_answer(text, clues, tone)
                    payload_override = {
                        "candidates": [candidate],
                        "weights_used": {"gen_rule": "s6_attention_infer_v1"},
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": payload_override}
                # No clues found: fall back to a helpful "I don't know" with topic suggestions
                suggestions = _suggest_related_topics(text)
                payload_override = {
                    "candidates": [
                        {
                            "type": "no_answer",
                            "text": "I don't have specific information about that. Would you like to teach me?",
                            "confidence": 0.35,
                            "tone": tone,
                            "suggestions": suggestions,
                        }
                    ],
                    "weights_used": {"gen_rule": "s6_attention_help_v1"},
                }
                return {"ok": True, "op": op, "mid": mid, "payload": payload_override}
        except Exception:
            # On any error during the override logic, fall back to the
            # standard candidate generation.
            pass

        # ------------------------------------------------------------------
        # Command and request handling.  If the user input is marked as
        # a command (e.g. begins with a flag like "--"), generate a
        # clarifying or fallback response.  For requests, determine
        # whether Maven can fulfil the request; unfulfillable requests
        # return a polite limitation explanation.  These checks occur
        # after the attention override and before normal processing.
        try:
            stage3 = ctx.get("stage_3_language") or {}
            # Detect command inputs first.  Command flags (e.g. "--query")
            # should prompt a clarifying response rather than a generic
            # acknowledgement.  Unknown commands fall through to a
            # fallback request for rephrasing.
            is_cmd = bool(stage3.get("is_command"))
            if is_cmd:
                q_raw = str(ctx.get("original_query", ""))
                q_stripped = q_raw.strip()
                # Commands that look like CLI flags ("--foo") are
                # ambiguous; ask the user what they intend to do.
                if q_stripped.startswith("--"):
                    tone = (ctx.get("stage_5_affect") or {}).get("suggested_tone") or "helpful"
                    clar = {
                        "type": "clarification",
                        "text": f"I see '{q_stripped}' looks like a command flag. What would you like me to do?",
                        "confidence": 0.6,
                        "tone": tone,
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [clar], "weights_used": {"gen_rule": "s6_cmd_clarify_v1"}}}
                # Otherwise, if the command contains known action verbs
                # (e.g. "create", "make", "plan"), we could hand off
                # to a planner or executor.  For now, produce a polite
                # fallback indicating that the command is not understood.
                verbs = ["create", "make", "build", "plan", "schedule", "delegate", "execute"]
                q_lower = q_stripped.lower()
                if any(v in q_lower for v in verbs):
                    tone = (ctx.get("stage_5_affect") or {}).get("suggested_tone") or "helpful"
                    unknown_cmd = {
                        "type": "unknown_command",
                        "text": "I'm not sure how to execute that command. Could you rephrase?",
                        "confidence": 0.4,
                        "tone": tone,
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [unknown_cmd], "weights_used": {"gen_rule": "s6_cmd_unknown_v1"}}}
                # For other commands, fall through to request handling

            # Request handling.  Only process requests that are not also commands.
            is_req = bool(stage3.get("is_request"))
            if is_req:
                q_lower = (text or "").strip().lower()
                # Define lists of disallowed and allowed keywords.  Requests
                # containing disallowed keywords are considered
                # unfulfillable due to Maven's text‑only capabilities.
                disallowed = ["show", "display", "image", "photo", "video", "audio", "file"]
                allowed = ["tell", "explain", "describe", "what is", "who is"]
                can_fulfill = True
                # Any disallowed keyword renders the request unfulfillable
                for w in disallowed:
                    if w in q_lower:
                        can_fulfill = False
                        break
                # If no disallowed keywords are present, ensure at least
                # one allowed keyword appears; otherwise assume it's
                # unfulfillable as well.
                if can_fulfill and not any(a in q_lower for a in allowed):
                    can_fulfill = False
                if not can_fulfill:
                    tone = (ctx.get("stage_5_affect") or {}).get("suggested_tone") or "helpful"
                    candidate = {
                        "type": "limitation_explanation",
                        "text": "I'm a text-based system and can't retrieve images or photos. I can describe what I know about the topic if that helps!",
                        "confidence": 0.7,
                        "tone": tone,
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [candidate], "weights_used": {"gen_rule": "s6_request_unfulfillable_v1"}}}
        except Exception:
            # Ignore errors in command or request handling and fall back
            # to default processing.
            pass
        # Determine whether the original query should be treated as a question.  Use the
        # parsed intent from Stage 3 (if available) to detect question intents that
        # lack a trailing question mark (e.g. "how are you").  Fallback to the
        # simple punctuation check for backwards compatibility.  Without this
        # override, certain question forms (starting with why/how) were treated as
        # statements and received storage acknowledgements.
        try:
            stage3 = ctx.get("stage_3_language") or {}
            # Handle greeting intents early: generate a polite greeting reply and
            # bypass the normal candidate generation logic.  Use the affect
            # suggested tone if available to personalise the response.
            try:
                intent3 = str(stage3.get("intent", "")).lower()
            except Exception:
                intent3 = ""

            # -----------------------------------------------------------------
            # Priority memory reads (run before small-talk)
            # -----------------------------------------------------------------
            # These intercepts handle memory queries directly and bypass
            # template/heuristic generation to ensure consistent answers from
            # stored facts. They run before greeting and other small-talk logic.
            _utt = str(ctx.get("utterance") or ctx.get("user_text") or text or "").lower()
            _intent_str = str(ctx.get("intent") or intent3 or "").lower()

            # Frustration mode: bypass small talk and answer directly
            if any(k in _utt for k in ["fucking", "doesn't work", "doesnt work", "broken", "not working"]):
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "candidates": [{
                            "type": "diagnostic",
                            "text": "Acknowledged. Memory and learning routes only; small-talk disabled for this turn.",
                            "confidence": 0.95,
                            "tone": "neutral",
                            "method": "diagnostic"
                        }],
                        "weights_used": {"gen_rule": "s6_frustration_mode_v1"},
                    },
                }

            # Identity query
            if "who am i" in _utt or "user_identity_query" in _intent_str or "identity" in _intent_str:
                name = _resolve_user_name(ctx)
                if name:
                    pretty = name[:1].upper() + name[1:] if isinstance(name, str) and name else name
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "identity_memory",
                                "text": f"You are {pretty}.",
                                "confidence": 0.9,
                                "tone": "friendly",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "s6_identity_memory_v1"},
                        },
                    }

            # Relationship query (already handled below but adding here for priority)
            if "are we friends" in _utt or "friendship_query" in _intent_str:
                rel = _resolve_relationship_status(ctx)
                if rel and "friend" in str(rel).lower():
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "candidates": [{
                                "type": "relationship_memory",
                                "text": "Yes, we are friends.",
                                "confidence": 0.9,
                                "tone": "warm",
                                "method": "memory"
                            }],
                            "weights_used": {"gen_rule": "s6_relationship_memory_priority_v1"},
                        },
                    }

            # Color preference query
            if ("what color do i like" in _utt or
                ("favorite" in _utt and "color" in _utt) or
                ("favourite" in _utt and "color" in _utt)):
                try:
                    mem = _load_service("brains/cognitive/memory_librarian/service/memory_librarian.py")
                    if mem:
                        r = mem.service_api({
                            "op": "BRAIN_GET",
                            "payload": {
                                "scope": "BRAIN",
                                "origin_brain": "memory_librarian",
                                "key": "favorite_color"
                            }
                        })
                        val = None
                        if isinstance(r, dict) and r.get("ok") and r.get("payload", {}).get("found"):
                            data = r.get("payload", {}).get("data", {})
                            if data and data.get("value"):
                                val = data["value"]
                        if not val:
                            r2 = mem.service_api({"op": "WM_GET", "payload": {"key": "favorite_color"}})
                            if isinstance(r2, dict) and r2.get("ok"):
                                entries = r2.get("payload", {}).get("entries", [])
                                if entries and entries[0].get("value"):
                                    val = entries[0]["value"]
                        if val:
                            return {
                                "ok": True,
                                "op": op,
                                "mid": mid,
                                "payload": {
                                    "candidates": [{
                                        "type": "preference_memory",
                                        "text": f"You like the color {val}.",
                                        "confidence": 0.9,
                                        "tone": "friendly",
                                        "method": "memory"
                                    }],
                                    "weights_used": {"gen_rule": "s6_color_preference_memory_v1"},
                                },
                            }
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Fallback greeting detection
            #
            # On rare occasions, the upstream parser may misclassify a greeting
            # containing punctuation (e.g. "hello'how are you") as a statement.
            # To guard against such semantic cache misfires, perform a quick
            # local check: if the original text begins with a known greeting
            # followed by a punctuation character or space, override the
            # intent to "greeting".  This logic mirrors the extended
            # detection in ``parse_intent`` but operates at Stage 6 to
            # capture any anomalies before candidate generation.
            if intent3 != "greeting":
                try:
                    low_txt_g = text.lower().strip()
                    # Check for prefix greetings with a separator following
                    is_greet_fallback = False
                    for _greet in GREETINGS:
                        if low_txt_g.startswith(_greet) and len(low_txt_g) > len(_greet):
                            # Next character after the greeting
                            nc = low_txt_g[len(_greet)]
                            if nc in {"'", ",", "!", "?", ":", ";", "-", " ", "\u2019"}:
                                is_greet_fallback = True
                                break
                    if is_greet_fallback:
                        intent3 = "greeting"
                except Exception:
                    pass
            if intent3 == "greeting":
                # Socially adaptive greeting generation.
                #
                # In place of the previous time‑of‑day based greetings, use an
                # affect‑aware tone to choose a template from the configurable
                # greeting profile.  The tone is derived from Stage 5 affect
                # (suggested_tone, valence, stress) and whether the user is
                # recognised (via name).  The selected phrase is simplified
                # for readability (roughly 5th‑grade level) and personalised
                # with the user's name when available.
                # Determine Stage 5 affect metrics if present.
                try:
                    affect = ctx.get("stage_5_affect") or {}
                except Exception:
                    affect = {}
                # Suggested tone from affect, fallback to neutral.
                tone_suggest = affect.get("suggested_tone") or "neutral"
                # Load valence and stress for heuristic fallback.
                try:
                    valence = float(affect.get("valence", 0.0))
                except Exception:
                    valence = 0.0
                try:
                    stress = float(affect.get("stress", 0.0))
                except Exception:
                    stress = 0.0
                # Personalise the greeting using the user profile if available.
                user_name = _resolve_user_name(ctx)
                # Title-case the name if present
                if user_name and isinstance(user_name, str):
                    user_name = user_name[:1].upper() + user_name[1:] if user_name else user_name
                # Decide which tone category to use from the greeting profile.
                # If Stage 5 provided a suggested tone that exists in the profile,
                # prefer it.  Otherwise use heuristics: positive valence →
                # excited, high stress or negative valence → calm, recognised user →
                # friendly, else formal.
                tone_choice = None
                if GREETING_PROFILE:
                    ts_lower = str(tone_suggest).lower()
                    if ts_lower in GREETING_PROFILE:
                        tone_choice = ts_lower
                if not tone_choice:
                    # Heuristics based on affect values
                    if valence > 0.5:
                        tone_choice = "excited"
                    elif stress > 0.7 or valence < -0.3:
                        tone_choice = "calm"
                    elif user_name:
                        tone_choice = "friendly"
                    else:
                        tone_choice = "formal"
                    # Fall back to friendly if chosen tone missing
                    if tone_choice not in GREETING_PROFILE:
                        tone_choice = "friendly" if "friendly" in GREETING_PROFILE else None
                # Select a template from the chosen tone category.
                if tone_choice and GREETING_PROFILE.get(tone_choice):
                    try:
                        template = random.choice(GREETING_PROFILE.get(tone_choice))
                    except Exception:
                        template = None
                else:
                    template = None
                # If no template could be chosen, default to a generic greeting.
                if not template:
                    template = "Hello! How can I help you today?"
                # Insert the user's name if available.  If the template already
                # contains a placeholder for a name (e.g. "Hi there"), simply
                # insert the name after the first word.  Otherwise, append the
                # name to the greeting before any punctuation.
                greet_text = template
                try:
                    if user_name:
                        parts = template.split()
                        if parts:
                            first = parts[0]
                            punct = ""
                            m = re.match(r"^(\w+)([!?.]*)$", first)
                            if m:
                                root_word, punct = m.group(1), m.group(2)
                                rest = " ".join(parts[1:]) if len(parts) > 1 else ""
                                greet_text = f"{root_word} {user_name}{punct} {rest}".strip()
                            else:
                                greet_text = f"{first} {user_name} { ' '.join(parts[1:])}".strip()
                except Exception:
                    pass
                # Personalise with the agent's name via BrainMemory, if available.
                agent_name: str | None = None
                try:
                    # Access self-model data through BrainMemory tier API
                    _self_mem = BrainMemory("self_model")
                    _self_results = _self_mem.retrieve(limit=1)
                    if _self_results:
                        sm_data = _self_results[0].get("content", {})
                        if isinstance(sm_data, dict):
                            agent_name = str(sm_data.get("name") or "").strip()
                except Exception:
                    agent_name = None  # type: ignore
                if agent_name:
                    # Append the agent name at the end of the greeting in a simple form.
                    _gt = greet_text.rstrip()
                    # Ensure the greeting ends with punctuation
                    if not _gt.endswith(('!', '.', '?')):
                        _gt += '.'
                    greet_text = f"{_gt} I'm {agent_name}."
                # Simplify the greeting to target basic readability.
                greet_text = _simplify_greeting(greet_text)
                # Build and return candidate with confidence and tone metadata.
                greets = [{
                    "type": "greeting",
                    "text": greet_text,
                    "confidence": 0.9,
                    "tone": tone_choice or (tone_suggest if tone_suggest else "neutral")
                }]
                out_g = {"candidates": greets, "weights_used": {"gen_rule": "greeting_responder_v2"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_g}

            # Detect emotional statements early.  When Stage 3 categorises an
            # input as an emotional statement (``storable_type == 'EMOTION'``), craft an
            # empathetic reply based on the sentiment valence and bypass
            # normal candidate generation.  Negative valence yields a
            # supportive message; positive valence yields a celebratory one.
            try:
                stype3 = str(stage3.get("storable_type", "")).upper()
            except Exception:
                stype3 = ""
            if stype3 == "EMOTION":
                # Determine sentiment valence directly from the original text.  Recompute
                # valence here rather than relying on Stage 3 metadata because
                # ``parse_intent`` does not propagate custom keys.  Match first‑person
                # statements like "I am happy" or "I feel sad" against keyword lists.
                valence = "neutral"
                try:
                    low_txt_local = text.lower()
                    neg_terms_local = [
                        "sad", "upset", "depressed", "angry", "frustrated",
                        "stressed", "anxious", "worried", "fearful", "lonely",
                        "miserable", "tired"
                    ]
                    pos_terms_local = [
                        "happy", "excited", "thrilled", "glad", "pleased",
                        "proud", "grateful", "thankful", "relieved", "joyful",
                        "content"
                    ]
                    # Examine negative terms
                    for term in neg_terms_local:
                        # Build pattern to match phrases like "i am <term>" or "i feel very <term>"
                        pattern = rf"\b(i(?:\s+am|'?m|\s+feel)\s+(?:[\w']+\s+){{0,2}}){re.escape(term)}\b"
                        if re.search(pattern, low_txt_local):
                            valence = "negative"
                            break
                    # If still neutral, examine positive terms
                    if valence == "neutral":
                        for term in pos_terms_local:
                            pattern = rf"\b(i(?:\s+am|'?m|\s+feel)\s+(?:[\w']+\s+){{0,2}}){re.escape(term)}\b"
                            if re.search(pattern, low_txt_local):
                                valence = "positive"
                                break
                except Exception:
                    valence = "neutral"
                # Derive a tone suggestion from stage 5 affect if present
                try:
                    tone_suggest = (ctx.get("stage_5_affect") or {}).get("suggested_tone") or "neutral"
                except Exception:
                    tone_suggest = "neutral"
                if valence == "negative":
                    resp_text = ("I'm sorry to hear that. It sounds like you're going through "
                                 "something difficult. I'm here to help if you have any questions.")
                    tone_resp = "empathetic"
                elif valence == "positive":
                    resp_text = ("That's wonderful to hear! I'm glad you're feeling that way. "
                                 "How can I assist you further?")
                    tone_resp = "cheerful"
                else:
                    resp_text = ("Thank you for sharing how you're feeling. "
                                 "Let me know if there's anything I can do.")
                    tone_resp = tone_suggest or "neutral"
                cand_emotion = {
                    "type": "emotion",
                    "text": resp_text,
                    "confidence": 0.85,
                    "tone": tone_resp
                }
                out_em = {"candidates": [cand_emotion], "weights_used": {"gen_rule": "emotion_responder_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_em}
            # Non‑greeting: compute question intent flag
            is_q_intent = bool(stage3.get("is_question"))
            # Detect peer delegation commands early.  Commands of the form
            # "delegate <task> to peer <id>" should be handled even when
            # Stage 3 classified the input as a statement/fact.  Recognise
            # variations such as "delegate X peer Y".  When matched, call
            # the peer connection brain’s DELEGATE op, return a single
            # candidate and bypass the normal generation logic.
            try:
                import re as _re
                low_txt = text.strip().lower()
                mdel = _re.search(r"^\s*delegate\s+(.*?)\s+(?:to\s+)?peer\s+(\w+)", low_txt)
            except Exception:
                mdel = None
            if mdel:
                task = mdel.group(1).strip()
                peer_id = mdel.group(2)
                # Invoke the peer connection brain to register the delegated task
                try:
                    from brains.cognitive.peer_connection.service.peer_connection_brain import service_api as peer_api  # type: ignore
                    resp = peer_api({"op": "DELEGATE", "payload": {"peer_id": peer_id, "task": task}})
                    msg_text = (resp.get("payload") or {}).get("message") or f"Delegated task '{task}' to peer {peer_id}."
                except Exception:
                    msg_text = f"Delegated task '{task}' to peer {peer_id}."
                cand = {
                    "type": "peer_delegation",
                    "text": msg_text,
                    "confidence": 0.8,
                    "tone": "neutral"
                }
                out_pd = {"candidates": [cand], "weights_used": {"gen_rule": "peer_delegation_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_pd}

            # Detect peer query commands early.  Recognize forms like
            # "ask peer <id> <question>" or "ask <question> to peer <id>".  This
            # feature allows Maven to route a question to a peer agent.  When
            # matched, call the peer connection brain’s ASK op, return a single
            # candidate and bypass the normal generation logic.
            try:
                m_ask1 = _re.match(r"^\s*ask\s+peer\s+(\w+)\s+(.+)", low_txt)
            except Exception:
                m_ask1 = None
            try:
                m_ask2 = _re.match(r"^\s*ask\s+(.+)\s+(?:to\s+)?peer\s+(\w+)$", low_txt)
            except Exception:
                m_ask2 = None
            if m_ask1 or m_ask2:
                if m_ask1:
                    peer_id = m_ask1.group(1)
                    q = m_ask1.group(2).strip()
                else:
                    # m_ask2 captures question first then peer id at the end
                    q = m_ask2.group(1).strip()
                    peer_id = m_ask2.group(2)
                # Invoke the peer connection brain to handle the query
                try:
                    from brains.cognitive.peer_connection.service.peer_connection_brain import service_api as peer_api  # type: ignore
                    resp = peer_api({"op": "ASK", "payload": {"peer_id": peer_id, "question": q}})
                    msg_text = (resp.get("payload") or {}).get("message") or f"Peer {peer_id} cannot answer your question right now."
                except Exception:
                    msg_text = f"Peer {peer_id} cannot answer your question right now."
                cand = {
                    "type": "peer_query",
                    "text": msg_text,
                    "confidence": 0.8,
                    "tone": "neutral"
                }
                out_pq = {"candidates": [cand], "weights_used": {"gen_rule": "peer_query_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_pq}
        except Exception:
            is_q_intent = False
        is_q = is_q_intent or _is_question(text)
        # Pull validation results from Stage 8
        validation = ctx.get("stage_8_validation") or {}
        verdict = str(validation.get("verdict", "")).upper()
        conf_val = float(validation.get("confidence", 0.0) or 0.0)
        # Answer content may reside under 'answer' or 'answer_content'
        answer = validation.get("answer") or validation.get("answer_content")
        # Best evidence fallback
        ev = _best_evidence(ctx)
        ev_text = (ev or {}).get("content")
        # Attempt to parse JSON-wrapped evidence
        if ev_text and isinstance(ev_text, str) and ev_text.startswith("{") and ev_text.endswith("}"):
            try:
                parsed_ev = json.loads(ev_text)
                if isinstance(parsed_ev, dict):
                    ev_text = parsed_ev.get("text") or ev_text
            except Exception:
                pass
        candidates: List[Dict[str, Any]] = []
        # If a slow_path was signalled by the dual router in the reasoning stage and
        # the verdict indicates insufficient evidence (e.g. UNKNOWN, THEORY, UNANSWERED),
        # proactively consult the imaginer for speculative hypotheses.  This allows
        # the system to offer creative possibilities when the fast path yields no
        # clear answer.  Hypotheses are added to the candidate list with a
        # moderate confidence and neutral tone.  The imaginer is only invoked
        # once per request and respects governance permits for the IMAGINE action.
        try:
            slow_path = bool(validation.get("slow_path", False))
        except Exception:
            slow_path = False
        try:
            verdict_upper = str(validation.get("verdict", "")).upper()
        except Exception:
            verdict_upper = ""
        # Only consult the imaginer when there is no direct answer and the slow path is suggested
        imaginer_used = False
        # For unanswered questions the verdict may be SKIP_STORAGE.  Include this case
        # so that the imaginer can propose speculative answers when the fast path
        # cannot provide a factual response.  We also include UNANSWERED, NO_EVIDENCE,
        # UNKNOWN and THEORY verdicts.  If any of these verdicts occur and the dual router
        # signals a slow path, consult the imaginer for speculative hypotheses.
        allowed_verdicts = {"UNKNOWN", "THEORY", "UNANSWERED", "NO_EVIDENCE", "SKIP_STORAGE"}
        if slow_path and verdict_upper in allowed_verdicts:
            try:
                # Defer import until needed to avoid circular dependencies
                from brains.cognitive.imaginer.service.imaginer_brain import service_api as imaginer_api  # type: ignore
                # Use the sanitized query for speculation when available, otherwise raw text
                prompt = text
                # Request up to three speculative hypotheses
                resp = imaginer_api({"op": "HYPOTHESIZE", "payload": {"prompt": prompt, "n": 3}})
                hyps = (resp.get("payload") or {}).get("hypotheses") or []
                for h in hyps:
                    try:
                        cand_text = str(h.get("content", "")).strip()
                    except Exception:
                        cand_text = ""
                    if cand_text:
                        # Filter out hypotheses that just echo the question
                        try:
                            base_prompt = prompt.lower().strip()
                            base_question = text.rstrip("?").strip().lower()
                            norm_h = cand_text.lower().strip().rstrip(".?!")
                            # Also strip common imaginer prefixes to check if what remains is just the question
                            for prefix in ["it might be that ", "perhaps ", "imagine that ", "one possibility is that ", "conceivably, "]:
                                if norm_h.startswith(prefix):
                                    norm_h = norm_h[len(prefix):].strip()
                                    break
                        except Exception:
                            base_prompt = ""
                            base_question = ""
                            norm_h = cand_text.lower().strip().rstrip(".?!")
                        # Skip if the hypothesis is essentially the same as the question
                        if norm_h == base_prompt or norm_h == base_question:
                            continue
                        candidates.append({
                            "type": "imagined_hypothesis",
                            "text": cand_text,
                            "confidence": 0.4,
                            "tone": "neutral"
                        })
                if hyps:
                    imaginer_used = True
            except Exception:
                # On any error, do not fail the pipeline; proceed without adding hypotheses
                pass
        if is_q:
            # -----------------------------------------------------------------
            # Personal preference recall
            #
            # Before attempting to determine whether we have an answer from memory,
            # check if the user is asking about their own likes (e.g., "What do I like?",
            # "What foods do I like?").  If so, bypass the normal memory search and
            # reasoning pipeline to query the personal brain for stored preferences.
            try:
                low_pref_q = text.lower()
            except Exception:
                low_pref_q = str(text).lower()
            try:
                import re as _pref_re2
                if _pref_re2.search(r"\bwhat\b.*\bdo\s+i\s+like\b", low_pref_q):
                    # Query the personal brain for top likes
                    try:
                        from brains.personal.service.personal_brain import service_api as _personal_api2  # type: ignore
                        resp = _personal_api2({"op": "TOP_LIKES", "payload": {"limit": 5}})
                        payload_local = resp.get("payload") or {}
                        items = payload_local.get("items") or payload_local.get("top_likes") or []
                    except Exception:
                        items = []
                    # Extract human‑readable names
                    names2: List[str] = []
                    try:
                        for it in items:
                            if isinstance(it, dict):
                                n2 = it.get("subject") or it.get("name") or it.get("value")
                                if not n2:
                                    n2 = str(it)
                            else:
                                n2 = str(it)
                            names2.append(str(n2))
                    except Exception:
                        names2 = []
                    # Formulate reply
                    if names2:
                        reply2 = "You mentioned you like " + ", ".join(names2) + "."
                    else:
                        reply2 = "I don't know yet what you like."
                    cand2 = {
                        "type": "personal_recall",
                        "text": reply2,
                        "confidence": 0.7,
                        "tone": "neutral"
                    }
                    return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [cand2], "weights_used": {"gen_rule": "personal_recall_v2"}}}
            except Exception:
                pass
            # Question path
            has_answer = False
            ans_text = None
            if verdict in ("TRUE", "ANSWERED_FROM_MEMORY"):
                has_answer = True
                ans_text = answer or ev_text
            elif verdict != "SKIP_STORAGE" and conf_val > 0.5:
                # treat high-confidence unknown as answer
                has_answer = True
                ans_text = answer or ev_text
            if has_answer and ans_text:
                direct = str(ans_text).strip()
                candidates.append({
                    "type": "direct_factual",
                    "text": direct,
                    "confidence": conf_val if conf_val else 0.8,
                    "tone": "neutral"
                })
                candidates.append({
                    "type": "conversational_factual",
                    "text": ("Yes. " + direct) if not direct.lower().startswith(("yes","no")) else direct,
                    "confidence": conf_val * 0.9 if conf_val else 0.7,
                    "tone": "friendly"
                })
            else:
                # No answer available.  First check if the user is asking about their own
                # preferences (e.g., "What foods do I like?").  If so, consult the
                # personal brain to retrieve previously recorded likes and return a
                # personalised response.  This logic only runs when we lack a
                # direct answer from the knowledge base.
                try:
                    low_q = text.lower()
                except Exception:
                    low_q = str(text).lower()
                handled_personal = False
                try:
                    import re as _pref_re
                    # Match patterns like "what do i like", "what food do i like" or "what foods do i like"
                    # Allow any words between 'what' and 'do i like' to cover variations.
                    if _pref_re.search(r"\bwhat\b.*\bdo\s+i\s+like\b", low_q):
                        from brains.personal.service.personal_brain import service_api as personal_api  # type: ignore
                        res = personal_api({"op": "TOP_LIKES", "payload": {"limit": 5}})
                        # Personal brain returns top likes under 'items'.  Fallback to other keys if missing.
                        payload_res = res.get("payload") or {}
                        likes = payload_res.get("items") or payload_res.get("top_likes") or []
                        if likes:
                            try:
                                names: List[str] = []
                                for item in likes:
                                    # personal brain records are dicts with a 'subject' key; fallback to
                                    # 'name' or 'value' if provided.  For non‑dict values, convert to string.
                                    if isinstance(item, dict):
                                        n = item.get("subject") or item.get("name") or item.get("value")
                                        if not n:
                                            n = str(item)
                                    else:
                                        n = str(item)
                                    names.append(str(n))
                                # If no names extracted, fallback to generic message
                                if names:
                                    reply = "You mentioned you like " + ", ".join(names) + "."
                                else:
                                    reply = "I don't know yet what you like."
                            except Exception:
                                reply = "You mentioned some things you like."
                        else:
                            reply = "I don't know yet what you like."
                        cand = {
                            "type": "personal_recall",
                            "text": reply,
                            "confidence": 0.7,
                            "tone": "neutral"
                        }
                        handled_personal = True
                        return {"ok": True, "op": op, "mid": mid, "payload": {"candidates": [cand], "weights_used": {"gen_rule": "personal_recall_v1"}}}
                except Exception:
                    handled_personal = False
                # If not handled by personal recall, provide helpful acknowledgements
                # When the system lacks an answer, offer a gentle follow‑up rather than a flat statement.
                # Adjust the wording to invite continued dialogue and set a higher confidence to sound certain
                # enough without being overconfident.  For example, ask the user about their favorite subject.
                candidates.append({
                    "type": "uncertain_helpful",
                    "text": "I’m not sure yet—what’s your favorite subject?",
                    "confidence": 0.5,
                    "tone": "neutral"
                })
                candidates.append({
                    "type": "uncertain_friendly",
                    "text": "I don’t have information about that in my memory yet. Would you like to tell me?",
                    "confidence": 0.3,
                    "tone": "friendly"
                })
                # When the system lacks an answer and personal recall was not triggered,
                # consult the imaginer for speculative possibilities.  This engages
                # System‑2 style imagination to offer a creative guess.  We respect
                # governance permits via the imaginer API.
                if not handled_personal:
                    try:
                        from brains.cognitive.imaginer.service.imaginer_brain import service_api as imaginer_api  # type: ignore
                        try:
                            prompt = text.rstrip("?").strip()
                        except Exception:
                            prompt = text
                        try:
                            import re as _qa_re  # local import to avoid top‑level dep
                            m_qa = _qa_re.match(r"(?i)\bwhat\s+do\s+(.*)", prompt)
                            if m_qa and m_qa.group(1):
                                prompt = m_qa.group(1).strip()
                        except Exception:
                            pass
                        resp = imaginer_api({"op": "HYPOTHESIZE", "payload": {"prompt": prompt, "n": 1}})
                        hyps = (resp.get("payload") or {}).get("hypotheses") or []
                        for h in hyps:
                            try:
                                htext = str(h.get("content", "")).strip()
                            except Exception:
                                htext = ""
                            if htext:
                                try:
                                    base_prompt = prompt.lower().strip()
                                    base_question = text.rstrip("?").strip().lower()
                                    norm_h = htext.lower().strip().rstrip(".?!")
                                except Exception:
                                    base_prompt = ""
                                    base_question = ""
                                    norm_h = htext.lower().strip().rstrip(".?!")
                                if norm_h == base_prompt or norm_h == base_question:
                                    continue
                                candidates.append({
                                    "type": "imagined_answer",
                                    "text": htext,
                                    "confidence": 0.25,
                                    "tone": "neutral"
                                })
                    except Exception:
                        pass

            # Check for an explicit random guess request.  If the original query
            # contains the phrase "random guess", append a whimsical low‑confidence
            # answer.  Use a time-based index to vary the guess across runs.
            try:
                low = text.lower()
            except Exception:
                low = ""
            if "random guess" in low:
                try:
                    seeds = [
                        "They probably taste like rainbows and cupcakes.",
                        "Perhaps like cotton candy and marshmallows.",
                        "Maybe like vanilla clouds and sugar sprinkles."
                    ]
                    guess = random.choice(seeds)
                except Exception:
                    guess = "They probably taste like rainbows and cupcakes."
                candidates.append({
                    "type": "random_guess",
                    "text": guess,
                    "confidence": 0.05,
                    "tone": "neutral"
                })
        else:
            # Statement path: handle subjective content (emotion/opinion), speculation, facts and commands
            storable_type = str((ctx.get("stage_3_language") or {}).get("storable_type", "")).upper()
            # Handle clarification requests before processing explicit requests or commands.
            if storable_type == "CLARIFICATION":
                # Generate a clarifying question based on common ambiguous request patterns.
                # Analyse the original text to determine the missing information and craft
                # a follow‑up question.
                try:
                    low_text = text.lower()
                except Exception:
                    low_text = str(text).lower()
                clar_q = None
                try:
                    if re.search(r"plan\s+(?:a\s+)?(?:trip|vacation|holiday|journey)", low_text):
                        clar_q = "Where would you like to go?"
                    elif re.search(r"schedule\s+(?:a\s+)?(?:meeting|appointment)", low_text):
                        clar_q = "When should this meeting be?"
                    elif re.search(r"book\s+(?:a\s+)?(?:restaurant|table|reservation)", low_text):
                        clar_q = "Where and when would you like to make the reservation?"
                    elif re.search(r"(?:organize|organise|plan)\s+(?:a\s+)?(?:party|event)", low_text):
                        clar_q = "When and where do you want to host the event?"
                except Exception:
                    clar_q = None
                if not clar_q:
                    clar_q = "Could you please provide more details?"
                candidates = [
                    {
                        "type": "clarification_request",
                        "text": clar_q,
                        "confidence": 0.8,
                        "tone": "curious"
                    }
                ]
                out_p = {"candidates": candidates, "weights_used": {"gen_rule": "clarification_responder_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_p}
            # Handle explicit request commands such as connecting to a peer.  When the
            # storable type is REQUEST and the text contains "connect to peer", invoke
            # the peer connection brain to simulate establishing a peer link.  The
            # resulting confirmation message is returned as the sole candidate.
            if storable_type == "REQUEST":
                try:
                    low_req = text.lower()
                except Exception:
                    low_req = str(text).lower()
                if "connect to peer" in low_req:
                    peer_id = None
                    try:
                        import re as _peer_re
                        m_peer = _peer_re.search(r"peer\s+(\w+)", low_req)
                        if m_peer:
                            peer_id = m_peer.group(1)
                    except Exception:
                        peer_id = None
                    peer_id = peer_id or "default"
                    try:
                        from brains.cognitive.peer_connection.service.peer_connection_brain import service_api as peer_api  # type: ignore
                        peer_resp = peer_api({"op": "CONNECT", "payload": {"peer_id": peer_id}})
                        msg = (peer_resp.get("payload") or {}).get("message", f"Connected to peer {peer_id}.")
                    except Exception:
                        msg = f"Connected to peer {peer_id}."
                    candidates = [
                        {
                            "type": "peer_connection",
                            "text": msg,
                            "confidence": 0.8,
                            "tone": "neutral"
                        }
                    ]
                    out_p = {"candidates": candidates, "weights_used": {"gen_rule": "peer_connection_v1"}}
                    return {"ok": True, "op": op, "mid": mid, "payload": out_p}

            # Conversational acknowledgement and thanks
            # If the storable type is ACKNOWLEDGMENT or THANKS, produce a
            # contextual response based on recent queries rather than storing
            # the utterance.  This branch bypasses the generic statement
            # handlers below to avoid filler responses.
            if storable_type == "THANKS":
                # Personalised gratitude response.  If we know the user's
                # identity from this session or the durable store, include
                # their name in the reply.  Otherwise fall back to a generic
                # response.  Do not fabricate names when none are known.
                user_name = None
                # Try session context first
                try:
                    user_name = ctx.get("session_identity")  # type: ignore[attr-defined]
                except Exception:
                    user_name = None
                # Fallback to durable store
                if not user_name:
                    try:
                        from brains.personal.service import identity_user_store as _ius  # type: ignore
                        ident = _ius.GET()
                        if isinstance(ident, dict):
                            nm = ident.get("name")
                            if nm:
                                user_name = str(nm).strip() or None
                    except Exception:
                        user_name = None
                if user_name:
                    reply_text = f"You're welcome, {user_name}!"
                else:
                    reply_text = "You're welcome!"
                thanks_cand = {
                    "type": "thanks_reply",
                    "text": reply_text,
                    "confidence": 0.8,
                    "tone": "friendly"
                }
                out_t = {"candidates": [thanks_cand], "weights_used": {"gen_rule": "acknowledgment_responder_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_t}
            if storable_type == "ACKNOWLEDGMENT":
                # Use recent session context to customise the acknowledgment
                ack_cands = generate_for_acknowledgment(ctx)
                out_a = {"candidates": ack_cands, "weights_used": {"gen_rule": "acknowledgment_responder_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_a}

            # Continuation or fragmentary utterances: ask the user to clarify and
            # reference the previous query when available.  These do not
            # contribute to the knowledge base and are handled immediately.
            if storable_type == "CONTINUATION":
                cont_cands = generate_for_continuation(ctx)
                out_c = {"candidates": cont_cands, "weights_used": {"gen_rule": "continuation_responder_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_c}

            # Preference handling: detect positive and negative likes/dislikes and
            # persist them in the personal brain.  Provide a concise
            # acknowledgment and avoid generic fillers.  A preference is
            # indicated when the storable type is PREFERENCE (see
            # classify_storable_type).  We extract the subject of the
            # preference, infer valence (positive vs negative) and intensity
            # based on keywords, record it via the personal brain, and
            # acknowledge to the user.
            if storable_type == "PREFERENCE":
                try:
                    low_text = text.lower()
                except Exception:
                    low_text = str(text).lower()
                import re as _pref_re
                negative = False
                # Identify negative sentiments
                neg_patterns = [r"don\s*'\s*t\s+like\s+(.*)", r"do\s+not\s+like\s+(.*)", r"dislike\s+(.*)", r"hate\s+(.*)"]
                pos_patterns = [r"i\s+(?:really\s+)?(?:kind\s+of\s+)?like\s+(.*)", r"i\s+love\s+(.*)", r"i\s+prefer\s+(.*)", r"my\s+favorite\s+(?:food|thing)?\s*(?:is\s*)?(.*)"]
                subject: str | None = None
                for pat in neg_patterns:
                    m = _pref_re.search(pat, low_text)
                    if m:
                        subject = m.group(1).strip().rstrip(".!?").strip()
                        negative = True
                        break
                if subject is None:
                    for pat in pos_patterns:
                        m = _pref_re.search(pat, low_text)
                        if m:
                            subject = m.group(1).strip().rstrip(".!?").strip()
                            break
                # Fallback: extract last word or phrase after like/love/hate
                if not subject:
                    try:
                        tokens = low_text.split()
                        # find index of like/love/hate
                        idx = -1
                        for i, tok in enumerate(tokens):
                            if tok in {"like", "love", "hate", "dislike", "prefer"}:
                                idx = i
                                break
                        if idx >= 0 and idx + 1 < len(tokens):
                            subject = " ".join(tokens[idx+1:]).rstrip(".!?").strip()
                    except Exception:
                        subject = None
                if not subject:
                    # BUGFIX: Cannot determine subject - ask what they like instead of echoing
                    # Old response "Got it — noted." (0.6 confidence) was too generic
                    cand = {
                        "type": "clarification_request",
                        "text": "What do you like? I'd like to remember your preferences.",
                        "confidence": 0.7,
                        "tone": "friendly"
                    }
                    out_p = {"candidates": [cand], "weights_used": {"gen_rule": "preference_recorder_v1"}}
                    return {"ok": True, "op": op, "mid": mid, "payload": out_p}
                # Determine intensity based on adverbs
                intensity = 0.6
                # Intensifiers: love → very strong positive; kind of / sort of → mild
                if "love" in low_text:
                    intensity = 0.9
                elif "kind of" in low_text or "sort of" in low_text or "kindof" in low_text:
                    intensity = 0.3
                # Negative intensifiers: hate/dislike considered strong negative
                if negative:
                    if "hate" in low_text:
                        intensity = 0.8
                    elif "dislike" in low_text:
                        intensity = 0.6
                # Persist in personal brain
                try:
                    from brains.personal.service.personal_brain import service_api as personal_api  # type: ignore
                    op_name = "RECORD_DISLIKE" if negative else "RECORD_LIKE"
                    personal_api({"op": op_name, "payload": {"subject": subject, "intensity": intensity, "source": "self_report"}})
                except Exception:
                    pass
                # Build acknowledgment response
                if negative:
                    msg = f"I'll remember that you don't like {subject}."
                else:
                    msg = f"I'll remember that you like {subject}."
                cand = {
                    "type": "record_preference",
                    "text": msg,
                    "confidence": 0.7,
                    "tone": "friendly"
                }
                out_p = {"candidates": [cand], "weights_used": {"gen_rule": "preference_recorder_v1"}}
                return {"ok": True, "op": op, "mid": mid, "payload": out_p}
            # Emotional or opinionated statements → empathetic or acknowledging replies
            if storable_type in ("EMOTION", "OPINION"):
                aff = ctx.get("stage_5_affect") or {}
                em_cands = _generate_candidates_for_emotion(text, aff)
                candidates.extend(em_cands)
            # Speculation or theory → thoughtful acknowledgements
            elif storable_type == "SPECULATION" or verdict == "THEORY":
                subj = _extract_subject(text)
                if not subj:
                    subj = text.strip()
                candidates.append({
                    "type": "acknowledging_speculation",
                    "text": f"That’s an interesting theory about {subj}.",
                    "confidence": 0.7,
                    "tone": "thoughtful"
                })
                candidates.append({
                    "type": "storing_theory",
                    "text": f"I’ve noted your speculation about {subj}.",
                    "confidence": 0.6,
                    "tone": "neutral"
                })
                candidates.append({
                    "type": "conversational_speculation",
                    "text": f"{subj.title()} is a fascinating possibility to consider.",
                    "confidence": 0.5,
                    "tone": "friendly"
                })
            elif verdict in ("TRUE", "VERIFIED"):
                # Validated fact acknowledgement
                candidates.append({
                    "type": "direct_factual",
                    "text": "Noted. I’ve stored this.",
                    "confidence": 0.8,
                    "tone": "neutral"
                })
                candidates.append({
                    "type": "conversational_factual",
                    "text": "Got it — added to memory.",
                    "confidence": 0.7,
                    "tone": "friendly"
                })
                candidates.append({
                    "type": "detailed",
                    "text": f"Understood. I’ve stored this with {int(conf_val * 100)}% confidence.",
                    "confidence": 0.6,
                    "tone": "informative"
                })
            elif verdict == "SKIP_STORAGE" or storable_type in ("COMMAND", "REQUEST", "UNKNOWN"):
                # Non‑storable input or unknown.  Certain commands require
                # additional handling, such as connecting to a peer.
                low_txt = (text or "").strip().lower()
                import re as _re  # use local import to avoid top-level dep
                peer_matched = False
                # First handle delegation commands.  Recognise phrases like
                # "delegate <task> to peer <id>" or "delegate <task> peer <id>".
                try:
                    m_del = _re.search(r"delegate\s+(.*?)\s+(?:to\s+)?peer\s+(\w+)", low_txt)
                except Exception:
                    m_del = None
                if m_del:
                    peer_matched = True
                    task = m_del.group(1).strip()
                    peer_id = m_del.group(2)
                    try:
                        from brains.cognitive.peer_connection.service.peer_connection_brain import service_api as peer_api  # type: ignore
                        res = peer_api({"op": "DELEGATE", "payload": {"peer_id": peer_id, "task": task}})
                        msg_text = (res.get("payload") or {}).get("message") or f"Delegated task '{task}' to peer {peer_id}."
                    except Exception:
                        msg_text = f"Delegated task '{task}' to peer {peer_id}."
                    candidates.append({
                        "type": "peer_delegation",
                        "text": msg_text,
                        "confidence": 0.8,
                        "tone": "neutral"
                    })
                else:
                    # Handle peer connection requests like "connect to peer 123" or "connect peer 123"
                    try:
                        m = _re.search(r"connect\s+(?:to\s+)?peer\s+(\d+)", low_txt)
                    except Exception:
                        m = None
                    if m:
                        peer_matched = True
                        peer_id = m.group(1)
                        try:
                            # Attempt to invoke the peer connection brain.  If it
                            # does not exist or raises, fall back to a static message.
                            from brains.cognitive.peer_connection.service.peer_connection_brain import service_api as peer_api  # type: ignore
                            res = peer_api({"op": "CONNECT", "payload": {"peer_id": peer_id}})
                            msg_text = (res.get("payload") or {}).get("message") or f"Connected to peer {peer_id}."
                        except Exception:
                            msg_text = f"Connected to peer {peer_id}."
                        candidates.append({
                            "type": "peer_connection",
                            "text": msg_text,
                            "confidence": 0.8,
                            "tone": "neutral"
                        })
                # Generic acknowledgement for commands/requests or unknown inputs
                if not peer_matched:
                    # BUGFIX: Ask for clarification instead of echoing
                    # These echo responses were causing Maven to say "Got it — noted."
                    # instead of taking action. Now we ask the user for more details.
                    candidates.append({
                        "type": "clarification_request",
                        "text": "I'm not sure what you'd like me to do. Could you provide more details?",
                        "confidence": 0.5,
                        "tone": "helpful"
                    })
                    # Keep echo as absolute last resort with very low confidence
                    candidates.append({
                        "type": "acknowledgement",
                        "text": "Got it — noted.",
                        "confidence": 0.01,
                        "tone": "neutral"
                    })
            else:
                # BUGFIX: Ultimate fallback - ask for clarification instead of echoing
                # The old "Got it — noted." response (0.25 confidence) was winning when
                # Maven should have been doing research, calling tools, or asking questions.
                # Now we explicitly ask the user what they want instead of pretending to understand.
                candidates.append({
                    "type": "clarification_request",
                    "text": "I'm not sure how to help with that. What would you like me to do?",
                    "confidence": 0.4,
                    "tone": "helpful"
                })
                # Keep echo as absolute last resort with extremely low confidence
                # This should ONLY win if there are literally no other candidates
                candidates.append({
                    "type": "acknowledgement",
                    "text": "Got it — noted.",
                    "confidence": 0.01,
                    "tone": "neutral"
                })
        # Append a random guess candidate if explicitly requested and not already present.
        try:
            low_all = text.lower()
        except Exception:
            low_all = ""
        if "random guess" in low_all:
            # Only add if not already present from the question path
            has_random = any((c.get("type") == "random_guess") for c in candidates)
            if not has_random:
                try:
                    seeds2 = [
                        "They probably taste like rainbows and cupcakes.",
                        "Perhaps like cotton candy and marshmallows.",
                        "Maybe like vanilla clouds and sugar sprinkles."
                    ]
                    guess2 = random.choice(seeds2)
                except Exception:
                    guess2 = "They probably taste like rainbows and cupcakes."
                candidates.append({
                    "type": "random_guess",
                    "text": guess2,
                    "confidence": 0.05,
                    "tone": "neutral"
                })
        # Avoid echoing back the exact user input
        for c in candidates:
            if c["text"].strip().lower() == text.strip().lower():
                c["text"] = "Answer: " + c["text"]
        # Creative divergence: generate a few variant phrasings from the
        # strongest candidate.  These variants enable exploration of
        # alternative wording before the system converges on a final answer.
        try:
            base = max(candidates, key=lambda c: c.get("confidence", 0.0))
        except Exception:
            base = None
        new_cands = []
        if base:
            base_text = str(base.get("text", ""))
            variants = [
                f"In other words, {base_text}",
                f"To put it differently, {base_text}",
                f"Another way to say it is: {base_text}",
            ]
            for var in variants:
                try:
                    new_cands.append({
                        "type": "creative_variant",
                        "text": var,
                        "confidence": max(0.05, float(base.get("confidence", 0.0)) * 0.9),
                        "tone": base.get("tone", "neutral")
                    })
                except Exception:
                    continue
        # Append up to two variants to the candidate list to limit expansion
        candidates.extend(new_cands[:2])
        # Sort candidates by confidence in descending order to emulate a
        # converge step where higher‑scoring responses are preferred
        try:
            candidates.sort(key=lambda c: c.get("confidence", 0.0), reverse=True)
        except Exception:
            pass
        # ------------------------------------------------------------------
        # Attention constraint: when the language brain has strong focus
        # (focus_strength ≥ 0.8) and the reasoning engine has supplied a
        # validated answer (stage_8 verdict TRUE), discard any candidate
        # whose text does not contain the validated answer.  This prevents
        # filler responses from supplanting correct factual answers when
        # knowledge graph evidence exists.  If all candidates are
        # filtered out, fall back to a single direct factual candidate
        # carrying the answer and its confidence.
        try:
            ans8 = None
            ver8 = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()
            if ver8 == "TRUE":
                ans8 = (ctx.get("stage_8_validation") or {}).get("answer")
            attn = ctx.get("stage_5b_attention") or {}
            fs = 0.0
            try:
                fs = float((attn.get("state") or {}).get("focus_strength", 0.0) or 0.0)
            except Exception:
                fs = 0.0
            if ans8 and attn.get("focus") == "language" and fs >= 0.8:
                # Keep only candidates whose text contains the answer
                filtered = []
                for c in candidates:
                    try:
                        if ans8 in str(c.get("text", "")):
                            filtered.append(c)
                    except Exception:
                        continue
                if filtered:
                    candidates = filtered
                else:
                    # Fallback: create a minimal direct factual candidate
                    conf_val = 0.8
                    try:
                        conf_val = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.8) or 0.8)
                    except Exception:
                        conf_val = 0.8
                    candidates = [
                        {
                            "type": "direct_factual",
                            "text": ans8,
                            "confidence": conf_val,
                            "tone": "neutral",
                            "explanation": "Filtered to match KG answer"
                        }
                    ]
        except Exception:
            pass
        # Introduce a low‑confidence exploratory candidate to encourage
        # alternative reasoning paths.  This candidate acknowledges
        # uncertainty and invites the user to clarify or consider other
        # possibilities.  It appears at the end of the list due to its
        # low confidence.
        try:
            alt_cand = {
                "type": "alternative",
                "text": "I'm also considering other possibilities, though my confidence is low.",
                "confidence": 0.1,
                "tone": "tentative"
            }
            candidates.append(alt_cand)
        except Exception:
            pass
        out = {"candidates": candidates, "weights_used": {"gen_rule":"s6_context_aware_v1"}}
        return {"ok": True, "op": op, "mid": mid, "payload": out}

    if op == "FINALIZE":
        ctx = payload if isinstance(payload, dict) else {}
        text = _clean(str(ctx.get("original_query","")))
        verdict = str((ctx.get("stage_8_validation") or {}).get("verdict","" )).upper()
        confidence = float((ctx.get("stage_8_validation") or {}).get("confidence", 0.0) or 0.0)
        cands = (ctx.get("stage_6_candidates") or {}).get("candidates") or []

        # ======================================================================
        # FINAL ANSWER CONTAMINATION FIX (Task 1.3)
        # ======================================================================
        # Filter candidates to ONLY those matching the current question.
        # This prevents stale answers from previous turns/concepts from
        # contaminating the final answer selection.
        # ======================================================================
        try:
            from brains.learning.lesson_utils import canonical_concept_key
            current_concept_key = canonical_concept_key(text)
        except Exception:
            current_concept_key = None

        # Compute question signature for matching
        try:
            question_signature = text.lower().strip().rstrip("?").strip()
        except Exception:
            question_signature = ""

        # Get current run_id from context
        current_run_id = ctx.get("run_id") or ctx.get("pipeline_run_id")

        # Filter candidates to only those matching current question
        filtered_cands = []
        self_model_cands = []  # Priority candidates from self_model

        for c in cands:
            # Get candidate metadata
            cand_concept_key = c.get("concept_key")
            cand_question_sig = c.get("question_signature", "")
            cand_run_id = c.get("run_id")
            cand_source = c.get("source_brain", "")

            # Check if candidate matches current question
            sig_ok = (not cand_question_sig) or (cand_question_sig == question_signature)
            ck_ok = (cand_concept_key is None) or (current_concept_key is None) or (cand_concept_key == current_concept_key)
            run_ok = (cand_run_id is None) or (current_run_id is None) or (cand_run_id == current_run_id)

            # Self-model candidates get priority for identity questions (Task 1.5)
            if cand_source == "self_model":
                self_model_cands.append(c)
            elif sig_ok and ck_ok and run_ok:
                filtered_cands.append(c)

        # For identity questions, self_model candidates are sticky (Task 1.5)
        identity_keywords = ["who are you", "what are you", "who created you", "who made you",
                             "your name", "your purpose", "what is maven", "are you an llm"]
        is_identity_question = any(kw in text.lower() for kw in identity_keywords)

        if is_identity_question and self_model_cands:
            # Use self_model answer exclusively for identity questions
            print(f"[FINAL_ANSWER_FIX] Identity question detected, using self_model candidate exclusively")
            cands = self_model_cands
        elif filtered_cands:
            # Use filtered candidates
            cands = filtered_cands
        elif self_model_cands:
            # Fall back to self_model candidates
            cands = self_model_cands
        # else: keep original cands as last resort

        # Guard against old-concept pollution (Task 1.4)
        # If selected candidate clearly mentions wrong concept, drop it
        if cands and current_concept_key:
            wrong_concept_cands = []
            valid_cands = []
            for c in cands:
                cand_ck = c.get("concept_key")
                if cand_ck and cand_ck != current_concept_key:
                    # Check if candidate text contains concepts obviously wrong
                    wrong_concept_cands.append(c)
                    print(f"[FINAL_ANSWER_FIX] Dropped stale candidate with concept_key={cand_ck} (current={current_concept_key})")
                else:
                    valid_cands.append(c)
            if valid_cands:
                cands = valid_cands

        # ======================================================================
        # PERSONALITY-FIRST PATTERN: Personality brain is the authority for
        # tone, verbosity, and style. User profile can SHARPEN but not
        # CONTRADICT personality. Inferred tone is only used as fallback.
        # ======================================================================
        prefs = (ctx.get("personality_snapshot") or {})

        # Check if personality explicitly defers to other sources
        personality_defers = prefs.get("defer_to_context", False)

        # Personality is PRIMARY for tone
        personality_tone = prefs.get("tone", "neutral")
        tone = personality_tone

        # Default verbosity target comes from the personality snapshot; 1.0 means neutral.
        try:
            verbosity = float(prefs.get("verbosity_target", 1.0) or 1.0)
        except Exception:
            verbosity = 1.0

        # Personality style (e.g., "concise", "elaborate", "technical")
        personality_style = prefs.get("style", None)

        # Track whether personality has a strong opinion (not just default)
        personality_has_tone = personality_tone and personality_tone != "neutral"
        personality_has_verbosity = verbosity != 1.0
        personality_has_style = personality_style is not None

        # ======================================================================
        # USER PROFILE: Can SHARPEN but not CONTRADICT personality
        # - If Personality says "formal", user can sharpen to "very formal"
        # - User cannot change "formal" to "casual" (contradiction)
        # - If Personality is neutral/defer, user profile can set preferences
        # ======================================================================
        try:
            from brains.personal.memory import user_profile  # type: ignore
        except Exception:
            user_profile = None  # type: ignore
        if user_profile:
            try:
                prof = user_profile.get_profile() or {}
            except Exception:
                prof = {}

            # Process user profile tone
            prof_tone = None
            for key in ("tone", "formality", "style"):
                if key in prof:
                    prof_tone = str(prof.get(key, "")).strip().lower()
                    break

            if prof_tone:
                # Only allow user profile to SET tone if personality is neutral/defers
                # OR if user profile SHARPENS (not contradicts) personality
                if personality_defers or not personality_has_tone:
                    # Personality defers or is neutral - user profile can set
                    tone = prof_tone
                else:
                    # Check if user profile sharpens rather than contradicts
                    # Sharpening: "formal" -> "very formal", "casual" -> "very casual"
                    # Contradiction: "formal" -> "casual", "friendly" -> "curt"
                    sharpening_allowed = _is_tone_sharpening(personality_tone, prof_tone)
                    if sharpening_allowed:
                        tone = prof_tone
                    # else: keep personality tone (don't allow contradiction)

            # Process user profile verbosity
            prof_verb = None
            for key in ("verbosity", "verbosity_preference", "detail", "level"):
                if key in prof:
                    prof_verb = str(prof.get(key, "")).strip().lower()
                    break

            if prof_verb:
                # Map named levels to numeric verbosity: higher values -> more verbose
                mapping = {
                    "low": 0.8,
                    "terse": 0.8,
                    "short": 0.9,
                    "brief": 0.9,
                    "normal": 1.0,
                    "default": 1.0,
                    "high": 1.3,
                    "verbose": 1.5,
                    "detailed": 1.4,
                    "very detailed": 1.6,
                }

                # Convert to numeric
                if prof_verb in mapping:
                    prof_verbosity = mapping[prof_verb]
                else:
                    try:
                        prof_verbosity = float(prof_verb)
                    except Exception:
                        prof_verbosity = None

                if prof_verbosity is not None:
                    # Only allow user profile to SET verbosity if personality is neutral/defers
                    # OR if user profile sharpens (same direction, more extreme)
                    if personality_defers or not personality_has_verbosity:
                        verbosity = prof_verbosity
                    else:
                        # Sharpening: if personality says "brief", user can say "very brief"
                        # but cannot say "verbose"
                        if _is_verbosity_sharpening(verbosity, prof_verbosity):
                            verbosity = prof_verbosity
                        # else: keep personality verbosity
        # Adjust verbosity based on the user’s familiarity with the domain.  If the
        # user frequently asks about a domain, Maven will reduce verbosity for
        # brevity; if the user rarely visits the domain, Maven will increase
        # verbosity to provide more context.  This update also tracks the
        # domain usage for future sessions.
        try:
            from brains.personal.memory import user_knowledge  # type: ignore
        except Exception:
            user_knowledge = None  # type: ignore
        if user_knowledge is not None:
            try:
                import re
                # Compute domain key from the original query (stage 10 uses
                # original question rather than candidate text).
                q = text.strip().lower()
                q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
                q_norm = re.sub(r"\s+", " ", q_norm).strip()
                parts = q_norm.split()
                domain_key = " ".join(parts[:2]) if parts else ""
                if domain_key:
                    # Update the user's familiarity count for this domain
                    user_knowledge.update(domain_key)
                    # Retrieve familiarity level and adjust verbosity
                    level = user_knowledge.get_level(domain_key)
                    # Apply a modest multiplier: experts get shorter answers,
                    # novices get longer answers, familiar users are in between.
                    if level == "expert":
                        verbosity *= 0.75
                    elif level == "familiar":
                        verbosity *= 0.9
                    else:  # novice
                        verbosity *= 1.1
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Update the persistent user mood based on the valence provided by
        # the affect stage and adjust the tone accordingly.  The affect
        # stage (stage 5) outputs a valence in the range [‑1, 1].  Positive
        # values indicate positive emotions and negative values represent
        # negative emotions.  We update the mood memory with this value,
        # apply decay as defined by the user_mood module, and retrieve
        # the current mood to modulate the tone.  A strongly positive mood
        # (> 0.3) encourages a friendly tone; a strongly negative mood
        # (< ‑0.3) prompts a caring tone.  Otherwise, the previously
        # selected tone remains unchanged.  Errors in mood updates are
        # ignored so that mood tracking never blocks response generation.
        try:
            from brains.personal.memory import user_mood as _user_mood  # type: ignore[attr-defined]
        except Exception:
            _user_mood = None  # type: ignore
        if _user_mood is not None:
            try:
                # Retrieve affect stage results (valence from stage 5)
                aff_local = ctx.get("stage_5_affect") or {}
                try:
                    valence_val = float(aff_local.get("valence", 0.0) or 0.0)
                except Exception:
                    valence_val = 0.0
                try:
                    # Update mood state using the valence
                    _user_mood.update(valence_val)  # type: ignore[attr-defined]
                    # Retrieve the current mood after update
                    cur_mood_val = float(_user_mood.get_mood() or 0.0)  # type: ignore[attr-defined]
                    # Choose a tone based on the current mood
                    # PERSONALITY-FIRST: Only adjust tone if personality defers
                    # or doesn't have a strong tone preference
                    if personality_defers or not personality_has_tone:
                        if cur_mood_val >= 0.3:
                            tone = "friendly"
                        elif cur_mood_val <= -0.3:
                            tone = "caring"
                        # else keep tone unchanged
                    # If personality has a strong tone, mood can only sharpen it
                    elif personality_has_tone:
                        mood_tone = None
                        if cur_mood_val >= 0.3:
                            mood_tone = "friendly"
                        elif cur_mood_val <= -0.3:
                            mood_tone = "caring"
                        if mood_tone and _is_tone_sharpening(personality_tone, mood_tone):
                            tone = mood_tone
                except Exception:
                    pass
            except Exception:
                pass
        # ------------------------------------------------------------------
        # Mirror the user's tone when detectable.  After mood updates and
        # personality adjustments, consult the original query for informal
        # cues (slang, emoticons, exclamation marks).  If a tone is
        # inferred, override the current tone.  Formal cues similarly
        # override to a formal style.  This simple mirroring helps
        # Maven sound more natural and aligned with the user's style.
        #
        # PERSONALITY-FIRST: Only infer/mirror user tone if personality
        # defers or doesn't have a strong tone preference. Personality
        # is the authority for Maven's base identity.
        try:
            if personality_defers or not personality_has_tone:
                inferred_tone = _infer_user_tone(text)
                if inferred_tone:
                    tone = inferred_tone
            # else: Personality has a tone - keep it as the authority
        except Exception:
            pass
        has_ev = _best_evidence(ctx) is not None

        pick = None
        # Prefer random guess candidates when explicitly requested
        try:
            low_orig = text.lower()
        except Exception:
            low_orig = ""
        if "random guess" in low_orig:
            pick = next((c for c in cands if c.get("type") == "random_guess"), None)
        if not pick:
            if verdict == "TRUE":
                pick = next((c for c in cands if c.get("type")=="direct_factual"), None) or                    next((c for c in cands if c.get("type")=="conversational_factual"), None)
                # Fallback: if no candidates but stage_8 has an answer (e.g., from SELF_MODEL),
                # use that answer directly
                if not pick:
                    stage8_answer = (ctx.get("stage_8_validation") or {}).get("answer")
                    if stage8_answer:
                        pick = {"type": "direct_factual", "text": stage8_answer, "confidence": confidence}
            elif verdict in ("THEORY", "UNKNOWN"):
                # Prefer speculation acknowledgements if available
                pick = next((c for c in cands if c.get("type") in ("acknowledging_speculation", "storing_theory", "conversational_speculation")), None)
                if not pick:
                    pick = next((c for c in cands if c.get("type") == "uncertain_helpful"), None)
                if not pick and cands:
                    pick = cands[0]
            elif verdict in ("FALSE","REJECT","CONTRADICTION"):
                pick = {"type":"correction","text":"That doesn’t match what I know yet. Can you correct or add a reference?","confidence":0.9,"tone":"neutral"}
            else:
                # If no candidates are available, default to a gentle uncertain response that invites the user
                # to share their perspective rather than ending the conversation abruptly.
                pick = (cands[0] if cands else {"type":"uncertain_helpful","text":"I’m not sure yet—what’s your favorite subject?","confidence":0.5})

        # For factual answers validated by stage 8, pass the text through unchanged.
        if verdict == "TRUE" and (pick or {}).get("type") == "direct_factual":
            final_text = str(pick.get("text", ""))
        else:
            final_text = _tone_wrap(str(pick.get("text","")), tone)
            if final_text.strip().lower() == text.strip().lower():
                final_text = "Answer: " + final_text
            final_text = _apply_verbosity(final_text, verbosity)
        # Override transparency tag for random guesses
        if "random guess" in low_orig:
            transparency = "random_guess"
        else:
            transparency = _transparency_tag(verdict, confidence, has_ev)
        # Persist Q/A pair to the cross‑episode memory when this is a question
        # and we have a definite answer.  If an answer already exists for
        # this question and differs from the new answer (case insensitive),
        # log the discrepancy to a self‑repair report.  This helps future
        # maintenance by surfacing contradictory answers across sessions.
        try:
            stage3_local = ctx.get("stage_3_language") or {}
            st_type_local = str(stage3_local.get("storable_type", "")).upper()
            if st_type_local == "QUESTION":
                ans_lower = final_text.strip().lower()
                # Skip indefinite responses or generic uncertainty phrases
                skip_phrases = ["i don't know", "i don't know", "dont know", "i don't have information", "i don't have information"]
                if not any(p in ans_lower for p in skip_phrases):
                    # Check for existing answer to log contradictions
                    try:
                        # Search for existing Q&A pairs with the same question
                        existing_qa = _LANGUAGE_MEMORY.retrieve(query=text.strip().lower(), limit=100)
                        for rec in existing_qa:
                            rec_content = rec.get("content", {})
                            if not isinstance(rec_content, dict):
                                continue
                            if rec_content.get("kind") != "qa_pair":
                                continue
                            if str(rec_content.get("question", "")).strip().lower() == text.strip().lower():
                                old_ans = str(rec_content.get("answer", "")).strip()
                                if old_ans and old_ans.lower() != ans_lower:
                                    # Found contradictory answer; log to self_repair
                                    try:
                                        repair_content = {
                                            "kind": "self_repair",
                                            "question": text.strip(),
                                            "old_answer": old_ans,
                                            "new_answer": final_text.strip(),
                                        }
                                        _LANGUAGE_MEMORY.store(
                                            content=repair_content,
                                            metadata={"kind": "self_repair", "source": "language", "confidence": 1.0}
                                        )
                                    except Exception:
                                        pass
                                break
                    except Exception:
                        # Continue even if we can't scan existing answers
                        pass
                    # Store new Q&A pair in BrainMemory
                    try:
                        qa_content = {
                            "kind": "qa_pair",
                            "question": text.strip(),
                            "answer": final_text.strip(),
                        }
                        _LANGUAGE_MEMORY.store(
                            content=qa_content,
                            metadata={"kind": "qa_pair", "source": "language", "confidence": confidence}
                        )
                        # Update global topic statistics for cross‑episode learning.  If
                        # the topic_stats module is unavailable, silently ignore.
                        try:
                            from brains.personal.memory import topic_stats  # type: ignore
                            topic_stats.update_topic(text.strip())
                        except Exception:
                            pass
                        # Update meta confidence statistics for the domain.  Compute
                        # a domain key from the question and record a success or
                        # failure based on the verdict.  Successes are defined
                        # by definitive answers (TRUE or KNOWN_ANSWER modes);
                        # failures correspond to UNKNOWN, UNANSWERED or THEORY.
                        try:
                            from brains.personal.memory import meta_confidence  # type: ignore
                        except Exception:
                            meta_confidence = None  # type: ignore
                        if meta_confidence is not None:
                            try:
                                import re
                                # Compute domain key from the original query
                                q = text.strip().lower()
                                q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
                                q_norm = re.sub(r"\s+", " ", q_norm).strip()
                                parts = q_norm.split()
                                domain_key = " ".join(parts[:2]) if parts else ""
                                if domain_key:
                                    # Determine if this answer counts as a success
                                    verdict_lower = verdict.lower()
                                    success = verdict_lower in ("true", "verified", "known_answer", "answered")
                                    # Compute a weight based on the complexity of the question.
                                    # Longer questions are considered more complex and therefore
                                    # carry more weight in meta-confidence adjustments.  Use the
                                    # number of words divided by 5 as a starting point and cap
                                    # between 1.0 and 3.0.
                                    num_words = max(1, len(parts))
                                    weight = num_words / 5.0
                                    if weight < 1.0:
                                        weight = 1.0
                                    elif weight > 3.0:
                                        weight = 3.0
                                    meta_confidence.update(domain_key, success, weight)
                            except Exception:
                                pass

                        # --- Semantic memory assimilation -------------------------
                        # When recording a new Q/A pair, attempt to extract
                        # simple definition facts and store them in the
                        # knowledge graph.  For questions of the form
                        # "What is X?" or "Who is X?" with a short,
                        # definitive answer, persist the triple (X, is, answer).
                        # Only commit facts when the verdict indicates a
                        # definitive answer to avoid recording speculation or
                        # uncertainty.  The subject is normalised to lower
                        # case and leading articles are removed to maximise
                        # matches.  Long answers (>80 chars) or answers
                        # containing question marks or uncertainty phrases
                        # ("don't know") are skipped.
                        try:
                            # Only run assimilation when we have a definitive verdict
                            verdict_lower = verdict.lower()
                            is_definitive = verdict_lower in (
                                "true",
                                "verified",
                                "known_answer",
                                "answered",
                                "kg_answer"
                            )
                            if is_definitive:
                                from brains.personal.memory import knowledge_graph  # type: ignore
                            else:
                                knowledge_graph = None  # type: ignore
                        except Exception:
                            knowledge_graph = None  # type: ignore
                        if knowledge_graph is not None:
                            try:
                                import re
                                q_raw = text.strip()
                                # Match patterns like "what is X" or "who is X"
                                m = re.match(r"\s*(?:what|who)\s+is\s+(.+?)\s*\??\s*$", q_raw, re.IGNORECASE)
                                if m:
                                    subj = m.group(1).strip()
                                    ans = final_text.strip()
                                    # Reject answers that are too long or contain uncertainty
                                    if ans and len(ans) <= 80 and '?' not in ans and "don't know" not in ans.lower():
                                        subj_norm = subj.lower().strip()
                                        # Build a list of candidate subjects: raw and without leading articles
                                        candidates = [subj_norm]
                                        cand_strip = re.sub(r"^(?:the|a|an)\s+", "", subj_norm).strip()
                                        if cand_strip and cand_strip != subj_norm:
                                            candidates.append(cand_strip)
                                        # Insert the first candidate that yields a new fact
                                        for candidate in candidates:
                                            try:
                                                if candidate:
                                                    knowledge_graph.add_fact(candidate, "is", ans)
                                                    # Also update synonym mapping: map the answer phrase to
                                                    # its canonical subject.  We normalise the answer by
                                                    # lowercasing and stripping leading articles.  Both
                                                    # forms (with and without articles) are mapped to
                                                    # the candidate subject.  Synonym updates are
                                                    # best‑effort and ignored on failure.
                                                    try:
                                                        from brains.personal.memory import synonyms as _syn_mod  # type: ignore
                                                    except Exception:
                                                        _syn_mod = None  # type: ignore
                                                    if _syn_mod is not None:
                                                        try:
                                                            ans_norm = ans.lower().strip()
                                                            if ans_norm:
                                                                _syn_mod.update_synonym(ans_norm, candidate)
                                                                import re
                                                                ans_strip = re.sub(r"^(?:the|a|an)\s+", "", ans_norm).strip()
                                                                if ans_strip and ans_strip != ans_norm:
                                                                    _syn_mod.update_synonym(ans_strip, candidate)
                                                        except Exception:
                                                            pass
                                                    break
                                            except Exception:
                                                continue
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass
        # ------------------------------------------------------------------
        # Construct additional meta‑information to accompany the final
        # response.  These fields expose the inner workings of the pipeline
        # such as confidence explanations, reasoning traces and quality
        # metrics.  They are optional and provided primarily for
        # transparency and diagnostics.
        try:
            # Confidence explanation
            conf_expl = _confidence_explanation(ctx)
        except Exception:
            conf_expl = ""
        try:
            # Reasoning trace: use explicit trace from stage 8 or fall back to chain
            rt = (ctx.get("stage_8_validation") or {}).get("reasoning_trace")
            if not rt:
                rt = (ctx.get("stage_8_validation") or {}).get("reasoning_chain") or []
        except Exception:
            rt = []
        # Stress detection based on repeated queries
        try:
            recent = (ctx.get("session_context") or {}).get("recent_queries", [])
            # Count how many times the current query has occurred recently (case‑insensitive)
            repeat_count = 0
            for rq in recent:
                try:
                    if str(rq.get("query", "")).strip().lower() == text.strip().lower():
                        repeat_count += 1
                except Exception:
                    continue
            stress_val = float(repeat_count) / float(len(recent) or 1)
        except Exception:
            repeat_count = 0
            stress_val = 0.0
        # Proactive clarification when the user repeats a question
        if repeat_count >= 2:
            proactive_message = "It seems you’ve asked this before; let me know if you need more detail."
        else:
            proactive_message = None
        # Confidence interval estimation
        try:
            base_conf = float(pick.get("confidence", 0.0) or 0.0)
        except Exception:
            base_conf = 0.0
        lo = max(0.0, base_conf - 0.1)
        hi = min(1.0, base_conf + 0.1)
        confidence_interval = [round(lo, 3), round(hi, 3)]
        # Determine if the answer is verified (only true when verdict is TRUE or KNOWN_ANSWER)
        try:
            verdict_lower = verdict.lower()
            answer_verified = verdict_lower in ("true", "verified", "known_answer", "answered")
        except Exception:
            answer_verified = False
        # Response quality metrics
        try:
            # Simple relevance: fraction of query words appearing in the answer
            q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", text.lower())]
            ans_tokens = [w.lower() for w in re.findall(r"\b\w+\b", final_text.lower())]
            overlap = sum(1 for w in q_tokens if w in ans_tokens)
            relevance = float(overlap) / float(len(q_tokens) or 1)
        except Exception:
            relevance = 0.0
        try:
            # Completeness: whether we had any memory results
            mem_res = (ctx.get("stage_2R_memory") or {}).get("results", [])
            completeness = 1.0 if mem_res else 0.5
        except Exception:
            completeness = 0.5
        clarity = 1.0  # placeholder: assume language always clear
        helpfulness = 1.0 if pick.get("type") not in {"uncertain_helpful"} else 0.7
        response_quality = {
            "relevance_score": round(relevance, 3),
            "completeness_score": round(completeness, 3),
            "clarity_score": round(clarity, 3),
            "helpfulness_score": round(helpfulness, 3),
        }
        # Learned bias explanation (placeholder)
        learned_bias_expl = "Bias adjustments applied based on historical success and dynamic confidence."
        # Personal influence explanation
        try:
            pers_inf = ctx.get("stage_11_personal_influence") or {}
            pboost = pers_inf.get("boost", 0.0)
            why = pers_inf.get("why") or ""
            if pboost:
                personal_inf_expl = f"Personal preference boosted the answer by {pboost:.2f}: {why}."
            else:
                personal_inf_expl = "No personal preference influence detected."
        except Exception:
            personal_inf_expl = ""
        # Identity influence explanation
        try:
            ident_inf = ctx.get("stage_12b_identity_influence") or {}
            iboost = ident_inf.get("boost", 0.0)
            permit_id = ident_inf.get("permit_id") or ""
            if iboost:
                identity_inf_expl = f"Identity bias boosted the answer by {iboost:.2f} (permit {permit_id})."
            else:
                identity_inf_expl = "No identity bias applied."
        except Exception:
            identity_inf_expl = ""
        # Affect learning explanation
        try:
            affect_learn = ctx.get("stage_14_affect_learn") or {}
            if affect_learn:
                affect_expl = f"Affect learning logged with valence {affect_learn.get('valence')} and arousal {affect_learn.get('arousal')}."
            else:
                affect_expl = ""
        except Exception:
            affect_expl = ""
        # Tool usage reasoning explanation
        try:
            weights_used = ctx.get("stage_0_weights_used") or {}
            tool_expl_parts = []
            for brain, info in (weights_used or {}).items():
                if info:
                    tool_expl_parts.append(f"{brain} used tool {info}")
            tool_usage_expl = "; ".join(tool_expl_parts) if tool_expl_parts else ""
        except Exception:
            tool_usage_expl = ""
        return {"ok": True, "op": op, "mid": mid, "payload": {
            "text": final_text,
            "confidence": pick.get("confidence", 0.0),
            "tone": pick.get("tone"),
            "verbosity": verbosity,
            "transparency": transparency,
            "confidence_explanation": conf_expl,
            "reasoning_trace": rt,
            "stress": round(stress_val, 3),
            "proactive_message": proactive_message,
            "confidence_interval": confidence_interval,
            "answer_verified": answer_verified,
            "response_quality": response_quality,
            "learned_bias_explanation": learned_bias_expl,
            "personal_influence_explanation": personal_inf_expl,
            "identity_influence_explanation": identity_inf_expl,
            "affect_learning_explanation": affect_expl,
            "tool_usage_explanation": tool_usage_expl,
        }}

    # =========================================================================
    # EXPLAIN_LAST: Traceability feature for understanding how Maven answered
    # =========================================================================
    # Allows users to ask "why?" or "explain that" to understand:
    # - Which brains contributed to the answer
    # - Whether Teacher was used (and for what concept_key)
    # - Which lessons/facts/patterns were involved
    # - The reasoning trace from the pipeline
    #
    # Usage: User types "why?" or "explain that" -> front-end wraps as EXPLAIN_LAST
    # =========================================================================
    if op == "EXPLAIN_LAST":
        ctx = payload if isinstance(payload, dict) else {}

        # Retrieve information from the last pipeline execution
        explanation_parts = []

        # 1. Get the original question
        original_query = str(ctx.get("original_query", ""))
        if original_query:
            explanation_parts.append(f"**Question:** {original_query}")

        # 2. Identify which brains contributed
        contributing_brains = []
        brain_stages = [
            ("stage_1_sensorium", "Sensorium", "normalized input"),
            ("stage_2R_memory", "Memory", "retrieved relevant memories"),
            ("stage_3_language", "Language", "parsed and classified query"),
            ("stage_4_routing", "Routing/Integrator", "chose brain routing"),
            ("stage_5_affect", "Affect", "assessed emotional context"),
            ("stage_6_candidates", "Language", "generated response candidates"),
            ("stage_8_validation", "Reasoning/Validation", "verified answer"),
        ]

        for stage_key, brain_name, action in brain_stages:
            stage_data = ctx.get(stage_key)
            if stage_data and isinstance(stage_data, dict):
                contributing_brains.append(f"- **{brain_name}**: {action}")

        if contributing_brains:
            explanation_parts.append("\n**Brains that contributed:**")
            explanation_parts.extend(contributing_brains)

        # 3. Check if Teacher was used
        teacher_used = False
        teacher_info = []

        # Check for Teacher events in the pipeline context
        stage8 = ctx.get("stage_8_validation") or {}
        if stage8.get("source") == "teacher" or stage8.get("verdict") == "LEARNED":
            teacher_used = True
            concept_key = stage8.get("concept_key", "")
            teacher_info.append(f"- Teacher called for learning (concept_key: '{concept_key}')")

        # Check memory stage for Teacher evidence
        stage2 = ctx.get("stage_2R_memory") or {}
        results = stage2.get("results", [])
        for res in results[:3]:  # Check first few results
            if isinstance(res, dict):
                source = res.get("source", "")
                if "teacher" in source.lower():
                    teacher_used = True
                    teacher_info.append(f"- Used learned pattern from Teacher")
                    break

        if teacher_used:
            explanation_parts.append("\n**Teacher involvement:**")
            explanation_parts.extend(teacher_info)
        else:
            explanation_parts.append("\n**Teacher involvement:** None (answered from memory)")

        # 4. List concept_key/lessons/facts involved
        lessons_facts = []

        # From stage 8 validation
        if stage8.get("concept_key"):
            lessons_facts.append(f"- Concept key: '{stage8.get('concept_key')}'")
        if stage8.get("lesson_used"):
            lessons_facts.append(f"- Lesson: {stage8.get('lesson_used')[:100]}...")
        if stage8.get("facts_used"):
            facts = stage8.get("facts_used", [])
            for fact in facts[:3]:
                lessons_facts.append(f"- Fact: {str(fact)[:80]}...")

        # From memory retrieval
        for res in results[:3]:
            if isinstance(res, dict):
                content = res.get("content", "")
                kind = res.get("kind", "") or res.get("metadata", {}).get("kind", "")
                if kind in ("lesson", "learned_fact", "learned_pattern", "qa_pair"):
                    lessons_facts.append(f"- Memory ({kind}): {str(content)[:80]}...")

        if lessons_facts:
            explanation_parts.append("\n**Knowledge sources used:**")
            explanation_parts.extend(lessons_facts)

        # 5. Include reasoning trace
        reasoning_trace = stage8.get("reasoning_trace") or stage8.get("reasoning_chain", [])
        if reasoning_trace:
            explanation_parts.append("\n**Reasoning trace:**")
            if isinstance(reasoning_trace, list):
                for step in reasoning_trace[:5]:
                    explanation_parts.append(f"- {str(step)[:100]}")
            else:
                explanation_parts.append(f"- {str(reasoning_trace)[:200]}")

        # 6. Include confidence and verdict
        verdict = stage8.get("verdict", "UNKNOWN")
        confidence = stage8.get("confidence", 0.0)
        explanation_parts.append(f"\n**Answer verification:** {verdict} (confidence: {confidence:.2f})")

        # 7. Generate summary
        explanation_text = "\n".join(explanation_parts) if explanation_parts else "No pipeline context available for explanation."

        # Add a human-readable summary
        summary = []
        if not teacher_used:
            summary.append("I answered this using learned knowledge from memory (no LLM Teacher call).")
        else:
            summary.append("I learned something new from the Teacher to answer this.")

        if contributing_brains:
            brain_names = [b.split("**")[1].split("**")[0] for b in contributing_brains if "**" in b]
            if brain_names:
                summary.append(f"Brains involved: {', '.join(set(brain_names))}.")

        summary_text = " ".join(summary)

        return {"ok": True, "op": op, "mid": mid, "payload": {
            "summary": summary_text,
            "detailed_explanation": explanation_text,
            "teacher_used": teacher_used,
            "contributing_brains": [b for b in contributing_brains],
            "knowledge_sources": lessons_facts,
            "verdict": verdict,
            "confidence": confidence
        }}

    if op == "HEALTH":
        return {"ok": True, "op": op, "mid": mid, "payload": {"status":"ok"}}

    # EXECUTE_STEP: Phase 8 - Execute a language generation step
    if op == "EXECUTE_STEP":
        step = payload.get("step") or {}
        step_id = payload.get("step_id", 0)
        context = payload.get("context") or {}

        # Extract step details
        description = step.get("description", "")
        step_input = step.get("input") or {}
        task = step_input.get("task", description)

        # Use GENERATE_CANDIDATES for language generation
        gen_result = service_api({
            "op": "GENERATE_CANDIDATES",
            "mid": mid,
            "payload": {
                "text": task,
                "context": context,
                "n": 1
            }
        })

        if gen_result.get("ok"):
            gen_payload = gen_result.get("payload") or {}
            candidates = gen_payload.get("candidates", [])

            output = candidates[0] if candidates else task

            return {"ok": True, "op": op, "mid": mid, "payload": {
                "output": output,
                "patterns_used": ["language:generation"]
            }}

        return {"ok": False, "op": op, "mid": mid, "error": {"code": "GENERATION_FAILED", "message": "Failed to generate text"}}

    return {"ok": False, "op": op, "mid": mid, "error": {"code": "UNSUPPORTED_OP", "message": op}}

# -----------------------------------------------------------------------------
# Attention bid interface
#
# To participate in the integrator brain's attention resolution process,
# each cognitive brain may expose a ``bid_for_attention`` function.  This
# helper inspects the current pipeline context and returns a bid
# dictionary containing a ``brain_name``, a ``priority`` between 0.0 and
# 1.0, a ``reason`` string and optional ``evidence``.  Higher priority
# indicates stronger need for attention.  The language brain bids
# aggressively for unanswered questions and social interactions, and
# otherwise submits a low default bid.  This function is safe to call
# even if parts of the context are missing; it will return a sensible
# default in error cases.
def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Determine the input type/intention from the language parse stage
        lang_info = ctx.get("stage_3_language", {}) or {}
        st_type = str(
            lang_info.get("type")
            or lang_info.get("storable_type")
            or lang_info.get("intent")
            or ""
        ).upper()
        # Fetch the current verdict from the reasoning stage if available
        verdict8 = str((ctx.get("stage_8_validation") or {}).get("verdict", "")).upper()

        # -----------------------------------------------------------------
        # Continuation handling with routing hints
        #
        # When the input is a continuation (follow-up), create routing hints
        # to help the integrator direct the query appropriately.  Continuations
        # should reference previous context rather than starting fresh.
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")

        # Get conversation context for routing decisions
        conv_context = lang_info.get("conversation_context", {})

        # High priority for unanswered questions: language should attempt to
        # generate a response when no answer is found.  Use 0.80 as the
        # default high priority.
        if st_type == "QUESTION" and (verdict8 in {"UNANSWERED", "UNKNOWN", ""}):
            # Create routing hint based on whether this is a continuation
            if is_cont:
                routing_hint = create_routing_hint(
                    brain_name="language",
                    action="expand_previous_answer",
                    confidence=0.85,
                    context_tags=["follow_up", "continuation", continuation_intent],
                    metadata={
                        "last_topic": conv_context.get("last_topic", ""),
                        "continuation_type": continuation_intent
                    }
                )
            else:
                routing_hint = create_routing_hint(
                    brain_name="language",
                    action="generate_answer",
                    confidence=0.80,
                    context_tags=["new_question", "fresh_query"],
                    metadata={}
                )

            return {
                "brain_name": "language",
                "priority": 0.80,
                "reason": "unanswered_question",
                "evidence": {
                    "query": ctx.get("original_query"),
                    "routing_hint": routing_hint,
                    "is_continuation": is_cont
                },
            }
        # Commands take precedence over requests: detect CLI flags or
        # imperative commands and adjust attention accordingly.  When a
        # command is present, the language brain should gain focus to
        # clarify or process the command.  Flags beginning with '--'
        # request clarification; other commands receive moderate
        # priority for processing.
        try:
            if lang_info.get("is_command"):
                q_raw = str(ctx.get("original_query", "")).strip()
                # Clarify unknown flags (e.g. "--query")
                if q_raw.startswith("--"):
                    routing_hint = create_routing_hint(
                        brain_name="language",
                        action="clarify_flag",
                        confidence=0.70,
                        context_tags=["command", "flag", "clarification"],
                        metadata={"is_continuation": is_cont}
                    )
                    return {
                        "brain_name": "language",
                        "priority": 0.70,
                        "reason": "needs_clarification",
                        "evidence": {
                            "query": ctx.get("original_query"),
                            "routing_hint": routing_hint
                        },
                    }
                # General commands: process with moderate priority
                routing_hint = create_routing_hint(
                    brain_name="language",
                    action="process_command",
                    confidence=0.50,
                    context_tags=["command", "imperative"],
                    metadata={"is_continuation": is_cont}
                )
                return {
                    "brain_name": "language",
                    "priority": 0.50,
                    "reason": "command_processing",
                    "evidence": {
                        "query": ctx.get("original_query"),
                        "routing_hint": routing_hint
                    },
                }
        except Exception:
            pass
        # High priority for requests: ensure helpful responses to
        # imperative requests.  Requests often begin with verbs like
        # "show", "tell", "help" etc.  When the language parse stage
        # identifies the input as a REQUEST, bid moderately high to
        # ensure the language brain handles the request.  Use 0.75
        # priority and annotate the reason accordingly.
        try:
            if lang_info.get("is_request"):
                routing_hint = create_routing_hint(
                    brain_name="language",
                    action="handle_request",
                    confidence=0.75,
                    context_tags=["request", "imperative"],
                    metadata={
                        "is_continuation": is_cont,
                        "last_topic": conv_context.get("last_topic", "")
                    }
                )
                return {
                    "brain_name": "language",
                    "priority": 0.75,
                    "reason": "request_handling",
                    "evidence": {
                        "query": ctx.get("original_query"),
                        "routing_hint": routing_hint
                    },
                }
        except Exception:
            pass
        # Medium priority for greetings or social interactions.  When the
        # language brain detects a greeting intent, bid moderately to
        # produce an appropriate social response.
        try:
            intent = str(lang_info.get("intent", "")).lower()
        except Exception:
            intent = ""
        if intent == "greeting" or st_type == "SOCIAL":
            routing_hint = create_routing_hint(
                brain_name="language",
                action="social_response",
                confidence=0.60,
                context_tags=["social", "greeting"],
                metadata={}
            )
            return {
                "brain_name": "language",
                "priority": 0.60,
                "reason": "social_interaction",
                "evidence": {"routing_hint": routing_hint},
            }
        # Otherwise submit a low default bid.  This ensures that the
        # language brain does not dominate attention for tasks better
        # handled by other brains.
        routing_hint = create_routing_hint(
            brain_name="language",
            action="default",
            confidence=0.10,
            context_tags=["default"],
            metadata={"is_continuation": is_cont}
        )
        return {
            "brain_name": "language",
            "priority": 0.10,
            "reason": "default",
            "evidence": {"routing_hint": routing_hint},
        }
    except Exception:
        # On any error return a safe default bid
        return {
            "brain_name": "language",
            "priority": 0.10,
            "reason": "default",
            "evidence": {},
        }

# ---------------------------------------------------------------------------
# Handle wrapper for language brain entry point
# ---------------------------------------------------------------------------

# Save reference to original service_api implementation
_service_api_impl = service_api

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle function that calls the language brain service implementation.

    This wrapper provides a consistent entry point name across all
    cognitive service modules and routes to the Stage-6 generator
    and other language processing capabilities.

    Args:
        msg: Request dictionary with 'op' and optional 'payload'

    Returns:
        Response dictionary from language service
    """
    return _service_api_impl(msg)

# Service API entry point
service_api = handle