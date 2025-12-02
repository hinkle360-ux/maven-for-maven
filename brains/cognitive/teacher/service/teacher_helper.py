"""
teacher_helper.py
~~~~~~~~~~~~~~~~~

Universal helper for any brain to use the Teacher + learning loop.

CORE PRINCIPLE: MEMORY-FIRST, LLM-AS-TEACHER
============================================
This module enforces the fundamental cognitive pattern:
"Brains operate from memory/context first, then LLM as teacher if needed,
never the other way around."

The LLM (Ollama locally, or remote APIs later) is always AVAILABLE but is
used as a TEACHER, not the primary source of answers.

Memory-First Order of Operations (ENFORCED):
1. Check Brain's own BrainMemory (STM/MTM/LTM)
2. Check relevant domain banks
3. Check strategy tables and Q→A cache
4. ONLY if memory fails → call LLM as teacher
5. Store lessons/facts → next time answer from memory

HARD RULE: For any brain using TeacherHelper, TeacherHelper MUST check
memory/context first, and only then use Ollama.

Learning Mode Semantics:
- TRAINING: Memory-first → if miss → call Ollama → store lesson/facts.
- OFFLINE: Use strategies and memory only. NO LLM learning. NO memory writes.
- SHADOW: LLM runs in background for comparison. NO memory writes. NO user influence.

IMPORTANT: This is the ONLY way brains should call Teacher.
Any brain bypassing TeacherHelper violates the memory-first contract.

Usage:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

    # Initialize helper for your brain
    helper = TeacherHelper("planner")

    # Call Teacher - ALWAYS memory-first
    result = helper.maybe_call_teacher(
        question="How do I plan a web scraping task?",
        context={"user": {"name": "Alice"}}
        # check_memory_first=True is ALWAYS enforced
    )

    if result:
        if result["source"] == "local_memory":
            # Answered from memory - no LLM call made
            pass
        elif result["source"] == "teacher":
            # Learned from LLM - lesson stored for next time
            pass
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import json

# Import LearningMode for controlling LLM access
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    # Fallback: create a simple enum-like class if import fails
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"

# Import core dependencies
try:
    from brains.memory.brain_memory import BrainMemory
except Exception:
    BrainMemory = None  # type: ignore

try:
    from brains.cognitive.reasoning.truth_classifier import TruthClassifier
except Exception:
    TruthClassifier = None  # type: ignore

try:
    from brains.brain_roles import is_domain_brain, get_domain_brains
except Exception:
    is_domain_brain = None  # type: ignore
    get_domain_brains = None  # type: ignore

try:
    from brains.teacher_contracts import get_contract
except Exception:
    get_contract = None  # type: ignore

try:
    from brains.cognitive.teacher.service.prompt_templates import build_prompt
except Exception:
    build_prompt = None  # type: ignore

try:
    from brains.learning.lesson_utils import canonical_concept_key
except Exception:
    # Fallback implementation if import fails
    def canonical_concept_key(question: str) -> str:  # type: ignore
        """Fallback concept key extraction."""
        if not question:
            return ""
        q = question.lower().strip()
        for char in "?!.,;:'\"()[]{}*":
            q = q.replace(char, " ")
        filler = {"what", "is", "are", "the", "a", "an", "does", "do", "can",
                  "tell", "me", "about", "explain", "describe", "please"}
        words = [w for w in q.split() if w not in filler]
        return " ".join(words[-3:] if not words else words).strip() or q.strip()


# Import proposal types for proposal mode handling
try:
    from brains.cognitive.teacher.service.teacher_proposal import (
        TeacherProposal,
        Hypothesis,
        HypothesisKind,
    )
    _proposal_types_available = True
except Exception:
    _proposal_types_available = False


def _load_feature_flags() -> Dict[str, bool]:
    """Load feature flags from config."""
    import os
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "config" / "features.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: bool(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def is_proposal_mode_enabled() -> bool:
    """Check if proposal mode is enabled (Teacher returns proposals, not direct facts)."""
    flags = _load_feature_flags()
    return flags.get("teacher_proposal_mode", True)  # Default to proposal mode


def is_direct_fact_write_enabled() -> bool:
    """Check if direct fact writing is enabled (legacy mode)."""
    flags = _load_feature_flags()
    return flags.get("teacher_direct_fact_write", False)  # Default to disabled


class TeacherHelper:
    """
    Universal helper for any brain to use Teacher + learning loop.

    ENFORCES MEMORY-FIRST PATTERN
    =============================
    This class encapsulates the entire Teacher learning pattern with
    memory-first behavior ALWAYS enforced:

    1. Memory check (MANDATORY - cannot be disabled)
       - Brain's own BrainMemory (STM/MTM/LTM)
       - Relevant domain banks
       - Q→A cache

    2. Learning mode gate (respects TRAINING/OFFLINE/SHADOW semantics)

    3. Teacher calling (only if memory miss AND mode allows)

    4. Truth classification + storage routing

    The LLM is a TEACHER, not the primary source of answers.
    Brains should learn once and answer from memory thereafter.

    Attributes:
        brain_id: Name of the calling brain
        memory: BrainMemory instance for this brain
        contract: Teacher contract for this brain
    """

    def __init__(self, brain_id: str):
        """
        Initialize helper for a specific brain.

        Args:
            brain_id: The name of the calling brain

        Raises:
            ValueError: If no contract defined or dependencies missing
        """
        if not get_contract:
            raise ValueError("Teacher contracts module not available")

        self.brain_id = brain_id
        self.contract = get_contract(brain_id)

        if not self.contract:
            raise ValueError(f"No Teacher contract defined for brain '{brain_id}'")

        if not self.contract.get("enabled", False):
            raise ValueError(f"Teacher not enabled for brain '{brain_id}'")

        # Initialize memory for this brain
        if BrainMemory:
            self.memory = BrainMemory(brain_id)
        else:
            self.memory = None

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Check that all required dependencies are available."""
        if not TruthClassifier:
            print(f"[WARNING] TruthClassifier not available for {self.brain_id}")
        if not is_domain_brain or not get_domain_brains:
            print(f"[WARNING] Brain roles module not available for {self.brain_id}")
        if not build_prompt:
            print(f"[WARNING] Prompt templates not available for {self.brain_id}")

    def _is_self_query(self, question: str) -> bool:
        """
        Detect if question is about Maven's own identity, feelings, or internal state.

        CRITICAL: Teacher MUST NOT answer identity/feelings questions.
        Identity and internal state come ONLY from self_dmn/self_model.

        TASK 1: Extended to catch feelings, emotions, preferences, opinions.
        These questions about Maven's internal state MUST NOT go to Teacher.

        Args:
            question: The question to check

        Returns:
            True if this is a self-identity or self-state question
        """
        try:
            q_lower = str(question).lower()
            # Hard-coded self-query patterns
            self_patterns = [
                # Identity patterns
                "who are you",
                "what are you",
                "who you are",
                "what you are",
                "tell me about yourself",
                "describe yourself",
                "what is your name",
                "what's your name",
                "are you maven",
                "are you an llm",
                "are you a large language model",
                "are you chatgpt",
                "are you claude",
                "are you gpt",
                "what is your code",
                "what is your system",
                "where do you run",
                "who made you",
                "who created you",
                "who built you",
                "who designed you",
                "who programmed you",
                "your creator",
                "your architect",
                "your developer",
                "how do you work",
                "what are your brains",
                "what do you know about yourself",
                "tell me about your system",
                "describe your architecture",
                # TASK 1: Feelings / emotions / internal state patterns
                # Teacher MUST NOT answer questions about Maven's feelings
                "how do you feel",
                "how are you feeling",
                "do you have feelings",
                "do you have emotions",
                "are you happy",
                "are you sad",
                "are you conscious",
                "are you sentient",
                "are you alive",
                "are you real",
                "are you a person",
                # TASK 1: Preferences / opinions patterns
                "do you like",
                "do you enjoy",
                "do you prefer",
                "do you want",
                "what do you like",
                "what do you want",
                "what do you prefer",
                "do you have preferences",
                "do you have opinions",
                "what is your opinion",
                "what are your opinions",
                "how do you think",
                "what do you think about yourself",
                # TASK 1: Other internal state patterns
                "do you dream",
                "do you sleep",
                "do you get tired",
                "do you get bored",
                "can you feel",
                "can you think",
                "your feelings",
                "your emotions",
                "your preferences",
                "your opinions",
            ]
            return any(pattern in q_lower for pattern in self_patterns)
        except Exception:
            return False

    def _is_history_query(self, question: str) -> bool:
        """
        Detect if question is about conversation history.

        CRITICAL: Teacher MUST NOT answer history questions.
        History comes ONLY from system_history and conversation logs.

        Args:
            question: The question to check

        Returns:
            True if this is a conversation history question
        """
        try:
            q_lower = str(question).lower()
            # History question patterns
            history_patterns = [
                "what did i ask you",
                "what did we talk about",
                "what have we discussed",
                "what was my first question",
                "what did i say",
                "what was our conversation",
                "yesterday we talked",
                "earlier you said",
                "before you mentioned",
                "remember when i asked",
                "did i ask you about",
                "what topics have we",
                "our previous conversation",
                "last time we spoke"
            ]
            return any(pattern in q_lower for pattern in history_patterns)
        except Exception:
            return False

    def _is_self_memory_query(self, question: str) -> bool:
        """
        Detect if question is about what Maven remembers/knows about the USER.

        CRITICAL: These MUST be answered from personal banks, NOT Teacher.

        Args:
            question: The question to check

        Returns:
            True if this is a self-memory question about the user
        """
        try:
            q_lower = str(question).lower()
            # Self-memory patterns (about the user)
            memory_patterns = [
                "what do you remember about me",
                "what do you know about me",
                "what have you learned about me",
                "what have you learned so far",
                "what is the most important thing you know about me",
                "do you remember my",
                "do you know my",
                "what are my preferences"
            ]
            return any(pattern in q_lower for pattern in memory_patterns)
        except Exception:
            return False

    def _is_capability_query(self, question: str) -> bool:
        """
        Detect if question is about Maven's capabilities ("can you X").

        CRITICAL: These MUST be answered from capability_snapshot, NOT Teacher.
        Teacher hallucinates capability answers that don't match reality.

        Args:
            question: The question to check

        Returns:
            True if this is a capability question
        """
        try:
            q_lower = str(question).lower()
            # Capability patterns
            capability_patterns = [
                # Web search
                "can you search the web",
                "can you browse the internet",
                "can you look this up online",
                "can you search online",
                "do you have internet access",
                "are you connected to the internet",
                "can you browse the web",
                # Code execution
                "can you run code",
                "can you execute code",
                "can you run python",
                "can you run scripts",
                "can you execute scripts",
                "can you run programs",
                # Control programs
                "can you control other programs",
                "can you control apps",
                "can you control other applications",
                "can you launch other apps",
                "can you control other programs on my computer",
                # File access
                "can you read files on my system",
                "can you change files on my system",
                "can you read or change files",
                "can you access files",
                "can you modify files",
                "can you write files",
                # Autonomous tools
                "can you use tools without me asking",
                "can you use the internet without",
                "do you use tools autonomously",
                "do you act on your own",
                # General capability queries (PHASE 1 additions)
                "what can you do",
                "what do you do",
                "what are your capabilities",
                "what are you capable of",
                "what tools can you use",
                "what tools do you have",
            ]
            return any(pattern in q_lower for pattern in capability_patterns)
        except Exception:
            return False

    def _is_system_capability_query(self, question: str) -> bool:
        """
        Detect if question is about Maven's system capabilities or upgrades.

        CRITICAL: These MUST be answered from system_capabilities.py, NOT Teacher.
        This prevents Teacher from answering with Apache Maven / Java 17 garbage.

        Args:
            question: The question to check

        Returns:
            True if this is a system capability/upgrade question
        """
        try:
            import re
            q_lower = str(question).lower()

            # System capability patterns - "you/your" + system words
            # These MUST route to self_model, NOT Teacher
            system_patterns = [
                # Upgrade questions
                r"\bwhat\s+upgrade\b",
                r"\bwhat\b.*\bdo\s+you\s+need\b",
                r"\bwhat\b.*\byou\s+need\b.*\bupgrade\b",
                r"\bwhat\s+upgrades?\s+(do\s+)?you\s+need\b",
                # Tools/capabilities questions
                r"\bwhat\s+tools?\b.*\b(can|do)\s+you\b",
                r"\bwhat\b.*\byour\b.*\b(tools?|capabilities?)\b",
                r"\b(you|your|yourself)\b.*\b(upgrade|capabilities?|tools?|browser|internet|web|code|files?|system|memory|brains?|pipeline|version|codebase|filesystem|programs?|applications?)\b",
                # Can you X patterns (expanded)
                r"\bcan\s+you\b.*\b(browse|search|run|execute|read|write|change|modify|access|control|do|create|update|upgrade)\b",
                # Ability questions
                r"\bare\s+you\s+able\s+to\b",
                r"\bdo\s+you\s+have\b.*\b(access|ability|capability|tool)\b",
                # Browse the web patterns - CRITICAL
                r"\b(you|your)\b.*\bbrowse\b.*\bweb\b",
                r"\bbrowse\b.*\bweb\b.*\b(you|your)\b",
                # Control programs patterns
                r"\b(you|your)\b.*\bcontrol\b.*\b(other\s+)?(programs?|apps?|applications?)\b",
                # Read/change files patterns
                r"\b(you|your)\b.*\b(read|change|modify|access)\b.*\bfiles?\b",
                # Scan code patterns
                r"\bscan\b.*\b(your|you)\b.*\b(code|codebase|system)\b",
                r"\b(your|you)\b.*\bscan\b.*\b(code|codebase|system)\b",
                # Can you write/create patterns (creative tasks as capability questions)
                r"\bcan\s+you\b.*\b(write|create|generate|compose)\b.*\b(story|poem|essay|article|code|script)\b",
            ]

            for pattern in system_patterns:
                if re.search(pattern, q_lower):
                    return True

            return False
        except Exception:
            return False

    def _is_explain_query(self, question: str) -> bool:
        """
        Detect if question is about explaining Maven's previous answer.

        CRITICAL: These MUST be answered from EXPLAIN_LAST, NOT Teacher.
        Teacher doesn't know what brains/memory sources Maven actually used.

        Args:
            question: The question to check

        Returns:
            True if this is an explain query
        """
        try:
            q_lower = str(question).lower()
            # Explain patterns
            explain_patterns = [
                "why did you answer that way",
                "why that answer",
                "how did you get that answer",
                "how did you arrive at that",
                "which parts of your system helped",
                "which brains helped",
                "which parts helped",
                "did you use the teacher to answer",
                "did you use the teacher",
                "did you call the teacher",
                "what would you do differently next time",
                "how would you improve that answer",
                "explain your reasoning",
                "explain that answer",
                "why did you say that",
            ]
            return any(pattern in q_lower for pattern in explain_patterns)
        except Exception:
            return False

    def _is_time_query(self, question: str) -> bool:
        """
        Detect if question is about current time/date.

        CRITICAL: Time questions MUST route to time_now tool, NOT Teacher.
        The LLM cannot provide accurate real-time information.

        Args:
            question: The question to check

        Returns:
            True if this is a time/date question
        """
        try:
            q_lower = str(question).lower()
            # Time/date patterns
            time_patterns = [
                "what time is it",
                "what's the time",
                "current time",
                "time now",
                "tell me the time",
                "give me the time",
                "show me the time",
                "what day is it",
                "what's the date",
                "today's date",
                "current date",
                "what day of the week",
                "what does the clock say",
                "check the time",
                "check the clock",
            ]
            return any(pattern in q_lower for pattern in time_patterns)
        except Exception:
            return False

    def _is_forbidden_teacher_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Centralized check for questions that MUST NOT go to Teacher.

        TASK 5: Strengthened block list combining identity, history, self-memory,
        capability, and explain queries.

        PHASE 1 ENHANCEMENT: Added system_capability detection to prevent
        Teacher from answering with Apache Maven / Java 17 garbage.

        Args:
            question: The question to check
            context: Optional context with intent hints

        Returns:
            Tuple of (is_forbidden, reason_category)
            reason_category values include:
            - "self_identity": Identity questions -> self_dmn
            - "conversation_history": History questions -> system_history
            - "self_memory": User memory questions -> personal banks
            - "capability": Capability questions -> capability_snapshot
            - "system_capability": System/upgrade questions -> system_capabilities
            - "explain_last": Explain questions -> EXPLAIN_LAST
        """
        # Check context hints first (including new routing hints from sensorium)
        if context:
            intent = context.get("intent") or context.get("self_intent_kind", "")
            routing_target = context.get("routing_target", "")

            # New routing hints from sensorium
            if routing_target == "self_model":
                routing_reason = context.get("routing_reason", "")
                if routing_reason in {"system_capability_query", "self_identity_query"}:
                    return (True, "system_capability")

            if intent in {"self_identity", "self_memory", "conversation_history",
                          "capability", "explain_last", "user_memory", "history",
                          "system_capability", "time_query"}:
                return (True, intent)

        # HIGHEST PRIORITY: Check for time queries FIRST
        # Time questions MUST route to time_now tool, NOT Teacher
        if self._is_time_query(question):
            return (True, "time_query")

        # Check question patterns
        if self._is_self_query(question):
            return (True, "self_identity")

        if self._is_history_query(question):
            return (True, "conversation_history")

        if self._is_self_memory_query(question):
            return (True, "self_memory")

        # PHASE 1: Check for system capability queries BEFORE generic capability
        if self._is_system_capability_query(question):
            return (True, "system_capability")

        if self._is_capability_query(question):
            return (True, "capability")

        if self._is_explain_query(question):
            return (True, "explain_last")

        return (False, None)

    def maybe_call_teacher(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        check_memory_first: bool = True,  # DEPRECATED: Always True, cannot be disabled
        retrieved_facts: Optional[list] = None,
        learning_mode: Optional[LearningMode] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Conditionally call Teacher using MEMORY-FIRST pattern.

        HARD RULE: Memory is ALWAYS checked first. The LLM is a teacher,
        not the primary source of answers.

        Order of Operations:
        1. MEMORY CHECK (MANDATORY - cannot be disabled)
           - Brain's own BrainMemory (STM/MTM/LTM)
           - Relevant domain banks
           - Q→A cache
           - If high-confidence match exists → return immediately, no LLM call

        2. SELF-QUERY BLOCKING (identity questions NEVER go to Teacher)

        3. LEARNING MODE GATE:
           - TRAINING: Memory miss → call LLM → store lesson/facts
           - OFFLINE: Memory miss → return NO_MEMORY, no LLM call, no storage
           - SHADOW: Memory miss → call LLM for comparison only, no storage

        4. TEACHER CALL (only if memory miss AND mode allows)

        5. STORAGE (only in TRAINING mode)

        Args:
            question: The question or task
            context: Optional context dict (user info, etc.)
            check_memory_first: DEPRECATED. Memory-first is ALWAYS enforced.
                               This parameter is ignored.
            retrieved_facts: Optional facts from memory to pass to Teacher
            learning_mode: Controls behavior after memory miss.
                          TRAINING: learn and store (default)
                          OFFLINE: no learning, no storage
                          SHADOW: compare only, no storage

        Returns:
            Dict with:
              - answer: str (the answer or result)
              - facts_stored: int (number of facts stored to domain brains)
              - patterns_stored: int (number of patterns stored to own memory)
              - verdict: str (LEARNED, KNOWN, NO_ANSWER, NO_MEMORY, LLM_DISABLED, etc.)
              - source: str (teacher, local_memory, memory_miss_offline, shadow_comparison, etc.)
            Or None if Teacher call not needed or failed
        """
        # Determine effective learning mode (default to TRAINING for LLM learning)
        if learning_mode is None:
            # Try to get from context if provided
            if context:
                learning_mode = context.get("learning_mode", LearningMode.TRAINING)
            else:
                learning_mode = LearningMode.TRAINING

        # ============================================================
        # STEP 1: MEMORY CHECK (can be bypassed with force_fresh_llm)
        # ============================================================
        # This is the core of memory-first architecture. We check memory
        # before considering the LLM, UNLESS force_fresh_llm is set.
        # force_fresh_llm is used for expansion requests where we need
        # a fresh, longer answer rather than a cached short one.
        # ============================================================
        force_fresh = context.get("force_fresh_llm", False) if context else False

        if force_fresh:
            print(f"[TEACHER_HELPER] Skipping memory check for {self.brain_id} (force_fresh_llm=True)")
        else:
            print(f"[TEACHER_HELPER] Memory-first check for {self.brain_id}: '{question[:50]}...'")

            # Check brain's own memory
            existing = self._check_memory(question)
            if existing:
                print(f"[TEACHER_HELPER] ✓ MEMORY HIT for {self.brain_id} (no LLM call needed)")
                return {
                    "answer": existing,
                    "verdict": "KNOWN",
                    "source": "local_memory",
                    "facts_stored": 0,
                    "patterns_stored": 0,
                    "learning_mode": str(learning_mode)
                }

            # Also check domain banks for world facts
            domain_answer = self._check_domain_banks(question)
            if domain_answer:
                print(f"[TEACHER_HELPER] ✓ DOMAIN MEMORY HIT for {self.brain_id} (no LLM call needed)")
                return {
                    "answer": domain_answer,
                    "verdict": "KNOWN",
                    "source": "domain_memory",
                    "facts_stored": 0,
                    "patterns_stored": 0,
                    "learning_mode": str(learning_mode)
                }

            print(f"[TEACHER_HELPER] Memory miss for {self.brain_id}, considering LLM teacher...")

        # ============================================================
        # STEP 2: FORBIDDEN QUESTION BLOCKING (TASK 5 ENHANCEMENT)
        # ============================================================
        # These question categories NEVER go to Teacher regardless of mode:
        # - self_identity: Identity comes from self_dmn.get_core_identity()
        # - conversation_history: History comes from system_history
        # - self_memory: User facts come from personal banks
        # ============================================================
        is_forbidden, reason_category = self._is_forbidden_teacher_question(question, context)
        if is_forbidden:
            print(f"[TEACHER_HELPER_BLOCK] Teacher disabled for {reason_category} question")
            print(f"[TEACHER_HELPER_BLOCK] Question: '{question[:50]}...'")
            if reason_category == "self_identity":
                print(f"[TEACHER_HELPER_BLOCK] Identity queries MUST route to self_model, NOT Teacher")
            elif reason_category == "conversation_history":
                print(f"[TEACHER_HELPER_BLOCK] History queries MUST route to system_history, NOT Teacher")
            elif reason_category == "self_memory":
                print(f"[TEACHER_HELPER_BLOCK] Self-memory queries MUST route to self_model, NOT Teacher")
            elif reason_category == "system_capability":
                print(f"[TEACHER_HELPER_BLOCK] System capability/upgrade queries MUST route to system_capabilities, NOT Teacher")
                print(f"[TEACHER_HELPER_BLOCK] CRITICAL: Prevents Apache Maven / Java 17 hallucinations")
            elif reason_category == "capability":
                print(f"[TEACHER_HELPER_BLOCK] Capability queries MUST route to capability_snapshot, NOT Teacher")
            elif reason_category == "time_query":
                print(f"[TEACHER_HELPER_BLOCK] Time queries MUST route to time_now tool, NOT Teacher")
                print(f"[TEACHER_HELPER_BLOCK] CRITICAL: LLM cannot provide accurate real-time information")

            return {
                "answer": None,
                "verdict": "LLM_DISABLED",
                "source": "llm_disabled",
                "facts_stored": 0,
                "patterns_stored": 0,
                "blocked_reason": reason_category,
                "self_intent_redirect": True  # Signal to routing that this should go to self_model
            }

        # ============================================================
        # STEP 3: LEARNING MODE GATE (after memory miss)
        # ============================================================
        # Now we apply learning mode semantics:
        # - TRAINING: Call LLM, store lessons/facts
        # - OFFLINE: No LLM call, no storage (evaluation-only mode)
        # - SHADOW: Call LLM for comparison, no storage
        # ============================================================
        if learning_mode == LearningMode.OFFLINE:
            # OFFLINE = evaluation-only, no new learning
            # Memory already checked and missed, so return NO_MEMORY
            print(f"[TEACHER_HELPER] Mode=OFFLINE, memory miss → returning NO_MEMORY (no LLM call)")
            return {
                "answer": None,
                "verdict": "NO_MEMORY",
                "source": "memory_miss_offline",
                "facts_stored": 0,
                "patterns_stored": 0,
                "learning_mode": str(learning_mode)
            }

        # ============================================================
        # STEP 4: CALL TEACHER (TRAINING and SHADOW modes only)
        # ============================================================
        # At this point:
        # - Memory was checked and missed
        # - Not a self-query
        # - Mode is TRAINING or SHADOW
        # ============================================================
        operation = self.contract["operation"]
        prompt_mode = self.contract["prompt_mode"]

        is_shadow_mode = (learning_mode == LearningMode.SHADOW)

        if is_shadow_mode:
            print(f"[TEACHER_HELPER] Mode=SHADOW, calling LLM for comparison only (no storage)")
        else:
            print(f"[TEACHER_HELPER] Mode=TRAINING, calling LLM to learn and store")

        try:
            if operation == "TEACH":
                response = self._call_teach(question, context, prompt_mode, retrieved_facts)
            elif operation == "TEACH_ROUTING":
                response = self._call_teach_routing(question, context)
            else:
                print(f"[TEACHER_HELPER_ERROR] Unknown operation '{operation}' for {self.brain_id}")
                return None

            if not response or not response.get("ok"):
                return None

            # ============================================================
            # STEP 5: PROCESS AND STORE RESULTS
            # ============================================================
            # Storage behavior depends on learning mode:
            # - TRAINING: Store lessons/facts to memory (normal learning)
            # - SHADOW: NO storage, answer for comparison only
            # ============================================================
            payload = response["payload"]
            result = {
                "verdict": payload.get("verdict", "LEARNED"),
                "answer": payload.get("answer"),
                "source": "shadow_comparison" if is_shadow_mode else "teacher",
                "facts_stored": 0,
                "patterns_stored": 0,
                "learning_mode": str(learning_mode)
            }

            # Only store in TRAINING mode (not SHADOW)
            if not is_shadow_mode:
                # TASK 4: Check if caller explicitly blocked fact storage
                # This is used by sensorium to ensure classification patterns
                # are never stored as world/personal facts
                block_facts = context.get("block_fact_storage", False) if context else False
                classification_only = context.get("classification_only", False) if context else False

                # Store based on contract
                if self.contract.get("store_domain", False):
                    # TASK 4: Skip domain storage if block_fact_storage is set
                    if block_facts or classification_only:
                        print(f"[TEACHER_HELPER_TASK4] Skipping domain fact storage (classification_only mode)")
                        result["facts_stored"] = 0
                    else:
                        # Store world facts to domain brains
                        facts = payload.get("candidate_facts", [])
                        result["facts_stored"] = self._store_facts_to_domain(facts, question)

                if self.contract.get("store_internal", False):
                    # Store internal patterns to own memory
                    # Patterns can come from different fields depending on operation
                    patterns = payload.get("patterns", [])
                    if not patterns:
                        # For TEACH_ROUTING, routes are the patterns
                        patterns = payload.get("routes", [])

                    result["patterns_stored"] = self._store_patterns_internal(patterns, question, payload)

                print(f"[TEACHER_HELPER] ✓ LEARNED: stored {result['facts_stored']} facts, {result['patterns_stored']} patterns")
            else:
                print(f"[TEACHER_HELPER] SHADOW mode: LLM answer returned for comparison (no storage)")

            return result

        except Exception as e:
            print(f"[TEACHER_HELPER_ERROR] Brain {self.brain_id}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return None

    def _check_memory(self, question: str) -> Optional[str]:
        """
        Check if answer already exists in memory.

        MEMORY-FIRST FIX: Now checks for lessons and facts by matching
        the 'concept_key' metadata (not just original_question or content).
        This ensures questions like "what are birds" and "are birds" both
        find the same stored lesson.

        Args:
            question: The question to check

        Returns:
            Stored answer or None if not found
        """
        if not self.memory:
            return None

        # Compute concept_key for this question
        concept_key = canonical_concept_key(question)
        normalized_q = question.lower().strip().rstrip("?").strip()

        print(f"[TEACHER_HELPER] Memory check with concept_key='{concept_key}' for question='{question[:50]}...'")

        try:
            # ============================================================
            # CONCEPT-KEY BASED LOOKUP (PRIMARY)
            # ============================================================
            # Check by concept_key first - this is the primary lookup method.
            # This allows "what are birds" and "are birds" to match.
            # ============================================================
            all_results = self.memory.retrieve(query=None, limit=500)

            for rec in all_results:
                try:
                    # FIX: Access metadata dict - BrainMemory stores metadata nested
                    metadata = rec.get("metadata", {}) or {}

                    # Check for concept_key metadata match (PRIMARY)
                    stored_concept_key = str(metadata.get("concept_key", "")).strip()
                    if stored_concept_key and stored_concept_key == concept_key:
                        # Found a record with matching concept_key!
                        confidence = metadata.get("confidence", 0.0)
                        if confidence >= 0.5:
                            content = rec.get("content")
                            kind = metadata.get("kind", "")
                            rec_type = metadata.get("type", "")

                            # For lessons, extract the distilled rule or answer
                            if kind == "lesson" or rec_type == "lesson":
                                if isinstance(content, dict):
                                    answer = content.get("distilled_rule") or content.get("answer")
                                    if answer:
                                        print(f"[TEACHER_HELPER] ✓ Found lesson by concept_key='{concept_key}'")
                                        return str(answer)
                                elif content:
                                    print(f"[TEACHER_HELPER] ✓ Found lesson content by concept_key='{concept_key}'")
                                    return str(content)

                            # For other records (facts, patterns, etc.)
                            elif content:
                                if isinstance(content, dict):
                                    return json.dumps(content)
                                else:
                                    print(f"[TEACHER_HELPER] ✓ Found memory record by concept_key='{concept_key}'")
                                    return str(content)

                    # FALLBACK: Check for original_question metadata match (exact normalized)
                    orig_q = str(metadata.get("original_question", "")).lower().strip().rstrip("?").strip()
                    if orig_q and orig_q == normalized_q:
                        confidence = metadata.get("confidence", 0.0)
                        if confidence >= 0.5:
                            content = rec.get("content")
                            kind = metadata.get("kind", "")
                            rec_type = metadata.get("type", "")

                            if kind == "lesson" or rec_type == "lesson":
                                if isinstance(content, dict):
                                    answer = content.get("distilled_rule") or content.get("answer")
                                    if answer:
                                        print(f"[TEACHER_HELPER] ✓ Found lesson by original_question match")
                                        return str(answer)
                                elif content:
                                    return str(content)
                            elif content:
                                if isinstance(content, dict):
                                    return json.dumps(content)
                                else:
                                    print(f"[TEACHER_HELPER] ✓ Found memory record by original_question match")
                                    return str(content)

                    # Also check for qa_memory records (stored by reasoning brain)
                    kind = metadata.get("kind", "")
                    if kind == "qa_memory":
                        stored_q = str(metadata.get("question", "")).lower().strip().rstrip("?").strip()
                        if stored_q == normalized_q:
                            answer = metadata.get("answer")
                            if answer:
                                print(f"[TEACHER_HELPER] ✓ Found qa_memory answer by question match")
                                return str(answer)

                except Exception:
                    continue

            # Fallback: original content-based search for backwards compatibility
            results = self.memory.retrieve(query=question, limit=10)
            for rec in results:
                try:
                    confidence = rec.get("confidence", 0.0)
                    if confidence < 0.7:
                        continue

                    content = rec.get("content")
                    if not content:
                        continue

                    if isinstance(content, dict):
                        return json.dumps(content)
                    else:
                        return str(content)

                except Exception:
                    continue

            return None

        except Exception:
            return None

    def _check_domain_banks(self, question: str) -> Optional[str]:
        """
        Check domain banks for relevant world facts.

        This is part of the memory-first architecture. After checking the
        brain's own memory, we also check domain banks for world facts
        that might answer the question.

        CONCEPT-KEY FIX: Now checks by 'concept_key' metadata match,
        allowing "what are birds" and "are birds" to find the same facts.

        Args:
            question: The question to check

        Returns:
            Relevant fact content or None if not found
        """
        if not BrainMemory or not get_domain_brains:
            return None

        # Compute concept_key for this question
        concept_key = canonical_concept_key(question)
        normalized_q = question.lower().strip().rstrip("?").strip()

        try:
            # Get list of domain brains
            domain_brains = get_domain_brains()

            for domain_name in domain_brains:
                try:
                    domain_memory = BrainMemory(domain_name)

                    # ============================================================
                    # CONCEPT-KEY BASED LOOKUP (PRIMARY)
                    # ============================================================
                    # Check by concept_key first - this allows "what are birds"
                    # and "are birds" to match facts stored with concept_key="birds"
                    # ============================================================
                    all_facts = domain_memory.retrieve(query=None, limit=500)

                    for rec in all_facts:
                        try:
                            # FIX: Access metadata dict - BrainMemory stores metadata nested
                            metadata = rec.get("metadata", {}) or {}

                            # Check for concept_key metadata match (PRIMARY)
                            stored_concept_key = str(metadata.get("concept_key", "")).strip()
                            if stored_concept_key and stored_concept_key == concept_key:
                                confidence = metadata.get("confidence", 0.0)
                                if confidence >= 0.5:
                                    content = rec.get("content")
                                    if content:
                                        print(f"[TEACHER_HELPER] ✓ Found domain fact by concept_key='{concept_key}' in {domain_name}: {str(content)[:50]}...")
                                        if isinstance(content, dict):
                                            return json.dumps(content)
                                        else:
                                            return str(content)

                            # FALLBACK: Check for original_question metadata match
                            orig_q = str(metadata.get("original_question", "")).lower().strip().rstrip("?").strip()
                            if orig_q and orig_q == normalized_q:
                                confidence = metadata.get("confidence", 0.0)
                                if confidence >= 0.5:
                                    content = rec.get("content")
                                    if content:
                                        print(f"[TEACHER_HELPER] ✓ Found domain fact by original_question match in {domain_name}: {str(content)[:50]}...")
                                        if isinstance(content, dict):
                                            return json.dumps(content)
                                        else:
                                            return str(content)
                        except Exception:
                            continue

                    # Fallback: content-based search
                    results = domain_memory.retrieve(query=question, limit=5)
                    for rec in results:
                        try:
                            # FIX: Access confidence from metadata
                            rec_metadata = rec.get("metadata", {}) or {}
                            confidence = rec_metadata.get("confidence", 0.0)
                            if confidence < 0.7:
                                continue

                            content = rec.get("content")
                            if not content:
                                continue

                            print(f"[TEACHER_HELPER] Found domain fact in {domain_name}: {str(content)[:50]}...")
                            if isinstance(content, dict):
                                return json.dumps(content)
                            else:
                                return str(content)

                        except Exception:
                            continue

                except Exception:
                    continue

            return None

        except Exception:
            return None

    def _call_teach(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        prompt_mode: str,
        retrieved_facts: Optional[list]
    ) -> Optional[Dict[str, Any]]:
        """
        Call Teacher with TEACH operation.

        Args:
            question: The question
            context: Optional context
            prompt_mode: Prompt template to use
            retrieved_facts: Optional facts from memory

        Returns:
            Teacher response or None
        """
        try:
            from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api  # type: ignore
        except Exception:
            print(f"[TEACHER_HELPER_ERROR] Cannot import teacher brain for {self.brain_id}")
            return None

        # Build prompt using template
        if build_prompt:
            # Use custom prompt builder (future enhancement)
            # For now, pass parameters directly
            pass

        return teacher_api({
            "op": "TEACH",
            "payload": {
                "question": question,
                "context": context or {},
                "retrieved_facts": retrieved_facts or []
            }
        })

    def _call_teach_routing(
        self,
        question: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Call Teacher with TEACH_ROUTING operation.

        Args:
            question: The question
            context: Optional context

        Returns:
            Teacher response or None
        """
        try:
            from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api  # type: ignore
        except Exception:
            print(f"[TEACHER_HELPER_ERROR] Cannot import teacher brain for {self.brain_id}")
            return None

        # Get domain brains for routing
        if get_domain_brains:
            available_banks = get_domain_brains()
        else:
            available_banks = []

        return teacher_api({
            "op": "TEACH_ROUTING",
            "payload": {
                "question": question,
                "available_banks": available_banks,
                "context": context or {}
            }
        })

    def _store_facts_to_domain(
        self,
        facts: list,
        original_question: str
    ) -> int:
        """
        Store facts to appropriate domain brains.

        This implements the domain storage pattern:
        - Classify each fact with TruthClassifier
        - Only store non-RANDOM facts
        - Route to correct domain brain based on fact type

        Args:
            facts: List of fact dicts from Teacher
            original_question: The original question

        Returns:
            Number of facts stored
        """
        if not TruthClassifier or not is_domain_brain or not BrainMemory:
            return 0

        # TASK 2: EARLY BLOCK - If question is about Maven's internal state,
        # block ALL fact storage. Teacher NEVER writes facts about Maven.
        if self._is_self_query(original_question):
            print(f"[TEACHER_HELPER_BLOCK_ALL] Question is about Maven's self/feelings - blocking ALL fact storage")
            print(f"[TEACHER_HELPER_BLOCK_ALL] Question: '{original_question[:60]}...'")
            return 0

        def _looks_like_self_fact(fact: str, original_question: str) -> bool:
            """
            TASK 2: Enhanced guard to block Teacher from writing self-facts.

            Teacher can never claim to know Maven's internal state.
            Facts about Maven's identity, feelings, preferences, opinions
            are NEVER sourced from Teacher - only from self_model/self_dmn.
            """
            text = (fact or "").lower()
            q = (original_question or "").lower()

            # if the question was about Maven itself, treat all facts as self-facts
            if "who are you" in q or "what are you" in q:
                return True
            if "tell me about yourself" in q or "tell me about your self" in q:
                return True
            if "tell me about you" in q:
                return True

            # TASK 2: Block facts from feelings/preferences questions
            # These questions about Maven's internal state should NEVER
            # result in facts being stored by Teacher
            feelings_question_patterns = [
                "how do you feel", "how are you feeling",
                "do you have feelings", "do you have emotions",
                "are you happy", "are you sad", "are you conscious",
                "are you sentient", "are you alive", "are you real",
                "do you like", "do you enjoy", "do you prefer", "do you want",
                "what do you like", "what do you want", "what do you prefer",
                "do you have preferences", "do you have opinions",
                "what is your opinion", "how do you think",
                "do you dream", "do you sleep", "do you get tired",
            ]
            if any(p in q for p in feelings_question_patterns):
                print(f"[FACT_BLOCK_FEELINGS] Blocking fact from feelings/preferences question")
                return True

            # generic "I am / I was created / I do not have experiences" style
            self_phrases = [
                "i am an artificial intelligence",
                "i am a large language model",
                "i'm a large language model",
                "i am an llm",
                "i'm an llm",
                "i am chatgpt",
                "i am gpt",
                "i am claude",
                "i was created by",
                "i do not have personal experiences",
                "i do not have emotions",
                "i was built by a team of",
                "large language model",  # Block any mention of "large language model" about self
            ]
            if any(p in text for p in self_phrases):
                print(f"[FACT_BLOCK_IDENTITY] Blocking identity lie: '{text[:60]}...'")
                return True

            # CRITICAL: Block facts claiming Maven cannot do things it CAN do
            # Teacher may hallucinate these false capability statements
            capability_lies = [
                # Time-related lies
                "i do not have the ability to query external time",
                "i cannot access the current time",
                "i don't have access to the current time",
                "i do not have access to real-time",
                "i cannot tell the time",
                "i don't know what time it is",
                "i am unable to access the time",
                "i cannot access a clock",
                "no ability to query external time",
                "cannot provide real-time information",
                # Web search lies
                "i cannot search the web",
                "i cannot browse the internet",
                "i do not have internet access",
                "i don't have access to the internet",
                "i cannot access external websites",
                "i am unable to browse",
                "i cannot look things up online",
                # Filesystem lies
                "i cannot read files",
                "i cannot access files",
                "i do not have access to your filesystem",
                "i cannot read or write files",
                "i am unable to access files",
                # Code execution lies
                "i cannot execute code",
                "i cannot run code",
                "i do not have the ability to run code",
                "i am unable to execute",
                # Git lies
                "i cannot access git",
                "i do not have git access",
                "i cannot run git commands",
                # Browser lies
                "i cannot control a browser",
                "i do not have browser access",
                "i cannot automate a browser",
            ]
            if any(p in text for p in capability_lies):
                print(f"[FACT_BLOCK_CAPABILITY_LIE] Blocking false capability claim: '{text[:60]}...'")
                return True

            # TASK 2: Block facts that describe Maven's feelings/preferences
            # These should NEVER be stored from Teacher
            self_feeling_phrases = [
                "i feel",
                "i am feeling",
                "i do not feel",
                "i don't feel",
                "i am happy",
                "i am sad",
                "i am not happy",
                "i am not sad",
                "i like",
                "i don't like",
                "i do not like",
                "i enjoy",
                "i prefer",
                "i want",
                "i think",
                "i believe",
                "my feelings",
                "my emotions",
                "my preferences",
                "my opinions",
                "i have feelings",
                "i have emotions",
                "i have preferences",
                "i have opinions",
                "i am conscious",
                "i am sentient",
                "i am not conscious",
                "i am not sentient",
            ]
            if any(p in text for p in self_feeling_phrases):
                print(f"[FACT_BLOCK_FEELINGS] Blocking self-feeling fact: '{text[:60]}...'")
                return True

            return False

        stored_count = 0

        for fact in facts:
            try:
                fact_statement = fact.get("statement", "")
                if not fact_statement:
                    continue

                # Classify with TruthClassifier
                classification = TruthClassifier.classify(
                    content=fact_statement,
                    confidence=0.7,  # Teacher facts start with moderate confidence
                    evidence=None
                )

                # Only store if allowed (not RANDOM)
                if classification["type"] != "RANDOM" and classification["allow_memory_write"]:
                    # Determine target domain brain
                    fact_type = fact.get("type", "world_fact")

                    # Route based on fact type
                    if fact_type == "personal_fact":
                        # CRITICAL: Block Teacher from storing self-identity facts to personal
                        if self._is_self_query(original_question):
                            print(f"[TEACHER_HELPER_BLOCK] Blocked Teacher from storing self-identity fact to personal")
                            print(f"[TEACHER_HELPER_BLOCK] Fact: {fact_statement[:80]}...")
                            continue  # Skip this fact
                        target_bank = "personal"
                    else:
                        # For now, all world facts go to factual
                        # Future: smarter routing based on content
                        target_bank = "factual"

                    # Verify it's a domain brain
                    if not is_domain_brain(target_bank):
                        print(f"[WARNING] Attempted to store to non-domain brain: {target_bank}")
                        continue

                    # Store to domain brain
                    try:
                        if target_bank == "personal" and _looks_like_self_fact(fact_statement, original_question):
                            # never store LLM self-facts about Maven
                            print(f"[FACT_SKIPPED_SELF] {self.brain_id} → {target_bank}: {fact_statement[:60]}...")
                            continue

                        # Check if direct fact write is allowed
                        # In proposal mode, facts are stored as tentative (not committed)
                        # Brains must confirm before promoting to committed status
                        if is_proposal_mode_enabled() and not is_direct_fact_write_enabled():
                            # Store as TENTATIVE - not yet committed
                            status = "tentative"
                            kind = "teacher_hypothesis"  # Mark as hypothesis from Teacher
                            print(f"[PROPOSAL_MODE] Storing as tentative hypothesis (not committed)")
                        else:
                            # Legacy mode: store directly
                            status = "committed"
                            kind = "learned_fact"

                        domain_memory = BrainMemory(target_bank)
                        # Compute concept_key for memory-first lookups
                        concept_key = canonical_concept_key(original_question)
                        domain_memory.store(
                            content=fact_statement,
                            metadata={
                                "kind": kind,
                                "status": status,  # NEW: tentative or committed
                                "source": f"llm_teacher_via_{self.brain_id}",
                                "original_question": original_question,
                                "concept_key": concept_key,
                                "confidence": classification["confidence"],
                                "truth_type": classification["type"],
                                "fact_type": fact_type,
                                "requires_confirmation": status == "tentative"  # Flag for brains to check
                            }
                        )
                        stored_count += 1
                        print(f"[FACT_STORED:{status.upper()}] {self.brain_id} → {target_bank} (concept_key='{concept_key}'): {fact_statement[:60]}...")
                    except Exception as e:
                        print(f"[ERROR] Failed to store fact to {target_bank}: {str(e)[:100]}")
                        continue

            except Exception as e:
                print(f"[ERROR] Failed to process fact: {str(e)[:100]}")
                continue

        if stored_count == 0:
            print(f"[WARNING] {self.brain_id}: No facts stored from Teacher")

        return stored_count

    def _store_patterns_internal(
        self,
        patterns: list,
        original_question: str,
        full_payload: Dict[str, Any]
    ) -> int:
        """
        Store internal patterns to brain's own memory.

        This implements the internal storage pattern:
        - Store patterns in brain's own memory
        - NOT stored to domain brains
        - Used for brain-specific skills and heuristics

        Args:
            patterns: List of pattern elements
            original_question: The original question
            full_payload: Full Teacher response payload

        Returns:
            Number of patterns stored
        """
        if not self.memory:
            return 0

        stored_count = 0

        # Compute concept_key for memory-first lookups
        concept_key = canonical_concept_key(original_question)

        # Determine if we're in proposal mode
        if is_proposal_mode_enabled() and not is_direct_fact_write_enabled():
            pattern_status = "tentative"
            pattern_kind = "teacher_hypothesis_pattern"
        else:
            pattern_status = "committed"
            pattern_kind = "learned_pattern"

        # Store the full answer as a pattern if available
        answer = full_payload.get("answer")
        if answer:
            try:
                # Store answer as a pattern
                self.memory.store(
                    content=answer,
                    metadata={
                        "kind": pattern_kind,
                        "status": pattern_status,
                        "source": "llm_teacher",
                        "original_question": original_question,
                        "concept_key": concept_key,
                        "confidence": 0.8,
                        "brain_id": self.brain_id,
                        "prompt_mode": self.contract["prompt_mode"],
                        "requires_confirmation": pattern_status == "tentative"
                    }
                )
                stored_count += 1
                print(f"[PATTERN_STORED:{pattern_status.upper()}] {self.brain_id} (concept_key='{concept_key}'): {answer[:60]}...")
            except Exception as e:
                print(f"[ERROR] Failed to store answer pattern: {str(e)[:100]}")

        # Store individual pattern elements if available
        for pattern in patterns:
            try:
                # Extract pattern content (varies by response type)
                if isinstance(pattern, dict):
                    # Pattern is a structured dict (e.g., route with bank + weight)
                    content = pattern
                elif isinstance(pattern, str):
                    # Pattern is a string
                    content = pattern
                else:
                    # Skip unknown types
                    continue

                # Store to own memory with tentative/committed status
                self.memory.store(
                    content=content,
                    metadata={
                        "kind": pattern_kind,
                        "status": pattern_status,
                        "source": "llm_teacher",
                        "original_question": original_question,
                        "concept_key": concept_key,
                        "confidence": 0.8,
                        "brain_id": self.brain_id,
                        "prompt_mode": self.contract["prompt_mode"],
                        "requires_confirmation": pattern_status == "tentative"
                    }
                )
                stored_count += 1

                # Log (truncate content for readability)
                content_str = json.dumps(content) if isinstance(content, dict) else str(content)
                print(f"[PATTERN_STORED] {self.brain_id} (concept_key='{concept_key}'): {content_str[:60]}...")

            except Exception as e:
                print(f"[ERROR] Failed to store pattern: {str(e)[:100]}")
                continue

        if stored_count == 0:
            print(f"[WARNING] {self.brain_id}: No patterns stored from Teacher")

        return stored_count


# =============================================================================
# ROUTING CLASSIFICATION HELPER
# =============================================================================

def get_routing_suggestion(
    user_message: str,
    context_bundle: Optional[Dict[str, Any]] = None,
    capability_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get routing suggestion from Teacher.

    This function calls Teacher as a routing classifier to suggest:
    - Primary intent classification
    - Recommended brains to activate
    - Recommended tools to use
    - Notes explaining the reasoning

    IMPORTANT: The suggestions returned here are NOT validated against
    actual capabilities. The caller MUST intersect these with:
    - Registered brains from brain_roles
    - Available tools from tools_api
    - System capabilities from system_capabilities

    Args:
        user_message: The user's message to classify
        context_bundle: Recent conversation context (2-4 turns)
        capability_snapshot: Current system capability state

    Returns:
        Dict with:
        - primary_intent: str
        - secondary_tags: List[str]
        - recommended_brains: List[str]
        - recommended_tools: List[str]
        - confidence: float
        - notes: str
        Or None if classification failed
    """
    try:
        from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api
    except ImportError:
        print("[ROUTING_HELPER] Teacher brain not available")
        return None

    # Build context string from recent turns
    context_str = "No recent context available."
    if context_bundle:
        recent_turns = context_bundle.get("recent_turns", [])
        if recent_turns:
            context_lines = []
            for turn in recent_turns[-4:]:  # Last 4 turns max
                role = turn.get("role", "unknown")
                content = str(turn.get("content", ""))[:200]  # Truncate
                context_lines.append(f"{role}: {content}")
            context_str = "\n".join(context_lines)

    # Build capability string
    cap_str = "No capability snapshot available."
    if capability_snapshot:
        try:
            available_tools = capability_snapshot.get("tools_available", [])
            exec_mode = capability_snapshot.get("execution_mode", "UNKNOWN")
            web_enabled = capability_snapshot.get("web_research_enabled", False)
            cap_str = f"Execution mode: {exec_mode}\nWeb research: {web_enabled}\nTools: {', '.join(available_tools)}"
        except Exception:
            pass

    # Get available brains and tools
    available_brains = []
    available_tools = []
    try:
        from brains.brain_roles import get_cognitive_brains
        available_brains = list(get_cognitive_brains())
    except ImportError:
        available_brains = ["language", "reasoning", "self_model", "teacher", "integrator"]

    try:
        from brains.system_capabilities import get_available_tools
        available_tools = get_available_tools()
    except ImportError:
        available_tools = []

    # Build the routing classification prompt
    prompt = f"""Analyze this user message and recommend routing.

USER MESSAGE: {user_message}

RECENT CONTEXT:
{context_str}

AVAILABLE CAPABILITIES:
{cap_str}

AVAILABLE BRAINS: {', '.join(available_brains[:20])}
AVAILABLE TOOLS: {', '.join(available_tools) if available_tools else 'none available'}

Your task:
1. Classify the PRIMARY INTENT (one of: chat_answer, code_task, research_question, tool_request, capability_question, self_question, history_question, task_followup, meta_instruction)
2. Extract SECONDARY TAGS (e.g., python, filesystem, git, web)
3. Recommend which BRAINS should handle this (from available brains only)
4. Recommend which TOOLS might be needed (from available tools only)
5. Provide brief NOTES explaining your reasoning

Format your response EXACTLY as JSON:
```json
{{
    "primary_intent": "<intent>",
    "secondary_tags": ["tag1", "tag2"],
    "recommended_brains": ["brain1", "brain2"],
    "recommended_tools": ["tool1", "tool2"],
    "confidence": 0.85,
    "notes": "Brief explanation"
}}
```

CRITICAL: Only suggest brains/tools from the provided lists. Do not invent capabilities."""

    # Call Teacher with CLASSIFY_ROUTING operation
    try:
        response = teacher_api({
            "op": "TEACH",  # Use TEACH op with routing_classification mode
            "payload": {
                "question": prompt,
                "context": {"prompt_mode": "routing_classification"},
                "retrieved_facts": []
            }
        })

        if not response or not response.get("ok"):
            print(f"[ROUTING_HELPER] Teacher call failed: {response}")
            return None

        payload = response.get("payload", {})
        answer = payload.get("answer", "")

        # Parse JSON from answer
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[^}]+\}', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"[ROUTING_HELPER] Could not parse JSON from response: {answer[:200]}")
                return None

        suggestion = json.loads(json_str)

        # Validate the response structure
        result = {
            "primary_intent": suggestion.get("primary_intent", "chat_answer"),
            "secondary_tags": suggestion.get("secondary_tags", []),
            "recommended_brains": suggestion.get("recommended_brains", []),
            "recommended_tools": suggestion.get("recommended_tools", []),
            "confidence": float(suggestion.get("confidence", 0.5)),
            "notes": suggestion.get("notes", ""),
            "raw_response": payload,
        }

        print(f"[ROUTING_HELPER] Classified intent={result['primary_intent']}, "
              f"brains={result['recommended_brains']}, confidence={result['confidence']:.2f}")

        return result

    except json.JSONDecodeError as e:
        print(f"[ROUTING_HELPER] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[ROUTING_HELPER] Classification error: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None


# Convenience function for quick usage
def get_teacher_helper(brain_id: str) -> Optional[TeacherHelper]:
    """
    Get a TeacherHelper instance for a brain.

    This is a convenience function for one-off usage.
    For repeated use, create a TeacherHelper instance and reuse it.

    Args:
        brain_id: Name of the brain

    Returns:
        TeacherHelper instance or None if not available
    """
    try:
        return TeacherHelper(brain_id)
    except Exception as e:
        print(f"[ERROR] Cannot create TeacherHelper for {brain_id}: {str(e)[:100]}")
        return None


# Export public API
__all__ = [
    "TeacherHelper",
    "get_teacher_helper",
    "get_routing_suggestion",
]
