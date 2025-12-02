from __future__ import annotations
from typing import Dict, Any, List
import json
import re
import time
from pathlib import Path

# Import LLM service for calling the external teacher
from brains.tools.llm_service import llm_service as _llm

# Import BrainMemory for telemetry
from brains.memory.brain_memory import BrainMemory

# Import continuation helpers for cognitive brain contract compliance
from brains.cognitive.continuation_helpers import (
    is_continuation,
    get_conversation_context,
    create_routing_hint
)

# Import TeacherProposal for proposal-only mode
from brains.cognitive.teacher.service.teacher_proposal import (
    TeacherProposal,
    Hypothesis,
    StrategySuggestion,
    HypothesisKind,
    create_proposal_from_response,
    create_empty_proposal,
    create_blocked_proposal,
    classify_hypothesis_kind,
)


def _load_feature_flags() -> Dict[str, bool]:
    """Load feature flags from config."""
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

# Initialize memory for teacher telemetry
memory = BrainMemory("teacher")

# Initialize continuation patterns storage (for learning brain coordination)
_continuation_patterns: List[Dict[str, Any]] = []


# ============================================================================
# TEACHER MODES - Direct mapping of brains to their teaching modes
# ============================================================================
# This dict tells Teacher how to behave for each cognitive brain.
# Each mode corresponds to a different prompting strategy.

TEACHER_MODES = {
    # Core learning brains (custom implementations)
    "reasoning": "world_question",
    "memory_librarian": "routing_learning",
    "teacher": None,  # Teacher doesn't learn from itself

    # Cognitive brains - internal patterns/skills
    "planner": "planning_patterns",
    "autonomy": "autonomy_strategies",
    "self_dmn": "self_model_rules",
    "self_model": "self_model_repairs",
    "self_review": "self_review_heuristics",
    "personality": "expression_patterns",
    "system_history": "history_summarization_patterns",
    "pattern_recognition": "pattern_detection_rules",
    "attention": "attention_alloc_rules",
    "learning": "learning_strategies",
    "coder": "coding_heuristics",
    "committee": "committee_voting_rules",
    "environment_context": "environment_cues",
    "external_interfaces": "interface_protocol_patterns",
    "motivation": "motivation_rules",
    "peer_connection": "peer_comm_patterns",
    "thought_synthesis": "synthesis_templates",
    "abstraction": "abstraction_patterns",
    "action_engine": "action_selection_patterns",
    "reasoning_trace": "trace_summarization_patterns",
    "belief_tracker": "belief_update_patterns",
    "context_management": "context_decay_strategies",
    "language": "style_meta",
    "sensorium": "sensorium_patterns",
    "affect_priority": "affect_rules",
    "integrator": "integration_rules",
    "research_manager": "world_question",
    "imaginer": "scenario_generation",

    # Router-Teacher / Normalizer mode (LLM-assisted routing)
    # CRITICAL: This mode is for text normalization and intent classification ONLY.
    # The Teacher MUST NOT answer questions or execute tools in this mode.
    "router_normalizer": "router_normalization",
}


def teach_for_brain(brain_name: str, situation: Dict[str, Any] | str) -> Dict[str, Any]:
    """
    Universal Teacher helper for any cognitive brain.

    This is the single entry point for brains to request teaching.
    Uses TEACHER_MODES to determine how to teach each brain.

    Args:
        brain_name: Name of the requesting brain (e.g., "planner", "autonomy")
        situation: Context about the current situation (dict or string)
                  If dict, should contain relevant context keys
                  If string, treated as a direct question

    Returns:
        Dict with:
            - patterns: List of learned patterns/rules (for internal storage)
            - facts: List of learned facts (for domain storage, if applicable)
            - confidence: float (0.0-1.0) confidence in the teaching
            - verdict: str ("LEARNED", "NO_ANSWER", "ERROR")
            - raw_response: str (full LLM response for debugging)
    """
    # Check if brain has a teaching mode
    mode = TEACHER_MODES.get(brain_name)
    if mode is None:
        return {
            "patterns": [],
            "facts": [],
            "confidence": 0.0,
            "verdict": "ERROR",
            "error": f"No teaching mode defined for brain '{brain_name}'"
        }

    # Check if LLM is available
    if not _llm or not _llm.enabled:
        return {
            "patterns": [],
            "facts": [],
            "confidence": 0.0,
            "verdict": "ERROR",
            "error": "LLM service not available or disabled"
        }

    # Build the teaching prompt based on mode and situation
    try:
        prompt = _build_mode_specific_prompt(brain_name, mode, situation)
    except Exception as e:
        return {
            "patterns": [],
            "facts": [],
            "confidence": 0.0,
            "verdict": "ERROR",
            "error": f"Failed to build prompt: {str(e)}"
        }

    # Call the LLM
    try:
        llm_response = _llm.call(
            prompt=prompt,
            max_tokens=600,
            temperature=0.3,  # Moderate temperature for consistent patterns
            context=situation if isinstance(situation, dict) else {}
        )

        if not llm_response.get("ok"):
            error_msg = llm_response.get("error", "Unknown LLM error")
            return {
                "patterns": [],
                "facts": [],
                "confidence": 0.0,
                "verdict": "ERROR",
                "error": error_msg
            }

        response_text = llm_response.get("text", "")

        # Parse response based on brain type
        # Most cognitive brains learn patterns, reasoning learns facts
        if brain_name == "reasoning":
            # Extract facts for reasoning brain
            facts = _extract_facts_from_response(response_text)
            patterns = []
        else:
            # Extract patterns for cognitive brains
            patterns = _extract_patterns_from_response(response_text, mode)
            facts = []

        # Determine verdict
        has_content = len(patterns) > 0 or len(facts) > 0
        verdict = "LEARNED" if has_content else "NO_ANSWER"

        # Log telemetry
        _store_telemetry("teach_for_brain", {
            "brain_name": brain_name,
            "mode": mode,
            "patterns_count": len(patterns),
            "facts_count": len(facts),
            "verdict": verdict
        })

        # Create routing hint for this teaching session
        routing_hint = create_routing_hint(
            brain_name="teacher",
            action=f"teach_{mode}",
            confidence=0.7 if has_content else 0.0,
            context_tags=["teaching", mode, brain_name]
        )

        return {
            "patterns": patterns,
            "facts": facts,
            "confidence": 0.7 if has_content else 0.0,
            "verdict": verdict,
            "raw_response": response_text,
            "routing_hint": routing_hint
        }

    except Exception as e:
        return {
            "patterns": [],
            "facts": [],
            "confidence": 0.0,
            "verdict": "ERROR",
            "error": f"LLM call failed: {str(e)}"
        }


def _build_mode_specific_prompt(brain_name: str, mode: str, situation: Dict[str, Any] | str) -> str:
    """
    Build a prompt tailored to the brain's specific learning mode.

    Args:
        brain_name: Name of the brain
        mode: Teaching mode from TEACHER_MODES
        situation: Context/question from the brain

    Returns:
        Formatted prompt string
    """
    # Extract question/context from situation
    if isinstance(situation, str):
        question = situation
        context_info = ""
    else:
        question = situation.get("question", str(situation.get("task", "")))
        # Build context string from other keys
        context_keys = {k: v for k, v in situation.items() if k not in ["question", "task"]}
        if context_keys:
            context_info = f"\nContext: {json.dumps(context_keys, indent=2)}\n"
        else:
            context_info = ""

    # Build prompt based on mode
    prompt = f"You are Maven's teacher, helping the {brain_name} brain learn {mode}.\n\n"
    prompt += f"SITUATION: {question}\n"
    if context_info:
        prompt += context_info

    # Mode-specific instructions
    mode_instructions = {
        "world_question": """
Your task: Answer this question factually and extract atomic facts.

Format:
ANSWER: <concise answer>
FACTS:
- <atomic fact 1>
- <atomic fact 2>
""",
        "planning_patterns": """
Your task: Provide a planning pattern or template for this type of task.

Format:
PATTERN:
- <step or principle 1>
- <step or principle 2>
- <step or principle 3>
""",
        "autonomy_strategies": """
Your task: Suggest a task prioritization or resource allocation strategy.

Format:
STRATEGY:
- <principle or rule 1>
- <principle or rule 2>
""",
        "attention_alloc_rules": """
Your task: Suggest attention allocation rules for this situation.

Format:
RULES:
- <attention rule 1>
- <attention rule 2>
""",
        "learning_strategies": """
Your task: Suggest a meta-learning strategy or approach.

Format:
STRATEGY:
- <learning principle 1>
- <learning principle 2>
""",
        "self_model_rules": """
Your task: Suggest self-modeling rules or patterns.

Format:
RULES:
- <self-model rule 1>
- <self-model rule 2>
""",
        "belief_update_patterns": """
Your task: Suggest belief tracking and update patterns.

Format:
PATTERNS:
- <belief pattern 1>
- <belief pattern 2>
""",
        "context_decay_strategies": """
Your task: Suggest context tracking and decay strategies.

Format:
STRATEGIES:
- <strategy 1>
- <strategy 2>
""",
        "router_normalization": """
Your task: Fix typos and classify the intent of the user message.
You are a message NORMALIZER only - do NOT answer the question.

CRITICAL RULES:
- Fix obvious typos and normalize the text
- Classify intent (is_time_query, is_date_query, is_web_search, etc.)
- Do NOT provide information or answer the question
- Return structured JSON output only

Format (JSON only):
{{
    "corrected_text": "<message with typos fixed>",
    "intent_hints": {{
        "is_time_query": <true/false>,
        "is_date_query": <true/false>,
        "is_calendar_query": <true/false>,
        "is_web_search": <true/false>,
        "is_capability_query": <true/false>,
        "is_self_identity": <true/false>,
        "is_follow_up": <true/false>,
        "is_greeting": <true/false>,
        "is_command": <true/false>
    }},
    "confidence": <0.0-1.0>,
    "typo_corrections": {{"<original>": "<corrected>"}},
    "notes": "<brief explanation>"
}}
""",
    }

    # Use mode-specific instructions or generic pattern extraction
    instructions = mode_instructions.get(mode, """
Your task: Provide patterns, rules, or heuristics for this situation.

Format:
PATTERNS:
- <pattern or rule 1>
- <pattern or rule 2>
- <pattern or rule 3>
""")

    prompt += instructions
    prompt += "\nBe concise and specific. Focus on reusable patterns, not one-off solutions.\n\nResponse:"

    return prompt


def _extract_patterns_from_response(response_text: str, mode: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response to extract patterns/rules.

    Args:
        response_text: The raw LLM response
        mode: The teaching mode (for context)

    Returns:
        A list of pattern dictionaries
    """
    patterns: List[Dict[str, Any]] = []

    try:
        lines = response_text.split('\n')
        in_pattern_section = False

        # Look for section markers (PATTERN:, PATTERNS:, RULES:, STRATEGY:, etc.)
        section_markers = ['PATTERN:', 'PATTERNS:', 'RULES:', 'RULE:', 'STRATEGY:', 'STRATEGIES:']

        for line in lines:
            line = line.strip()

            # Detect pattern section
            if any(line.upper().startswith(marker) for marker in section_markers):
                in_pattern_section = True
                # Check if pattern is on same line as marker
                for marker in section_markers:
                    if line.upper().startswith(marker):
                        remainder = line[len(marker):].strip()
                        if remainder and not remainder.startswith('-'):
                            # Single-line pattern
                            patterns.append({
                                "pattern": remainder,
                                "mode": mode,
                                "type": "rule"
                            })
                continue

            # Skip ANSWER/FACTS sections (for reasoning brain)
            if line.upper().startswith('ANSWER:') or line.upper().startswith('FACTS:'):
                in_pattern_section = False
                continue

            # Extract patterns (lines starting with - or *)
            if in_pattern_section and line:
                if line.startswith('-') or line.startswith('*'):
                    pattern_text = line.lstrip('-*').strip()
                    if pattern_text:
                        patterns.append({
                            "pattern": pattern_text,
                            "mode": mode,
                            "type": "rule"
                        })
    except Exception as e:
        # If parsing fails, log error and return empty list
        print(f"[TEACHER] Warning: Failed to parse patterns from response: {str(e)[:100]}")
        pass

    return patterns


def _extract_facts_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response to extract atomic facts.

    The LLM is prompted to return facts in a structured format:
    - ANSWER: <the answer>
    - FACTS:
      - fact 1
      - fact 2

    Args:
        response_text: The raw LLM response

    Returns:
        A list of fact dictionaries with 'statement' and 'type' keys
    """
    facts: List[Dict[str, Any]] = []

    try:
        lines = response_text.split('\n')
        in_facts_section = False

        for line in lines:
            line = line.strip()

            # Detect FACTS section
            if line.upper().startswith('FACTS:'):
                in_facts_section = True
                continue

            # Skip ANSWER section
            if line.upper().startswith('ANSWER:'):
                in_facts_section = False
                continue

            # Extract facts (lines starting with - or *)
            if in_facts_section and line:
                if line.startswith('-') or line.startswith('*'):
                    fact_text = line.lstrip('-*').strip()
                    if fact_text:
                        # Classify fact type based on content
                        fact_type = "world_fact"
                        if any(word in fact_text.lower() for word in ["my", "i ", "you ", "your"]):
                            fact_type = "personal_fact"

                        facts.append({
                            "statement": fact_text,
                            "type": fact_type,
                            "source": "llm_teacher"
                        })
    except Exception as e:
        # If parsing fails, log error and return empty list
        print(f"[TEACHER] Warning: Failed to parse facts from response: {str(e)[:100]}")
        pass

    return facts


def _extract_answer_from_response(response_text: str) -> str | None:
    """
    Extract the answer portion from the LLM response.

    Skips debug headers like PATTERNS:, STRATEGIES:, RULES:, FACTS:, etc.

    Args:
        response_text: The raw LLM response

    Returns:
        The answer text or None if not found
    """
    try:
        lines = response_text.split('\n')

        for line in lines:
            line = line.strip()
            if line.upper().startswith('ANSWER:'):
                # Get everything after "ANSWER:"
                answer = line[7:].strip()
                if answer:
                    return answer

        # Skip response headers that aren't actual answers
        headers = {
            'PATTERNS:', 'PATTERN:',
            'STRATEGIES:', 'STRATEGY:',
            'RULES:', 'RULE:',
            'FACTS:', 'FACT:',
            'NOTES:', 'NOTE:',
            'STEPS:', 'STEP:',
            'ANSWER:'
        }

        # If no explicit ANSWER: marker, try to use the first non-empty line
        # that doesn't start with headers, bullets, or asterisks
        for line in lines:
            line = line.strip()
            if (line and
                not line.startswith('-') and
                not line.startswith('*') and
                not any(line.upper().startswith(h) for h in headers)):
                return line
    except Exception as e:
        print(f"[TEACHER] Warning: Failed to extract answer from response: {str(e)[:100]}")
        pass

    return None


def _build_teacher_prompt(question: str, context: Dict[str, Any], retrieved_facts: List[Dict[str, Any]]) -> str:
    """
    Build a structured prompt for the LLM teacher.

    The prompt asks the LLM to:
    1. Answer the question directly
    2. Provide a list of atomic facts that support the answer

    Args:
        question: The user's question
        context: Context dict with user info, etc.
        retrieved_facts: Any facts already retrieved from memory (may be low confidence)

    Returns:
        A formatted prompt string
    """
    prompt = "You are Maven's teacher. A question has been asked that Maven cannot answer from its own memory.\n\n"
    prompt += f"QUESTION: {question}\n\n"

    # Add user context if available
    try:
        user = context.get("user") or {}
        name = user.get("name")
        if name:
            prompt += f"User's name: {name}\n\n"
    except Exception:
        pass

    # Add retrieved facts if available (even if low confidence)
    if retrieved_facts:
        prompt += "Potentially relevant context from Maven's memory (low confidence):\n"
        for fact in retrieved_facts[:3]:  # Limit to top 3
            try:
                content = str(fact.get("content", "")).strip()
                if content:
                    prompt += f"- {content}\n"
            except Exception:
                pass
        prompt += "\n"

    prompt += """Your task:
1. Provide a direct, concise answer to the question
2. List 2-5 atomic facts that support this answer (facts should be simple, verifiable statements)

Format your response EXACTLY as follows:
ANSWER: <your concise answer here>
FACTS:
- <fact 1>
- <fact 2>
- <fact 3>

Remember:
- Keep the answer concise (1-2 sentences)
- Make facts atomic (one idea per fact)
- Focus on factual information, not speculation
- For personal questions, use facts about the user if mentioned

Response:"""

    return prompt


def _build_routing_prompt(question: str, available_banks: List[str], context: Dict[str, Any]) -> str:
    """
    Build a structured prompt for the LLM teacher to suggest memory bank routing.

    The prompt asks the LLM to:
    1. Identify which memory banks are most relevant for the question
    2. Assign weights to each relevant bank
    3. Suggest keyword aliases for future routing

    Args:
        question: The user's question
        available_banks: List of available memory banks
        context: Context dict with additional info

    Returns:
        A formatted prompt string
    """
    prompt = "You are Maven's routing teacher. Your task is to help Maven learn which memory banks to query for different types of questions.\n\n"
    prompt += f"QUESTION: {question}\n\n"
    prompt += f"AVAILABLE BANKS: {', '.join(available_banks)}\n\n"

    prompt += """Your task:
1. Identify the 1-3 most relevant memory banks for this question
2. Assign a weight (0.0-1.0) to each relevant bank, where 1.0 is most relevant
3. Suggest 2-4 keyword aliases or related phrases that should route to the same banks

Format your response EXACTLY as follows:
ROUTES:
- bank_name: weight (e.g., science: 0.9)
- bank_name: weight
ALIASES:
- alias phrase 1
- alias phrase 2

Example for "How do frogs jump?":
ROUTES:
- science: 0.9
- working_theories: 0.3
ALIASES:
- frog locomotion
- frog jumping biomechanics
- amphibian movement

Remember:
- science: biology, physics, chemistry, natural sciences
- history: historical events, dates, people from the past
- geography: locations, capitals, countries, landmarks
- math: numbers, calculations, equations
- personal: information about the user or Maven itself
- working_theories: unverified hypotheses, low-confidence facts
- theories_and_contradictions: conflicting information
- Use higher weights (0.8-1.0) for primary banks
- Use lower weights (0.2-0.5) for secondary banks

Response:"""

    return prompt


def _extract_routing_from_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the LLM response to extract routing suggestions.

    Expected format:
    ROUTES:
    - bank_name: weight
    - bank_name: weight
    ALIASES:
    - alias 1
    - alias 2

    Args:
        response_text: The raw LLM response

    Returns:
        A dict with 'routes' (list of {bank, weight}) and 'aliases' (list of strings)
    """
    routes: List[Dict[str, Any]] = []
    aliases: List[str] = []

    try:
        lines = response_text.split('\n')
        in_routes_section = False
        in_aliases_section = False

        for line in lines:
            line = line.strip()

            # Detect ROUTES section
            if line.upper().startswith('ROUTES:'):
                in_routes_section = True
                in_aliases_section = False
                continue

            # Detect ALIASES section
            if line.upper().startswith('ALIASES:'):
                in_routes_section = False
                in_aliases_section = True
                continue

            # Extract routes (format: "- bank_name: weight")
            if in_routes_section and line:
                if line.startswith('-') or line.startswith('*'):
                    route_text = line.lstrip('-*').strip()
                    if ':' in route_text:
                        parts = route_text.split(':', 1)
                        bank_name = parts[0].strip()
                        try:
                            weight = float(parts[1].strip())
                            # Clamp weight to 0.0-1.0
                            weight = max(0.0, min(1.0, weight))
                            routes.append({
                                "bank": bank_name,
                                "weight": weight
                            })
                        except (ValueError, IndexError):
                            pass

            # Extract aliases (lines starting with - or *)
            if in_aliases_section and line:
                if line.startswith('-') or line.startswith('*'):
                    alias_text = line.lstrip('-*').strip()
                    if alias_text:
                        aliases.append(alias_text)
    except Exception:
        # If parsing fails, return empty lists
        pass

    return {
        "routes": routes,
        "aliases": aliases
    }


def _store_telemetry(event_type: str, details: Dict[str, Any]) -> None:
    """
    Store telemetry about teacher usage.

    Args:
        event_type: Type of event (e.g., "teacher_called", "facts_extracted")
        details: Additional details to log
    """
    try:
        memory.store(
            content=f"teacher_event:{event_type}",
            metadata={
                "kind": "teacher_telemetry",
                "event_type": event_type,
                "details": details,
                "confidence": 1.0
            }
        )
    except Exception as e:
        # Silently ignore telemetry errors (non-critical)
        # Note: Could log to stderr if needed: print(f"[TEACHER] Telemetry error: {e}", file=sys.stderr)
        pass


# ============================================================================
# CONTINUATION PATTERN LEARNING - Cognitive Brain Contract Compliance
# ============================================================================

def _extract_query_pattern(query: str) -> str:
    """
    Extract pattern from query for future matching.

    Args:
        query: User query text

    Returns:
        Pattern classification string
    """
    query_lower = query.lower()

    # Common continuation patterns
    patterns = {
        "tell me more": "expansion_request",
        "what about": "specific_expansion",
        "continue": "continuation_request",
        "expand on": "expansion_request",
        "more details": "detail_request",
        "what else": "alternative_request",
        "can you elaborate": "elaboration_request",
        "go on": "continuation_request",
        "keep going": "continuation_request",
        "and then": "sequence_continuation"
    }

    for phrase, pattern in patterns.items():
        if phrase in query_lower:
            return pattern

    return "general_follow_up"


def _calculate_response_quality(integrator_response: Dict[str, Any]) -> float:
    """
    Calculate quality score for response (0-1).

    Args:
        integrator_response: Response from Integrator

    Returns:
        Quality score between 0.0 and 1.0
    """
    # Check if Self-Review brain was activated and what it said
    activated_brains = integrator_response.get("activated_brains", [])

    for brain in activated_brains:
        if brain.get("brain_name") == "self_review":
            evidence = brain.get("evidence", {})
            quality = evidence.get("quality_score", 0.5)
            return float(quality)

    # If no self-review, check for error indicators
    if integrator_response.get("error"):
        return 0.3

    # Default: assume moderate quality if no review
    return 0.7


def _store_continuation_pattern(pattern: Dict[str, Any]) -> None:
    """
    Store learned continuation pattern for future use.

    Args:
        pattern: Pattern dictionary with trigger, pipeline, success info
    """
    global _continuation_patterns

    # Add to in-memory pattern bank
    _continuation_patterns.append(pattern)

    # Also store in memory for persistence
    try:
        memory.store(
            content=f"continuation_pattern:{pattern.get('user_query_pattern', 'unknown')}",
            metadata={
                "kind": "continuation_pattern",
                "pattern": pattern,
                "confidence": pattern.get("success_score", 0.7)
            }
        )
    except Exception as e:
        print(f"[TEACHER] Warning: Failed to persist pattern: {str(e)[:100]}")


def learn_from_execution(integrator_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Learn from how Integrator coordinated brains for this query.

    This is the core learning function that enables Teacher to improve
    Maven's brain coordination over time by detecting successful patterns.

    Args:
        integrator_response: The response from Integrator after brain coordination
        context: Context dict with user_query and other metadata

    Returns:
        Learning result with patterns learned, confidence, etc.
    """
    try:
        # Get conversation context
        conv_context = get_conversation_context()

        # Detect if this was a continuation query
        user_query = context.get("user_query", "")
        is_follow_up = is_continuation(user_query, context)

        # Extract activated brains
        activated_brains = integrator_response.get("activated_brains", [])

        if not activated_brains:
            return {
                "learned": False,
                "reason": "no_brains_activated",
                "patterns_learned": 0
            }

        # Collect routing hints from all brains
        routing_hints = []
        for brain_result in activated_brains:
            evidence = brain_result.get("evidence", {})
            if "routing_hint" in evidence:
                routing_hints.append({
                    "brain": brain_result.get("brain_name"),
                    "hint": evidence["routing_hint"],
                    "priority": brain_result.get("priority", 0)
                })

        # If this was a successful follow-up, learn the pattern
        if is_follow_up and routing_hints:
            success_score = _calculate_response_quality(integrator_response)

            # Only learn from successful patterns (threshold: 0.7)
            if success_score > 0.7:
                pattern = {
                    "trigger_type": "follow_up_question",
                    "base_topic": conv_context.get("last_topic", "unknown"),
                    "user_query_pattern": _extract_query_pattern(user_query),
                    "user_query_sample": user_query[:100],  # Store sample for debugging
                    "successful_pipeline": [b["brain"] for b in routing_hints],
                    "routing_hints_observed": routing_hints,
                    "success_score": success_score,
                    "timestamp": time.time()
                }

                # Store this successful continuation pattern
                _store_continuation_pattern(pattern)

                # Log learning
                print(f"[TEACHER] ✓ Learned continuation pattern: "
                      f"{pattern['user_query_pattern']} -> "
                      f"{pattern['successful_pipeline']}")

                # Store telemetry
                _store_telemetry("pattern_learned", {
                    "pattern_type": pattern['user_query_pattern'],
                    "pipeline": pattern['successful_pipeline'],
                    "success_score": success_score
                })

                return {
                    "learned": True,
                    "reason": "successful_continuation_pattern",
                    "patterns_learned": 1,
                    "pattern": pattern,
                    "success_score": success_score
                }
            else:
                return {
                    "learned": False,
                    "reason": "quality_below_threshold",
                    "success_score": success_score,
                    "threshold": 0.7
                }

        # Not a follow-up, but still collect data
        return {
            "learned": False,
            "reason": "not_a_follow_up" if not is_follow_up else "no_routing_hints",
            "is_follow_up": is_follow_up,
            "routing_hints_count": len(routing_hints)
        }

    except Exception as e:
        return {
            "learned": False,
            "reason": "error",
            "error": str(e)
        }


def get_learned_patterns(query_pattern: str = None) -> List[Dict[str, Any]]:
    """
    Retrieve learned continuation patterns, optionally filtered by pattern type.

    Args:
        query_pattern: Optional pattern type to filter by (e.g., "expansion_request")

    Returns:
        List of matching patterns
    """
    global _continuation_patterns

    if query_pattern:
        return [p for p in _continuation_patterns
                if p.get("user_query_pattern") == query_pattern]

    return _continuation_patterns


def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for the teacher brain.

    Supports operations:
    - TEACH: Ask the LLM teacher for an answer and extract facts
    - TEACH_ROUTING: Ask the LLM teacher for routing suggestions
    - STATS: Get teacher usage statistics
    - HEALTH: Health check

    Args:
        msg: Message dict with 'op' and 'payload'

    Returns:
        Response dict with 'ok', 'op', and 'payload'
    """
    op = (msg or {}).get("op", "").upper()
    mid = (msg or {}).get("mid")
    payload = (msg or {}).get("payload") or {}

    # Health check
    if op == "HEALTH":
        return {"ok": True, "op": op, "mid": mid, "payload": {"status": "ok"}}

    # Get teacher usage statistics
    if op == "STATS":
        try:
            # Retrieve telemetry records
            results = memory.retrieve(query="teacher_event:", limit=1000)

            # Count by event type
            event_counts: Dict[str, int] = {}
            for rec in results:
                if rec.get("kind") == "teacher_telemetry":
                    event_type = rec.get("event_type", "unknown")
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1

            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "total_teacher_calls": event_counts.get("teacher_called", 0),
                    "total_facts_extracted": event_counts.get("facts_extracted", 0),
                    "total_facts_stored": event_counts.get("facts_stored", 0),
                    "event_breakdown": event_counts
                }
            }
        except Exception as e:
            return {"ok": False, "op": op, "mid": mid, "error": {"code": "STATS_ERROR", "message": str(e)}}

    # TEACH operation: ask the LLM and extract facts
    if op == "TEACH":
        # Check if LLM service is available
        if not _llm or not _llm.enabled:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LLM_UNAVAILABLE", "message": "LLM service not available or disabled"}
            }

        # Extract parameters
        question = str(payload.get("question", "")).strip()
        context = payload.get("context") or {}
        retrieved_facts = payload.get("retrieved_facts") or []

        if not question:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_QUESTION", "message": "Question is required"}
            }

        # COGNITIVE BRAIN CONTRACT: Signal 1 - Detect continuation
        continuation_detected = is_continuation(question, context)

        # COGNITIVE BRAIN CONTRACT: Signal 2 - Pull conversation context if continuation
        if continuation_detected:
            conv_context = get_conversation_context()
            # Enrich context with conversation history
            context["continuation_detected"] = True
            context["last_topic"] = conv_context.get("last_topic", "")
            context["conversation_depth"] = conv_context.get("conversation_depth", 0)

        # --------------------------------------------------------------------
        # SELF-KNOWLEDGE GUARD: Block Teacher from teaching self-identity facts
        # --------------------------------------------------------------------
        # Teacher MUST NOT teach facts about Maven's own identity, systems, or code.
        # These questions should be handled by self_model/self_dmn ONLY.
        #
        # If a self-identity question leaks through to Teacher (shouldn't happen
        # if reasoning gate works, but defense in depth), reject it here.
        try:
            q_lower = question.strip().lower()

            # Define self-identity patterns (same as reasoning brain gate)
            self_identity_patterns = [
                r"\bwho are you\b",
                r"\bwhat.*your\s+(own\s+)?code\b",
                r"\bwhat.*your\s+(own\s+)?systems?\b",
                r"\bwhat.*you\s+built\b",
                r"\bhow\s+do\s+you\s+work\b",
                r"\bare\s+you\s+(an?\s+)?llm\b",
                r"\bwhat.*you\s+know\s+about\s+your\s*(self|own)?\b",
                r"\btell\s+me\s+about\s+yourself\b",
                r"\bwhat.*your\s+name\b",
                r"\bare\s+you\s+maven\b",
                r"\bwhat.*maven.*you\b",
                r"\byour\s+identity\b",
                r"\byour\s+architecture\b",
                r"\byour\s+implementation\b",
                # Additional patterns that detect "Maven" as subject
                r"\bmaven\s+(is|was|can|does)\b",
                r"\bwhat\s+is\s+maven\b",
                r"\bwho\s+(created|built|made)\s+maven\b"
            ]

            # Check if question matches any self-identity pattern
            is_self_query = any(re.search(pattern, q_lower) for pattern in self_identity_patterns)

            if is_self_query:
                print(f"[TEACHER_SELF_BLOCK] Rejecting self-identity question: {question[:60]}...")
                print(f"[TEACHER_SELF_BLOCK] Teacher must NOT teach self-facts")

                # In proposal mode, return a blocked proposal
                if is_proposal_mode_enabled():
                    blocked_proposal = create_blocked_proposal(
                        question,
                        "Teacher cannot teach self-identity facts. Route to self_model instead."
                    )
                    return {
                        "ok": True,
                        "op": op,
                        "mid": mid,
                        "payload": {
                            "proposal": blocked_proposal.to_dict(),
                            "answer": None,
                            "candidate_facts": [],
                            "verdict": "SELF_KNOWLEDGE_FORBIDDEN",
                            "raw_response": "",
                            "llm_source": "blocked",
                            "confidence": 0.0,
                            "error": "Teacher cannot teach self-identity facts. Route to self_model instead."
                        }
                    }

                # Legacy mode: return old format
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "answer": None,
                        "candidate_facts": [],  # NO facts for self-queries
                        "verdict": "SELF_KNOWLEDGE_FORBIDDEN",  # Special verdict
                        "raw_response": "",
                        "llm_source": "blocked",
                        "confidence": 0.0,
                        "error": "Teacher cannot teach self-identity facts. Route to self_model instead."
                    }
                }
        except Exception as e:
            print(f"[TEACHER_SELF_BLOCK_ERROR] Pattern check failed: {str(e)[:100]}")
            # Continue to LLM call if pattern check fails (fail open)
            pass

        # Build the teacher prompt
        prompt = _build_teacher_prompt(question, context, retrieved_facts)

        # Call the LLM teacher
        try:
            llm_response = _llm.call(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,  # Lower temperature for more factual responses
                context=context
            )

            if not llm_response.get("ok"):
                error_msg = llm_response.get("error", "Unknown LLM error")
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "LLM_CALL_FAILED", "message": error_msg}
                }

            response_text = llm_response.get("text", "")

            # Extract answer and facts
            answer = _extract_answer_from_response(response_text)
            facts = _extract_facts_from_response(response_text)

            # Log telemetry
            _store_telemetry("teacher_called", {
                "question": question[:100],  # Truncate for storage
                "answer_provided": answer is not None,
                "facts_count": len(facts),
                "llm_source": llm_response.get("source", "unknown"),
                "proposal_mode": is_proposal_mode_enabled()
            })

            if len(facts) > 0:
                _store_telemetry("facts_extracted", {"count": len(facts)})

            # Determine verdict based on whether we learned something
            verdict = "PROPOSAL" if is_proposal_mode_enabled() else ("LEARNED" if answer and len(facts) > 0 else "NO_ANSWER")

            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            routing_hint = create_routing_hint(
                brain_name="teacher",
                action=f"teach_{verdict.lower()}",
                confidence=0.7 if (answer and len(facts) > 0) else 0.3,
                context_tags=[
                    "teaching",
                    "proposal_mode" if is_proposal_mode_enabled() else "legacy_mode",
                    "fact_extraction" if len(facts) > 0 else "no_facts",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )

            # In proposal mode, wrap everything in a TeacherProposal
            if is_proposal_mode_enabled():
                # Create proposal from extracted facts/answer
                # NOTE: Facts are now hypotheses with status="proposal"
                # Brains must evaluate and decide what to commit
                proposal = create_proposal_from_response(
                    response_text=response_text,
                    answer=answer,
                    facts=facts,
                    patterns=None,
                    original_question=question
                )

                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {
                        "proposal": proposal.to_dict(),
                        "answer": answer,
                        "candidate_facts": facts,  # Legacy field for backwards compat
                        "hypotheses": [h.to_dict() for h in proposal.hypotheses],
                        "verdict": verdict,
                        "raw_response": response_text,
                        "llm_source": llm_response.get("source", "unknown"),
                        "confidence": 0.7,
                        "routing_hint": routing_hint
                    }
                }

            # Legacy mode: return the teaching result directly
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "answer": answer,
                    "candidate_facts": facts,
                    "verdict": verdict,
                    "raw_response": response_text,
                    "llm_source": llm_response.get("source", "unknown"),
                    "confidence": 0.7,
                    "routing_hint": routing_hint
                }
            }

        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "TEACH_EXCEPTION", "message": str(e)}
            }

    # TEACH_ROUTING operation: ask the LLM for routing suggestions
    if op == "TEACH_ROUTING":
        # Check if LLM service is available
        if not _llm or not _llm.enabled:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LLM_UNAVAILABLE", "message": "LLM service not available or disabled"}
            }

        # Extract parameters
        question = str(payload.get("question", "")).strip()
        available_banks = payload.get("available_banks") or []
        context = payload.get("context") or {}

        if not question:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_QUESTION", "message": "Question is required"}
            }

        if not available_banks:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_BANKS", "message": "Available banks list is required"}
            }

        # COGNITIVE BRAIN CONTRACT: Signal 1 - Detect continuation
        continuation_detected = is_continuation(question, context)

        # COGNITIVE BRAIN CONTRACT: Signal 2 - Pull conversation context if continuation
        if continuation_detected:
            conv_context = get_conversation_context()
            context["continuation_detected"] = True
            context["last_topic"] = conv_context.get("last_topic", "")
            context["conversation_depth"] = conv_context.get("conversation_depth", 0)

        # Build the routing prompt
        prompt = _build_routing_prompt(question, available_banks, context)

        # Call the LLM teacher
        try:
            llm_response = _llm.call(
                prompt=prompt,
                max_tokens=400,
                temperature=0.2,  # Low temperature for more consistent routing
                context=context
            )

            if not llm_response.get("ok"):
                error_msg = llm_response.get("error", "Unknown LLM error")
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "LLM_CALL_FAILED", "message": error_msg}
                }

            response_text = llm_response.get("text", "")

            # Extract routes and aliases
            routing_info = _extract_routing_from_response(response_text)

            # Log telemetry
            _store_telemetry("routing_taught", {
                "question": question[:100],
                "routes_count": len(routing_info.get("routes", [])),
                "aliases_count": len(routing_info.get("aliases", [])),
                "llm_source": llm_response.get("source", "unknown")
            })

            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            routing_hint = create_routing_hint(
                brain_name="teacher",
                action="teach_routing",
                confidence=0.8 if len(routing_info.get("routes", [])) > 0 else 0.3,
                context_tags=[
                    "routing_learning",
                    "memory_bank_routing",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )

            # Return the routing suggestion
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "routes": routing_info.get("routes", []),
                    "aliases": routing_info.get("aliases", []),
                    "raw_response": response_text,
                    "llm_source": llm_response.get("source", "unknown"),
                    "routing_hint": routing_hint  # Add routing hint
                }
            }

        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "TEACH_ROUTING_EXCEPTION", "message": str(e)}
            }

    # LEARN_FROM_EXECUTION operation: learn from Integrator coordination patterns
    if op == "LEARN_FROM_EXECUTION":
        integrator_response = payload.get("integrator_response") or {}
        context = payload.get("context") or {}

        if not integrator_response:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MISSING_RESPONSE", "message": "integrator_response is required"}
            }

        # Learn from the execution
        learning_result = learn_from_execution(integrator_response, context)

        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        routing_hint = create_routing_hint(
            brain_name="teacher",
            action="learn_from_execution",
            confidence=learning_result.get("success_score", 0.5),
            context_tags=[
                "pattern_learning",
                "learned" if learning_result.get("learned") else "not_learned",
                learning_result.get("reason", "unknown")
            ]
        )

        learning_result["routing_hint"] = routing_hint

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": learning_result
        }

    # GET_PATTERNS operation: retrieve learned patterns
    if op == "GET_PATTERNS":
        query_pattern = payload.get("query_pattern")

        patterns = get_learned_patterns(query_pattern)

        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        routing_hint = create_routing_hint(
            brain_name="teacher",
            action="get_patterns",
            confidence=0.9 if len(patterns) > 0 else 0.3,
            context_tags=[
                "pattern_retrieval",
                "patterns_found" if len(patterns) > 0 else "no_patterns",
                f"count_{len(patterns)}"
            ]
        )

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "patterns": patterns,
                "total_patterns": len(_continuation_patterns),
                "filtered_count": len(patterns),
                "routing_hint": routing_hint
            }
        }

    # Unsupported operation
    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"}
    }


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
