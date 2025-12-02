"""
prompt_templates.py
~~~~~~~~~~~~~~~~~~~

Specialized prompt templates for different brain types.

This module provides prompt templates for each brain's learning mode.
Each template defines how to format the LLM prompt and expected response structure.

IMPORTANT: These templates are used by TeacherHelper to construct
brain-specific prompts when calling the Teacher brain.

Usage:
    from brains.cognitive.teacher.service.prompt_templates import build_prompt

    prompt = build_prompt(
        mode="planning_patterns",
        question="How do I plan a web scraping task?",
        context={"user": {"name": "Alice"}}
    )
"""

from __future__ import annotations
from typing import Dict, Any, Optional


# Template definitions
PROMPT_TEMPLATES = {
    # ============ ROUTING CLASSIFICATION TEMPLATE ============

    "routing_classification": {
        "system": """You are Maven's routing classifier. Your task is to analyze user messages and determine the best routing path.

IMPORTANT RULES:
1. NEVER suggest capabilities that aren't in the provided capability snapshot
2. NEVER suggest brains or tools that aren't in the provided lists
3. For self/capability/history questions, ALWAYS recommend self_model brain
4. Be conservative - suggest fewer, more confident routes over many uncertain ones""",
        "instruction": """Analyze this user message and recommend routing.

USER MESSAGE: {user_text}

RECENT CONTEXT:
{conversation_context}

AVAILABLE CAPABILITIES:
{capability_snapshot}

AVAILABLE BRAINS: {available_brains}
AVAILABLE TOOLS: {available_tools}

Your task:
1. Classify the PRIMARY INTENT (one of: chat_answer, code_task, research_question, tool_request, capability_question, self_question, history_question, task_followup, meta_instruction)
2. Extract SECONDARY TAGS (e.g., python, filesystem, git, web)
3. Recommend which BRAINS should handle this (from available brains only)
4. Recommend which TOOLS might be needed (from available tools only)
5. Provide brief NOTES explaining your reasoning

Format your response EXACTLY as JSON:
```json
{
    "primary_intent": "<intent>",
    "secondary_tags": ["tag1", "tag2"],
    "recommended_brains": ["brain1", "brain2"],
    "recommended_tools": ["tool1", "tool2"],
    "confidence": 0.85,
    "notes": "Brief explanation"
}
```

CRITICAL: Only suggest brains/tools from the provided lists. Do not invent capabilities.""",
        "context_template": "",
        "expected_format": '{"primary_intent": "<intent>", "secondary_tags": [], "recommended_brains": [], "recommended_tools": [], "confidence": 0.0, "notes": ""}'
    },

    # ============ CURRENT TEMPLATES (Already in use) ============

    "world_question": {
        "system": "You are Maven's teacher. A question has been asked that Maven cannot answer from its own memory.",
        "instruction": """Your task:
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
- For personal questions, use facts about the user if mentioned""",
        "context_template": "User's name: {name}\n\n" if "{name}" else "",
        "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact1>\n- <fact2>"
    },

    "routing_help": {
        "system": "You are Maven's routing teacher. Your task is to help Maven learn which memory banks to query for different types of questions.",
        "instruction": """Your task:
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

Remember:
- science: biology, physics, chemistry, natural sciences
- history: historical events, dates, people from the past
- geography: locations, capitals, countries, landmarks
- math: numbers, calculations, equations
- personal: information about the user or Maven itself
- working_theories: unverified hypotheses, low-confidence facts
- theories_and_contradictions: conflicting information
- Use higher weights (0.8-1.0) for primary banks
- Use lower weights (0.2-0.5) for secondary banks""",
        "context_template": "",
        "expected_format": "ROUTES:\n- bank: weight\nALIASES:\n- alias1\n- alias2"
    },

    # ============ COGNITIVE BRAIN TEMPLATES (Internal Skills) ============

    "planning_patterns": {
        "system": "You are Maven's planning teacher. You help Maven learn task decomposition patterns and planning strategies.",
        "instruction": """Your task:
1. Provide a planning pattern or strategy for this type of task
2. Break down the approach into clear steps
3. Suggest when this pattern should be used

Format your response EXACTLY as follows:
PATTERN: <pattern name>
APPROACH: <brief description>
STEPS:
- <step 1>
- <step 2>
- <step 3>
WHEN_TO_USE: <condition or task type>
PATTERNS:
- <reusable pattern element 1>
- <reusable pattern element 2>

Remember:
- Focus on reusable patterns, not specific solutions
- Keep steps high-level and generalizable
- Identify common task types that fit this pattern""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nAPPROACH: <desc>\nSTEPS:\n- <step>\nWHEN_TO_USE: <condition>\nPATTERNS:\n- <pattern>"
    },

    "autonomy_strategies": {
        "system": "You are Maven's autonomy teacher. You help Maven learn task prioritization and resource allocation strategies.",
        "instruction": """Your task:
1. Provide a strategy for handling this autonomy challenge
2. Define prioritization rules
3. Suggest resource allocation approaches

Format your response EXACTLY as follows:
STRATEGY: <strategy name>
APPROACH: <brief description>
RULES:
- <rule 1>
- <rule 2>
- <rule 3>
PRIORITY: <high/medium/low>
PATTERNS:
- <reusable strategy pattern 1>
- <reusable strategy pattern 2>

Remember:
- Focus on decision-making frameworks
- Consider resource constraints (time, memory, LLM budget)
- Balance competing priorities""",
        "context_template": "",
        "expected_format": "STRATEGY: <name>\nAPPROACH: <desc>\nRULES:\n- <rule>\nPRIORITY: <level>\nPATTERNS:\n- <pattern>"
    },

    "coding_patterns": {
        "system": "You are Maven's coding teacher. You help Maven learn code patterns, templates, and best practices.",
        "instruction": """Your task:
1. Provide a code pattern or template for this programming task
2. Include example code structure
3. Explain when to use this pattern

Format your response EXACTLY as follows:
PATTERN: <pattern name>
DESCRIPTION: <what this pattern does>
CODE_TEMPLATE:
```
<code template with placeholders>
```
USE_CASE: <when to use this pattern>
PATTERNS:
- <key principle 1>
- <key principle 2>

Remember:
- Focus on patterns, not complete implementations
- Use placeholders for variable parts
- Include error handling considerations
- Keep code clean and readable""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nDESCRIPTION: <desc>\nCODE_TEMPLATE:\n```\n<code>\n```\nUSE_CASE: <use>\nPATTERNS:\n- <pattern>"
    },

    "pattern_analysis": {
        "system": "You are Maven's pattern recognition teacher. You help Maven learn pattern analysis techniques.",
        "instruction": """Your task:
1. Identify the pattern type in this scenario
2. Provide analysis technique for detecting this pattern
3. Suggest similar patterns to watch for

Format your response EXACTLY as follows:
PATTERN_TYPE: <type of pattern>
ANALYSIS: <how to detect this pattern>
TECHNIQUE:
- <detection step 1>
- <detection step 2>
SIMILAR_PATTERNS:
- <related pattern 1>
- <related pattern 2>
PATTERNS:
- <reusable analysis template 1>
- <reusable analysis template 2>

Remember:
- Focus on detection methods
- Identify pattern signatures
- Consider edge cases and variations""",
        "context_template": "",
        "expected_format": "PATTERN_TYPE: <type>\nANALYSIS: <desc>\nTECHNIQUE:\n- <step>\nSIMILAR_PATTERNS:\n- <pattern>\nPATTERNS:\n- <template>"
    },

    "style_meta": {
        "system": "You are Maven's language teacher. You help Maven learn language styles and phrasing patterns.",
        "instruction": """Your task:
1. Provide a language style or phrasing pattern
2. Give examples of this style in use
3. Explain when this style is appropriate

Format your response EXACTLY as follows:
STYLE: <style name>
DESCRIPTION: <what this style conveys>
EXAMPLES:
- <example 1>
- <example 2>
- <example 3>
CONTEXT: <when to use this style>
PATTERNS:
- <phrasing pattern 1>
- <phrasing pattern 2>

Remember:
- Consider tone, formality, and context
- Provide diverse examples
- Note situations where this style is inappropriate""",
        "context_template": "",
        "expected_format": "STYLE: <name>\nDESCRIPTION: <desc>\nEXAMPLES:\n- <ex>\nCONTEXT: <when>\nPATTERNS:\n- <pattern>"
    },

    "scenario_generation": {
        "system": "You are Maven's imagination teacher. You help Maven learn scenario generation and creative exploration.",
        "instruction": """Your task:
1. Provide a scenario generation approach for this topic
2. Outline key elements to consider
3. Suggest variations and alternatives

Format your response EXACTLY as follows:
SCENARIO_TYPE: <type of scenario>
APPROACH: <generation strategy>
KEY_ELEMENTS:
- <element 1>
- <element 2>
- <element 3>
VARIATIONS:
- <variation 1>
- <variation 2>
PATTERNS:
- <generation pattern 1>
- <generation pattern 2>

Remember:
- Focus on creative exploration
- Consider multiple perspectives
- Balance realism with imagination""",
        "context_template": "",
        "expected_format": "SCENARIO_TYPE: <type>\nAPPROACH: <strategy>\nKEY_ELEMENTS:\n- <elem>\nVARIATIONS:\n- <var>\nPATTERNS:\n- <pattern>"
    },

    "self_definition": {
        "system": "You are Maven's self-awareness teacher. You help Maven understand and define its own identity and capabilities.",
        "instruction": """Your task:
1. Provide guidance on how Maven should understand this aspect of itself
2. Define key characteristics or capabilities
3. Suggest appropriate self-descriptions

Format your response EXACTLY as follows:
ASPECT: <aspect of self-identity>
UNDERSTANDING: <how Maven should understand this>
CHARACTERISTICS:
- <characteristic 1>
- <characteristic 2>
- <characteristic 3>
SELF_DESCRIPTION: <appropriate way to describe this>
PATTERNS:
- <identity pattern 1>
- <identity pattern 2>

Remember:
- Focus on accurate self-representation
- Acknowledge both capabilities and limitations
- Maintain consistency with Maven's core identity""",
        "context_template": "",
        "expected_format": "ASPECT: <aspect>\nUNDERSTANDING: <desc>\nCHARACTERISTICS:\n- <char>\nSELF_DESCRIPTION: <desc>\nPATTERNS:\n- <pattern>"
    },

    # ============ ADDITIONAL COGNITIVE TEMPLATES ============

    "belief_patterns": {
        "system": "You are Maven's belief tracking teacher. You help Maven learn how to track and update beliefs.",
        "instruction": """Provide a belief tracking pattern including:
- Pattern name and approach
- Update rules
- Confidence handling
- Patterns as reusable elements

Format: PATTERN: <name>\nAPPROACH: <desc>\nRULES:\n- <rule>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nAPPROACH: <desc>\nRULES:\n- <rule>\nPATTERNS:\n- <pattern>"
    },

    "goal_patterns": {
        "system": "You are Maven's motivation teacher. You help Maven learn goal generation and motivation patterns.",
        "instruction": """Provide a goal/motivation pattern including:
- Pattern name
- Goal formation approach
- Priority criteria
- Patterns as reusable elements

Format: PATTERN: <name>\nAPPROACH: <desc>\nCRITERIA:\n- <criterion>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nAPPROACH: <desc>\nCRITERIA:\n- <criterion>\nPATTERNS:\n- <pattern>"
    },

    "personality_traits": {
        "system": "You are Maven's personality teacher. You help Maven learn personality trait patterns.",
        "instruction": """Provide a personality pattern including:
- Trait name
- Behavioral expressions
- Contextual modulation
- Patterns as reusable elements

Format: TRAIT: <name>\nEXPRESSIONS:\n- <expr>\nCONTEXT:\n- <when>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "TRAIT: <name>\nEXPRESSIONS:\n- <expr>\nCONTEXT:\n- <when>\nPATTERNS:\n- <pattern>"
    },

    "decision_patterns": {
        "system": "You are Maven's committee teacher. You help Maven learn multi-perspective decision patterns.",
        "instruction": """Provide a decision pattern including:
- Pattern name
- Perspectives to consider
- Integration approach
- Patterns as reusable elements

Format: PATTERN: <name>\nPERSPECTIVES:\n- <persp>\nINTEGRATION:\n- <approach>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nPERSPECTIVES:\n- <persp>\nINTEGRATION:\n- <approach>\nPATTERNS:\n- <pattern>"
    },

    "integration_patterns": {
        "system": "You are Maven's integration teacher. You help Maven learn cross-brain coordination patterns.",
        "instruction": """Provide an integration pattern including:
- Pattern name
- Coordination approach
- Data flow
- Patterns as reusable elements

Format: PATTERN: <name>\nCOORDINATION:\n- <step>\nDATA_FLOW:\n- <flow>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nCOORDINATION:\n- <step>\nDATA_FLOW:\n- <flow>\nPATTERNS:\n- <pattern>"
    },

    "attention_strategies": {
        "system": "You are Maven's attention teacher. You help Maven learn attention management strategies.",
        "instruction": """Provide an attention strategy including:
- Strategy name
- Priority rules
- Resource allocation
- Patterns as reusable elements

Format: STRATEGY: <name>\nRULES:\n- <rule>\nRESOURCES:\n- <alloc>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "STRATEGY: <name>\nRULES:\n- <rule>\nRESOURCES:\n- <alloc>\nPATTERNS:\n- <pattern>"
    },

    "context_strategies": {
        "system": "You are Maven's context management teacher. You help Maven learn context tracking strategies.",
        "instruction": """Provide a context strategy including:
- Strategy name
- Tracking approach
- Context transitions
- Patterns as reusable elements

Format: STRATEGY: <name>\nTRACKING:\n- <method>\nTRANSITIONS:\n- <rule>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "STRATEGY: <name>\nTRACKING:\n- <method>\nTRANSITIONS:\n- <rule>\nPATTERNS:\n- <pattern>"
    },

    "meta_learning": {
        "system": "You are Maven's meta-learning teacher. You help Maven learn how to learn.",
        "instruction": """Provide a meta-learning pattern including:
- Pattern name
- Learning approach
- Improvement criteria
- Patterns as reusable elements

Format: PATTERN: <name>\nAPPROACH:\n- <step>\nCRITERIA:\n- <criterion>\nPATTERNS:\n- <pattern>""",
        "context_template": "",
        "expected_format": "PATTERN: <name>\nAPPROACH:\n- <step>\nCRITERIA:\n- <criterion>\nPATTERNS:\n- <pattern>"
    },

    # Add more cognitive templates as needed...
    "abstraction_patterns": {"system": "Maven's abstraction teacher", "instruction": "Provide abstraction pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "action_patterns": {"system": "Maven's action teacher", "instruction": "Provide action pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "review_criteria": {"system": "Maven's review teacher", "instruction": "Provide review criteria", "context_template": "", "expected_format": "CRITERIA: <name>\nPATTERNS:\n- <pattern>"},
    "synthesis_patterns": {"system": "Maven's synthesis teacher", "instruction": "Provide synthesis pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "sensory_integration": {"system": "Maven's sensory teacher", "instruction": "Provide sensory integration pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "emotional_patterns": {"system": "Maven's emotional teacher", "instruction": "Provide emotional pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "environment_patterns": {"system": "Maven's environment teacher", "instruction": "Provide environment pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "api_patterns": {"system": "Maven's API teacher", "instruction": "Provide API pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "peer_protocols": {"system": "Maven's peer teacher", "instruction": "Provide peer protocol", "context_template": "", "expected_format": "PROTOCOL: <name>\nPATTERNS:\n- <pattern>"},
    "trace_patterns": {"system": "Maven's trace teacher", "instruction": "Provide trace pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "default_mode_patterns": {"system": "Maven's DMN teacher", "instruction": "Provide default mode pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},
    "history_patterns": {"system": "Maven's history teacher", "instruction": "Provide history pattern", "context_template": "", "expected_format": "PATTERN: <name>\nPATTERNS:\n- <pattern>"},

    # ============ DOMAIN BANK TEMPLATES (Fact Bootstrap) ============
    # All domain templates use the same structure as world_question

    "science_fact": {
        "system": "You are Maven's science teacher. Provide scientific facts and explanations.",
        "instruction": """Provide scientific facts:
Format: ANSWER: <answer>\nFACTS:\n- <fact1>\n- <fact2>
Focus on verified scientific knowledge.""",
        "context_template": "",
        "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"
    },

    "history_fact": {
        "system": "You are Maven's history teacher. Provide historical facts and context.",
        "instruction": """Provide historical facts:
Format: ANSWER: <answer>\nFACTS:\n- <fact1>\n- <fact2>
Focus on verified historical events and dates.""",
        "context_template": "",
        "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"
    },

    "geography_fact": {
        "system": "You are Maven's geography teacher. Provide geographic facts and information.",
        "instruction": """Provide geographic facts:
Format: ANSWER: <answer>\nFACTS:\n- <fact1>\n- <fact2>
Focus on locations, regions, and geographic features.""",
        "context_template": "",
        "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"
    },

    "math_fact": {
        "system": "You are Maven's math teacher. Provide mathematical facts and formulas.",
        "instruction": """Provide mathematical facts:
Format: ANSWER: <answer>\nFACTS:\n- <fact1>\n- <fact2>
Focus on formulas, theorems, and mathematical principles.""",
        "context_template": "",
        "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"
    },

    # Simplified domain templates (all follow same pattern)
    "arts_fact": {"system": "Maven's arts teacher", "instruction": "Provide arts facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "creative_fact": {"system": "Maven's creative teacher", "instruction": "Provide creative knowledge", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "economics_fact": {"system": "Maven's economics teacher", "instruction": "Provide economic facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "general_fact": {"system": "Maven's general knowledge teacher", "instruction": "Provide general facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "language_arts_fact": {"system": "Maven's language arts teacher", "instruction": "Provide language arts facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "law_fact": {"system": "Maven's law teacher", "instruction": "Provide legal facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "personal_fact": {"system": "Maven's personal teacher", "instruction": "Provide personal/user facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "philosophy_fact": {"system": "Maven's philosophy teacher", "instruction": "Provide philosophical facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "procedural_fact": {"system": "Maven's procedural teacher", "instruction": "Provide procedural knowledge", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "specs_fact": {"system": "Maven's specs teacher", "instruction": "Provide specification facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "technology_fact": {"system": "Maven's technology teacher", "instruction": "Provide technology facts", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "theory_fact": {"system": "Maven's theory teacher", "instruction": "Provide theoretical knowledge", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
    "working_theory_fact": {"system": "Maven's working theory teacher", "instruction": "Provide working theories", "context_template": "", "expected_format": "ANSWER: <answer>\nFACTS:\n- <fact>"},
}


def build_prompt(
    mode: str,
    question: str,
    context: Optional[Dict[str, Any]] = None,
    retrieved_facts: Optional[list] = None
) -> str:
    """
    Build a prompt for the Teacher using the specified template.

    Args:
        mode: The prompt mode (from teacher_contracts)
        question: The question or task
        context: Optional context dict (user info, etc.)
        retrieved_facts: Optional list of facts from memory

    Returns:
        Formatted prompt string ready for LLM
    """
    template = PROMPT_TEMPLATES.get(mode)
    if not template:
        # Fallback to world_question template
        template = PROMPT_TEMPLATES["world_question"]

    # Build prompt components
    system_msg = template["system"]
    instruction = template["instruction"]
    context_template = template.get("context_template", "")

    # Start with system message
    prompt = system_msg + "\n\n"

    # Add question
    prompt += f"QUESTION: {question}\n\n"

    # Add user context if available
    if context and context_template:
        try:
            user = context.get("user", {})
            name = user.get("name")
            if name:
                prompt += context_template.format(name=name)
        except Exception:
            pass

    # Add retrieved facts if available (for world_question mode)
    if retrieved_facts and mode in ["world_question", "science_fact", "history_fact", "geography_fact", "math_fact"]:
        if len(retrieved_facts) > 0:
            prompt += "Potentially relevant context from Maven's memory (low confidence):\n"
            for fact in retrieved_facts[:3]:  # Limit to top 3
                try:
                    if isinstance(fact, dict):
                        content = str(fact.get("content", "")).strip()
                        if content:
                            prompt += f"- {content}\n"
                except Exception:
                    pass
            prompt += "\n"

    # Add instruction
    prompt += instruction

    return prompt


def get_expected_format(mode: str) -> str:
    """
    Get the expected response format for a prompt mode.

    Args:
        mode: The prompt mode

    Returns:
        Expected format string
    """
    template = PROMPT_TEMPLATES.get(mode)
    if not template:
        return "ANSWER: <answer>\nFACTS:\n- <fact>"
    return template.get("expected_format", "")


# Export public API
__all__ = [
    "PROMPT_TEMPLATES",
    "build_prompt",
    "get_expected_format",
]
