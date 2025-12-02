"""
teacher_contracts.py
~~~~~~~~~~~~~~~~~~~~

Per-brain Teacher contracts defining what each brain learns from the LLM.

This module defines the canonical mapping of brains to their Teacher usage:
- Which Teacher operation to use (TEACH or TEACH_ROUTING)
- Which prompt template to use
- Where to store learned content (own memory vs domain brains)
- Whether Teacher is enabled for this brain

IMPORTANT: This configuration determines:
1. What each brain learns from Teacher
2. Where learned content is stored
3. Which prompt template is used
4. Budget allocation per brain

Usage:
    from brains.teacher_contracts import get_contract, is_teacher_enabled

    contract = get_contract("planner")
    if contract:
        operation = contract["operation"]  # "TEACH" or "TEACH_ROUTING"
        prompt_mode = contract["prompt_mode"]
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Literal

# Teacher operation types
TeacherOperation = Literal["TEACH", "TEACH_ROUTING"]

# Teacher contract definition
TEACHER_CONTRACTS: Dict[str, Dict[str, Any]] = {
    # ============ CURRENT IMPLEMENTATIONS (Keep as-is) ============

    "reasoning": {
        "operation": "TEACH",
        "prompt_mode": "world_question",
        "store_internal": False,  # Q&A pairs handled separately
        "store_domain": True,     # Facts go to domain brains
        "enabled": True,
        "description": "Answers world/user questions, extracts facts"
    },

    "memory_librarian": {
        "operation": "TEACH_ROUTING",
        "prompt_mode": "routing_help",
        "store_internal": True,   # Routing rules go to own memory
        "store_domain": False,
        "enabled": True,
        "description": "Learns routing rules for memory bank selection"
    },

    "routing_classifier": {
        "operation": "CLASSIFY_ROUTING",
        "prompt_mode": "routing_classification",
        "store_internal": True,   # Store routing patterns for learning
        "store_domain": False,
        "enabled": True,
        "description": "Classifies user intent and suggests routing paths"
    },

    # ============ COGNITIVE BRAINS (Internal Skills & Patterns) ============

    "planner": {
        "operation": "TEACH",
        "prompt_mode": "planning_patterns",
        "store_internal": True,   # Planning templates/patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns task decomposition patterns and planning templates"
    },

    "autonomy": {
        "operation": "TEACH",
        "prompt_mode": "autonomy_strategies",
        "store_internal": True,   # Task prioritization patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns task prioritization and resource allocation strategies"
    },

    "coder": {
        "operation": "TEACH",
        "prompt_mode": "coding_patterns",
        "store_internal": True,   # Code patterns/templates
        "store_domain": False,
        "enabled": True,
        "description": "Learns code patterns, templates, and best practices"
    },

    "pattern_recognition": {
        "operation": "TEACH",
        "prompt_mode": "pattern_analysis",
        "store_internal": True,   # Pattern templates
        "store_domain": False,
        "enabled": True,
        "description": "Learns pattern analysis techniques and templates"
    },

    "language": {
        "operation": "TEACH",
        "prompt_mode": "style_meta",
        "store_internal": True,   # Phrasing styles
        "store_domain": False,
        "enabled": True,
        "description": "Learns language styles and phrasing patterns"
    },

    "imaginer": {
        "operation": "TEACH",
        "prompt_mode": "scenario_generation",
        "store_internal": True,   # Scenario templates
        "store_domain": False,
        "enabled": True,
        "description": "Learns scenario generation patterns and creative templates"
    },

    "self_model": {
        "operation": "TEACH",
        "prompt_mode": "self_definition",
        "store_internal": True,   # Self-identity patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns self-definition and identity patterns"
    },

    "belief_tracker": {
        "operation": "TEACH",
        "prompt_mode": "belief_patterns",
        "store_internal": True,   # Belief tracking patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns belief tracking and update patterns"
    },

    "motivation": {
        "operation": "TEACH",
        "prompt_mode": "goal_patterns",
        "store_internal": True,   # Goal patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns goal generation and motivation patterns"
    },

    "personality": {
        "operation": "TEACH",
        "prompt_mode": "personality_traits",
        "store_internal": True,   # Personality patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns personality trait patterns and behaviors"
    },

    "committee": {
        "operation": "TEACH",
        "prompt_mode": "decision_patterns",
        "store_internal": True,   # Decision patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns multi-perspective decision-making patterns"
    },

    "integrator": {
        "operation": "TEACH",
        "prompt_mode": "integration_patterns",
        "store_internal": True,   # Integration patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns cross-brain integration and coordination patterns"
    },

    "research_manager": {
        "operation": "TEACH",
        "prompt_mode": "world_question",
        "store_internal": True,
        "store_domain": True,
        "enabled": True,
        "description": "Learns how to run structured research tasks and capture facts"
    },

    "attention": {
        "operation": "TEACH",
        "prompt_mode": "attention_strategies",
        "store_internal": True,   # Attention strategies
        "store_domain": False,
        "enabled": True,
        "description": "Learns attention management and priority strategies"
    },

    "context_management": {
        "operation": "TEACH",
        "prompt_mode": "context_strategies",
        "store_internal": True,   # Context strategies
        "store_domain": False,
        "enabled": True,
        "description": "Learns context tracking and management strategies"
    },

    "learning": {
        "operation": "TEACH",
        "prompt_mode": "meta_learning",
        "store_internal": True,   # Meta-learning patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns meta-learning strategies and patterns"
    },

    "abstraction": {
        "operation": "TEACH",
        "prompt_mode": "abstraction_patterns",
        "store_internal": True,   # Abstraction patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns abstraction and generalization patterns"
    },

    "action_engine": {
        "operation": "TEACH",
        "prompt_mode": "action_patterns",
        "store_internal": True,   # Action patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns action execution and coordination patterns"
    },

    "self_review": {
        "operation": "TEACH",
        "prompt_mode": "review_criteria",
        "store_internal": True,   # Review criteria
        "store_domain": False,
        "enabled": True,
        "description": "Learns self-review criteria and quality standards"
    },

    "thought_synthesis": {
        "operation": "TEACH",
        "prompt_mode": "synthesis_patterns",
        "store_internal": True,   # Synthesis patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns thought synthesis and integration patterns"
    },

    "sensorium": {
        "operation": "TEACH",
        "prompt_mode": "sensory_integration",
        "store_internal": True,   # Sensory patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns sensory integration and processing patterns"
    },

    "affect_priority": {
        "operation": "TEACH",
        "prompt_mode": "emotional_patterns",
        "store_internal": True,   # Emotional patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns emotional response and priority patterns"
    },

    "environment_context": {
        "operation": "TEACH",
        "prompt_mode": "environment_patterns",
        "store_internal": True,   # Environment patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns environment awareness and adaptation patterns"
    },

    "external_interfaces": {
        "operation": "TEACH",
        "prompt_mode": "api_patterns",
        "store_internal": True,   # API patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns external API interaction patterns"
    },

    "peer_connection": {
        "operation": "TEACH",
        "prompt_mode": "peer_protocols",
        "store_internal": True,   # Peer protocols
        "store_domain": False,
        "enabled": True,
        "description": "Learns peer communication and collaboration protocols"
    },

    "reasoning_trace": {
        "operation": "TEACH",
        "prompt_mode": "trace_patterns",
        "store_internal": True,   # Trace patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns reasoning trace and explanation patterns"
    },

    "self_dmn": {
        "operation": "TEACH",
        "prompt_mode": "default_mode_patterns",
        "store_internal": True,   # Default mode patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns default mode network and idle-state patterns"
    },

    "system_history": {
        "operation": "TEACH",
        "prompt_mode": "history_patterns",
        "store_internal": True,   # History patterns
        "store_domain": False,
        "enabled": True,
        "description": "Learns history tracking and analysis patterns"
    },

    # Teacher itself doesn't need a contract (it IS the LLM interface)
    "teacher": {
        "operation": None,
        "prompt_mode": None,
        "store_internal": False,
        "store_domain": False,
        "enabled": False,
        "description": "Central LLM interface (doesn't learn from itself)"
    },

    # ============ DOMAIN BANKS (World Knowledge Bootstrap) ============
    # All domain banks use the same pattern: learn facts, store to own bank

    "arts": {
        "operation": "TEACH",
        "prompt_mode": "arts_fact",
        "store_internal": False,
        "store_domain": True,     # Facts go to arts bank
        "enabled": True,
        "description": "Bootstraps arts knowledge and facts"
    },

    "creative": {
        "operation": "TEACH",
        "prompt_mode": "creative_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps creative knowledge and techniques"
    },

    "economics": {
        "operation": "TEACH",
        "prompt_mode": "economics_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps economic knowledge and principles"
    },

    "factual": {
        "operation": "TEACH",
        "prompt_mode": "general_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps general factual knowledge"
    },

    "geography": {
        "operation": "TEACH",
        "prompt_mode": "geography_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps geographic knowledge and facts"
    },

    "history": {
        "operation": "TEACH",
        "prompt_mode": "history_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps historical knowledge and events"
    },

    "language_arts": {
        "operation": "TEACH",
        "prompt_mode": "language_arts_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps language arts knowledge and literature"
    },

    "law": {
        "operation": "TEACH",
        "prompt_mode": "law_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps legal knowledge and principles"
    },

    "math": {
        "operation": "TEACH",
        "prompt_mode": "math_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps mathematical knowledge and formulas"
    },

    "personal": {
        "operation": "TEACH",
        "prompt_mode": "personal_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps personal/user facts and preferences"
    },

    "philosophy": {
        "operation": "TEACH",
        "prompt_mode": "philosophy_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps philosophical knowledge and concepts"
    },

    "procedural": {
        "operation": "TEACH",
        "prompt_mode": "procedural_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps procedural knowledge (how-to)"
    },

    "science": {
        "operation": "TEACH",
        "prompt_mode": "science_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps scientific knowledge and facts"
    },

    "specs": {
        "operation": "TEACH",
        "prompt_mode": "specs_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps specification and standards knowledge"
    },

    "technology": {
        "operation": "TEACH",
        "prompt_mode": "technology_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps technology knowledge and facts"
    },

    "theories_and_contradictions": {
        "operation": "TEACH",
        "prompt_mode": "theory_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps theoretical and conflicting knowledge"
    },

    "working_theories": {
        "operation": "TEACH",
        "prompt_mode": "working_theory_fact",
        "store_internal": False,
        "store_domain": True,
        "enabled": True,
        "description": "Bootstraps unverified hypotheses and theories"
    },

    # stm_only is a special tier, not a knowledge domain
    "stm_only": {
        "operation": None,
        "prompt_mode": None,
        "store_internal": False,
        "store_domain": False,
        "enabled": False,
        "description": "Memory tier, not a learning brain"
    },
}


# Public API functions

def get_contract(brain_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the Teacher contract for a brain.

    Args:
        brain_id: Name of the brain

    Returns:
        Contract dict or None if not defined
    """
    return TEACHER_CONTRACTS.get(brain_id)


def is_teacher_enabled(brain_id: str) -> bool:
    """
    Check if Teacher is enabled for a brain.

    Args:
        brain_id: Name of the brain

    Returns:
        True if Teacher is enabled for this brain
    """
    contract = get_contract(brain_id)
    if not contract:
        return False
    return contract.get("enabled", False)


def get_prompt_mode(brain_id: str) -> Optional[str]:
    """
    Get the prompt mode for a brain.

    Args:
        brain_id: Name of the brain

    Returns:
        Prompt mode string or None
    """
    contract = get_contract(brain_id)
    if not contract:
        return None
    return contract.get("prompt_mode")


def get_operation(brain_id: str) -> Optional[TeacherOperation]:
    """
    Get the Teacher operation for a brain.

    Args:
        brain_id: Name of the brain

    Returns:
        "TEACH", "TEACH_ROUTING", or None
    """
    contract = get_contract(brain_id)
    if not contract:
        return None
    return contract.get("operation")


def should_store_internal(brain_id: str) -> bool:
    """
    Check if brain should store patterns in its own memory.

    Args:
        brain_id: Name of the brain

    Returns:
        True if patterns should be stored internally
    """
    contract = get_contract(brain_id)
    if not contract:
        return False
    return contract.get("store_internal", False)


def should_store_domain(brain_id: str) -> bool:
    """
    Check if brain should store facts to domain brains.

    Args:
        brain_id: Name of the brain

    Returns:
        True if facts should be stored to domain brains
    """
    contract = get_contract(brain_id)
    if not contract:
        return False
    return contract.get("store_domain", False)


def list_all_contracts() -> Dict[str, Dict[str, Any]]:
    """
    Get all Teacher contracts.

    Returns:
        Dict mapping brain_id to contract
    """
    return TEACHER_CONTRACTS.copy()


def list_enabled_brains() -> list[str]:
    """
    Get list of brain IDs with Teacher enabled.

    Returns:
        List of brain IDs
    """
    return [
        brain_id
        for brain_id, contract in TEACHER_CONTRACTS.items()
        if contract.get("enabled", False)
    ]


# Export public API
__all__ = [
    "TeacherOperation",
    "TEACHER_CONTRACTS",
    "get_contract",
    "is_teacher_enabled",
    "get_prompt_mode",
    "get_operation",
    "should_store_internal",
    "should_store_domain",
    "list_all_contracts",
    "list_enabled_brains",
]
