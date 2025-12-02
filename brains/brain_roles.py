"""
brain_roles.py
~~~~~~~~~~~~~~

Brain role configuration with EXPLICIT whitelist.

This module provides the canonical mapping of brains to their functional roles:
- Cognitive brains: Internal processing, skills, quality metrics
- Domain brains: World knowledge, user facts, preferences

IMPORTANT: Brain discovery is WHITELIST-BASED, not auto-discovery.
Only brains explicitly listed in CANONICAL_COGNITIVE_BRAINS are treated as
cognitive brains. Folders under brains/cognitive/ that are not in this list
are ignored (they may be helpers, data, or obsolete).

This prevents the brain count from exploding due to stray folders.

IMPORTANT: This configuration determines:
1. Where each brain's memory is stored (brains/ vs domain_banks/)
2. Which brains Teacher can write facts to (domain only)
3. Which brains Librarian routes over (domain only)

Usage:
    from brains.brain_roles import get_brain_role, get_domain_brains, get_cognitive_brains

    role = get_brain_role("reasoning")  # Returns "cognitive"
    domain_list = get_domain_brains()   # Returns list of all domain brains
"""

from __future__ import annotations
from typing import List, Literal, Set
from pathlib import Path

# Import scan function to get live data
from brains.tools.scan_brains import scan_all_brains


# =============================================================================
# CANONICAL BRAIN WHITELIST
# =============================================================================
# Only brains in this set are recognized as cognitive brains.
# To add a new cognitive brain, add it here explicitly.
# Folders under brains/cognitive/ NOT in this list are treated as helpers/data.

CANONICAL_COGNITIVE_BRAINS: Set[str] = {
    # Core processing
    "sensorium",           # Input normalization, message filtering
    "integrator",          # Routing, tool selection, orchestration
    "language",            # Language understanding and generation
    "reasoning",           # Logical reasoning, inference
    "reasoning_trace",     # Reasoning audit trails

    # Skills and capabilities
    "coder",               # Code generation, pattern-based coding
    "research_manager",    # Research tasks, fact verification
    "planner",             # Task planning and decomposition
    "imaginer",            # Creative generation, hypotheticals
    "abstraction",         # Concept abstraction

    # Learning and memory
    "teacher",             # Teaching, fact injection
    "learning",            # Meta-learning, skill acquisition
    "memory_librarian",    # Memory organization, retrieval
    "belief_tracker",      # Belief management
    "pattern_recognition", # Pattern detection

    # Self-awareness and introspection
    "self_model",          # Self-representation, introspection
    "self_dmn",            # Self-critique, default mode network
    "self_review",         # Output review
    "system_history",      # System event logging

    # Context and attention
    "attention",           # Focus management
    "context_management",  # Context window management
    "environment_context", # Environment awareness

    # Motivation and personality
    "affect_priority",     # Emotional/priority weighting
    "motivation",          # Goal motivation
    "personality",         # Personality traits
    "autonomy",            # Autonomous behavior

    # Collaboration
    "committee",           # Multi-brain deliberation
    "peer_connection",     # External peer communication

    # Action and execution
    "action_engine",       # Action planning and execution
    "thought_synthesis",   # Thought integration
    "external_interfaces", # External system connections

    # System management
    "inventory",           # Brain inventory and cataloging
}

# Domain banks are auto-discovered from brains/domain_banks/
# No whitelist needed as they are data stores, not processing brains.


# Role type
BrainRole = Literal["cognitive", "domain", "other"]


class BrainRoleConfig:
    """
    Brain role configuration using EXPLICIT whitelist.

    This class provides the authoritative mapping of brain names to roles.
    Only brains in CANONICAL_COGNITIVE_BRAINS are recognized as cognitive.
    """

    def __init__(self):
        """Initialize using canonical whitelist + domain bank scan."""
        self._inventory = scan_all_brains()

        # Use EXPLICIT whitelist for cognitive brains
        # Only brains in CANONICAL_COGNITIVE_BRAINS are cognitive
        self._cognitive_set = CANONICAL_COGNITIVE_BRAINS.copy()

        # Domain brains are auto-discovered (they're data stores, not code)
        self._domain_set = set()
        for brain in self._inventory["domain"]:
            if brain.startswith("_top_level/"):
                brain_name = brain.replace("_top_level/", "")
                self._domain_set.add(brain_name)
            else:
                self._domain_set.add(brain)

        # Store self/user brain for quick access
        self._self_brain = self._inventory.get("self_brain")

        # Log any folders under cognitive/ that are NOT in whitelist
        # These are helpers/data/obsolete, not brains
        scanned_cognitive = set(self._inventory.get("cognitive", []))
        unrecognized = scanned_cognitive - self._cognitive_set - {"__pycache__"}
        if unrecognized:
            # These folders exist but are not canonical brains
            # They are helpers, data dirs, or need cleanup
            self._unrecognized_folders = unrecognized
        else:
            self._unrecognized_folders = set()

    def get_role(self, brain_name: str) -> BrainRole:
        """
        Get the role of a brain.

        Args:
            brain_name: Name of the brain

        Returns:
            "cognitive", "domain", or "other"
        """
        if brain_name in self._cognitive_set:
            return "cognitive"
        elif brain_name in self._domain_set:
            return "domain"
        else:
            return "other"

    def is_cognitive(self, brain_name: str) -> bool:
        """Check if brain is cognitive."""
        return brain_name in self._cognitive_set

    def is_domain(self, brain_name: str) -> bool:
        """Check if brain is domain."""
        return brain_name in self._domain_set

    def get_cognitive_brains(self) -> List[str]:
        """Get list of all cognitive brain names."""
        return sorted(self._cognitive_set)

    def get_domain_brains(self) -> List[str]:
        """Get list of all domain brain names (including self/user brain)."""
        return sorted(self._domain_set)

    def get_self_brain(self) -> str | None:
        """Get the name of the self/user-facts brain."""
        return self._self_brain

    def get_domain_banks(self) -> List[str]:
        """
        Get list of domain bank names only (excluding top-level self brain).

        Returns:
            List of brain names under brains/domain_banks/
        """
        domain_banks = []
        for brain in self._inventory["domain"]:
            if not brain.startswith("_top_level/"):
                domain_banks.append(brain)
        return sorted(domain_banks)

    def is_top_level_brain(self, brain_name: str) -> bool:
        """
        Check if brain lives at top level (brains/<name>) vs in a subdirectory.

        Args:
            brain_name: Name of the brain

        Returns:
            True if brain is at top level
        """
        # Check if it's the self brain
        if brain_name == self._self_brain:
            return True

        # Check if it's one of the other top-level brains
        if brain_name in self._inventory.get("other", []):
            return True

        return False

    def get_unrecognized_folders(self) -> List[str]:
        """
        Get list of folders under cognitive/ that are NOT canonical brains.

        These may be:
        - Helper modules (not brains)
        - Data directories
        - Obsolete/stray folders to clean up

        Returns:
            List of folder names not in CANONICAL_COGNITIVE_BRAINS
        """
        return sorted(self._unrecognized_folders)


# Global singleton config
_config: BrainRoleConfig | None = None


def _get_config() -> BrainRoleConfig:
    """Get or create the global brain role config."""
    global _config
    if _config is None:
        _config = BrainRoleConfig()
    return _config


# Public API functions

def get_brain_role(brain_name: str) -> BrainRole:
    """
    Get the role of a brain.

    Args:
        brain_name: Name of the brain

    Returns:
        "cognitive", "domain", or "other"

    Example:
        >>> get_brain_role("reasoning")
        'cognitive'
        >>> get_brain_role("factual")
        'domain'
    """
    return _get_config().get_role(brain_name)


def is_cognitive_brain(brain_name: str) -> bool:
    """Check if brain is cognitive."""
    return _get_config().is_cognitive(brain_name)


def is_domain_brain(brain_name: str) -> bool:
    """Check if brain is domain."""
    return _get_config().is_domain(brain_name)


def get_cognitive_brains() -> List[str]:
    """Get list of all cognitive brain names."""
    return _get_config().get_cognitive_brains()


def get_domain_brains() -> List[str]:
    """
    Get list of all domain brain names.

    This includes:
    - All brains under brains/domain_banks/
    - The self/user-facts brain (e.g., "personal")

    This is the canonical list of brains that:
    - Teacher should write facts to
    - Librarian should route over

    Returns:
        Sorted list of domain brain names
    """
    return _get_config().get_domain_brains()


def get_domain_banks() -> List[str]:
    """
    Get list of domain bank names only (excluding self brain).

    Returns:
        List of brain names under brains/domain_banks/
    """
    return _get_config().get_domain_banks()


def get_self_brain() -> str | None:
    """Get the name of the self/user-facts brain."""
    return _get_config().get_self_brain()


def is_top_level_brain(brain_name: str) -> bool:
    """Check if brain lives at top level vs in a subdirectory."""
    return _get_config().is_top_level_brain(brain_name)


def get_unrecognized_folders() -> List[str]:
    """
    Get folders under cognitive/ that are NOT canonical brains.

    Use this for diagnostics to find stray folders that need cleanup.
    """
    return _get_config().get_unrecognized_folders()


def get_canonical_brain_list() -> Set[str]:
    """
    Get the canonical whitelist of cognitive brains.

    This is the authoritative list. Only brains in this set are
    recognized as cognitive brains by the system.
    """
    return CANONICAL_COGNITIVE_BRAINS.copy()


# Export public API
__all__ = [
    "BrainRole",
    "CANONICAL_COGNITIVE_BRAINS",
    "get_brain_role",
    "is_cognitive_brain",
    "is_domain_brain",
    "get_cognitive_brains",
    "get_domain_brains",
    "get_domain_banks",
    "get_self_brain",
    "is_top_level_brain",
    "get_unrecognized_folders",
    "get_canonical_brain_list",
]
