"""
Semantic Normalizer / Synonyms Engine

Purpose
-------
Provide a robust, human-like normalization layer for all text going into
routing, pattern learning, and tool intent:

- Lowercasing, whitespace cleanup, basic punctuation stripping.
- Multi-word phrase canonicalization (e.g. "scan self" → "self_scan").
- Token-level synonym mapping (e.g. "fs", "file system" → "filesystem").
- Very simple rule-based lemmatization (run/running/ran → run where safe).
- Domain-aware normalization for Maven (tools, brains, self-intent, etc.).
- Configurable via synonyms.json in project or ~/.maven/synonyms.json.
- Safe: invalid config is handled gracefully, defaults still work.
- Transparent: returns structured NormalizationResult including applied rules.

No stubs. If config is missing or broken, normalizer still works with built-ins.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
DEFAULT_SYNONYMS_PATH = MAVEN_DIR / "synonyms.json"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PhraseRule:
    """Multi-word phrase normalization rule."""
    canonical: str
    variants: List[str]
    # internal, built from variants
    _compiled: List[re.Pattern] = field(default_factory=list, repr=False)


@dataclass
class TokenRule:
    """Single-token synonym rule."""
    canonical: str
    variants: List[str]


@dataclass
class NormalizationConfig:
    """
    Configuration for the SemanticNormalizer.

    - phrase_rules: for multi-word expressions.
    - token_rules: for single tokens.
    """
    phrase_rules: List[PhraseRule] = field(default_factory=list)
    token_rules: List[TokenRule] = field(default_factory=list)


@dataclass
class NormalizationResult:
    """
    Result of a normalization pass.

    - original: original text.
    - normalized: final normalized string.
    - tokens: list of final tokens.
    - applied_phrase_rules: canonical names of phrase rules that fired.
    - applied_token_rules: canonical names of token rules that fired.
    - intent_kind: classified intent type (e.g., "system_capability", "self_identity")
    """
    original: str
    normalized: str
    tokens: List[str]
    applied_phrase_rules: List[str] = field(default_factory=list)
    applied_token_rules: List[str] = field(default_factory=list)
    intent_kind: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# =============================================================================
# Main normalizer class
# =============================================================================

class SemanticNormalizer:
    """
    Central normalization engine.

    Usage:
        normalizer = SemanticNormalizer(project_root="/path/to/maven2_fix")
        res = normalizer.normalize_for_routing(user_text)

    It will:
    - Load config from project_root/config/synonyms.json or ~/.maven/synonyms.json.
    - Fall back to built-in defaults if files are missing/invalid.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None
        self.config = NormalizationConfig()
        # lookup tables
        self._variant_to_canonical_token: Dict[str, str] = {}
        self._phrase_rules: List[PhraseRule] = []

        self._load_config()
        self._build_lookup_tables()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def normalize(
        self,
        text: str,
        *,
        for_routing: bool = False,
        for_memory: bool = False,
        lowercase: bool = True,
        strip_punctuation: bool = True,
        lemmatize: bool = True,
    ) -> NormalizationResult:
        """
        Core normalization.

        Options:
        - for_routing: tune behavior for routing intent (aggressive canonicalization).
        - for_memory: tune behavior for memory storage (slightly less aggressive).
        - lowercase: always recommended.
        - strip_punctuation: remove punctuation except underscores.
        - lemmatize: apply simple stem/lemma rules.

        Returns NormalizationResult with full detail.
        """
        original = text
        work = text

        if lowercase:
            work = work.lower()

        work = self._collapse_whitespace(work)

        applied_phrase_rules: List[str] = []
        work = self._apply_phrase_rules(work, applied_phrase_rules)

        # Optionally strip punctuation (preserving underscores used for canonicals)
        if strip_punctuation:
            work = self._strip_punctuation(work)

        # Tokenize on whitespace after phrase replacement
        raw_tokens = work.split()
        tokens, applied_token_rules = self._normalize_tokens(
            raw_tokens,
            lemmatize=lemmatize,
            aggressive=for_routing,
        )

        normalized = " ".join(tokens)

        # Classify intent using original text (before normalization)
        intent_kind = self._classify_intent(original)

        return NormalizationResult(
            original=original,
            normalized=normalized,
            tokens=tokens,
            applied_phrase_rules=applied_phrase_rules,
            applied_token_rules=applied_token_rules,
            intent_kind=intent_kind,
        )

    def normalize_for_routing(self, text: str) -> NormalizationResult:
        """
        Opinionated preset for routing:
        - lowercase
        - punctuation stripped (except underscores)
        - lemmatization
        - aggressive synonym canonicalization
        """
        return self.normalize(
            text,
            for_routing=True,
            for_memory=False,
            lowercase=True,
            strip_punctuation=True,
            lemmatize=True,
        )

    def normalize_for_memory(self, text: str) -> NormalizationResult:
        """
        Opinionated preset for memory:
        - lowercase
        - punctuation stripped (except underscores)
        - light lemmatization
        - synonym mapping, but slightly less aggressive than routing
        (implemented as same pipeline but you can extend for domain flags).
        """
        return self.normalize(
            text,
            for_routing=False,
            for_memory=True,
            lowercase=True,
            strip_punctuation=True,
            lemmatize=True,
        )

    def normalize_for_pattern(self, text: str) -> NormalizationResult:
        """
        Opinionated preset for pattern learning:
        - lowercase
        - punctuation stripped
        - moderate lemmatization
        - synonym mapping
        """
        return self.normalize(
            text,
            for_routing=False,
            for_memory=False,
            lowercase=True,
            strip_punctuation=True,
            lemmatize=True,
        )

    def get_canonical_for_token(self, token: str) -> Optional[str]:
        """
        Look up the canonical form for a single token.
        Returns None if no mapping exists.
        """
        norm_token = self._normalize_space(token.lower())
        return self._variant_to_canonical_token.get(norm_token)

    def get_config_summary(self) -> Dict[str, Any]:
        """Return a summary of the current configuration."""
        return {
            "phrase_rules_count": len(self.config.phrase_rules),
            "token_rules_count": len(self.config.token_rules),
            "phrase_canonicals": [r.canonical for r in self.config.phrase_rules],
            "token_canonicals": [r.canonical for r in self.config.token_rules],
        }

    # -------------------------------------------------------------------------
    # Config loading
    # -------------------------------------------------------------------------

    def _load_config(self) -> None:
        """
        Load config from:
        1) project_root/config/synonyms.json if present
        2) ~/.maven/synonyms.json
        3) fall back to built-in defaults

        Invalid JSON is logged and skipped; defaults are still applied.
        """
        loaded = False
        merged_config: Dict[str, Any] = {}

        # 1) project config
        if self.project_root is not None:
            project_cfg = self.project_root / "config" / "synonyms.json"
            if project_cfg.exists():
                try:
                    with project_cfg.open("r", encoding="utf-8") as f:
                        merged_config = json.load(f)
                        loaded = True
                        logger.info("Loaded synonyms config from %s", project_cfg)
                except Exception as e:
                    logger.error("Failed to load project synonyms.json (%s): %s", project_cfg, e)

        # 2) user config (~/.maven/synonyms.json) overrides / extends
        if DEFAULT_SYNONYMS_PATH.exists():
            try:
                with DEFAULT_SYNONYMS_PATH.open("r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                if loaded:
                    merged_config = self._deep_merge_dicts(merged_config, user_cfg)
                else:
                    merged_config = user_cfg
                    loaded = True
                logger.info("Loaded synonyms config from %s", DEFAULT_SYNONYMS_PATH)
            except Exception as e:
                logger.error("Failed to load ~/.maven/synonyms.json: %s", e)

        if not loaded:
            logger.warning("No valid synonyms.json found; using built-in defaults")
            merged_config = self._default_config_dict()

        self.config = self._parse_config_dict(merged_config)

    @staticmethod
    def _deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Simple recursive dict merge: values in b override a."""
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = SemanticNormalizer._deep_merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _default_config_dict() -> Dict[str, Any]:
        """
        Built-in defaults for Maven.

        These cover:
        - self-intent
        - tools (fs/git/browser)
        - research
        - memory
        """
        return {
            "phrases": [
                {
                    "canonical": "self_scan",
                    "variants": [
                        "scan self",
                        "self system scan",
                        "self-system scan",
                        "run self scan",
                        "system scan",
                    ],
                },
                {
                    "canonical": "scan_memory",
                    "variants": [
                        "scan memory",
                        "memory scan",
                        "check memory health",
                        "memory health scan",
                    ],
                },
                {
                    "canonical": "browser_search",
                    "variants": [
                        "use the browser",
                        "search the web",
                        "look up online",
                        "web search",
                        "search for",
                        "look up",
                        "find online",
                        "google it",
                        "google search",
                        "bing search",
                        "ddg search",
                        "duckduckgo search",
                    ],
                },
                {
                    "canonical": "git_status",
                    "variants": [
                        "git status",
                        "check git status",
                        "scan git status",
                    ],
                },
                {
                    "canonical": "git_diff",
                    "variants": [
                        "git diff",
                        "show git diff",
                        "check git diff",
                    ],
                },
                {
                    "canonical": "filesystem_action",
                    "variants": [
                        "file system action",
                        "filesystem operation",
                        "fs operation",
                    ],
                },
                {
                    "canonical": "create_directory",
                    "variants": [
                        "create directory",
                        "make directory",
                        "mkdir",
                        "create folder",
                        "make folder",
                    ],
                },
                {
                    "canonical": "read_file",
                    "variants": [
                        "read file",
                        "show file",
                        "display file",
                        "cat file",
                    ],
                },
                {
                    "canonical": "write_file",
                    "variants": [
                        "write file",
                        "save file",
                        "create file",
                    ],
                },
                {
                    "canonical": "enable_execution",
                    "variants": [
                        "enable execution",
                        "turn on execution",
                        "allow execution",
                        "enable exec",
                    ],
                },
                {
                    "canonical": "disable_execution",
                    "variants": [
                        "disable execution",
                        "turn off execution",
                        "stop execution",
                        "disable exec",
                    ],
                },
            ],
            "tokens": [
                {
                    "canonical": "filesystem",
                    "variants": ["fs", "file system", "file-system"],
                },
                {
                    "canonical": "browser",
                    "variants": ["web", "chrome", "firefox", "edge", "safari"],
                },
                {
                    "canonical": "search",
                    "variants": ["lookup", "look-up", "look up", "find", "query", "seek"],
                },
                {
                    "canonical": "google",
                    "variants": ["goog", "google.com"],
                },
                {
                    "canonical": "bing",
                    "variants": ["bing.com", "microsoft search"],
                },
                {
                    "canonical": "duckduckgo",
                    "variants": ["ddg", "duck duck go", "duckduckgo.com"],
                },
                {
                    "canonical": "research",
                    "variants": ["lookup", "look-up", "investigate"],
                },
                {
                    "canonical": "teacher",
                    "variants": ["oracle", "expert"],
                },
                {
                    "canonical": "self_model",
                    "variants": ["self model", "self brain"],
                },
                {
                    "canonical": "git",
                    "variants": ["repository", "repo", "git repo"],
                },
                {
                    "canonical": "tool",
                    "variants": ["agent", "action"],
                },
                {
                    "canonical": "directory",
                    "variants": ["folder", "dir"],
                },
                {
                    "canonical": "file",
                    "variants": ["document", "doc"],
                },
                {
                    "canonical": "execute",
                    "variants": ["run", "exec", "invoke"],
                },
                {
                    "canonical": "memory",
                    "variants": ["recall", "remember", "storage"],
                },
                {
                    "canonical": "verify",
                    "variants": ["check", "validate", "confirm"],
                },
            ],
        }

    @staticmethod
    def _parse_config_dict(cfg: Dict[str, Any]) -> NormalizationConfig:
        """Turn raw dict into NormalizationConfig."""
        phrase_rules: List[PhraseRule] = []
        token_rules: List[TokenRule] = []

        for item in cfg.get("phrases", []):
            try:
                canonical = str(item["canonical"]).strip()
                variants = [str(v).strip() for v in item.get("variants", []) if str(v).strip()]
                if not canonical or not variants:
                    continue
                phrase_rules.append(PhraseRule(canonical=canonical, variants=variants))
            except Exception:
                continue

        for item in cfg.get("tokens", []):
            try:
                canonical = str(item["canonical"]).strip()
                variants = [str(v).strip() for v in item.get("variants", []) if str(v).strip()]
                if not canonical or not variants:
                    continue
                token_rules.append(TokenRule(canonical=canonical, variants=variants))
            except Exception:
                continue

        return NormalizationConfig(
            phrase_rules=phrase_rules,
            token_rules=token_rules,
        )

    # -------------------------------------------------------------------------
    # Lookup table preparation
    # -------------------------------------------------------------------------

    def _build_lookup_tables(self) -> None:
        """
        Build:
        - variant_to_canonical_token: mapping for tokens.
        - compiled phrase regexes (longest variants first) for phrase rules.
        """
        self._variant_to_canonical_token.clear()
        self._phrase_rules = []

        # Token lookup (normalize variants to lowercase for matching)
        for rule in self.config.token_rules:
            for variant in rule.variants:
                norm_variant = self._normalize_space(variant.lower())
                self._variant_to_canonical_token[norm_variant] = rule.canonical.lower()

        # Phrase rules: compile patterns
        phrase_rules_sorted: List[PhraseRule] = sorted(
            self.config.phrase_rules,
            key=lambda r: max(len(v) for v in r.variants),
            reverse=True,
        )

        for rule in phrase_rules_sorted:
            compiled_variants: List[re.Pattern] = []
            for v in rule.variants:
                # Escape for regex, then enforce word boundaries
                pattern_str = re.escape(self._normalize_space(v.lower()))
                pattern = re.compile(r"\b" + pattern_str + r"\b", re.IGNORECASE)
                compiled_variants.append(pattern)

            rule._compiled = compiled_variants
            self._phrase_rules.append(rule)

    # -------------------------------------------------------------------------
    # Phrase handling
    # -------------------------------------------------------------------------

    def _apply_phrase_rules(self, text: str, applied_rules_out: List[str]) -> str:
        """
        Replace matched phrases with canonical tokens (often with underscores).

        Example:
            "scan self and memory" → "self_scan and memory"
        """
        result = text
        for rule in self._phrase_rules:
            for pattern in rule._compiled:
                if pattern.search(result):
                    if rule.canonical not in applied_rules_out:
                        applied_rules_out.append(rule.canonical)
                    result = pattern.sub(rule.canonical, result)
        return result

    # -------------------------------------------------------------------------
    # Token-level normalization
    # -------------------------------------------------------------------------

    def _normalize_tokens(
        self,
        tokens: List[str],
        *,
        lemmatize: bool,
        aggressive: bool,
    ) -> Tuple[List[str], List[str]]:
        """
        Normalize each token:
        - collapse spaces
        - apply synonyms
        - apply simple lemma rules

        Returns:
        - final_tokens
        - applied_token_rules (list of canonicals)
        """
        final_tokens: List[str] = []
        applied_rules: List[str] = []

        for tok in tokens:
            norm_tok = self._normalize_space(tok.lower())

            # token synonym mapping
            canonical = self._variant_to_canonical_token.get(norm_tok)
            if canonical:
                final_tok = canonical
                if canonical not in applied_rules:
                    applied_rules.append(canonical)
            else:
                final_tok = norm_tok

            if lemmatize:
                final_tok = self._lemmatize(final_tok, aggressive=aggressive)

            if final_tok:
                final_tokens.append(final_tok)

        return final_tokens, applied_rules

    # -------------------------------------------------------------------------
    # Lemmatizer (simple rule-based)
    # -------------------------------------------------------------------------

    @staticmethod
    def _lemmatize(token: str, aggressive: bool) -> str:
        """
        Very basic, rule-based lemmatization.

        We do not pull in external NLP libs; this is deterministic.

        Strategy:
        - handle obvious English plurals (tools → tool, agents → agent).
        - handle "-ing" / "-ed" where safe.

        aggressive=False → more conservative (memory).
        aggressive=True → more willing to strip suffixes (routing).
        """
        if len(token) <= 3:
            return token

        # Preserve certain words that shouldn't be lemmatized
        preserve_words = {
            "status", "this", "that", "was", "has", "does", "is", "as",
            "class", "pass", "process", "access", "address", "compress",
        }
        if token in preserve_words:
            return token

        # Plurals: agents -> agent, tools -> tool
        if token.endswith("s") and len(token) > 3:
            # avoid stripping 'ss' (e.g. "class", "pass")
            if not token.endswith("ss") and not token.endswith("us"):
                candidate = token[:-1]
                if aggressive or len(candidate) > 3:
                    token = candidate

        # -ing
        if token.endswith("ing") and len(token) > 5:
            base = token[:-3]
            # running -> run (simplified)
            if len(base) > 1 and base[-1] == base[-2] and len(base) > 3:
                base = base[:-1]
            if aggressive or len(base) > 3:
                token = base

        # -ed
        if token.endswith("ed") and len(token) > 4:
            base = token[:-2]
            if aggressive or len(base) > 3:
                token = base

        return token

    # -------------------------------------------------------------------------
    # Intent Classification
    # -------------------------------------------------------------------------

    # Patterns for system_capability intent detection
    # These match questions like "what can you do", "can you browse the web", etc.
    CAPABILITY_PATTERNS = [
        # "you/your" + capability words (expanded list)
        re.compile(r"\b(you|your|yourself)\b.*\b(upgrade|capability|capabilities|tool|tools|browser|internet|web|code|files?|system|memory|brains?|pipeline|version|codebase|filesystem|programs?|applications?)\b", re.IGNORECASE),
        # "can you X" patterns (expanded verb list)
        re.compile(r"\bcan\s+you\b.*\b(browse|search|run|execute|read|write|change|modify|access|control|do|create|update|upgrade)\b", re.IGNORECASE),
        # "what can you" patterns
        re.compile(r"\bwhat\s+(can|could)\s+you\b", re.IGNORECASE),
        # "are you able to" patterns
        re.compile(r"\bare\s+you\s+able\s+to\b", re.IGNORECASE),
        # "do you have" capability patterns
        re.compile(r"\bdo\s+you\s+have\b.*\b(access|ability|capability|tool)\b", re.IGNORECASE),
        # "what upgrade" patterns
        re.compile(r"\bwhat\s+upgrade\b", re.IGNORECASE),
        # "what do you need" / "what * you need" patterns
        re.compile(r"\bwhat\b.*\b(do\s+)?you\s+need\b", re.IGNORECASE),
        # "what tools" patterns
        re.compile(r"\bwhat\s+tools?\b.*\b(can|do)\s+you\b", re.IGNORECASE),
        # "browse the web" + "you" context - CRITICAL: captures "can you browse the web"
        re.compile(r"\b(you|your)\b.*\bbrowse\b.*\bweb\b", re.IGNORECASE),
        re.compile(r"\bbrowse\b.*\bweb\b.*\b(you|your)\b", re.IGNORECASE),
        # "control other programs" patterns
        re.compile(r"\b(you|your)\b.*\bcontrol\b.*\b(other\s+)?(programs?|apps?|applications?)\b", re.IGNORECASE),
        # "read/change files" patterns
        re.compile(r"\b(you|your)\b.*\b(read|change|modify|access)\b.*\bfiles?\b", re.IGNORECASE),
        # "scan your code" patterns
        re.compile(r"\bscan\b.*\b(your|you)\b.*\b(code|codebase|system)\b", re.IGNORECASE),
        re.compile(r"\b(your|you)\b.*\bscan\b.*\b(code|codebase|system)\b", re.IGNORECASE),
        # "what upgrades do you need" - full phrase match
        re.compile(r"\bwhat\s+upgrades?\s+(do\s+)?you\s+need\b", re.IGNORECASE),
        # "can you write/create" patterns for creative tasks (these are capability questions)
        re.compile(r"\bcan\s+you\b.*\b(write|create|generate|compose)\b.*\b(story|poem|essay|article|code|script)\b", re.IGNORECASE),
    ]

    # Patterns for self_identity intent detection
    IDENTITY_PATTERNS = [
        re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+you\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+about\s+yourself\b", re.IGNORECASE),
        re.compile(r"\byour\s+name\b", re.IGNORECASE),
        re.compile(r"\bare\s+you\s+(an?\s+)?(ai|llm|bot|assistant|maven)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+do\s+you\s+work\b", re.IGNORECASE),
    ]

    # ==========================================================================
    # TIME/DATE DETECTION - AGGRESSIVE APPROACH
    # ==========================================================================
    # CRITICAL: Time/date questions MUST route to time_now tool, NOT to Teacher.
    # The LLM cannot provide accurate real-time information and will hallucinate.
    #
    # Strategy:
    # 1. First normalize text (fix typos, collapse spaces)
    # 2. Check against explicit trigger phrases (fast, exact match)
    # 3. Fall back to regex patterns for more complex queries
    # ==========================================================================

    # Explicit trigger phrases (checked BEFORE regex patterns)
    # These catch common queries and typos directly
    TIME_TRIGGERS = {
        # Exact time queries
        "time", "the time", "what time", "what time is it", "whats the time",
        "what is the time", "current time", "time now", "time right now",
        "tell me the time", "give me the time", "show me the time",
        "check the time", "check time", "what does the clock say",
        # Typos
        "wat time", "wats the time", "wat time is it",
    }

    DATE_TRIGGERS = {
        # Exact date queries
        "date", "the date", "what date", "what is the date", "whats the date",
        "current date", "todays date", "today date", "what day",
        "what day is it", "what day is today", "what is today", "whats today",
        # "what is the day" variants (CRITICAL - these were missing)
        "what is the day", "whats the day", "the day", "what day today",
        # "which day" variants
        "which day", "which day is it", "which day is today", "which day today",
        # Today variations
        "today", "to day", "2day",
        # Day of week
        "what day of the week", "what day of week", "day of the week",
        # Typos
        "wat day", "wat is today", "wat day is it", "what is to day",
        "wats the date", "wats today", "whats 2day", "wat is the day",
        "wats the day", "wich day", "wich day is it",
    }

    CALENDAR_TRIGGERS = {
        # Month/year queries
        "what month", "what month is it", "current month", "this month",
        "what year", "what year is it", "current year", "this year",
        "what week", "week number", "current week",
        # Calendar
        "calendar", "show calendar", "show me the calendar", "the calendar",
    }

    # Common typo corrections applied before matching
    TYPO_CORRECTIONS = {
        "to day": "today",
        "2 day": "today",
        "2day": "today",
        "wat ": "what ",
        "wats ": "whats ",
        "whats ": "what is ",
        "wat's ": "what is ",
    }

    # Regex patterns for more complex queries (fallback after trigger check)
    TIME_QUERY_PATTERNS = [
        # Direct time questions
        re.compile(r"\bwhat\s+time\s+is\s+it\b", re.IGNORECASE),
        re.compile(r"\bwhat's\s+the\s+time\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+time\b", re.IGNORECASE),
        re.compile(r"\bcurrent\s+time\b", re.IGNORECASE),
        re.compile(r"\btime\s+(right\s+)?now\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+time\s+is\s+it\s+(now|right\s+now|currently)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+time\s+is\s+it\s+here\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+time\s+is\s+it\s+(in\s+)?(my\s+)?(local|location|area|zone)\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+the\s+time\b", re.IGNORECASE),
        re.compile(r"\bgive\s+me\s+the\s+time\b", re.IGNORECASE),
        re.compile(r"\bshow\s+me\s+the\s+time\b", re.IGNORECASE),
        # Date questions (also served by time tool)
        re.compile(r"\bwhat\s+day\s+is\s+(it|today)\b", re.IGNORECASE),
        re.compile(r"\bwhat's\s+the\s+date\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+date\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+today('s)?\s+date\b", re.IGNORECASE),
        re.compile(r"\btoday's\s+date\b", re.IGNORECASE),
        re.compile(r"\bcurrent\s+date\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+day\s+of\s+(the\s+)?week\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+to\s*day\b", re.IGNORECASE),  # "what is today" / "what is to day"
        re.compile(r"\bwhat's\s+today\b", re.IGNORECASE),  # "what's today"
        re.compile(r"^today\??$", re.IGNORECASE),  # Just "today" or "today?"
        re.compile(r"^to\s*day\??$", re.IGNORECASE),  # "to day" typo
        # "what is the day" variants (CRITICAL - these were missing)
        re.compile(r"\bwhat\s+is\s+the\s+day\b", re.IGNORECASE),  # "what is the day"
        re.compile(r"\bwhat's\s+the\s+day\b", re.IGNORECASE),  # "what's the day"
        # "which day" variants
        re.compile(r"\bwhich\s+day\s+is\s+(it|today)\b", re.IGNORECASE),  # "which day is it"
        re.compile(r"\bwhich\s+day\s+today\b", re.IGNORECASE),  # "which day today"
        re.compile(r"^which\s+day\??$", re.IGNORECASE),  # Just "which day"
        # Calendar/month/year queries
        re.compile(r"\bwhat\s+month\s+is\s+it\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+year\s+is\s+it\b", re.IGNORECASE),
        re.compile(r"\bcurrent\s+(month|year)\b", re.IGNORECASE),
        re.compile(r"\bwhat's\s+the\s+(month|year)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+(current\s+)?(month|year)\b", re.IGNORECASE),
        re.compile(r"^calendar\??$", re.IGNORECASE),
        re.compile(r"\bshow\s+(me\s+)?(the\s+)?calendar\b", re.IGNORECASE),
        re.compile(r"\btoday\s+is\s+what\s+(day|date)\b", re.IGNORECASE),
        # Abbreviated time queries
        re.compile(r"^time\??$", re.IGNORECASE),
        re.compile(r"^the\s+time\??$", re.IGNORECASE),
        re.compile(r"^what\s+time\??$", re.IGNORECASE),
        re.compile(r"^date\??$", re.IGNORECASE),
        # Clock queries
        re.compile(r"\bwhat\s+does\s+(the\s+)?clock\s+say\b", re.IGNORECASE),
        re.compile(r"\bcheck\s+(the\s+)?time\b", re.IGNORECASE),
        re.compile(r"\bcheck\s+(the\s+)?clock\b", re.IGNORECASE),
    ]

    @classmethod
    def _normalize_for_time_detection(cls, text: str) -> str:
        """
        Aggressively normalize text for time/date detection.

        Applies:
        - Lowercase
        - Strip punctuation (except apostrophes)
        - Collapse multiple spaces
        - Fix common typos
        """
        if not text:
            return ""

        # Lowercase
        normalized = text.lower()

        # Strip punctuation except apostrophes (for contractions like "what's")
        normalized = re.sub(r"[^\w\s']", "", normalized)

        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Apply typo corrections
        for typo, fix in cls.TYPO_CORRECTIONS.items():
            normalized = normalized.replace(typo, fix)

        return normalized

    def _classify_intent(self, text: str) -> Optional[str]:
        """
        Classify the intent of the text.

        Returns:
            "time_query" - for time/date questions (HIGHEST PRIORITY - routes to time_now tool)
            "system_capability" - for capability/upgrade/tool questions about Maven
            "self_identity" - for identity questions ("who are you")
            None - for general questions
        """
        if not text:
            return None

        text_lower = text.lower()

        # =================================================================
        # HIGHEST PRIORITY: TIME/DATE DETECTION
        # Time questions MUST route to time_now tool, NEVER to Teacher.
        # The LLM will hallucinate dates/times if this falls through.
        # =================================================================

        # Step 1: Normalize text aggressively (fix typos, collapse spaces)
        normalized = self._normalize_for_time_detection(text)

        # Step 2: Check explicit trigger phrases (fast exact match)
        # This catches common queries and typos that regex might miss
        if normalized in self.TIME_TRIGGERS:
            return "time_query"
        if normalized in self.DATE_TRIGGERS:
            return "time_query"
        if normalized in self.CALENDAR_TRIGGERS:
            return "time_query"

        # Step 3: Check regex patterns (handles more complex queries)
        for pattern in self.TIME_QUERY_PATTERNS:
            if pattern.search(text_lower):
                return "time_query"

        # Also check patterns against the normalized text (catches typo-corrected versions)
        if normalized != text_lower:
            for pattern in self.TIME_QUERY_PATTERNS:
                if pattern.search(normalized):
                    return "time_query"

        # =================================================================
        # Lower priority intents
        # =================================================================

        # Check for system_capability patterns
        for pattern in self.CAPABILITY_PATTERNS:
            if pattern.search(text_lower):
                return "system_capability"

        # Check for self_identity patterns
        for pattern in self.IDENTITY_PATTERNS:
            if pattern.search(text_lower):
                return "self_identity"

        return None

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        """
        Remove punctuation characters, but keep underscores because we use them
        in canonical phrase tokens (e.g. self_scan).
        """
        # Replace punctuation (except underscore) with spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple spaces
        return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Default normalizer instance (lazy initialization)
_default_normalizer: Optional[SemanticNormalizer] = None


def _get_normalizer() -> SemanticNormalizer:
    """Get or create default normalizer instance."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = SemanticNormalizer()
    return _default_normalizer


def normalize_for_routing(text: str) -> NormalizationResult:
    """Module-level function for routing normalization."""
    return _get_normalizer().normalize_for_routing(text)


def normalize_for_memory(text: str) -> NormalizationResult:
    """Module-level function for memory normalization."""
    return _get_normalizer().normalize_for_memory(text)


def normalize_for_pattern(text: str) -> NormalizationResult:
    """Module-level function for pattern learning normalization."""
    return _get_normalizer().normalize_for_pattern(text)


def get_canonical_token(token: str) -> Optional[str]:
    """Module-level function to get canonical form of a token."""
    return _get_normalizer().get_canonical_for_token(token)


def classify_intent(text: str) -> Optional[str]:
    """
    Module-level function to classify intent of text.

    Returns:
        "system_capability" - for capability/upgrade/tool questions about Maven
        "self_identity" - for identity questions ("who are you")
        None - for general questions
    """
    return _get_normalizer()._classify_intent(text)


def is_system_capability_query(text: str) -> bool:
    """Check if the text is a system capability question."""
    return classify_intent(text) == "system_capability"


def is_self_identity_query(text: str) -> bool:
    """Check if the text is a self-identity question."""
    return classify_intent(text) == "self_identity"


def is_time_query(text: str) -> bool:
    """
    Check if the text is a time/date query.

    CRITICAL: Time queries MUST route to time_now tool, NOT Teacher.
    The Teacher/LLM cannot provide accurate real-time information.

    Args:
        text: The user's query

    Returns:
        True if this is a time/date question
    """
    return classify_intent(text) == "time_query"


def get_time_query_type(text: str) -> Optional[str]:
    """
    Classify what type of time query this is: time, date, or calendar.

    Args:
        text: The user's query

    Returns:
        'time' for clock/hour questions
        'date' for day/date questions
        'calendar' for month/year/calendar questions
        None if not a time query
    """
    if not is_time_query(text):
        return None

    # Normalize text aggressively for matching
    normalized = SemanticNormalizer._normalize_for_time_detection(text)
    text_lower = text.lower()

    # Check trigger sets first (fast exact match on normalized text)
    if normalized in SemanticNormalizer.CALENDAR_TRIGGERS:
        return "calendar"
    if normalized in SemanticNormalizer.DATE_TRIGGERS:
        return "date"
    if normalized in SemanticNormalizer.TIME_TRIGGERS:
        return "time"

    # Fall back to pattern matching
    # Calendar patterns (month, year, calendar)
    calendar_patterns = [
        r"\bmonth\b",
        r"\byear\b",
        r"\bcalendar\b",
        r"\bweek\s+number\b",
    ]
    for pattern in calendar_patterns:
        if re.search(pattern, text_lower) or re.search(pattern, normalized):
            return "calendar"

    # Date patterns (day, date, today)
    date_patterns = [
        r"\bdate\b",
        r"\bday\b",
        r"\btoday\b",
        r"\bto\s*day\b",  # Handle typo
    ]
    for pattern in date_patterns:
        if re.search(pattern, text_lower) or re.search(pattern, normalized):
            return "date"

    # Default to time (clock, time, now, etc.)
    return "time"


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for semantic normalizer.

    Supported operations:
    - NORMALIZE: Normalize text with options
    - NORMALIZE_ROUTING: Normalize for routing
    - NORMALIZE_MEMORY: Normalize for memory
    - NORMALIZE_PATTERN: Normalize for pattern learning
    - GET_CANONICAL: Get canonical form of a token
    - CONFIG_SUMMARY: Get configuration summary
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    normalizer = _get_normalizer()

    if op == "NORMALIZE":
        try:
            text = payload.get("text", "")
            if not text:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TEXT", "message": "text required"},
                }

            result = normalizer.normalize(
                text,
                for_routing=payload.get("for_routing", False),
                for_memory=payload.get("for_memory", False),
                lowercase=payload.get("lowercase", True),
                strip_punctuation=payload.get("strip_punctuation", True),
                lemmatize=payload.get("lemmatize", True),
            )
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "NORMALIZE_FAILED", "message": str(e)},
            }

    if op == "NORMALIZE_ROUTING":
        try:
            text = payload.get("text", "")
            if not text:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TEXT", "message": "text required"},
                }

            result = normalizer.normalize_for_routing(text)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "NORMALIZE_FAILED", "message": str(e)},
            }

    if op == "NORMALIZE_MEMORY":
        try:
            text = payload.get("text", "")
            if not text:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TEXT", "message": "text required"},
                }

            result = normalizer.normalize_for_memory(text)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "NORMALIZE_FAILED", "message": str(e)},
            }

    if op == "NORMALIZE_PATTERN":
        try:
            text = payload.get("text", "")
            if not text:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TEXT", "message": "text required"},
                }

            result = normalizer.normalize_for_pattern(text)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "NORMALIZE_FAILED", "message": str(e)},
            }

    if op == "GET_CANONICAL":
        try:
            token = payload.get("token", "")
            if not token:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TOKEN", "message": "token required"},
                }

            canonical = normalizer.get_canonical_for_token(token)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"token": token, "canonical": canonical},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LOOKUP_FAILED", "message": str(e)},
            }

    if op == "CONFIG_SUMMARY":
        try:
            summary = normalizer.get_config_summary()
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": summary,
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "SUMMARY_FAILED", "message": str(e)},
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "semantic_normalizer",
                "config_loaded": len(normalizer.config.phrase_rules) > 0 or len(normalizer.config.token_rules) > 0,
                "phrase_rules_count": len(normalizer.config.phrase_rules),
                "token_rules_count": len(normalizer.config.token_rules),
                "available_operations": [
                    "NORMALIZE", "NORMALIZE_ROUTING", "NORMALIZE_MEMORY",
                    "NORMALIZE_PATTERN", "GET_CANONICAL", "CONFIG_SUMMARY", "HEALTH"
                ],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
