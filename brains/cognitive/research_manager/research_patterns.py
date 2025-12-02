"""
Research Pattern Learning System

Purpose
-------
Provide reusable research patterns learned from past successful research tasks.

Core concepts:
- ResearchTaskRecord: one completed research run (topic, subquestions, sources, outcome).
- ResearchPattern: reusable template capturing:
    - topic signature (normalized tokens)
    - subquestion templates
    - preferred sources/domains
    - typical depth and options

Behavior:
- record_research_task(...) is called by the research manager whenever a task completes.
- Successful tasks (or tasks marked "promote_to_pattern") are turned into patterns.
- suggest_research_plan(topic, metadata) returns a plan that reuses learned patterns.

Storage:
- ~/.maven/research_tasks.jsonl      (raw task history)
- ~/.maven/research_patterns.jsonl   (pattern definitions)
- ~/.maven/research_sources.json     (per-domain stats: successes/failures)

No stubs. Everything in this module works on its own.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"

RESEARCH_TASKS_LOG_PATH = MAVEN_DIR / "research_tasks.jsonl"
RESEARCH_PATTERNS_LOG_PATH = MAVEN_DIR / "research_patterns.jsonl"
RESEARCH_SOURCES_STATE_PATH = MAVEN_DIR / "research_sources.json"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ResearchSubquestion:
    """
    One sub-question within a research task.

    Fields:
      text: actual sub-question asked.
      kind: optional label (e.g. "definition", "comparison", "timeline").
      notes: arbitrary metadata.
    """
    text: str
    kind: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSourceUsage:
    """
    Snapshot of one source used in a research task.

    Fields:
      name: logical name, e.g. "wikipedia", "openai", "nytimes".
      url: URL if applicable.
      domain: domain extracted from url (if any).
      reliability_hint: optional; caller can mark "trusted", "untrusted", etc.
    """
    name: str
    url: Optional[str] = None
    domain: Optional[str] = None
    reliability_hint: Optional[str] = None


@dataclass
class ResearchTaskRecord:
    """
    One completed research task.

    Fields:
      task_id: unique id for the task (provided or auto-generated).
      topic: original topic string.
      normalized_topic: normalized form for matching.
      tokens: normalized tokens.
      subquestions: list of ResearchSubquestion.
      sources: list of ResearchSourceUsage.
      depth: requested research depth.
      success: bool indicating whether the task achieved its goal.
      promote_to_pattern: bool, if caller wants to force pattern learning.
      metadata: arbitrary dict.
      ts: timestamp.
    """
    task_id: str
    topic: str
    normalized_topic: str
    tokens: List[str]
    subquestions: List[ResearchSubquestion]
    sources: List[ResearchSourceUsage]
    depth: int
    success: bool
    promote_to_pattern: bool
    metadata: Dict[str, Any]
    ts: str


@dataclass
class ResearchPattern:
    """
    Reusable research pattern template.

    Fields:
      id: unique pattern id.
      topic_signature: canonical topic string (e.g. "birds general facts").
      topic_tokens: normalized tokens describing the topic family.
      example_topic: example actual topic that generated this pattern.
      subquestion_templates: list of template strings, may contain {topic} etc.
      preferred_domains: domains that worked well for this pattern.
      typical_depth: typical depth used (int).
      tags: labels for what this pattern is about.
      usage_count: how many times the pattern was used.
      success_count: how many times pattern-based runs succeeded.
      last_used_ts: last time used.
      created_ts: creation time.
    """
    id: str
    topic_signature: str
    topic_tokens: List[str]
    example_topic: str
    subquestion_templates: List[str]
    preferred_domains: List[str]
    typical_depth: int
    tags: List[str]
    usage_count: int
    success_count: int
    last_used_ts: Optional[str]
    created_ts: str


@dataclass
class SuggestedResearchPlan:
    """
    Plan suggested for a new research topic.

    Fields:
      pattern_id: id of pattern used (if any).
      topic: original topic.
      normalized_topic: normalized topic string.
      tokens: normalized tokens.
      depth: recommended depth.
      subquestions: list of ResearchSubquestion (actual texts for this run).
      preferred_domains: domains to prioritize for sources.
      debug: any additional info.
    """
    pattern_id: Optional[str]
    topic: str
    normalized_topic: str
    tokens: List[str]
    depth: int
    subquestions: List[ResearchSubquestion]
    preferred_domains: List[str]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["subquestions"] = [asdict(sq) for sq in self.subquestions]
        return result


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        _ensure_dir(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error("Failed to append to %s: %s", path, e)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load JSON from %s: %s", path, e)
        return None


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        _ensure_dir(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.error("Failed to save JSON to %s: %s", path, e)


def _normalize_tokens_simple(text: str) -> List[str]:
    text = text.lower()
    tokens = re.split(r"[^a-z0-9_]+", text)
    return [t for t in tokens if t]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    if not union:
        return 0.0
    return len(inter) / len(union)


# =============================================================================
# Source reliability tracking
# =============================================================================

@dataclass
class SourceStats:
    uses: int = 0
    successes: int = 0
    failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses


class SourceReliability:
    """
    Tracks per-domain reliability.

    Stored in ~/.maven/research_sources.json:

        {
          "example.com": {"uses": 10, "successes": 7, "failures": 3},
          ...
        }
    """

    def __init__(self, path: Path = RESEARCH_SOURCES_STATE_PATH) -> None:
        self.path = path
        self.stats: Dict[str, SourceStats] = {}
        self._load()

    def _load(self) -> None:
        data = _load_json(self.path)
        if not data:
            self.stats = {}
            return
        self.stats = {}
        for domain, v in data.items():
            try:
                self.stats[domain] = SourceStats(
                    uses=int(v.get("uses", 0)),
                    successes=int(v.get("successes", 0)),
                    failures=int(v.get("failures", 0)),
                )
            except Exception:
                continue

    def _save(self) -> None:
        data = {
            domain: {
                "uses": s.uses,
                "successes": s.successes,
                "failures": s.failures,
            }
            for domain, s in self.stats.items()
        }
        _save_json(self.path, data)

    def update_from_task(self, sources: List[ResearchSourceUsage], success: bool) -> None:
        for s in sources:
            domain = (s.domain or "").strip().lower()
            if not domain:
                continue
            stats = self.stats.get(domain, SourceStats())
            stats.uses += 1
            if success:
                stats.successes += 1
            else:
                stats.failures += 1
            self.stats[domain] = stats
        self._save()

    def score_domain(self, domain: str) -> float:
        domain = domain.strip().lower()
        if not domain:
            return 0.0
        stats = self.stats.get(domain)
        if not stats:
            return 0.0
        # Simple mapping: success_rate in [0,1] -> [0,1]
        return stats.success_rate

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all source statistics."""
        return {
            domain: {
                "uses": s.uses,
                "successes": s.successes,
                "failures": s.failures,
                "success_rate": s.success_rate,
            }
            for domain, s in self.stats.items()
        }


# =============================================================================
# Pattern manager
# =============================================================================

class ResearchPatternManager:
    """
    Central manager for research pattern learning.

    Responsibilities:
      - Record completed research tasks.
      - Learn patterns from successful tasks.
      - Suggest research plans based on existing patterns.
      - Track source reliability and prefer better domains.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None

        # Try to use SemanticNormalizer, fall back to simple normalization
        try:
            from brains.normalization.semantic_normalizer import SemanticNormalizer
            self.normalizer = SemanticNormalizer(project_root=project_root)
            self._has_normalizer = True
        except ImportError:
            self.normalizer = None
            self._has_normalizer = False
            logger.warning("SemanticNormalizer not available; using simple normalization")

        self.source_reliability = SourceReliability()
        self.patterns: Dict[str, ResearchPattern] = {}
        self._load_patterns()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def record_research_task(
        self,
        topic: str,
        subquestions: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        depth: int,
        success: bool,
        promote_to_pattern: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> ResearchTaskRecord:
        """
        Record a completed research task.

        subquestions: list of dicts with keys:
            - text (required)
            - kind (optional)
            - notes (optional dict)

        sources: list of dicts with keys:
            - name
            - url (optional)
            - domain (optional)
            - reliability_hint (optional)

        Returns: ResearchTaskRecord
        """
        metadata = metadata or {}
        depth = int(depth) if depth is not None else 1

        # Normalize topic
        if self._has_normalizer and self.normalizer is not None:
            norm_res = self.normalizer.normalize_for_routing(topic)
            normalized_topic = norm_res.normalized
            tokens = norm_res.tokens
        else:
            normalized_topic = topic.lower().strip()
            tokens = _normalize_tokens_simple(topic)

        ts = _now_iso()
        if not task_id:
            task_id = f"research-{ts}"

        subs = [
            ResearchSubquestion(
                text=str(sq.get("text", "")),
                kind=sq.get("kind"),
                notes=sq.get("notes") or {},
            )
            for sq in subquestions
            if str(sq.get("text", "")).strip()
        ]

        srcs = [
            ResearchSourceUsage(
                name=str(s.get("name") or "unknown"),
                url=s.get("url"),
                domain=s.get("domain"),
                reliability_hint=s.get("reliability_hint"),
            )
            for s in sources
        ]

        record = ResearchTaskRecord(
            task_id=task_id,
            topic=topic,
            normalized_topic=normalized_topic,
            tokens=tokens,
            subquestions=subs,
            sources=srcs,
            depth=depth,
            success=bool(success),
            promote_to_pattern=bool(promote_to_pattern),
            metadata=metadata,
            ts=ts,
        )

        # Append to task log
        rec = {
            "kind": "research_task",
            "ts": ts,
            "task": asdict(record),
        }
        _append_jsonl(RESEARCH_TASKS_LOG_PATH, rec)

        # Update source reliability
        self.source_reliability.update_from_task(srcs, success=bool(success))

        # Maybe learn pattern
        if success or promote_to_pattern:
            self._learn_pattern_from_task(record)

        return record

    def suggest_research_plan(
        self,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None,
        default_depth: int = 2,
    ) -> SuggestedResearchPlan:
        """
        Suggest a research plan for a new topic based on existing patterns.

        If no pattern is suitable, returns a generic plan with a single
        subquestion equal to the topic.

        metadata may contain:
            - tags: list[str] hinting the kind of research (e.g., ["definition", "overview"])
        """
        metadata = metadata or {}

        # Normalize topic
        if self._has_normalizer and self.normalizer is not None:
            norm_res = self.normalizer.normalize_for_routing(topic)
            normalized_topic = norm_res.normalized
            tokens = norm_res.tokens
        else:
            normalized_topic = topic.lower().strip()
            tokens = _normalize_tokens_simple(topic)

        # 1) pick best pattern
        best, score = self._select_best_pattern(tokens, metadata)
        if best is None:
            # generic fallback
            subquestions = [
                ResearchSubquestion(
                    text=f"What are the key facts about {topic}?",
                    kind="overview",
                    notes={},
                ),
                ResearchSubquestion(
                    text=f"What are the most important aspects of {topic}?",
                    kind="key_points",
                    notes={},
                ),
            ]
            return SuggestedResearchPlan(
                pattern_id=None,
                topic=topic,
                normalized_topic=normalized_topic,
                tokens=tokens,
                depth=default_depth,
                subquestions=subquestions,
                preferred_domains=[],
                debug={
                    "note": "No suitable pattern found; using generic overview plan.",
                    "pattern_score": None,
                    "available_patterns": len(self.patterns),
                },
            )

        # 2) instantiate pattern subquestions
        subquestions = self._instantiate_subquestions(best, topic)

        # 3) choose depth (use pattern typical_depth if reasonable)
        depth = max(1, best.typical_depth or default_depth)

        # 4) prefer domains that have good reliability
        ordered_domains = sorted(
            best.preferred_domains,
            key=lambda d: self.source_reliability.score_domain(d),
            reverse=True,
        )

        plan = SuggestedResearchPlan(
            pattern_id=best.id,
            topic=topic,
            normalized_topic=normalized_topic,
            tokens=tokens,
            depth=depth,
            subquestions=subquestions,
            preferred_domains=ordered_domains,
            debug={
                "pattern_score": score,
                "pattern_topic_signature": best.topic_signature,
                "pattern_tags": best.tags,
            },
        )

        # Update usage counters and persist
        self._mark_pattern_used(best.id, success=None)

        return plan

    def mark_plan_result(
        self,
        pattern_id: Optional[str],
        success: bool,
    ) -> None:
        """
        Inform the pattern manager whether a pattern-based plan succeeded.

        This is optional but improves learning: it updates usage and success_count.
        """
        if not pattern_id or pattern_id not in self.patterns:
            return

        self._mark_pattern_used(pattern_id, success=success)

    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all learned patterns with summary info."""
        return [
            {
                "id": p.id,
                "topic_signature": p.topic_signature,
                "example_topic": p.example_topic,
                "subquestion_count": len(p.subquestion_templates),
                "domain_count": len(p.preferred_domains),
                "usage_count": p.usage_count,
                "success_count": p.success_count,
                "tags": p.tags,
            }
            for p in self.patterns.values()
        ]

    def get_pattern_details(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a specific pattern."""
        pat = self.patterns.get(pattern_id)
        if pat is None:
            return None
        return asdict(pat)

    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get source reliability statistics."""
        return self.source_reliability.get_all_stats()

    # -------------------------------------------------------------------------
    # Pattern learning
    # -------------------------------------------------------------------------

    def _learn_pattern_from_task(self, task: ResearchTaskRecord) -> None:
        """
        Learn or update a pattern based on a successful research task.

        Strategy:
        - Derive topic_signature from tokens and (optional) metadata hints.
        - Use subquestions as templates with {topic} where appropriate.
        - Collect domains from sources as preferred_domains.
        - If a similar pattern exists, don't create a duplicate; instead, merge.

        No guessing about subquestion structure: we only use what was given.
        """
        # Build topic_signature by joining unique tokens
        unique_tokens = []
        for t in task.tokens:
            if t not in unique_tokens:
                unique_tokens.append(t)
        topic_signature = " ".join(unique_tokens)

        # Build initial pattern candidate
        subquestion_templates: List[str] = []
        for sq in task.subquestions:
            text = sq.text.strip()
            if not text:
                continue
            # naive replacement: use {topic} where original topic appears
            lowered = text.lower()
            topic_lower = task.topic.lower()
            if topic_lower and topic_lower in lowered:
                templ = text.replace(task.topic, "{topic}")
            else:
                templ = text
            subquestion_templates.append(templ)

        # Collect domains from sources
        domains: List[str] = []
        for s in task.sources:
            d = (s.domain or "").strip().lower()
            if d and d not in domains:
                domains.append(d)

        # Derive tags loosely from metadata and tokens
        tags = list(task.metadata.get("tags", []))
        if not tags:
            tags = task.tokens[:10]  # first few tokens as crude tags

        candidate = ResearchPattern(
            id=f"pattern-{task.task_id}",
            topic_signature=topic_signature,
            topic_tokens=task.tokens,
            example_topic=task.topic,
            subquestion_templates=subquestion_templates or [f"What are the key facts about {{topic}}?"],
            preferred_domains=domains,
            typical_depth=max(1, task.depth),
            tags=tags,
            usage_count=0,
            success_count=0,
            last_used_ts=None,
            created_ts=task.ts,
        )

        # Check for an existing similar pattern
        existing_id, sim_score = self._find_similar_pattern(candidate)
        if existing_id is not None and sim_score >= 0.6:
            # Merge into existing pattern
            self._merge_pattern(existing_id, candidate)
            pid = existing_id
        else:
            # Add as new pattern
            self.patterns[candidate.id] = candidate
            self._persist_pattern(candidate)
            pid = candidate.id

        logger.info("Research pattern learned/updated: %s", pid)

    def _find_similar_pattern(self, candidate: ResearchPattern) -> Tuple[Optional[str], float]:
        """
        Find the most similar existing pattern based on topic_tokens Jaccard.
        """
        best_id = None
        best_score = 0.0
        for pid, pat in self.patterns.items():
            score = _jaccard(candidate.topic_tokens, pat.topic_tokens)
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id, best_score

    def _merge_pattern(self, pattern_id: str, candidate: ResearchPattern) -> None:
        """
        Merge a candidate into an existing pattern conservatively.

        - Union of topic tokens.
        - Union of subquestion templates.
        - Union of preferred domains.
        - Typical depth = max(current, candidate).
        - Tags = union.
        """
        pat = self.patterns[pattern_id]

        # Merge tokens
        tokens = list(pat.topic_tokens)
        for t in candidate.topic_tokens:
            if t not in tokens:
                tokens.append(t)
        pat.topic_tokens = tokens
        pat.topic_signature = " ".join(tokens)

        # Merge subquestion templates
        for tq in candidate.subquestion_templates:
            if tq not in pat.subquestion_templates:
                pat.subquestion_templates.append(tq)

        # Merge domains
        for d in candidate.preferred_domains:
            if d not in pat.preferred_domains:
                pat.preferred_domains.append(d)

        # Depth: keep the larger
        pat.typical_depth = max(pat.typical_depth, candidate.typical_depth)

        # Merge tags
        for tag in candidate.tags:
            if tag not in pat.tags:
                pat.tags.append(tag)

        self.patterns[pattern_id] = pat
        self._persist_pattern(pat, replace=True)

    def _select_best_pattern(
        self,
        topic_tokens: List[str],
        metadata: Dict[str, Any],
    ) -> Tuple[Optional[ResearchPattern], Optional[float]]:
        """
        Choose the best pattern for the current topic.

        Scoring:
        - Topic Jaccard similarity.
        - Tag hint overlap (if metadata["tags"] present).
        - Soft bonus for patterns with higher success_rate.
        """
        tags_hint = metadata.get("tags") or []
        tags_hint_tokens = []
        for t in tags_hint:
            tags_hint_tokens.extend(_normalize_tokens_simple(str(t)))

        best_pat: Optional[ResearchPattern] = None
        best_score: float = 0.0

        for pat in self.patterns.values():
            sim_topic = _jaccard(topic_tokens, pat.topic_tokens)
            sim_tags = _jaccard(tags_hint_tokens, _normalize_tokens_simple(" ".join(pat.tags))) if tags_hint_tokens else 0.0
            usage = max(1, pat.usage_count)
            success_rate = pat.success_count / usage

            score = 0.6 * sim_topic + 0.2 * sim_tags + 0.2 * success_rate

            if score > best_score:
                best_score = score
                best_pat = pat

        if best_pat is None or best_score < 0.3:
            return None, None

        return best_pat, best_score

    def _instantiate_subquestions(self, pattern: ResearchPattern, topic: str) -> List[ResearchSubquestion]:
        """
        Turn pattern.subquestion_templates into concrete ResearchSubquestion list
        for the given topic.
        """
        subs: List[ResearchSubquestion] = []
        for templ in pattern.subquestion_templates:
            try:
                text = templ.format(topic=topic)
            except Exception:
                text = templ
            subs.append(
                ResearchSubquestion(
                    text=text,
                    kind=None,
                    notes={"source_pattern": pattern.id},
                )
            )
        return subs

    def _mark_pattern_used(self, pattern_id: str, success: Optional[bool]) -> None:
        pat = self.patterns.get(pattern_id)
        if not pat:
            return
        pat.usage_count += 1
        if success is True:
            pat.success_count += 1
        pat.last_used_ts = _now_iso()
        self.patterns[pattern_id] = pat
        self._persist_pattern(pat, replace=True)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load_patterns(self) -> None:
        """
        Load patterns from JSONL file.

        Each line:
          {
            "kind": "research_pattern",
            "ts": "...",
            "pattern": { ... ResearchPattern fields ... }
          }
        """
        self.patterns = {}
        if not RESEARCH_PATTERNS_LOG_PATH.exists():
            return

        try:
            with RESEARCH_PATTERNS_LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("kind") != "research_pattern":
                        continue
                    payload = obj.get("pattern") or {}
                    try:
                        pat = ResearchPattern(
                            id=str(payload["id"]),
                            topic_signature=str(payload["topic_signature"]),
                            topic_tokens=list(payload.get("topic_tokens", [])),
                            example_topic=str(payload.get("example_topic", "")),
                            subquestion_templates=list(payload.get("subquestion_templates", [])),
                            preferred_domains=list(payload.get("preferred_domains", [])),
                            typical_depth=int(payload.get("typical_depth", 1)),
                            tags=list(payload.get("tags", [])),
                            usage_count=int(payload.get("usage_count", 0)),
                            success_count=int(payload.get("success_count", 0)),
                            last_used_ts=payload.get("last_used_ts"),
                            created_ts=str(payload.get("created_ts", _now_iso())),
                        )
                        self.patterns[pat.id] = pat
                    except Exception as e:
                        logger.error("Failed to parse research pattern line: %s", e)
        except Exception as e:
            logger.error("Failed to load research patterns: %s", e)
            self.patterns = {}

    def _persist_pattern(self, pattern: ResearchPattern, replace: bool = False) -> None:
        """
        Persist a pattern.

        Strategy:
        - If replace=False, append a new line for this pattern.
        - If replace=True, we rewrite the entire file using self.patterns.
          (We keep this simple and robust rather than clever.)
        """
        if not replace:
            rec = {
                "kind": "research_pattern",
                "ts": _now_iso(),
                "pattern": asdict(pattern),
            }
            _append_jsonl(RESEARCH_PATTERNS_LOG_PATH, rec)
            return

        # rewrite all patterns
        try:
            _ensure_dir(RESEARCH_PATTERNS_LOG_PATH)
            with RESEARCH_PATTERNS_LOG_PATH.open("w", encoding="utf-8") as f:
                for pat in self.patterns.values():
                    rec = {
                        "kind": "research_pattern",
                        "ts": _now_iso(),
                        "pattern": asdict(pat),
                    }
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.error("Failed to rewrite research patterns: %s", e)


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Default manager instance (lazy initialization)
_default_manager: Optional[ResearchPatternManager] = None


def _get_manager() -> ResearchPatternManager:
    """Get or create default manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ResearchPatternManager()
    return _default_manager


def get_default_research_pattern_manager() -> ResearchPatternManager:
    """Get the default research pattern manager instance."""
    return _get_manager()


def record_research_task(
    topic: str,
    subquestions: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    depth: int,
    success: bool,
    promote_to_pattern: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> ResearchTaskRecord:
    """Module-level function to record a research task."""
    return _get_manager().record_research_task(
        topic=topic,
        subquestions=subquestions,
        sources=sources,
        depth=depth,
        success=success,
        promote_to_pattern=promote_to_pattern,
        metadata=metadata,
        task_id=task_id,
    )


def suggest_research_plan(
    topic: str,
    metadata: Optional[Dict[str, Any]] = None,
    default_depth: int = 2,
) -> SuggestedResearchPlan:
    """Module-level function to suggest a research plan."""
    return _get_manager().suggest_research_plan(
        topic=topic,
        metadata=metadata,
        default_depth=default_depth,
    )


def mark_plan_result(pattern_id: Optional[str], success: bool) -> None:
    """Module-level function to mark plan result."""
    _get_manager().mark_plan_result(pattern_id, success)


def list_research_patterns() -> List[Dict[str, Any]]:
    """Module-level function to list all patterns."""
    return _get_manager().list_patterns()


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for research pattern learning.

    Supported operations:
    - RECORD_TASK: Record a completed research task
    - SUGGEST_PLAN: Get a suggested research plan for a topic
    - MARK_RESULT: Mark the result of a pattern-based plan
    - LIST_PATTERNS: List all learned patterns
    - GET_PATTERN: Get details of a specific pattern
    - SOURCE_STATS: Get source reliability statistics
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    manager = _get_manager()

    if op == "RECORD_TASK":
        try:
            topic = payload.get("topic", "")
            subquestions = payload.get("subquestions", [])
            sources = payload.get("sources", [])
            depth = payload.get("depth", 1)
            success = payload.get("success", False)
            promote_to_pattern = payload.get("promote_to_pattern", False)
            metadata = payload.get("metadata", {})
            task_id = payload.get("task_id")

            if not topic:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TOPIC", "message": "topic required"},
                }

            record = manager.record_research_task(
                topic=topic,
                subquestions=subquestions,
                sources=sources,
                depth=depth,
                success=success,
                promote_to_pattern=promote_to_pattern,
                metadata=metadata,
                task_id=task_id,
            )
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "task_id": record.task_id,
                    "pattern_count": len(manager.patterns),
                },
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "RECORD_FAILED", "message": str(e)},
            }

    if op == "SUGGEST_PLAN":
        try:
            topic = payload.get("topic", "")
            metadata = payload.get("metadata", {})
            default_depth = payload.get("default_depth", 2)

            if not topic:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TOPIC", "message": "topic required"},
                }

            plan = manager.suggest_research_plan(
                topic=topic,
                metadata=metadata,
                default_depth=default_depth,
            )
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": plan.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "SUGGEST_FAILED", "message": str(e)},
            }

    if op == "MARK_RESULT":
        try:
            pattern_id = payload.get("pattern_id")
            success = payload.get("success", False)

            manager.mark_plan_result(pattern_id, success)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"message": "Result marked", "pattern_id": pattern_id},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "MARK_FAILED", "message": str(e)},
            }

    if op == "LIST_PATTERNS":
        try:
            patterns = manager.list_patterns()
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"patterns": patterns, "count": len(patterns)},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LIST_FAILED", "message": str(e)},
            }

    if op == "GET_PATTERN":
        try:
            pattern_id = payload.get("pattern_id")
            if not pattern_id:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_ID", "message": "pattern_id required"},
                }

            details = manager.get_pattern_details(pattern_id)
            if details is None:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "NOT_FOUND", "message": f"Pattern '{pattern_id}' not found"},
                }

            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": details,
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "GET_FAILED", "message": str(e)},
            }

    if op == "SOURCE_STATS":
        try:
            stats = manager.get_source_stats()
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"sources": stats, "count": len(stats)},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "STATS_FAILED", "message": str(e)},
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "research_patterns",
                "pattern_count": len(manager.patterns),
                "source_count": len(manager.source_reliability.stats),
                "tasks_log_path": str(RESEARCH_TASKS_LOG_PATH),
                "patterns_log_path": str(RESEARCH_PATTERNS_LOG_PATH),
                "available_operations": [
                    "RECORD_TASK", "SUGGEST_PLAN", "MARK_RESULT",
                    "LIST_PATTERNS", "GET_PATTERN", "SOURCE_STATS", "HEALTH"
                ],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
