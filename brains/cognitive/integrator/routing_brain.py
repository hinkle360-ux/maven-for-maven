"""
Routing Brain (Reinforcement Version)

Purpose
-------
Decide which cognitive route(s) to use for a given user question, and learn from
feedback over time.

Core behavior:
- For each incoming question:
    - Normalize text (SemanticNormalizer).
    - Score each candidate route based on:
        - language/domain suitability
        - tag similarity
        - learned bias (from past feedback)
    - Select top-N routes.
    - Log the decision to JSONL for audit.

- When feedback arrives:
    - Convert verdict (ok / minor_issue / major_issue / fail) to reward.
    - Update per-route stats and bias.
    - Persist new state.

No stubs. If no good route is found, router falls back to a default route and
says so explicitly.

This module is self-contained and does not depend on any LLMs.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
ROUTING_STATE_PATH = MAVEN_DIR / "routing_state.json"
ROUTING_DECISIONS_LOG_PATH = MAVEN_DIR / "routing_decisions.jsonl"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RouteConfig:
    """
    Static config for a single route/brain/bank.

    Fields:
      id: stable identifier, e.g., "language", "technology", "fs_tool"
      label: human-readable label
      tags: keywords describing what this route is good at
      base_priority: float baseline preference (0–1)
      allowed_kinds: optional list of question kinds ("direct_question", "task_request", ...)
      disallowed_prefixes: optional list of string prefixes that should not use this route
      max_parallel: how many sibling routes this one is allowed to participate with
    """
    id: str
    label: str
    tags: List[str]
    base_priority: float = 0.5
    allowed_kinds: List[str] = field(default_factory=list)
    disallowed_prefixes: List[str] = field(default_factory=list)
    max_parallel: int = 2


@dataclass
class RouteStats:
    """
    Learned state for a route.

    Fields:
      n: number of feedback updates
      total_reward: sum of rewards
      avg_reward: running average reward
      bias: learned adjustment term added to base_priority
      last_verdict: last verdict seen
    """
    n: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    bias: float = 0.0
    last_verdict: Optional[str] = None


@dataclass
class RoutingState:
    """
    Global routing state.

    - routes: per-route stats
    - meta: any extra metadata (version, etc.)
    """
    routes: Dict[str, RouteStats] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=lambda: {"version": 1})


@dataclass
class RoutingDecision:
    """
    Output from a routing call.

    Fields:
      decision_id: UUID for this routing decision.
      chosen_routes: list of route ids selected (sorted by score).
      scores: map from route id to score.
      normalized_text: normalized question string.
      tokens: normalized tokens.
      debug: extra info (sig, kind, etc.).
    """
    decision_id: str
    chosen_routes: List[str]
    scores: Dict[str, float]
    normalized_text: str
    tokens: List[str]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class RoutingFeedback:
    """
    Feedback for a previous routing decision.

    Fields:
      decision_id: UUID from RoutingDecision (optional, but recommended).
      route_id: the specific route we are giving feedback on.
      verdict: "ok" | "minor_issue" | "major_issue" | "fail"
      metadata: optional extra context (e.g. brain name, error info).
    """
    decision_id: Optional[str]
    route_id: str
    verdict: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def _normalize_tokens(text: str) -> List[str]:
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


def _verdict_to_reward(verdict: str) -> float:
    """
    Map feedback verdict to reward.

    You can tune these numbers; they define how strongly routing adapts.
    """
    v = verdict.lower()
    if v == "ok":
        return 1.0
    if v == "minor_issue":
        return 0.3
    if v == "major_issue":
        return -0.5
    if v == "fail":
        return -1.0
    # Unknown verdict: neutral
    return 0.0


def _reward_to_bias(avg_reward: float, scale: float = 0.3) -> float:
    """
    Convert running average reward to a bias term in [-scale, +scale].

    Uses tanh squashing for smooth bounds.
    """
    return float(scale * math.tanh(avg_reward))


# =============================================================================
# RoutingBrain
# =============================================================================

class RoutingBrain:
    """
    Routing Brain with simple reinforcement learning.

    Initialization:
        normalizer: SemanticNormalizer instance (optional)
        routes_config: optional list[RouteConfig]; if None, use built-in defaults.

    Public methods:
        route(question, metadata) -> RoutingDecision
        apply_feedback(RoutingFeedback) -> None
        get_route_stats() -> Dict[str, RouteStats]
    """

    def __init__(
        self,
        normalizer: Optional[Any] = None,
        routes_config: Optional[List[RouteConfig]] = None,
    ) -> None:
        # Lazy import normalizer to avoid circular dependencies
        if normalizer is None:
            try:
                from brains.normalization.semantic_normalizer import SemanticNormalizer
                self.normalizer = SemanticNormalizer()
            except ImportError:
                self.normalizer = None
                logger.warning("SemanticNormalizer not available; using basic normalization")
        else:
            self.normalizer = normalizer

        self.routes_config: Dict[str, RouteConfig] = {}
        self.state = RoutingState()

        if routes_config is None:
            routes_config = self._default_routes()

        for rc in routes_config:
            self.routes_config[rc.id] = rc

        self._load_state()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def route(
        self,
        question: str,
        metadata: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Main routing entrypoint.

        Inputs:
            question: raw user text
            metadata: may include:
                - question_kind: "direct_question", "task_request", "self_diag", etc.
            top_n: optional limit on number of routes to return (default: 3)

        Returns:
            RoutingDecision:
                - decision_id
                - chosen_routes (list of route ids)
                - scores (per route)
                - normalized_text
                - tokens
                - debug info
        """
        metadata = metadata or {}
        question_kind = str(metadata.get("question_kind", "")).lower()
        max_routes = top_n if top_n is not None else 3

        # 1) Normalize question
        if self.normalizer is not None:
            norm_res = self.normalizer.normalize_for_routing(question)
            norm_text = norm_res.normalized
            tokens = norm_res.tokens
        else:
            # Fallback basic normalization
            norm_text = question.lower().strip()
            tokens = _normalize_tokens(norm_text)

        # 2) Score each configured route
        scores: Dict[str, float] = {}
        route_debug: Dict[str, Any] = {}
        for route_id, cfg in self.routes_config.items():
            score, sd = self._score_route(cfg, tokens, question_kind)
            scores[route_id] = score
            route_debug[route_id] = sd

        # 3) Select top routes
        chosen_routes = self._select_routes(scores, max_routes=max_routes)

        decision_id = str(uuid.uuid4())
        decision = RoutingDecision(
            decision_id=decision_id,
            chosen_routes=chosen_routes,
            scores=scores,
            normalized_text=norm_text,
            tokens=tokens,
            debug={
                "question_kind": question_kind,
                "route_debug": route_debug,
            },
        )

        # 4) Log decision
        self._log_decision(question, metadata, decision)

        return decision

    def apply_feedback(self, feedback: RoutingFeedback) -> None:
        """
        Apply feedback for a specific route decision.

        This updates the learned bias for route_id and persists state.
        """
        route_id = feedback.route_id
        verdict = feedback.verdict.lower()

        if route_id not in self.routes_config:
            logger.warning("Feedback for unknown route_id=%s; ignoring", route_id)
            return

        reward = _verdict_to_reward(verdict)
        stats = self.state.routes.get(route_id, RouteStats())

        stats.n += 1
        stats.total_reward += reward
        stats.avg_reward = stats.total_reward / max(1, stats.n)
        stats.bias = _reward_to_bias(stats.avg_reward)
        stats.last_verdict = verdict

        self.state.routes[route_id] = stats
        self._save_state()

        # Also log feedback for audit
        rec = {
            "kind": "routing_feedback",
            "ts": _now_iso(),
            "decision_id": feedback.decision_id,
            "route_id": route_id,
            "verdict": verdict,
            "reward": reward,
            "route_stats": asdict(stats),
            "metadata": feedback.metadata,
        }
        _append_jsonl(ROUTING_DECISIONS_LOG_PATH, rec)

    def get_route_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current stats for all routes."""
        return {rid: asdict(stats) for rid, stats in self.state.routes.items()}

    def get_route_configs(self) -> List[Dict[str, Any]]:
        """Get configuration for all routes."""
        return [asdict(cfg) for cfg in self.routes_config.values()]

    def add_route(self, config: RouteConfig) -> None:
        """Add or update a route configuration."""
        self.routes_config[config.id] = config

    # -------------------------------------------------------------------------
    # Scoring & selection
    # -------------------------------------------------------------------------

    def _score_route(
        self,
        cfg: RouteConfig,
        tokens: List[str],
        question_kind: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute a score for a given route based on:

        - base_priority (0–1)
        - learned bias from RouteStats
        - tag similarity (Jaccard) between tokens and route tags
        - suitability for question_kind
        """
        # Start with base + bias
        stats = self.state.routes.get(cfg.id, RouteStats())
        base = max(0.0, min(1.0, cfg.base_priority))
        bias = stats.bias
        raw_base = base + bias

        # Kind suitability
        kind_factor = 1.0
        if cfg.allowed_kinds:
            if question_kind and question_kind in [k.lower() for k in cfg.allowed_kinds]:
                kind_factor = 1.0
            else:
                # If kinds are specified and this one doesn't match, downweight heavily
                kind_factor = 0.3

        # Disallowed prefixes
        for prefix in cfg.disallowed_prefixes:
            if tokens and " ".join(tokens).startswith(prefix.lower()):
                kind_factor *= 0.1

        # Tag similarity
        tag_tokens = []
        for tag in cfg.tags:
            tag_tokens.extend(_normalize_tokens(tag))
        sim = _jaccard(tokens, tag_tokens)

        # Combine: base + similarity bonus, modulated by kind factor
        score = raw_base
        score += 0.4 * sim  # Similarity bonus
        score *= kind_factor

        score = max(0.0, min(1.0, score))

        debug = {
            "base_priority": base,
            "bias": bias,
            "raw_base": raw_base,
            "kind_factor": kind_factor,
            "similarity": sim,
            "n_feedback": stats.n,
        }
        return score, debug

    def _select_routes(self, scores: Dict[str, float], max_routes: int = 3) -> List[str]:
        """
        Choose final routes based on scores.

        Strategy:
          - Sort routes by score descending.
          - Take top routes with score > threshold.
          - If all scores are below threshold, pick 'language' as a safe fallback.
        """
        threshold = 0.2
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen: List[str] = []

        for route_id, s in items:
            if s < threshold:
                continue
            if len(chosen) >= max_routes:
                break
            chosen.append(route_id)

        if not chosen:
            # Safe fallback
            if "language" in self.routes_config:
                chosen = ["language"]
            elif items:
                chosen = [items[0][0]]
            else:
                chosen = []

        return chosen

    # -------------------------------------------------------------------------
    # State load/save
    # -------------------------------------------------------------------------

    def _load_state(self) -> None:
        data = _load_json(ROUTING_STATE_PATH)
        if not data:
            self.state = RoutingState()
            return

        try:
            routes_data = data.get("routes", {})
            meta = data.get("meta", {"version": 1})
            routes: Dict[str, RouteStats] = {}
            for route_id, rs in routes_data.items():
                routes[route_id] = RouteStats(
                    n=int(rs.get("n", 0)),
                    total_reward=float(rs.get("total_reward", 0.0)),
                    avg_reward=float(rs.get("avg_reward", 0.0)),
                    bias=float(rs.get("bias", 0.0)),
                    last_verdict=rs.get("last_verdict"),
                )
            self.state = RoutingState(routes=routes, meta=meta)
        except Exception as e:
            logger.error("Failed to parse routing state; starting fresh: %s", e)
            self.state = RoutingState()

    def _save_state(self) -> None:
        data = {
            "routes": {rid: asdict(stats) for rid, stats in self.state.routes.items()},
            "meta": self.state.meta,
        }
        _save_json(ROUTING_STATE_PATH, data)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_decision(
        self,
        question: str,
        metadata: Dict[str, Any],
        decision: RoutingDecision,
    ) -> None:
        rec = {
            "kind": "routing_decision",
            "ts": _now_iso(),
            "decision_id": decision.decision_id,
            "question": question[:500],  # Truncate long questions
            "metadata": metadata,
            "chosen_routes": decision.chosen_routes,
            "scores": decision.scores,
            "normalized_text": decision.normalized_text[:200],
            "tokens": decision.tokens[:50],
            "debug": decision.debug,
        }
        _append_jsonl(ROUTING_DECISIONS_LOG_PATH, rec)

    # -------------------------------------------------------------------------
    # Default route config
    # -------------------------------------------------------------------------

    @staticmethod
    def _default_routes() -> List[RouteConfig]:
        """
        Built-in default routes for Maven.

        You/Claude can override or extend this by passing routes_config explicitly.
        """
        return [
            RouteConfig(
                id="language",
                label="General Language/Reasoning Brain",
                tags=["explanation", "definition", "summarize", "describe", "explain", "what", "why", "how"],
                base_priority=0.7,
                allowed_kinds=["direct_question", "small_talk", "explanation"],
            ),
            RouteConfig(
                id="technology",
                label="Technical / Programming Brain",
                tags=["code", "python", "javascript", "api", "implementation", "bug", "stack", "trace", "error", "function", "class"],
                base_priority=0.7,
                allowed_kinds=["direct_question", "task_request"],
            ),
            RouteConfig(
                id="factual",
                label="Factual / Knowledge Brain",
                tags=["fact", "who", "what", "when", "where", "history", "science", "geography", "math"],
                base_priority=0.6,
                allowed_kinds=["direct_question"],
            ),
            RouteConfig(
                id="research_reports",
                label="Research / Reports Brain",
                tags=["research", "report", "sources", "web", "evidence", "investigate", "find", "search"],
                base_priority=0.5,
                allowed_kinds=["direct_question", "task_request"],
            ),
            RouteConfig(
                id="fs_tool",
                label="Filesystem Tool Brain",
                tags=["file", "directory", "folder", "path", "sandbox", "write", "read", "create", "delete", "filesystem"],
                base_priority=0.6,
                allowed_kinds=["task_request"],
            ),
            RouteConfig(
                id="git_tool",
                label="Git Tool Brain",
                tags=["git", "commit", "branch", "merge", "status", "diff", "push", "pull", "repository"],
                base_priority=0.5,
                allowed_kinds=["task_request"],
            ),
            RouteConfig(
                id="self_diag",
                label="Self-Diagnostics Brain",
                tags=["scan", "self", "self_scan", "scan_memory", "health", "status", "diagnostic", "memory"],
                base_priority=0.5,
                allowed_kinds=["self_diag", "task_request"],
            ),
            RouteConfig(
                id="pattern_coder",
                label="Pattern-based Code Generation",
                tags=["template", "pattern", "boilerplate", "skeleton", "scaffold", "generate"],
                base_priority=0.4,
                allowed_kinds=["task_request"],
            ),
        ]


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Default router instance (lazy initialization)
_default_router: Optional[RoutingBrain] = None


def _get_router() -> RoutingBrain:
    """Get or create default router instance."""
    global _default_router
    if _default_router is None:
        _default_router = RoutingBrain()
    return _default_router


def get_default_routing_brain() -> RoutingBrain:
    """Get the default routing brain instance."""
    return _get_router()


def route_question(question: str, metadata: Optional[Dict[str, Any]] = None) -> RoutingDecision:
    """Module-level function to route a question."""
    return _get_router().route(question, metadata)


def apply_routing_feedback(
    decision_id: Optional[str],
    route_id: str,
    verdict: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Module-level function to apply feedback."""
    feedback = RoutingFeedback(
        decision_id=decision_id,
        route_id=route_id,
        verdict=verdict,
        metadata=metadata or {},
    )
    _get_router().apply_feedback(feedback)


def get_all_route_stats() -> Dict[str, Dict[str, Any]]:
    """Module-level function to get all route stats."""
    return _get_router().get_route_stats()


def list_recent_decisions(limit: int = 50, max_age_days: int = 7) -> List[Dict[str, Any]]:
    """Read recent routing decisions from the log."""
    if not ROUTING_DECISIONS_LOG_PATH.exists():
        return []

    decisions: List[Dict[str, Any]] = []
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    try:
        with ROUTING_DECISIONS_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") != "routing_decision":
                    continue

                ts_str = rec.get("ts")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                except Exception:
                    ts = None

                if ts is not None and ts < cutoff:
                    continue

                decisions.append(rec)
    except Exception as e:
        logger.error("Failed to read routing decisions log: %s", e)
        return []

    # Most recent first
    decisions.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return decisions[:limit]


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for routing brain.

    Supported operations:
    - ROUTE: Route a question
    - FEEDBACK: Apply feedback for a routing decision
    - GET_STATS: Get route statistics
    - LIST_ROUTES: List configured routes
    - LIST_DECISIONS: List recent routing decisions
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    router = _get_router()

    if op == "ROUTE":
        try:
            question = payload.get("question", "")
            metadata = payload.get("metadata", {})

            if not question:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_QUESTION", "message": "question required"},
                }

            decision = router.route(question, metadata)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": decision.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "ROUTE_FAILED", "message": str(e)},
            }

    if op == "FEEDBACK":
        try:
            decision_id = payload.get("decision_id")
            route_id = payload.get("route_id")
            verdict = payload.get("verdict")
            metadata = payload.get("metadata", {})

            if not route_id or not verdict:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_FIELDS", "message": "route_id and verdict required"},
                }

            feedback = RoutingFeedback(
                decision_id=decision_id,
                route_id=route_id,
                verdict=verdict,
                metadata=metadata,
            )
            router.apply_feedback(feedback)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"message": "Feedback applied", "route_id": route_id, "verdict": verdict},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "FEEDBACK_FAILED", "message": str(e)},
            }

    if op == "GET_STATS":
        try:
            stats = router.get_route_stats()
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"stats": stats},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "STATS_FAILED", "message": str(e)},
            }

    if op == "LIST_ROUTES":
        try:
            routes = router.get_route_configs()
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"routes": routes, "count": len(routes)},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LIST_FAILED", "message": str(e)},
            }

    if op == "LIST_DECISIONS":
        try:
            limit = int(payload.get("limit", 50))
            max_age_days = int(payload.get("max_age_days", 7))
            decisions = list_recent_decisions(limit=limit, max_age_days=max_age_days)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"decisions": decisions, "count": len(decisions)},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LIST_FAILED", "message": str(e)},
            }

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "routing_brain",
                "route_count": len(router.routes_config),
                "state_path": str(ROUTING_STATE_PATH),
                "log_path": str(ROUTING_DECISIONS_LOG_PATH),
                "available_operations": ["ROUTE", "FEEDBACK", "GET_STATS", "LIST_ROUTES", "LIST_DECISIONS", "HEALTH"],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
