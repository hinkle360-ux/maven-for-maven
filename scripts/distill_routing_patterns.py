#!/usr/bin/env python3
"""
distill_routing_patterns.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distill Teacher routing suggestions into local pattern rules.

This script reads the smart routing decision log and:
1. Clusters similar queries that resulted in the same routing
2. Extracts common patterns (keywords, phrases)
3. Creates or updates pattern rules in the pattern store
4. Prunes patterns with consistently poor feedback

The goal is to gradually reduce dependency on Teacher by learning
effective local routing patterns.

Usage:
    python scripts/distill_routing_patterns.py [--days 7] [--min-count 3]

Options:
    --days: Number of days of logs to process (default: 7)
    --min-count: Minimum occurrences to create a pattern (default: 3)
    --dry-run: Show what would be done without making changes
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Paths
HOME_DIR = Path.home()
MAVEN_DIR = HOME_DIR / ".maven"
SMART_ROUTING_LOG_PATH = MAVEN_DIR / "smart_routing_decisions.jsonl"


@dataclass
class RoutingCluster:
    """A cluster of similar routing decisions."""
    intent: str
    brains: Tuple[str, ...]
    tools: Tuple[str, ...]
    examples: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    total_reward: float = 0.0
    feedback_count: int = 0
    decision_ids: List[str] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        if self.feedback_count == 0:
            return 0.0
        return self.total_reward / self.feedback_count


def load_routing_logs(days: int = 7) -> List[Dict[str, Any]]:
    """Load routing decision logs from the last N days."""
    if not SMART_ROUTING_LOG_PATH.exists():
        print(f"No routing log found at {SMART_ROUTING_LOG_PATH}")
        return []

    cutoff = datetime.utcnow() - timedelta(days=days)
    decisions = []
    feedback_by_id: Dict[str, Dict[str, Any]] = {}

    with SMART_ROUTING_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Parse timestamp
            ts_str = record.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
            except Exception:
                ts = None

            if ts and ts < cutoff:
                continue

            # Separate decisions from feedback
            if record.get("kind") == "routing_feedback":
                decision_id = record.get("decision_id")
                if decision_id:
                    feedback_by_id[decision_id] = record
            elif "decision_id" in record and "final_plan" in record:
                decisions.append(record)

    # Merge feedback into decisions
    for decision in decisions:
        decision_id = decision.get("decision_id")
        if decision_id in feedback_by_id:
            decision["feedback"] = feedback_by_id[decision_id]

    print(f"Loaded {len(decisions)} decisions from last {days} days")
    return decisions


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    # Normalize
    text_lower = text.lower()

    # Remove common stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "and", "but", "if",
        "or", "because", "until", "while", "about", "against", "this", "that",
        "these", "those", "what", "which", "who", "whom", "me", "my", "i", "you",
        "your", "he", "she", "it", "we", "they", "them", "his", "her", "its",
        "our", "their", "please", "help", "want", "like", "know", "think",
        "make", "get", "go", "see", "come", "take", "give", "find", "tell",
    }

    # Extract words
    words = re.findall(r'\b[a-z][a-z0-9_]+\b', text_lower)
    keywords = {w for w in words if w not in stopwords and len(w) > 2}

    return keywords


def cluster_decisions(decisions: List[Dict[str, Any]]) -> List[RoutingCluster]:
    """Cluster decisions by intent and routing."""
    clusters: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], RoutingCluster] = {}

    for decision in decisions:
        final_plan = decision.get("final_plan", {})
        intent = final_plan.get("intent", {}).get("primary_intent", "unknown")
        brains = tuple(sorted(final_plan.get("final_brains", [])))
        tools = tuple(sorted(final_plan.get("final_tools", [])))
        user_text = decision.get("user_text", "")
        decision_id = decision.get("decision_id", "")

        key = (intent, brains, tools)

        if key not in clusters:
            clusters[key] = RoutingCluster(
                intent=intent,
                brains=brains,
                tools=tools,
            )

        cluster = clusters[key]
        cluster.examples.append(user_text)
        cluster.keywords.update(extract_keywords(user_text))
        cluster.decision_ids.append(decision_id)

        # Add feedback if available
        feedback = decision.get("feedback", {})
        if feedback:
            reward = feedback.get("reward", 0.0)
            cluster.total_reward += reward
            cluster.feedback_count += 1

    return list(clusters.values())


def generate_pattern_rules(
    clusters: List[RoutingCluster],
    min_count: int = 3,
) -> List[Dict[str, Any]]:
    """Generate pattern rules from clusters."""
    rules = []

    for cluster in clusters:
        # Skip clusters with too few examples
        if len(cluster.examples) < min_count:
            continue

        # Skip clusters with very negative feedback
        if cluster.feedback_count > 0 and cluster.avg_reward < -0.5:
            print(f"Skipping poor-performing cluster: {cluster.intent} -> {cluster.brains} "
                  f"(avg_reward={cluster.avg_reward:.2f})")
            continue

        # Find common keywords (appear in at least 30% of examples)
        keyword_counts = defaultdict(int)
        for example in cluster.examples:
            example_keywords = extract_keywords(example)
            for kw in example_keywords:
                keyword_counts[kw] += 1

        min_occurrences = max(2, len(cluster.examples) * 0.3)
        common_keywords = [
            kw for kw, count in keyword_counts.items()
            if count >= min_occurrences
        ]

        if not common_keywords:
            continue

        # Create pattern rule
        rule = {
            "id": f"distilled_{cluster.intent}_{hash(cluster.brains) % 10000}",
            "intent": cluster.intent,
            "keywords": sorted(common_keywords)[:10],  # Top 10 keywords
            "brains": list(cluster.brains),
            "tools": list(cluster.tools),
            "example_count": len(cluster.examples),
            "avg_reward": cluster.avg_reward,
            "feedback_count": cluster.feedback_count,
            "confidence": min(0.9, 0.5 + len(cluster.examples) * 0.05),
        }
        rules.append(rule)

    return rules


def save_pattern_rules(
    rules: List[Dict[str, Any]],
    dry_run: bool = False,
) -> int:
    """Save pattern rules to the pattern store."""
    if dry_run:
        print("\n[DRY RUN] Would save the following rules:\n")
        for rule in rules:
            print(f"  Intent: {rule['intent']}")
            print(f"  Keywords: {rule['keywords']}")
            print(f"  -> Brains: {rule['brains']}")
            print(f"  -> Tools: {rule['tools']}")
            print(f"  Confidence: {rule['confidence']:.2f}")
            print(f"  Examples: {rule['example_count']}, Avg reward: {rule['avg_reward']:.2f}")
            print()
        return 0

    try:
        from brains.cognitive.pattern_store import (
            get_pattern_store,
            Pattern,
            create_pattern_id,
        )

        store = get_pattern_store()
        saved = 0

        for rule in rules:
            pattern = Pattern(
                id=create_pattern_id("smart_routing", rule["id"]),
                brain="smart_routing",
                signature=f"intent:{rule['intent']}",
                context_tags=rule["keywords"],
                action={
                    "brains": rule["brains"],
                    "tools": rule["tools"],
                    "distilled": True,
                },
                score=rule["confidence"],
            )
            store.store_pattern(pattern)
            saved += 1
            print(f"Saved pattern: {rule['intent']} -> {rule['brains']}")

        return saved

    except ImportError as e:
        print(f"Error: Could not import pattern store: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Distill Teacher routing suggestions into local patterns"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days of logs to process (default: 7)"
    )
    parser.add_argument(
        "--min-count", type=int, default=3,
        help="Minimum occurrences to create a pattern (default: 3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ROUTING PATTERN DISTILLATION")
    print("=" * 60)

    # Load logs
    decisions = load_routing_logs(days=args.days)
    if not decisions:
        print("No decisions to process")
        return

    # Cluster decisions
    clusters = cluster_decisions(decisions)
    print(f"Found {len(clusters)} unique routing clusters")

    # Generate pattern rules
    rules = generate_pattern_rules(clusters, min_count=args.min_count)
    print(f"Generated {len(rules)} pattern rules")

    if not rules:
        print("No patterns to save (increase data or lower min-count)")
        return

    # Save patterns
    saved = save_pattern_rules(rules, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nSaved {saved} patterns to pattern store")

    print("=" * 60)
    print("DISTILLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
