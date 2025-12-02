"""Utility to display LLM learning statistics and template usage.

Running this module prints a dashboard summarising how many
interactions have been logged, how many templates have been learned,
and how often templates are being used compared to fresh LLM calls.
It is intended for debugging and monitoring and uses only the
standard library.
"""
from __future__ import annotations

import json
from pathlib import Path
from brains.maven_paths import get_brains_path

# This monitor reads data files directly from the learned_patterns
# directory rather than importing the LLM service.  This avoids
# requiring the ``brains`` directory to be a package and keeps
# dependencies minimal.  All paths are resolved relative to the
# project root using the Maven path helper.
PATTERNS_DIR = get_brains_path("personal", "memory", "learned_patterns")


def _load_json(path: Path) -> dict:
    """Load JSON from a file path, returning an empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore
    except Exception:
        return {}


def _load_interactions(path: Path) -> list:
    """Load a list of interaction dicts from a JSONL file."""
    if not path.exists():
        return []
    interactions: list = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    interactions.append(json.loads(ln))  # type: ignore
                except Exception:
                    continue
    except Exception:
        pass
    return interactions


def print_learning_dashboard() -> None:
    """Print a summary of learning statistics to stdout.

    This function reads the learned patterns directory directly,
    computing the total number of interactions, templates and
    template usage without importing the LLM service.  If no
    patterns directory exists yet (e.g., before any LLM usage), it
    displays zeros accordingly.
    """
    stats_path = PATTERNS_DIR / "pattern_stats.json"
    templates_path = PATTERNS_DIR / "learned_templates.json"
    interactions_path = PATTERNS_DIR / "llm_interactions.jsonl"
    stats = _load_json(stats_path)
    templates = _load_json(templates_path)
    interactions = _load_interactions(interactions_path)
    total_interactions = len(interactions)
    total_templates = len(templates)
    print("\n" + "=" * 60)
    print("MAVEN LEARNING DASHBOARD")
    print("=" * 60)
    print(f"\n📊 Overall Stats:")
    print(f"  Total interactions: {total_interactions}")
    print(f"  Learned templates: {total_templates}")
    # Compute independence: template hits divided by interactions
    if total_interactions > 0:
        try:
            template_hits = sum(t.get("use_count", 0) for t in templates.values())
        except Exception:
            template_hits = 0
        independence = (template_hits / total_interactions) * 100.0
        llm_usage = 100.0 - independence
        print(f"\n🎯 LLM Independence:")
        print(f"  Template usage: {independence:.1f}%")
        print(f"  LLM usage: {llm_usage:.1f}%")
        print(f"  Goal: 90% template, 10% LLM")
        bars = int(independence / 10.0)
        bar_str = "█" * bars + "░" * (10 - bars)
        print(f"\n  [{bar_str}] {independence:.0f}%")
    else:
        independence = 0.0
        print(f"\n🎯 LLM Independence:")
        print(f"  Template usage: 0.0%")
        print(f"  LLM usage: 100.0%")
        print(f"  Goal: 90% template, 10% LLM")
        print(f"\n  [░░░░░░░░░░] 0%")
    print(f"\n📈 Most Used Templates:")
    # Sort templates by use count descending
    try:
        sorted_t = sorted(templates.items(), key=lambda x: x[1].get("use_count", 0), reverse=True)
    except Exception:
        sorted_t = []
    for idx, (h, tpl) in enumerate(sorted_t[:5]):
        uses = tpl.get("use_count", 0)
        pattern = str(tpl.get("prompt_pattern", ""))[:40]
        print(f"  {idx + 1}. {pattern}... ({uses} uses)")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print_learning_dashboard()