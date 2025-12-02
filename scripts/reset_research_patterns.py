"""One-off helper to clear learned research_manager patterns.

This preserves other brains' patterns while removing any entries
associated with the research_manager brain.
"""

from __future__ import annotations

from brains.cognitive.pattern_store import PatternStore, get_pattern_store


def _filter_out_research_patterns(store: PatternStore) -> int:
    """Remove research_manager patterns from the shared pattern store."""

    try:
        all_patterns = store._load_patterns()  # type: ignore[attr-defined]
    except Exception:
        all_patterns = []

    before_count = len(all_patterns)
    remaining = [p for p in all_patterns if p.brain != "research_manager"]
    removed = before_count - len(remaining)

    try:
        store._save_patterns(remaining)  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[RESET_RESEARCH_PATTERNS] Failed to save filtered patterns: {e}")
        return 0

    return removed


def main() -> None:
    store = get_pattern_store()
    removed = _filter_out_research_patterns(store)
    print(f"[RESET_RESEARCH_PATTERNS] Removed {removed} research_manager pattern(s)")


if __name__ == "__main__":
    main()
