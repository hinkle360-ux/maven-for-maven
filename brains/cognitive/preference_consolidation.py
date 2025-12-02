"""
Phase 5D: Long-Term Preference Consolidation

This module provides deterministic preference consolidation logic for Maven's
continuous learning system. It consolidates repeated preferences, detects
conflicts, and manages preference evolution over time.

Key operations:
- Consolidate repeated preferences into canonical forms
- Detect and store preference conflicts
- Track preference stability and importance
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path


def consolidate_preferences(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate repeated preferences into canonical forms.

    If user says multiple times:
    - "I like green"
    - "green is my favorite color"
    - "I prefer green objects"
    - "my room is green"

    Consolidate into:
    {
        "type": "PREFERENCE",
        "canonical": "user_likes_green",
        "tier": "MID",
        "importance": 1.0,
        "evidence_count": 4,
        "sources": [list of seq_ids]
    }

    Args:
        records: List of preference records

    Returns:
        List of consolidated preference records.
    """
    if not records:
        return []

    # Group preferences by subject
    preference_groups: Dict[str, List[Dict]] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue

        content = str(rec.get("content", "")).lower()
        verdict = str(rec.get("verdict", "")).upper()
        tags = rec.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags_set = {str(t).lower() for t in tags}

        # Only process preference records
        if verdict != "PREFERENCE" and "preference" not in tags_set:
            continue

        # Extract preference subject
        subject = _extract_preference_subject(content)
        if not subject:
            continue

        if subject not in preference_groups:
            preference_groups[subject] = []

        preference_groups[subject].append(rec)

    # Consolidate each group
    consolidated = []

    for subject, group in preference_groups.items():
        if len(group) < 2:
            # Not enough evidence to consolidate
            continue

        # Calculate importance based on repetition
        evidence_count = len(group)
        importance = min(1.0, 0.7 + (evidence_count * 0.1))

        # Extract source seq_ids
        sources = [rec.get("seq_id") for rec in group if "seq_id" in rec]

        # Determine sentiment (like/dislike)
        sentiment = _determine_sentiment(group)

        canonical_pref = {
            "type": "PREFERENCE",
            "canonical": f"user_{sentiment}_{subject}",
            "subject": subject,
            "sentiment": sentiment,
            "tier": "MID",
            "importance": importance,
            "evidence_count": evidence_count,
            "sources": sources,
            "confidence": min(1.0, 0.8 + (evidence_count * 0.05))
        }

        consolidated.append(canonical_pref)

    return consolidated


def detect_conflicts(preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect preference conflicts.

    If contradictions exist:
    - "I like cats"
    - "I don't like cats"

    Store a CONFLICT record and activate reasoning brain when answering.

    Args:
        preferences: List of preference records

    Returns:
        List of conflict records.
    """
    if not preferences:
        return []

    conflicts = []

    # Group by subject
    subject_map: Dict[str, List[Dict]] = {}

    for pref in preferences:
        if not isinstance(pref, dict):
            continue

        subject = pref.get("subject", "")
        if not subject:
            continue

        if subject not in subject_map:
            subject_map[subject] = []

        subject_map[subject].append(pref)

    # Detect conflicts within each subject
    for subject, prefs in subject_map.items():
        sentiments = {p.get("sentiment") for p in prefs}

        # Conflict: both positive and negative sentiments
        if "likes" in sentiments and "dislikes" in sentiments:
            conflict = {
                "type": "CONFLICT",
                "subject": subject,
                "conflicting_preferences": prefs,
                "tier": "MID",
                "importance": 0.9,
                "resolution_strategy": "present_both_ask_user"
            }

            conflicts.append(conflict)

    return conflicts


def _extract_preference_subject(content: str) -> str:
    """
    Extract the subject of a preference statement.

    Examples:
    - "I like green" → "green"
    - "green is my favorite color" → "green"
    - "I prefer green objects" → "green"

    Args:
        content: Preference statement text

    Returns:
        Subject string or empty string.
    """
    content = content.lower().strip()

    # Pattern 1: "I like X"
    if "like" in content:
        parts = content.split("like")
        if len(parts) > 1:
            subject = parts[1].strip().split()[0] if parts[1].strip().split() else ""
            return subject

    # Pattern 2: "X is my favorite"
    if "favorite" in content:
        parts = content.split("is")
        if len(parts) > 1:
            subject = parts[0].strip().split()[-1] if parts[0].strip().split() else ""
            return subject

    # Pattern 3: "I prefer X"
    if "prefer" in content:
        parts = content.split("prefer")
        if len(parts) > 1:
            subject = parts[1].strip().split()[0] if parts[1].strip().split() else ""
            return subject

    # Pattern 4: "my X is Y" (e.g., "my room is green")
    if "my" in content and "is" in content:
        parts = content.split("is")
        if len(parts) > 1:
            subject = parts[1].strip().split()[0] if parts[1].strip().split() else ""
            return subject

    return ""


def _determine_sentiment(preferences: List[Dict[str, Any]]) -> str:
    """
    Determine overall sentiment from a list of preferences.

    Args:
        preferences: List of preference records

    Returns:
        "likes" or "dislikes" based on majority.
    """
    like_count = 0
    dislike_count = 0

    for pref in preferences:
        content = str(pref.get("content", "")).lower()

        if "like" in content and "don't like" not in content and "do not like" not in content:
            like_count += 1
        elif "don't like" in content or "do not like" in content or "dislike" in content:
            dislike_count += 1
        elif "favorite" in content or "love" in content:
            like_count += 1
        elif "hate" in content:
            dislike_count += 1

    return "likes" if like_count >= dislike_count else "dislikes"


def merge_duplicate_preferences(preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge duplicate preference records.

    Args:
        preferences: List of preference records (possibly with duplicates)

    Returns:
        Deduplicated list with merged evidence counts.
    """
    if not preferences:
        return []

    # Group by canonical name
    canonical_map: Dict[str, Dict[str, Any]] = {}

    for pref in preferences:
        if not isinstance(pref, dict):
            continue

        canonical = pref.get("canonical", "")
        if not canonical:
            continue

        if canonical not in canonical_map:
            canonical_map[canonical] = pref.copy()
        else:
            # Merge evidence
            existing = canonical_map[canonical]
            existing["evidence_count"] = existing.get("evidence_count", 0) + pref.get("evidence_count", 0)

            # Merge sources
            existing_sources = existing.get("sources", [])
            new_sources = pref.get("sources", [])
            merged_sources = list(set(existing_sources + new_sources))
            existing["sources"] = merged_sources

            # Update importance
            existing["importance"] = min(1.0, 0.7 + (existing["evidence_count"] * 0.1))
            existing["confidence"] = min(1.0, 0.8 + (existing["evidence_count"] * 0.05))

    return list(canonical_map.values())


def save_consolidated_preferences(preferences: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save consolidated preferences to a file.

    Args:
        preferences: List of consolidated preferences
        output_path: Path to save file
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pref in preferences:
                f.write(json.dumps(pref) + "\n")
    except Exception:
        pass


def load_consolidated_preferences(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load consolidated preferences from a file.

    Args:
        input_path: Path to load from

    Returns:
        List of preference records.
    """
    if not input_path.exists():
        return []

    preferences = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pref = json.loads(line.strip())
                    preferences.append(pref)
                except Exception:
                    continue
    except Exception:
        return []

    return preferences
