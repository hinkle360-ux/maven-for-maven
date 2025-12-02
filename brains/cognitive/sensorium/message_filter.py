"""Utilities for filtering out separator/comment-only messages before pipeline run."""

from __future__ import annotations

SEPARATOR_CHARS = set("-_=*•·—–# ")


def is_visual_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    if stripped.startswith("#"):
        core = stripped[1:].strip()
    else:
        core = stripped

    if not core:
        return True

    return all(ch in SEPARATOR_CHARS for ch in core)


def is_meta_or_comment_message(text: str) -> bool:
    """Return True when the incoming text is only separators or meta markers."""

    if not text or not str(text).strip():
        return True

    lines = [line for line in str(text).splitlines() if line.strip()]

    if all(is_visual_separator(line) for line in lines):
        return True

    meta_prefixes = (
        "# VERIFY ",
        "# TIME-BUDGETED",
        "# DEEP RESEARCH",
        "# ------------",
    )
    stripped_first = lines[0].strip()
    if stripped_first.startswith(meta_prefixes):
        if len(lines) == 1 or all(is_visual_separator(line) for line in lines[1:]):
            return True

    return False
