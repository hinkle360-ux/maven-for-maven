"""
Input Handler for Maven
=======================

Provides enhanced input capabilities:
1. File upload/attachment via @file or /file commands
2. Multi-line paste support (triple backticks or /paste mode)
3. URL content fetching via @url command
4. Clipboard access (when available)

Usage in chat:
    @file path/to/document.txt     - Attach file contents
    @file /path/to/code.py         - Attach code file
    /paste                          - Enter multi-line mode (end with /end)
    ```                             - Start/end code block (auto-detected)
    @url https://example.com        - Fetch and include URL content

The handler processes raw user input and returns structured content
that includes any attachments or expanded content.
"""

from __future__ import annotations

import os
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Supported file extensions and their types
FILE_TYPES = {
    # Code files
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".r": "r",
    ".m": "matlab",
    ".lua": "lua",
    ".pl": "perl",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    ".lisp": "lisp",
    ".vim": "vim",
    ".el": "elisp",

    # Config/Data files
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "config",
    ".conf": "config",
    ".env": "env",
    ".properties": "properties",

    # Document files
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "restructuredtext",
    ".tex": "latex",
    ".csv": "csv",
    ".tsv": "tsv",

    # Web files
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",

    # Binary/special (read as base64 or skip)
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".svg": "svg",
    ".pdf": "pdf",
}

# Maximum file size to read (500 MB - as big as reasonable for text processing)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Maximum lines for text files
MAX_LINES = 100000


@dataclass
class Attachment:
    """Represents an attached file or content."""
    name: str
    content: str
    file_type: str
    size: int
    is_binary: bool = False
    encoding: str = "utf-8"
    line_count: Optional[int] = None
    truncated: bool = False
    error: Optional[str] = None


@dataclass
class ProcessedInput:
    """Result of processing user input."""
    text: str  # The main text query (without attachment commands)
    attachments: List[Attachment] = field(default_factory=list)
    is_multiline: bool = False
    raw_input: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_file_type(path: Path) -> str:
    """Detect file type from extension."""
    suffix = path.suffix.lower()
    return FILE_TYPES.get(suffix, "text")


def is_binary_file(path: Path) -> bool:
    """Check if a file is binary."""
    binary_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip",
                        ".tar", ".gz", ".bz2", ".exe", ".dll", ".so", ".dylib"}
    return path.suffix.lower() in binary_extensions


def read_file_safe(path: Path, max_size: int = MAX_FILE_SIZE) -> Attachment:
    """
    Safely read a file and return an Attachment.

    Handles:
    - Size limits
    - Binary vs text detection
    - Encoding issues
    - Line count limits
    """
    name = path.name
    file_type = detect_file_type(path)

    if not path.exists():
        return Attachment(
            name=name,
            content="",
            file_type=file_type,
            size=0,
            error=f"File not found: {path}"
        )

    try:
        size = path.stat().st_size

        if size > max_size:
            return Attachment(
                name=name,
                content="",
                file_type=file_type,
                size=size,
                error=f"File too large: {size} bytes (max {max_size})"
            )

        if is_binary_file(path):
            # Read binary files as base64
            content = base64.b64encode(path.read_bytes()).decode("ascii")
            return Attachment(
                name=name,
                content=content,
                file_type=file_type,
                size=size,
                is_binary=True,
                encoding="base64"
            )

        # Try to read as text with various encodings
        content = None
        encoding_used = "utf-8"

        for encoding in ["utf-8", "utf-16", "latin-1", "cp1252"]:
            try:
                content = path.read_text(encoding=encoding)
                encoding_used = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            # Fall back to binary reading
            content = base64.b64encode(path.read_bytes()).decode("ascii")
            return Attachment(
                name=name,
                content=content,
                file_type=file_type,
                size=size,
                is_binary=True,
                encoding="base64",
                error="Could not decode as text, stored as base64"
            )

        # Count lines and potentially truncate
        lines = content.split("\n")
        line_count = len(lines)
        truncated = False

        if line_count > MAX_LINES:
            lines = lines[:MAX_LINES]
            lines.append(f"\n... (truncated, {line_count - MAX_LINES} more lines)")
            content = "\n".join(lines)
            truncated = True

        return Attachment(
            name=name,
            content=content,
            file_type=file_type,
            size=size,
            is_binary=False,
            encoding=encoding_used,
            line_count=line_count,
            truncated=truncated
        )

    except PermissionError:
        return Attachment(
            name=name,
            content="",
            file_type=file_type,
            size=0,
            error=f"Permission denied: {path}"
        )
    except Exception as e:
        return Attachment(
            name=name,
            content="",
            file_type=file_type,
            size=0,
            error=f"Error reading file: {str(e)}"
        )


def expand_path(path_str: str) -> Path:
    """Expand user paths and environment variables."""
    # Expand ~ to home directory
    path_str = os.path.expanduser(path_str)
    # Expand environment variables
    path_str = os.path.expandvars(path_str)
    return Path(path_str)


def extract_file_commands(text: str) -> Tuple[str, List[str]]:
    """
    Extract @file commands from text.

    Returns:
        (remaining_text, list_of_file_paths)
    """
    # Pattern: @file followed by path (quoted or unquoted)
    # Supports: @file path.txt, @file "path with spaces.txt", @file 'path.txt'
    pattern = r'@file\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))'

    matches = re.findall(pattern, text, re.IGNORECASE)
    file_paths = []

    for match in matches:
        # match is a tuple of (quoted_double, quoted_single, unquoted)
        path = match[0] or match[1] or match[2]
        if path:
            file_paths.append(path)

    # Remove the @file commands from text
    remaining = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return remaining, file_paths


def extract_url_commands(text: str) -> Tuple[str, List[str]]:
    """
    Extract @url commands from text.

    Returns:
        (remaining_text, list_of_urls)
    """
    pattern = r'@url\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))'

    matches = re.findall(pattern, text, re.IGNORECASE)
    urls = []

    for match in matches:
        url = match[0] or match[1] or match[2]
        if url:
            urls.append(url)

    remaining = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return remaining, urls


def detect_pasted_code(text: str) -> Optional[str]:
    """
    Detect if text looks like code that was pasted with newlines stripped.
    If detected, attempt to restore newlines.

    Common patterns when code is pasted without newlines:
    - "import asynciofrom pathlib" -> "import asyncio\nfrom pathlib"
    - "def foo():    return" -> "def foo():\n    return"
    - Multiple statements concatenated without separation
    """
    if not text or len(text) < 50:
        return None

    # Check if it looks like concatenated Python code
    python_indicators = [
        r'import\s+\w+from\s+',       # import Xfrom
        r'from\s+\w+import\s+',        # from X import (normal, but check context)
        r'def\s+\w+\([^)]*\):\s*\w',   # def foo(): directly followed by code
        r'class\s+\w+.*:\s*def',       # class X: def (no newline)
        r':\s*return\s',               # :return (missing newline after :)
        r':\s*if\s',                   # :if
        r':\s*for\s',                  # :for
        r':\s*while\s',                # :while
        r':\s*try:',                   # :try:
        r'#[^#\n]+#',                  # Two comments merged
    ]

    import re
    for pattern in python_indicators:
        if re.search(pattern, text):
            # Looks like pasted code - try to restore newlines
            return restore_python_newlines(text)

    # Check for very long lines without any newlines (likely pasted)
    if '\n' not in text and len(text) > 200:
        # Try to detect if it's code by looking for keywords
        code_keywords = ['import', 'def ', 'class ', 'return ', 'if ', 'for ', 'while ']
        keyword_count = sum(1 for kw in code_keywords if kw in text.lower())
        if keyword_count >= 2:
            return restore_python_newlines(text)

    return None


def restore_python_newlines(text: str) -> str:
    """
    Attempt to restore newlines in Python code that had them stripped.

    This uses heuristics to insert newlines at likely positions.
    """
    import re

    # Already has newlines? Don't process
    if '\n' in text and text.count('\n') > 5:
        return text

    result = text

    # Pattern 1: Xfrom Y import -> X\nfrom Y import (catches pathimport, jsonimport, etc)
    result = re.sub(r'(\w)(from\s+[\w.]+\s+import)', r'\1\n\2', result)

    # Pattern 2: Ximport Y -> X\nimport Y (catches json# commentimport, etc)
    result = re.sub(r'([a-z\"\'\)\]])(\s*import\s+)', r'\1\n\2', result)

    # Pattern 3: After closing paren followed by def/class/async
    result = re.sub(r'(\))(async\s+def\s+|def\s+|class\s+)', r'\1\n\n\2', result)

    # Pattern 4: After comments followed by async def/def/class/import/from
    result = re.sub(r'(\)[^\n]*)(\s*#[^#\n]+)(async\s+def\s+|def\s+|class\s+|import\s+|from\s+)', r'\1\2\n\3', result)
    result = re.sub(r'(#[^\n]+)(async\s+def\s+|def\s+|class\s+|import\s+|from\s+)', r'\1\n\2', result)

    # Pattern 5: Word directly followed by async def (like "locally)async def")
    result = re.sub(r'(\w)\s*(async\s+def\s+)', r'\1\n\n\2', result)

    # Pattern 6: After colon + content followed by keywords
    result = re.sub(r'(:\s*)([^:\n]+)(return\s+|if\s+|for\s+|while\s+|try:|except|finally:|else:|elif\s+)', r'\1\2\n    \3', result)

    # Pattern 7: Split statements after common endings
    result = re.sub(r'(\)\s*)(Path\(|result\s*=|return\s+)', r'\1\n    \2', result)

    # Pattern 8: After string literal followed by keywords
    result = re.sub(r'([\"\'])\s*(import\s+|from\s+|def\s+|class\s+|async\s+)', r'\1\n\2', result)

    # Pattern 9: Handle "    #" at various indentation levels - these are comments that need newlines before
    result = re.sub(r'(\w)(    #)', r'\1\n\2', result)

    # Pattern 10: Multiple consecutive spaces often indicate original indentation
    result = re.sub(r'(\S)(    )(\w)', r'\1\n\2\3', result)

    return result


def is_multiline_start(text: str) -> bool:
    """Check if input starts multiline mode."""
    triggers = ["/paste", "/multi", "```"]
    return text.strip().lower() in triggers or text.strip().startswith("```")


def collect_multiline_input(initial: str = "") -> str:
    """
    Collect multi-line input from user.

    Ends when user types:
    - /end
    - ``` (if started with ```)
    - Two consecutive empty lines
    """
    lines = []
    if initial and initial not in ["/paste", "/multi"]:
        # Started with ```, include language hint if present
        if initial.startswith("```"):
            lang = initial[3:].strip()
            if lang:
                lines.append(f"# Language: {lang}")

    print("[Multi-line mode: type /end or ``` to finish, or press Enter twice]")

    empty_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break

        # Check for end markers
        if line.strip().lower() == "/end" or line.strip() == "```":
            break

        # Check for double empty line
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0

        lines.append(line)

    return "\n".join(lines)


def process_input(raw_input: str, allow_multiline: bool = True) -> ProcessedInput:
    """
    Process raw user input and extract files, URLs, and handle multi-line.

    Args:
        raw_input: The raw string from user input
        allow_multiline: Whether to allow multi-line input mode

    Returns:
        ProcessedInput with text, attachments, and metadata
    """
    result = ProcessedInput(
        text="",
        raw_input=raw_input
    )

    text = raw_input.strip()

    # Check for multi-line mode
    if allow_multiline and is_multiline_start(text):
        text = collect_multiline_input(text)
        result.is_multiline = True

    # Check for pasted code that had newlines stripped
    restored = detect_pasted_code(text)
    if restored and restored != text:
        text = restored
        result.is_multiline = True
        result.metadata["paste_restored"] = True

    # Extract file commands
    text, file_paths = extract_file_commands(text)

    # Process file attachments
    for path_str in file_paths:
        path = expand_path(path_str)
        attachment = read_file_safe(path)
        result.attachments.append(attachment)

    # Extract URL commands (mark for later fetching by web tool)
    text, urls = extract_url_commands(text)
    if urls:
        result.metadata["urls_to_fetch"] = urls

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    result.text = text

    return result


def format_attachments_for_context(attachments: List[Attachment]) -> str:
    """
    Format attachments as context for the query.

    Returns a string that can be prepended to the user's query.
    """
    if not attachments:
        return ""

    parts = []

    for att in attachments:
        if att.error:
            parts.append(f"[Attachment Error: {att.name}] {att.error}")
        elif att.is_binary:
            parts.append(f"[Binary Attachment: {att.name}] ({att.file_type}, {att.size} bytes, base64 encoded)")
        else:
            header = f"[Attachment: {att.name}]"
            if att.truncated:
                header += " (truncated)"
            if att.line_count:
                header += f" ({att.line_count} lines)"

            # Format code files with markdown code blocks
            if att.file_type in ["python", "javascript", "typescript", "java", "c",
                                 "cpp", "rust", "go", "ruby", "php", "shell", "sql"]:
                parts.append(f"{header}\n```{att.file_type}\n{att.content}\n```")
            else:
                parts.append(f"{header}\n{att.content}")

    return "\n\n".join(parts)


def enhanced_input(prompt: str = "> ") -> ProcessedInput:
    """
    Enhanced input function that replaces standard input().

    Supports:
    - @file path/to/file - attach file
    - /paste - multi-line mode
    - ``` - code block mode

    Returns ProcessedInput with text and any attachments.
    """
    try:
        raw = input(prompt)
    except EOFError:
        return ProcessedInput(text="exit", raw_input="")

    return process_input(raw)


# Convenience function for direct use
def get_input_with_attachments(prompt: str = "> ") -> Tuple[str, List[Attachment]]:
    """
    Get user input with any file attachments.

    Returns:
        (query_text, list_of_attachments)
    """
    result = enhanced_input(prompt)
    return result.text, result.attachments


# CLI test
if __name__ == "__main__":
    print("Maven Input Handler Test")
    print("Commands: @file <path>, /paste, ```, /exit")
    print("-" * 40)

    while True:
        result = enhanced_input("test> ")

        if result.text.lower() in ["exit", "quit", "/exit"]:
            print("Goodbye!")
            break

        print(f"\nProcessed Input:")
        print(f"  Text: {result.text}")
        print(f"  Multi-line: {result.is_multiline}")
        print(f"  Attachments: {len(result.attachments)}")

        for att in result.attachments:
            print(f"    - {att.name} ({att.file_type}, {att.size} bytes)")
            if att.error:
                print(f"      Error: {att.error}")
            elif not att.is_binary:
                preview = att.content[:100] + "..." if len(att.content) > 100 else att.content
                print(f"      Preview: {preview}")

        if result.metadata.get("urls_to_fetch"):
            print(f"  URLs to fetch: {result.metadata['urls_to_fetch']}")

        print()
