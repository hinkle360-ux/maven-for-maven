"""
Pattern-based Coder Brain

Purpose
-------
Provide a *real* pattern-based code generation path:

- Stores reusable code patterns (templates) with metadata.
- Selects an appropriate pattern for a given coding task.
- Fills in template slots using structured slot_values.
- Performs basic sanity checks on generated code.
- Logs generations and pattern usage for later learning.

This replaces any previous "pattern learned" NotImplemented paths.

No stubs. No fake learning.
If no suitable pattern is found, we explicitly say so and let the caller
fall back to the general LLM / Teacher coder path.
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
PATTERN_STORE_PATH = MAVEN_DIR / "pattern_coder.jsonl"
PATTERN_USAGE_LOG_PATH = MAVEN_DIR / "pattern_coder_usage.jsonl"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CodePattern:
    """
    One reusable code-generation pattern.

    Fields:
      id: unique pattern id (string)
      language: "python", "bash", etc.
      domain: e.g. "filesystem", "http", "cli", "data_processing"
      description: human-readable description of what the pattern does
      template: code template with {slot_name} placeholders
      slots: list of expected slots, each a dict with "name" and "description"
      tags: arbitrary tags for pattern selection
      tests: optional list of simple checks to run on generated code
    """
    id: str
    language: str
    domain: str
    description: str
    template: str
    slots: List[Dict[str, Any]]
    tags: List[str]
    tests: List[Dict[str, Any]]


@dataclass
class PatternGenerationRequest:
    """
    Input to the pattern coder.

    Fields:
      task_description: natural-language description of the coding task
      language: target language (e.g. "python")
      domain: high-level domain (optional; narrows pattern search)
      slot_values: dict from slot name to concrete values
      metadata: arbitrary extra info (caller, context ids, etc.)
    """
    task_description: str
    language: str
    domain: Optional[str]
    slot_values: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class PatternGenerationResult:
    """
    Result of pattern-based code generation.

    Fields:
      ok: whether generation succeeded
      reason: human-readable reason if not ok
      pattern_id: id of the pattern used (if any)
      code: generated code (if ok)
      tests_passed: list of test ids that passed
      tests_failed: list of test ids that failed
      debug: extra info (scores, slot info, etc.)
    """
    ok: bool
    reason: str
    pattern_id: Optional[str]
    code: Optional[str]
    tests_passed: List[str]
    tests_failed: List[str]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# =============================================================================
# Pattern storage utilities
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


class PatternStore:
    """
    Simple JSONL-backed pattern store.

    - Loads all patterns at startup.
    - Allows adding new patterns at runtime.
    - Persists new patterns to PATTERN_STORE_PATH.
    """

    def __init__(self, path: Path = PATTERN_STORE_PATH) -> None:
        self.path = path
        self._patterns: Dict[str, CodePattern] = {}
        self._load()

    def _load(self) -> None:
        self._patterns.clear()
        if not self.path.exists():
            logger.info("Pattern store %s does not exist; starting with defaults", self.path)
            self._seed_default_patterns()
            return

        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("kind") != "code_pattern":
                        continue
                    payload = obj.get("pattern") or {}
                    try:
                        pat = CodePattern(
                            id=str(payload["id"]),
                            language=str(payload["language"]),
                            domain=str(payload["domain"]),
                            description=str(payload.get("description", "")),
                            template=str(payload["template"]),
                            slots=list(payload.get("slots", [])),
                            tags=list(payload.get("tags", [])),
                            tests=list(payload.get("tests", [])),
                        )
                        self._patterns[pat.id] = pat
                    except Exception as e:
                        logger.error("Failed to parse pattern line: %s", e)
        except Exception as e:
            logger.error("Failed to load pattern store %s: %s", self.path, e)

        # If no patterns loaded, seed defaults
        if not self._patterns:
            self._seed_default_patterns()

    def _seed_default_patterns(self) -> None:
        """Seed the store with useful default patterns."""
        defaults = self._get_default_patterns()
        for pat in defaults:
            self._patterns[pat.id] = pat
            rec = {
                "kind": "code_pattern",
                "ts": _now_iso(),
                "pattern": asdict(pat),
            }
            _append_jsonl(self.path, rec)
        logger.info("Seeded %d default patterns", len(defaults))

    def _get_default_patterns(self) -> List[CodePattern]:
        """Return built-in default patterns."""
        return [
            CodePattern(
                id="python_cli_argparse_v1",
                language="python",
                domain="cli",
                description="Python CLI skeleton using argparse with main() entrypoint.",
                template='''import argparse

def main():
    parser = argparse.ArgumentParser(description="{description}")
    parser.add_argument("--input", required=True, help="{input_help}")
    parser.add_argument("--output", required=True, help="{output_help}")
    args = parser.parse_args()

    # TODO: implement logic here
    print("Reading from", args.input)
    print("Writing to", args.output)

if __name__ == "__main__":
    main()
''',
                slots=[
                    {"name": "description", "description": "CLI tool description"},
                    {"name": "input_help", "description": "Help text for --input"},
                    {"name": "output_help", "description": "Help text for --output"},
                ],
                tags=["cli", "argparse", "skeleton", "command-line"],
                tests=[
                    {"id": "contains_argparse", "type": "contains", "text": "import argparse"},
                    {"id": "contains_main", "type": "contains", "text": "def main()"},
                    {"id": "no_eval", "type": "not_contains", "text": "eval("},
                ],
            ),
            CodePattern(
                id="python_read_file_v1",
                language="python",
                domain="filesystem",
                description="Read a text file and process its contents.",
                template='''def read_file(filepath: str) -> str:
    """Read and return contents of {filepath_desc}."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# Usage
content = read_file("{example_path}")
print(f"Read {{len(content)}} characters")
''',
                slots=[
                    {"name": "filepath_desc", "description": "Description of what file is being read"},
                    {"name": "example_path", "description": "Example file path"},
                ],
                tags=["file", "read", "text", "filesystem"],
                tests=[
                    {"id": "contains_open", "type": "contains", "text": "open("},
                    {"id": "contains_encoding", "type": "contains", "text": "encoding="},
                    {"id": "no_eval", "type": "not_contains", "text": "eval("},
                ],
            ),
            CodePattern(
                id="python_write_file_v1",
                language="python",
                domain="filesystem",
                description="Write content to a text file.",
                template='''def write_file(filepath: str, content: str) -> None:
    """Write content to {filepath_desc}."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {{len(content)}} characters to {{filepath}}")

# Usage
write_file("{example_path}", "{example_content}")
''',
                slots=[
                    {"name": "filepath_desc", "description": "Description of what file is being written"},
                    {"name": "example_path", "description": "Example file path"},
                    {"name": "example_content", "description": "Example content to write"},
                ],
                tags=["file", "write", "text", "filesystem"],
                tests=[
                    {"id": "contains_open", "type": "contains", "text": "open("},
                    {"id": "contains_write_mode", "type": "contains", "text": '"w"'},
                    {"id": "no_eval", "type": "not_contains", "text": "eval("},
                ],
            ),
            CodePattern(
                id="python_http_get_v1",
                language="python",
                domain="http",
                description="Make an HTTP GET request using requests library.",
                template='''import requests

def fetch_url(url: str) -> dict:
    """Fetch data from {url_desc}."""
    response = requests.get(url, timeout={timeout})
    response.raise_for_status()
    return response.json()

# Usage
data = fetch_url("{example_url}")
print(f"Received: {{data}}")
''',
                slots=[
                    {"name": "url_desc", "description": "Description of what URL is being fetched"},
                    {"name": "timeout", "description": "Request timeout in seconds"},
                    {"name": "example_url", "description": "Example URL"},
                ],
                tags=["http", "get", "request", "api", "web"],
                tests=[
                    {"id": "contains_requests", "type": "contains", "text": "import requests"},
                    {"id": "contains_timeout", "type": "contains", "text": "timeout="},
                    {"id": "no_eval", "type": "not_contains", "text": "eval("},
                ],
            ),
            CodePattern(
                id="python_class_dataclass_v1",
                language="python",
                domain="data_processing",
                description="Python dataclass definition for structured data.",
                template='''from dataclasses import dataclass
from typing import Optional, List

@dataclass
class {class_name}:
    """{class_description}"""
    {field_name}: {field_type}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {{
            "{field_name}": self.{field_name},
        }}
''',
                slots=[
                    {"name": "class_name", "description": "Name of the dataclass"},
                    {"name": "class_description", "description": "Docstring for the class"},
                    {"name": "field_name", "description": "Name of the primary field"},
                    {"name": "field_type", "description": "Type of the primary field"},
                ],
                tags=["dataclass", "class", "data", "structure"],
                tests=[
                    {"id": "contains_dataclass", "type": "contains", "text": "@dataclass"},
                    {"id": "contains_import", "type": "contains", "text": "from dataclasses import"},
                    {"id": "no_eval", "type": "not_contains", "text": "eval("},
                ],
            ),
            CodePattern(
                id="python_try_except_v1",
                language="python",
                domain="error_handling",
                description="Try-except block with logging.",
                template='''import logging

logger = logging.getLogger(__name__)

def {function_name}({params}) -> {return_type}:
    """{function_description}"""
    try:
        # TODO: implement {operation_desc}
        result = None
        return result
    except {exception_type} as e:
        logger.error("{error_message}: %s", e)
        raise
''',
                slots=[
                    {"name": "function_name", "description": "Name of the function"},
                    {"name": "params", "description": "Function parameters"},
                    {"name": "return_type", "description": "Return type annotation"},
                    {"name": "function_description", "description": "Docstring"},
                    {"name": "operation_desc", "description": "Description of the operation"},
                    {"name": "exception_type", "description": "Exception type to catch"},
                    {"name": "error_message", "description": "Error log message"},
                ],
                tags=["error", "exception", "try", "except", "logging"],
                tests=[
                    {"id": "contains_try", "type": "contains", "text": "try:"},
                    {"id": "contains_except", "type": "contains", "text": "except"},
                    {"id": "contains_logging", "type": "contains", "text": "import logging"},
                    {"id": "no_eval", "type": "not_contains", "text": "eval("},
                ],
            ),
            CodePattern(
                id="bash_script_v1",
                language="bash",
                domain="cli",
                description="Basic bash script with argument handling.",
                template='''#!/bin/bash
# {script_description}

set -e  # Exit on error

if [ $# -lt {min_args} ]; then
    echo "Usage: $0 {usage_args}"
    exit 1
fi

{arg_name}="$1"
echo "Processing: ${arg_name}"

# TODO: implement logic here
''',
                slots=[
                    {"name": "script_description", "description": "Description of what the script does"},
                    {"name": "min_args", "description": "Minimum number of arguments"},
                    {"name": "usage_args", "description": "Usage string for arguments"},
                    {"name": "arg_name", "description": "Name for the first argument variable"},
                ],
                tags=["bash", "script", "shell", "cli"],
                tests=[
                    {"id": "contains_shebang", "type": "contains", "text": "#!/bin/bash"},
                    {"id": "contains_set_e", "type": "contains", "text": "set -e"},
                    {"id": "no_rm_rf", "type": "not_contains", "text": "rm -rf /"},
                ],
            ),
        ]

    def add_pattern(self, pattern: CodePattern) -> None:
        """
        Add a new pattern and persist it immediately.
        """
        self._patterns[pattern.id] = pattern
        rec = {
            "kind": "code_pattern",
            "ts": _now_iso(),
            "pattern": asdict(pattern),
        }
        _append_jsonl(self.path, rec)

    def get_pattern(self, pattern_id: str) -> Optional[CodePattern]:
        return self._patterns.get(pattern_id)

    def all_patterns(self) -> List[CodePattern]:
        return list(self._patterns.values())

    def count(self) -> int:
        return len(self._patterns)


# =============================================================================
# Pattern selection and scoring
# =============================================================================

def _normalize_text(text: str) -> List[str]:
    """
    Very simple normalization: lowercase, split on non-alphanumerics.
    """
    t = text.lower()
    tokens = re.split(r"[^a-z0-9_]+", t)
    return [tok for tok in tokens if tok]


def _jaccard_score(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    if not union:
        return 0.0
    return len(inter) / len(union)


def _pattern_score(pattern: CodePattern, req: PatternGenerationRequest) -> float:
    """
    Score how well a pattern matches the request.

    Consider:
      - language match (hard)
      - domain match (bonus)
      - description/tags vs task_description tokens
    """
    if pattern.language.lower() != req.language.lower():
        return 0.0

    score = 0.1  # base if language matches

    # Domain bonus
    if req.domain and pattern.domain.lower() == req.domain.lower():
        score += 0.2

    task_tokens = _normalize_text(req.task_description)
    patt_tokens = _normalize_text(pattern.description) + [
        t for tag in pattern.tags for t in _normalize_text(tag)
    ]

    score += 0.7 * _jaccard_score(task_tokens, patt_tokens)

    return score


def _select_best_pattern(
    store: PatternStore,
    req: PatternGenerationRequest,
    min_score: float = 0.15,
) -> Optional[Tuple[CodePattern, float]]:
    """
    Choose the best pattern for this request.

    Returns (pattern, score) or None if no suitable pattern.
    """
    best_pat: Optional[CodePattern] = None
    best_score: float = 0.0

    for pat in store.all_patterns():
        s = _pattern_score(pat, req)
        if s > best_score:
            best_score = s
            best_pat = pat

    if best_pat is None or best_score < min_score:
        return None

    return best_pat, best_score


# =============================================================================
# Template filling and tests
# =============================================================================

def _fill_template(template: str, slot_values: Dict[str, Any]) -> str:
    """
    Fill the template using Python's .format() mechanism.

    - All slot_values keys must match the template placeholders.
    - If any slot is missing, we raise a KeyError (caller handles it).
    """
    # Convert non-str values to str for formatting
    safe_values = {k: str(v) for k, v in slot_values.items()}
    return template.format(**safe_values)


def _run_tests(code: str, tests: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Run simple sanity tests on generated code.

    Supported test types:
      - { "id": "contains_import_os", "type": "contains", "text": "import os" }
      - { "id": "not_contains_eval", "type": "not_contains", "text": "eval(" }
      - { "id": "regex_match", "type": "regex", "pattern": "def \\w+\\(" }

    Returns: (passed_ids, failed_ids)
    """
    passed: List[str] = []
    failed: List[str] = []

    for t in tests:
        tid = str(t.get("id") or f"test_{len(passed) + len(failed)}")
        ttype = str(t.get("type", "contains")).lower()
        text = str(t.get("text", ""))
        pattern = str(t.get("pattern", ""))

        if ttype == "contains":
            if not text:
                continue
            if text in code:
                passed.append(tid)
            else:
                failed.append(tid)
        elif ttype == "not_contains":
            if not text:
                continue
            if text in code:
                failed.append(tid)
            else:
                passed.append(tid)
        elif ttype == "regex":
            if not pattern:
                continue
            try:
                if re.search(pattern, code):
                    passed.append(tid)
                else:
                    failed.append(tid)
            except re.error:
                failed.append(tid)
        else:
            # unsupported test type: treat as failed to be safe
            failed.append(tid)

    return passed, failed


# =============================================================================
# PatternCoder main class
# =============================================================================

class PatternCoder:
    """
    Main pattern-based generation engine.

    - Uses PatternStore for persistence.
    - Handles selection, formatting, testing, and logging.

    This is what the Coder Brain calls instead of raising NotImplemented.
    """

    def __init__(self, store: Optional[PatternStore] = None) -> None:
        self.store = store or PatternStore()

    def generate(self, req: PatternGenerationRequest) -> PatternGenerationResult:
        """
        Attempt pattern-based generation for a given request.

        Behavior:
          - If no suitable pattern: ok=False, reason="no_pattern", pattern_id=None.
          - If slots missing: ok=False, reason="missing_slots".
          - If tests fail: ok=True (by default), but tests_failed is non-empty and
            reason describes the failure; caller can decide what to do.
        """
        # 1. Pattern selection
        sel = _select_best_pattern(self.store, req)
        if sel is None:
            result = PatternGenerationResult(
                ok=False,
                reason="no_suitable_pattern",
                pattern_id=None,
                code=None,
                tests_passed=[],
                tests_failed=[],
                debug={"note": "No pattern scored above threshold"},
            )
            self._log_usage(req, result, pattern_score=None)
            return result

        pattern, score = sel

        # 2. Slot checking
        required_slots = [s.get("name") for s in pattern.slots if s.get("name")]
        missing_slots = [name for name in required_slots if name not in req.slot_values]

        if missing_slots:
            result = PatternGenerationResult(
                ok=False,
                reason="missing_slots",
                pattern_id=pattern.id,
                code=None,
                tests_passed=[],
                tests_failed=[],
                debug={
                    "missing_slots": missing_slots,
                    "required_slots": required_slots,
                    "pattern_score": score,
                },
            )
            self._log_usage(req, result, pattern_score=score, pattern=pattern)
            return result

        # 3. Template filling
        try:
            code = _fill_template(pattern.template, req.slot_values)
        except KeyError as e:
            result = PatternGenerationResult(
                ok=False,
                reason=f"slot_format_error: missing {e}",
                pattern_id=pattern.id,
                code=None,
                tests_passed=[],
                tests_failed=[],
                debug={"pattern_score": score},
            )
            self._log_usage(req, result, pattern_score=score, pattern=pattern)
            return result
        except Exception as e:
            result = PatternGenerationResult(
                ok=False,
                reason=f"template_error: {e}",
                pattern_id=pattern.id,
                code=None,
                tests_passed=[],
                tests_failed=[],
                debug={"pattern_score": score},
            )
            self._log_usage(req, result, pattern_score=score, pattern=pattern)
            return result

        # 4. Run tests
        tests_passed, tests_failed = _run_tests(code, pattern.tests)

        # By default, we still return ok=True even if some tests failed.
        # Caller (Coder Brain / governance) can decide to reject it.
        reason = "ok"
        if tests_failed:
            reason = "tests_failed"

        result = PatternGenerationResult(
            ok=True,
            reason=reason,
            pattern_id=pattern.id,
            code=code,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            debug={"pattern_score": score},
        )
        self._log_usage(req, result, pattern_score=score, pattern=pattern)
        return result

    def learn_pattern_from_example(
        self,
        pattern_id: str,
        language: str,
        domain: str,
        description: str,
        template: str,
        slots: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
        tests: Optional[List[Dict[str, Any]]] = None,
    ) -> CodePattern:
        """
        Add a new pattern (e.g. when Teacher or human provides a reusable template).

        This does NOT guess. Caller must supply a real template and slots.
        """
        pat = CodePattern(
            id=pattern_id,
            language=language,
            domain=domain,
            description=description,
            template=template,
            slots=slots,
            tags=tags or [],
            tests=tests or [],
        )
        self.store.add_pattern(pat)
        return pat

    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available patterns with summary info."""
        return [
            {
                "id": p.id,
                "language": p.language,
                "domain": p.domain,
                "description": p.description,
                "tags": p.tags,
                "slot_count": len(p.slots),
            }
            for p in self.store.all_patterns()
        ]

    def get_pattern_details(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a pattern."""
        pat = self.store.get_pattern(pattern_id)
        if pat is None:
            return None
        return asdict(pat)

    def _log_usage(
        self,
        req: PatternGenerationRequest,
        res: PatternGenerationResult,
        pattern_score: Optional[float],
        pattern: Optional[CodePattern] = None,
    ) -> None:
        """
        Log every attempt for audit and learning.

        Stored in ~/.maven/pattern_coder_usage.jsonl so identity/governance
        can see how pattern-based coding behaves.
        """
        rec: Dict[str, Any] = {
            "kind": "pattern_coder_usage",
            "ts": _now_iso(),
            "request": {
                "task_description": req.task_description,
                "language": req.language,
                "domain": req.domain,
                "slot_values": req.slot_values,
                "metadata": req.metadata,
            },
            "result": {
                "ok": res.ok,
                "reason": res.reason,
                "pattern_id": res.pattern_id,
                "tests_passed": res.tests_passed,
                "tests_failed": res.tests_failed,
                "debug": res.debug,
            },
            "pattern_score": pattern_score,
        }
        if pattern is not None:
            rec["pattern_summary"] = {
                "id": pattern.id,
                "language": pattern.language,
                "domain": pattern.domain,
                "tags": pattern.tags,
            }

        _append_jsonl(PATTERN_USAGE_LOG_PATH, rec)


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Default coder instance (lazy initialization)
_default_coder: Optional[PatternCoder] = None


def _get_coder() -> PatternCoder:
    """Get or create default coder instance."""
    global _default_coder
    if _default_coder is None:
        _default_coder = PatternCoder()
    return _default_coder


def generate_code_with_pattern(
    task_description: str,
    language: str,
    domain: Optional[str],
    slot_values: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> PatternGenerationResult:
    """
    Module-level convenience function for pattern-based code generation.
    """
    req = PatternGenerationRequest(
        task_description=task_description,
        language=language,
        domain=domain,
        slot_values=slot_values,
        metadata=metadata or {},
    )
    return _get_coder().generate(req)


def learn_pattern(
    pattern_id: str,
    language: str,
    domain: str,
    description: str,
    template: str,
    slots: List[Dict[str, Any]],
    tags: Optional[List[str]] = None,
    tests: Optional[List[Dict[str, Any]]] = None,
) -> CodePattern:
    """Module-level function to learn a new pattern."""
    return _get_coder().learn_pattern_from_example(
        pattern_id=pattern_id,
        language=language,
        domain=domain,
        description=description,
        template=template,
        slots=slots,
        tags=tags,
        tests=tests,
    )


def list_available_patterns() -> List[Dict[str, Any]]:
    """Module-level function to list all available patterns."""
    return _get_coder().list_patterns()


def get_default_pattern_coder() -> PatternCoder:
    """Get or create the default PatternCoder instance.

    This is the canonical way to get a PatternCoder instance from outside the module.
    """
    return _get_coder()


# =============================================================================
# Service API
# =============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for pattern coder.

    Supported operations:
    - GENERATE: Generate code from a pattern
    - LEARN: Learn a new pattern
    - LIST_PATTERNS: List all available patterns
    - GET_PATTERN: Get details of a specific pattern
    - HEALTH: Health check
    """
    op = (msg or {}).get("op", "").upper()
    mid = msg.get("mid")
    payload = msg.get("payload") or {}

    coder = _get_coder()

    if op == "GENERATE":
        try:
            task_description = payload.get("task_description", "")
            language = payload.get("language", "python")
            domain = payload.get("domain")
            slot_values = payload.get("slot_values", {})
            metadata = payload.get("metadata", {})

            if not task_description:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_TASK", "message": "task_description required"},
                }

            req = PatternGenerationRequest(
                task_description=task_description,
                language=language,
                domain=domain,
                slot_values=slot_values,
                metadata=metadata,
            )
            result = coder.generate(req)
            return {
                "ok": result.ok,
                "op": op,
                "mid": mid,
                "payload": result.to_dict(),
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "GENERATE_FAILED", "message": str(e)},
            }

    if op == "LEARN":
        try:
            pattern_id = payload.get("pattern_id")
            language = payload.get("language")
            domain = payload.get("domain")
            description = payload.get("description")
            template = payload.get("template")
            slots = payload.get("slots", [])
            tags = payload.get("tags", [])
            tests = payload.get("tests", [])

            if not all([pattern_id, language, domain, template]):
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_FIELDS", "message": "pattern_id, language, domain, template required"},
                }

            pat = coder.learn_pattern_from_example(
                pattern_id=pattern_id,
                language=language,
                domain=domain,
                description=description or "",
                template=template,
                slots=slots,
                tags=tags,
                tests=tests,
            )
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"pattern_id": pat.id, "message": "Pattern learned successfully"},
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "LEARN_FAILED", "message": str(e)},
            }

    if op == "LIST_PATTERNS":
        try:
            patterns = coder.list_patterns()
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

            details = coder.get_pattern_details(pattern_id)
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

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "healthy",
                "service": "pattern_coder",
                "pattern_count": coder.store.count(),
                "store_path": str(PATTERN_STORE_PATH),
                "usage_log_path": str(PATTERN_USAGE_LOG_PATH),
                "available_operations": ["GENERATE", "LEARN", "LIST_PATTERNS", "GET_PATTERN", "HEALTH"],
            },
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Operation '{op}' not supported"},
    }
