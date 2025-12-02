"""
Natural Language Chat Interface for Maven (Improved)
===================================================

This module provides a conversational entry point into the Maven system.  It
leverages the Language brain to parse user utterances and determine
communicative intent before dispatching the request to the appropriate
subsystem.  Statements and questions are evaluated via the full pipeline
through the Memory Librarian.  Self‑DMN maintenance operations (tick,
reflect, dissent scan) are invoked when explicitly requested.  The
confidence used for pipeline execution is derived from the language
brain's confidence penalty, allowing subjective content (emotions,
opinions, speculation) to be treated with appropriate caution.

To use this interface from the command line, run:

    python -m maven.ui.maven_chat

and enter queries at the prompt.  Type ``exit`` or ``quit`` to stop.  This
file lives in the ``ui`` package and is imported by ``run_maven.py`` when
no arguments are provided.
"""

from __future__ import annotations

import re
import time
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

# At runtime this module may be executed either via ``python -m maven.ui.maven_chat``
# or directly as a script.  When executed as a script, Python's import
# machinery does not automatically include the Maven project root (which
# contains the ``api`` package) on ``sys.path``.  To ensure that
# ``from api.utils import generate_mid`` succeeds regardless of how this
# file is launched, attempt to import ``api.utils`` and, on failure,
# insert the project root into ``sys.path`` dynamically.  The project
# root is two directories up from this file (``.../maven/ui`` → ``.../maven``).
try:
    from api.utils import generate_mid, CFG  # type: ignore
except ModuleNotFoundError:
    # Compute the absolute path to the project root (two parents up)
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parents[1]
    # Prepend project root to sys.path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from api.utils import generate_mid, CFG  # type: ignore

# Import query preprocessor for multi-question handling
try:
    from api.query_preprocessor import preprocess_query, should_process_as_multi_query  # type: ignore
except Exception:
    # Fallback if preprocessor not available
    preprocess_query = None  # type: ignore
    should_process_as_multi_query = None  # type: ignore

# Import brains dynamically to avoid cyclic dependencies when run as a script.
try:
    from brains.cognitive.language.service import language_brain  # type: ignore
except Exception:
    language_brain = None  # type: ignore
try:
    from brains.cognitive.memory_librarian.service import memory_librarian  # type: ignore
except Exception:
    memory_librarian = None  # type: ignore
try:
    from brains.cognitive.self_dmn.service import self_dmn_brain  # type: ignore
except Exception:
    self_dmn_brain = None  # type: ignore
try:
    from brains.cognitive import correction_handler  # type: ignore
except Exception:
    correction_handler = None  # type: ignore

from brains.maven_paths import get_maven_root, get_reports_path

# Import enhanced input handler for file uploads and multi-line paste
try:
    from brains.tools.input_handler import (
        process_input,
        format_attachments_for_context,
        ProcessedInput,
    )
    _input_handler_enabled = True
except Exception as e:
    print(f"[MAVEN_CHAT] Warning: Input handler not available: {e}")
    _input_handler_enabled = False
    process_input = None  # type: ignore
    format_attachments_for_context = None  # type: ignore

# Initialize host tools and inject them into brain facades
# This enables LLM, web search, and other tools to work
try:
    from host_tools.factory import create_host_tools
    from brains.tool_injection import inject_tools
    _host_tools = create_host_tools(
        enable_web=True,
        enable_llm=True,
        enable_shell=True,
        enable_git=True,
        enable_sandbox=True,
        root_dir=str(get_maven_root())
    )
    inject_tools(_host_tools)
except Exception as e:
    print(f"[MAVEN_CHAT] Warning: Failed to inject host tools: {e}")

MAVEN_ROOT = get_maven_root()
CHAT_LOG_DIR = get_reports_path("agent", "chat")

# Module level variables for chat logging and pending actions.
#
# When a chat session is started via ``repl()``, ``_CONV_FILE`` is set to
# the path of a JSONL file under ``reports/agent/chat``.  Each turn of
# the conversation is appended as a JSON object with ``ts``, ``user``,
# ``intent`` and ``response`` fields.  Logging is best effort:
# failures to open or write the file are silently ignored.
#
# The ``_PENDING_ACTION`` variable stores a tuple describing a
# deferred operation that requires user confirmation.  It is set by
# commands that would cause side effects (e.g. storing a fact or
# registering a claim).  When set, the next call to ``process``
# expects the user to respond "yes" or "no".  A positive response
# triggers the stored callable and clears the pending action; a
# negative response clears the pending action without performing any
# operation.  Any other response also clears the pending action and
# proceeds to handle the input normally.
_CONV_FILE: str | None = None
# Pending action tuple (callable, args tuple, kwargs dict, name) or None
_PENDING_ACTION: tuple | None = None

# Learning mode for training. When True, the system will call LLM for learning.
# Default is True (TRAINING mode - LLM learning enabled).
_TRAINING_MODE_ENABLED: bool = True

# Import LearningMode enum for training mode control
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    # Fallback if import fails
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"


def _enable_execution_confirmed() -> str:
    """
    Helper function called when user confirms execution enablement.

    Returns:
        Success message string
    """
    try:
        from brains.tools.execution_guard import enable_execution_via_config

        success, message = enable_execution_via_config()
        if success:
            return "Execution has been enabled successfully. Settings saved to ~/.maven/config.json. You can now use filesystem, git, and other agency tools."
        else:
            return f"Failed to enable execution: {message}"

    except Exception as e:
        return f"Error enabling execution: {e}"


def _enable_full_agency_confirmed() -> str:
    """
    Helper function called when user confirms full agency mode enablement.

    Returns:
        Success message string
    """
    try:
        from brains.tools.execution_guard import enable_full_agency

        cfg = enable_full_agency("user confirmed full agency via chat")
        return """Full Agency mode has been enabled successfully!

You now have unrestricted access to:
  - Read/write files anywhere on disk (OS permissions apply)
  - Run shell commands
  - Run Python code
  - Browse the web
  - Use git (all operations)
  - Run autonomous agents

Settings saved to ~/.maven/config.json. Only truly destructive commands (rm -rf /, mkfs, etc.) are blocked.
"""
    except Exception as e:
        return f"Error enabling full agency: {e}"


def _check_full_agency_profile() -> bool:
    """Check if FULL_AGENCY profile should be activated from environment."""
    import os
    profile = os.getenv("MAVEN_CAPABILITIES_PROFILE", "").upper()
    mode = os.getenv("MAVEN_EXECUTION_MODE", "").upper()
    return profile == "FULL_AGENCY" or mode == "FULL_AGENCY"


def _check_safe_chat_profile() -> bool:
    """Check if SAFE_CHAT profile should be activated from environment."""
    import os
    profile = os.getenv("MAVEN_CAPABILITIES_PROFILE", "").upper()
    return profile == "SAFE_CHAT"


def _get_active_profile() -> str:
    """Get the active capability profile from environment."""
    import os
    profile = os.getenv("MAVEN_CAPABILITIES_PROFILE", "").upper()
    if profile in ("SAFE_CHAT", "FULL_AGENCY"):
        return profile
    mode = os.getenv("MAVEN_EXECUTION_MODE", "").upper()
    if mode == "FULL_AGENCY":
        return "FULL_AGENCY"
    return "DEFAULT"


def _activate_profile_at_startup() -> None:
    """Activate the capability profile specified in environment at chat startup."""
    profile = _get_active_profile()

    if profile == "FULL_AGENCY":
        try:
            from capabilities import activate_profile
            activate_profile("FULL_AGENCY", "activated at chat startup from environment")
            print("[CHAT] Profile: FULL_AGENCY - unrestricted access to all tools")
        except Exception as e:
            try:
                from brains.tools.execution_guard import enable_full_agency
                enable_full_agency("activated at chat startup from environment")
                print("[CHAT] FULL_AGENCY mode enabled")
            except Exception:
                pass
    elif profile == "SAFE_CHAT":
        try:
            from capabilities import activate_profile
            activate_profile("SAFE_CHAT", "activated at chat startup from environment")
            print("[CHAT] Profile: SAFE_CHAT - pure conversation, no tools")
        except Exception as e:
            try:
                from brains.tools.execution_guard import enable_safe_chat
                enable_safe_chat("activated at chat startup from environment")
                print("[CHAT] SAFE_CHAT mode enabled")
            except Exception:
                pass


def _extract_web_settings(topic: str) -> Tuple[str, bool, Any, int, bool]:
    """Parse web hints from the topic string and return cleaned settings."""

    default_web = bool((CFG or {}).get("ENABLE_WEB_RESEARCH", True)) if 'CFG' in globals() else True
    max_seconds = int((CFG or {}).get("WEB_RESEARCH_MAX_SECONDS", 1200)) if 'CFG' in globals() else 1200
    max_requests = int((CFG or {}).get("WEB_RESEARCH_MAX_REQUESTS", 20)) if 'CFG' in globals() else 20

    web_enabled = default_web
    time_budget_seconds = None
    parts: List[str] = []
    hint_seen = False

    for token in str(topic).split():
        cleaned = token.strip().strip("\"'“”‘’").rstrip(",")
        if not cleaned.lower().startswith("web:"):
            parts.append(token)
            continue

        hint_seen = True

        value = cleaned.split(":", 1)[1].strip().lower() if ":" in cleaned else ""
        if value.isdigit():
            try:
                time_budget_seconds = max(1, min(max_seconds, int(value)))
                web_enabled = True
            except Exception:
                pass
        elif value in {"true", "on", "yes"}:
            web_enabled = True
        elif value in {"false", "off", "no"}:
            web_enabled = False
        else:
            parts.append(token)

    cleaned_topic = " ".join(parts).strip().strip("\"'")
    return cleaned_topic or topic, web_enabled, time_budget_seconds, max_requests, hint_seen


def _sanitize_for_log(text: str) -> str:
    """
    Sanitize a string before logging by masking email addresses and
    long alphanumeric tokens that may represent secrets or IDs.

    This helper replaces anything that looks like an email address with
    ``<EMAIL>`` and any contiguous run of 16 or more alphanumeric
    characters with ``<TOKEN>``.  It is intended to prevent the
    accidental recording of sensitive data in chat logs while still
    retaining the overall shape of the conversation for auditing.

    Args:
        text: The text to sanitize.

    Returns:
        A sanitized version of the text safe for logging.
    """
    try:
        # Mask email addresses
        text = re.sub(r"([A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+)", "<EMAIL>", text)
        # Mask long tokens (16+ alphanumeric characters)
        text = re.sub(r"[A-Za-z0-9]{16,}", "<TOKEN>", text)
    except Exception:
        pass
    return text


def _log_turn(user_text: str, intent: str, response: str) -> None:
    """Append a single chat turn to the conversation log.

    If a conversation file has been established by the REPL, this
    function writes a single JSON line containing the timestamp,
    user input, interpreted intent and system response.  Prior to
    writing, the user and response strings are sanitized to mask
    potential secrets such as email addresses or long tokens.  Logging
    errors are ignored to avoid disrupting the chat flow.

    Args:
        user_text: The raw input entered by the user.
        intent: The high‑level intent determined for this turn.
        response: The response returned to the user.
    """
    global _CONV_FILE
    if not _CONV_FILE:
        return
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "user": _sanitize_for_log(user_text),
        "intent": intent,
        "response": _sanitize_for_log(response),
    }
    try:
        with open(_CONV_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        # Do not surface logging errors
        pass


def _parse_language(text: str) -> Dict[str, Any]:
    """Call the language brain to parse the text and return the parsed payload.

    If the language brain is unavailable, a fallback classification is
    returned: the input is treated as a FACT statement with no penalty.

    Args:
        text: The user utterance.

    Returns:
        A dictionary of parsed metadata including ``storable_type`` and
        ``confidence_penalty``.
    """
    if language_brain is None:
        # Fallback: treat everything as a fact with zero penalty
        return {
            "storable_type": "FACT",
            "storable": True,
            "confidence_penalty": 0.0,
        }
    mid = generate_mid()
    try:
        resp = language_brain.service_api({"op": "PARSE", "mid": mid, "payload": {"text": text}})
        parsed = (resp or {}).get("payload") or {}
        return parsed
    except Exception:
        # On error, fallback classification
        return {
            "storable_type": "FACT",
            "storable": True,
            "confidence_penalty": 0.0,
        }


def _interpret_intent(text: str, parsed: Dict[str, Any]) -> str:
    """Determine a high‑level intent based on the raw text and parse result.

    This function inspects the user utterance for explicit maintenance
    commands (tick, reflect, dissent) and otherwise uses the storable_type
    to decide whether to run the full pipeline.  Commands take priority
    over the storable_type classification.

    Args:
        text: The raw user utterance.
        parsed: The parsed metadata from the language brain.

    Returns:
        One of ``dmn_tick``, ``dmn_reflect``, ``dmn_dissent`` or ``pipeline``.
    """
    lower = text.strip().lower()
    # Explicit Self‑DMN commands override other intents
    if any(word in lower for word in ["tick", "advance hum", "oscillator"]):
        return "dmn_tick"
    if "reflect" in lower:
        return "dmn_reflect"
    # Only trigger dissent scan for explicit dissent-related queries
    # NOT for "scan your memory" or other self-introspection queries
    # Self-memory queries (e.g., "scan your memory") should go to pipeline
    # where the self-intent gate will route them to self_model
    if "dissent" in lower:
        return "dmn_dissent"
    # Diagnostics command
    if lower in ["diag", "diagnostics", "diagnostic"]:
        return "diag"
    # Shell/tool commands should go to pipeline (not intercepted by status/health checks)
    shell_cmd_prefixes = (
        "pc:", "pc ", "human:", "human ", "x:", "x ",
        "git ", "ls", "cd ", "mkdir ", "rm ", "cat ", "pwd",
        "pip ", "python ", "npm ", "docker ", "curl ",
    )
    if lower.startswith(shell_cmd_prefixes):
        return "pipeline"
    # Status or health requests (but NOT git status, ls, etc. which are handled above)
    if any(word in lower for word in ["status", "health", "counts"]):
        return "status"
    # Summaries or reports and export requests
    if any(word in lower for word in ["summary", "summarize", "report", "dashboard", "export"]):
        return "summary"
    # Retrieval requests (search memory)
    if ("search" in lower or "find" in lower or "lookup" in lower) and "memory" in lower:
        return "retrieve"
    # Router explanation requests
    if "router" in lower and ("explain" in lower or "why" in lower or "route" in lower or "bank" in lower):
        return "router_explain"
    # Register claim
    if "register" in lower and "claim" in lower:
        return "register_claim"
    # Explicit store command (but will ultimately run through pipeline)
    if lower.startswith("store ") or lower.startswith("remember ") or lower.startswith("save "):
        return "pipeline"
    # Default: treat as pipeline request (question or statement)
    return "pipeline"


def process(text: str) -> str:
    """Handle a user utterance by dispatching to the appropriate brain.

    The utterance is first parsed by the language brain to determine
    intent and confidence penalties.  Self‑DMN commands are executed
    directly.  All other utterances are passed through the full
    pipeline via the Memory Librarian.  The resulting answer is
    returned along with a confidence score when available.

    Args:
        text: The raw user utterance.

    Returns:
        A human‑friendly response string.
    """
    # If there is a pending action awaiting user consent, handle it first.
    global _PENDING_ACTION
    if _PENDING_ACTION is not None:
        # Unpack pending action
        cb, args, kwargs, action_name = _PENDING_ACTION
        reply = text.strip().lower()
        # Acceptable affirmative responses
        yes_set = {"yes", "y", "ok", "okay", "sure", "proceed", "apply"}
        # Acceptable negative responses
        no_set = {"no", "n", "cancel", "stop", "abort"}
        if reply in yes_set:
            # Perform the deferred action
            try:
                result = cb(*args, **kwargs)
            except Exception as e:
                # Clear pending action before raising error
                _PENDING_ACTION = None
                return f"An error occurred while performing the {action_name.replace('_', ' ')}: {e}"
            # Clear the pending action
            _PENDING_ACTION = None
            return result
        elif reply in no_set:
            # Cancel the pending operation
            _PENDING_ACTION = None
            return "Operation cancelled."
        else:
            # Unrecognized response: clear and continue with normal processing
            _PENDING_ACTION = None
            # Fall through to standard handling below

    # Check if the input is user feedback (positive or negative) about the last answer
    if correction_handler is not None:
        try:
            if correction_handler.is_positive_feedback(text):
                # User confirmed the last answer was correct
                lib_api = memory_librarian.service_api if memory_librarian is not None else None
                return correction_handler.handle_positive_feedback(lib_api)
            elif correction_handler.is_negative_feedback(text):
                # User indicated the last answer was incorrect
                lib_api = memory_librarian.service_api if memory_librarian is not None else None
                return correction_handler.handle_negative_feedback(lib_api)
        except Exception:
            # If feedback handling fails, continue with normal processing
            pass

    # ------------------------------------------------------------------
    # EXECUTION ENABLEMENT COMMAND
    # ------------------------------------------------------------------
    # Detect "enable execution" command and handle with confirmation
    try:
        import re as _re
        lower_text = text.strip().lower()

        # Patterns for enable execution command
        enable_patterns = [
            r"\benable\s+execution\b",
            r"\benable\s+code\s+execution\b",
            r"\ballow\s+execution\b",
            r"\bturn\s+on\s+execution\b",
            r"\bactivate\s+execution\b"
        ]

        for pattern in enable_patterns:
            if _re.search(pattern, lower_text):
                # Check if already enabled
                try:
                    from brains.tools.execution_guard import get_execution_status, enable_execution_via_config

                    status = get_execution_status()
                    if status["enabled"]:
                        return f"Execution is already enabled (source: {status['source']}). You can use filesystem, git, and other agency tools."

                    # Not enabled yet - show confirmation prompt
                    confirmation_msg = """
Enabling execution will allow me to:
- Read and write files within the Maven directory
- Execute git operations (commit, push, etc.)
- Reload modules dynamically
- Perform other system operations

This setting will be saved to ~/.maven/config.json and persist across sessions.

Do you want to enable execution? (yes/no)
""".strip()

                    # Set up pending action for confirmation
                    _PENDING_ACTION = (
                        lambda: _enable_execution_confirmed(),
                        (),
                        {},
                        "enable_execution"
                    )

                    return confirmation_msg

                except Exception as e:
                    return f"Error checking execution status: {e}"

        # Patterns for disable execution command
        disable_patterns = [
            r"\bdisable\s+execution\b",
            r"\bturn\s+off\s+execution\b",
            r"\bdeactivate\s+execution\b"
        ]

        for pattern in disable_patterns:
            if _re.search(pattern, lower_text):
                try:
                    from brains.tools.execution_guard import disable_execution_via_config

                    success, message = disable_execution_via_config()
                    if success:
                        return "Execution has been disabled. Agency tools (filesystem, git, etc.) will not be available until you enable execution again."
                    else:
                        return f"Failed to disable execution: {message}"

                except Exception as e:
                    return f"Error disabling execution: {e}"

        # Patterns for enable full agency command
        full_agency_patterns = [
            r"\benable\s+full\s+agency\b",
            r"\bfull\s+agency\s+mode\b",
            r"\bunrestricted\s+mode\b",
            r"\benable\s+full\s+access\b",
            r"\bfull\s+access\s+mode\b"
        ]

        for pattern in full_agency_patterns:
            if _re.search(pattern, lower_text):
                # Check if already in full agency mode
                try:
                    from brains.tools.execution_guard import get_execution_status, ExecMode

                    status = get_execution_status()
                    if status.mode == ExecMode.FULL_AGENCY and status.effective:
                        return f"Full Agency mode is already enabled (source: {status.source}). You have unrestricted access to all tools and capabilities."

                    # Not enabled yet - show confirmation prompt
                    confirmation_msg = """
Enabling Full Agency mode will give me UNRESTRICTED access to:
  - Read/write files ANYWHERE on disk (only limited by OS permissions)
  - Execute arbitrary shell commands
  - Run Python code
  - Browse the web
  - Full git operations (clone, push, etc.)
  - Run autonomous agents

Only truly destructive commands (rm -rf /, mkfs, etc.) will be blocked.

This is a powerful mode. Are you sure? (yes/no)
""".strip()

                    # Set up pending action for confirmation
                    _PENDING_ACTION = (
                        lambda: _enable_full_agency_confirmed(),
                        (),
                        {},
                        "enable_full_agency"
                    )

                    return confirmation_msg

                except Exception as e:
                    return f"Error checking full agency status: {e}"

    except Exception as e:
        # If command handling fails, continue with normal processing
        pass

    # ------------------------------------------------------------------
    # PERSONAL INFORMATION EXTRACTION AND STORAGE
    # ------------------------------------------------------------------
    # Detect and store personal statements like "i am josh", "i like the color green", etc.
    # Store via personal_brain.SET_USER_SLOT for structured data
    try:
        import re as _re
        lower_text = text.strip().lower()

        # SKIP personal info extraction for tool commands
        # These should pass through to memory_librarian for proper routing
        tool_prefixes = ["use grok:", "use grok ", "grok:", "x:", "post:", "human:", "use chatgpt:"]
        is_tool_command = any(lower_text.startswith(prefix) for prefix in tool_prefixes)

        if not is_tool_command:
            # Detect "i am [name]" or "call me [name]" or "my name is [name]"
            name_patterns = [
                r"\bi\s+am\s+([a-z]+)\b",
                r"\bcall\s+me\s+([a-z]+)\b",
                r"\bmy\s+name\s+is\s+([a-z]+)\b"
            ]
            for pattern in name_patterns:
                match = _re.search(pattern, lower_text)
                if match:
                    name = match.group(1).strip().capitalize()
                    # Store via personal_brain
                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api
                        personal_api({
                            "op": "SET_USER_SLOT",
                            "payload": {
                                "slot_name": "name",
                                "value": name
                            }
                        })
                        return f"I've noted what you've shared."
                    except Exception:
                        pass
                    break

            # Detect "i like the color [color]" or "my favorite color is [color]"
            color_patterns = [
                r"\bi\s+like\s+the\s+color\s+([a-z]+)\b",
                r"\bmy\s+favorite\s+colou?r\s+is\s+([a-z]+)\b",
                r"\bmy\s+favourite\s+colou?r\s+is\s+([a-z]+)\b",
                r"\bfavorite\s+colou?r\s+([a-z]+)\b",
                r"\bfavourite\s+colou?r\s+([a-z]+)\b"
            ]
            for pattern in color_patterns:
                match = _re.search(pattern, lower_text)
                if match:
                    color = match.group(1).strip()
                    # Store via personal_brain
                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api
                        personal_api({
                            "op": "SET_USER_SLOT",
                            "payload": {
                                "slot_name": "favorite_color",
                                "value": color
                            }
                        })
                        return f"I'll remember that you like the color {color}."
                    except Exception:
                        pass
                    break

            # Detect "i like the animal [animal]" or "my favorite animal is [animal]"
            animal_patterns = [
                r"\bi\s+like\s+the\s+animal\s+([a-z]+)\b",
                r"\bi\s+like\s+([a-z]+)\b.*\banimal",
                r"\bmy\s+favorite\s+animal\s+is\s+([a-z]+)\b",
                r"\bmy\s+favourite\s+animal\s+is\s+([a-z]+)\b",
                r"\bfavorite\s+animal\s+([a-z]+)\b",
                r"\bfavourite\s+animal\s+([a-z]+)\b"
            ]
            for pattern in animal_patterns:
                match = _re.search(pattern, lower_text)
                if match:
                    animal = match.group(1).strip()
                    # Store via personal_brain
                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api
                        personal_api({
                            "op": "SET_USER_SLOT",
                            "payload": {
                                "slot_name": "favorite_animal",
                                "value": animal
                            }
                        })
                        return f"I'll remember that you like {animal}."
                    except Exception:
                        pass
                    break

            # Detect "food i like [food]" or "my favorite food is [food]" or "i like [food]" (when "food" appears in the sentence)
            food_patterns = [
                r"\bfood\s+i\s+like\s+([a-z]+)\b",
                r"\bi\s+like\s+([a-z]+).*\bfood",
                r"\bmy\s+favorite\s+food\s+is\s+([a-z]+)\b",
                r"\bmy\s+favourite\s+food\s+is\s+([a-z]+)\b",
                r"\bfavorite\s+food\s+([a-z]+)\b",
                r"\bfavourite\s+food\s+([a-z]+)\b"
            ]
            for pattern in food_patterns:
                match = _re.search(pattern, lower_text)
                if match:
                    food = match.group(1).strip()
                    # Store via personal_brain
                    try:
                        from brains.personal.service.personal_brain import service_api as personal_api
                        personal_api({
                            "op": "SET_USER_SLOT",
                            "payload": {
                                "slot_name": "favorite_food",
                                "value": food
                            }
                        })
                        return f"I'll remember that you like {food}."
                    except Exception:
                        pass
                    break
    except Exception:
        pass  # Fall through to normal processing

    # ------------------------------------------------------------------
    # PERSONAL QUESTION ROUTING
    # ------------------------------------------------------------------
    # Route personal questions to personal_brain.ANSWER_PERSONAL_QUESTION
    try:
        lower_text = text.strip().lower()

        # Detect "who am i" or "who am I"
        if lower_text in ["who am i", "who am i?"]:
            try:
                from brains.personal.service.personal_brain import service_api as personal_api
                resp = personal_api({
                    "op": "ANSWER_PERSONAL_QUESTION",
                    "payload": {
                        "question_type": "who_am_i"
                    }
                })
                if resp.get("ok"):
                    return resp.get("payload", {}).get("answer", "I don't know.")
            except Exception:
                pass

        # Detect "what color do i like" or "what do i like" with "color"
        if "what" in lower_text and "color" in lower_text and "like" in lower_text:
            try:
                from brains.personal.service.personal_brain import service_api as personal_api
                resp = personal_api({
                    "op": "ANSWER_PERSONAL_QUESTION",
                    "payload": {
                        "question_type": "what_color"
                    }
                })
                if resp.get("ok"):
                    return resp.get("payload", {}).get("answer", "I don't know.")
            except Exception:
                pass

        # Detect "what animal do i like"
        if "what" in lower_text and "animal" in lower_text and "like" in lower_text:
            try:
                from brains.personal.service.personal_brain import service_api as personal_api
                resp = personal_api({
                    "op": "ANSWER_PERSONAL_QUESTION",
                    "payload": {
                        "question_type": "what_animal"
                    }
                })
                if resp.get("ok"):
                    return resp.get("payload", {}).get("answer", "I don't know.")
            except Exception:
                pass

        # Detect "what food do i like"
        if "what" in lower_text and "food" in lower_text and "like" in lower_text:
            try:
                from brains.personal.service.personal_brain import service_api as personal_api
                resp = personal_api({
                    "op": "ANSWER_PERSONAL_QUESTION",
                    "payload": {
                        "question_type": "what_food"
                    }
                })
                if resp.get("ok"):
                    return resp.get("payload", {}).get("answer", "I don't know.")
            except Exception:
                pass

        # Detect generic "what do i like"
        if lower_text in ["what do i like", "what do i like?"]:
            try:
                from brains.personal.service.personal_brain import service_api as personal_api
                resp = personal_api({
                    "op": "ANSWER_PERSONAL_QUESTION",
                    "payload": {
                        "question_type": "what_do_i_like"
                    }
                })
                if resp.get("ok"):
                    return resp.get("payload", {}).get("answer", "I don't know.")
            except Exception:
                pass
    except Exception:
        pass  # Fall through to normal processing

    # Obtain parse metadata
    parsed = _parse_language(text)
    st_type = str(parsed.get("storable_type", ""))
    penalty = 0.0
    try:
        penalty = float(parsed.get("confidence_penalty") or 0.0)
    except Exception:
        penalty = 0.0

    # Generate message ID early (needed for research mode and all other operations)
    mid = generate_mid()

    # ------------------------------------------------------------------
    # RESEARCH MODE COMMANDS
    # ------------------------------------------------------------------
    # Detect research commands in various formats and route to research_manager
    # Supported formats:
    #   - "research: <topic>" or "research <topic>" → depth=2, offline
    #   - "deep research on <topic>" or "deep research <topic>" → depth=3, with web
    lower_text_cmd = text.lower().strip()

    # Check all research command patterns
    topic = None
    depth = 2
    default_web_flag = bool((CFG or {}).get("ENABLE_WEB_RESEARCH", True)) if 'CFG' in globals() else True
    use_web = default_web_flag

    if lower_text_cmd.startswith("deep research on "):
        topic = text[len("deep research on "):].strip()
        depth = 3
        use_web = True
    elif lower_text_cmd.startswith("deep research "):
        topic = text[len("deep research "):].strip()
        depth = 3
        use_web = True
    elif lower_text_cmd.startswith("research: "):
        topic = text[len("research: "):].strip()
        depth = 2
        use_web = default_web_flag
    elif lower_text_cmd.startswith("research "):
        # Check it's not "research:" with space after colon
        topic = text[len("research "):].strip()
        depth = 2
        use_web = default_web_flag

    if topic:
        # Research command detected
        if not topic:
            return "Please specify a topic to research. Example: 'research computers' or 'deep research photosynthesis'"

        topic, hint_web_enabled, time_budget_seconds, max_web_requests, hint_seen = _extract_web_settings(topic)
        final_web_enabled = hint_web_enabled if hint_seen else use_web

        print(f"[RESEARCH_MODE] Starting research on topic: {topic} (depth={depth}, web={final_web_enabled})")

        try:
            from brains.cognitive.research_manager.service.research_manager_brain import service_api as research_api

            # Build sources list based on web flag
            sources = ['memory', 'teacher']
            if final_web_enabled:
                sources.append('web')

            research_resp = research_api({
                "op": "RUN_RESEARCH",
                "mid": mid,
                "payload": {
                    "topic": topic,
                    "depth": depth,
                    "sources": sources,
                    "full_prompt": text,
                    "web_enabled": final_web_enabled,
                    "time_budget_seconds": time_budget_seconds,
                    "max_web_requests": max_web_requests,
                }
            })

            if research_resp.get("ok"):
                resp_payload = research_resp.get("payload") or {}
                summary = resp_payload.get("summary", "")
                # Get facts_collected from research_manager (not facts_stored)
                facts_collected = resp_payload.get("facts_collected", 0)
                sources_list = resp_payload.get("sources", [])
                sources_count = len(sources_list)

                if summary:
                    # Return the research summary with accurate metrics
                    result = f"{summary}\n\n"
                    result += f"Research complete: {facts_collected} facts stored from {sources_count} sources."
                    return result
                else:
                    return f"Research completed on '{topic}' but no summary was generated. {facts_collected} facts were stored."
            else:
                error = research_resp.get("error", "unknown error")
                return f"Research failed: {error}"

        except Exception as e:
            print(f"[RESEARCH_MODE_ERROR] {e}")
            return f"Research mode error: {str(e)[:100]}"

    # Determine high‑level intent
    intent = _interpret_intent(text, parsed)
    try:
        # Handle Self‑DMN maintenance operations and meta commands
        if intent == "dmn_tick":
            if self_dmn_brain is None:
                return "Self‑DMN brain unavailable."
            self_dmn_brain.service_api({"op": "TICK", "mid": mid})
            return "Self‑DMN tick complete."
        if intent == "dmn_reflect":
            if self_dmn_brain is None:
                return "Self‑DMN brain unavailable."
            # Extract a numeric window if present
            match = re.search(r"(\d+)", text)
            window = int(match.group(1)) if match else 10
            resp = self_dmn_brain.service_api({"op": "REFLECT", "mid": mid, "payload": {"window": window}})
            metrics = (resp.get("payload") or {}).get("metrics") or {}
            counts = metrics.get("counts", {})
            return f"Reflection complete: {counts.get('runs', 0)} runs analysed." if counts else "Reflection complete."
        if intent == "dmn_dissent":
            if self_dmn_brain is None:
                return "Self‑DMN brain unavailable."
            match = re.search(r"(\d+)", text)
            window = int(match.group(1)) if match else 10
            resp = self_dmn_brain.service_api({"op": "DISSENT_SCAN", "mid": mid, "payload": {"window": window}})
            flagged = (resp.get("payload") or {}).get("flagged") or []
            return f"Dissent scan complete: {len(flagged)} claims flagged." if flagged else "No dissent found."

        # Diagnostics request
        if intent == "diag":
            # Real diagnostic checks: runtime root, memory root, self-intent gate, memory system, Teacher
            from pathlib import Path as _Path
            failures = 0
            checks = []

            # 1. Check runtime root
            try:
                from brains.maven_paths import MAVEN_ROOT
                runtime_root = str(MAVEN_ROOT)
                if _Path(runtime_root).exists():
                    checks.append(f"runtime_root: OK ({runtime_root})")
                else:
                    checks.append(f"runtime_root: ERROR (path not found: {runtime_root})")
                    failures += 1
            except Exception as e:
                checks.append(f"runtime_root: ERROR ({str(e)[:60]})")
                failures += 1

            # 2. Check runtime memory root
            try:
                from brains.maven_paths import get_runtime_memory_root
                memory_root = get_runtime_memory_root()
                if memory_root.exists():
                    checks.append(f"memory_root: OK ({str(memory_root)})")
                else:
                    checks.append(f"memory_root: ERROR (expected {str(memory_root)}, not found)")
                    failures += 1
            except Exception as e:
                checks.append(f"memory_root: ERROR ({str(e)[:60]})")
                failures += 1

            # 3. Check self-intent gate (simulate a self-query internally)
            try:
                # Test if self-intent gate can detect a self-query
                from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api
                test_resp = self_model_api({
                    "op": "QUERY_SELF",
                    "payload": {
                        "query": "who are you",
                        "self_kind": "identity"
                    }
                })
                if test_resp.get("ok"):
                    checks.append("self_intent_gate: OK")
                else:
                    checks.append("self_intent_gate: ERROR (self_model returned error)")
                    failures += 1
            except Exception as e:
                checks.append(f"self_intent_gate: ERROR ({str(e)[:60]})")
                failures += 1

            # 4. Check memory system (sample bank read/write test)
            try:
                from brains.memory.brain_memory import BrainMemory
                test_bank = BrainMemory("personal")
                # Try to get stats (read test)
                stats = test_bank.get_stats()
                checks.append(f"sample_bank(personal): OK (read/write test passed)")
            except Exception as e:
                checks.append(f"sample_bank(personal): ERROR ({str(e)[:60]})")
                failures += 1

            # 5. Check Teacher connectivity (optional)
            try:
                from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api
                health = teacher_api({"op": "HEALTH", "mid": mid})
                if health.get("ok"):
                    checks.append("teacher: OK")
                else:
                    checks.append("teacher: ERROR (health check failed)")
                    failures += 1
            except Exception as e:
                checks.append(f"teacher: WARNING (not available: {str(e)[:40]})")

            # Build result
            result = f"Diagnostics: {failures} failure(s)\n"
            result += "\n".join([f"- {check}" for check in checks])
            return result

        # Status/health request
        if intent == "status":
            # Collate health from memory librarian and self‑DMN if available
            parts: list[str] = []
            if memory_librarian is not None:
                try:
                    hlth = memory_librarian.service_api({"op": "HEALTH", "mid": mid})
                    payload = hlth.get("payload") or {}
                    mh = payload.get("memory_health", {})
                    parts.append(f"Memory counts: STM={mh.get('stm', 0)}, MTM={mh.get('mtm', 0)}, LTM={mh.get('ltm', 0)}, COLD={mh.get('cold', 0)}")
                except Exception:
                    pass
            if self_dmn_brain is not None:
                try:
                    h = self_dmn_brain.service_api({"op": "HEALTH", "mid": mid})
                    pay = h.get("payload") or {}
                    parts.append(f"Self‑DMN status: {pay.get('status', 'unknown')}")
                except Exception:
                    pass
            if parts:
                return "; ".join(parts)
            return "Status unavailable."

        # Summary/report request
        if intent == "summary":
            try:
                import importlib
                sys_mod = importlib.import_module("brains.cognitive.system_history.service.system_history_brain")
                # Extract window if a number appears in the text
                match = re.search(r"(\d+)", text)
                window = int(match.group(1)) if match else 10
                res = sys_mod.service_api({"op": "SUMMARIZE", "mid": mid, "payload": {"window": window}})
                summ = (res.get("payload") or {}).get("summary") or {}
                agg = summ.get("aggregated", {})
                runs = agg.get("runs_analyzed", 0)
                decisions = agg.get("decisions", {})
                bank_use = agg.get("bank_usage", {})
                msg_parts = [f"Analysed {runs} runs"]
                if decisions:
                    dec_parts = [f"{k.lower()}: {v}" for k, v in decisions.items() if v]
                    if dec_parts:
                        msg_parts.append("decisions " + ", ".join(dec_parts))
                if bank_use:
                    bu_parts = [f"{b}: {c}" for b, c in bank_use.items()]
                    msg_parts.append("bank usage " + ", ".join(bu_parts))
                return "; ".join(msg_parts)
            except Exception:
                return "Could not generate summary."

        # Router explanation request
        if intent == "router_explain":
            # Explain which bank the routers would choose for the given text
            try:
                from importlib import import_module
                simple_bank = None
                learned_target = None
                learned_scores = None
                # Compute simple router suggestion if memory_librarian available
                if memory_librarian is not None:
                    try:
                        simple_bank = memory_librarian._simple_route_to_bank(text)
                    except Exception:
                        simple_bank = None
                # Compute learned router suggestion
                try:
                    lr_mod = import_module("brains.cognitive.reasoning.service.learned_router")
                    lr_resp = lr_mod.service_api({"op": "ROUTE", "payload": {"text": text}})
                    lp = lr_resp.get("payload") or {}
                    learned_target = lp.get("target_bank")
                    learned_scores = lp.get("scores")
                except Exception:
                    learned_target = None
                    learned_scores = None
                parts = []
                if simple_bank:
                    parts.append(f"Simple router suggests: {simple_bank}")
                if learned_target:
                    # Format scores for top few banks if available
                    score_desc = ""
                    if isinstance(learned_scores, dict):
                        # Get top 3 scores in descending order
                        try:
                            items = sorted(learned_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                            score_desc = ", ".join([f"{b}: {v:.2f}" for b, v in items])
                        except Exception:
                            score_desc = ""
                    if score_desc:
                        parts.append(f"Learned router suggests: {learned_target} (scores {score_desc})")
                    else:
                        parts.append(f"Learned router suggests: {learned_target}")
                if not parts:
                    return "Unable to determine routing explanation."
                return "; ".join(parts)
            except Exception:
                return "Could not explain routing."

        # Memory retrieval request
        if intent == "retrieve":
            if memory_librarian is None:
                return "Memory Librarian module unavailable."
            # Attempt to extract a query from the utterance.  Look for
            # patterns like 'for <text>' or 'about <text>'.  If none are
            # found, remove command words and treat the rest as the query.
            lower = text.lower()
            query: str | None = None
            m = re.search(r"(?:for|about)\s+(.+)", lower)
            if m:
                query = m.group(1).strip()
            else:
                # Remove the words 'search', 'find', 'lookup', 'memory'
                q = re.sub(r"\b(search|find|lookup|memory|for|about)\b", "", lower)
                query = q.strip()
            if not query:
                return "Please specify what to search for."
            # Perform retrieval across banks.  Use the librarian's internal helper
            # rather than a service op, since the service API does not
            # directly expose a RETRIEVE operation.  This helper returns
            # results aggregated from all banks with deduplication.
            try:
                results_data: Dict[str, Any] | None = None
                # Prefer the parallel implementation if available
                try:
                    # Some versions expose a parallel helper
                    results_data = memory_librarian._retrieve_from_banks_parallel(query, k=5)
                except Exception:
                    # Fallback to the serial implementation
                    results_data = memory_librarian._retrieve_from_banks(query, k=5)
                if not results_data:
                    return f"No memory entries found for '{query}'."
                res_list = results_data.get("results") or []
                if not res_list:
                    return f"No memory entries found for '{query}'."
                # Present up to three results with their bank names
                lines: list[str] = []
                for i, item in enumerate(res_list[:3], start=1):
                    content = item.get("content") or item.get("text") or str(item)
                    bank = item.get("source_bank") or item.get("bank") or "?"
                    summary = str(content)
                    # Truncate long summaries
                    if len(summary) > 60:
                        summary = summary[:57] + "..."
                    lines.append(f"{i}. {summary} (bank: {bank})")
                extra = "" if len(res_list) <= 3 else f" and {len(res_list) - 3} more"
                return f"Found {len(res_list)} result{'s' if len(res_list) != 1 else ''} for '{query}':\n" + "\n".join(lines) + extra
            except Exception:
                return "Failed to search memory."

        # Register claim request: defer execution until user confirms
        if intent == "register_claim":
            # Extract a proposition after the keyword 'claim' or 'register'
            prop = None
            m = re.search(r"claim\s+(.+)", text, flags=re.IGNORECASE)
            if m:
                prop = m.group(1).strip()
            if not prop:
                m = re.search(r"register\s+(.+)", text, flags=re.IGNORECASE)
                if m:
                    prop = m.group(1).strip()
            if not prop:
                return "Please specify a claim to register after the word 'claim'."

            # Define a closure that registers the claim when executed
            def _do_register_claim() -> str:
                # Import the skeptic module lazily
                import importlib  # type: ignore
                skeptic_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_skeptic")
                cid = f"CL-{int(time.time()*1000)}"
                payload = {
                    "claim_id": cid,
                    "proposition": prop,
                    "consensus_score": 0.5,
                    "skeptic_score": 0.5,
                    "expiry": time.time() + 24*3600,
                }
                res = skeptic_mod.service_api({"op":"REGISTER","mid": mid, "payload": payload})
                claim = (res.get("payload") or {}).get("claim") or payload
                return f"Registered claim {claim.get('claim_id')} with status {claim.get('status', 'unknown')}"
            # Save pending action and return prompt
            _PENDING_ACTION = (_do_register_claim, tuple(), {}, "register_claim")  # type: ignore
            return "This will register a claim in Self‑DMN. Proceed? (yes/no)"

        # Otherwise run the full pipeline (question or statement).  If the user
        # explicitly used a store/remember/save prefix, defer execution until
        # user confirmation to avoid accidental writes.  On confirmation,
        # ``_do_store`` will execute the pipeline with the cleaned text.
        if memory_librarian is None:
            return "Memory Librarian module unavailable."
        # Determine if the user explicitly wants to store information
        lowered = text.strip().lower()
        cleaned_text = text
        store_used = False
        for prefix in ["store ", "remember ", "save "]:
            if lowered.startswith(prefix):
                cleaned_text = text[len(prefix):].strip()
                store_used = True
                break
        # Compute the confidence outside of any closure to capture penalty
        conf = 1.0 - penalty
        if conf < 0.1:
            conf = 0.1
        if conf > 1.0:
            conf = 1.0
        # Helper to get current learning mode
        def _get_learning_mode():
            return LearningMode.TRAINING if _TRAINING_MODE_ENABLED else LearningMode.OFFLINE

        # If an explicit store command was used, set up a pending action
        if store_used:
            # use the module-level _PENDING_ACTION variable defined above
            def _do_store() -> str:
                resp_inner = memory_librarian.service_api({"op": "RUN_PIPELINE", "mid": mid, "payload": {"text": cleaned_text, "confidence": conf, "learning_mode": _get_learning_mode()}})
                context_inner = ((resp_inner or {}).get("payload") or {}).get("context") or {}
                ans = context_inner.get("final_answer")
                cval = context_inner.get("final_confidence")
                if ans:
                    return str(ans)
                return "I'm not sure how to respond to that."
            _PENDING_ACTION = (_do_store, tuple(), {}, "store_fact")  # type: ignore
            return "This action will store new information. Proceed? (yes/no)"

        # -------------------------------------------------------------------
        # MULTI-QUESTION PREPROCESSING: Split compound questions before pipeline
        # Handles cases like "what is a dog and what is lighting what is 32+44"
        # -------------------------------------------------------------------
        # NEW IMPLEMENTATION: Marker-based splitting for natural language + math
        import re as _re_multi
        lower_cleaned = cleaned_text.lower().strip()

        # Question markers to split on
        question_markers = [
            "what is", "what are", "who is", "who are",
            "tell me", "define", "explain"
        ]

        # Check if this looks like a multi-question query
        marker_count = sum(lower_cleaned.count(marker) for marker in question_markers)
        has_math = bool(_re_multi.search(r"\d+\s*[\+\-\*/]\s*\d+", cleaned_text))

        if marker_count > 1 or (marker_count >= 1 and has_math):
            # Try to split into sub-queries
            sub_q_list: List[str] = []

            # Step 1: Find all marker positions in lowercased text
            marker_positions = []
            for marker in question_markers:
                for match in _re_multi.finditer(_re_multi.escape(marker), lower_cleaned):
                    marker_positions.append((match.start(), marker))

            # Sort by position
            marker_positions.sort(key=lambda x: x[0])

            # Step 2: Extract questions between markers
            if marker_positions:
                for i, (start_pos, marker) in enumerate(marker_positions):
                    # Find end position (next marker or end of string)
                    if i + 1 < len(marker_positions):
                        end_pos = marker_positions[i + 1][0]
                    else:
                        end_pos = len(lower_cleaned)

                    # Extract segment from ORIGINAL text (preserve case)
                    segment = cleaned_text[start_pos:end_pos].strip()

                    # Clean up trailing "and" or "what"
                    segment = _re_multi.sub(r"\s+(and|what|who|tell|define|explain)\s*$", "", segment, flags=_re_multi.IGNORECASE).strip()

                    if segment and len(segment) > 4:
                        sub_q_list.append(segment)

            # Step 3: Extract math expressions separately
            math_pattern = r"(\d+\s*[\+\-\*/]\s*\d+)"
            for match in _re_multi.finditer(math_pattern, cleaned_text):
                expr = match.group(1).strip()
                if expr:
                    # Remove math from question segments
                    sub_q_list = [_re_multi.sub(math_pattern, "", q).strip() for q in sub_q_list]
                    # Add math as separate query
                    if expr not in sub_q_list:
                        sub_q_list.append(expr)

            # Clean up: remove empty strings, duplicates, and bare markers
            seen = set()
            clean_list = []
            # List of bare markers that should be filtered out
            bare_markers = ["what is", "what are", "who is", "who are", "tell me", "define", "explain"]
            for q in sub_q_list:
                # Filter out: empty, too short, duplicates, or bare markers without content
                if q and len(q) > 2 and q not in seen and q.lower() not in bare_markers:
                    seen.add(q)
                    clean_list.append(q)
            sub_q_list = clean_list

            # If we extracted multiple sub-queries, process them separately
            if len(sub_q_list) > 1:
                print(f"[MULTI_QUERY] Split into {len(sub_q_list)} sub-queries")
                answers: List[str] = []

                for sq_text in sub_q_list:
                    print(f"[MULTI_QUERY] Processing: {sq_text[:50]}")

                    # Check if this is a math expression
                    if _re_multi.match(r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*$", sq_text):
                        # Math query - evaluate directly
                        try:
                            from brains.agent.tools.math_tool import service_api as math_api
                            math_resp = math_api({"op": "CALC", "payload": {"expression": sq_text}})
                            if math_resp.get("ok"):
                                result = math_resp.get("payload", {}).get("result")
                                answers.append(f"{sq_text} = {result}")
                                print(f"[MULTI_QUERY_MATH] {sq_text} = {result}")
                            else:
                                answers.append(f"[Math error for: {sq_text}]")
                        except Exception as e:
                            print(f"[MULTI_QUERY_MATH_ERROR] {e}")
                            answers.append(f"[Math error for: {sq_text}]")
                    else:
                        # Factual query - send through full pipeline
                        try:
                            print(f"[MULTI_QUERY] Routing '{sq_text[:50]}' through full pipeline...")
                            sq_resp = memory_librarian.service_api({"op": "RUN_PIPELINE", "mid": mid, "payload": {"text": sq_text, "confidence": conf, "learning_mode": _get_learning_mode()}})
                            sq_payload = (sq_resp or {}).get("payload") or {}
                            sq_ctx = sq_payload.get("context") or {}

                            # PRIORITY 1: Check for Teacher answer in reasoning stage
                            reasoning = sq_ctx.get("stage_7_reasoning") or {}
                            reasoning_verdict = str(reasoning.get("verdict", "")).upper()

                            sq_ans = None
                            if reasoning_verdict == "LEARNED":
                                sq_ans = reasoning.get("answer")
                                if sq_ans:
                                    print(f"[MULTI_QUERY_TEACHER] Got Teacher answer: {str(sq_ans)[:60]}...")

                            # PRIORITY 2: Check final_answer from context
                            if not sq_ans:
                                sq_ans = sq_ctx.get("final_answer")
                                if sq_ans:
                                    print(f"[MULTI_QUERY_FINAL] Got final_answer: {str(sq_ans)[:60]}...")

                            # PRIORITY 3: Check language brain output
                            if not sq_ans:
                                lang_stage = sq_ctx.get("stage_8_language") or {}
                                sq_ans = lang_stage.get("text")
                                if sq_ans:
                                    print(f"[MULTI_QUERY_LANGUAGE] Got language brain: {str(sq_ans)[:60]}...")

                            # Validate the answer is not just echoing the question or a fallback message
                            if sq_ans and isinstance(sq_ans, str):
                                # Filter out fallback messages and question echoes
                                lower_ans = sq_ans.lower().strip()
                                if (not lower_ans.startswith("i don't yet have enough") and
                                    not lower_ans.startswith("i don't have enough") and
                                    lower_ans != sq_text.lower().strip() and
                                    len(sq_ans) > 10):
                                    answers.append(sq_ans)
                                else:
                                    print(f"[MULTI_QUERY_SKIP] Skipping fallback/echo answer")
                                    answers.append(f"[No reliable answer for: {sq_text}]")
                            else:
                                print(f"[MULTI_QUERY_EMPTY] No answer found")
                                answers.append(f"[No answer for: {sq_text}]")
                        except Exception as e:
                            print(f"[MULTI_QUERY_ERROR] {e}")
                            import traceback
                            traceback.print_exc()
                            answers.append(f"[Error for: {sq_text}]")

                # Return combined answers
                if answers:
                    combined = "\n\n".join(answers)
                    print(f"[MULTI_QUERY] Returning {len(answers)} combined answers")
                    return combined
                # If no answers were found, fall through to single query processing

        # -------------------------------------------------------------------
        # OLD MULTI-QUESTION PREPROCESSING (fallback for non-math cases)
        # -------------------------------------------------------------------
        sub_queries: List[Dict[str, Any]] = []
        if False and preprocess_query is not None and should_process_as_multi_query is not None:  # DISABLED
            try:
                sub_queries = preprocess_query(cleaned_text)
                if should_process_as_multi_query(sub_queries):
                    # Process each sub-query separately and combine answers
                    answers: List[str] = []
                    for sq in sub_queries:
                        sq_text = sq.get("text", "")
                        if sq_text:
                            sq_resp = memory_librarian.service_api({"op": "RUN_PIPELINE", "mid": mid, "payload": {"text": sq_text, "confidence": conf, "learning_mode": _get_learning_mode()}})
                            sq_ctx = ((sq_resp or {}).get("payload") or {}).get("context") or {}
                            sq_ans = sq_ctx.get("final_answer")
                            # Also check reasoning stage for Teacher answers
                            if not sq_ans:
                                reasoning = sq_ctx.get("stage_7_reasoning") or {}
                                sq_ans = reasoning.get("answer")
                            if sq_ans:
                                answers.append(f"{sq_ans}")
                            else:
                                answers.append(f"[No answer for: {sq_text[:40]}...]")
                    if answers:
                        return "\n".join(answers)
                    # If no answers were found, fall through to single query processing
            except Exception as e:
                # If preprocessing fails, fall through to single query
                pass

        # Otherwise, run the pipeline immediately for questions and statements (single query)
        resp = memory_librarian.service_api({"op": "RUN_PIPELINE", "mid": mid, "payload": {"text": cleaned_text, "confidence": conf, "learning_mode": _get_learning_mode()}})

        # Check if this is a bypassed pipeline response (SELF_INTENT_GATE or similar)
        payload = (resp or {}).get("payload") or {}
        verdict = payload.get("verdict")
        is_bypassed = payload.get("bypassed_pipeline", False)

        # CRITICAL: If this is a self-intent query (bypassed pipeline), use ONLY the self-model answer
        # NEVER fall back to Teacher, NEVER allow Teacher to override
        if is_bypassed or verdict in ("SKIP_STORAGE", "SELF_MODEL_DIRECT", "SELF_IDENTITY_DIRECT", "SELF_CODE_DIRECT", "SELF_RUNTIME_DIRECT"):
            # Self-model answered directly - this is the final authority
            answer = payload.get("answer")
            confidence = payload.get("confidence", 1.0)
            print(f"[CHAT] Self-intent query: using self-model answer (verdict={verdict})")

            # Store exchange and return immediately - no fallbacks, no Teacher override
            if correction_handler is not None:
                try:
                    words = cleaned_text.strip().split()
                    domain = " ".join(words[:2]) if len(words) >= 2 else words[0] if words else ""
                    conf_val = confidence if isinstance(confidence, float) else 1.0
                    correction_handler.set_last_exchange(
                        question=cleaned_text,
                        answer=str(answer) if answer else "",
                        confidence=conf_val,
                        domain=domain
                    )
                except Exception:
                    pass

            return str(answer) if answer else "I'm not sure how to respond to that."

        # Normal pipeline result (not bypassed)
        context = payload.get("context") or {}
        answer = context.get("final_answer")
        confidence = context.get("final_confidence")

        # FIX: Check reasoning stage for Teacher answers FIRST
        # Teacher answers have verdict="LEARNED" and are in stage_7_reasoning.answer
        # Do this BEFORE accepting a fallback message from language brain
        reasoning = context.get("stage_7_reasoning") or {}
        reasoning_verdict = str(reasoning.get("verdict", "")).upper()

        # If Teacher provided an answer with LEARNED verdict, use it regardless of what's in final_answer
        if reasoning_verdict == "LEARNED":
            teacher_answer = reasoning.get("answer")
            if teacher_answer:
                # Teacher answer takes priority over language brain fallback
                answer = teacher_answer
                confidence = reasoning.get("confidence", 0.7)
                print(f"[CHAT] Using Teacher answer (verdict=LEARNED): {str(answer)[:60]}...")

        # If we still don't have an answer OR the answer is a fallback message,
        # check if there's a Teacher answer we missed
        if not answer or (isinstance(answer, str) and "i don't yet have enough information" in answer.lower()):
            if reasoning_verdict == "LEARNED":
                teacher_answer = reasoning.get("answer")
                if teacher_answer:
                    answer = teacher_answer
                    confidence = reasoning.get("confidence", 0.7)
                    print(f"[CHAT] Replaced fallback with Teacher answer: {str(answer)[:60]}...")
        # If a final answer exists, publish it to the shared blackboard before returning
        if answer:
            # Best‑effort publish: do not raise errors if blackboard is unavailable
            try:
                from brains.agent.service import blackboard  # type: ignore
                payload: Dict[str, Any] = {
                    "type": "utterance",
                    "role": "assistant",
                    "text": answer,
                }
                # Include confidence if numeric
                try:
                    if isinstance(confidence, float):
                        payload["confidence"] = confidence
                except Exception:
                    pass
                blackboard.put("dialogue", payload)
            except Exception:
                # Silently ignore publish errors to avoid disrupting chat
                pass
            # Store this exchange for potential feedback processing
            if correction_handler is not None:
                try:
                    # Extract domain from the question (first 1-2 words)
                    words = cleaned_text.strip().split()
                    domain = " ".join(words[:2]) if len(words) >= 2 else words[0] if words else ""
                    conf_val = confidence if isinstance(confidence, float) else 0.4
                    correction_handler.set_last_exchange(
                        question=cleaned_text,
                        answer=str(answer),
                        confidence=conf_val,
                        domain=domain
                    )
                except Exception:
                    pass
            return str(answer)
        # ------------------------------------------------------------------
        # LLM fallback: When no final answer exists or the confidence is low or
        # the generic "I don't yet have enough information" message is
        # returned, attempt to generate a response using the local LLM.
        # This ensures the chat remains helpful even when the pipeline
        # provides no answer.  Only call the LLM when it is available.
        try:
            # Determine if the pipeline produced a fallback message
            fallback_trigger = False
            raw_ans = str(answer or "").strip().lower()
            if not answer:
                fallback_trigger = True
            elif raw_ans.startswith("i don't yet have enough information"):
                fallback_trigger = True
            # Fall back when the generic limitation message appears anywhere in the answer.  This catches
            # cases where the answer is quoted or has minor variations, e.g. "I don't yet have enough
            # information about photosynthesis simply to provide a summary."
            elif "i don't yet have enough information" in raw_ans:
                fallback_trigger = True
            # Also trigger when confidence is extremely low (<0.5)
            if isinstance(confidence, float) and confidence < 0.5:
                fallback_trigger = True
            if fallback_trigger:
                from brains.tools.llm_service import llm_service as _chat_llm  # type: ignore
                if _chat_llm is not None:
                    # Build a simple context with the session user name if available
                    user_name = None
                    try:
                        # Use the session identity from the pipeline context if present
                        user_name = context.get("session_identity") or None
                    except Exception:
                        user_name = None
                    call_ctx = {}
                    if user_name:
                        call_ctx["user"] = {"name": user_name}
                    # Generate a response directly from the LLM using the raw user text
                    llm_res = _chat_llm.call(prompt=text, context=call_ctx)
                    if llm_res and llm_res.get("ok") and llm_res.get("text"):
                        llm_text = str(llm_res.get("text"))
                        # Use provided confidence if available; default to 0.75
                        try:
                            cval = float(llm_res.get("confidence", 0.75) or 0.75)
                        except Exception:
                            cval = 0.75
                        # Store this exchange for potential feedback processing
                        if correction_handler is not None:
                            try:
                                words = cleaned_text.strip().split()
                                domain = " ".join(words[:2]) if len(words) >= 2 else words[0] if words else ""
                                correction_handler.set_last_exchange(
                                    question=cleaned_text,
                                    answer=llm_text,
                                    confidence=cval,
                                    domain=domain
                                )
                            except Exception:
                                pass
                        return llm_text
        except Exception:
            # Ignore any errors in LLM fallback to avoid crashing
            pass
        return "I'm not sure how to respond to that."
    except Exception as e:
        return f"An error occurred: {e}"


def repl() -> None:
    """Simple read‑eval‑print loop for interactive use."""
    # Activate profile from environment at startup
    _activate_profile_at_startup()

    print("Welcome to the Maven chat interface. Type 'exit' or 'quit' to leave.")

    # Print dynamic version information from Git + capabilities
    try:
        from version_utils import get_version_info

        version_info = get_version_info()
        print(
            "MAVEN BUILD: commit={commit} branch={branch} features={features}".format(
                commit=version_info.get("commit", "unknown"),
                branch=version_info.get("branch", "unknown"),
                features=version_info.get("features", "unknown"),
            )
        )
    except Exception:
        pass
    # Initialise a conversation log file on first entry.  We defer the
    # creation until here so that import of this module does not have
    # side effects.  The log directory is reports/agent/chat to reuse
    # the existing agent logging area.  No new top‑level folders are
    # created.  The filename includes the start timestamp.
    global _CONV_FILE
    if _CONV_FILE is None:
        try:
            log_dir = CHAT_LOG_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            import time
            _CONV_FILE = str(log_dir / f"conv_{int(time.time())}.jsonl")
        except Exception:
            _CONV_FILE = None
    pending_execution_enable = False
    _attachment_context = ""  # Store attachment context for queries

    # Print input handler help on startup
    if _input_handler_enabled:
        print("[Input commands: @file <path> to attach files, /paste for multi-line, ``` for code blocks]")

    while True:
        try:
            raw = input("You: ")
        except EOFError:
            break

        # Process input through enhanced handler (file uploads, paste mode)
        _attachment_context = ""
        if _input_handler_enabled and process_input:
            try:
                processed = process_input(raw)
                raw = processed.text  # Use processed text
                # Format any attachments as context
                if processed.attachments:
                    _attachment_context = format_attachments_for_context(processed.attachments)
                    print(f"[Attached {len(processed.attachments)} file(s)]")
                    for att in processed.attachments:
                        status = "✓" if not att.error else f"✗ {att.error}"
                        print(f"  - {att.name}: {status}")
                # Store URLs for later fetching
                if processed.metadata.get("urls_to_fetch"):
                    print(f"[URLs to fetch: {len(processed.metadata['urls_to_fetch'])}]")
                # Notify if paste was restored
                if processed.metadata.get("paste_restored"):
                    print("[Detected pasted code - newlines restored]")
            except Exception as e:
                print(f"[Input processing error: {e}]")

        # If nothing was entered, prompt again
        if raw is None:
            continue
        # Trim whitespace from both ends
        line = str(raw).strip()
        # Exit or quit commands (case-insensitive) before sanitisation
        if line.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if pending_execution_enable:
            if line.strip().upper() == "YES":
                try:
                    from brains.tools.execution_guard import set_execution_config

                    set_execution_config(True, True)
                    print("Execution enabled and persisted in ~/.maven/config.json")
                except Exception as e:
                    print(f"Failed to enable execution: {e}")
                pending_execution_enable = False
                continue
            else:
                print("Execution enablement cancelled (explicit YES required).")
                pending_execution_enable = False
                continue
        lower_line = line.lower()
        if lower_line in {"enable execution", "enable code execution", "i understand the risks, enable code execution"}:
            print("Enabling execution allows Maven to read/write files and run tools. Type YES to confirm.")
            pending_execution_enable = True
            continue
        if lower_line in {"disable execution", "turn off execution"}:
            try:
                from brains.tools.execution_guard import set_execution_config

                set_execution_config(False, False)
                print("Execution disabled and persisted in ~/.maven/config.json")
            except Exception as e:
                print(f"Failed to disable execution: {e}")
            continue
        # Training mode commands
        global _TRAINING_MODE_ENABLED
        if lower_line in {"enable training", "training on", "training mode on", "enable learning"}:
            _TRAINING_MODE_ENABLED = True
            print("Training mode ENABLED. Reasoning Brain will now call LLM for learning.")
            print("Lessons will be stored and strategies will be built from LLM responses.")
            continue
        if lower_line in {"disable training", "training off", "training mode off", "disable learning"}:
            _TRAINING_MODE_ENABLED = False
            print("Training mode DISABLED. System is now in OFFLINE mode (no LLM calls).")
            continue
        if lower_line in {"training status", "learning status", "show training"}:
            mode = "TRAINING" if _TRAINING_MODE_ENABLED else "OFFLINE"
            print(f"Current learning mode: {mode}")
            continue
        # Sanitize the input: remove an accidental leading "You:" prefix
        # Users sometimes paste the "You:" prompt back into the input; strip it off
        lower_line = line.lower()
        if lower_line.startswith("you:"):
            line = line[4:].strip()
        # Remove surrounding quotes if both ends have a double quote
        if line.startswith('"') and line.endswith('"') and len(line) >= 2:
            line = line[1:-1].strip()
        # If only one quote appears at either end, strip all quotes
        elif line.startswith('"') or line.endswith('"'):
            line = line.replace('"', '').strip()
        # After sanitisation, if the line is empty, skip
        if not line:
            continue

        # Prepend attachment context if files were attached
        query_with_context = line
        if _attachment_context:
            query_with_context = f"{_attachment_context}\n\n---\nUser query: {line}"

        # Determine intent ahead of processing so we can log it
        parsed = _parse_language(line)
        intent = _interpret_intent(line, parsed)
        response = process(query_with_context)
        # Log the turn (best effort)
        _log_turn(line, intent, response)
        print(f"Maven: {response}")


if __name__ == "__main__":
    # When run as a script, adjust sys.path to ensure Maven modules can be imported
    maven_root = MAVEN_ROOT
    if str(maven_root) not in sys.path:
        sys.path.insert(0, str(maven_root))
    repl()
