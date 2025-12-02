"""
Self‑Model Service
==================

The self‑model provides a reflective layer that allows Maven to
estimate whether it can answer a given query based on its current
beliefs.  It exposes a minimal API with a ``CAN_ANSWER`` operation
and a helper method ``can_answer`` for direct use by other modules.
This model does not attempt deep semantic understanding; instead it
uses heuristics over stored facts to judge its knowledge state.

Future enhancements may incorporate confidence calibration,
meta‑learning and a richer belief representation.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import json
import os
from pathlib import Path
from brains.memory.brain_memory import BrainMemory
from brains.maven_paths import get_maven_root, validate_path_confinement
from capabilities import get_capabilities as get_capability_registry
from brains.tools.execution_guard import get_execution_status, execution_status_snapshot

try:
    from brains.tools import fs_scan_tool, git_tool
except Exception as e:
    print(f"[SELF_MODEL] Tool layer unavailable: {e}")
    fs_scan_tool = None  # type: ignore
    git_tool = None  # type: ignore

# Teacher integration for learning self-definition and identity patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("self_model")
except Exception as e:
    print(f"[SELF_MODEL] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Identity inferencer for behavioral trait analysis
try:
    from brains.personal.service.identity_inferencer import (
        identity_snapshot_for_self_model,
        update_identity_profile
    )
    _identity_inferencer_available = True
except Exception as e:
    print(f"[SELF_MODEL] Identity inferencer not available: {e}")
    _identity_inferencer_available = False

# Continuous Introspector for periodic self-analysis and cleanup
try:
    from brains.cognitive.self_model.continuous_introspector import (
        ContinuousIntrospector,
        get_default_introspector,
        scan_for_upgrades,
    )
    _continuous_introspector = get_default_introspector()
    _continuous_introspector_available = True
except Exception as e:
    print(f"[SELF_MODEL] Continuous introspector not available: {e}")
    _continuous_introspector = None  # type: ignore
    _continuous_introspector_available = False

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[SELF_MODEL] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("self_model")

# Import the belief tracker to retrieve related beliefs.  If the
# module is unavailable (e.g. in older Maven versions), fallback to
# the internal stub implementation defined below.
try:
    from brains.cognitive.belief_tracker.service.belief_tracker import find_related_beliefs as _bt_find_related  # type: ignore
except Exception:
    _bt_find_related = None  # type: ignore


class SelfModel:
    """A simple self‑model for estimating answerability.

    The current implementation inspects a list of related beliefs to
    determine if any are sufficiently confident to support an answer.
    Callers are responsible for populating the belief list; this class
    does not perform retrieval itself.
    """

    def __init__(self) -> None:
        pass

    def handle_system_scan(self) -> str:
        root = get_maven_root()
        brains_ok = (root / "brains").is_dir()
        api_ok = (root / "api").is_dir()
        reports_ok = (root / "reports").is_dir()
        config_ok = (root / "config").is_dir()

        git_status_line = "- git status: tool unavailable"
        if git_tool:
            try:
                status = git_tool.git_status()
                first_line = status.strip().splitlines()[0] if status else "clean"
                git_status_line = f"- git status: {first_line}"
            except Exception as e:
                git_status_line = f"- git status: {str(e)}"

        # Get execution status
        exec_status_line = "- execution: unavailable"
        try:
            exec_snap = execution_status_snapshot()
            mode = exec_snap.get("mode", "UNKNOWN")
            effective = exec_snap.get("effective", False)
            source = exec_snap.get("source", "unknown")
            exec_status_line = f"- execution: mode={mode}, effective={effective}, source={source}"
        except Exception as e:
            exec_status_line = f"- execution: error ({e})"

        return (
            "System scan:\n"
            f"- root: {root}\n"
            f"- brains/: {'OK' if brains_ok else 'MISSING'}\n"
            f"- api/: {'OK' if api_ok else 'MISSING'}\n"
            f"- config/: {'OK' if config_ok else 'MISSING'}\n"
            f"- reports/: {'OK' if reports_ok else 'MISSING'}\n"
            f"{git_status_line}\n"
            f"{exec_status_line}"
        )

    def handle_routing_scan(self) -> str:
        root = get_maven_root()
        command_router_path = root / "brains" / "cognitive" / "command_router.py"
        routing_diag_path = root / "brains" / "cognitive" / "routing_diagnostics.py"
        sensorium_path = root / "brains" / "cognitive" / "sensorium"
        registry_path = root / "config" / "commands.json"

        try:
            if registry_path.exists():
                with open(registry_path, "r", encoding="utf-8") as fh:
                    registry_data = json.load(fh)
            else:
                registry_data = {}
        except Exception:
            registry_data = {}

        registered_commands = list(registry_data.keys()) if isinstance(registry_data, dict) else []

        lines = [
            "Routing scan:",
            f"- command_router.py: {'OK' if command_router_path.exists() else 'MISSING'}",
            f"- routing_diagnostics.py: {'OK' if routing_diag_path.exists() else 'MISSING'}",
            f"- sensorium brain: {'OK' if sensorium_path.exists() else 'MISSING'}",
        ]

        if registered_commands:
            lines.append(f"- commands.json entries: {len(registered_commands)} ({', '.join(registered_commands)})")
        else:
            lines.append("- commands.json entries: none")

        return "\n".join(lines)

    def handle_code_scan(self) -> str:
        summary = scan_own_code()
        brain_count = len(summary.get("brains", []))
        domain_count = len(summary.get("domain_banks", []))
        total_py = summary.get("total_python_files", 0)
        scanned_from = summary.get("scanned_from", "unknown")

        return (
            "Codebase scan:\n"
            f"- scanned from: {scanned_from}\n"
            f"- cognitive brains: {brain_count}\n"
            f"- domain banks: {domain_count}\n"
            f"- total Python files: {total_py}"
        )

    def handle_cognitive_scan(self) -> str:
        root = get_maven_root()
        brains_dir = root / "brains" / "cognitive"
        brains: List[str] = []
        if brains_dir.exists():
            try:
                brains = [p.name for p in sorted(brains_dir.iterdir()) if p.is_dir() and not p.name.startswith("_")]
            except Exception:
                brains = []

        lines = ["Cognitive brain scan:"]
        lines.append(f"- brains directory: {'OK' if brains_dir.exists() else 'MISSING'}")
        if brains:
            lines.append(f"- discovered brains ({len(brains)}): {', '.join(brains)}")
        else:
            lines.append("- discovered brains: none")

        return "\n".join(lines)

    def describe_capabilities(self) -> str:
        """
        Describe Maven's current capabilities using the REAL capability registry.

        This method provides HONEST, ACCURATE information about what Maven can and cannot do
        based on the actual runtime state (feature flags, execution status, module availability).

        CRITICAL: This uses actual probes, NOT Teacher/LLM, to avoid hallucination.

        Returns:
            Human-readable description of current capabilities and limitations.
        """
        print("[SELF_MODEL] Generating capability description from runtime capability registry...")

        try:
            # Try new system_capabilities module first (provides probe-based truth)
            try:
                from brains.system_capabilities import get_current_capabilities, get_capability_truth
                truth = get_capability_truth()
                current = get_current_capabilities()

                lines = []
                lines.append("Here's what I can actually do right now, based on runtime probes:\n")

                # Available capabilities
                if current.get("available"):
                    lines.append("AVAILABLE:")
                    for cap in current["available"]:
                        lines.append(f"  [+] {cap}")

                # Unavailable capabilities (with reasons)
                if current.get("unavailable"):
                    lines.append("\nNOT AVAILABLE:")
                    for cap in current["unavailable"]:
                        lines.append(f"  [-] {cap}")

                # Add summary
                lines.append(f"\n{truth.get('summary', '')}")

                # Core capabilities always available
                lines.append("\nCore capabilities (always available):")
                lines.append("  - Natural language understanding and response generation")
                lines.append("  - Memory storage and retrieval across multiple tiers")
                lines.append("  - Self-awareness and introspection of my own code and state")
                lines.append("  - Learning from conversations and pattern recognition")

                result = "\n".join(lines)
                print(f"[SELF_MODEL] Generated capability description from probes ({len(result)} chars)")
                return result

            except ImportError:
                pass

            # Fallback to legacy capabilities module
            from capabilities import describe_capabilities, get_capabilities, get_enabled_capabilities
            from brains.tools.execution_guard import get_execution_status

            # Get the full capability description
            caps_description = describe_capabilities()

            # Get execution status for context
            exec_status = get_execution_status()

            # Build the response
            lines = []
            lines.append("Here's what I can actually do right now, based on my current configuration:\n")
            lines.append(caps_description)
            lines.append("\n")

            # Add execution context if relevant
            if not exec_status["enabled"]:
                lines.append(f"Note: Code execution is currently disabled. {exec_status['reason']}")
                lines.append("This affects capabilities like filesystem operations and git commands.")

            # Add core reasoning capabilities that don't require execution
            lines.append("\n")
            lines.append("Core capabilities (always available):")
            lines.append("  - Natural language understanding and response generation")
            lines.append("  - Memory storage and retrieval across multiple tiers")
            lines.append("  - Self-awareness and introspection of my own code and state")
            lines.append("  - Learning from conversations and pattern recognition")
            lines.append("  - Reasoning and inference across multiple cognitive brains")

            result = "\n".join(lines)
            print(f"[SELF_MODEL] Generated capability description ({len(result)} chars)")
            return result

        except Exception as e:
            print(f"[SELF_MODEL] ERROR: Failed to get capabilities from registry: {e}")
            # Fallback: minimal honest response
            return (
                "I encountered an error accessing my capability registry. "
                "Please use the 'scan self' command to check my system status, "
                "or contact my developer if this persists."
            )

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get complete system status including capabilities, execution status, and paths.

        This method aggregates information from multiple sources:
        - Capability registry (feature flags, execution status)
        - Version information (git commit, branch, features)
        - Memory statistics
        - Code structure

        Returns:
            Dictionary with comprehensive system status.
        """
        print("[SELF_MODEL] Getting complete system status...")

        status = {
            "maven_root": str(get_maven_root()),
            "capabilities": {},
            "execution": {},
            "version": {},
            "memory": {},
            "code": {}
        }

        try:
            # Get capabilities
            from capabilities import get_capabilities, get_enabled_capabilities
            status["capabilities"] = get_capabilities()
            status["enabled_capabilities"] = get_enabled_capabilities()
        except Exception as e:
            print(f"[SELF_MODEL] Warning: Could not get capabilities: {e}")
            status["capabilities"] = {}

        try:
            # Get execution status
            from brains.tools.execution_guard import get_execution_status
            exec_status = get_execution_status()
            # Convert to dict for JSON serialization compatibility
            status["execution"] = exec_status.to_dict() if hasattr(exec_status, 'to_dict') else exec_status
        except Exception as e:
            print(f"[SELF_MODEL] Warning: Could not get execution status: {e}")
            status["execution"] = {"enabled": False, "error": str(e)}

        try:
            # Get version information
            from version_utils import get_version_info
            status["version"] = get_version_info()
        except Exception as e:
            print(f"[SELF_MODEL] Warning: Could not get version info: {e}")
            status["version"] = {"commit": "unknown", "branch": "unknown"}

        try:
            # Get memory stats
            memory_stats = get_memory_stats()
            status["memory"] = {
                "total_facts": memory_stats.get("total_facts", 0),
                "bank_count": len(memory_stats.get("banks", {})),
                "tiers": memory_stats.get("tiers_total", {})
            }
        except Exception as e:
            print(f"[SELF_MODEL] Warning: Could not get memory stats: {e}")
            status["memory"] = {"total_facts": 0, "bank_count": 0}

        try:
            # Get code summary
            code_summary = scan_own_code()
            status["code"] = {
                "total_brains": len(code_summary.get("brains", [])),
                "total_domain_banks": len(code_summary.get("domain_banks", [])),
                "total_python_files": code_summary.get("total_python_files", 0)
            }
        except Exception as e:
            print(f"[SELF_MODEL] Warning: Could not scan code: {e}")
            status["code"] = {"total_brains": 0, "total_domain_banks": 0}

        print(f"[SELF_MODEL] System status complete: {len(status.get('enabled_capabilities', []))} capabilities enabled")
        return status

    def answer_capability_question(self, question_text: str, runtime_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer a capability or upgrade question using REAL runtime state.

        This method provides HONEST, ACCURATE answers about what Maven can and
        cannot do based on:
        1. system_capabilities.py (runtime probes)
        2. execution_guard (execution status)
        3. feature flags and configuration

        CRITICAL: Never mentions Apache Maven / Java 17 - those are WRONG.

        Args:
            question_text: The user's question text
            runtime_state: Optional pre-computed runtime state

        Returns:
            Dict with 'answer', 'confidence', 'self_origin', and 'source' fields
        """
        print(f"[SELF_MODEL] Answering capability question: {question_text[:50]}...")

        try:
            # Get runtime state from system_capabilities
            from brains.system_capabilities import (
                get_capability_truth,
                get_current_capabilities,
                is_tool_available,
            )

            truth = get_capability_truth()
            current = get_current_capabilities()

            # Get execution status
            exec_status = {}
            try:
                exec_snap = execution_status_snapshot()
                exec_status = {
                    "mode": exec_snap.get("mode", "UNKNOWN"),
                    "effective": exec_snap.get("effective", False),
                    "source": exec_snap.get("source", "unknown"),
                }
            except Exception:
                exec_status = {"mode": "UNKNOWN", "effective": False}

            # Get feature flags
            feature_flags = {}
            try:
                caps = get_capability_registry()
                feature_flags = caps.get("features", {}) if isinstance(caps, dict) else {}
            except Exception:
                pass

            # Parse question to determine what's being asked
            ql = question_text.lower().strip()
            answer = ""
            specific_capability = None

            # Handle specific capability questions
            if "upgrade" in ql or "need" in ql:
                # "what upgrade do you need"
                answer = "I don't currently track required software upgrades internally. "
                if exec_status.get("mode") == "DISABLED":
                    answer += f"My execution mode is currently DISABLED ({exec_status.get('source', 'config')}). "
                else:
                    answer += f"My execution mode is: {exec_status.get('mode', 'UNKNOWN')}. "
                answer += "I use runtime probes to check my current capabilities rather than tracking upgrade requirements."

            elif "browse" in ql or "web" in ql or "internet" in ql:
                # "can you browse the web"
                web_search = truth.get("tools", {}).get("web_search", "unavailable")
                browser = truth.get("tools", {}).get("browser_runtime", "unavailable")

                if web_search == "available" and browser == "available":
                    answer = "Yes, I can browse the web. Both web search and visual browsing are available."
                elif web_search == "available":
                    answer = "I can search the web (web_search is available), but visual browser automation is not configured. "
                    browser_reason = truth.get("tool_details", {}).get("browser_runtime", {}).get("reason", "")
                    if browser_reason:
                        answer += f"Browser: {browser_reason}"
                else:
                    answer = "No, I cannot browse the web right now. "
                    web_reason = truth.get("tool_details", {}).get("web_search", {}).get("reason", "")
                    if web_reason:
                        answer += f"Web search: {web_reason}"
                specific_capability = "web_browsing"

            elif "code" in ql or "run" in ql or "execute" in ql:
                # "can you run code"
                shell = truth.get("tools", {}).get("shell", "unavailable")

                if shell == "available":
                    answer = "Yes, I can execute shell commands and run code. "
                    if exec_status.get("mode") == "FULL":
                        answer += "Execution mode is FULL (read/write access)."
                    elif exec_status.get("mode") == "READ_ONLY":
                        answer += "Execution mode is READ_ONLY (I can read but not write files)."
                else:
                    answer = "No, I cannot run code right now. "
                    shell_reason = truth.get("tool_details", {}).get("shell", {}).get("reason", "")
                    if shell_reason:
                        answer += f"Reason: {shell_reason}"
                specific_capability = "code_execution"

            elif "control" in ql or "program" in ql:
                # "can you control other programs"
                answer = "No, I cannot control other programs or perform autonomous actions without your explicit instruction. "
                answer += "I operate through a structured pipeline where each action requires going through my cognitive brains. "
                answer += "This is by design for safety and predictability."
                specific_capability = "program_control"

            elif "file" in ql or "read" in ql or "write" in ql or "change" in ql:
                # "can you read or change files"
                filesystem = truth.get("tools", {}).get("filesystem", "unavailable")

                if filesystem == "available":
                    answer = "Yes, I can access files. "
                    fs_details = truth.get("tool_details", {}).get("filesystem", {}).get("details", {})
                    scope = fs_details.get("scope", "unknown")
                    if scope == "read_write":
                        answer += "I have read/write access to files within my configured boundaries."
                    elif scope == "read_only":
                        answer += "I currently have read-only access to files."
                    else:
                        answer += f"Access scope: {scope}."
                else:
                    answer = "No, file access is not currently available. "
                    fs_reason = truth.get("tool_details", {}).get("filesystem", {}).get("reason", "")
                    if fs_reason:
                        answer += f"Reason: {fs_reason}"
                specific_capability = "filesystem"

            elif "what can" in ql or "what do" in ql or "capabilities" in ql:
                # General "what can you do" - use describe_capabilities
                answer = self.describe_capabilities()
                specific_capability = "general"

            else:
                # Default: provide general capability overview
                available = current.get("available", [])
                unavailable = current.get("unavailable", [])

                answer = "Based on my current runtime state:\n\n"
                if available:
                    answer += "AVAILABLE:\n"
                    for cap in available[:5]:
                        answer += f"  [+] {cap}\n"
                if unavailable:
                    answer += "\nNOT AVAILABLE:\n"
                    for cap in unavailable[:3]:
                        answer += f"  [-] {cap}\n"

                answer += f"\nExecution mode: {exec_status.get('mode', 'UNKNOWN')}"

            return {
                "answer": answer,
                "confidence": 1.0,  # Maximum confidence - reading from actual runtime state
                "self_origin": True,
                "source": "system_capabilities",
                "specific_capability": specific_capability,
                "runtime_state": {
                    "execution_mode": exec_status.get("mode"),
                    "tools_available": list(k for k, v in truth.get("tools", {}).items() if v == "available"),
                }
            }

        except Exception as e:
            print(f"[SELF_MODEL] Error answering capability question: {e}")
            return {
                "answer": f"I encountered an error checking my capabilities: {str(e)[:100]}. Please try 'scan self' for a full system check.",
                "confidence": 0.5,
                "self_origin": True,
                "source": "error_fallback",
                "error": str(e)
            }

    def find_related_beliefs(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve beliefs related to the query string.

        This method attempts to use the external belief tracker if
        available.  It falls back to an empty list when the belief
        tracker is not installed or an error occurs.  Each belief
        returned should contain at least a ``confidence`` key.

        Args:
            query: The user query string.
        Returns:
            A list of belief dictionaries, potentially empty.
        """
        # Prefer the belief tracker if present
        try:
            if _bt_find_related:
                return _bt_find_related(query) or []
        except Exception:
            # Ignore belief tracker errors and fall back to stub
            pass
        # Fallback stub: no beliefs available
        return []

    def can_answer(self, query: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Determine if the agent believes it can answer the query.

        This method retrieves related beliefs and checks whether the
        highest confidence exceeds a threshold (default 0.7).  If so it
        returns ``True`` along with the related beliefs; otherwise it
        returns ``False`` and an empty list.  Callers may adjust the
        threshold or extend this logic for more nuanced reasoning.

        Args:
            query: The user query.
        Returns:
            A tuple ``(can_answer, beliefs)`` where ``can_answer`` is
            ``True`` if the agent believes it can respond and
            ``beliefs`` contains the supporting evidence.
        """
        try:
            related = self.find_related_beliefs(query) or []
        except Exception:
            related = []
        if not related:
            return False, []
        # Extract the highest confidence from related beliefs
        try:
            highest = max((float(b.get("confidence", 0.0) or 0.0) for b in related))
        except Exception:
            highest = 0.0
        if highest > 0.7:
            return True, related
        return False, []

    # ------------------------------------------------------------------
    # New: load and provide self facts for direct identity queries.  The
    # self model maintains a minimal bank of immutable facts about the
    # agent (e.g. name, type, age policy).  These facts are stored in
    # ``brains/cognitive/self_model/memory/self_facts.json``.  The
    # ``query_self`` helper uses simple pattern matching to answer
    # questions like "who are you" or "how old are you".  It returns
    # both the response text and a flag indicating that the answer
    # originates from the self model.  Unsupported queries return
    # ``None`` so that callers may fallback to other modules.

    def _load_self_facts(self) -> Dict[str, Any]:
        """Load baseline self facts from BrainMemory.

        Returns an empty dict on error or if no facts exist.
        """
        try:
            results = _memory.retrieve(limit=1)
            if results:
                # Get the most recent self facts record
                content = results[0].get("content", {})
                return content if isinstance(content, dict) else {}
        except Exception:
            pass
        return {}

    def query_self(self, query: str, self_kind: Optional[str] = None, self_mode: Optional[str] = None) -> Dict[str, Any]:
        """Attempt to answer a self‑referential question.

        Inspect the provided query for common identity questions.  If a
        match is found, return a success dict with answer and confidence.
        When no appropriate self answer exists, return an error dict.

        CRITICAL: Identity ALWAYS comes from self_dmn.get_core_identity(),
        NEVER from Teacher or LLM generation.

        NOW WITH CONTINUATION AWARENESS:
        - Detects meta-questions about previous responses ("why did you say that?")
        - Provides self-explanation and clarification across turns
        - Maintains self-awareness continuity in conversations

        Args:
            query: Raw user query.
            self_kind: Optional hint about query type ("identity", "code", "memory")
            self_mode: Optional hint about memory mode ("stats", "health")
        Returns:
            A dict with ok, payload (text, confidence, self_origin) or error.
        """
        try:
            ql = (query or "").strip().lower()
        except Exception:
            return {
                "ok": False,
                "error": {
                    "code": "SELF_MODEL_FAILURE",
                    "message": "Self-model could not process the request."
                },
                "payload": {}
            }

        # CONTINUATION AWARENESS: Handle meta-questions about previous responses
        if _continuation_helpers_available:
            try:
                is_follow_up = is_continuation(query, {"query": query})

                if is_follow_up:
                    print(f"[SELF_MODEL] ✓ Detected continuation/meta-question")

                    # Get conversation context
                    conv_context = get_conversation_context()
                    last_topic = conv_context.get("last_topic", "")
                    last_response = conv_context.get("last_maven_response", "")

                    # Detect type of meta-question
                    if "why" in ql and ("say" in ql or "tell" in ql or "answer" in ql):
                        # "Why did you say that?" - Explain reasoning
                        print(f"[SELF_MODEL] Handling 'why' meta-question")

                        answer = f"You're asking about my previous response regarding {last_topic}. "
                        answer += f"I provided that information based on my self-knowledge system, which tracks my identity, capabilities, and internal structure. "

                        if "code" in last_topic.lower() or "brain" in last_topic.lower():
                            answer += "I scanned my actual source code to give you accurate information about my implementation. "
                        elif "identity" in last_topic.lower() or "who" in last_topic.lower():
                            answer += "I retrieved this from my core identity definition (self_dmn), which defines who I am. "

                        return {
                            "ok": True,
                            "payload": {
                                "text": answer,
                                "confidence": 0.85,
                                "self_origin": True,
                                "is_meta_explanation": True
                            }
                        }

                    elif "what" in ql and ("mean" in ql or "meant" in ql):
                        # "What did you mean by X?" - Clarify meaning
                        print(f"[SELF_MODEL] Handling 'what did you mean' clarification")

                        # Try to extract what phrase they're asking about
                        import re
                        # Look for "mean by X" pattern
                        match = re.search(r'mean (?:by )?(.+?)(?:\?|$)', ql)
                        phrase = match.group(1).strip() if match else "that"

                        answer = f"When I mentioned '{phrase}' in my previous response about {last_topic}, "
                        answer += f"I was referring to specific aspects of my self-model. "

                        # Check if phrase appears in last response for context
                        if last_response and phrase in last_response.lower():
                            # Extract surrounding context
                            response_lower = last_response.lower()
                            phrase_idx = response_lower.find(phrase)
                            context_start = max(0, phrase_idx - 50)
                            context_end = min(len(last_response), phrase_idx + len(phrase) + 50)
                            context = last_response[context_start:context_end]
                            answer += f"\n\nContext: '...{context}...'"

                        return {
                            "ok": True,
                            "payload": {
                                "text": answer,
                                "confidence": 0.80,
                                "self_origin": True,
                                "is_clarification": True
                            }
                        }

                    elif "did you" in ql or "have you" in ql:
                        # "Did you just say X?" - Confirm statement
                        print(f"[SELF_MODEL] Handling confirmation question")

                        # Extract potential claim
                        claim_match = re.search(r'(?:did you|have you) (?:just )?(?:say|tell|mention) (.+?)(?:\?|$)', ql)
                        claim = claim_match.group(1).strip() if claim_match else "that"

                        # Check if claim appears in last response
                        confirmation = False
                        if last_response:
                            confirmation = claim.lower() in last_response.lower()

                        answer = f"{'Yes' if confirmation else 'No'}, "
                        if confirmation:
                            answer += f"I did mention {claim} in my previous response about {last_topic}. "
                            answer += "This information came from my self-model knowledge base."
                        else:
                            answer += f"I don't recall mentioning {claim} in my previous response. "
                            answer += f"I was discussing {last_topic}, which may have been related but different."

                        return {
                            "ok": True,
                            "payload": {
                                "text": answer,
                                "confidence": 0.95,
                                "self_origin": True,
                                "is_confirmation": True,
                                "confirmed": confirmation
                            }
                        }

            except Exception as e:
                print(f"[SELF_MODEL] Warning: Continuation detection failed: {str(e)[:100]}")
                # Fall through to standard processing

        # PRIORITY ROUTING: Handle different self_kind values FIRST

        # Runtime queries: questions about Maven's root directory and runtime location
        if self_kind == "runtime":
            print(f"[SELF_MODEL] Handling self-runtime query via self_kind hint")

            # Get Maven's actual runtime root from maven_paths
            try:
                from brains.maven_paths import MAVEN_ROOT
                runtime_root = str(MAVEN_ROOT)
            except Exception:
                # Fallback: compute from current file location
                current_file = Path(__file__).resolve()
                runtime_root = str(current_file.parent.parent.parent.parent.parent)

            answer = f"I'm running from {runtime_root}."

            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,  # Maximum confidence - reading from local filesystem!
                    "self_origin": True
                }
            }

        # Code queries: questions about Maven's own source code
        if self_kind == "code":
            print(f"[SELF_MODEL] Handling self-code query via self_kind hint")

            # Scan the actual code from disk
            code_summary = scan_own_code()

            # Get core identity for context
            try:
                import importlib
                self_dmn_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_brain")
                core_identity = self_dmn_mod.get_core_identity()
            except Exception:
                core_identity = {
                    "name": "Maven",
                    "system_type": "offline synthetic cognition system",
                    "creator": "Josh / Hink"
                }

            name = str(core_identity.get("name", "Maven"))
            system_type = str(core_identity.get("system_type", "offline synthetic cognition system"))
            creator = str(core_identity.get("creator", "Josh / Hink"))

            # Build answer from REAL scanned data
            answer = f"I've scanned my actual source code. I'm {name}, {system_type}. "
            answer += f"I was created by {creator} and run locally from {code_summary['scanned_from']}. "

            brain_count = len(code_summary['brains'])
            domain_count = len(code_summary['domain_banks'])
            answer += f"\n\nI consist of {brain_count} cognitive brains and {domain_count} domain banks. "

            # List some key brains
            brain_names = [b['name'] for b in code_summary['brains'][:10]]
            if brain_names:
                answer += f"My cognitive brains include: {', '.join(brain_names)}"
                if brain_count > 10:
                    answer += f", and {brain_count - 10} more. "
                else:
                    answer += ". "

            answer += f"\n\nTotal Python files: {code_summary['total_python_files']}. "
            answer += "\n\nThis knowledge comes from scanning my actual source directory, NOT from Teacher or external databases."

            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,  # Maximum confidence - we read from disk!
                    "self_origin": True,
                    "code_scan": code_summary
                }
            }

        if self_kind == "system_scan":
            print(f"[SELF_MODEL] Handling self-system scan via self_kind hint")
            answer = self.handle_system_scan()
            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,
                    "self_origin": True,
                },
            }

        if self_kind == "routing_scan":
            print(f"[SELF_MODEL] Handling self-routing scan via self_kind hint")
            answer = self.handle_routing_scan()
            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,
                    "self_origin": True,
                },
            }

        if self_kind == "code_scan":
            print(f"[SELF_MODEL] Handling self-code scan via self_kind hint")
            answer = self.handle_code_scan()
            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,
                    "self_origin": True,
                },
            }

        if self_kind == "cognitive_scan":
            print(f"[SELF_MODEL] Handling self-cognitive scan via self_kind hint")
            answer = self.handle_cognitive_scan()
            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,
                    "self_origin": True,
                },
            }

        # Identity queries: questions about who Maven is
        if self_kind == "identity":
            print(f"[SELF_MODEL] Handling self-identity query via self_kind hint")

            # Get core identity from self_dmn (NOT from Teacher!)
            try:
                import importlib
                self_dmn_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_brain")
                core_identity = self_dmn_mod.get_core_identity()
                print(f"[SELF_MODEL] Got core_identity from self_dmn: {core_identity}")
            except Exception as e:
                print(f"[SELF_MODEL] Failed to load core_identity from self_dmn: {e}, using fallback")
                core_identity = {
                    "name": "Maven",
                    "is_llm": False,
                    "system_type": "offline synthetic cognition system",
                    "creator": "Josh / Hink"
                }

            name = str(core_identity.get("name", "Maven"))
            system_type = str(core_identity.get("system_type", "offline synthetic cognition system"))
            creator = str(core_identity.get("creator", "Josh / Hink"))

            # Build DYNAMIC identity answer using real introspection
            # This scans codebase, reads memory stats, and checks capabilities
            try:
                intro_result = generate_self_introduction(detail_level="standard")
                answer = intro_result.get("introduction_text", "")
                if not answer:
                    # Fallback to static if introspection failed
                    answer = f"I'm {name}, a {system_type}. I was created by {creator}."
                print(f"[SELF_MODEL] Built DYNAMIC identity answer from introspection")
            except Exception as e:
                print(f"[SELF_MODEL] Introspection failed: {e}, using static answer")
                answer = f"I'm {name}, a {system_type}. I was created by {creator}."
            print(f"[SELF_MODEL] Identity answer: '{answer[:100]}...'")

            result = {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,  # Maximum confidence - from core identity!
                    "self_origin": True
                }
            }
            print(f"[SELF_MODEL] Returning result: {result}")
            return result

        # Memory queries: questions about Maven's OWN stored facts, NOT general knowledge!
        if self_kind == "memory":
            print(f"[SELF_MODEL] Handling self-memory query via self_kind hint")

            answer = ""

            # Use self_mode to determine routing
            if self_mode == "health":
                # Health/scan request
                print("[SELF_MODEL] Running memory health scan...")
                health = scan_memory_health()
                answer = health.get("summary", "Memory health check completed.")
            elif self_mode == "stats":
                # Stats/counting request
                print("[SELF_MODEL] Getting memory stats...")
                stats = get_memory_stats()
                answer = stats.get("summary", "Memory stats retrieved.")
            else:
                # No mode specified, provide both
                print("[SELF_MODEL] Running both memory health and stats...")
                stats = get_memory_stats()
                health = scan_memory_health()
                answer = f"{stats.get('summary', '')} {health.get('summary', '')}"

            answer = answer.strip()
            if not answer:
                answer = "I inspected my memory system, but couldn't generate a summary."

            return {
                "ok": True,
                "payload": {
                    "text": answer,
                    "confidence": 1.0,  # Maximum confidence - reading from local storage!
                    "self_origin": True
                }
            }

        # Upgrade queries: questions about self-improvement and upgrade planning
        if self_kind == "upgrade":
            print(f"[SELF_MODEL] Handling self-upgrade planning via self_kind hint")

            upgrade_plan = plan_self_upgrade()

            return {
                "ok": True,
                "payload": {
                    "text": upgrade_plan["summary"],
                    "confidence": 0.95,
                    "self_origin": True,
                    "upgrade_plan": upgrade_plan
                }
            }

        # Get core identity from self_dmn (NOT from Teacher!)
        try:
            import importlib
            self_dmn_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_brain")
            core_identity = self_dmn_mod.get_core_identity()
        except Exception as e:
            print(f"[SELF_MODEL] Could not load core identity: {e}")
            core_identity = {
                "name": "Maven",
                "is_llm": False,
                "system_type": "offline synthetic cognition system",
                "creator": "Josh / Hink"
            }

        name = str(core_identity.get("name", "Maven"))
        system_type = str(core_identity.get("system_type", "offline synthetic cognition system"))
        is_llm = core_identity.get("is_llm", False)
        creator = str(core_identity.get("creator", "Josh / Hink"))

        # For backwards compatibility, also check stored facts for extended info
        facts = self._load_self_facts() or {}
        kind = system_type  # Use core identity, not stored facts
        # Extended self facts: creation date, co‑creator, archivist and purpose
        # NOTE: creator comes from core_identity, not from facts!
        creation_date = str(facts.get("creation_date")) if facts.get("creation_date") else None
        co_creator = str(facts.get("co_creator")) if facts.get("co_creator") else None
        archivist = str(facts.get("archivist")) if facts.get("archivist") else None
        purpose = str(facts.get("purpose")) if facts.get("purpose") else None
        # Recognise questions about identity (who/what) and age
        # Who/what queries
        try:
            import re

            # ======================================================================
            # TASK 3.1: HANDLE "WHO CREATED YOU" QUESTIONS
            # ======================================================================
            # These MUST be answered from self_dmn.get_core_identity(), NEVER Teacher
            # ======================================================================
            if re.search(r"\b(who\s+(?:created|made|built|designed|wrote|programmed)\s+you|your\s+creator|who\s+is\s+your\s+(?:creator|author|developer|architect))\b", ql):
                print("[SELF_MODEL] Handling 'who created you' question from core identity")
                # creator is already loaded from core_identity above
                answer = f"I was created by {creator}."
                if core_identity.get("system_type"):
                    answer += f" I'm {name}, a {system_type}."
                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 1.0,  # Maximum confidence - from core identity card
                        "self_origin": True,
                        "source_brain": "self_model"  # For filtering in language_brain
                    }
                }

            # ======================================================================
            # TASK 3.2: HANDLE "WHAT DO YOU REMEMBER ABOUT ME" QUESTIONS
            # ======================================================================
            # These should query personal/preference banks for actual USER facts,
            # NOT just return memory stats. Only fall back to stats if no personal
            # facts exist.
            # ======================================================================
            if re.search(r"\b(what\s+(?:do\s+you\s+)?(?:remember|know)\s+about\s+me|what\s+have\s+you\s+learned\s+(?:about\s+me|so\s+far)|what.*most\s+important.*know\s+about\s+me)\b", ql):
                print("[SELF_MODEL] Handling 'what do you remember about me' question")

                # Query personal banks for actual user-facing facts
                personal_facts = []
                try:
                    # Check personal bank
                    personal_memory = BrainMemory("personal")
                    personal_results = personal_memory.retrieve(
                        query="user preference fact",
                        limit=10,
                        tiers=["stm", "mtm", "ltm"]
                    )

                    for rec in personal_results:
                        content = rec.get("content", {})
                        metadata = rec.get("metadata", {})
                        confidence = metadata.get("confidence", 0.5)

                        # Only include high-confidence personal facts
                        if confidence >= 0.6:
                            if isinstance(content, dict):
                                # Extract human-readable fact
                                fact_text = content.get("fact", "") or content.get("content", "") or content.get("preference", "")
                                if fact_text and len(fact_text) < 200:
                                    personal_facts.append(fact_text)
                            elif isinstance(content, str) and len(content) < 200:
                                personal_facts.append(content)
                except Exception as e:
                    print(f"[SELF_MODEL] Personal bank lookup error: {e}")

                # Also check identity/preference brains
                try:
                    identity_memory = BrainMemory("identity_user_store")
                    identity_results = identity_memory.retrieve(
                        query="user",
                        limit=5,
                        tiers=["stm", "mtm", "ltm"]
                    )
                    for rec in identity_results:
                        content = rec.get("content", {})
                        if isinstance(content, dict):
                            for key, value in content.items():
                                if key in ["name", "nickname", "preference", "likes", "dislikes"]:
                                    personal_facts.append(f"Your {key}: {value}")
                except Exception:
                    pass

                # Build answer
                if personal_facts:
                    # Have real personal facts about the user
                    unique_facts = list(dict.fromkeys(personal_facts[:5]))  # Dedupe, limit to 5
                    answer = "Here's what I remember about you:\n\n"
                    for fact in unique_facts:
                        answer += f"- {fact}\n"
                    answer += "\nThis comes from my personal memory banks, not from guessing."
                else:
                    # No personal facts yet - be honest
                    answer = "I don't have any personal information stored about you yet. "
                    answer += "As we interact, I'll learn your preferences and important details. "
                    answer += "Feel free to tell me about yourself if you'd like me to remember something."

                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 0.95,
                        "self_origin": True,
                        "source_brain": "self_model",
                        "personal_facts_count": len(personal_facts)
                    }
                }

            # CRITICAL: Check for "are you an LLM" first and answer explicitly
            if re.search(r"\b(are\s+you\s+(an?\s+)?llm|are\s+you\s+(a\s+)?large\s+language\s+model|are\s+you\s+chatgpt|are\s+you\s+claude|are\s+you\s+gpt)\b", ql):
                # HARD RULE: Maven is NOT an LLM
                answer = f"No, I'm not an LLM. I'm {name}, {system_type}. "
                answer += f"I was created by {creator} and run locally from {core_identity.get('home_directory', 'maven2_fix')}. "
                answer += "I use an external LLM (Teacher) for language generation and learning patterns, but I'm not an LLM myself. "
                answer += "All my reasoning, memory, and decisions are handled by my own specialized cognitive brains."
                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 1.0,  # Maximum confidence - this is a core fact
                        "self_origin": True
                    }
                }

            # Step B: Handle SELF-MEMORY questions (scan your memory, memory stats, what have you learned, etc.)
            # CRITICAL: These questions are about Maven's OWN memory/learning, NOT general knowledge!
            # They must NEVER go to Teacher - only internal memory inspection.
            memory_patterns = [
                r"\bscan\s+your\s+memory",
                r"\bscan\s+your\s+memory\s+system",
                r"\bdiagnose\s+your\s+memory",
                r"\bdiagnose\s+memory",
                r"\bwhat.*you\s+know\s+about\s+your\s+memory",
                r"\bhow\s+many\s+facts.*you.*learned",
                r"\bhow\s+much.*you.*learned",
                r"\bhow\s+many\s+facts.*you.*know",
                r"\bwhat.*you.*learned\s+so\s+far",
                r"\bwhat.*you\s+remember",
                r"\byour\s+memory\s+(system|health|stats|status)",
                r"\bmemory\s+(stats|statistics|count|summary)"
            ]

            is_memory_query = any(re.search(pattern, ql) for pattern in memory_patterns)

            if is_memory_query:
                print("[SELF_MODEL] Detected self-memory question, inspecting local memory system...")

                # Determine if this is a health scan or stats request
                is_health_scan = bool(re.search(r"\b(scan|diagnose|health|status)\b", ql))
                is_stats_request = bool(re.search(r"\b(how\s+many|how\s+much|stats|statistics|count|summary|learned|know)\b", ql))

                answer = ""

                # If it's a health/scan request, run scan_memory_health
                if is_health_scan:
                    print("[SELF_MODEL] Running memory health scan...")
                    health = scan_memory_health()
                    answer = health.get("summary", "Memory health check completed.")

                # If it's a stats request, run get_memory_stats
                if is_stats_request:
                    print("[SELF_MODEL] Getting memory stats...")
                    stats = get_memory_stats()
                    answer = stats.get("summary", "Memory stats retrieved.")

                # If both or neither matched, provide both
                if not answer or (is_health_scan and is_stats_request):
                    print("[SELF_MODEL] Running both memory health and stats...")
                    stats = get_memory_stats()
                    health = scan_memory_health()
                    answer = f"{stats.get('summary', '')} {health.get('summary', '')}"

                answer = answer.strip()
                if not answer:
                    answer = "I inspected my memory system, but couldn't generate a summary."

                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 1.0,  # Maximum confidence - reading from local storage!
                        "self_origin": True
                    }
                }

            # Step C: Handle self-knowledge questions (what do you know about yourself, your code, etc.)
            # These patterns trigger the REAL code scanner - reading from disk, NOT Teacher!
            if re.search(r"\b(what\s+do\s+you\s+know\s+about\s+(yourself|your\s+(own\s+)?code|your\s+systems|your\s+brains|your\s+architecture)|describe\s+your\s+code|how\s+are\s+you\s+built|what\s+are\s+your\s+brains|list\s+your\s+brains)\b", ql):
                print("[SELF_MODEL] Detected self-code question, scanning actual source files...")

                # Scan the actual code from disk (NOT Teacher!)
                code_summary = scan_own_code()

                # Build answer from REAL scanned data
                answer = f"I've scanned my actual source code. I'm {name}, {system_type}. "
                answer += f"I was created by {creator} and run locally from {code_summary['scanned_from']}. "

                brain_count = len(code_summary['brains'])
                domain_count = len(code_summary['domain_banks'])
                answer += f"\n\nI consist of {brain_count} cognitive brains and {domain_count} domain banks. "

                # List some key brains
                brain_names = [b['name'] for b in code_summary['brains'][:10]]
                if brain_names:
                    answer += f"My cognitive brains include: {', '.join(brain_names)}"
                    if brain_count > 10:
                        answer += f", and {brain_count - 10} more. "
                    else:
                        answer += ". "

                answer += f"\n\nTotal Python files: {code_summary['total_python_files']}. "
                answer += f"Memory architecture: {core_identity.get('architectural_facts', {}).get('memory_architecture', 'tiered memory system')}. "
                answer += "\n\nThis knowledge comes from scanning my actual source directory, NOT from Teacher or external databases."

                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 1.0,  # Maximum confidence - we read from disk!
                        "self_origin": True,
                        "code_scan": code_summary  # Include full scan data
                    }
                }

            # Step B: Handle "how do you work" questions
            if re.search(r"\b(how\s+do\s+you\s+work|how\s+do\s+you\s+function|how\s+does\s+maven\s+work)\b", ql):
                answer = f"I'm {name}, {system_type}. "
                answer += "I operate through a 14-stage broadcast pipeline with 27 specialized cognitive brains. "
                answer += "Each brain handles specific tasks: language understanding, memory, reasoning, planning, and more. "
                answer += "My responses come from deterministic brain operations, not random LLM generation. "
                answer += "I use Teacher (an external LLM) only for language pattern learning, but all my core decisions are mine."
                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 0.95,
                        "self_origin": True
                    }
                }

            # Step B: Handle "where do you run" questions
            if re.search(r"\b(where\s+do\s+you\s+run|where\s+are\s+you\s+running|where\s+is\s+maven\s+running)\b", ql):
                answer = f"I run locally from {core_identity.get('home_directory', 'maven2_fix')} on this machine. "
                answer += "I'm not a cloud service or remote API. All my processing, memory, and reasoning happen here locally. "
                answer += "My runtime memory is stored in maven2_fix/brains/runtime_memory/, and all my brain code lives in maven2_fix/brains/."
                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 0.95,
                        "self_origin": True
                    }
                }

            # Match phrases like "who are you", "who you are", "what is your name",
            # "what's your name", "are you maven", etc.
            if re.search(r"\b(who\s+are\s+you|who\s+you\s+are|what\s+is\s+your\s+name|what's\s+your\s+name|tell\s+me\s+about\s+yourself|are\s+you\s+maven)\b", ql):
                # HARD RULE: Identity always comes from Maven's own card, NOT from Teacher
                # Query the personal identity brain for Maven's identity
                try:
                    from brains.personal.service import identity_user_store
                    identity_card = identity_user_store.GET()
                    # Check if we have Maven's identity (not the user's)
                    # Maven's identity should be in self_facts from memory
                    pass
                except Exception:
                    pass

                # Compose identity from self facts (the source of truth)
                # Compose a richer self description by including extended self facts when available.
                # Start with the basic identity.  Then, if creation details exist, append them in one
                # coherent sentence.  Finally, mention the purpose if provided.
                parts = [f"I'm {name}, a {kind}."]
                # Include creation information when present
                if creation_date or creator or co_creator or archivist:
                    creation_bits = []
                    if creation_date:
                        creation_bits.append(f"in {creation_date}")
                    if creator:
                        creation_bits.append(f"by my architect {creator}")
                    if co_creator:
                        creation_bits.append(f"with the help of {co_creator}")
                    if archivist:
                        creation_bits.append(f"and documented by {archivist}")
                    # Join bits into a human‑readable fragment
                    creation_phrase = ", ".join(creation_bits)
                    parts.append(f"I was created {creation_phrase}.")
                # Include purpose
                if purpose:
                    parts.append(f"My purpose is {purpose}.")
                answer = " ".join(parts)

                # DO NOT CALL TEACHER FOR IDENTITY
                # Identity card is the source of truth, not LLM guesses

                return {
                    "ok": True,
                    "payload": {
                        "text": answer,
                        "confidence": 0.92,
                        "self_origin": True
                    }
                }
            # Age queries: "how old are you", "how old you", "your age"
            if re.search(r"\bhow\s+old\s+are\s+you\b", ql) or re.search(r"\bhow\s+old\s+you\b", ql) or re.search(r"\byour\s+age\b", ql):
                has_age = bool(facts.get("has_age", False))
                if not has_age:
                    # The agent does not have a biological age.  Offer to share
                    # uptime if available.  The actual uptime reporting is
                    # delegated to other modules, so only mention the
                    # possibility here.
                    return {
                        "ok": True,
                        "payload": {
                            "text": "I don't have a biological age; I'm software. I can share my uptime if you want.",
                            "confidence": 0.95,
                            "self_origin": True
                        }
                    }
                # If an age is explicitly provided in the facts, use it.
                explicit_age = facts.get("age")
                if explicit_age:
                    return {
                        "ok": True,
                        "payload": {
                            "text": f"I'm {explicit_age}.",
                            "confidence": 0.90,
                            "self_origin": True
                        }
                    }
                # Fallback: no age information
                return {
                    "ok": True,
                    "payload": {
                        "text": "I don't have a biological age; I'm software.",
                        "confidence": 0.90,
                        "self_origin": True
                    }
                }
            # Location queries: "where are you", "where are we"
            if re.search(r"\bwhere\s+are\s+you\b", ql) or re.search(r"\bwhere\s+are\s+we\b", ql):
                return {
                    "ok": True,
                    "payload": {
                        "text": "I'm a digital system running on a server, so I don't occupy a physical location like a person does.",
                        "confidence": 0.88,
                        "self_origin": True
                    }
                }

            # Capabilities queries.  Users may ask what the assistant can do or is capable of.
            # Match phrases like "what can you do", "what do you do", "what are your capabilities",
            # "what are you capable of", and related variations.  If detected, use the REAL
            # capability registry to provide HONEST, ACCURATE information about what Maven can/cannot do.
            try:
                if re.search(r"\bwhat\s+(?:can|do)\s+you\s+(?:do)?\b", ql) or \
                   re.search(r"\bwhat\s+are\s+(?:your|you)\s+(?:capabilities|abilities|skills)\b", ql) or \
                   re.search(r"\bwhat\s+are\s+you\s+capable\b", ql):
                    # CRITICAL: Use the REAL capability registry, NOT generic boilerplate
                    print("[SELF_MODEL] Detected capability query, using real capability registry...")
                    cap_reply = self.describe_capabilities()
                    return {
                        "ok": True,
                        "payload": {
                            "text": cap_reply,
                            "confidence": 1.0,  # Maximum confidence - reading from actual runtime state!
                            "self_origin": True
                        }
                    }
            except Exception as e:
                print(f"[SELF_MODEL] Warning: Capability query handler failed: {e}")
                pass

            # Preference/likes queries.  Detect when a question asks about
            # Maven's personal tastes using a robust token‑based approach.
            # Examples include "what do you like", "do you prefer", "what are your
            # preferences", "do you enjoy", "are you into", or misspellings like
            # "likee" and "preferances".  We require the presence of a second
            # person pronoun (you/your/yourself) together with a token that
            # begins with a recognised preference root.  This avoids triggering
            # on arbitrary uses of preference words that do not involve
            # the assistant.  If detected, return a generic explanation that the
            # assistant has no personal preferences.
            try:
                import re
                tokens = re.findall(r"\b\w+\b", ql)
                pronouns = {"you", "your", "yourself"}
                pref_roots = [
                    "like", "lik", "prefer", "preferenc", "preferanc",
                    "favor", "favour", "favorite", "favourite",
                    "enjoy", "into", "interested", "love"
                ]
                found_pronoun = any(tok in pronouns for tok in tokens)
                found_pref = any(any(tok.startswith(root) for root in pref_roots) for tok in tokens)
                if found_pronoun and found_pref:
                    pref_reply = (
                        "I don't have personal likes or preferences—I'm a software "
                        "assistant designed to help answer questions, provide information "
                        "and assist with tasks. If you have something specific you'd like "
                        "to know or do, feel free to ask!"
                    )
                    return {
                        "ok": True,
                        "payload": {
                            "text": pref_reply,
                            "confidence": 0.88,
                            "self_origin": True
                        }
                    }
            except Exception:
                pass
        except Exception:
            # On any regex error, fall back to no answer
            pass
        return {
            "ok": False,
            "error": {
                "code": "NO_SELF_ANSWER",
                "message": "Unsupported self query"
            },
            "payload": {}
        }


def scan_own_code(base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan Maven's own source code to build a structural summary.

    This function walks through maven2_fix/ directory and catalogs:
    - Cognitive brains
    - Domain banks
    - Key files and their purposes

    This is the REAL introspection - reading actual files from disk,
    NOT asking Teacher about Apache Maven!

    Args:
        base_path: Optional path to maven2_fix/. If None, auto-detect from current file.

    Returns:
        Dictionary with:
        - brains: list of cognitive brains with their files and purposes
        - domain_banks: list of domain banks
        - total_python_files: count of .py files
        - structure: overview of directory structure
    """
    if base_path is None:
        base_path_obj = get_maven_root()
    else:
        base_path_obj = validate_path_confinement(base_path, "self_model_scan")

    base_path_obj = Path(base_path_obj)
    brains_dir = base_path_obj / "brains" / "cognitive"
    domain_banks_dir = base_path_obj / "brains" / "domain_banks"

    result = {
        "brains": [],
        "domain_banks": [],
        "total_python_files": 0,
        "structure": {},
        "scanned_from": str(base_path_obj)
    }

    scanned_files = []
    if fs_scan_tool:
        try:
            scanned_files = fs_scan_tool.scan_codebase(root=base_path_obj, pattern="*.py")
        except Exception as e:
            print(f"[SELF_MODEL_CODE_SCAN] Tool scan failed: {e}")

    # Scan cognitive brains
    try:
        if brains_dir.exists():
            for brain_dir in sorted(brains_dir.iterdir()):
                if brain_dir.is_dir() and not brain_dir.name.startswith("_"):
                    brain_info = {
                        "name": brain_dir.name,
                        "files": [],
                        "purpose": ""
                    }

                    # Find Python files in this brain (prefer tool output)
                    py_files = []
                    if scanned_files:
                        for entry in scanned_files:
                            path_obj = Path(entry.get("path", ""))
                            try:
                                path_obj.relative_to(Path("brains") / "cognitive" / brain_dir.name)
                                rel_path = path_obj.relative_to(Path("brains") / "cognitive" / brain_dir.name)
                                py_files.append(str(rel_path))
                            except Exception:
                                continue
                    else:
                        for root, dirs, files in os.walk(brain_dir):
                            for file in files:
                                if file.endswith(".py"):
                                    rel_path = os.path.relpath(
                                        os.path.join(root, file),
                                        brain_dir
                                    )
                                    py_files.append(rel_path)

                    brain_info["files"] = py_files

                    # Try to extract purpose from main brain file
                    main_brain_file = brain_dir / "service" / f"{brain_dir.name}_brain.py"
                    if main_brain_file.exists():
                        try:
                            with open(main_brain_file, 'r', encoding='utf-8') as f:
                                content = f.read(500)
                                if '"""' in content:
                                    parts = content.split('"""')
                                    if len(parts) >= 2:
                                        docstring = parts[1].strip()
                                        first_line = docstring.split('\n')[0].strip()
                                        if first_line:
                                            brain_info["purpose"] = first_line[:100]
                        except Exception:
                            pass

                    result["brains"].append(brain_info)
    except Exception as e:
        print(f"[SELF_MODEL_CODE_SCAN] Error scanning brains: {e}")

    # Scan domain banks
    try:
        if domain_banks_dir.exists():
            for bank_dir in sorted(domain_banks_dir.iterdir()):
                if bank_dir.is_dir() and not bank_dir.name.startswith("_"):
                    result["domain_banks"].append(bank_dir.name)
    except Exception as e:
        print(f"[SELF_MODEL_CODE_SCAN] Error scanning domain banks: {e}")

    # Add structure overview
    result["structure"] = {
        "cognitive_brains_count": len(result["brains"]),
        "domain_banks_count": len(result["domain_banks"]),
        "has_pipeline": (base_path_obj / "brains" / "pipeline").exists(),
        "has_memory_system": (base_path_obj / "memory_system").exists(),
    }

    if scanned_files:
        result["total_python_files"] = len(scanned_files)
    else:
        try:
            total = 0
            for _root, _dirs, files in os.walk(base_path_obj):
                total += len([f for f in files if f.endswith(".py")])
            result["total_python_files"] = total
        except Exception:
            result["total_python_files"] = 0

    # Store the scan results in memory for future reference
    try:
        _memory.store(
            content=result,
            metadata={
                "kind": "self_code_summary",
                "source": "self_model_code_scanner",
                "confidence": 1.0,
                "scope": "self_core"
            }
        )
        print(f"[SELF_MODEL_CODE_SCAN] Stored code summary: {len(result['brains'])} brains, {result['total_python_files']} Python files")
    except Exception as e:
        print(f"[SELF_MODEL_CODE_SCAN] Could not store scan results: {e}")

    return result


def get_memory_stats() -> Dict[str, Any]:
    """
    Return a structured summary of Maven's stored facts across all memory banks.

    Uses ONLY the local BrainMemory API and domain banks - NO Teacher calls.

    Returns:
        Dictionary with:
        - total_facts: total count across all banks and tiers
        - banks: dict mapping bank name -> tier counts
        - tiers_total: dict with totals per tier (STM, MTM, LTM, Archive)
        - summary: human-readable string
    """
    from pathlib import Path

    result = {
        "total_facts": 0,
        "banks": {},
        "tiers_total": {"stm": 0, "mtm": 0, "ltm": 0, "archive": 0},
        "summary": ""
    }

    # Discover domain banks by scanning the directory
    try:
        # Auto-detect maven2_fix/ from current file location
        current_file = Path(__file__).resolve()
        base_path = current_file.parent.parent.parent.parent.parent
        domain_banks_dir = base_path / "brains" / "domain_banks"

        bank_names = []
        if domain_banks_dir.exists():
            for item in domain_banks_dir.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    bank_names.append(item.name)

        # Also include key cognitive banks that store facts
        cognitive_banks = ["personal", "self_model", "reasoning", "memory_librarian"]
        all_banks = bank_names + cognitive_banks

        print(f"[SELF_MODEL_MEMORY_STATS] Scanning {len(all_banks)} memory banks...")

        # For each bank, get tier counts
        for bank_id in all_banks:
            try:
                # Create BrainMemory instance for this bank
                bank_memory = BrainMemory(bank_id)
                stats = bank_memory.get_stats()
                tier_counts = stats.get("by_tier", {}) if isinstance(stats, dict) else {}
                bank_total = stats.get("total", 0) if isinstance(stats, dict) else 0

                if bank_total > 0:  # Only include banks with stored facts
                    result["banks"][bank_id] = {
                        "stm": tier_counts.get("stm", 0),
                        "mtm": tier_counts.get("mtm", 0),
                        "ltm": tier_counts.get("ltm", 0),
                        "archive": tier_counts.get("archive", 0),
                        "total": bank_total
                    }

                    # Update totals
                    result["total_facts"] += bank_total
                    result["tiers_total"]["stm"] += tier_counts.get("stm", 0)
                    result["tiers_total"]["mtm"] += tier_counts.get("mtm", 0)
                    result["tiers_total"]["ltm"] += tier_counts.get("ltm", 0)
                    result["tiers_total"]["archive"] += tier_counts.get("archive", 0)
            except Exception as e:
                # Skip banks that can't be accessed
                print(f"[SELF_MODEL_MEMORY_STATS] Could not access bank '{bank_id}': {e}")
                continue

        # Build human-readable summary
        bank_count = len(result["banks"])
        if result["total_facts"] == 0:
            result["summary"] = "I currently have no stored facts in my memory system. All memory tiers are empty."
        else:
            summary_parts = []
            summary_parts.append(f"I currently have {result['total_facts']} stored facts across {bank_count} memory banks.")

            # List top banks by fact count
            top_banks = sorted(
                result["banks"].items(),
                key=lambda x: x[1]["total"],
                reverse=True
            )[:5]  # Top 5 banks

            if top_banks:
                bank_list = ", ".join([f"{name} ({counts['total']})" for name, counts in top_banks])
                summary_parts.append(f"Top banks: {bank_list}.")

            # Tier distribution
            tier_parts = []
            for tier in ["stm", "mtm", "ltm", "archive"]:
                count = result["tiers_total"][tier]
                if count > 0:
                    tier_parts.append(f"{tier.upper()}: {count}")
            if tier_parts:
                summary_parts.append(f"Distribution across tiers: {', '.join(tier_parts)}.")

            result["summary"] = " ".join(summary_parts)

        print(f"[SELF_MODEL_MEMORY_STATS] Total: {result['total_facts']} facts across {bank_count} banks")

    except Exception as e:
        print(f"[SELF_MODEL_MEMORY_STATS] Error scanning memory: {e}")
        result["summary"] = f"Error scanning memory system: {str(e)[:100]}"

    return result


def scan_memory_health() -> Dict[str, Any]:
    """
    Light health-check for Maven's memory system.

    Checks:
    - Can we read/write to each tier (STM/MTM/LTM/Archive)?
    - Are the directories and files present?
    - Any errors accessing memory banks?

    Uses only local paths under maven2_fix - NO external directories.

    Returns:
        Dictionary with:
        - status: "healthy" or "errors"
        - checks: list of health check results
        - errors: list of any errors encountered
        - summary: human-readable string
    """
    from pathlib import Path

    result = {
        "status": "healthy",
        "checks": [],
        "errors": [],
        "summary": ""
    }

    try:
        # Sample a few key banks and check their tiers
        # Note: BrainMemory handles all path resolution via maven_paths.py,
        # ensuring everything stays inside maven2_fix
        sample_banks = ["personal", "self_model", "factual", "science"]

        print(f"[SELF_MODEL_MEMORY_HEALTH] Checking {len(sample_banks)} sample banks...")

        for bank_id in sample_banks:
            try:
                # Create BrainMemory instance
                bank_memory = BrainMemory(bank_id)

                # Try to get stats (this reads from each tier)
                stats = bank_memory.get_stats()

                # Check if all tiers are accessible
                tiers = ["stm", "mtm", "ltm", "archive"]
                tier_counts = stats.get("by_tier", {}) if isinstance(stats, dict) else {}
                for tier in tiers:
                    if tier_counts.get(tier) is None:
                        result["errors"].append(f"Bank '{bank_id}': tier '{tier}' not accessible")
                        result["status"] = "errors"

                total_records = stats.get("total", 0) if isinstance(stats, dict) else 0
                if isinstance(tier_counts, dict) and tier_counts:
                    total_records = sum(tier_counts.values())

                result["checks"].append(
                    f"Bank '{bank_id}': all tiers accessible ({total_records} records)"
                )

            except Exception as e:
                error_msg = f"Bank '{bank_id}': {str(e)[:100]}"
                result["errors"].append(error_msg)
                result["status"] = "errors"
                print(f"[SELF_MODEL_MEMORY_HEALTH] {error_msg}")

        # Build summary
        if result["status"] == "healthy":
            checks_count = len(result["checks"])
            result["summary"] = f"Memory system is healthy. All {checks_count} health checks passed. "
            result["summary"] += "All memory tiers (STM/MTM/LTM/Archive) are accessible with no errors."
        else:
            error_count = len(result["errors"])
            result["summary"] = f"Memory system has {error_count} error(s). "
            result["summary"] += "Issues: " + "; ".join(result["errors"][:3])  # Show first 3 errors

        print(f"[SELF_MODEL_MEMORY_HEALTH] Status: {result['status']}, Checks: {len(result['checks'])}, Errors: {len(result['errors'])}")

    except Exception as e:
        result["status"] = "errors"
        result["errors"].append(f"Health check failed: {str(e)[:100]}")
        result["summary"] = f"Memory health check failed: {str(e)[:100]}"
        print(f"[SELF_MODEL_MEMORY_HEALTH] Fatal error: {e}")

    return result


def plan_self_upgrade() -> Dict[str, Any]:
    """
    Generate a structured self-upgrade plan for Maven using DYNAMIC introspection.

    This function NOW:
    1. Dynamically scans ALL cognitive brains from actual Python files
    2. Analyzes brain contract compliance by comparing actual vs expected operations
    3. Identifies missing cognitive brain contract signals (3 signals per brain)
    4. Generates phased upgrade roadmap based on REAL findings, not hardcoded assumptions
    5. Returns structured plan WITHOUT calling Teacher

    This is TRUE self-awareness - Maven scans its own code and identifies gaps dynamically.

    Returns:
        Dictionary with:
        - summary: Human-readable upgrade plan
        - phases: List of upgrade phases with tasks (generated from scan results)
        - current_state: Snapshot of current capabilities (from dynamic scan)
        - priority_areas: High-priority improvements (identified dynamically)
        - scan_details: Full brain scan results
    """
    print("[SELF_MODEL] Generating DYNAMIC self-upgrade plan from code introspection...")

    # Import the introspection module
    try:
        from brains.cognitive.self_model.service.self_introspection import scan_self
    except Exception as e:
        print(f"[SELF_MODEL] Warning: Could not import introspection module: {e}")
        print("[SELF_MODEL] Falling back to basic scan...")
        # Fallback to basic code scan
        return _plan_self_upgrade_fallback()

    # STEP 1: Dynamically scan all brains
    introspection_results = scan_self()
    scan_results = introspection_results["scan_results"]
    priorities = introspection_results["priorities"]

    # STEP 2: Build current state from REAL scan data
    current_state = {
        "brains": {
            "total_cognitive": scan_results["total_brains"],
            "operation_compliant": scan_results["compliant_brains"],
            "operation_partial": scan_results["partial_brains"],
            "operation_non_compliant": scan_results["non_compliant_brains"],
            "signal_compliant": scan_results["signal_compliant_brains"],
            "signal_partial": scan_results["signal_partial_brains"],
            "signal_non_compliant": scan_results["signal_non_compliant_brains"]
        },
        "has_continuation_helpers": True,  # Check by import
        "signal_compliance_percentage": 0,
        "operation_compliance_percentage": 0
    }

    # Calculate overall compliance
    total_brains = scan_results["total_brains"]
    if total_brains > 0:
        current_state["signal_compliance_percentage"] = (
            scan_results["signal_compliant_brains"] / total_brains * 100
        )
        current_state["operation_compliance_percentage"] = (
            scan_results["compliant_brains"] / total_brains * 100
        )

    # Get memory stats
    try:
        memory_stats = get_memory_stats()
        current_state["memory"] = {
            "total_facts": memory_stats.get("total_facts", 0),
            "bank_count": len(memory_stats.get("banks", {}))
        }
    except Exception:
        current_state["memory"] = {"total_facts": 0, "bank_count": 0}

    # Get code summary
    try:
        code_summary = scan_own_code()
        current_state["code"] = {
            "total_python_files": code_summary.get("total_python_files", 0),
            "domain_banks": len(code_summary.get("domain_banks", []))
        }
    except Exception:
        current_state["code"] = {"total_python_files": 0, "domain_banks": 0}

    # STEP 3: Convert priorities to phases
    phases = []
    for idx, priority in enumerate(priorities, start=1):
        phases.append({
            "phase": idx,
            "name": priority["area"],
            "priority": priority["priority"],
            "tasks": priority["tasks"],
            "affected_brains": priority.get("affected_brains", []),
            "estimated_impact": priority["impact"],
            "status": "pending"
        })

    # STEP 4: Build human-readable summary
    total_brains = current_state["brains"]["total_cognitive"]
    signal_compliant = current_state["brains"]["signal_compliant"]
    signal_percent = current_state["signal_compliance_percentage"]
    op_compliant = current_state["brains"]["operation_compliant"]
    op_percent = current_state["operation_compliance_percentage"]

    summary = "# Maven Self-Upgrade Plan (Dynamic Analysis)\n\n"
    summary += "## Current State (Scanned from Actual Code)\n\n"
    summary += f"**Total Cognitive Brains:** {total_brains}\n\n"

    summary += f"**Signal Compliance (3 Signals per Brain):**\n"
    summary += f"- ✅ Fully compliant: {signal_compliant} brains ({signal_percent:.1f}%)\n"
    summary += f"- ⚠️ Partially compliant: {current_state['brains']['signal_partial']} brains\n"
    summary += f"- ❌ Non-compliant: {current_state['brains']['signal_non_compliant']} brains\n\n"

    summary += f"**Operation Contract Compliance:**\n"
    summary += f"- ✅ Fully compliant: {op_compliant} brains ({op_percent:.1f}%)\n"
    summary += f"- ⚠️ Partially compliant: {current_state['brains']['operation_partial']} brains\n"
    summary += f"- ❌ Non-compliant: {current_state['brains']['operation_non_compliant']} brains\n\n"

    summary += f"**Memory System:**\n"
    summary += f"- Total facts stored: {current_state['memory']['total_facts']}\n"
    summary += f"- Memory banks: {current_state['memory']['bank_count']}\n\n"

    summary += f"**Codebase:**\n"
    summary += f"- Total Python files: {current_state['code']['total_python_files']}\n"
    summary += f"- Domain banks: {current_state['code']['domain_banks']}\n\n"

    # Priority areas
    summary += f"## Priority Upgrade Areas ({len(priorities)})\n\n"
    for idx, priority in enumerate(priorities, start=1):
        priority_marker = "🔴" if priority["priority"] == "critical" else "🟠" if priority["priority"] == "high" else "🟡"
        summary += f"{idx}. {priority_marker} **{priority['area']}** (Priority: {priority['priority'].upper()})\n"
        summary += f"   - Affected brains: {len(priority.get('affected_brains', []))}\n"
        summary += f"   - Impact: {priority['impact']}\n"
        summary += f"   - Tasks: {len(priority['tasks'])} items\n\n"

    # Top phases
    summary += f"## Recommended Upgrade Phases\n\n"
    for phase in phases[:4]:  # Show top 4 phases
        priority_marker = "🔴" if phase["priority"] == "critical" else "🟠" if phase["priority"] == "high" else "🟡"
        summary += f"{priority_marker} **Phase {phase['phase']}: {phase['name']}**\n"
        summary += f"- Priority: {phase['priority'].upper()}\n"
        summary += f"- Brains affected: {len(phase['affected_brains'])}\n"
        summary += f"- Tasks: {len(phase['tasks'])} items\n"
        summary += f"- Impact: {phase['estimated_impact']}\n\n"

        # Show first 3 tasks
        for task in phase["tasks"][:3]:
            summary += f"  - {task}\n"
        if len(phase["tasks"]) > 3:
            summary += f"  - ...and {len(phase['tasks']) - 3} more tasks\n"
        summary += "\n"

    summary += "---\n\n"
    summary += "**This plan was generated by Maven's self-introspection system.**\n"
    summary += "All data comes from scanning actual Python source files, NOT from hardcoded assumptions.\n"
    summary += f"Scanned {total_brains} cognitive brains and analyzed {current_state['code']['total_python_files']} Python files.\n"

    return {
        "summary": summary,
        "phases": phases,
        "current_state": current_state,
        "priority_areas": priorities,
        "total_phases": len(phases),
        "scan_details": scan_results
    }


def _plan_self_upgrade_fallback() -> Dict[str, Any]:
    """
    Fallback upgrade plan if introspection module is not available.

    This provides a basic analysis without dynamic scanning.
    """
    print("[SELF_MODEL] Using fallback upgrade planning...")

    current_state = {
        "brains": {"cognitive_count": "unknown"},
        "memory": {},
        "code": {}
    }

    try:
        code_summary = scan_own_code()
        current_state["brains"]["cognitive_count"] = len(code_summary.get("brains", []))
        current_state["code"]["domain_banks"] = len(code_summary.get("domain_banks", []))
    except Exception:
        pass

    summary = "# Maven Self-Upgrade Plan (Fallback Mode)\n\n"
    summary += "⚠️ **Note:** Dynamic introspection module not available. Using basic analysis.\n\n"
    summary += "To enable full dynamic self-analysis, ensure self_introspection.py is accessible.\n\n"
    summary += f"## Current State\n"
    summary += f"- Cognitive brains: {current_state['brains'].get('cognitive_count', 'unknown')}\n"
    summary += f"- Domain banks: {current_state['code'].get('domain_banks', 'unknown')}\n\n"
    summary += "## Recommended Action\n"
    summary += "1. Fix introspection module import\n"
    summary += "2. Re-run upgrade planning for full dynamic analysis\n"

    return {
        "summary": summary,
        "phases": [],
        "current_state": current_state,
        "priority_areas": [],
        "total_phases": 0
    }


def describe_self_structured() -> Dict[str, Any]:
    """
    Merge core identity + live state into a coherent structured model.

    This function:
    1. Calls self_dmn.get_core_identity() for canonical facts
    2. Pulls relevant self facts from brains/personal (scope: self_core, self_reflection)
    3. Adds runtime state (enabled features, counts of Teacher calls, etc.)
    4. Returns one structured object – no text yet

    Returns:
        Structured dictionary with:
        - core_identity: from self_dmn identity card
        - personal_facts: from brains/personal with self_core/self_reflection scope
        - runtime_state: live system state
    """
    # 1. Get core identity from self_dmn
    try:
        import importlib
        self_dmn_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_brain")
        core_identity = self_dmn_mod.get_core_identity()
    except Exception as e:
        print(f"[SELF_MODEL] Could not load core identity: {e}")
        core_identity = {
            "name": "Maven",
            "is_llm": False,
            "system_type": "offline synthetic cognition system",
            "creator": "Josh / Hink",
            "home_directory": "maven2_fix"
        }

    # 2. Pull self facts from brains/personal with scope filtering
    personal_facts = []
    try:
        from brains.personal.service.personal_brain import service_api as personal_api
        # Try to get facts with self_core or self_reflection scope
        # For now, we'll retrieve from personal memory and filter by metadata
        from brains.memory.brain_memory import BrainMemory
        personal_memory = BrainMemory("personal")
        all_facts = personal_memory.retrieve(limit=100, tiers=["stm", "mtm", "ltm"])
        for fact in all_facts:
            metadata = fact.get("metadata", {})
            scope = metadata.get("scope")
            if scope in ["self_core", "self_reflection"]:
                personal_facts.append({
                    "content": fact.get("content"),
                    "scope": scope,
                    "confidence": metadata.get("confidence", 0.5)
                })
    except Exception as e:
        print(f"[SELF_MODEL] Could not load personal facts: {e}")

    # 3. Add runtime state
    runtime_state = {}
    try:
        # Count memory tier usage
        memory_stats = _memory.get_stats()
        runtime_state["memory_tiers"] = memory_stats
    except Exception:
        runtime_state["memory_tiers"] = {}

    try:
        exec_status = get_execution_status()
        # Convert to dict for JSON serialization compatibility
        runtime_state["execution_status"] = exec_status.to_dict() if hasattr(exec_status, 'to_dict') else exec_status
    except Exception:
        runtime_state["execution_status"] = {"enabled": False, "source": "unknown", "reason": "unavailable"}

    # 4. Add behavioral identity from identity inferencer
    behavioral_identity = {}
    if _identity_inferencer_available:
        try:
            behavioral_identity = identity_snapshot_for_self_model()
        except Exception as e:
            print(f"[SELF_MODEL] Could not get identity snapshot: {e}")
            behavioral_identity = {"error": str(e)}
    else:
        behavioral_identity = {"available": False}

    try:
        # Get Teacher call statistics if available
        from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api
        teacher_health = teacher_api({"op": "HEALTH", "mid": "self_model_check"})
        if teacher_health.get("ok"):
            runtime_state["teacher_available"] = True
            runtime_state["teacher_stats"] = teacher_health.get("payload", {})
        else:
            runtime_state["teacher_available"] = False
    except Exception:
        runtime_state["teacher_available"] = False

    # Add enabled features
    runtime_state["enabled_features"] = [
        "multi_tier_memory",
        "knowledge_graph",
        "goal_tracking",
        "self_reflection",
        "governance_system"
    ]

    return {
        "core_identity": core_identity,
        "personal_facts": personal_facts,
        "runtime_state": runtime_state,
        "behavioral_identity": behavioral_identity
    }

def describe_self(mode: str = "short") -> Dict[str, Any]:
    """Generate a structured self-description.

    Args:
        mode: Either "short" or "detailed"

    Returns:
        Dictionary with identity, capabilities, and limitations
    """
    model = SelfModel()
    facts = model._load_self_facts()

    capabilities = describe_capabilities()
    limitations = describe_limitations()

    if mode == "detailed":
        return {
            "identity": {
                "name": facts.get("name", "Maven"),
                "creator": facts.get("creator", "Josh Hinkle (Hink)"),
                "origin": facts.get("origin", "November 2025"),
                "role": facts.get("role", "offline personal intelligence"),
                "goals": facts.get("goals", [])
            },
            "capabilities": capabilities,
            "limitations": limitations
        }
    else:
        return {
            "identity": {
                "name": facts.get("name", "Maven"),
                "role": facts.get("role", "offline personal intelligence")
            },
            "capabilities": capabilities,
            "limitations": limitations
        }


def describe_capabilities() -> List[str]:
    """Generate capability descriptions from the capability registry."""

    registry = get_capability_registry()
    descriptions: List[str] = []
    for name, state in sorted(registry.items()):
        if not state.get("available"):
            descriptions.append(f"{name}: not available")
        elif state.get("enabled"):
            descriptions.append(f"{name}: enabled")
        else:
            reason = state.get("reason") or "disabled"
            descriptions.append(f"{name}: disabled ({reason})")
    return descriptions


def describe_limitations() -> List[str]:
    """Describe current limitations derived from disabled capabilities."""

    registry = get_capability_registry()
    limitations: List[str] = []
    for name, state in registry.items():
        if not state.get("available"):
            limitations.append(f"{name} unavailable")
        elif not state.get("enabled"):
            reason = state.get("reason") or "disabled"
            limitations.append(f"{name} disabled: {reason}")
    return limitations


def get_capabilities() -> List[str]:
    return describe_capabilities()


def get_limitations() -> List[str]:
    return describe_limitations()


def generate_self_introduction(detail_level: str = "full") -> Dict[str, Any]:
    """
    Generate a comprehensive, dynamic self-introduction based on real introspection.

    This function performs ACTUAL introspection by:
    1. Scanning the codebase to understand structure
    2. Reading memory statistics from all tiers
    3. Checking active capabilities via runtime probes
    4. Aggregating brain inventory and health
    5. Generating a dynamic, truthful introduction

    Args:
        detail_level: "brief", "standard", or "full"

    Returns:
        Dictionary with:
        - introduction_text: Natural language introduction
        - identity: Core identity facts
        - architecture: Codebase structure analysis
        - capabilities: Active capabilities and tools
        - memory_stats: Aggregated memory statistics
        - brain_inventory: Active brains and their status
        - runtime_state: Current system state
    """
    result = {
        "introduction_text": "",
        "identity": {},
        "architecture": {},
        "capabilities": {},
        "memory_stats": {},
        "brain_inventory": {},
        "runtime_state": {},
        "generated_at": None
    }

    import time
    from datetime import datetime
    result["generated_at"] = datetime.now().isoformat()

    # ==========================================================================
    # 1. IDENTITY - Who am I?
    # ==========================================================================
    try:
        import importlib
        self_dmn_mod = importlib.import_module("brains.cognitive.self_dmn.service.self_dmn_brain")
        core_identity = self_dmn_mod.get_core_identity()
    except Exception:
        core_identity = {
            "name": "Maven",
            "is_llm": False,
            "system_type": "offline synthetic cognition system",
            "creator": "Josh / Hink",
        }

    result["identity"] = core_identity

    # ==========================================================================
    # 2. ARCHITECTURE - What is my structure?
    # ==========================================================================
    try:
        code_summary = scan_own_code()
        result["architecture"] = {
            "cognitive_brains": len(code_summary.get("brains", [])),
            "domain_banks": len(code_summary.get("domain_banks", [])),
            "total_python_files": code_summary.get("total_python_files", 0),
            "brain_list": code_summary.get("brains", [])[:15],  # First 15
            "domain_list": code_summary.get("domain_banks", [])[:10],  # First 10
            "root_directory": code_summary.get("scanned_from", "unknown")
        }
    except Exception as e:
        result["architecture"] = {"error": str(e)}

    # ==========================================================================
    # 3. CAPABILITIES - What can I do right now?
    # ==========================================================================
    try:
        from brains.system_capabilities import (
            get_capability_truth,
            get_capability_summary,
            get_available_tools,
            is_web_research_enabled,
            is_browser_runtime_configured,
            get_execution_mode,
        )

        capability_truth = get_capability_truth()
        capability_summary = get_capability_summary()

        result["capabilities"] = {
            "tools_available": get_available_tools(),
            "tools_status": capability_truth.get("tools", {}),
            "brains_status": capability_truth.get("brains", {}),
            "execution_mode": get_execution_mode(),
            "web_research": is_web_research_enabled(),
            "browser_runtime": is_browser_runtime_configured(),
            "summary": capability_truth.get("summary", "unknown")
        }
    except Exception as e:
        # Fallback to basic capability detection
        result["capabilities"] = {
            "tools_available": [],
            "error": str(e)
        }
        try:
            registry = get_capability_registry()
            result["capabilities"]["from_registry"] = {
                name: state.get("enabled", False)
                for name, state in registry.items()
            }
        except Exception:
            pass

    # ==========================================================================
    # 4. MEMORY STATS - What do I remember?
    # ==========================================================================
    try:
        from brains.brain_roles import get_cognitive_brains, get_domain_banks
        from brains.memory.brain_memory import BrainMemory

        total_records = 0
        memory_by_tier = {"stm": 0, "mtm": 0, "ltm": 0, "archive": 0}
        brain_memory_stats = {}

        # Aggregate memory from cognitive brains
        for brain_name in get_cognitive_brains()[:10]:  # Sample first 10
            try:
                brain_mem = BrainMemory(brain_name, brain_category="cognitive")
                stats = brain_mem.get_stats()
                brain_memory_stats[brain_name] = stats.get("total", 0)
                total_records += stats.get("total", 0)
                for tier, count in stats.get("by_tier", {}).items():
                    memory_by_tier[tier] = memory_by_tier.get(tier, 0) + count
            except Exception:
                continue

        # Aggregate memory from domain banks
        domain_memory = 0
        for domain_name in get_domain_banks()[:5]:  # Sample first 5
            try:
                domain_mem = BrainMemory(domain_name, brain_category="domain")
                stats = domain_mem.get_stats()
                domain_memory += stats.get("total", 0)
                total_records += stats.get("total", 0)
            except Exception:
                continue

        result["memory_stats"] = {
            "total_records": total_records,
            "by_tier": memory_by_tier,
            "cognitive_brain_samples": brain_memory_stats,
            "domain_memory_total": domain_memory
        }
    except Exception as e:
        result["memory_stats"] = {"error": str(e)}

    # ==========================================================================
    # 5. BRAIN INVENTORY - What brains are active?
    # ==========================================================================
    try:
        inventory = get_brain_inventory()
        result["brain_inventory"] = {
            "total_cognitive": len(inventory.get("canonical_cognitive", [])),
            "present_cognitive": len(inventory.get("present_cognitive", [])),
            "domain_banks": len(inventory.get("domain_banks", [])),
            "summary": inventory.get("summary", "")
        }
    except Exception as e:
        result["brain_inventory"] = {"error": str(e)}

    # ==========================================================================
    # 6. RUNTIME STATE - What is my current state?
    # ==========================================================================
    try:
        exec_status = get_execution_status()
        result["runtime_state"] = {
            "execution_mode": exec_status.mode.value if hasattr(exec_status, 'mode') else "unknown",
            "execution_effective": exec_status.effective if hasattr(exec_status, 'effective') else False,
            "teacher_available": False,
            "learning_enabled": False
        }

        # Check Teacher availability
        try:
            from brains.cognitive.teacher.service.teacher_brain import service_api as teacher_api
            health = teacher_api({"op": "HEALTH", "mid": "intro_check"})
            result["runtime_state"]["teacher_available"] = health.get("ok", False)
        except Exception:
            pass

        # Check feature flags
        try:
            root = get_maven_root()
            features_path = root / "config" / "features.json"
            if features_path.exists():
                import json
                with open(features_path, 'r') as f:
                    features = json.load(f)
                result["runtime_state"]["learning_enabled"] = features.get("teacher_learning", False)
                result["runtime_state"]["feature_flags"] = features
        except Exception:
            pass

    except Exception as e:
        result["runtime_state"] = {"error": str(e)}

    # ==========================================================================
    # 7. GENERATE NATURAL LANGUAGE INTRODUCTION
    # ==========================================================================
    intro_parts = []

    # Opening - Don't mention creator unless specifically asked
    # Creator info is stored in identity but only revealed for "who created you" questions
    name = result["identity"].get("name", "Maven")
    system_type = result["identity"].get("system_type", "synthetic cognition system")

    # Use correct article (a/an) based on first letter
    article = "an" if system_type[0].lower() in "aeiou" else "a"
    intro_parts.append(
        f"I am {name}, {article} {system_type}."
    )

    # Architecture
    arch = result.get("architecture", {})
    if arch and not arch.get("error"):
        brain_count = arch.get("cognitive_brains", 0)
        domain_count = arch.get("domain_banks", 0)
        intro_parts.append(
            f"My cognitive architecture consists of {brain_count} specialized processing brains "
            f"and {domain_count} domain knowledge banks."
        )

    # Capabilities
    caps = result.get("capabilities", {})
    if caps and not caps.get("error"):
        tools = caps.get("tools_available", [])
        exec_mode = caps.get("execution_mode", "unknown")
        if tools:
            intro_parts.append(
                f"I currently have access to: {', '.join(tools)}. "
                f"Execution mode: {exec_mode}."
            )

        if caps.get("web_research"):
            intro_parts.append("I can search the web for information.")
        if caps.get("browser_runtime"):
            intro_parts.append("I have visual browser automation capabilities.")

    # Memory
    mem = result.get("memory_stats", {})
    if mem and not mem.get("error"):
        total = mem.get("total_records", 0)
        if total > 0:
            intro_parts.append(
                f"I maintain a tiered memory system with {total} stored records "
                f"across short-term, mid-term, long-term, and archival tiers."
            )

    # Brain inventory
    inv = result.get("brain_inventory", {})
    if inv and not inv.get("error"):
        present = inv.get("present_cognitive", 0)
        total = inv.get("total_cognitive", 0)
        if present and total:
            intro_parts.append(
                f"Of my {total} defined cognitive brains, {present} are currently active."
            )

    # Learning state
    runtime = result.get("runtime_state", {})
    if runtime.get("teacher_available") and runtime.get("learning_enabled"):
        intro_parts.append(
            "My Teacher brain is active, allowing me to learn new patterns from interactions."
        )

    # Closing based on detail level
    if detail_level == "brief":
        result["introduction_text"] = intro_parts[0] if intro_parts else f"I am {name}."
    elif detail_level == "standard":
        result["introduction_text"] = " ".join(intro_parts[:4])
    else:  # full
        result["introduction_text"] = " ".join(intro_parts)

    return result


def get_brain_inventory() -> Dict[str, Any]:
    """
    Get a complete inventory of all brains in the Maven system.

    This function provides:
    - List of all canonical cognitive brains (from whitelist)
    - Which cognitive brains are actually present on disk
    - Unrecognized folders under brains/cognitive/
    - Domain banks
    - Overall brain health summary

    Uses the brain_roles module for canonical whitelist and scan_brains for discovery.

    Returns:
        Dictionary with:
        - canonical_cognitive: list of whitelisted cognitive brain names
        - present_cognitive: list of cognitive brains actually found on disk
        - missing_cognitive: list of whitelisted brains not found on disk
        - unrecognized_folders: folders under cognitive/ not in whitelist
        - domain_banks: list of domain bank names
        - summary: human-readable inventory summary
    """
    result = {
        "canonical_cognitive": [],
        "present_cognitive": [],
        "missing_cognitive": [],
        "unrecognized_folders": [],
        "domain_banks": [],
        "summary": ""
    }

    try:
        from brains.brain_roles import (
            CANONICAL_COGNITIVE_BRAINS,
            get_cognitive_brains,
            get_domain_banks,
            get_unrecognized_folders,
        )

        # Get canonical list (whitelist)
        result["canonical_cognitive"] = sorted(CANONICAL_COGNITIVE_BRAINS)

        # Get what's actually present on disk via brain_roles
        present_brains = get_cognitive_brains()
        result["present_cognitive"] = sorted(present_brains)

        # Calculate missing (in whitelist but not on disk)
        present_set = set(present_brains)
        canonical_set = CANONICAL_COGNITIVE_BRAINS
        result["missing_cognitive"] = sorted(canonical_set - present_set)

        # Get unrecognized folders (on disk but not in whitelist)
        result["unrecognized_folders"] = get_unrecognized_folders()

        # Get domain banks
        result["domain_banks"] = get_domain_banks()

        # Build summary
        canonical_count = len(result["canonical_cognitive"])
        present_count = len(result["present_cognitive"])
        missing_count = len(result["missing_cognitive"])
        unrecognized_count = len(result["unrecognized_folders"])
        domain_count = len(result["domain_banks"])

        summary_parts = []
        summary_parts.append(f"Brain Inventory Summary:")
        summary_parts.append(f"- Canonical cognitive brains: {canonical_count}")
        summary_parts.append(f"- Present on disk: {present_count}")

        if missing_count > 0:
            summary_parts.append(f"- MISSING from disk: {missing_count} ({', '.join(result['missing_cognitive'][:5])}{'...' if missing_count > 5 else ''})")

        if unrecognized_count > 0:
            summary_parts.append(f"- Unrecognized folders: {unrecognized_count} ({', '.join(result['unrecognized_folders'][:5])}{'...' if unrecognized_count > 5 else ''})")

        summary_parts.append(f"- Domain banks: {domain_count}")

        # Overall health
        if missing_count == 0 and unrecognized_count == 0:
            summary_parts.append("- Status: HEALTHY - All canonical brains present, no stray folders")
        elif missing_count > 0:
            summary_parts.append(f"- Status: INCOMPLETE - {missing_count} canonical brain(s) missing")
        elif unrecognized_count > 0:
            summary_parts.append(f"- Status: NEEDS CLEANUP - {unrecognized_count} unrecognized folder(s)")

        result["summary"] = "\n".join(summary_parts)

        print(f"[SELF_MODEL] Brain inventory: {present_count}/{canonical_count} cognitive brains present, {domain_count} domain banks")

    except Exception as e:
        print(f"[SELF_MODEL] Error getting brain inventory: {e}")
        result["summary"] = f"Error scanning brain inventory: {str(e)[:100]}"

    return result

def update_self_facts(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update self facts in a controlled way.

    Args:
        updates: Dictionary of updates to apply

    Returns:
        Updated facts or error
    """
    try:
        memory = BrainMemory("self_model")

        # Retrieve current facts
        results = memory.retrieve(limit=1)
        if results:
            current_facts = results[0].get("content", {})
            if not isinstance(current_facts, dict):
                current_facts = {}
        else:
            current_facts = {}

        for key, value in updates.items():
            if key in ["capabilities", "limitations"]:
                if isinstance(value, list):
                    if "add" in updates.get("_mode", {}):
                        current_facts.setdefault(key, []).extend(value)
                    else:
                        current_facts[key] = value
            elif key not in ["name", "creator", "origin", "role"]:
                current_facts[key] = value

        # Store updated facts with metadata
        memory.store(
            content=current_facts,
            metadata={
                "kind": "self_facts",
                "source": "self_model",
                "confidence": 1.0
            }
        )
        return {"ok": True, "updated_facts": current_facts}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the self‑model brain.

    Supports operations for identity, capabilities, and limitations.
    """
    op = str((msg or {}).get("op", "")).upper()
    mid = (msg or {}).get("mid")
    payload = (msg or {}).get("payload") or {}
    model = SelfModel()

    if op == "DESCRIBE_SELF_STRUCTURED":
        # NEW: Return structured identity without calling Teacher
        description = describe_self_structured()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": description,
        }

    if op == "DESCRIBE_SELF":
        mode = str(payload.get("mode", "short"))
        description = describe_self(mode)
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": description,
        }

    if op == "SELF_INTRODUCTION":
        # Generate comprehensive self-introduction based on real introspection
        # This scans codebase, reads memory stats, checks capabilities dynamically
        print("[SELF_MODEL] service_api: SELF_INTRODUCTION operation")
        detail_level = str(payload.get("detail_level", "full"))
        introduction = generate_self_introduction(detail_level)
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": introduction,
        }

    if op == "DESCRIBE_CAPABILITIES":
        # NEW: Return capability description from REAL capability registry
        print(f"[SELF_MODEL] service_api: DESCRIBE_CAPABILITIES operation")
        capability_description = model.describe_capabilities()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "text": capability_description,
                "self_origin": True
            },
        }

    if op == "GET_SYSTEM_STATUS":
        # NEW: Return complete system status aggregating multiple sources
        print(f"[SELF_MODEL] service_api: GET_SYSTEM_STATUS operation")
        system_status = model.get_system_status()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": system_status,
        }

    if op == "ANSWER_CAPABILITY_QUESTION":
        # NEW: Answer capability/upgrade questions using real runtime state
        # CRITICAL: This is used by routing to answer "what can you do", "can you browse", etc.
        print(f"[SELF_MODEL] service_api: ANSWER_CAPABILITY_QUESTION operation")
        question = str(payload.get("question", payload.get("query", "")))
        runtime_state = payload.get("runtime_state")

        result = model.answer_capability_question(question, runtime_state)
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": result,
        }

    if op == "GET_CAPABILITIES":
        capabilities = get_capabilities()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {"capabilities": capabilities},
        }

    if op == "GET_LIMITATIONS":
        limitations = get_limitations()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {"limitations": limitations},
        }

    if op == "UPDATE_SELF_FACTS":
        updates = payload.get("updates", {})
        result = update_self_facts(updates)
        if result.get("ok"):
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result,
            }
        else:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": {"code": "UPDATE_FAILED", "message": result.get("error", "Unknown error")},
            }

    if op == "CAN_ANSWER":
        q = str(payload.get("query", ""))
        can_ans, beliefs = model.can_answer(q)
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "can_answer": can_ans,
                "beliefs": beliefs,
            },
        }

    if op == "QUERY_SELF":
        q = str(payload.get("query", ""))
        self_kind = payload.get("self_kind")  # Extract self_kind (identity/code/memory)
        self_mode = payload.get("self_mode")  # Extract self_mode (stats/health)
        result = model.query_self(q, self_kind=self_kind, self_mode=self_mode)

        # Add routing hint for Teacher learning
        routing_hint = None
        if _continuation_helpers_available and result.get("ok"):
            try:
                is_follow_up = is_continuation(q, {"query": q})
                result_payload = result.get("payload", {})
                confidence = result_payload.get("confidence", 0.8)

                # Determine action based on meta-question type
                action = "self_query"
                context_tags = ["self_awareness"]

                if result_payload.get("is_meta_explanation"):
                    action = "self_explanation"
                    context_tags.append("meta_cognition")
                elif result_payload.get("is_clarification"):
                    action = "self_clarification"
                    context_tags.append("clarification")
                elif result_payload.get("is_confirmation"):
                    action = "self_confirmation"
                    context_tags.append("confirmation")
                elif is_follow_up:
                    action = "self_query_continuation"
                    context_tags.append("continuation")

                routing_hint = create_routing_hint(
                    brain_name="self_model",
                    action=action,
                    confidence=confidence,
                    context_tags=context_tags
                )

                # Add routing hint to payload
                result["payload"]["routing_hint"] = routing_hint
            except Exception as e:
                print(f"[SELF_MODEL] Warning: Failed to create routing hint: {str(e)[:100]}")

        if result.get("ok"):
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result.get("payload", {}),
            }
        return {
            "ok": False,
            "op": op,
            "mid": mid,
            "error": result.get("error", {"code": "NO_SELF_ANSWER", "message": "Unsupported self query"}),
        }

    if op == "SCAN_CODE":
        # Explicitly scan Maven's own code and return structured summary
        # This is the REAL introspection - reading actual files from disk
        print("[SELF_MODEL] Executing code scan operation...")
        base_path = payload.get("base_path")  # Optional override
        code_summary = scan_own_code(base_path)
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": code_summary,
        }

    if op == "BRAIN_INVENTORY":
        # Get complete inventory of all brains in the system
        # Uses canonical whitelist vs actual disk presence
        print("[SELF_MODEL] Executing brain inventory operation...")
        inventory = get_brain_inventory()
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": inventory,
        }

    if op == "GET_SELF_KNOWLEDGE":
        # Read the SELF_KNOWLEDGE.md file for self-modification guidance
        print("[SELF_MODEL] Reading self-knowledge reference...")
        section = payload.get("section")  # Optional: filter to specific section
        try:
            root = get_maven_root()
            knowledge_path = root / "SELF_KNOWLEDGE.md"
            if knowledge_path.exists():
                content = knowledge_path.read_text(encoding="utf-8")
                # If section requested, try to extract it
                if section:
                    import re
                    pattern = rf"## {re.escape(section)}.*?(?=\n## |\Z)"
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(0)
                return {
                    "ok": True,
                    "op": op,
                    "mid": mid,
                    "payload": {"content": content, "source": str(knowledge_path)},
                }
            else:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": "SELF_KNOWLEDGE.md not found",
                }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": str(e),
            }

    if op == "GET_CODEBASE_STRUCTURE":
        # Return structured view of where files belong
        print("[SELF_MODEL] Generating codebase structure map...")
        try:
            root = get_maven_root()
            structure = {
                "root": str(root),
                "directories": {
                    "brains/cognitive/[name]/service/": "Cognitive brains (processing)",
                    "brains/agent/tools/": "Tool facades (no I/O)",
                    "brains/personal/": "User identity and preferences",
                    "brains/memory/": "Memory subsystem",
                    "brains/tools/": "Core tool facades",
                    "brains/utils/": "Shared utilities",
                    "host_tools/": "External I/O (network, LLM, browser)",
                    "optional/": "Optional features",
                    "config/": "Configuration files",
                    "docs/": "Documentation",
                    "reports/": "Runtime data and logs",
                    "tests/": "Test suites",
                },
                "key_files": {
                    "brains/cognitive/integrator/service/integrator_brain.py": "Query routing",
                    "brains/cognitive/language/service/language_brain.py": "NLU/generation",
                    "brains/tools/self_upgrade_tool.py": "Self-modification",
                    "brains/tools/execution_guard.py": "Safety controls",
                    "config/features.json": "Feature flags",
                    "SELF_KNOWLEDGE.md": "Self-knowledge reference",
                },
            }
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": structure,
            }
        except Exception as e:
            return {
                "ok": False,
                "op": op,
                "mid": mid,
                "error": str(e),
            }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": op},
    }


# Standard service contract: handle is the entry point
service_api = handle