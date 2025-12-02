"""
pipeline_runner.py
~~~~~~~~~~~~~~~~~~

Pipeline runner that orchestrates the canonical Maven cognition pipeline.

This module provides:
1. Service wrapper functions for each of the 9 canonical pipeline stages
2. A run_pipeline() function that executes the full pipeline
3. Error handling and logging for pipeline execution

P1 PIPELINE REQUIREMENT:
All cognitive processing must flow through this runner using the canonical
pipeline defined in pipeline_definition.py. The runner ensures proper stage
ordering, service invocation, and blackboard management.

STDLIB ONLY, OFFLINE, WINDOWS 10 COMPATIBLE
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import traceback

from brains.cognitive.sensorium.message_filter import is_meta_or_comment_message

from brains.pipeline.pipeline_executor import PipelineExecutor
from brains.pipeline.pipeline_definition import CanonicalPipeline, PipelineStage
from brains.learning.learning_mode import LearningMode


# =============================================================================
# Stage 1: NLU (Natural Language Understanding)
# =============================================================================

def nlu_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for NLU stage using the language brain.

    Parses user intent and extracts entities from the input query.

    Args:
        stage_input: Stage input from PipelineExecutor containing:
            - stage: Stage name
            - context: Blackboard context
            - previous_outputs: Outputs from previous stages

    Returns:
        Stage output with NLU results and blackboard updates
    """
    try:
        from brains.cognitive.language.service.language_brain import service_api

        # Extract user query from context
        context = stage_input.get("context", {})
        user_query = context.get("user_query", "")

        # Build service message for NLU operation
        # language_brain uses "PARSE" operation for NLU
        msg = {
            "op": "PARSE",
            "mid": context.get("mid", "pipeline_nlu"),
            "payload": {
                "text": user_query,
                "context": context
            }
        }

        # Call language brain service
        response = service_api(msg)

        # Extract NLU results
        nlu_result = response.get("payload", {})

        # Return stage output with blackboard updates
        return {
            "stage": "NLU",
            "status": "success",
            "nlu_result": nlu_result,
            "blackboard_updates": {
                "intent": nlu_result.get("intent"),
                "entities": nlu_result.get("entities", []),
                "parsed_query": nlu_result.get("parsed_text", user_query)
            }
        }

    except Exception as e:
        return {
            "stage": "NLU",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {}
        }


# =============================================================================
# Stage 2: PATTERN_RECOGNITION
# =============================================================================

def pattern_recognition_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Pattern Recognition stage.

    Detects patterns in the user input and context.

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with pattern detection results
    """
    try:
        from brains.cognitive.pattern_recognition.service.pattern_recognition_brain import service_api

        context = stage_input.get("context", {})
        user_query = context.get("parsed_query") or context.get("user_query", "")

        # Build service message for pattern detection
        # pattern_recognition_brain uses "EXTRACT_PATTERNS" operation
        msg = {
            "op": "EXTRACT_PATTERNS",
            "mid": context.get("mid", "pipeline_pattern"),
            "payload": {
                "text": user_query,
                "entities": context.get("entities", []),
                "context": context
            }
        }

        # Call pattern recognition service
        response = service_api(msg)

        # Extract pattern results
        pattern_result = response.get("payload", {})

        return {
            "stage": "PATTERN_RECOGNITION",
            "status": "success",
            "patterns": pattern_result.get("patterns", []),
            "blackboard_updates": {
                "detected_patterns": pattern_result.get("patterns", []),
                "pattern_confidence": pattern_result.get("confidence", 0.0)
            }
        }

    except Exception as e:
        return {
            "stage": "PATTERN_RECOGNITION",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {}
        }


# =============================================================================
# Stage 3: MEMORY (Memory Librarian) - MANDATORY
# =============================================================================

def memory_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Memory stage using the memory librarian.

    Retrieves relevant memories and context from all memory banks.
    This is a MANDATORY stage (required=True in pipeline).

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with memory retrieval results
    """
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import service_api

        context = stage_input.get("context", {})
        user_query = context.get("parsed_query") or context.get("user_query", "")

        # Build service message for unified memory retrieval
        msg = {
            "op": "UNIFIED_RETRIEVE",
            "mid": context.get("mid", "pipeline_memory"),
            "payload": {
                "query": user_query,
                "intent": context.get("intent"),
                "entities": context.get("entities", []),
                "patterns": context.get("detected_patterns", []),
                "limit": 10,
                "context": context
            }
        }

        # Call memory librarian service
        response = service_api(msg)

        # Extract memory results
        memory_payload = response.get("payload", {})
        memory_results = memory_payload.get("results", [])

        return {
            "stage": "MEMORY",
            "status": "success",
            "memory_count": len(memory_results),
            "memory_results": memory_results,
            "blackboard_updates": {
                "memories": memory_results,
                "memory_banks_queried": memory_payload.get("banks_queried", []),
                "memory_context": memory_payload
            }
        }

    except Exception as e:
        return {
            "stage": "MEMORY",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {}
        }


# =============================================================================
# Stage 4: REASONING - MANDATORY
# =============================================================================

def reasoning_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Reasoning stage.

    Evaluates facts, performs inference, and generates structured reasoning.
    This is a MANDATORY stage (required=True in pipeline).

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with reasoning results
    """
    try:
        from brains.cognitive.reasoning.service.reasoning_brain import service_api

        context = stage_input.get("context", {})
        user_query = context.get("parsed_query") or context.get("user_query", "")
        memories = context.get("memories", [])

        # Build service message for reasoning
        msg = {
            "op": "GENERATE_THOUGHTS",
            "mid": context.get("mid", "pipeline_reasoning"),
            "payload": {
                "query": user_query,
                "intent": context.get("intent"),
                "evidence": memories,
                "context": context
            }
        }

        # Call reasoning service
        response = service_api(msg)

        # Extract reasoning results
        reasoning_payload = response.get("payload", {})

        return {
            "stage": "REASONING",
            "status": "success",
            "thoughts": reasoning_payload.get("thoughts", []),
            "inferences": reasoning_payload.get("inferences", []),
            "blackboard_updates": {
                "reasoning_thoughts": reasoning_payload.get("thoughts", []),
                "inferences": reasoning_payload.get("inferences", []),
                "reasoning_confidence": reasoning_payload.get("confidence", 0.5)
            }
        }

    except Exception as e:
        return {
            "stage": "REASONING",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {}
        }


# =============================================================================
# Stage 5: VALIDATION (Council Brain) - MANDATORY
# =============================================================================

def validation_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Validation stage using the council brain.

    Validates truth tags, checks governance rules, and ensures safety.
    This is a MANDATORY stage (required=True in pipeline).

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with validation results
    """
    try:
        from brains.governance.council.service.council_brain import service_api

        context = stage_input.get("context", {})

        # Build service message for validation
        # council_brain uses "ARBITRATE" operation
        msg = {
            "op": "ARBITRATE",
            "mid": context.get("mid", "pipeline_validation"),
            "payload": {
                "query": context.get("parsed_query") or context.get("user_query", ""),
                "intent": context.get("intent"),
                "memories": context.get("memories", []),
                "thoughts": context.get("reasoning_thoughts", []),
                "inferences": context.get("inferences", []),
                "context": context
            }
        }

        # Call council brain service
        response = service_api(msg)

        # Extract validation results
        validation_payload = response.get("payload", {})

        return {
            "stage": "VALIDATION",
            "status": "success",
            "validation_passed": validation_payload.get("valid", True),
            "governance_checks": validation_payload.get("checks", []),
            "blackboard_updates": {
                "validation_passed": validation_payload.get("valid", True),
                "governance_violations": validation_payload.get("violations", []),
                "safety_checks": validation_payload.get("safety_checks", {})
            }
        }

    except Exception as e:
        return {
            "stage": "VALIDATION",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {
                "validation_passed": False,
                "validation_error": str(e)
            }
        }


# =============================================================================
# Stage 6: GENERATION - MANDATORY
# =============================================================================

def generation_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Generation stage using the language brain.

    Generates the final response based on all previous stage outputs.
    This is a MANDATORY stage (required=True in pipeline).

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with generated response
    """
    try:
        from brains.cognitive.language.service.language_brain import service_api

        context = stage_input.get("context", {})
        user_query = context.get("parsed_query") or context.get("user_query", "")

        # Build service message for generation
        # language_brain uses "GENERATE_CANDIDATES" operation
        msg = {
            "op": "GENERATE_CANDIDATES",
            "mid": context.get("mid", "pipeline_generation"),
            "payload": {
                "query": user_query,
                "intent": context.get("intent"),
                "memories": context.get("memories", []),
                "thoughts": context.get("reasoning_thoughts", []),
                "inferences": context.get("inferences", []),
                "validation_passed": context.get("validation_passed", True),
                "context": context
            }
        }

        # Call language brain service
        response = service_api(msg)

        # Extract generation results
        # GENERATE_CANDIDATES returns a list of candidates
        generation_payload = response.get("payload", {})
        candidates = generation_payload.get("candidates", [])

        # Use the first candidate if available
        if candidates and len(candidates) > 0:
            first_candidate = candidates[0]
            generated_text = first_candidate.get("text", "")
            confidence = first_candidate.get("confidence", 0.5)
        else:
            generated_text = ""
            confidence = 0.0

        return {
            "stage": "GENERATION",
            "status": "success",
            "generated_text": generated_text,
            "candidates": candidates,
            "blackboard_updates": {
                "generated_response": generated_text,
                "generation_confidence": confidence,
                "generation_candidates": candidates,
                "response_metadata": generation_payload
            }
        }

    except Exception as e:
        return {
            "stage": "GENERATION",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {
                "generated_response": "I encountered an error generating a response."
            }
        }


# =============================================================================
# Stage 7: FINALIZATION
# =============================================================================

def finalization_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Finalization stage using the integrator brain.

    Formats and finalizes the output for presentation.

    CRITICAL: If capability_disabled is present in context, pass through the
    response verbatim without any formatting or modification. This preserves
    clear error messages about disabled capabilities.

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with finalized response
    """
    try:
        from brains.cognitive.integrator.service.integrator_brain import service_api

        context = stage_input.get("context", {})
        generated_response = context.get("generated_response", "")

        # CRITICAL: Check for capability_disabled in context
        # If present, pass through the existing response verbatim without modification
        if context.get("capability_disabled"):
            capability_name = context.get("capability_disabled")
            reason = context.get("reason", "Capability disabled")
            print(f"[FINALIZATION] Detected capability_disabled ({capability_name}), passing through response verbatim")

            # Use the existing response from context without any formatting
            passthrough_response = context.get("response") or generated_response

            return {
                "stage": "FINALIZATION",
                "status": "success",
                "final_response": passthrough_response,
                "blackboard_updates": {
                    "final_response": passthrough_response,
                    "capability_disabled": capability_name,
                    "reason": reason,
                    "passthrough": True
                }
            }

        # Build service message for finalization
        msg = {
            "op": "FINALIZE",
            "mid": context.get("mid", "pipeline_finalization"),
            "payload": {
                "response": generated_response,
                "intent": context.get("intent"),
                "metadata": context.get("response_metadata", {}),
                "context": context
            }
        }

        # Call integrator brain service
        response = service_api(msg)

        # Extract finalization results
        finalization_payload = response.get("payload", {})
        final_response = finalization_payload.get("final_response") or generated_response

        return {
            "stage": "FINALIZATION",
            "status": "success",
            "final_response": final_response,
            "blackboard_updates": {
                "final_response": final_response,
                "formatting_applied": finalization_payload.get("formatting", [])
            }
        }

    except Exception as e:
        # If finalization fails, use the generated response as-is
        context = stage_input.get("context", {})
        fallback_response = context.get("generated_response", "")

        return {
            "stage": "FINALIZATION",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "final_response": fallback_response,
            "blackboard_updates": {
                "final_response": fallback_response,
                "finalization_error": str(e)
            }
        }


# =============================================================================
# Stage 8: HISTORY
# =============================================================================

def history_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for History stage using the system history brain.

    Updates conversation history with the current interaction.

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with history update confirmation
    """
    try:
        from brains.cognitive.system_history.service.system_history_brain import service_api

        context = stage_input.get("context", {})
        user_query = context.get("user_query", "")
        final_response = context.get("final_response", "")

        # Build service message for history update
        msg = {
            "op": "UPDATE_HISTORY",
            "mid": context.get("mid", "pipeline_history"),
            "payload": {
                "user_query": user_query,
                "response": final_response,
                "intent": context.get("intent"),
                "metadata": {
                    "stage_outputs": stage_input.get("previous_outputs", {}),
                    "confidence": context.get("generation_confidence", 0.5)
                },
                "context": context
            }
        }

        # Call system history service
        response = service_api(msg)

        # Extract history update results
        history_payload = response.get("payload", {})

        return {
            "stage": "HISTORY",
            "status": "success",
            "history_updated": history_payload.get("updated", True),
            "blackboard_updates": {
                "history_entry_id": history_payload.get("entry_id"),
                "conversation_depth": history_payload.get("conversation_depth", 0)
            }
        }

    except Exception as e:
        return {
            "stage": "HISTORY",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {}
        }


# =============================================================================
# Stage 9: AUTONOMY
# =============================================================================

def autonomy_service_wrapper(stage_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Autonomy stage using the autonomy brain.

    Considers autonomous actions based on the interaction.

    Args:
        stage_input: Stage input from PipelineExecutor

    Returns:
        Stage output with autonomy evaluation results
    """
    try:
        from brains.cognitive.autonomy.service.autonomy_brain import service_api

        context = stage_input.get("context", {})

        # Build service message for autonomy evaluation
        msg = {
            "op": "EVALUATE_AUTONOMY",
            "mid": context.get("mid", "pipeline_autonomy"),
            "payload": {
                "query": context.get("user_query", ""),
                "response": context.get("final_response", ""),
                "intent": context.get("intent"),
                "context": context
            }
        }

        # Call autonomy brain service
        response = service_api(msg)

        # Extract autonomy results
        autonomy_payload = response.get("payload", {})

        return {
            "stage": "AUTONOMY",
            "status": "success",
            "autonomous_actions": autonomy_payload.get("actions", []),
            "blackboard_updates": {
                "autonomous_actions": autonomy_payload.get("actions", []),
                "autonomy_triggered": len(autonomy_payload.get("actions", [])) > 0
            }
        }

    except Exception as e:
        return {
            "stage": "AUTONOMY",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "blackboard_updates": {}
        }


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_pipeline(
    user_query: str,
    confidence: float = 0.8,
    initial_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the canonical Maven cognition pipeline.

    This function orchestrates the entire 9-stage pipeline:
    1. NLU - Parse intent and entities
    2. PATTERN_RECOGNITION - Detect patterns
    3. MEMORY - Retrieve relevant memories (MANDATORY)
    4. REASONING - Generate thoughts and inferences (MANDATORY)
    5. VALIDATION - Validate truth and governance (MANDATORY)
    6. GENERATION - Generate response (MANDATORY)
    7. FINALIZATION - Format output
    8. HISTORY - Update conversation history
    9. AUTONOMY - Consider autonomous actions

    Args:
        user_query: The user's input query
        confidence: Initial confidence threshold (default 0.8)
        initial_context: Optional additional context to include

    Returns:
        Dict containing:
            - ok: Boolean indicating success
            - response: Final response text (if successful)
            - blackboard: Final blackboard state
            - stage_outputs: All stage outputs
            - execution_log: Pipeline execution log
            - error: Error message (if failed)
    """
    # PRE-PIPELINE AGENCY DETECTION WITH CAPABILITY CHECKING
    # Check if this is an agency query (filesystem, git, introspection, etc.)
    # Verify capability is enabled before executing, or return clear error
    try:
        from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern
        from brains.cognitive.integrator.agency_executor import execute_agency_tool, format_agency_response
        from capabilities import is_capability_enabled, get_capability_reason

        agency_match = match_agency_pattern(user_query, threshold=0.7)
        if agency_match:
            print(f"[PIPELINE] Agency query detected: {agency_match['tool']} (confidence: {agency_match['confidence']})")

            # Map tool paths to capability names
            tool_to_capability = {
                "brains.tools.filesystem_agency": "filesystem_agency",
                "brains.tools.git_tool": "git_agency",
                "brains.tools.hot_reload": "hot_reload",
                "brains.tools.self_introspection": "self_introspection",
            }

            # Determine which capability this tool needs
            tool_path = agency_match['tool']
            capability_name = None
            for tool_prefix, cap_name in tool_to_capability.items():
                if tool_path.startswith(tool_prefix):
                    capability_name = cap_name
                    break

            # Check if capability is enabled
            if capability_name and not is_capability_enabled(capability_name):
                reason = get_capability_reason(capability_name)
                print(f"[PIPELINE] Capability {capability_name} is disabled: {reason}")

                # Return clear "capability disabled" response
                disabled_response = f"I cannot execute that operation. {reason}"

                return {
                    "ok": True,
                    "response": disabled_response,
                    "blackboard": {
                        "user_query": user_query,
                        "capability_disabled": capability_name,
                        "reason": reason,
                        "bypass_pipeline": True
                    },
                    "stage_outputs": {
                        "CAPABILITY_CHECK": {
                            "capability": capability_name,
                            "enabled": False,
                            "reason": reason
                        }
                    },
                    "execution_log": [
                        {
                            "stage": "CAPABILITY_CHECK",
                            "success": False,
                            "capability": capability_name,
                            "reason": "capability_disabled"
                        }
                    ]
                }

            # Capability is enabled, execute tool
            tool_result = execute_agency_tool(
                tool_path=agency_match['tool'],
                method_name=agency_match.get('method'),
                args=agency_match.get('args')
            )

            # Format response
            formatted_response = format_agency_response(user_query, tool_result)

            # Return immediately with agency tool result
            return {
                "ok": True,
                "response": formatted_response,
                "blackboard": {
                    "user_query": user_query,
                    "agency_tool": agency_match['tool'],
                    "agency_result": tool_result,
                    "bypass_pipeline": True
                },
                "stage_outputs": {
                    "AGENCY_TOOL": {
                        "tool": agency_match['tool'],
                        "result": tool_result,
                        "confidence": agency_match['confidence']
                    }
                },
                "execution_log": [
                    {
                        "stage": "AGENCY_TOOL",
                        "success": tool_result.get('status') == 'success',
                        "duration_ms": 0,
                        "tool": agency_match['tool']
                    }
                ]
            }
    except Exception as e:
        # If agency detection fails, just continue with normal pipeline
        print(f"[PIPELINE] Agency detection failed: {e}")
        import traceback
        traceback.print_exc()
        pass

    # PRE-PIPELINE NORM INTROSPECTION DETECTION
    # Route "repeat my message" / "show normalized" requests to sensorium
    try:
        import re
        query_lower = user_query.strip().lower()

        # Detect norm introspection requests
        norm_introspection_patterns = [
            r"repeat\s+(?:this\s+)?(?:message\s+)?(?:back\s+)?normalized",
            r"show\s+(?:me\s+)?(?:the\s+)?(?:last\s+)?normalized\s+(?:input|text|message)",
            r"how\s+did\s+you\s+(?:normalize|process)\s+(?:that|my\s+(?:message|input))",
            r"what\s+(?:did\s+you\s+)?normalize(?:d)?\s+(?:to|as)",
            r"echo\s+(?:back\s+)?(?:the\s+)?normalized",
        ]

        is_norm_introspection = any(re.search(pattern, query_lower) for pattern in norm_introspection_patterns)

        if is_norm_introspection:
            print(f"[PIPELINE] Norm introspection request detected, routing to sensorium")

            try:
                from brains.cognitive.sensorium.service.sensorium_brain import service_api as sensorium_api

                # Get the last normalized input
                result = sensorium_api({
                    "op": "GET_LAST_NORMALIZED",
                    "mid": "norm_introspection"
                })

                if result.get("ok"):
                    payload = result.get("payload", {})
                    raw_text = payload.get("raw_text", "")
                    normalized_text = payload.get("normalized_text", "")
                    norm_type = payload.get("norm_type", "")
                    tokens = payload.get("tokens", [])

                    # Format response - return the normalized text directly
                    response_text = f"**Original text:**\n{raw_text}\n\n**Normalized text:**\n{normalized_text}"
                    if norm_type:
                        response_text += f"\n\n**Classification:** {norm_type}"
                    if tokens:
                        response_text += f"\n\n**Tokens:** {', '.join(tokens)}"

                    print(f"[PIPELINE] Norm introspection completed")

                    return {
                        "ok": True,
                        "response": response_text,
                        "blackboard": {
                            "user_query": user_query,
                            "routed_to": "sensorium_introspection",
                            "raw_text": raw_text,
                            "normalized_text": normalized_text,
                            "bypass_pipeline": True
                        },
                        "stage_outputs": {
                            "NORM_INTROSPECTION": {
                                "raw_text": raw_text,
                                "normalized_text": normalized_text
                            }
                        },
                        "execution_log": [
                            {
                                "stage": "NORM_INTROSPECTION",
                                "success": True,
                                "duration_ms": 0
                            }
                        ]
                    }
                else:
                    # No previous input to show
                    return {
                        "ok": True,
                        "response": "No input has been normalized yet. Please send a message first.",
                        "blackboard": {
                            "user_query": user_query,
                            "routed_to": "sensorium_introspection",
                            "bypass_pipeline": True
                        },
                        "stage_outputs": {},
                        "execution_log": []
                    }

            except Exception as e:
                print(f"[PIPELINE] Norm introspection failed: {e}, continuing to full pipeline")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"[PIPELINE] Norm introspection detection failed: {e}")
        import traceback
        traceback.print_exc()
        pass

    # PRE-PIPELINE RESEARCH DETECTION
    # Route research requests directly to research_manager and return results
    try:
        import re
        query_lower = user_query.strip().lower()

        # Detect explicit research requests
        research_patterns = [
            r"^research\s*:",  # "research:"
            r"^deep\s+research\s*:",  # "deep research:"
            r"^investigate\s*:",  # "investigate:"
        ]

        is_research_request = any(re.search(pattern, query_lower) for pattern in research_patterns)

        if is_research_request:
            print(f"[PIPELINE] Research request detected, routing to research_manager")

            try:
                from brains.cognitive.research_manager.service.research_manager_brain import service_api as research_api

                # Extract the topic from the query
                topic = user_query
                topic = re.sub(r"^(?:deep\s+)?research\s*:\s*", "", topic, flags=re.IGNORECASE)
                topic = re.sub(r"^investigate\s*:\s*", "", topic, flags=re.IGNORECASE)

                # Call research manager
                research_result = research_api({
                    "op": "RESEARCH",
                    "mid": "research_run",
                    "payload": {
                        "topic": topic,
                        "depth": 2,
                        "sources": ["memory", "teacher"],  # Use available sources
                        "max_web_requests": 0  # No web for now
                    }
                })

                if research_result.get("ok"):
                    payload = research_result.get("payload", {})
                    summary = payload.get("summary", payload.get("answer", ""))
                    facts_collected = payload.get("facts_collected", 0)

                    # Format response
                    if summary:
                        response_text = summary
                        if facts_collected > 0:
                            response_text += f"\n\n*({facts_collected} facts collected)*"
                    else:
                        # Fallback: use teacher for basic facts
                        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
                        teacher = TeacherHelper("research_fallback")
                        teacher_result = teacher.maybe_call_teacher(
                            question=f"Give me 5 key facts with years about: {topic}",
                            context={"topic": topic},
                            check_memory_first=True
                        )
                        if teacher_result and teacher_result.get("answer"):
                            response_text = teacher_result.get("answer", "")
                        else:
                            response_text = f"I researched '{topic}' but couldn't find specific information."

                    print(f"[PIPELINE] Research completed (facts={facts_collected})")

                    return {
                        "ok": True,
                        "response": response_text,
                        "blackboard": {
                            "user_query": user_query,
                            "routed_to": "research_manager",
                            "topic": topic,
                            "facts_collected": facts_collected,
                            "bypass_pipeline": True
                        },
                        "stage_outputs": {
                            "RESEARCH": {
                                "topic": topic,
                                "facts_collected": facts_collected
                            }
                        },
                        "execution_log": [
                            {
                                "stage": "RESEARCH",
                                "success": True,
                                "duration_ms": 0
                            }
                        ]
                    }
                else:
                    print(f"[PIPELINE] Research failed, continuing to full pipeline")

            except Exception as e:
                print(f"[PIPELINE] Research routing failed: {e}, continuing to full pipeline")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"[PIPELINE] Research detection failed: {e}")
        import traceback
        traceback.print_exc()
        pass

    # PRE-PIPELINE BRAIN INVENTORY DETECTION
    # Route "inventory:" requests to list all cognitive brains using InventoryBrain
    try:
        import re
        query_lower = user_query.strip().lower()

        # Detect brain inventory requests
        inventory_patterns = [
            r"^inventory\s*[:\?]?\s*$",  # "inventory:" or "inventory?" or "inventory"
            r"^inventory\s*:\s*list",  # "inventory: list"
            r"list\s+(?:all\s+)?(?:cognitive\s+)?brains",  # "list all brains"
            r"show\s+(?:me\s+)?(?:all\s+)?(?:cognitive\s+)?brains",  # "show all brains"
            r"what\s+brains\s+(?:are\s+)?(?:there|available|exist)",  # "what brains are there"
            r"brain\s+inventory",  # "brain inventory"
            r"count\s+(?:all\s+)?brains",  # "count brains"
        ]

        is_inventory_request = any(re.search(pattern, query_lower) for pattern in inventory_patterns)

        if is_inventory_request:
            print(f"[PIPELINE] Brain inventory request detected, routing to InventoryBrain")

            try:
                from brains.cognitive.inventory.service.inventory_brain import service_api as inventory_api

                # Call inventory brain
                inventory_result = inventory_api({
                    "op": "LIST",
                    "mid": "inventory_request"
                })

                if inventory_result.get("ok"):
                    payload = inventory_result.get("payload", {})
                    brains_list = payload.get("brains", [])
                    total_count = payload.get("total", 0)
                    unclassified = payload.get("unclassified", [])

                    # Format response
                    response_text = f"**Maven: Cognitive brains detected:**\n\n"
                    for brain_info in brains_list:
                        brain_name = brain_info.get("name", "unknown")
                        response_text += f"- {brain_name}\n"

                    response_text += f"\n**Total:** {total_count}\n"

                    # Add unclassified notice if any
                    if unclassified:
                        response_text += f"\n**Unclassified items ({len(unclassified)}):**\n"
                        for item in unclassified:
                            item_name = item.get("name", "unknown")
                            response_text += f"- Unclassified item found: {item_name}. Should this be added?\n"

                    print(f"[PIPELINE] Brain inventory completed: {total_count} brains found")

                    return {
                        "ok": True,
                        "response": response_text,
                        "blackboard": {
                            "user_query": user_query,
                            "routed_to": "inventory_brain",
                            "brain_count": total_count,
                            "brains": [b.get("name") for b in brains_list],
                            "bypass_pipeline": True
                        },
                        "stage_outputs": {
                            "BRAIN_INVENTORY": {
                                "count": total_count,
                                "brains": brains_list
                            }
                        },
                        "execution_log": [
                            {
                                "stage": "BRAIN_INVENTORY",
                                "success": True,
                                "duration_ms": 0
                            }
                        ]
                    }
                else:
                    print(f"[PIPELINE] InventoryBrain returned error, continuing to full pipeline")

            except Exception as e:
                print(f"[PIPELINE] Brain inventory failed: {e}, continuing to full pipeline")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"[PIPELINE] Brain inventory detection failed: {e}")
        import traceback
        traceback.print_exc()
        pass

    # PRE-PIPELINE CODER DETECTION
    # Route coding requests directly to coder brain and return code without summarization
    try:
        import re
        query_lower = user_query.strip().lower()

        # Detect explicit coder requests
        coder_patterns = [
            r"^(?:use\s+)?coder\s*:",  # "coder:" or "use coder:"
            r"^write\s+(?:a\s+)?(?:python\s+)?function\b",  # "write a function"
            r"^generate\s+(?:a\s+)?(?:python\s+)?(?:function|code)\b",  # "generate code"
            r"^create\s+(?:a\s+)?(?:python\s+)?function\b",  # "create function"
        ]

        # Check if this is a coder request
        is_coder_request = any(re.search(pattern, query_lower) for pattern in coder_patterns)

        # Also detect code follow-up requests that should refine previous code
        code_only_patterns = [
            r"return\s+only\s+(?:the\s+)?(?:full\s+)?(?:updated\s+)?(?:code|function)",
            r"(?:just|only)\s+(?:the\s+)?code",
            r"show\s+(?:me\s+)?the\s+(?:full\s+)?code",
            r"if\s+both\s+are\s+(?:lists|arrays)",  # spec extension
            r"if\s+one\s+is\s+a\s+scalar",  # spec extension
        ]
        is_code_follow_up = any(re.search(pattern, query_lower) for pattern in code_only_patterns)

        if is_coder_request or is_code_follow_up:
            print(f"[PIPELINE] Coder request detected, routing to coder brain")

            try:
                from brains.cognitive.coder.service.coder_brain import service_api as coder_api

                # Extract the actual spec from the query
                spec = user_query
                # Remove "coder:" or "use coder:" prefix
                spec = re.sub(r"^(?:use\s+)?coder\s*:\s*", "", spec, flags=re.IGNORECASE)

                # Determine if this is a refinement request
                is_refinement = is_code_follow_up

                # First create a plan
                plan_result = coder_api({
                    "op": "PLAN",
                    "mid": "coder_plan",
                    "payload": {"spec": spec}
                })

                plan = plan_result.get("payload", {}) if plan_result.get("ok") else {}

                # Generate code
                gen_result = coder_api({
                    "op": "GENERATE",
                    "mid": "coder_generate",
                    "payload": {
                        "spec": spec,
                        "plan": plan,
                        "return_code_only": True
                    }
                })

                if gen_result.get("ok"):
                    payload = gen_result.get("payload", {})
                    code = payload.get("code", "")
                    test_code = payload.get("test_code", "")
                    summary = payload.get("summary", "")

                    # Verify the code
                    verify_result = coder_api({
                        "op": "VERIFY",
                        "mid": "coder_verify",
                        "payload": {
                            "code": code,
                            "test_code": test_code,
                            "spec": spec
                        }
                    })

                    verify_payload = verify_result.get("payload", {}) if verify_result.get("ok") else {}
                    tests_passed = verify_payload.get("tests_passed", False)

                    # Format response - return code directly
                    if code:
                        # Format as a code block
                        response_text = f"```python\n{code}\n```"

                        # Add test results if there are issues
                        if not tests_passed and verify_payload.get("test_error"):
                            response_text += f"\n\n*Note: Some tests did not pass:*\n```\n{verify_payload.get('test_error', '')[:500]}\n```"
                    else:
                        response_text = f"I couldn't generate code for: {spec}"

                    print(f"[PIPELINE] Coder generated code (tests_passed={tests_passed})")

                    return {
                        "ok": True,
                        "response": response_text,
                        "blackboard": {
                            "user_query": user_query,
                            "routed_to": "coder",
                            "code": code,
                            "test_code": test_code,
                            "tests_passed": tests_passed,
                            "bypass_pipeline": True
                        },
                        "stage_outputs": {
                            "CODER": {
                                "code": code,
                                "tests_passed": tests_passed,
                                "is_refinement": is_refinement
                            }
                        },
                        "execution_log": [
                            {
                                "stage": "CODER",
                                "success": True,
                                "duration_ms": 0,
                                "tests_passed": tests_passed
                            }
                        ]
                    }
                else:
                    print(f"[PIPELINE] Coder failed to generate, continuing to full pipeline")

            except Exception as e:
                print(f"[PIPELINE] Coder routing failed: {e}, continuing to full pipeline")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"[PIPELINE] Coder detection failed: {e}")
        import traceback
        traceback.print_exc()
        pass

    # PRE-PIPELINE SELF-MODEL META-QUESTION DETECTION
    # Route meta-questions about Maven's capabilities/identity to self_model instead of Teacher
    # This ensures honest, accurate responses based on actual runtime state
    try:
        import re
        query_lower = user_query.strip().lower()

        # Detect meta-questions that should go to self_model
        meta_patterns = [
            r"\bwhat\s+(?:can|do)\s+you\s+(?:do)?\b",  # "what can you do"
            r"\bwhat\s+are\s+(?:your|you)\s+(?:capabilities|abilities|skills|features)\b",  # "what are your capabilities"
            r"\bwhat\s+are\s+you\s+capable\b",  # "what are you capable of"
            r"\bdescribe\s+(?:yourself|your\s+capabilities|your\s+abilities)\b",  # "describe yourself"
            r"\bwho\s+are\s+you\b",  # "who are you"
            r"\bwhat\s+(?:is|are)\s+your\s+(?:capabilities|abilities|features)\b",  # "what is your capability"
            r"\blist\s+your\s+(?:capabilities|abilities|features)\b",  # "list your capabilities"
            r"\bscan\s+(?:self|yourself)\b",  # "scan self"
            r"\bwhat\s+do\s+you\s+know\s+about\s+yourself\b",  # "what do you know about yourself"
        ]

        is_meta_question = any(re.search(pattern, query_lower) for pattern in meta_patterns)

        if is_meta_question:
            print(f"[PIPELINE] Meta-question detected, routing to self_model instead of Teacher")

            # Route to self_model brain
            try:
                from brains.cognitive.self_model.service.self_model_brain import service_api as self_model_api

                # Determine query type for routing hint
                if re.search(r"\bscan\s+(?:self|yourself)\b", query_lower):
                    self_kind = "system_scan"
                elif re.search(r"\bwho\s+are\s+you\b", query_lower):
                    self_kind = "identity"
                elif re.search(r"\bwhat\s+do\s+you\s+know\s+about\s+yourself\b", query_lower):
                    self_kind = "code"
                else:
                    self_kind = None  # Let query_self handle routing based on pattern matching

                # Call self_model with QUERY_SELF operation
                self_model_result = self_model_api({
                    "op": "QUERY_SELF",
                    "mid": "meta_question_routing",
                    "payload": {
                        "query": user_query,
                        "self_kind": self_kind
                    }
                })

                if self_model_result.get("ok"):
                    response_text = self_model_result.get("payload", {}).get("text", "")
                    confidence = self_model_result.get("payload", {}).get("confidence", 0.9)

                    print(f"[PIPELINE] Self-model answered meta-question (confidence: {confidence})")

                    return {
                        "ok": True,
                        "response": response_text,
                        "blackboard": {
                            "user_query": user_query,
                            "routed_to": "self_model",
                            "self_kind": self_kind,
                            "confidence": confidence,
                            "bypass_pipeline": True
                        },
                        "stage_outputs": {
                            "SELF_MODEL": {
                                "query_type": self_kind or "capability",
                                "confidence": confidence,
                                "self_origin": True
                            }
                        },
                        "execution_log": [
                            {
                                "stage": "SELF_MODEL",
                                "success": True,
                                "duration_ms": 0,
                                "query_type": self_kind or "capability"
                            }
                        ]
                    }
                else:
                    print(f"[PIPELINE] Self-model could not answer, continuing to full pipeline")

            except Exception as e:
                print(f"[PIPELINE] Self-model routing failed: {e}, continuing to full pipeline")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"[PIPELINE] Meta-question detection failed: {e}")
        import traceback
        traceback.print_exc()
        pass

    if is_meta_or_comment_message(user_query):
        return {
            "ok": True,
            "response": "",
            "blackboard": {
                "user_query": user_query,
                "meta_ignored": True,
                "skip_pipeline": True,
            },
            "stage_outputs": {},
            "execution_log": [],
            "meta_ignored": True,
            "skip_pipeline": True,
        }

    try:
        # Create pipeline executor
        pipeline = CanonicalPipeline()
        executor = PipelineExecutor(pipeline)

        # Register all brain services
        executor.register_service(
            "language.service.language_brain",
            nlu_service_wrapper
        )
        executor.register_service(
            "pattern_recognition.service.pattern_recognition_brain",
            pattern_recognition_service_wrapper
        )
        executor.register_service(
            "memory_librarian.service.memory_librarian",
            memory_service_wrapper
        )
        executor.register_service(
            "reasoning.service.reasoning_brain",
            reasoning_service_wrapper
        )
        executor.register_service(
            "governance.council.service.council_brain",
            validation_service_wrapper
        )
        # GENERATION uses language brain (registered separately for clarity)
        executor.register_service(
            "language.service.language_brain",  # Same brain, different stage
            generation_service_wrapper
        )
        executor.register_service(
            "integrator.service.integrator_brain",
            finalization_service_wrapper
        )
        executor.register_service(
            "system_history.service.system_history_brain",
            history_service_wrapper
        )
        executor.register_service(
            "autonomy.service.autonomy_brain",
            autonomy_service_wrapper
        )

        # Build initial context
        context = initial_context or {}
        context.update({
            "user_query": user_query,
            "confidence_threshold": confidence,
            "pipeline_version": "canonical_v1",
            # Learning mode controls cognitive behavior (memory-first, LLM-as-teacher).
            # TRAINING (default): Memory-first -> if miss -> call LLM -> store lesson/facts
            # OFFLINE: Memory-first -> if miss -> no LLM call, no storage (evaluation only)
            # SHADOW: Memory-first -> if miss -> LLM for comparison only, no storage
            # NOTE: Do NOT set OFFLINE here by default. TRAINING enables learning.
            "learning_mode": context.get("learning_mode", LearningMode.TRAINING)
        })

        # Execute pipeline
        result = executor.execute(initial_context=context)

        # Check if pipeline succeeded
        if result.get("ok"):
            # Extract final response from blackboard
            blackboard = result.get("blackboard", {})
            final_response = blackboard.get("final_response") or blackboard.get("generated_response", "")

            return {
                "ok": True,
                "response": final_response,
                "blackboard": blackboard,
                "stage_outputs": result.get("stage_outputs", {}),
                "execution_log": result.get("execution_log", []),
                "bypass_attempts": result.get("bypass_attempts", [])
            }
        else:
            # Pipeline failed
            return {
                "ok": False,
                "error": result.get("error", "Pipeline execution failed"),
                "blackboard": result.get("blackboard", {}),
                "execution_log": result.get("execution_log", [])
            }

    except Exception as e:
        # Top-level error handling
        return {
            "ok": False,
            "error": f"Pipeline runner failed: {str(e)}",
            "traceback": traceback.format_exc(),
            "blackboard": {},
            "stage_outputs": {},
            "execution_log": []
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "run_pipeline",
    "nlu_service_wrapper",
    "pattern_recognition_service_wrapper",
    "memory_service_wrapper",
    "reasoning_service_wrapper",
    "validation_service_wrapper",
    "generation_service_wrapper",
    "finalization_service_wrapper",
    "history_service_wrapper",
    "autonomy_service_wrapper",
]
