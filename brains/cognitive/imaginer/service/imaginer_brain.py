"""
Imaginer Brain
==============

The imaginer brain provides a safe sandbox for generating hypothetical
statements.  It allows the system to speculate without immediately
committing those speculations to long‑term memory.  Hypotheses are
tagged as transient and must be validated by the reasoning brain before
promotion to factual knowledge or working theory.

Operations:

  HYPOTHESIZE
      Accepts a ``prompt`` in the payload and returns a list of
      hypothetical statements.  Each hypothesis is returned as a dict
      with keys ``content`` and ``transient``.

The current implementation generates a single speculation by prefixing
the prompt with "It might be that".  Future versions could employ
more sophisticated mechanisms such as pattern completion or analogy.
"""

from __future__ import annotations
from typing import Dict, Any, List
from brains.memory.brain_memory import BrainMemory

# Teacher integration for learning scenario generation patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("imaginer")
except Exception as e:
    print(f"[IMAGINER] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Import continuation helpers for cognitive brain contract compliance
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint
    )
    _continuation_helpers_available = True
except Exception as e:
    print(f"[IMAGINER] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Initialize memory at module level for reuse
_memory = BrainMemory("imaginer")

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op", "").upper()
    payload = (msg or {}).get("payload", {}) or {}
# COGNITIVE BRAIN CONTRACT: Signal 1 & 2 - Detect continuation and get context
    continuation_detected = False
    conv_context = {}

    try:
        # Extract query from payload
        query = (payload.get("query") or
                payload.get("question") or
                payload.get("user_query") or
                payload.get("text") or "")

        if query:
            continuation_detected = is_continuation(query, payload)

            if continuation_detected:
                conv_context = get_conversation_context()
                # Enrich payload with conversation context
                payload["continuation_detected"] = True
                payload["last_topic"] = conv_context.get("last_topic", "")
                payload["conversation_depth"] = conv_context.get("conversation_depth", 0)
    except Exception as e:
        # Silently continue if continuation detection fails
        pass

    # Get conversation context for continuation detection
    conv_context = {}
    is_follow_up = False
    if _continuation_helpers_available:
        try:
            conv_context = get_conversation_context()
            prompt = payload.get("prompt", "") or payload.get("topic", "")
            is_follow_up = is_continuation(prompt, payload)
        except Exception:
            pass

    if op == "HYPOTHESIZE":
        # Extract the prompt/topic.  If missing return an empty list.
        prompt = str(payload.get("prompt") or payload.get("topic") or "").strip()
        if not prompt:
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="imaginer",
                    action="hypothesize",
                    confidence=0.7,
                    context_tags=[
                        "hypothesize",
                        "continuation" if continuation_detected else "fresh_query"
                    ]
                )
                if isinstance(result, dict):
                    result["routing_hint"] = routing_hint
                elif isinstance(payload_result, dict):
                    payload_result["routing_hint"] = routing_hint
            except Exception:
                pass  # Routing hint generation is non-critical
            return {"ok": True, "op": op, "payload": {"hypotheses": []}}

        # Check for learned scenario generation patterns first
        learned_templates = []
        if _teacher_helper and _memory:
            try:
                learned_patterns = _memory.retrieve(
                    query=f"scenario generation template: {prompt[:50]}",
                    limit=5,
                    tiers=["stm", "mtm", "ltm"]
                )

                for pattern_rec in learned_patterns:
                    if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.6:
                        content = pattern_rec.get("content", "")
                        if isinstance(content, str) and "{p}" in content:
                            learned_templates.append(content)
                            if len(learned_templates) >= 3:
                                break

                if learned_templates:
                    print(f"[IMAGINER] Using {len(learned_templates)} learned scenario templates from Teacher")
            except Exception:
                pass

        # Compose up to five speculative statements.  Each hypothesis is
        # transient and must be validated by the reasoning brain before
        # promotion to factual knowledge.  Different prefixes encourage
        # creative thinking without committing to memory.
        # Use learned templates if available, otherwise use built-in templates
        if learned_templates:
            templates = learned_templates
        else:
            templates = [
                "It might be that {p}.",
                "Perhaps {p}.",
                "Imagine that {p}.",
                "One possibility is that {p}.",
                "Conceivably, {p}."
            ]

            # If no learned templates and Teacher available, try to learn
            if _teacher_helper:
                try:
                    print(f"[IMAGINER] No learned templates for this scenario type, calling Teacher...")
                    teacher_result = _teacher_helper.maybe_call_teacher(
                        question=f"What creative templates should I use to generate hypotheses about: {prompt}",
                        context={"prompt": prompt, "current_templates": templates},
                        check_memory_first=True
                    )

                    if teacher_result and teacher_result.get("answer"):
                        patterns_stored = teacher_result.get("patterns_stored", 0)
                        print(f"[IMAGINER] Learned from Teacher: {patterns_stored} scenario templates stored")
                        # Learned templates now in memory for future use
                except Exception as e:
                    print(f"[IMAGINER] Teacher call failed: {str(e)[:100]}")

        hyps: List[Dict[str, Any]] = []
        for tmpl in templates:
            try:
                content = tmpl.format(p=prompt)
            except Exception:
                content = f"It might be that {prompt}."
            hyps.append({
                "content": content,
                "transient": True,
                "confidence": 0.4,
                "source": "imaginer"
            })
        # Respect any governance permit specifying max rollouts.  A permit
        # request is issued to the policy engine.  If the request is denied,
        # return an empty list.  Otherwise include the permit_id on each
        # hypothesis and truncate the list to the allowed number.
        try:
            n_requested = int(payload.get("n", len(hyps)))
        except Exception:
            n_requested = len(hyps)
        # Request a permit from governance
        permit_id = None
        allowed = True
        try:
            import importlib
            permits_mod = importlib.import_module(
                "brains.governance.policy_engine.service.permits"
            )
            perm_res = permits_mod.service_api({
                "op": "REQUEST",
                "payload": {"action": "IMAGINE", "n": n_requested}
            })
            perm_payload = perm_res.get("payload") or {}
            allowed = bool(perm_payload.get("allowed", False))
            permit_id = perm_payload.get("permit_id")
            # If denied, note the reason but drop hypotheses
            if not allowed:
                # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
                try:
                    routing_hint = create_routing_hint(
                        brain_name="imaginer",
                        action="unknown",
                        confidence=0.7,
                        context_tags=[
                            "unknown",
                            "continuation" if continuation_detected else "fresh_query"
                        ]
                    )
                    if isinstance(result, dict):
                        result["routing_hint"] = routing_hint
                    elif isinstance(payload_result, dict):
                        payload_result["routing_hint"] = routing_hint
                except Exception:
                    pass  # Routing hint generation is non-critical
                return {
                    "ok": True,
                    "op": op,
                    "payload": {"hypotheses": []}
                }
        except Exception:
            # On permit failure, proceed cautiously with allowed number but no proof
            permit_id = None
            allowed = True
        # Bound number of hypotheses.  Respect a configurable maximum
        # number of roll‑outs specified in ``config/imagination.json``.  This
        # allows deployments to tune the imagination sandbox depth without
        # modifying the code.  If the configuration is missing or invalid,
        # default to the length of the available templates (currently 5).
        max_rollouts = len(hyps)
        try:
            from pathlib import Path
            # Determine repository root (maven_extracted/maven) relative to this file
            root = Path(__file__).resolve().parents[4]
            cfg_path = root / "config" / "imagination.json"
            if cfg_path.exists():
                import json as _json
                with open(cfg_path, "r", encoding="utf-8") as cfgfh:
                    cfg = _json.load(cfgfh) or {}
                mr = int(cfg.get("max_rollouts", max_rollouts))
                # Ensure a sensible value (1 <= mr <= 20)
                if 1 <= mr <= 20:
                    max_rollouts = mr
        except Exception:
            # Fall back to default
            max_rollouts = len(hyps)
        # Limit n_requested by both available templates and configuration
        n = max(1, min(n_requested, max_rollouts, len(hyps)))
        hyps = hyps[:n]
        # Attach simple novelty scores and governance proof id
        out_hyps: List[Dict[str, Any]] = []
        for h in hyps:
            new_h = dict(h)
            # Score is a placeholder reflecting nominal novelty; could be extended
            new_h["score"] = 0.5
            if permit_id:
                new_h["permit_id"] = permit_id
            out_hyps.append(new_h)
        # Persist hypotheses and permit results using BrainMemory tier API
        try:
            # Store permit request result
            _memory.store(
                content={"action": "IMAGINE", "n": n_requested, "allowed": allowed, "permit_id": permit_id, "is_continuation": is_follow_up},
                metadata={"kind": "permit_request", "source": "imaginer", "confidence": 1.0}
            )
            # Store generated hypotheses
            _memory.store(
                content={"prompt": prompt, "hypotheses": out_hyps, "n_generated": len(out_hyps), "is_continuation": is_follow_up},
                metadata={"kind": "hypotheses_generation", "source": "imaginer", "confidence": 0.4}
            )
        except Exception:
            pass

        result_payload = {
            "hypotheses": out_hyps,
            "is_continuation": is_follow_up
        }

        # Track creative expansion for continuations
        if is_follow_up:
            result_payload["base_scenario"] = conv_context.get("last_topic", "")

        # Add routing hint for Teacher learning
        if _continuation_helpers_available:
            try:
                result_payload["routing_hint"] = create_routing_hint(
                    brain_name="imaginer",
                    action="creative_expansion" if is_follow_up else "hypothesize",
                    confidence=0.7,
                    context_tags=["creative", "expansion"] if is_follow_up else ["creative"]
                )
            except Exception:
                pass

        return {
            "ok": True,
            "op": op,
            "payload": result_payload
        }

    # EXECUTE_STEP: Phase 8 - Execute a creative/imagination step
    if op == "EXECUTE_STEP":
        step = payload.get("step") or {}
        step_id = payload.get("step_id", 0)
        context = payload.get("context") or {}

        # Extract step details
        description = step.get("description", "")
        step_input = step.get("input") or {}
        task = step_input.get("task", description)

        # Use HYPOTHESIZE to generate creative ideas
        hyp_result = service_api({"op": "HYPOTHESIZE", "payload": {"prompt": task, "n": 3}})

        if hyp_result.get("ok"):
            hyp_payload = hyp_result.get("payload") or {}
            hypotheses = hyp_payload.get("hypotheses", [])

            output = {
                "ideas": [h.get("content") for h in hypotheses],
                "hypotheses": hypotheses,
                "task": task,
                "is_continuation": is_follow_up
            }

            # Add routing hint for creative step execution
            if _continuation_helpers_available:
                try:
                    output["routing_hint"] = create_routing_hint(
                        brain_name="imaginer",
                        action="creative_expansion" if is_follow_up else "creative_step",
                        confidence=0.75,
                        context_tags=["creative", "execution", "expansion"] if is_follow_up else ["creative", "execution"]
                    )
                except Exception:
                    pass

            return {"ok": True, "payload": {
                "output": output,
                "patterns_used": ["creative:brainstorming"]
            }}

        return {"ok": False, "error": {"code": "HYPOTHESIZE_FAILED", "message": "Failed to generate ideas"}}

    return {"ok": False, "op": op, "error": "unknown operation"}

# Standard service contract: handle is the entry point
service_api = handle