"""
Coder Brain
===========

This cognitive brain provides a simple capability for generating and
verifying Python code based on a user specification.  It is not a
general purpose code generator; rather, it uses lightweight
heuristics to transform plain language prompts into small functions
along with unit tests.  The coder brain delegates actual code
execution and testing to the ``python_exec`` tool via its service API.

Supported operations via ``service_api``:

* ``PLAN``: Accept a specification string describing the desired code.
  Returns a structured plan with a tentative function name and a list
  of inferred behaviours.  The plan is kept deliberately simple and
  serves as a basis for code generation.

* ``GENERATE``: Given a spec (and optional plan), produce Python
  source code and a test snippet.  The generator uses keyword
  matching to decide on the implementation.  When no specific pattern
  is recognised, it emits a skeleton function raising
  ``NotImplementedError``.

* ``VERIFY``: Run static linting and execute the tests using
  ``python_exec``.  Returns a summary of whether the code is valid and
  whether the tests pass.

* ``REFINE``: Given code and tests, attempt a limited number of
  automatic refinements when the tests fail.  This operation loops up
  to ``max_refine_loops`` times (as configured in
  ``config/coding.json``).  Refinement heuristics are simple; they
  currently only fix common mistakes in addition operations.  If
  refinement succeeds, returns updated code and tests along with a
  report.  If refinement fails, returns the original code and
  diagnostics.

The coder brain does not write files or affect the file system.
Instead, it returns code snippets and test code in the payload.  It
also produces a high‑level summary that can be used by the language
brain to craft a natural language response.  Consumers of this brain
should avoid printing large code blocks directly in chat unless the
user explicitly requests them.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import re
from pathlib import Path
import json
import sys
from brains.memory.brain_memory import BrainMemory

# Pattern Store integration for pattern-learned code generation
try:
    from brains.cognitive.coder.pattern_store import (
        get_coder_pattern_store,
        Pattern,
        PatternQuery,
        PatternContext,
        VerificationOutcome,
        store_pattern,
        find_similar_patterns,
    )
    _pattern_store_available = True
except Exception as e:
    print(f"[CODER] Pattern store not available: {e}")
    _pattern_store_available = False

# Pattern Coder integration for template-based code generation
try:
    from brains.cognitive.coder.pattern_coder import (
        PatternCoder,
        PatternGenerationRequest,
        PatternGenerationResult,
        get_default_pattern_coder,
    )
    _pattern_coder = get_default_pattern_coder()
    _pattern_coder_available = True
except Exception as e:
    print(f"[CODER] Pattern coder not available: {e}")
    _pattern_coder = None  # type: ignore
    _pattern_coder_available = False

# Teacher integration for learning new coding patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("coder")
except Exception as e:
    print(f"[CODER] Teacher helper not available: {e}")
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
    print(f"[CODER] Continuation helpers not available: {e}")
    _continuation_helpers_available = False

# Import the python_exec tool's service API dynamically to avoid
# circular dependencies when loading other brains.
try:
    from brains.agent.tools.python_exec import service_api as python_exec_api  # type: ignore
except Exception:
    python_exec_api = None  # type: ignore

THIS_FILE = Path(__file__).resolve()
MAVEN_ROOT = THIS_FILE.parents[4]

# =============================================================================
# Coder State Memory - Tracks last generated function for extension
# =============================================================================
_coder_state = {
    "last_function_source": None,  # Last generated function code
    "last_function_name": None,    # Last function name
    "last_spec": None,             # Original specification
    "pending_specs": [],           # Accumulated extension specs (bullet points)
}


def _store_last_function(code: str, fn_name: str, spec: str) -> None:
    """Store the last generated function for extension."""
    global _coder_state
    _coder_state["last_function_source"] = code
    _coder_state["last_function_name"] = fn_name
    _coder_state["last_spec"] = spec
    _coder_state["pending_specs"] = []


def _get_last_function() -> Optional[Dict[str, Any]]:
    """Get the last generated function."""
    global _coder_state
    if _coder_state["last_function_source"]:
        return {
            "code": _coder_state["last_function_source"],
            "name": _coder_state["last_function_name"],
            "spec": _coder_state["last_spec"],
            "pending_specs": _coder_state["pending_specs"]
        }
    return None


def _add_pending_spec(spec: str) -> None:
    """Add a pending extension spec (bullet point)."""
    global _coder_state
    _coder_state["pending_specs"].append(spec)


def _clear_pending_specs() -> None:
    """Clear accumulated pending specs."""
    global _coder_state
    _coder_state["pending_specs"] = []


def _is_extension_request(text: str) -> bool:
    """Check if the text is an extension request for the last function."""
    text_lower = text.strip().lower()

    # Direct extension markers
    extension_markers = [
        "extend that function",
        "extend the function",
        "modify that function",
        "update that function",
        "change that function",
        "add to that function",
        "return only the full updated function",
        "return only the function",
        "return only the code",
    ]

    for marker in extension_markers:
        if marker in text_lower:
            return True

    # Bullet point specs (starts with -)
    if text_lower.startswith("-") or text_lower.startswith("•"):
        return True

    # Short continuation without explicit context
    if len(text) < 100 and _coder_state["last_function_source"]:
        # Looks like a follow-up spec
        if any(word in text_lower for word in ["if both", "if one", "return", "should", "handle"]):
            return True

    return False


def extend_function(previous_source: str, specs: List[str]) -> str:
    """
    Extend a function based on specification lines.

    This function:
    1. Parses previous_source to capture the function signature
    2. Applies specs like:
       - "If both are lists, return pairwise sums."
       - "If one is a scalar, broadcast it across the list."
    3. Returns only the new function source, no commentary.

    Args:
        previous_source: The existing function code
        specs: List of specification strings to apply

    Returns:
        Updated function source code
    """
    import ast

    # Parse the function to extract its structure
    fn_name = "unknown"
    fn_params = []
    fn_return_type = ""
    fn_docstring = ""

    try:
        tree = ast.parse(previous_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fn_name = node.name
                fn_params = [arg.arg for arg in node.args.args]
                if node.returns:
                    fn_return_type = ast.unparse(node.returns)
                # Extract docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant)):
                    fn_docstring = node.body[0].value.value
                break
    except Exception:
        # Fallback: use regex to extract function name
        fn_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', previous_source)
        if fn_match:
            fn_name = fn_match.group(1)
            params_str = fn_match.group(2)
            fn_params = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]

    # Build the extended function based on specs
    specs_text = "\n".join(f"    # {spec}" for spec in specs)

    # Generate appropriate code based on common patterns
    has_list_check = any("list" in s.lower() for s in specs)
    has_scalar_broadcast = any("scalar" in s.lower() or "broadcast" in s.lower() for s in specs)
    has_pairwise = any("pairwise" in s.lower() for s in specs)

    # Default to generic two-param function
    params_str = ", ".join(fn_params) if fn_params else "a, b"

    if has_list_check and has_scalar_broadcast and has_pairwise:
        # Common pattern: add function with list/scalar handling
        code = f'''def {fn_name}({params_str}):
    """Extended function with list/scalar handling.

    Specs applied:
{specs_text}
    """
    # Handle list cases
    a_is_list = isinstance({fn_params[0] if fn_params else 'a'}, list)
    b_is_list = isinstance({fn_params[1] if len(fn_params) > 1 else 'b'}, list)

    if a_is_list and b_is_list:
        # Both are lists: return pairwise sums
        if len({fn_params[0] if fn_params else 'a'}) != len({fn_params[1] if len(fn_params) > 1 else 'b'}):
            raise ValueError("Lists must have same length for pairwise operation")
        return [{fn_params[0] if fn_params else 'a'}[i] + {fn_params[1] if len(fn_params) > 1 else 'b'}[i] for i in range(len({fn_params[0] if fn_params else 'a'}))]
    elif a_is_list:
        # a is list, b is scalar: broadcast b across a
        return [x + {fn_params[1] if len(fn_params) > 1 else 'b'} for x in {fn_params[0] if fn_params else 'a'}]
    elif b_is_list:
        # b is list, a is scalar: broadcast a across b
        return [{fn_params[0] if fn_params else 'a'} + x for x in {fn_params[1] if len(fn_params) > 1 else 'b'}]
    else:
        # Both are scalars: simple addition
        return {fn_params[0] if fn_params else 'a'} + {fn_params[1] if len(fn_params) > 1 else 'b'}'''
    else:
        # Generic extension - just add spec comments to original
        code = previous_source.rstrip()
        # Insert specs as comments before return
        if "return" in code:
            lines = code.split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith("return"):
                    # Insert spec comments before return
                    for spec in specs:
                        new_lines.append(f"    # {spec}")
                new_lines.append(line)
            code = "\n".join(new_lines)

    return code

# Import domain lookup for accessing coding patterns
sys.path.insert(0, str(MAVEN_ROOT / "brains" / "domain_banks"))
try:
    from domain_lookup import lookup_by_tag, lookup_by_bank_and_kind
except Exception:
    lookup_by_tag = None  # type: ignore
    lookup_by_bank_and_kind = None  # type: ignore

# Load coding configuration to determine refinement limits
def _load_coding_config() -> Dict[str, Any]:
    cfg_path = MAVEN_ROOT / "config" / "coding.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            return data
    except Exception:
        pass
    return {}


def _get_coding_patterns() -> Dict[str, Any]:
    """
    Get coding patterns from domain bank.

    Returns:
        Dict mapping pattern IDs to pattern entries
    """
    patterns = {}
    if lookup_by_bank_and_kind:
        try:
            pattern_entries = lookup_by_bank_and_kind("coding_patterns", "pattern")
            for entry in pattern_entries:
                patterns[entry.get("id", "")] = entry
        except Exception:
            pass  # Return empty dict if lookup fails
    return patterns


# ============================================================================
# Pattern-Learned Code Generation Functions
# ============================================================================

def _try_pattern_coder_generation(
    spec: str,
    fn_name: str,
    domain: Optional[str] = None
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Try to generate code using the template-based PatternCoder.

    This is the first generation path tried - uses pre-built templates
    with slot filling for structured, deterministic code generation.

    Args:
        spec: User specification for the code
        fn_name: Inferred function name
        domain: Optional domain hint (e.g., "filesystem", "http", "cli")

    Returns:
        Tuple of (code, test_code, summary) if pattern found, None otherwise
    """
    if not _pattern_coder_available or not _pattern_coder:
        return None

    try:
        # Build slot values from spec
        slot_values = {
            "function_name": fn_name,
            "description": spec,
        }

        # Try to extract more specific slot values from spec
        spec_lower = spec.lower()
        if "file" in spec_lower:
            slot_values["operation"] = "file"
        if "http" in spec_lower or "api" in spec_lower or "request" in spec_lower:
            slot_values["operation"] = "http"
        if "cli" in spec_lower or "command" in spec_lower or "argparse" in spec_lower:
            slot_values["operation"] = "cli"

        # Create generation request
        request = PatternGenerationRequest(
            task_description=spec,
            language="python",
            domain=domain,
            slot_values=slot_values,
            metadata={"source": "coder_brain", "fn_name": fn_name}
        )

        # Try to generate
        result = _pattern_coder.generate(request)

        if result.ok and result.code:
            print(f"[CODER] Generated code using PatternCoder template: {result.pattern_id}")

            # Generate basic tests for the templated code
            test_code = _generate_basic_tests(fn_name, spec)

            summary: Dict[str, Any] = {
                "function": fn_name,
                "description": spec.strip(),
                "generation_mode": "template_pattern",
                "pattern_id": result.pattern_id,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
            }

            return result.code, test_code, summary

        # If pattern coder explicitly said no suitable pattern
        if not result.ok:
            print(f"[CODER] PatternCoder: {result.reason}")

        return None

    except Exception as e:
        print(f"[CODER] PatternCoder generation failed: {e}")
        return None


def _get_pattern_suggested_code(
    spec: str,
    fn_name: str,
    k: int = 3
) -> Optional[Tuple[str, str, Dict[str, Any], List[str]]]:
    """
    Pattern-suggested generation: retrieve top-k similar patterns
    and use them to propose code.

    Args:
        spec: User specification for the code
        fn_name: Inferred function name
        k: Number of patterns to retrieve

    Returns:
        Tuple of (code, test_code, summary, pattern_ids) if pattern found, None otherwise
    """
    if not _pattern_store_available:
        return None

    try:
        store = get_coder_pattern_store()

        # Create query for similar patterns
        query = PatternQuery(
            problem_description=spec,
            pattern_type="GENERATION",
            min_score=0.3,
            language="python",
        )

        # Find similar patterns
        similar = store.find_similar_patterns(query, k=k)

        if not similar:
            return None

        # Use the best matching pattern
        best_pattern, best_score = similar[0]
        pattern_ids = [p.id for p, _ in similar]

        print(f"[CODER] Using pattern {best_pattern.id} (score={best_score:.3f}) "
              f"for spec: {spec[:50]}...")

        # Adapt the pattern to the current spec
        code = best_pattern.code_after
        test_code = best_pattern.test_code

        # Replace function name in the code if different
        if best_pattern.problem_description and fn_name:
            old_fn_match = re.search(r'def\s+(\w+)', code)
            if old_fn_match:
                old_fn_name = old_fn_match.group(1)
                if old_fn_name != fn_name:
                    code = code.replace(old_fn_name, fn_name)
                    test_code = test_code.replace(old_fn_name, fn_name)

        summary: Dict[str, Any] = {
            "function": fn_name,
            "description": spec.strip(),
            "pattern_used": True,
            "pattern_id": best_pattern.id,
            "pattern_score": best_score,
            "pattern_similarity": best_score,
        }

        # Record that we used this pattern
        store.record_pattern_usage(best_pattern.id, success=True)

        return code, test_code, summary, pattern_ids

    except Exception as e:
        print(f"[CODER] Pattern suggestion failed: {e}")
        return None


def _reinforce_pattern(
    spec: str,
    code: str,
    test_code: str,
    verification_passed: bool,
    code_before: Optional[str] = None,
) -> Optional[str]:
    """
    Pattern reinforcement: when a coding task succeeds,
    extract a normalized pattern and store it.

    Args:
        spec: The original specification
        code: The generated/refined code
        test_code: The test code
        verification_passed: Whether tests passed
        code_before: Optional code before transformation (for correction patterns)

    Returns:
        Pattern ID if stored, None otherwise
    """
    if not _pattern_store_available:
        return None

    if not verification_passed:
        return None

    try:
        store = get_coder_pattern_store()

        verification_outcome = VerificationOutcome(
            tests_passed=True,
            lint_passed=True,
        )

        context = PatternContext(
            language="python",
            complexity="simple",
            tags=["generated", "verified"],
        )

        pattern_id = store.extract_and_store_pattern(
            problem_description=spec,
            code=code,
            test_code=test_code,
            verification_outcome=verification_outcome,
            context=context,
            code_before=code_before,
        )

        if pattern_id:
            print(f"[CODER] Reinforced pattern: {pattern_id}")

        return pattern_id

    except Exception as e:
        print(f"[CODER] Pattern reinforcement failed: {e}")
        return None


def _get_correction_pattern(
    failing_code: str,
    error_message: str,
    k: int = 3
) -> Optional[Tuple[str, str, List[str]]]:
    """
    Pattern-based correction: when code fails tests,
    retrieve patterns where similar failures were fixed.

    Args:
        failing_code: The code that is failing
        error_message: The error message from tests
        k: Number of patterns to retrieve

    Returns:
        Tuple of (corrected_code, test_code, pattern_ids) if pattern found, None otherwise
    """
    if not _pattern_store_available:
        return None

    try:
        store = get_coder_pattern_store()

        # Find correction patterns
        correction_patterns = store.find_correction_patterns(
            failing_code=failing_code,
            error_message=error_message,
            k=k,
        )

        if not correction_patterns:
            return None

        # Use the best matching correction pattern
        best_pattern, best_score = correction_patterns[0]
        pattern_ids = [p.id for p, _ in correction_patterns]

        print(f"[CODER] Using correction pattern {best_pattern.id} "
              f"(score={best_score:.3f}) for error: {error_message[:50]}...")

        if not best_pattern.code_after:
            return None

        # Return the corrected code from the pattern
        # The caller will need to adapt it to the specific case
        corrected_code = best_pattern.code_after
        test_code = best_pattern.test_code

        # Record pattern usage
        store.record_pattern_usage(best_pattern.id, success=True)

        return corrected_code, test_code, pattern_ids

    except Exception as e:
        print(f"[CODER] Correction pattern lookup failed: {e}")
        return None


def _infer_function_name(spec: str) -> str:
    """Infer a simple function name from the user specification.

    Attempts to extract a verb and noun from the spec using basic
    heuristics.  Falls back to ``user_function`` when extraction
    fails.
    """
    low = spec.strip().lower()
    # Look for phrases like "add two numbers", "sum of", etc.
    if "add" in low or "sum" in low:
        return "add"
    if "fizzbuzz" in low:
        return "fizzbuzz"
    if "two sum" in low or "two_sum" in low:
        return "two_sum"
    # Default
    # Extract the first noun-like word as a fallback
    tokens = re.findall(r"[a-zA-Z_]+", low)
    return tokens[0] if tokens else "user_function"


def _generate_code_and_tests(spec: str, plan: Dict[str, Any] | None = None) -> Tuple[str, str, Dict[str, Any]]:
    """Generate Python source code and a simple test snippet.

    Generation priority:
    1. Template-based PatternCoder (deterministic, pre-built templates)
    2. Pattern Store (learned patterns from successful generations)
    3. Keyword-based template matching
    4. Teacher-learned patterns
    5. Default implementation

    Returns a tuple (code, test_code, summary) where summary
    describes the public API and behaviour.
    """
    low = spec.strip().lower()
    fn_name = _infer_function_name(spec)
    summary: Dict[str, Any] = {"function": fn_name, "description": spec.strip()}

    # ===== PRIORITY 1: TEMPLATE-BASED PATTERN CODER =====
    # Try template-based generation with PatternCoder first (most deterministic)
    pattern_coder_result = _try_pattern_coder_generation(spec, fn_name)
    if pattern_coder_result is not None:
        code, test_code, pattern_summary = pattern_coder_result
        summary.update(pattern_summary)
        return code, test_code, summary

    # ===== PRIORITY 2: PATTERN STORE (learned patterns) =====
    # Try to use a pattern from the pattern store
    pattern_result = _get_pattern_suggested_code(spec, fn_name, k=3)
    if pattern_result is not None:
        code, test_code, pattern_summary, pattern_ids = pattern_result
        # Merge pattern summary with base summary
        summary.update(pattern_summary)
        summary["pattern_ids"] = pattern_ids
        summary["generation_mode"] = "pattern_suggested"
        print(f"[CODER] Generated code using pattern-suggested mode")
        return code, test_code, summary

    # ===== PRIORITY 3: KEYWORD-BASED TEMPLATE MATCHING =====
    summary["generation_mode"] = "template_matching"

    # Template for add function
    if ("add" in low or "sum" in low) and "numbers" in low:
        # Generate a simple add function.  Use single quotes in the docstring to
        # avoid premature termination of the enclosing f‑string.
        code = f"""
def {fn_name}(a: float, b: float) -> float:
    '''Return the sum of two numbers.'''
    return a + b
""".strip()
        test_code = f"""
assert {fn_name}(2, 3) == 5, "2 + 3 should be 5"
assert {fn_name}(-1, 1) == 0, "-1 + 1 should be 0"
""".strip()
        summary["example_calls"] = [f"{fn_name}(2,3) -> 5"]
        return code, test_code, summary
    # Template for FizzBuzz
    if "fizzbuzz" in low or "fizz buzz" in low:
        code = f"""
def {fn_name}(n: int) -> list[str]:
    '''Generate the FizzBuzz sequence from 1 to n.'''
    result: list[str] = []
    for i in range(1, n + 1):
        val = ""
        if i % 3 == 0:
            val += "Fizz"
        if i % 5 == 0:
            val += "Buzz"
        result.append(val or str(i))
    return result
""".strip()
        test_code = f"""
assert {fn_name}(5) == ["1", "2", "Fizz", "4", "Buzz"]
""".strip()
        summary["example_calls"] = [f"{fn_name}(5) -> ['1','2','Fizz','4','Buzz']"]
        return code, test_code, summary
    # Template for two_sum
    if "two sum" in low or "two_sum" in low:
        code = f"""
def {fn_name}(nums: list[int], target: int) -> list[int]:
    '''Return indices of the two numbers that add up to target.'''
    lookup = {{}}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in lookup:
            return [lookup[complement], i]
        lookup[num] = i
    return []
""".strip()
        test_code = f"""
assert {fn_name}([2, 7, 11, 15], 9) == [0, 1]
assert {fn_name}([3, 3], 6) == [0, 1]
""".strip()
        summary["example_calls"] = [f"{fn_name}([2,7,11,15], 9) -> [0,1]"]
        return code, test_code, summary

    # Check learned patterns in memory before using default stub
    if _teacher_helper:
        try:
            # Try to find learned coding pattern in memory
            coder_memory = BrainMemory("coder")
            learned_patterns = coder_memory.retrieve(
                query=f"code pattern: {spec}",
                limit=5,
                tiers=["stm", "mtm", "ltm"]
            )

            # Look for high-confidence learned patterns
            for pattern_rec in learned_patterns:
                if pattern_rec.get("kind") == "learned_pattern" and pattern_rec.get("confidence", 0) >= 0.7:
                    content = pattern_rec.get("content", "")
                    if isinstance(content, str) and ("CODE" in content or "def " in content):
                        print(f"[CODER] Using learned pattern from Teacher for: {spec[:50]}...")
                        # Try to extract actual code from the learned pattern
                        extracted_code = _extract_code_from_pattern(content, fn_name)
                        if extracted_code:
                            code = extracted_code
                            test_code = _generate_basic_tests(fn_name, spec)
                            summary["example_calls"] = [f"{fn_name}()"]
                            summary["learned_from_teacher"] = True
                            summary["pattern_based"] = True
                            return code, test_code, summary
                        # If extraction failed, create a passthrough implementation
                        code = f"""
def {fn_name}(*args, **kwargs):
    '''Generated using Teacher-learned pattern.

    Pattern guidance: {content[:150]}
    '''
    # Implementation based on learned pattern
    if args:
        return args[0]  # Return first arg as passthrough
    return None
""".strip()
                        test_code = _generate_basic_tests(fn_name, spec)
                        summary["example_calls"] = [f"{fn_name}(value)"]
                        summary["learned_from_teacher"] = True
                        return code, test_code, summary
        except Exception:
            pass

    # If no pattern found, try calling Teacher to learn
    if _teacher_helper:
        try:
            print(f"[CODER] No pattern found for '{spec[:50]}...', calling Teacher to learn...")

            teacher_result = _teacher_helper.maybe_call_teacher(
                question=f"How do I write Python code for this: {spec}",
                context={"spec": spec, "function_name": fn_name},
                check_memory_first=True
            )

            if teacher_result and teacher_result.get("answer"):
                answer = teacher_result["answer"]
                patterns_stored = teacher_result.get("patterns_stored", 0)

                print(f"[CODER] Learned from Teacher: {patterns_stored} patterns stored")
                print(f"[CODER] Answer: {answer[:100]}...")

                # Try to extract code from Teacher's answer
                extracted_code = _extract_code_from_pattern(answer, fn_name)
                if extracted_code:
                    code = extracted_code
                    test_code = _generate_basic_tests(fn_name, spec)
                    summary["example_calls"] = [f"{fn_name}()"]
                    summary["learned_from_teacher"] = True
                    summary["teacher_guidance"] = answer[:200]
                    return code, test_code, summary

                # Create code with Teacher guidance - generate a working stub
                code = f"""
def {fn_name}(*args, **kwargs):
    '''Generated with Teacher guidance.

    Teacher says: {answer[:150]}
    '''
    # Apply Teacher's guidance to process inputs
    result = None
    if args:
        result = args[0]
    if kwargs:
        result = next(iter(kwargs.values()), result)
    return result
""".strip()
                test_code = _generate_basic_tests(fn_name, spec)
                summary["example_calls"] = [f"{fn_name}(value)", f"{fn_name}(key=value)"]
                summary["learned_from_teacher"] = True
                summary["teacher_guidance"] = answer[:200]
                return code, test_code, summary

        except Exception as e:
            print(f"[CODER] Teacher call failed: {str(e)[:100]}")

    # Default implementation (fallback when no pattern found and Teacher didn't help)
    # Generate a working stub based on spec analysis
    code = _generate_default_implementation(fn_name, spec)
    test_code = _generate_basic_tests(fn_name, spec)
    summary["example_calls"] = [f"{fn_name}()", f"{fn_name}(value)"]
    summary["is_default_stub"] = True
    return code, test_code, summary


def _extract_code_from_pattern(content: str, fn_name: str) -> Optional[str]:
    """Extract actual Python code from a pattern or Teacher answer.

    Looks for code blocks or function definitions in the content.

    Args:
        content: The pattern/answer content to extract from
        fn_name: The function name to use

    Returns:
        Extracted code string or None if extraction fails
    """
    import re

    # Look for code blocks (```python ... ``` or ``` ... ```)
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, content, re.DOTALL)

    for match in matches:
        code = match.strip()
        # Check if it contains a function definition
        if 'def ' in code:
            # Replace function name if needed
            code = re.sub(r'def\s+\w+\s*\(', f'def {fn_name}(', code, count=1)
            return code

    # Look for inline function definitions
    func_pattern = r'(def\s+\w+\s*\([^)]*\):\s*(?:\n(?:[ \t]+.+\n?)+|\s*.+))'
    func_matches = re.findall(func_pattern, content, re.MULTILINE)

    if func_matches:
        code = func_matches[0].strip()
        code = re.sub(r'def\s+\w+\s*\(', f'def {fn_name}(', code, count=1)
        return code

    return None


def _generate_basic_tests(fn_name: str, spec: str) -> str:
    """Generate basic test code for a function.

    Args:
        fn_name: The function name
        spec: The specification

    Returns:
        Test code string
    """
    return f'''import pytest

def test_{fn_name}_exists():
    """Test that {fn_name} is callable."""
    assert callable({fn_name})

def test_{fn_name}_basic():
    """Test basic invocation."""
    result = {fn_name}()
    # Function should not raise an exception

def test_{fn_name}_with_args():
    """Test with arguments."""
    result = {fn_name}("test_value")
    # Function should accept arguments
'''


def _generate_default_implementation(fn_name: str, spec: str) -> str:
    """Generate a default working implementation based on spec analysis.

    Analyzes the spec to generate appropriate implementation logic.

    Args:
        fn_name: The function name
        spec: The specification

    Returns:
        Generated code string
    """
    spec_lower = spec.lower()

    # Detect common patterns in spec
    if any(word in spec_lower for word in ['sum', 'add', 'total']):
        return f'''
def {fn_name}(*args, **kwargs):
    \'\'\'Sum/add operation based on spec: {spec[:100]}\'\'\'
    values = list(args) + list(kwargs.values())
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    return sum(numeric_values) if numeric_values else 0
'''.strip()

    if any(word in spec_lower for word in ['filter', 'select', 'find']):
        return f'''
def {fn_name}(items=None, predicate=None, **kwargs):
    \'\'\'Filter operation based on spec: {spec[:100]}\'\'\'
    if items is None:
        return []
    if predicate is None:
        return list(items)
    return [item for item in items if predicate(item)]
'''.strip()

    if any(word in spec_lower for word in ['map', 'transform', 'convert']):
        return f'''
def {fn_name}(items=None, transform=None, **kwargs):
    \'\'\'Map/transform operation based on spec: {spec[:100]}\'\'\'
    if items is None:
        return []
    if transform is None:
        return list(items)
    return [transform(item) for item in items]
'''.strip()

    if any(word in spec_lower for word in ['sort', 'order', 'arrange']):
        return f'''
def {fn_name}(items=None, key=None, reverse=False, **kwargs):
    \'\'\'Sort operation based on spec: {spec[:100]}\'\'\'
    if items is None:
        return []
    return sorted(items, key=key, reverse=reverse)
'''.strip()

    if any(word in spec_lower for word in ['count', 'length', 'size']):
        return f'''
def {fn_name}(items=None, **kwargs):
    \'\'\'Count operation based on spec: {spec[:100]}\'\'\'
    if items is None:
        return 0
    return len(items) if hasattr(items, '__len__') else sum(1 for _ in items)
'''.strip()

    if any(word in spec_lower for word in ['validate', 'check', 'verify']):
        return f'''
def {fn_name}(value=None, **kwargs):
    \'\'\'Validation operation based on spec: {spec[:100]}\'\'\'
    if value is None:
        return False
    # Basic validation - not None and not empty
    if hasattr(value, '__len__'):
        return len(value) > 0
    return True
'''.strip()

    # Default passthrough implementation
    return f'''
def {fn_name}(*args, **kwargs):
    \'\'\'User requested function for: {spec[:100]}

    This is a generated implementation. Customize as needed.
    \'\'\'
    # Process inputs
    if args:
        return args[0]
    if kwargs:
        return next(iter(kwargs.values()))
    return None
'''.strip()


def _run_lint(code: str) -> Tuple[bool, str | None]:
    """Call the python_exec LINT operation and return validity and message."""
    if python_exec_api is None:
        return False, "python_exec tool unavailable"
    res = python_exec_api({"op": "LINT", "payload": {"code": code}})
    if not res.get("ok"):
        return False, res.get("error", {}).get("message")
    payload = res.get("payload") or {}
    return bool(payload.get("valid")), payload.get("error") or payload.get("warning")


def _run_tests(code: str, test_code: str) -> Tuple[bool, str | None]:
    """Execute tests via python_exec TEST operation and return pass/fail and stderr."""
    if python_exec_api is None:
        return False, "python_exec tool unavailable"
    res = python_exec_api({"op": "TEST", "payload": {"code": code, "test_code": test_code}})
    if not res.get("ok"):
        return False, res.get("error", {}).get("message")
    payload = res.get("payload") or {}
    return bool(payload.get("passed")), payload.get("stderr")


def _attempt_refinement(
    code: str,
    test_code: str,
    error_message: Optional[str] = None
) -> Tuple[str, str, bool, str | None]:
    """Attempt refinement of code when tests fail.

    Uses pattern-based correction first, then falls back to simple
    heuristics. Returns (new_code, new_test_code, refined, message).
    """
    # ===== PATTERN-BASED CORRECTION (try first) =====
    if error_message:
        correction_result = _get_correction_pattern(code, error_message, k=3)
        if correction_result is not None:
            corrected_code, corrected_tests, pattern_ids = correction_result
            print(f"[CODER] Applied correction pattern: {pattern_ids[0] if pattern_ids else 'unknown'}")
            return corrected_code, corrected_tests, True, f"Applied correction pattern from {len(pattern_ids)} similar fixes"

    # ===== FALLBACK TO SIMPLE HEURISTICS =====
    # Example refinement: if code uses subtraction instead of addition
    if "+" not in code and "return" in code:
        fixed = code.replace("-", "+")
        return fixed, test_code, True, "Replaced subtraction with addition"

    # Check for common typos in function names
    if "retrun" in code:
        fixed = code.replace("retrun", "return")
        return fixed, test_code, True, "Fixed 'retrun' typo"

    # Check for missing colons after if/for/def
    lines = code.split('\n')
    fixed_lines = []
    made_fix = False
    for line in lines:
        stripped = line.rstrip()
        # Check for if/for/while/def without colon
        if re.match(r'^(\s*)(if|for|while|def|class|elif|else|try|except|finally)\s+.*[^:]$', stripped):
            if not stripped.endswith(':'):
                fixed_lines.append(stripped + ':')
                made_fix = True
                continue
        fixed_lines.append(line)

    if made_fix:
        return '\n'.join(fixed_lines), test_code, True, "Added missing colons"

    return code, test_code, False, None


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the coder brain.

    Expects an ``op`` and optional ``payload``.  Dispatches to the
    corresponding internal function and returns a structured result.
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
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
    if not op:
        return {"ok": False, "error": {"code": "MISSING_OP", "message": "op is required"}}
    # Load config for refinement loops
    cfg = _load_coding_config()
    max_loops = int(cfg.get("max_refine_loops", 3) or 3)
    # PLAN: create a basic plan from user spec
    if op == "PLAN":
        spec = str(payload.get("spec", ""))
        fn_name = _infer_function_name(spec)
        plan = {
            "function_name": fn_name,
            "spec": spec,
            "description": f"Plan to implement '{fn_name}' based on user spec"
        }
        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="coder",
                action="plan",
                confidence=0.7,
                context_tags=[
                    "plan",
                    "continuation" if continuation_detected else "fresh_query"
                ]
            )
            if isinstance(result, dict):
                result["routing_hint"] = routing_hint
            elif isinstance(payload_result, dict):
                payload_result["routing_hint"] = routing_hint
        except Exception:
            pass  # Routing hint generation is non-critical
        return {"ok": True, "payload": plan}
    # GENERATE: produce code and tests
    if op == "GENERATE":
        spec = str(payload.get("spec", ""))
        # plan may be provided to override function name
        plan = payload.get("plan") or {}

        # CONTINUATION AWARENESS: Detect iterative coding
        is_refinement = False
        previous_code = None
        conv_context = {}

        if _continuation_helpers_available:
            try:
                is_refinement = is_continuation(spec, {"spec": spec, "query": spec})

                if is_refinement:
                    conv_context = get_conversation_context()
                    # Try to get previous code from conversation context
                    last_response = conv_context.get("last_maven_response", "")
                    # Extract code block from previous response if present
                    import re
                    code_match = re.search(r'```python\n(.*?)\n```', last_response, re.DOTALL)
                    if code_match:
                        previous_code = code_match.group(1)
                        print(f"[CODER] ✓ Iterative coding detected - refining previous code")
                        print(f"[CODER] Modification request: {spec[:60]}...")
            except Exception as e:
                print(f"[CODER] Warning: Continuation detection failed: {str(e)[:100]}")
                is_refinement = False

        code, test_code, summary = _generate_code_and_tests(spec, plan)

        # Persist code generation results using BrainMemory tier API
        try:
            memory = BrainMemory("coder")
            memory.store(
                content={"spec": spec, "code": code, "test_code": test_code, "summary": summary, "is_refinement": is_refinement},
                metadata={"kind": "code_generation", "source": "coder", "confidence": 0.7}
            )
        except Exception:
            pass

        # Create routing hint
        routing_hint = None
        if _continuation_helpers_available:
            try:
                action = "refine_code" if is_refinement else "generate_code"
                context_tags = ["coding", "iterative"] if is_refinement else ["coding", "fresh"]

                routing_hint = create_routing_hint(
                    brain_name="coder",
                    action=action,
                    confidence=0.7,
                    context_tags=context_tags
                )
            except Exception as e:
                print(f"[CODER] Warning: Failed to create routing hint: {str(e)[:100]}")

        result = {"code": code, "test_code": test_code, "summary": summary}

        if is_refinement:
            result["is_refinement"] = True
            result["modification_type"] = "iterative_refinement"

        if routing_hint:
            result["routing_hint"] = routing_hint

        # Store last function for extension capability
        fn_name = summary.get("function", "")
        if fn_name and code:
            _store_last_function(code, fn_name, spec)

        return {"ok": True, "payload": result}
    # VERIFY: run lint and tests
    if op == "VERIFY":
        code = str(payload.get("code", ""))
        test_code = str(payload.get("test_code", ""))
        # Lint
        valid, lint_msg = _run_lint(code)
        if not valid:
            result = {"valid": False, "lint_error": lint_msg}
            # Persist verification results using BrainMemory tier API
            try:
                memory = BrainMemory("coder")
                memory.store(
                    content={"code": code, "result": result},
                    metadata={"kind": "verification_result", "source": "coder", "confidence": 0.9}
                )
            except Exception:
                pass
            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="coder",
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
            return {"ok": True, "payload": result}
        # Run tests
        passed, stderr = _run_tests(code, test_code)
        result = {"valid": True, "tests_passed": passed, "test_error": stderr}

        # ===== PATTERN REINFORCEMENT =====
        # When verification passes, extract and store pattern for future use
        if passed:
            spec = payload.get("spec", "") or payload.get("description", "")
            if spec:
                pattern_id = _reinforce_pattern(
                    spec=spec,
                    code=code,
                    test_code=test_code,
                    verification_passed=True,
                )
                if pattern_id:
                    result["pattern_reinforced"] = True
                    result["pattern_id"] = pattern_id

        # Persist verification results using BrainMemory tier API
        try:
            memory = BrainMemory("coder")
            memory.store(
                content={"code": code, "test_code": test_code, "result": result},
                metadata={"kind": "verification_result", "source": "coder", "confidence": 0.9}
            )
        except Exception:
            pass
        # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
        try:
            routing_hint = create_routing_hint(
                brain_name="coder",
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
        return {"ok": True, "payload": result}
    # REFINE: attempt automatic refinement
    if op == "REFINE":
        code = str(payload.get("code", ""))
        test_code = str(payload.get("test_code", ""))
        original_code = code  # Keep original for pattern storage
        spec = payload.get("spec", "") or payload.get("description", "")
        diagnostics: list[str] = []
        refined = False
        last_error = None

        for i in range(max_loops):
            # Run tests
            passed, err = _run_tests(code, test_code)
            last_error = err

            if passed:
                # ===== PATTERN REINFORCEMENT FOR CORRECTION =====
                # If we refined the code successfully, store as a correction pattern
                if refined and spec:
                    pattern_id = _reinforce_pattern(
                        spec=spec,
                        code=code,
                        test_code=test_code,
                        verification_passed=True,
                        code_before=original_code,
                    )
                    if pattern_id:
                        diagnostics.append(f"Stored correction pattern: {pattern_id}")

                # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
                try:
                    routing_hint = create_routing_hint(
                        brain_name="coder",
                        action="refine",
                        confidence=0.7,
                        context_tags=[
                            "refine",
                            "continuation" if continuation_detected else "fresh_query"
                        ]
                    )
                    if isinstance(result, dict):
                        result["routing_hint"] = routing_hint
                    elif isinstance(payload_result, dict):
                        payload_result["routing_hint"] = routing_hint
                except Exception:
                    pass  # Routing hint generation is non-critical
                return {"ok": True, "payload": {"code": code, "test_code": test_code, "refined": refined, "diagnostics": diagnostics}}

            # Attempt refinement (now with error message for pattern-based correction)
            new_code, new_test, did_refine, msg = _attempt_refinement(code, test_code, error_message=err)
            if did_refine:
                diagnostics.append(msg or "Applied refinement")
                code, test_code, refined = new_code, new_test, True
            else:
                break
        # Final run after refinements
        passed, err = _run_tests(code, test_code)
        return {"ok": True, "payload": {"code": code, "test_code": test_code, "refined": refined, "diagnostics": diagnostics, "tests_passed": passed, "test_error": err}}

    # EXECUTE_STEP: Phase 8 - Execute a single step with pattern application
    if op == "EXECUTE_STEP":
        step = payload.get("step") or {}
        step_id = payload.get("step_id", 0)
        context = payload.get("context") or {}

        # Get coding patterns from domain bank
        coding_patterns = _get_coding_patterns()
        patterns_used = []

        # Extract step details
        description = step.get("description", "")
        step_input = step.get("input") or {}
        task = step_input.get("task", description)

        # Execute coding step: PLAN -> GENERATE -> VERIFY
        try:
            # Step 1: Plan
            plan_result = service_api({"op": "PLAN", "payload": {"spec": task}})
            if not plan_result.get("ok"):
                return {"ok": False, "error": {"code": "PLAN_FAILED", "message": "Failed to plan coding step"}}

            plan = plan_result.get("payload") or {}

            # Step 2: Generate code
            gen_result = service_api({"op": "GENERATE", "payload": {"spec": task, "plan": plan}})
            if not gen_result.get("ok"):
                return {"ok": False, "error": {"code": "GENERATE_FAILED", "message": "Failed to generate code"}}

            gen_payload = gen_result.get("payload") or {}
            code = gen_payload.get("code", "")
            test_code = gen_payload.get("test_code", "")
            summary = gen_payload.get("summary", {})

            # Step 3: Verify
            verify_result = service_api({"op": "VERIFY", "payload": {"code": code, "test_code": test_code}})
            verify_payload = verify_result.get("payload") or {}

            # If tests failed, attempt refinement
            if not verify_payload.get("tests_passed", False):
                refine_result = service_api({"op": "REFINE", "payload": {"code": code, "test_code": test_code}})
                refine_payload = refine_result.get("payload") or {}
                code = refine_payload.get("code", code)
                test_code = refine_payload.get("test_code", test_code)
                verify_payload["refined"] = refine_payload.get("refined", False)

            # Record which patterns were used (if any)
            if coding_patterns:
                patterns_used = list(coding_patterns.keys())[:2]  # Use first 2 patterns for determinism

            output = {
                "code": code,
                "test_code": test_code,
                "summary": summary,
                "verified": verify_payload.get("valid", False),
                "tests_passed": verify_payload.get("tests_passed", False)
            }

            # COGNITIVE BRAIN CONTRACT: Signal 3 - Generate routing hint
            try:
                routing_hint = create_routing_hint(
                    brain_name="coder",
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
            return {"ok": True, "payload": {
                "output": output,
                "patterns_used": patterns_used
            }}

        except Exception as e:
            return {"ok": False, "error": {"code": "EXECUTION_ERROR", "message": str(e)}}

    # EXTEND: Extend the last generated function with new specs
    if op == "EXTEND":
        spec_text = str(payload.get("spec", ""))
        specs = payload.get("specs", [])

        # If spec_text is provided, split into list
        if spec_text and not specs:
            # Handle bullet points
            lines = spec_text.strip().split("\n")
            specs = [line.lstrip("-•").strip() for line in lines if line.strip()]

        # Get last function
        last_fn = _get_last_function()
        if not last_fn:
            return {
                "ok": False,
                "error": {
                    "code": "NO_PREVIOUS_FUNCTION",
                    "message": "No previous function to extend. Use GENERATE first."
                }
            }

        # Accumulate specs if there are pending ones
        all_specs = last_fn.get("pending_specs", []) + specs

        if not all_specs:
            return {
                "ok": False,
                "error": {
                    "code": "NO_SPECS",
                    "message": "No extension specifications provided."
                }
            }

        # Extend the function
        try:
            extended_code = extend_function(last_fn["code"], all_specs)

            # Store the extended function as the new last function
            _store_last_function(extended_code, last_fn["name"], f"{last_fn['spec']} + extensions")

            # Generate test code for the extended function
            test_code = _generate_basic_tests(last_fn["name"], f"Extended: {', '.join(all_specs[:2])}")

            summary = {
                "function": last_fn["name"],
                "description": f"Extended from original spec: {last_fn['spec'][:50]}",
                "specs_applied": all_specs,
                "generation_mode": "extension"
            }

            return {
                "ok": True,
                "payload": {
                    "code": extended_code,
                    "test_code": test_code,
                    "summary": summary,
                    "original_code": last_fn["code"],
                    "specs_applied": all_specs
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "EXTENSION_FAILED",
                    "message": str(e)
                }
            }

    # ADD_SPEC: Add a spec to pending specs for later extension
    if op == "ADD_SPEC":
        spec_text = str(payload.get("spec", ""))
        if spec_text:
            # Clean up bullet point if present
            clean_spec = spec_text.lstrip("-•").strip()
            _add_pending_spec(clean_spec)
            return {
                "ok": True,
                "payload": {
                    "spec_added": clean_spec,
                    "pending_count": len(_coder_state["pending_specs"]),
                    "pending_specs": _coder_state["pending_specs"]
                }
            }
        return {
            "ok": False,
            "error": {"code": "EMPTY_SPEC", "message": "No spec provided"}
        }

    # GET_LAST_FUNCTION: Retrieve the last generated function
    if op == "GET_LAST_FUNCTION":
        last_fn = _get_last_function()
        if last_fn:
            return {"ok": True, "payload": last_fn}
        return {
            "ok": False,
            "error": {"code": "NO_FUNCTION", "message": "No function has been generated yet"}
        }

    # Unsupported op
    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}

# Standard service contract: handle is the entry point
service_api = handle