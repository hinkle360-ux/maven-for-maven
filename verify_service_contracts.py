#!/usr/bin/env python3
"""
Phase A Service Contract Verification Script
=============================================

Verifies that all cognitive brains and domain banks have proper service contracts:
- def handle(msg: Dict[str, Any]) -> Dict[str, Any]
- service_api = handle

Tests HEALTH operation for each brain.
"""

import sys
from pathlib import Path

MAVEN_ROOT = Path(__file__).parent
sys.path.insert(0, str(MAVEN_ROOT))

def test_brain_service(brain_type: str, brain_name: str, service_module_path: str) -> bool:
    """
    Test that a brain has valid service contract.

    Args:
        brain_type: "cognitive" or "domain_bank"
        brain_name: Name of the brain
        service_module_path: Import path for service module

    Returns:
        True if tests pass, False otherwise
    """
    try:
        # Import the service module
        parts = service_module_path.split('.')
        module = __import__(service_module_path, fromlist=[parts[-1]])

        # Check for service_api
        if not hasattr(module, "service_api"):
            print(f"✗ {brain_name}: missing service_api")
            return False

        # Check for handle
        if not hasattr(module, "handle"):
            print(f"✗ {brain_name}: missing handle()")
            return False

        # Check they're callable
        if not callable(module.service_api):
            print(f"✗ {brain_name}: service_api not callable")
            return False

        if not callable(module.handle):
            print(f"✗ {brain_name}: handle not callable")
            return False

        # Test HEALTH op
        try:
            response = module.service_api({"op": "HEALTH"})
            if not isinstance(response, dict):
                print(f"✗ {brain_name}: response not dict")
                return False

            if "ok" not in response:
                print(f"✗ {brain_name}: missing 'ok' field in response")
                return False

            print(f"✓ {brain_name}")
            return True
        except Exception as e:
            print(f"✗ {brain_name}: HEALTH op failed: {e}")
            return False

    except Exception as e:
        print(f"✗ {brain_name}: import/test failed: {e}")
        return False


def main():
    """Run verification tests on all brains."""
    print("=" * 70)
    print("Phase A Service Contract Verification")
    print("=" * 70)

    passed = 0
    failed = 0

    # Cognitive brains to test
    cognitive_brains = [
        "abstraction",
        "action_engine",
        "affect_priority",
        "attention",
        "autonomy",
        "belief_tracker",
        "coder",
        "committee",
        "context_management",
        "environment_context",
        "external_interfaces",
        "imaginer",
        "integrator",
        "language",
        "learning",
        "memory_librarian",
        "motivation",
        "pattern_recognition",
        "peer_connection",
        "personality",
        "planner",
        "reasoning",
        "reasoning_trace",
        "self_dmn",
        "self_model",
        "self_review",
        "sensorium",
        "system_history",
        "thought_synthesis",
    ]

    # Domain banks to test
    domain_banks = [
        "arts",
        "creative",
        "economics",
        "factual",
        "geography",
        "history",
        "language_arts",
        "law",
        "math",
        "personal",
        "philosophy",
        "procedural",
        "science",
        "stm_only",
        "technology",
        "theories_and_contradictions",
        "working_theories",
    ]

    print("\nCognitive Brains:")
    print("-" * 70)

    for brain in cognitive_brains:
        # Determine service module path
        # Most use {brain}_brain.py, but some exceptions:
        if brain == "attention":
            module_path = "brains.cognitive.attention.service.attention_service"
        elif brain == "reasoning_trace":
            module_path = "brains.cognitive.reasoning_trace.service.trace_service"
        elif brain == "reasoning":
            module_path = "brains.cognitive.reasoning.service.dual_router"
        elif brain == "context_management":
            module_path = "brains.cognitive.context_management.service.context_manager"
        elif brain == "external_interfaces":
            module_path = "brains.cognitive.external_interfaces.service.connector"
        elif brain == "learning":
            module_path = "brains.cognitive.learning.service.meta_learning"
        elif brain == "thought_synthesis":
            module_path = "brains.cognitive.thought_synthesis.service.thought_synthesizer"
        elif brain == "memory_librarian":
            module_path = "brains.cognitive.memory_librarian.service.memory_librarian"
        elif brain == "planner":
            module_path = "brains.cognitive.planner.service.planner_brain"
        elif brain == "language":
            module_path = "brains.cognitive.language.service.language_brain"
        elif brain == "action_engine":
            module_path = "brains.cognitive.action_engine.service.action_engine"
        elif brain == "belief_tracker":
            module_path = "brains.cognitive.belief_tracker.service.belief_tracker"
        elif brain == "environment_context":
            module_path = "brains.cognitive.environment_context.service.environment_brain"
        else:
            module_path = f"brains.cognitive.{brain}.service.{brain}_brain"

        if test_brain_service("cognitive", brain, module_path):
            passed += 1
        else:
            failed += 1

    print("\nDomain Banks:")
    print("-" * 70)

    for bank in domain_banks:
        module_path = f"brains.domain_banks.{bank}.service.{bank}_bank"

        if test_brain_service("domain_bank", bank, module_path):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        print("\n❌ Some service contracts are missing or broken!")
        return 1
    else:
        print("\n✅ All service contracts are properly implemented!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
