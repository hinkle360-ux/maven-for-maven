"""
Runtime Version Verification Script
====================================

This script verifies which version of the code is actually being executed.
Run this to confirm you're using the updated codebase with routing learning.

Usage:
    python verify_runtime_version.py

The script will check for:
1. Routing learning functions
2. Diagnostic markers in the code
3. File locations and timestamps
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_version():
    print("="* 70)
    print("MAVEN RUNTIME VERSION VERIFICATION")
    print("="* 70)

    # 1. Check working directory
    print(f"\n1. Current Working Directory:")
    print(f"   {os.getcwd()}")

    # 2. Check Python path
    print(f"\n2. Python Executable:")
    print(f"   {sys.executable}")

    # 3. Check sys.path
    print(f"\n3. Python Path (sys.path):")
    for i, p in enumerate(sys.path[:5]):
        print(f"   [{i}] {p}")

    # 4. Check if routing learning code exists
    print(f"\n4. Checking for Updated Routing Code:")

    try:
        # Try to import the librarian memory module
        spec = importlib.util.find_spec("brains.cognitive.memory_librarian.service.librarian_memory")
        if spec and spec.origin:
            print(f"   ✓ librarian_memory.py found at:")
            print(f"     {spec.origin}")

            # Read the file and check for routing learning functions
            with open(spec.origin, 'r', encoding='utf-8') as f:
                content = f.read()

            has_learn = 'def learn_routing_for_question' in content
            has_store = 'def store_learned_routing_rule' in content
            has_retrieve = 'def retrieve_routing_rules' in content

            print(f"   ✓ learn_routing_for_question: {'FOUND' if has_learn else 'MISSING'}")
            print(f"   ✓ store_learned_routing_rule: {'FOUND' if has_store else 'MISSING'}")
            print(f"   ✓ retrieve_routing_rules: {'FOUND' if has_retrieve else 'MISSING'}")

            if not (has_learn and has_store and has_retrieve):
                print(f"\n   ⚠️  WARNING: Routing learning functions are MISSING!")
                print(f"   This explains why routing learning isn't working.")
                return False

        else:
            print(f"   ✗ librarian_memory.py NOT FOUND in Python path!")
            print(f"   This means Python cannot import the updated code.")
            return False

    except Exception as e:
        print(f"   ✗ Error checking librarian memory: {e}")
        return False

    # 5. Check reasoning brain for routing logs
    print(f"\n5. Checking Reasoning Brain for Routing Logs:")

    try:
        spec = importlib.util.find_spec("brains.cognitive.reasoning.service.reasoning_brain")
        if spec and spec.origin:
            print(f"   ✓ reasoning_brain.py found at:")
            print(f"     {spec.origin}")

            with open(spec.origin, 'r', encoding='utf-8') as f:
                content = f.read()

            has_learning_log = '[ROUTING_LEARNING]' in content
            has_learned_log = '[ROUTING_LEARNED_FOR]' in content
            has_rule_match = 'ROUTING_RULE_MATCH' in content

            print(f"   ✓ [ROUTING_LEARNING] log: {'FOUND' if has_learning_log else 'MISSING'}")
            print(f"   ✓ [ROUTING_LEARNED_FOR] log: {'FOUND' if has_learned_log else 'MISSING'}")
            print(f"   ✓ ROUTING_RULE_MATCH: {'FOUND' if has_rule_match else 'MISSING'}")

            if not (has_learning_log and has_learned_log):
                print(f"\n   ⚠️  WARNING: Routing log markers are MISSING!")
                print(f"   This explains why you don't see routing logs in output.")
                return False

        else:
            print(f"   ✗ reasoning_brain.py NOT FOUND in Python path!")
            return False

    except Exception as e:
        print(f"   ✗ Error checking reasoning brain: {e}")
        return False

    # 6. Summary
    print("\n" + "="* 70)
    print("✓ ALL CHECKS PASSED")
    print("="* 70)
    print("\nThe updated routing learning code IS present and importable.")
    print("If you're still not seeing routing logs, the issue is likely:")
    print("  1. The code path isn't reaching the routing learning section")
    print("  2. Teacher Brain isn't being called with TEACH_ROUTING operation")
    print("  3. Runtime conditions prevent routing learning from triggering")
    print("\nNext steps: Add debug prints at the entry point of your question")
    print("processing to trace execution flow.")
    return True

if __name__ == "__main__":
    success = check_version()
    sys.exit(0 if success else 1)
