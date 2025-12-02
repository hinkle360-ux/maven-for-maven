#!/usr/bin/env python3
"""
Phase 8 Test Runner
Executes the comprehensive Phase 8 test suite for the Task Execution Engine.
"""

import sys
from pathlib import Path

# Add maven2_fix to path
MAVEN_ROOT = Path(__file__).parent
sys.path.insert(0, str(MAVEN_ROOT))

# Import and run tests
from tests.test_phase8_task_execution import run_all_tests

if __name__ == "__main__":
    print("Starting Phase 8 Test Suite...")
    print(f"Maven Root: {MAVEN_ROOT}")
    print()

    success = run_all_tests()

    if success:
        print("\n✅ Phase 8 tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some Phase 8 tests failed.")
        sys.exit(1)
