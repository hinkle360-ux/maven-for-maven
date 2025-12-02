#!/usr/bin/env python3
"""
Agency Tools Validation Test Suite
===================================

Comprehensive test suite to validate all Maven agency tools are working correctly.

This tests:
1. Filesystem agency (read, write, scan, analyze)
2. Git operations (status, commit simulation)
3. Hot-reload (module info, dependency tracking)
4. Self-introspection (architecture scan, brain analysis)
5. Routing intelligence (pattern matching, tool execution)
6. Execution guards (permission checks, audit logging)

Usage:
    python3 test_agency_tools.py
    python3 test_agency_tools.py --verbose
    python3 test_agency_tools.py --category filesystem
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Add Maven root to path
maven_root = Path(__file__).parent
sys.path.insert(0, str(maven_root))

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "tests": []
}


def test_category(name: str, color: str = "\033[94m"):
    """Print test category header."""
    print(f"\n{color}{'=' * 70}\033[0m")
    print(f"{color}{name}\033[0m")
    print(f"{color}{'=' * 70}\033[0m")


def test_case(name: str, func: callable, verbose: bool = False) -> bool:
    """
    Run a test case and report results.

    Args:
        name: Test name
        func: Test function to run
        verbose: Whether to print verbose output

    Returns:
        True if test passed, False otherwise
    """
    global test_results

    print(f"\n  Testing: {name}...", end=" ")

    try:
        result = func()

        if result:
            print("\033[92m✓ PASS\033[0m")
            test_results["passed"] += 1
            test_results["tests"].append({"name": name, "status": "pass"})

            if verbose and isinstance(result, dict):
                print(f"    Result: {result}")

            return True
        else:
            print("\033[91m✗ FAIL\033[0m")
            test_results["failed"] += 1
            test_results["tests"].append({"name": name, "status": "fail", "reason": "returned False"})
            return False

    except Exception as e:
        print(f"\033[91m✗ ERROR: {str(e)[:100]}\033[0m")
        test_results["failed"] += 1
        test_results["tests"].append({"name": name, "status": "fail", "reason": str(e)})

        if verbose:
            print(f"    Traceback: {traceback.format_exc()}")

        return False


# ============================================================================
# FILESYSTEM AGENCY TESTS
# ============================================================================

def test_filesystem_import():
    """Test filesystem agency can be imported."""
    from brains.tools.filesystem_agency import get_filesystem_agency
    agency = get_filesystem_agency()
    return agency is not None


def test_filesystem_scan_tree():
    """Test directory tree scanning."""
    from brains.tools.filesystem_agency import get_filesystem_agency
    agency = get_filesystem_agency()
    tree = agency.scan_directory_tree("brains/tools", max_depth=2)
    return tree and "files" in tree and tree["file_count"] > 0


def test_filesystem_list_python_files():
    """Test listing Python files."""
    from brains.tools.filesystem_agency import get_filesystem_agency
    agency = get_filesystem_agency()
    files = agency.list_python_files("brains/tools")
    return files and len(files) > 0


def test_filesystem_file_exists():
    """Test file existence checking."""
    from brains.tools.filesystem_agency import get_filesystem_agency
    agency = get_filesystem_agency()
    # Check this test file exists
    exists = agency.file_exists("test_agency_tools.py")
    return exists is True


def test_filesystem_get_file_info():
    """Test getting file information."""
    from brains.tools.filesystem_agency import get_filesystem_agency
    agency = get_filesystem_agency()
    info = agency.get_file_info("test_agency_tools.py")
    return info and "size" in info and "modified" in info


# ============================================================================
# GIT OPERATIONS TESTS
# ============================================================================

def test_git_import():
    """Test git tool can be imported."""
    from brains.tools import git_tool
    return git_tool is not None


def test_git_repo_info():
    """Test getting git repository info."""
    from brains.tools.git_tool import git_get_repo_info
    info = git_get_repo_info()
    return info and "branch" in info and "commit" in info


def test_git_current_branch():
    """Test getting current git branch."""
    from brains.tools.git_tool import git_get_current_branch
    branch = git_get_current_branch()
    return branch and len(branch) > 0


def test_git_status_detailed():
    """Test getting detailed git status."""
    from brains.tools.git_tool import git_status_detailed
    status = git_status_detailed()
    return status and "branch" in status


# ============================================================================
# HOT-RELOAD TESTS
# ============================================================================

def test_hotreload_import():
    """Test hot-reload can be imported."""
    from brains.tools.hot_reload import get_hot_reload_manager
    manager = get_hot_reload_manager()
    return manager is not None


def test_hotreload_list_modules():
    """Test listing loaded Maven modules."""
    from brains.tools.hot_reload import get_hot_reload_manager
    manager = get_hot_reload_manager()
    modules = manager.get_loaded_maven_modules()
    return modules and len(modules) > 0


def test_hotreload_check_syntax():
    """Test syntax checking."""
    from brains.tools.hot_reload import get_hot_reload_manager
    manager = get_hot_reload_manager()
    # Check this test file syntax
    valid, error = manager.check_module_syntax("test_agency_tools.py")
    return valid is True


# ============================================================================
# SELF-INTROSPECTION TESTS
# ============================================================================

def test_introspection_import():
    """Test self-introspection can be imported."""
    from brains.tools.self_introspection import get_self_introspection
    introspector = get_self_introspection()
    return introspector is not None


def test_introspection_scan_architecture():
    """Test scanning full architecture."""
    from brains.tools.self_introspection import get_self_introspection
    introspector = get_self_introspection()
    arch = introspector.scan_full_architecture()
    return arch and "total_files" in arch and arch["total_files"] > 100


def test_introspection_analyze_brains():
    """Test analyzing all brains."""
    from brains.tools.self_introspection import get_self_introspection
    introspector = get_self_introspection()
    analysis = introspector.analyze_all_brains()
    return analysis and "total_brains" in analysis


def test_introspection_analyze_dependencies():
    """Test dependency analysis."""
    from brains.tools.self_introspection import get_self_introspection
    introspector = get_self_introspection()
    deps = introspector.analyze_dependencies()
    return deps and "external_dependencies" in deps


# ============================================================================
# ROUTING INTELLIGENCE TESTS
# ============================================================================

def test_routing_patterns_import():
    """Test routing patterns can be imported."""
    from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern
    return match_agency_pattern is not None


def test_routing_pattern_match_introspection():
    """Test pattern matching for introspection query."""
    from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern
    match = match_agency_pattern("scan my codebase", threshold=0.7)
    return match is not None and match["match_score"] >= 0.7


def test_routing_pattern_match_git():
    """Test pattern matching for git query."""
    from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern
    match = match_agency_pattern("what's the git status?", threshold=0.7)
    return match is not None and match["match_score"] >= 0.7


def test_routing_pattern_match_filesystem():
    """Test pattern matching for filesystem query."""
    from brains.cognitive.integrator.agency_routing_patterns import match_agency_pattern
    match = match_agency_pattern("list all python files", threshold=0.7)
    return match is not None and match["match_score"] >= 0.7


def test_routing_executor_import():
    """Test agency executor can be imported."""
    from brains.cognitive.integrator.agency_executor import execute_agency_tool
    return execute_agency_tool is not None


# ============================================================================
# EXECUTION GUARD TESTS
# ============================================================================

def test_execution_guard_import():
    """Test execution guard can be imported."""
    from brains.tools.execution_guard import check_execution_enabled
    return check_execution_enabled is not None


def test_execution_guard_risk_classification():
    """Test operation risk classification."""
    from brains.tools.execution_guard import get_operation_risk
    risk = get_operation_risk("git_push")
    return risk == "critical"


def test_execution_guard_dangerous_ops():
    """Test listing dangerous operations."""
    from brains.tools.execution_guard import list_dangerous_operations
    ops = list_dangerous_operations()
    return ops and len(ops) > 0 and "git_push" in ops


def test_execution_guard_audit_log():
    """Test audit log access."""
    from brains.tools.execution_guard import get_audit_log
    log = get_audit_log()
    return log is not None


# ============================================================================
# INTEGRATOR BRAIN TESTS
# ============================================================================

def test_integrator_agency_helpers():
    """Test integrator brain agency helper functions."""
    from brains.cognitive.integrator.service.integrator_brain import get_current_pattern, get_agency_tool_info
    # These should not raise errors
    pattern = get_current_pattern()
    tool_info = get_agency_tool_info()
    return True  # Just check they're callable


# ============================================================================
# ACTION ENGINE TESTS
# ============================================================================

def test_action_engine_import():
    """Test action engine can be imported."""
    from brains.cognitive.action_engine.service.action_engine import service_api
    return service_api is not None


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(verbose: bool = False, category: str = "all"):
    """
    Run all agency tool tests.

    Args:
        verbose: Whether to print verbose output
        category: Category of tests to run (all, filesystem, git, hotreload, introspection, routing, guard)
    """
    print("\033[1;36m")
    print("=" * 70)
    print("Maven Agency Tools Validation Test Suite")
    print("=" * 70)
    print("\033[0m")

    categories = {
        "filesystem": [
            ("Import filesystem agency", test_filesystem_import),
            ("Scan directory tree", test_filesystem_scan_tree),
            ("List Python files", test_filesystem_list_python_files),
            ("Check file exists", test_filesystem_file_exists),
            ("Get file info", test_filesystem_get_file_info),
        ],
        "git": [
            ("Import git tool", test_git_import),
            ("Get repo info", test_git_repo_info),
            ("Get current branch", test_git_current_branch),
            ("Get detailed status", test_git_status_detailed),
        ],
        "hotreload": [
            ("Import hot-reload", test_hotreload_import),
            ("List loaded modules", test_hotreload_list_modules),
            ("Check syntax", test_hotreload_check_syntax),
        ],
        "introspection": [
            ("Import self-introspection", test_introspection_import),
            ("Scan architecture", test_introspection_scan_architecture),
            ("Analyze brains", test_introspection_analyze_brains),
            ("Analyze dependencies", test_introspection_analyze_dependencies),
        ],
        "routing": [
            ("Import routing patterns", test_routing_patterns_import),
            ("Match introspection pattern", test_routing_pattern_match_introspection),
            ("Match git pattern", test_routing_pattern_match_git),
            ("Match filesystem pattern", test_routing_pattern_match_filesystem),
            ("Import routing executor", test_routing_executor_import),
        ],
        "guard": [
            ("Import execution guard", test_execution_guard_import),
            ("Risk classification", test_execution_guard_risk_classification),
            ("List dangerous ops", test_execution_guard_dangerous_ops),
            ("Audit log access", test_execution_guard_audit_log),
        ],
        "integration": [
            ("Integrator agency helpers", test_integrator_agency_helpers),
            ("Action engine import", test_action_engine_import),
        ],
    }

    # Run tests
    if category == "all":
        for cat_name, tests in categories.items():
            test_category(f"{cat_name.upper()} TESTS")
            for test_name, test_func in tests:
                test_case(test_name, test_func, verbose)
    elif category in categories:
        test_category(f"{category.upper()} TESTS")
        for test_name, test_func in categories[category]:
            test_case(test_name, test_func, verbose)
    else:
        print(f"\033[91mUnknown category: {category}\033[0m")
        print(f"Available categories: {', '.join(categories.keys())}, all")
        return

    # Print summary
    print("\n\033[1;36m" + "=" * 70 + "\033[0m")
    print("\033[1;36mTEST SUMMARY\033[0m")
    print("\033[1;36m" + "=" * 70 + "\033[0m")
    print(f"\n  \033[92mPassed: {test_results['passed']}\033[0m")
    print(f"  \033[91mFailed: {test_results['failed']}\033[0m")
    print(f"  \033[93mSkipped: {test_results['skipped']}\033[0m")
    print(f"  Total: {test_results['passed'] + test_results['failed'] + test_results['skipped']}")

    success_rate = (test_results['passed'] / (test_results['passed'] + test_results['failed'])) * 100 if (test_results['passed'] + test_results['failed']) > 0 else 0
    print(f"\n  Success Rate: {success_rate:.1f}%")

    if test_results['failed'] == 0:
        print("\n  \033[92m✓ ALL TESTS PASSED!\033[0m\n")
        return 0
    else:
        print(f"\n  \033[91m✗ {test_results['failed']} TEST(S) FAILED\033[0m\n")
        return 1


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    category = "all"

    # Parse category argument
    for arg in sys.argv[1:]:
        if arg.startswith("--category="):
            category = arg.split("=")[1]
        elif not arg.startswith("-"):
            category = arg

    exit_code = run_all_tests(verbose=verbose, category=category)
    sys.exit(exit_code)
