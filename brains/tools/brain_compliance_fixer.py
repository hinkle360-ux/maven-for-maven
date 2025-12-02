"""
Brain Compliance Fixer
======================

Automated tool to fix brain compliance issues by adding missing process() methods.

This tool:
1. Scans all brain modules for compliance
2. Identifies missing process() methods
3. Generates appropriate process() stubs
4. Can auto-fix with backup creation

Usage:
    from brains.tools.brain_compliance_fixer import fix_brain_compliance

    # Scan and report
    report = fix_brain_compliance(dry_run=True)

    # Auto-fix with backups
    report = fix_brain_compliance(dry_run=False, create_backup=True)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import ast
import re
from pathlib import Path
from datetime import datetime


def get_maven_root() -> Path:
    """Get Maven root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / "brains").exists() and (current / "api").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find Maven root directory")


def is_brain_module(file_path: Path) -> bool:
    """
    Check if a file is likely a brain module.

    Heuristics:
    - Located in brains/cognitive/**/service/ or brains/*/service/
    - Ends with _brain.py
    - Contains "brain" in the name
    """
    path_str = str(file_path)

    # Must be in brains directory
    if "brains" not in path_str:
        return False

    # Skip test files
    if "test_" in file_path.name or "_test.py" in file_path.name:
        return False

    # Skip __init__ files
    if file_path.name == "__init__.py":
        return False

    # Check if in service directory or has brain in name
    is_service_brain = "/service/" in path_str and "_brain.py" in file_path.name
    has_brain_name = "brain" in file_path.name.lower()

    return is_service_brain or has_brain_name


def has_process_method(file_path: Path) -> bool:
    """Check if a Python file has a process() method or function."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content, filename=str(file_path))

        # Check for process function or method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process":
                return True
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "process":
                return True

        return False
    except Exception:
        return False


def get_brain_type(file_path: Path) -> str:
    """
    Determine the type of brain based on path and content.

    Returns:
        Type string: "cognitive", "service", "tool", "utility", "unknown"
    """
    path_str = str(file_path)

    if "/cognitive/" in path_str:
        return "cognitive"
    elif "/service/" in path_str:
        return "service"
    elif "/tools/" in path_str:
        return "tool"
    else:
        return "utility"


def generate_process_stub(file_path: Path, brain_name: str, brain_type: str) -> str:
    """
    Generate an appropriate process() method stub based on brain type.

    Args:
        file_path: Path to the brain file
        brain_name: Name of the brain
        brain_type: Type of brain (cognitive, service, tool, utility)

    Returns:
        Python code for the process() stub
    """
    if brain_type == "cognitive":
        stub = f'''

def process(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a query through the {brain_name} brain.

    Args:
        query: Input query to process
        context: Optional context dictionary

    Returns:
        Processing result dictionary
    """
    # TODO: Implement {brain_name} processing logic
    print(f"[{brain_name.upper()}] Processing query: {{query}}")

    return {{
        "status": "not_implemented",
        "brain": "{brain_name}",
        "query": query,
        "result": None,
        "message": "{brain_name} process() method is a stub - needs implementation"
    }}
'''
    elif brain_type == "service":
        stub = f'''

def process(data: Any) -> Any:
    """
    Process data through the {brain_name} service.

    Args:
        data: Input data to process

    Returns:
        Processed result
    """
    # TODO: Implement {brain_name} service processing
    print(f"[{brain_name.upper()}] Processing data")
    return data
'''
    elif brain_type == "tool":
        stub = f'''

def process(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Process a tool operation.

    Args:
        operation: Operation to perform
        **kwargs: Operation parameters

    Returns:
        Operation result dictionary
    """
    # TODO: Implement {brain_name} tool operations
    print(f"[{brain_name.upper()}] Tool operation: {{operation}}")

    return {{
        "status": "not_implemented",
        "operation": operation,
        "message": "{brain_name} tool process() is a stub"
    }}
'''
    else:  # utility
        stub = f'''

def process(*args, **kwargs):
    """
    Process utility operation.

    Note: This is a stub. This module may not need a process() method.
    """
    pass
'''

    return stub


def scan_brain_compliance() -> Dict[str, Any]:
    """
    Scan all brain modules and check for process() method compliance.

    Returns:
        Dictionary with compliance report
    """
    maven_root = get_maven_root()
    brains_dir = maven_root / "brains"

    compliant = []
    non_compliant = []
    brain_files = []

    # Find all potential brain files
    for file_path in brains_dir.rglob("*.py"):
        if is_brain_module(file_path):
            brain_files.append(file_path)

            if has_process_method(file_path):
                compliant.append(str(file_path.relative_to(maven_root)))
            else:
                non_compliant.append(str(file_path.relative_to(maven_root)))

    return {
        "total_brains": len(brain_files),
        "compliant": compliant,
        "non_compliant": non_compliant,
        "compliance_rate": len(compliant) / len(brain_files) if brain_files else 0,
        "scan_time": datetime.utcnow().isoformat()
    }


def add_process_method(file_path: Path, create_backup: bool = True) -> Dict[str, Any]:
    """
    Add a process() method stub to a brain file.

    Args:
        file_path: Path to the brain file
        create_backup: Whether to create a backup before modifying

    Returns:
        Result dictionary with success status
    """
    result = {
        "file": str(file_path),
        "success": False,
        "backup_created": False,
        "error": None
    }

    try:
        # Read current content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Create backup if requested
        if create_backup:
            backup_path = file_path.with_suffix(f".py.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            result["backup_created"] = True
            result["backup_path"] = str(backup_path)

        # Determine brain info
        brain_name = file_path.stem
        brain_type = get_brain_type(file_path)

        # Generate stub
        stub = generate_process_stub(file_path, brain_name, brain_type)

        # Add stub at the end of the file
        new_content = content.rstrip() + "\n" + stub + "\n"

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        result["success"] = True
        result["brain_name"] = brain_name
        result["brain_type"] = brain_type

    except Exception as e:
        result["error"] = str(e)

    return result


def fix_brain_compliance(dry_run: bool = True, create_backup: bool = True,
                        max_fixes: int = 100) -> Dict[str, Any]:
    """
    Fix brain compliance issues by adding process() stubs.

    Args:
        dry_run: If True, only report what would be fixed (don't modify files)
        create_backup: Create backups before modifying files
        max_fixes: Maximum number of files to fix in one run

    Returns:
        Dictionary with fix results
    """
    maven_root = get_maven_root()

    # First, scan for compliance
    compliance_report = scan_brain_compliance()

    result = {
        "dry_run": dry_run,
        "total_brains": compliance_report["total_brains"],
        "compliant": len(compliance_report["compliant"]),
        "non_compliant": len(compliance_report["non_compliant"]),
        "fixes_attempted": 0,
        "fixes_successful": 0,
        "fixes_failed": 0,
        "fixed_files": [],
        "failed_files": [],
        "skipped": 0
    }

    if dry_run:
        result["message"] = "Dry run - no files modified"
        result["would_fix"] = compliance_report["non_compliant"][:max_fixes]
        return result

    # Fix non-compliant brains
    for file_rel_path in compliance_report["non_compliant"][:max_fixes]:
        file_path = maven_root / file_rel_path
        result["fixes_attempted"] += 1

        fix_result = add_process_method(file_path, create_backup)

        if fix_result["success"]:
            result["fixes_successful"] += 1
            result["fixed_files"].append({
                "file": file_rel_path,
                "brain_name": fix_result.get("brain_name"),
                "brain_type": fix_result.get("brain_type"),
                "backup": fix_result.get("backup_path")
            })
        else:
            result["fixes_failed"] += 1
            result["failed_files"].append({
                "file": file_rel_path,
                "error": fix_result.get("error")
            })

    # Count how many we skipped
    result["skipped"] = len(compliance_report["non_compliant"]) - max_fixes
    if result["skipped"] < 0:
        result["skipped"] = 0

    return result


def get_brain_compliance_fixer():
    """
    Factory function to get a brain compliance fixer interface.

    This provides a simple interface for other modules to use.
    """
    return {
        "scan": scan_brain_compliance,
        "fix": fix_brain_compliance,
        "add_method": add_process_method
    }


# CLI interface
if __name__ == "__main__":
    import json
    import sys

    # Parse command line args
    dry_run = "--fix" not in sys.argv
    create_backup = "--no-backup" not in sys.argv
    max_fixes = 10  # Default to 10 files at a time

    if "--all" in sys.argv:
        max_fixes = 1000

    print("[BRAIN_COMPLIANCE_FIXER] Starting compliance check...")

    result = fix_brain_compliance(dry_run=dry_run, create_backup=create_backup, max_fixes=max_fixes)

    print(json.dumps(result, indent=2))

    if dry_run:
        print(f"\nDry run complete. Would fix {len(result.get('would_fix', []))} brains.")
        print("Run with --fix to actually modify files.")
    else:
        print(f"\nFixed {result['fixes_successful']}/{result['fixes_attempted']} brains.")
        if result['fixes_failed'] > 0:
            print(f"Failed to fix {result['fixes_failed']} brains.")
