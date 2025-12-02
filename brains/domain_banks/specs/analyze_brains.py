#!/usr/bin/env python3
"""
Brain Inventory Analysis Script for Phase 6

This script scans all brain modules and classifies them based on:
- role: core/specialist/diagnostic
- status: implemented/partial/stub
- ops: list of supported operations

Output: brain_inventory_phase6.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set

from brains.maven_paths import MAVEN_ROOT, get_brains_path

def extract_operations(file_path: Path) -> List[str]:
    """Extract all supported operations from a brain file."""
    operations = []
    try:
        content = file_path.read_text(encoding='utf-8')

        # Find operation checks in service_api
        op_patterns = [
            r'if\s+op\s*==\s*["\']([A-Z_]+)["\']',
            r'elif\s+op\s*==\s*["\']([A-Z_]+)["\']',
            r'if\s+op\s+in\s*\(([^)]+)\)',
        ]

        for pattern in op_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if len(match.groups()) == 1:
                    op_value = match.group(1)
                    # Handle both single ops and lists
                    if ',' in op_value:
                        # Parse list like ("OP1", "OP2")
                        ops_in_list = re.findall(r'["\']([A-Z_]+)["\']', op_value)
                        operations.extend(ops_in_list)
                    else:
                        operations.append(op_value)

        return sorted(list(set(operations)))
    except Exception as e:
        print(f"Error extracting operations from {file_path}: {e}")
        return []

def assess_implementation_status(file_path: Path, operations: List[str]) -> str:
    """Determine if brain is implemented, partial, or stub."""
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        # Count indicators
        todo_count = len([l for l in lines if 'TODO' in l or 'FIXME' in l])
        stub_patterns = ['pass', 'return {}', 'return None', 'not implemented']
        stub_count = sum(1 for l in lines if any(p in l.lower() for p in stub_patterns))

        # Check for real logic
        has_imports = 'import' in content
        has_classes = 'class ' in content or 'def ' in content
        has_logic = len(lines) > 50  # Arbitrary but reasonable

        total_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])

        if total_lines < 30:
            return "stub"
        elif todo_count > 5 or stub_count > 10:
            return "partial"
        elif has_imports and has_classes and has_logic:
            return "implemented"
        else:
            return "partial"
    except Exception as e:
        print(f"Error assessing {file_path}: {e}")
        return "unknown"

def classify_brain_role(brain_name: str, operations: List[str]) -> str:
    """Classify brain as core, specialist, or diagnostic."""

    # Core brains: essential for normal query processing
    core_brains = {
        'language_brain', 'memory_librarian', 'reasoning_brain',
        'planner_brain', 'thought_synthesizer', 'self_model_brain',
        'self_review_brain', 'self_dmn_brain', 'motivation_brain',
        'integrator_brain', 'autonomy_brain', 'pattern_recognition_brain',
        'personal_brain'
    }

    # Diagnostic brains: for health checks and introspection
    diagnostic_ops = {'HEALTH', 'STATUS', 'DIAGNOSE', 'REPORT'}

    if brain_name in core_brains:
        return "core"
    elif any(op in diagnostic_ops for op in operations):
        return "diagnostic"
    else:
        return "specialist"

def analyze_brain_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a single brain file and return its metadata."""
    brain_name = file_path.stem

    # Extract operations
    operations = extract_operations(file_path)

    # Assess implementation status
    status = assess_implementation_status(file_path, operations)

    # Classify role
    role = classify_brain_role(brain_name, operations)

    # Get relative module path
    module_path = str(file_path.relative_to(MAVEN_ROOT))

    return {
        "module": module_path,
        "role": role,
        "status": status,
        "ops": operations
    }

def main():
    """Main analysis function."""
    base_path = get_brains_path()

    # Find all brain files
    brain_files = []

    # Cognitive brains
    cognitive_path = base_path / "cognitive"
    if cognitive_path.exists():
        brain_files.extend(cognitive_path.glob("**/service/*_brain.py"))
        # Include memory_librarian and thought_synthesizer
        brain_files.extend(cognitive_path.glob("**/service/memory_librarian.py"))
        brain_files.extend(cognitive_path.glob("**/service/thought_synthesizer.py"))

    # Personal brains
    personal_path = base_path / "personal"
    if personal_path.exists():
        brain_files.extend(personal_path.glob("**/service/*_brain.py"))

    # Governance brains
    governance_path = base_path / "governance"
    if governance_path.exists():
        brain_files.extend(governance_path.glob("**/service/*_brain.py"))

    # Build inventory
    inventory = {}

    for brain_file in sorted(brain_files):
        brain_name = brain_file.stem
        print(f"Analyzing {brain_name}...")

        try:
            brain_info = analyze_brain_file(brain_file)
            inventory[brain_name] = brain_info
        except Exception as e:
            print(f"Error analyzing {brain_name}: {e}")

    # Save inventory
    output_path = Path(__file__).parent / "brain_inventory_phase6.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    print(f"\nInventory saved to {output_path}")
    print(f"Total brains analyzed: {len(inventory)}")

    # Summary statistics
    by_role = {}
    by_status = {}
    for name, info in inventory.items():
        role = info['role']
        status = info['status']
        by_role[role] = by_role.get(role, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    print("\nSummary by role:")
    for role, count in sorted(by_role.items()):
        print(f"  {role}: {count}")

    print("\nSummary by status:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

if __name__ == "__main__":
    main()
