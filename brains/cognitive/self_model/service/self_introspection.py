"""
Self‑Introspection Module
=========================

This module provides dynamic codebase introspection for Maven's self-awareness.
It enables Maven to:
1. Scan its own Python files to understand brain implementations
2. Analyze brain contract compliance by comparing actual vs expected operations
3. Identify gaps and missing functionality
4. Generate upgrade plans based on ACTUAL state, not hardcoded assumptions

This is TRUE self-awareness - reading actual code from disk and analyzing it
dynamically, not returning hardcoded responses.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from brains.maven_paths import get_maven_root


class BrainIntrospector:
    """
    Analyzes Maven's cognitive brains by scanning actual Python source code.

    Provides methods to:
    - Extract implemented operations from brain files
    - Compare against expected contracts
    - Identify compliance gaps
    - Generate structured improvement recommendations
    """

    def __init__(self):
        self.maven_root = get_maven_root()
        self.brains_dir = self.maven_root / "brains" / "cognitive"
        self.specs_dir = self.maven_root / "brains" / "domain_banks" / "specs"

        # Load brain contracts and inventory
        self.contracts = self._load_brain_contracts()
        self.inventory = self._load_brain_inventory()
        self.cognitive_brain_contract = self._load_cognitive_brain_contract()

    def _load_brain_contracts(self) -> Dict[str, Any]:
        """Load expected brain contracts from specs."""
        contracts_path = self.specs_dir / "brain_contracts_phase6.json"
        try:
            if contracts_path.exists():
                with open(contracts_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[INTROSPECTION] Could not load brain contracts: {e}")
        return {}

    def _load_brain_inventory(self) -> Dict[str, Any]:
        """Load brain inventory from specs."""
        inventory_path = self.specs_dir / "brain_inventory_phase6.json"
        try:
            if inventory_path.exists():
                with open(inventory_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[INTROSPECTION] Could not load brain inventory: {e}")
        return {}

    def _load_cognitive_brain_contract(self) -> Dict[str, Any]:
        """Load cognitive brain contract requirements (3 signals)."""
        contract_path = self.maven_root / "COGNITIVE_BRAIN_CONTRACT.md"

        # Define the 3 required signals
        return {
            "required_signals": [
                {
                    "name": "follow_up_detection",
                    "description": "Detect and classify continuation vs. fresh queries",
                    "indicators": [
                        "is_continuation(",
                        "follow_up_question",
                        "_continuation_helpers_available"
                    ]
                },
                {
                    "name": "history_access",
                    "description": "Pull and use conversation history",
                    "indicators": [
                        "get_conversation_context(",
                        "last_topic",
                        "system_history"
                    ]
                },
                {
                    "name": "routing_learning",
                    "description": "Output routing hints for the Integrator",
                    "indicators": [
                        "create_routing_hint(",
                        "routing_hint",
                        "context_tags"
                    ]
                }
            ]
        }

    def scan_brain_implementation(self, brain_name: str) -> Dict[str, Any]:
        """
        Scan a specific brain's Python file to extract implemented operations.

        Args:
            brain_name: Name of the brain (e.g., "language_brain")

        Returns:
            Dictionary with:
            - operations: List of implemented operations
            - signals: Which cognitive brain contract signals are implemented
            - file_path: Path to the brain file
            - line_count: Number of lines in the file
        """
        result = {
            "brain_name": brain_name,
            "operations": [],
            "signals": {
                "follow_up_detection": False,
                "history_access": False,
                "routing_learning": False
            },
            "file_path": None,
            "line_count": 0,
            "exists": False
        }

        # Find brain file
        brain_dir = self.brains_dir / brain_name.replace("_brain", "")
        brain_file = brain_dir / "service" / f"{brain_name}.py"

        if not brain_file.exists():
            return result

        result["exists"] = True
        result["file_path"] = str(brain_file)

        try:
            with open(brain_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            result["line_count"] = len(source_code.splitlines())

            # Extract operations using AST
            operations = self._extract_operations_from_ast(source_code)
            result["operations"] = operations

            # Check for cognitive brain contract signals
            signals = self._detect_cognitive_signals(source_code)
            result["signals"] = signals

        except Exception as e:
            print(f"[INTROSPECTION] Error scanning {brain_name}: {e}")

        return result

    def _extract_operations_from_ast(self, source_code: str) -> List[str]:
        """
        Parse Python source code and extract operation names from service_api function.

        Looks for patterns like:
        - if op == "OPERATION_NAME":
        - elif op == "OPERATION_NAME":
        """
        operations = []

        try:
            tree = ast.parse(source_code)

            # Look for service_api function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "service_api":
                    # Walk through the function body
                    for item in ast.walk(node):
                        if isinstance(item, ast.Compare):
                            # Check if this is comparing 'op' with a string
                            if isinstance(item.left, ast.Name) and item.left.id == "op":
                                for comparator in item.comparators:
                                    if isinstance(comparator, ast.Constant):
                                        operations.append(str(comparator.value))

        except Exception as e:
            print(f"[INTROSPECTION] AST parsing error: {e}")

        # Fallback: regex search for operation patterns
        if not operations:
            operations = self._extract_operations_regex(source_code)

        return sorted(set(operations))

    def _extract_operations_regex(self, source_code: str) -> List[str]:
        """Fallback: use regex to find operation names."""
        operations = []

        # Pattern: if op == "OPERATION" or elif op == "OPERATION"
        pattern = r'(?:if|elif)\s+op\s*==\s*["\']([A-Z_]+)["\']'
        matches = re.findall(pattern, source_code)
        operations.extend(matches)

        return sorted(set(operations))

    def _detect_cognitive_signals(self, source_code: str) -> Dict[str, bool]:
        """
        Detect which cognitive brain contract signals are implemented.

        Checks for presence of key functions and patterns that indicate
        the brain implements the required signals.
        """
        signals = {
            "follow_up_detection": False,
            "history_access": False,
            "routing_learning": False
        }

        # Check each signal
        for signal_spec in self.cognitive_brain_contract["required_signals"]:
            signal_name = signal_spec["name"]
            indicators = signal_spec["indicators"]

            # If ANY indicator is found, signal is implemented
            for indicator in indicators:
                if indicator in source_code:
                    signals[signal_name] = True
                    break

        return signals

    def analyze_contract_compliance(self, brain_name: str) -> Dict[str, Any]:
        """
        Compare brain's actual implementation against expected contract.

        Args:
            brain_name: Name of the brain to analyze

        Returns:
            Dictionary with:
            - expected_operations: Operations the brain SHOULD implement
            - implemented_operations: Operations the brain DOES implement
            - missing_operations: Gap between expected and actual
            - extra_operations: Operations not in contract but implemented
            - compliance_score: Percentage of expected operations implemented
            - signal_compliance: Which cognitive signals are implemented
        """
        # Scan actual implementation
        actual = self.scan_brain_implementation(brain_name)

        # Get expected operations from contract
        expected_ops = []
        if brain_name in self.contracts:
            expected_ops = list(self.contracts[brain_name].keys())

        implemented_ops = actual["operations"]

        # Calculate gaps
        missing_ops = [op for op in expected_ops if op not in implemented_ops]
        extra_ops = [op for op in implemented_ops if op not in expected_ops]

        # Compliance score
        if expected_ops:
            compliance_score = (len(implemented_ops) - len(extra_ops)) / len(expected_ops) * 100
            compliance_score = max(0, min(100, compliance_score))  # Clamp to 0-100
        else:
            compliance_score = 100 if not implemented_ops else 0

        # Signal compliance (cognitive brain contract)
        signal_compliance = actual["signals"]
        signals_implemented = sum(signal_compliance.values())
        signals_total = len(signal_compliance)
        signal_compliance_score = (signals_implemented / signals_total) * 100 if signals_total > 0 else 0

        return {
            "brain_name": brain_name,
            "exists": actual["exists"],
            "file_path": actual["file_path"],
            "line_count": actual["line_count"],
            "expected_operations": expected_ops,
            "implemented_operations": implemented_ops,
            "missing_operations": missing_ops,
            "extra_operations": extra_ops,
            "compliance_score": round(compliance_score, 1),
            "signal_compliance": signal_compliance,
            "signals_implemented": signals_implemented,
            "signals_total": signals_total,
            "signal_compliance_score": round(signal_compliance_score, 1)
        }

    def scan_all_brains(self) -> Dict[str, Any]:
        """
        Scan all cognitive brains and analyze their compliance.

        Returns:
            Dictionary with:
            - total_brains: Total number of brains discovered
            - compliant_brains: Brains with 100% operation compliance
            - partial_brains: Brains with some compliance
            - non_compliant_brains: Brains with 0% compliance
            - signal_compliant_brains: Brains with all 3 signals
            - details: Per-brain compliance analysis
        """
        results = {
            "total_brains": 0,
            "compliant_brains": 0,
            "partial_brains": 0,
            "non_compliant_brains": 0,
            "signal_compliant_brains": 0,
            "signal_partial_brains": 0,
            "signal_non_compliant_brains": 0,
            "details": []
        }

        # Get list of all cognitive brains
        brain_names = []

        # From inventory
        if self.inventory:
            brain_names = list(self.inventory.keys())

        # Fallback: scan directory
        if not brain_names and self.brains_dir.exists():
            for item in self.brains_dir.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    brain_name = f"{item.name}_brain"
                    brain_names.append(brain_name)

        results["total_brains"] = len(brain_names)

        # Analyze each brain
        for brain_name in sorted(brain_names):
            analysis = self.analyze_contract_compliance(brain_name)
            results["details"].append(analysis)

            # Count compliance categories
            if analysis["exists"]:
                if analysis["compliance_score"] >= 100:
                    results["compliant_brains"] += 1
                elif analysis["compliance_score"] > 0:
                    results["partial_brains"] += 1
                else:
                    results["non_compliant_brains"] += 1

                # Signal compliance
                if analysis["signal_compliance_score"] >= 100:
                    results["signal_compliant_brains"] += 1
                elif analysis["signal_compliance_score"] > 0:
                    results["signal_partial_brains"] += 1
                else:
                    results["signal_non_compliant_brains"] += 1

        return results

    def identify_upgrade_priorities(self, scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze scan results and identify high-priority upgrade areas.

        Returns:
            List of priority areas with:
            - area: Description of the improvement area
            - priority: "critical", "high", "medium", "low"
            - affected_brains: List of brain names
            - tasks: Specific implementation tasks
            - impact: Expected improvement from fixing
        """
        priorities = []

        # Priority 1: Critical brains with low signal compliance
        critical_brains = ["language_brain", "reasoning_brain", "self_model_brain",
                          "memory_librarian", "integrator_brain"]

        low_signal_brains = [
            detail for detail in scan_results["details"]
            if detail["brain_name"] in critical_brains and detail["signal_compliance_score"] < 100
        ]

        if low_signal_brains:
            priorities.append({
                "area": "Complete Cognitive Brain Contract for Critical Brains",
                "priority": "critical",
                "affected_brains": [b["brain_name"] for b in low_signal_brains],
                "tasks": self._generate_signal_tasks(low_signal_brains),
                "impact": "Enable automatic learning of multi-turn conversation behavior",
                "phase": 1
            })

        # Priority 2: All brains missing signal compliance
        signal_incomplete = [
            detail for detail in scan_results["details"]
            if detail["signal_compliance_score"] < 100 and detail["exists"]
        ]

        if len(signal_incomplete) > len(low_signal_brains):
            remaining_brains = [
                b for b in signal_incomplete
                if b["brain_name"] not in critical_brains
            ]

            priorities.append({
                "area": "Complete Cognitive Brain Contract for All Brains",
                "priority": "high",
                "affected_brains": [b["brain_name"] for b in remaining_brains],
                "tasks": self._generate_signal_tasks(remaining_brains),
                "impact": "Full multi-turn conversation learning across all cognitive systems",
                "phase": 2
            })

        # Priority 3: Brains with missing contract operations
        missing_ops_brains = [
            detail for detail in scan_results["details"]
            if detail["missing_operations"] and detail["exists"]
        ]

        if missing_ops_brains:
            priorities.append({
                "area": "Implement Missing Contract Operations",
                "priority": "high",
                "affected_brains": [b["brain_name"] for b in missing_ops_brains],
                "tasks": self._generate_operation_tasks(missing_ops_brains),
                "impact": "Complete service contracts and enable full brain interoperability",
                "phase": 3
            })

        # Priority 4: Non-existent brains
        non_existent = [
            detail for detail in scan_results["details"]
            if not detail["exists"]
        ]

        if non_existent:
            priorities.append({
                "area": "Implement Missing Brains",
                "priority": "medium",
                "affected_brains": [b["brain_name"] for b in non_existent],
                "tasks": [f"Create {b['brain_name']} implementation" for b in non_existent],
                "impact": "Complete cognitive architecture",
                "phase": 4
            })

        return priorities

    def _generate_signal_tasks(self, brains: List[Dict[str, Any]]) -> List[str]:
        """Generate specific tasks for implementing cognitive signals."""
        tasks = []

        for brain in brains:
            brain_name = brain["brain_name"]
            signals = brain["signal_compliance"]

            if not signals.get("follow_up_detection"):
                tasks.append(f"Implement follow-up detection in {brain_name}")

            if not signals.get("history_access"):
                tasks.append(f"Implement history access in {brain_name}")

            if not signals.get("routing_learning"):
                tasks.append(f"Implement routing hints in {brain_name}")

        return tasks

    def _generate_operation_tasks(self, brains: List[Dict[str, Any]]) -> List[str]:
        """Generate specific tasks for implementing missing operations."""
        tasks = []

        for brain in brains:
            brain_name = brain["brain_name"]
            missing = brain["missing_operations"]

            if len(missing) <= 3:
                for op in missing:
                    tasks.append(f"Implement {op} operation in {brain_name}")
            else:
                tasks.append(f"Implement {len(missing)} missing operations in {brain_name}: {', '.join(missing[:3])}, ...")

        return tasks


def scan_self() -> Dict[str, Any]:
    """
    Entry point for dynamic self-scanning.

    Returns comprehensive analysis of Maven's current state by:
    1. Scanning all cognitive brains
    2. Analyzing contract compliance
    3. Identifying improvement priorities
    """
    introspector = BrainIntrospector()

    print("[INTROSPECTION] Scanning all cognitive brains...")
    scan_results = introspector.scan_all_brains()

    print(f"[INTROSPECTION] Found {scan_results['total_brains']} brains")
    print(f"[INTROSPECTION] Operation compliance: {scan_results['compliant_brains']} compliant, "
          f"{scan_results['partial_brains']} partial, {scan_results['non_compliant_brains']} non-compliant")
    print(f"[INTROSPECTION] Signal compliance: {scan_results['signal_compliant_brains']} compliant, "
          f"{scan_results['signal_partial_brains']} partial, {scan_results['signal_non_compliant_brains']} non-compliant")

    print("[INTROSPECTION] Identifying upgrade priorities...")
    priorities = introspector.identify_upgrade_priorities(scan_results)

    print(f"[INTROSPECTION] Identified {len(priorities)} priority areas")

    return {
        "scan_results": scan_results,
        "priorities": priorities,
        "introspector": introspector  # Return introspector for further analysis
    }
