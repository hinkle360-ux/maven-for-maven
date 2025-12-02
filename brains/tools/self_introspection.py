"""
True Self-Introspection System for Maven
========================================

Provides Maven with deep introspection capabilities:
- Scan and analyze its own directory structure
- Map all brain modules and their relationships
- Detect missing or broken functions
- Analyze code structure and dependencies
- Identify architectural inconsistencies
- Generate self-knowledge reports
- Detect outdated or inconsistent brain implementations

This replaces the canned responses with real introspection.
"""

from __future__ import annotations

import ast
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json

from brains.maven_paths import MAVEN_ROOT, get_maven_root


class SelfIntrospection:
    """Maven's self-introspection and self-knowledge system."""

    def __init__(self):
        self.root = get_maven_root()
        self.cache: Dict[str, Any] = {}
        self.last_scan: Optional[datetime] = None

    # ==================== DIRECTORY STRUCTURE ANALYSIS ====================

    def scan_full_architecture(self, use_cache: bool = False) -> Dict[str, Any]:
        """
        Perform a complete scan of Maven's architecture.

        Returns comprehensive structure including:
        - All directories and their purposes
        - All Python files categorized by type
        - Brain modules and their structure
        - API modules
        - Configuration files
        - Memory directories
        """
        if use_cache and "full_architecture" in self.cache:
            return self.cache["full_architecture"]

        print("[SELF_INTROSPECTION] Scanning full Maven architecture...")

        architecture = {
            "timestamp": datetime.utcnow().isoformat(),
            "root": str(self.root),
            "directories": {},
            "python_files": {},
            "brains": {},
            "api_modules": {},
            "config_files": [],
            "total_files": 0,
            "total_lines": 0
        }

        # Scan directory structure
        for path in self.root.rglob("*"):
            if path.name.startswith('.') or '__pycache__' in path.parts:
                continue

            rel_path = str(path.relative_to(self.root))

            if path.is_dir():
                architecture["directories"][rel_path] = {
                    "path": rel_path,
                    "type": self._classify_directory(path),
                    "file_count": len(list(path.glob("*.py")))
                }
            elif path.suffix == '.py':
                architecture["total_files"] += 1
                file_info = self._analyze_python_file(path)
                architecture["python_files"][rel_path] = file_info
                architecture["total_lines"] += file_info.get("lines", 0)

                # Categorize brains
                if "brains/cognitive" in rel_path or "brains/domain_banks" in rel_path:
                    brain_name = self._extract_brain_name(rel_path)
                    if brain_name:
                        if brain_name not in architecture["brains"]:
                            architecture["brains"][brain_name] = []
                        architecture["brains"][brain_name].append(file_info)

                # Categorize API modules
                if rel_path.startswith("api/"):
                    architecture["api_modules"][rel_path] = file_info

            elif path.suffix == '.json':
                if "config" in path.parts:
                    architecture["config_files"].append(rel_path)

        self.cache["full_architecture"] = architecture
        self.last_scan = datetime.utcnow()

        print(f"[SELF_INTROSPECTION] Scanned {architecture['total_files']} files, {architecture['total_lines']} lines")
        return architecture

    def _classify_directory(self, path: Path) -> str:
        """Classify a directory by its purpose."""
        name = path.name
        parts = path.parts

        if "cognitive" in parts:
            return "cognitive_brain"
        elif "domain_banks" in parts:
            return "domain_bank"
        elif name == "memory":
            return "memory_storage"
        elif name == "service":
            return "brain_service"
        elif name == "config":
            return "configuration"
        elif name == "tools":
            return "utility_tools"
        elif name == "api":
            return "api_layer"
        elif name == "ui":
            return "user_interface"
        elif name == "tests":
            return "test_suite"
        elif name == "reports":
            return "runtime_reports"
        elif name == "docs":
            return "documentation"
        else:
            return "other"

    def _extract_brain_name(self, file_path: str) -> Optional[str]:
        """Extract brain name from file path."""
        parts = file_path.split('/')

        if "cognitive" in parts:
            idx = parts.index("cognitive")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        elif "domain_banks" in parts:
            idx = parts.index("domain_banks")
            if idx + 1 < len(parts):
                return parts[idx + 1]

        return None

    # ==================== PYTHON FILE ANALYSIS ====================

    def _analyze_python_file(self, path: Path) -> Dict[str, Any]:
        """Analyze a Python file and extract structure."""
        info = {
            "path": str(path.relative_to(self.root)),
            "size": path.stat().st_size,
            "lines": 0,
            "classes": [],
            "functions": [],
            "imports": [],
            "docstring": None,
            "has_process_method": False,
            "has_brain_contract": False
        }

        try:
            content = path.read_text(encoding='utf-8')
            info["lines"] = len(content.splitlines())

            tree = ast.parse(content, filename=str(path))
            info["docstring"] = ast.get_docstring(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [],
                        "docstring": ast.get_docstring(node)
                    }

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info["methods"].append(item.name)
                            if item.name == "process":
                                info["has_process_method"] = True

                    info["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    # Top-level functions only
                    if node.col_offset == 0:
                        func_info = {
                            "name": node.name,
                            "line": node.lineno,
                            "args": [arg.arg for arg in node.args.args],
                            "docstring": ast.get_docstring(node)
                        }
                        info["functions"].append(func_info)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            info["imports"].append(alias.name)
                    else:
                        if node.module:
                            info["imports"].append(node.module)

            # Check for brain contract compliance
            if any("process" in str(c["methods"]) for c in info["classes"]):
                info["has_brain_contract"] = True

        except Exception as e:
            info["error"] = str(e)

        return info

    # ==================== BRAIN ANALYSIS ====================

    def analyze_all_brains(self) -> Dict[str, Any]:
        """Analyze all brain modules and their compliance."""
        print("[SELF_INTROSPECTION] Analyzing all brains...")

        architecture = self.scan_full_architecture(use_cache=True)
        brain_analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_brains": len(architecture["brains"]),
            "brains": {},
            "compliant": [],
            "non_compliant": [],
            "missing_methods": {}
        }

        for brain_name, files in architecture["brains"].items():
            analysis = {
                "name": brain_name,
                "files": len(files),
                "total_lines": sum(f.get("lines", 0) for f in files),
                "has_service": False,
                "has_memory": False,
                "has_process": False,
                "has_brain_contract": False,
                "missing_methods": [],
                "structure": files
            }

            # Check for required components
            for file in files:
                if "service" in file["path"]:
                    analysis["has_service"] = True
                if "memory" in file["path"]:
                    analysis["has_memory"] = True
                if file.get("has_process_method"):
                    analysis["has_process"] = True
                if file.get("has_brain_contract"):
                    analysis["has_brain_contract"] = True

            # Check for missing required methods
            required_methods = ["process"]
            found_methods = set()

            for file in files:
                for cls in file.get("classes", []):
                    found_methods.update(cls.get("methods", []))

            analysis["missing_methods"] = [m for m in required_methods if m not in found_methods]

            # Determine compliance
            is_compliant = (
                analysis["has_service"] and
                analysis["has_process"] and
                len(analysis["missing_methods"]) == 0
            )

            if is_compliant:
                brain_analysis["compliant"].append(brain_name)
            else:
                brain_analysis["non_compliant"].append(brain_name)
                brain_analysis["missing_methods"][brain_name] = analysis["missing_methods"]

            brain_analysis["brains"][brain_name] = analysis

        print(f"[SELF_INTROSPECTION] Analyzed {brain_analysis['total_brains']} brains")
        print(f"  Compliant: {len(brain_analysis['compliant'])}")
        print(f"  Non-compliant: {len(brain_analysis['non_compliant'])}")

        return brain_analysis

    def get_brain_structure(self, brain_name: str) -> Dict[str, Any]:
        """Get detailed structure of a specific brain."""
        architecture = self.scan_full_architecture(use_cache=True)

        if brain_name not in architecture["brains"]:
            return {"error": f"Brain not found: {brain_name}"}

        brain_files = architecture["brains"][brain_name]

        structure = {
            "name": brain_name,
            "files": brain_files,
            "total_lines": sum(f.get("lines", 0) for f in brain_files),
            "classes": [],
            "functions": [],
            "dependencies": set()
        }

        for file in brain_files:
            structure["classes"].extend(file.get("classes", []))
            structure["functions"].extend(file.get("functions", []))
            structure["dependencies"].update(file.get("imports", []))

        structure["dependencies"] = sorted(list(structure["dependencies"]))

        return structure

    # ==================== DEPENDENCY ANALYSIS ====================

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies across all Maven modules."""
        print("[SELF_INTROSPECTION] Analyzing dependencies...")

        architecture = self.scan_full_architecture(use_cache=True)

        dependency_map = {
            "timestamp": datetime.utcnow().isoformat(),
            "modules": {},
            "circular_dependencies": [],
            "missing_imports": [],
            "external_dependencies": set()
        }

        for file_path, file_info in architecture["python_files"].items():
            module_name = file_path.replace('/', '.').replace('.py', '')
            deps = file_info.get("imports", [])

            dependency_map["modules"][module_name] = {
                "path": file_path,
                "imports": deps,
                "internal": [d for d in deps if d.startswith('brains.') or d.startswith('api.')],
                "external": [d for d in deps if not d.startswith('brains.') and not d.startswith('api.')]
            }

            dependency_map["external_dependencies"].update(dependency_map["modules"][module_name]["external"])

        dependency_map["external_dependencies"] = sorted(list(dependency_map["external_dependencies"]))

        # Detect circular dependencies
        dependency_map["circular_dependencies"] = self._find_circular_dependencies(dependency_map["modules"])

        print(f"[SELF_INTROSPECTION] Analyzed {len(dependency_map['modules'])} modules")
        print(f"  External dependencies: {len(dependency_map['external_dependencies'])}")
        print(f"  Circular dependencies: {len(dependency_map['circular_dependencies'])}")

        return dependency_map

    def _find_circular_dependencies(self, modules: Dict[str, Dict]) -> List[List[str]]:
        """Find circular dependencies in the module graph."""
        circular = []

        def visit(module: str, path: List[str], visited: Set[str]):
            if module in path:
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                if cycle not in circular and list(reversed(cycle)) not in circular:
                    circular.append(cycle)
                return

            if module in visited or module not in modules:
                return

            visited.add(module)
            path.append(module)

            for dep in modules[module]["internal"]:
                visit(dep, path.copy(), visited)

        for module in modules:
            visit(module, [], set())

        return circular

    # ==================== FUNCTION DETECTION ====================

    def detect_missing_functions(self, required_functions: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Detect missing or broken functions across Maven.

        Args:
            required_functions: Dict mapping brain names to lists of required function names

        Returns:
            Report of missing functions
        """
        if required_functions is None:
            required_functions = {
                "all_brains": ["process"],
                "reasoning": ["process", "reflect", "self_review"],
                "teacher": ["process", "store_fact", "critique"],
                "integrator": ["process", "integrate"],
                "memory_librarian": ["process", "route_query"]
            }

        architecture = self.scan_full_architecture(use_cache=True)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "missing_functions": {},
            "broken_imports": []
        }

        for brain_name, required in required_functions.items():
            if brain_name == "all_brains":
                # Check all brains for these functions
                for b_name in architecture["brains"]:
                    missing = self._check_brain_functions(b_name, required, architecture)
                    if missing:
                        report["missing_functions"][b_name] = missing
            else:
                missing = self._check_brain_functions(brain_name, required, architecture)
                if missing:
                    report["missing_functions"][brain_name] = missing

        return report

    def _check_brain_functions(self, brain_name: str, required_functions: List[str], architecture: Dict) -> List[str]:
        """Check if a brain has all required functions."""
        if brain_name not in architecture["brains"]:
            return required_functions  # All functions missing if brain doesn't exist

        found_functions = set()

        for file in architecture["brains"][brain_name]:
            for cls in file.get("classes", []):
                found_functions.update(cls.get("methods", []))
            for func in file.get("functions", []):
                found_functions.add(func["name"])

        missing = [f for f in required_functions if f not in found_functions]
        return missing

    # ==================== ROUTING TABLE INTROSPECTION ====================

    def scan_routing_table(self) -> Dict[str, Any]:
        """Scan and analyze the routing table from memory."""
        print("[SELF_INTROSPECTION] Scanning routing table...")

        routing_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "patterns": [],
            "brain_counts": {},
            "total_patterns": 0
        }

        # Try to load routing patterns from memory
        try:
            from brains.cognitive.memory_librarian.service.memory_librarian import get_routing_patterns

            patterns = get_routing_patterns()

            for pattern_data in patterns:
                routing_info["patterns"].append({
                    "pattern": pattern_data.get("pattern"),
                    "brain": pattern_data.get("brain"),
                    "confidence": pattern_data.get("confidence", 0.0),
                    "uses": pattern_data.get("uses", 0)
                })

                brain = pattern_data.get("brain", "unknown")
                routing_info["brain_counts"][brain] = routing_info["brain_counts"].get(brain, 0) + 1

            routing_info["total_patterns"] = len(patterns)

        except Exception as e:
            routing_info["error"] = f"Failed to load routing patterns: {str(e)}"

        return routing_info

    # ==================== MEMORY INTROSPECTION ====================

    def scan_memory_system(self) -> Dict[str, Any]:
        """Scan Maven's memory system structure."""
        print("[SELF_INTROSPECTION] Scanning memory system...")

        memory_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "brain_memories": {},
            "tier_structure": {},
            "total_size": 0
        }

        # Scan brain memory directories
        brains_dir = self.root / "brains"

        for brain_path in brains_dir.rglob("memory"):
            if brain_path.is_dir():
                brain_name = brain_path.parent.name

                brain_memory = {
                    "path": str(brain_path.relative_to(self.root)),
                    "tiers": {},
                    "total_files": 0,
                    "total_size": 0
                }

                # Check for tier directories
                for tier in ["STM", "MTM", "LTM", "Archive"]:
                    tier_path = brain_path / tier
                    if tier_path.exists():
                        files = list(tier_path.glob("*.json"))
                        size = sum(f.stat().st_size for f in files)

                        brain_memory["tiers"][tier] = {
                            "files": len(files),
                            "size": size
                        }
                        brain_memory["total_files"] += len(files)
                        brain_memory["total_size"] += size

                memory_info["brain_memories"][brain_name] = brain_memory
                memory_info["total_size"] += brain_memory["total_size"]

        print(f"[SELF_INTROSPECTION] Found memory for {len(memory_info['brain_memories'])} brains")

        return memory_info

    # ==================== SELF-KNOWLEDGE REPORT ====================

    def generate_self_knowledge_report(self) -> Dict[str, Any]:
        """Generate a comprehensive self-knowledge report."""
        print("[SELF_INTROSPECTION] Generating self-knowledge report...")

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "architecture": self.scan_full_architecture(),
            "brains": self.analyze_all_brains(),
            "dependencies": self.analyze_dependencies(),
            "missing_functions": self.detect_missing_functions(),
            "routing": self.scan_routing_table(),
            "memory": self.scan_memory_system()
        }

        # Add summary
        report["summary"] = {
            "total_files": report["architecture"]["total_files"],
            "total_lines": report["architecture"]["total_lines"],
            "total_brains": report["brains"]["total_brains"],
            "compliant_brains": len(report["brains"]["compliant"]),
            "non_compliant_brains": len(report["brains"]["non_compliant"]),
            "external_dependencies": len(report["dependencies"]["external_dependencies"]),
            "circular_dependencies": len(report["dependencies"]["circular_dependencies"]),
            "total_memory_size": report["memory"]["total_size"]
        }

        print("[SELF_INTROSPECTION] Self-knowledge report complete")
        return report

    def save_report(self, report: Dict[str, Any], filename: str = "self_knowledge_report.json") -> str:
        """Save a report to disk."""
        report_path = self.root / "reports" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"[SELF_INTROSPECTION] Report saved to {report_path}")
        return str(report_path)


# Global instance
_introspection = None


def get_self_introspection() -> SelfIntrospection:
    """Get the global self-introspection instance."""
    global _introspection
    if _introspection is None:
        _introspection = SelfIntrospection()
    return _introspection


# Convenience functions
def scan_architecture() -> Dict[str, Any]:
    """Scan Maven's full architecture."""
    return get_self_introspection().scan_full_architecture()


def analyze_brains() -> Dict[str, Any]:
    """Analyze all brain modules."""
    return get_self_introspection().analyze_all_brains()


def generate_report() -> Dict[str, Any]:
    """Generate a complete self-knowledge report."""
    return get_self_introspection().generate_self_knowledge_report()
