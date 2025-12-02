"""
Hot-Reload System for Maven
============================

Provides Maven with the ability to:
- Reload Python modules at runtime using importlib
- Replace active modules without restarting
- Switch between different implementations
- Modify its own code and reload changes
- Track reloaded modules and their versions
- Validate reloads before applying them

This enables true self-modification capabilities.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import ast
import traceback

from brains.maven_paths import MAVEN_ROOT, get_maven_root, validate_path_confinement
from brains.tools.execution_guard import require_execution_enabled


class HotReloadManager:
    """Manages hot-reloading of Maven modules at runtime."""

    def __init__(self):
        self.root = get_maven_root()
        self.reload_history: List[Dict[str, Any]] = []
        self.module_versions: Dict[str, int] = {}
        self.original_modules: Dict[str, Any] = {}

    # ==================== MODULE RELOADING ====================

    def reload_module(self, module_name: str, validate: bool = True) -> Dict[str, Any]:
        """
        Reload a Python module at runtime.

        Args:
            module_name: Full module name (e.g., 'brains.cognitive.reasoning.service.reasoning_brain')
            validate: Whether to validate the module before reloading

        Returns:
            Dictionary with reload status and information
        """
        require_execution_enabled("hot_reload")

        result = {
            "module": module_name,
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "version": self.module_versions.get(module_name, 0) + 1,
            "error": None
        }

        try:
            # Check if module is already loaded
            if module_name not in sys.modules:
                result["status"] = "not_loaded"
                result["error"] = f"Module {module_name} is not currently loaded"
                return result

            # Store original module for rollback
            if module_name not in self.original_modules:
                self.original_modules[module_name] = sys.modules[module_name]

            # Validate module if requested
            if validate:
                validation = self._validate_module(module_name)
                if not validation["valid"]:
                    result["status"] = "validation_failed"
                    result["error"] = validation["error"]
                    return result

            # Reload the module
            old_module = sys.modules[module_name]
            reloaded_module = importlib.reload(old_module)

            # Update version tracking
            self.module_versions[module_name] = result["version"]

            result["status"] = "success"
            result["module_file"] = getattr(reloaded_module, '__file__', 'unknown')

            print(f"[HOT_RELOAD] Successfully reloaded {module_name} (v{result['version']})")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"[HOT_RELOAD] Error reloading {module_name}: {e}")

        # Record in history
        self.reload_history.append(result)

        return result

    def reload_module_from_file(self, file_path: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Reload a module by specifying its file path.

        Args:
            file_path: Path to the Python file
            module_name: Optional module name (will be inferred if not provided)

        Returns:
            Dictionary with reload status
        """
        require_execution_enabled("hot_reload")

        target = self.root / file_path if not Path(file_path).is_absolute() else Path(file_path)
        target = validate_path_confinement(target, "hot_reload:file")

        if not target.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        # Infer module name from file path if not provided
        if not module_name:
            module_name = self._file_path_to_module_name(target)

        return self.reload_module(module_name)

    def reload_brain(self, brain_name: str) -> Dict[str, Any]:
        """
        Reload a specific brain module.

        Args:
            brain_name: Name of the brain (e.g., 'reasoning', 'teacher', 'integrator')

        Returns:
            Dictionary with reload status
        """
        require_execution_enabled("hot_reload")

        # Try to find the brain's main service module
        possible_paths = [
            f"brains.cognitive.{brain_name}.service.{brain_name}_brain",
            f"brains.cognitive.{brain_name}.{brain_name}_brain",
            f"brains.domain_banks.{brain_name}.service.{brain_name}_bank",
            f"brains.{brain_name}_brain"
        ]

        for module_name in possible_paths:
            if module_name in sys.modules:
                return self.reload_module(module_name)

        return {
            "status": "error",
            "error": f"Brain module not found: {brain_name}",
            "searched": possible_paths
        }

    def reload_multiple(self, module_names: List[str], stop_on_error: bool = False) -> List[Dict[str, Any]]:
        """
        Reload multiple modules.

        Args:
            module_names: List of module names to reload
            stop_on_error: Whether to stop if a reload fails

        Returns:
            List of reload results
        """
        require_execution_enabled("hot_reload")

        results = []
        for module_name in module_names:
            result = self.reload_module(module_name)
            results.append(result)

            if stop_on_error and result["status"] != "success":
                print(f"[HOT_RELOAD] Stopping batch reload due to error in {module_name}")
                break

        return results

    # ==================== DEPENDENCY MANAGEMENT ====================

    def reload_with_dependencies(self, module_name: str) -> Dict[str, Any]:
        """
        Reload a module and all its dependencies.

        Args:
            module_name: Module name to reload

        Returns:
            Dictionary with reload status including dependencies
        """
        require_execution_enabled("hot_reload")

        # Find all dependencies
        deps = self._find_module_dependencies(module_name)

        result = {
            "module": module_name,
            "dependencies": deps,
            "reload_results": []
        }

        # Reload dependencies first (topological order)
        for dep in reversed(deps):
            if dep in sys.modules:
                dep_result = self.reload_module(dep, validate=False)
                result["reload_results"].append(dep_result)

        # Reload main module
        main_result = self.reload_module(module_name)
        result["reload_results"].append(main_result)
        result["status"] = main_result["status"]

        return result

    def _find_module_dependencies(self, module_name: str, visited: Optional[Set[str]] = None) -> List[str]:
        """Find all dependencies of a module."""
        if visited is None:
            visited = set()

        if module_name in visited or module_name not in sys.modules:
            return []

        visited.add(module_name)
        deps = []

        try:
            module = sys.modules[module_name]
            module_file = getattr(module, '__file__', None)

            if module_file:
                # Parse the module file to find imports
                with open(module_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('brains.') or alias.name.startswith('api.'):
                                deps.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and (node.module.startswith('brains.') or node.module.startswith('api.')):
                            deps.append(node.module)

        except Exception:
            pass

        return deps

    # ==================== VALIDATION ====================

    def _validate_module(self, module_name: str) -> Dict[str, Any]:
        """
        Validate a module before reloading.

        Checks for:
        - Syntax errors
        - Import errors
        - Missing required functions/classes
        """
        result = {"valid": True, "error": None, "warnings": []}

        try:
            if module_name not in sys.modules:
                result["valid"] = False
                result["error"] = "Module not loaded"
                return result

            module = sys.modules[module_name]
            module_file = getattr(module, '__file__', None)

            if not module_file:
                result["valid"] = False
                result["error"] = "Module has no file"
                return result

            # Check syntax
            with open(module_file, 'r', encoding='utf-8') as f:
                code = f.read()

            try:
                ast.parse(code)
            except SyntaxError as e:
                result["valid"] = False
                result["error"] = f"Syntax error: {e}"
                return result

            # Additional validation for brain modules
            if ".service." in module_name and "_brain" in module_name:
                # Check for required brain contract methods
                required_methods = ["process"]
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name in required_methods:
                            required_methods.remove(node.name)

                if required_methods:
                    result["warnings"].append(f"Missing recommended methods: {required_methods}")

        except Exception as e:
            result["valid"] = False
            result["error"] = f"Validation error: {str(e)}"

        return result

    # ==================== MODULE REPLACEMENT ====================

    def replace_module_function(self, module_name: str, function_name: str, new_function: Any) -> Dict[str, Any]:
        """
        Replace a specific function in a loaded module.

        Args:
            module_name: Module name
            function_name: Function name to replace
            new_function: New function object

        Returns:
            Dictionary with replacement status
        """
        require_execution_enabled("hot_reload")

        result = {
            "module": module_name,
            "function": function_name,
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            if module_name not in sys.modules:
                result["status"] = "error"
                result["error"] = f"Module {module_name} not loaded"
                return result

            module = sys.modules[module_name]

            if not hasattr(module, function_name):
                result["status"] = "error"
                result["error"] = f"Function {function_name} not found in {module_name}"
                return result

            # Store original function
            original_function = getattr(module, function_name)
            setattr(module, f"_original_{function_name}", original_function)

            # Replace function
            setattr(module, function_name, new_function)

            result["status"] = "success"
            print(f"[HOT_RELOAD] Replaced {module_name}.{function_name}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def restore_module_function(self, module_name: str, function_name: str) -> Dict[str, Any]:
        """Restore a previously replaced function."""
        require_execution_enabled("hot_reload")

        result = {
            "module": module_name,
            "function": function_name,
            "status": "unknown"
        }

        try:
            if module_name not in sys.modules:
                result["status"] = "error"
                result["error"] = "Module not loaded"
                return result

            module = sys.modules[module_name]
            original_attr = f"_original_{function_name}"

            if not hasattr(module, original_attr):
                result["status"] = "error"
                result["error"] = "No backup found"
                return result

            original_function = getattr(module, original_attr)
            setattr(module, function_name, original_function)
            delattr(module, original_attr)

            result["status"] = "success"
            print(f"[HOT_RELOAD] Restored {module_name}.{function_name}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    # ==================== ROLLBACK ====================

    def rollback_module(self, module_name: str) -> Dict[str, Any]:
        """
        Rollback a module to its original state.

        Args:
            module_name: Module name to rollback

        Returns:
            Dictionary with rollback status
        """
        require_execution_enabled("hot_reload")

        result = {
            "module": module_name,
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            if module_name not in self.original_modules:
                result["status"] = "error"
                result["error"] = "No backup available"
                return result

            # Restore original module
            sys.modules[module_name] = self.original_modules[module_name]

            result["status"] = "success"
            print(f"[HOT_RELOAD] Rolled back {module_name}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    # ==================== INTROSPECTION ====================

    def get_loaded_maven_modules(self) -> List[Dict[str, Any]]:
        """Get all currently loaded Maven modules."""
        modules = []

        for name, module in sys.modules.items():
            if name.startswith('brains.') or name.startswith('api.') or name.startswith('ui.'):
                module_info = {
                    "name": name,
                    "file": getattr(module, '__file__', None),
                    "version": self.module_versions.get(name, 0),
                    "has_backup": name in self.original_modules
                }
                modules.append(module_info)

        return sorted(modules, key=lambda x: x["name"])

    def get_reload_history(self, module_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get reload history."""
        history = self.reload_history

        if module_name:
            history = [h for h in history if h["module"] == module_name]

        return history[-limit:]

    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Get detailed information about a module."""
        info = {
            "name": module_name,
            "loaded": module_name in sys.modules,
            "version": self.module_versions.get(module_name, 0),
            "has_backup": module_name in self.original_modules,
            "file": None,
            "dependencies": []
        }

        if module_name in sys.modules:
            module = sys.modules[module_name]
            info["file"] = getattr(module, '__file__', None)
            info["dependencies"] = self._find_module_dependencies(module_name)

        return info

    # ==================== HELPER METHODS ====================

    def _file_path_to_module_name(self, file_path: Path) -> str:
        """Convert a file path to a module name."""
        try:
            rel_path = file_path.relative_to(self.root)
        except ValueError:
            rel_path = file_path

        # Remove .py extension
        module_path = str(rel_path).replace('.py', '')

        # Replace path separators with dots
        module_name = module_path.replace('/', '.').replace('\\', '.')

        return module_name

    def list_reloadable_modules(self, pattern: Optional[str] = None) -> List[str]:
        """List all modules that can be reloaded."""
        reloadable = []

        for name in sys.modules.keys():
            if name.startswith('brains.') or name.startswith('api.') or name.startswith('ui.'):
                if pattern is None or pattern in name:
                    reloadable.append(name)

        return sorted(reloadable)


# Global instance
_hot_reload_manager = None


def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot-reload manager instance."""
    global _hot_reload_manager
    if _hot_reload_manager is None:
        _hot_reload_manager = HotReloadManager()
    return _hot_reload_manager


# Convenience functions
def reload_module(module_name: str) -> Dict[str, Any]:
    """Convenience function to reload a module."""
    return get_hot_reload_manager().reload_module(module_name)


def reload_brain(brain_name: str) -> Dict[str, Any]:
    """Convenience function to reload a brain."""
    return get_hot_reload_manager().reload_brain(brain_name)


def get_loaded_modules() -> List[Dict[str, Any]]:
    """Convenience function to get loaded modules."""
    return get_hot_reload_manager().get_loaded_maven_modules()
