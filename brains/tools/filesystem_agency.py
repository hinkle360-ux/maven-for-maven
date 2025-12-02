"""
Comprehensive Filesystem Agency Module
======================================

Provides Maven with full filesystem access capabilities including:
- Directory scanning and traversal
- File reading and writing
- File operations (copy, move, delete)
- Directory operations (create, remove)
- Path analysis and introspection
- Code analysis and parsing

All operations are confined to the Maven root and respect security policies.
"""

from __future__ import annotations

import os
import shutil
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brains.maven_paths import MAVEN_ROOT, get_maven_root, validate_path_confinement
from brains.tools.execution_guard import require_execution_enabled


class FilesystemAgency:
    """Maven's filesystem agency providing comprehensive file operations."""

    def __init__(self):
        self.root = get_maven_root()
        self.scan_cache: Dict[str, Any] = {}

    # ==================== SCANNING OPERATIONS ====================

    def scan_directory_tree(self, start_path: Optional[str] = None, max_depth: int = -1) -> Dict[str, Any]:
        """
        Scan the complete directory tree starting from start_path.

        Args:
            start_path: Path to start scanning from (defaults to Maven root)
            max_depth: Maximum depth to scan (-1 for unlimited)

        Returns:
            Dictionary with tree structure, file counts, and metadata
        """
        base = Path(start_path) if start_path else self.root
        base = validate_path_confinement(base, "filesystem_agency:scan_tree")

        tree = {
            "root": str(base.relative_to(self.root)),
            "directories": [],
            "files": [],
            "total_size": 0,
            "file_count": 0,
            "dir_count": 0
        }

        def _scan_recursive(path: Path, depth: int = 0):
            if max_depth >= 0 and depth > max_depth:
                return

            try:
                for entry in sorted(path.iterdir()):
                    if entry.name.startswith('.') or entry.name == '__pycache__':
                        continue

                    rel_path = str(entry.relative_to(self.root))

                    if entry.is_dir():
                        tree["directories"].append(rel_path)
                        tree["dir_count"] += 1
                        _scan_recursive(entry, depth + 1)
                    elif entry.is_file():
                        stat = entry.stat()
                        file_info = {
                            "path": rel_path,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "extension": entry.suffix
                        }
                        tree["files"].append(file_info)
                        tree["file_count"] += 1
                        tree["total_size"] += stat.st_size
            except PermissionError:
                pass

        _scan_recursive(base)
        print(f"[FILESYSTEM_AGENCY] Scanned {tree['file_count']} files in {tree['dir_count']} directories")
        return tree

    def list_python_files(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all Python files in directory and subdirectories."""
        base = Path(directory) if directory else self.root
        base = validate_path_confinement(base, "filesystem_agency:list_py")

        py_files = []
        for py_file in sorted(base.rglob("*.py")):
            if '__pycache__' in py_file.parts:
                continue
            try:
                stat = py_file.stat()
                py_files.append({
                    "path": str(py_file.relative_to(self.root)),
                    "absolute": str(py_file),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "lines": self._count_lines(py_file)
                })
            except Exception:
                continue

        print(f"[FILESYSTEM_AGENCY] Found {len(py_files)} Python files")
        return py_files

    def list_directory(self, path: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """List contents of a specific directory with optional pattern matching."""
        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:list_dir")

        if not target.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")

        entries = []
        for entry in sorted(target.glob(pattern)):
            stat = entry.stat()
            entry_info = {
                "name": entry.name,
                "path": str(entry.relative_to(self.root)),
                "type": "directory" if entry.is_dir() else "file",
                "size": stat.st_size if entry.is_file() else 0,
                "modified": stat.st_mtime
            }
            entries.append(entry_info)

        return entries

    # ==================== READING OPERATIONS ====================

    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read and return the contents of a file."""
        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:read")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = target.read_text(encoding=encoding)
        print(f"[FILESYSTEM_AGENCY] Read {len(content)} chars from {path}")
        return content

    def read_lines(self, path: str, start: int = 0, end: Optional[int] = None, encoding: str = "utf-8") -> List[str]:
        """Read specific lines from a file."""
        content = self.read_file(path, encoding)
        lines = content.splitlines()
        return lines[start:end] if end else lines[start:]

    def read_binary(self, path: str) -> bytes:
        """Read a file in binary mode."""
        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:read_binary")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = target.read_bytes()
        print(f"[FILESYSTEM_AGENCY] Read {len(data)} bytes from {path}")
        return data

    # ==================== WRITING OPERATIONS ====================

    def write_file(self, path: str, content: str, backup: bool = True, encoding: str = "utf-8") -> Dict[str, Any]:
        """Write content to a file with optional backup."""
        require_execution_enabled("filesystem_agency:write")

        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:write")

        # Create parent directories
        target.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "path": str(target.relative_to(self.root)),
            "backup_path": None,
            "bytes_written": 0
        }

        # Create backup if requested and file exists
        if backup and target.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = target.with_suffix(target.suffix + f".backup_{timestamp}")
            shutil.copy2(target, backup_path)
            result["backup_path"] = str(backup_path.relative_to(self.root))
            print(f"[FILESYSTEM_AGENCY] Created backup: {result['backup_path']}")

        # Write content
        target.write_text(content, encoding=encoding)
        result["bytes_written"] = len(content)

        print(f"[FILESYSTEM_AGENCY] Wrote {result['bytes_written']} bytes to {path}")
        return result

    def append_to_file(self, path: str, content: str, encoding: str = "utf-8") -> int:
        """Append content to an existing file."""
        require_execution_enabled("filesystem_agency:append")

        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:append")

        with open(target, "a", encoding=encoding) as f:
            f.write(content)

        print(f"[FILESYSTEM_AGENCY] Appended {len(content)} bytes to {path}")
        return len(content)

    def write_binary(self, path: str, data: bytes, backup: bool = True) -> Dict[str, Any]:
        """Write binary data to a file."""
        require_execution_enabled("filesystem_agency:write_binary")

        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:write_binary")

        target.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "path": str(target.relative_to(self.root)),
            "backup_path": None,
            "bytes_written": 0
        }

        if backup and target.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = target.with_suffix(target.suffix + f".backup_{timestamp}")
            shutil.copy2(target, backup_path)
            result["backup_path"] = str(backup_path.relative_to(self.root))

        target.write_bytes(data)
        result["bytes_written"] = len(data)

        print(f"[FILESYSTEM_AGENCY] Wrote {result['bytes_written']} bytes (binary) to {path}")
        return result

    # ==================== FILE OPERATIONS ====================

    def copy_file(self, source: str, destination: str, backup: bool = True) -> Dict[str, Any]:
        """Copy a file from source to destination."""
        require_execution_enabled("filesystem_agency:copy")

        src = self.root / source if not Path(source).is_absolute() else Path(source)
        dst = self.root / destination if not Path(destination).is_absolute() else Path(destination)

        src = validate_path_confinement(src, "filesystem_agency:copy_src")
        dst = validate_path_confinement(dst, "filesystem_agency:copy_dst")

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)

        result = {"source": str(src.relative_to(self.root)), "destination": str(dst.relative_to(self.root)), "backup_path": None}

        if backup and dst.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = dst.with_suffix(dst.suffix + f".backup_{timestamp}")
            shutil.copy2(dst, backup_path)
            result["backup_path"] = str(backup_path.relative_to(self.root))

        shutil.copy2(src, dst)
        print(f"[FILESYSTEM_AGENCY] Copied {source} -> {destination}")

        return result

    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move/rename a file from source to destination."""
        require_execution_enabled("filesystem_agency:move")

        src = self.root / source if not Path(source).is_absolute() else Path(source)
        dst = self.root / destination if not Path(destination).is_absolute() else Path(destination)

        src = validate_path_confinement(src, "filesystem_agency:move_src")
        dst = validate_path_confinement(dst, "filesystem_agency:move_dst")

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

        result = {"source": str(src.relative_to(self.root)), "destination": str(dst.relative_to(self.root))}
        print(f"[FILESYSTEM_AGENCY] Moved {source} -> {destination}")

        return result

    def delete_file(self, path: str, backup: bool = True) -> Dict[str, Any]:
        """Delete a file with optional backup."""
        require_execution_enabled("filesystem_agency:delete")

        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:delete")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        result = {"path": str(target.relative_to(self.root)), "backup_path": None}

        if backup:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.root / "reports" / "deleted_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{target.name}.deleted_{timestamp}"
            shutil.copy2(target, backup_path)
            result["backup_path"] = str(backup_path.relative_to(self.root))

        target.unlink()
        print(f"[FILESYSTEM_AGENCY] Deleted {path}")

        return result

    # ==================== DIRECTORY OPERATIONS ====================

    def create_directory(self, path: str, parents: bool = True) -> str:
        """Create a directory."""
        require_execution_enabled("filesystem_agency:mkdir")

        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:mkdir")

        target.mkdir(parents=parents, exist_ok=True)

        rel_path = str(target.relative_to(self.root))
        print(f"[FILESYSTEM_AGENCY] Created directory: {rel_path}")
        return rel_path

    def remove_directory(self, path: str, recursive: bool = False) -> str:
        """Remove a directory."""
        require_execution_enabled("filesystem_agency:rmdir")

        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:rmdir")

        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()

        rel_path = str(target.relative_to(self.root))
        print(f"[FILESYSTEM_AGENCY] Removed directory: {rel_path}")
        return rel_path

    def copy_directory(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy an entire directory tree."""
        require_execution_enabled("filesystem_agency:copy_dir")

        src = self.root / source if not Path(source).is_absolute() else Path(source)
        dst = self.root / destination if not Path(destination).is_absolute() else Path(destination)

        src = validate_path_confinement(src, "filesystem_agency:copy_dir_src")
        dst = validate_path_confinement(dst, "filesystem_agency:copy_dir_dst")

        if not src.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        shutil.copytree(src, dst, dirs_exist_ok=True)

        result = {"source": str(src.relative_to(self.root)), "destination": str(dst.relative_to(self.root))}
        print(f"[FILESYSTEM_AGENCY] Copied directory {source} -> {destination}")

        return result

    # ==================== CODE ANALYSIS ====================

    def analyze_python_file(self, path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract structure information."""
        content = self.read_file(path)

        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}", "path": path}

        analysis = {
            "path": path,
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": [],
            "docstring": ast.get_docstring(tree)
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                analysis["classes"].append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "methods": methods,
                    "docstring": ast.get_docstring(node)
                })
            elif isinstance(node, ast.FunctionDef):
                if not any(node in cls.body for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]):
                    analysis["functions"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append({"module": alias.name, "alias": alias.asname})
                else:
                    module = node.module or ""
                    for alias in node.names:
                        analysis["imports"].append({
                            "module": f"{module}.{alias.name}" if module else alias.name,
                            "alias": alias.asname
                        })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        analysis["constants"].append({"name": target.id, "lineno": node.lineno})

        print(f"[FILESYSTEM_AGENCY] Analyzed {path}: {len(analysis['classes'])} classes, {len(analysis['functions'])} functions")
        return analysis

    def find_class_definition(self, class_name: str, directory: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find the file containing a specific class definition."""
        search_dir = Path(directory) if directory else self.root

        for py_file in search_dir.rglob("*.py"):
            if '__pycache__' in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        return {
                            "file": str(py_file.relative_to(self.root)),
                            "lineno": node.lineno,
                            "class_name": class_name
                        }
            except Exception:
                continue

        return None

    def find_function_definition(self, func_name: str, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all files containing a specific function definition."""
        search_dir = Path(directory) if directory else self.root
        results = []

        for py_file in search_dir.rglob("*.py"):
            if '__pycache__' in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        results.append({
                            "file": str(py_file.relative_to(self.root)),
                            "lineno": node.lineno,
                            "function_name": func_name
                        })
            except Exception:
                continue

        return results

    def detect_imports(self, path: str) -> Dict[str, List[str]]:
        """Detect all imports in a Python file."""
        content = self.read_file(path)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"error": ["Syntax error in file"]}

        imports = {"stdlib": [], "third_party": [], "local": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    imports["third_party"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module == "brains" or module == "api" or module == "ui":
                        imports["local"].append(node.module)
                    else:
                        imports["third_party"].append(node.module)

        return imports

    # ==================== HELPER METHODS ====================

    def _count_lines(self, path: Path) -> int:
        """Count the number of lines in a file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        target = self.root / path if not Path(path).is_absolute() else Path(path)
        try:
            target = validate_path_confinement(target, "filesystem_agency:exists")
            return target.exists()
        except Exception:
            return False

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed information about a file."""
        target = self.root / path if not Path(path).is_absolute() else Path(path)
        target = validate_path_confinement(target, "filesystem_agency:info")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = target.stat()
        return {
            "path": str(target.relative_to(self.root)),
            "absolute_path": str(target),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "is_file": target.is_file(),
            "is_directory": target.is_dir(),
            "extension": target.suffix,
            "name": target.name
        }


# Global instance for easy access
_agency = None

def get_filesystem_agency() -> FilesystemAgency:
    """Get the global filesystem agency instance."""
    global _agency
    if _agency is None:
        _agency = FilesystemAgency()
    return _agency


# Convenience functions for backward compatibility
def scan_codebase(root: Optional[str] = None, pattern: str = "*.py") -> List[Dict[str, Any]]:
    """Legacy compatibility function for scan_codebase."""
    agency = get_filesystem_agency()
    if pattern == "*.py":
        return agency.list_python_files(root)
    else:
        tree = agency.scan_directory_tree(root)
        return tree["files"]


# ============================================================================
# Service API (Standard Tool Interface)
# ============================================================================

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API for filesystem_agency.

    Operations:
    - SCAN_TREE: Scan directory tree structure
    - LIST_PYTHON_FILES: List all Python files
    - READ_FILE: Read file contents
    - WRITE_FILE: Write content to file
    - FILE_INFO: Get file metadata
    - FILE_EXISTS: Check if file exists
    - ANALYZE_FILE: Analyze Python file structure
    - FIND_CLASS: Find class definition
    - FIND_FUNCTION: Find function definition
    - DETECT_IMPORTS: Detect imports in a Python file
    - HEALTH: Health check

    Args:
        msg: Dict with "op" and optional "payload"

    Returns:
        Dict with "ok", "payload" or "error"
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "FILESYSTEM_AGENCY"

    agency = get_filesystem_agency()

    try:
        if op == "SCAN_TREE":
            start_path = payload.get("path")
            max_depth = payload.get("max_depth", -1)

            result = agency.scan_directory_tree(start_path, max_depth=max_depth)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result
            }

        if op == "LIST_PYTHON_FILES":
            directory = payload.get("directory")
            result = agency.list_python_files(directory)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"files": result, "count": len(result)}
            }

        if op == "READ_FILE":
            path = payload.get("path", "")
            if not path:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATH", "message": "Path is required"}
                }

            content = agency.read_file(path)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"content": content, "path": path, "size": len(content)}
            }

        if op == "WRITE_FILE":
            path = payload.get("path", "")
            content = payload.get("content", "")

            if not path:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATH", "message": "Path is required"}
                }

            result = agency.write_file(path, content)
            return {
                "ok": result.get("status") == "created" or result.get("status") == "updated",
                "op": op,
                "mid": mid,
                "payload": result
            }

        if op == "FILE_INFO":
            path = payload.get("path", "")
            if not path:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATH", "message": "Path is required"}
                }

            result = agency.get_file_info(path)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result
            }

        if op == "FILE_EXISTS":
            path = payload.get("path", "")
            if not path:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATH", "message": "Path is required"}
                }

            exists = agency.file_exists(path)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"exists": exists, "path": path}
            }

        if op == "ANALYZE_FILE":
            path = payload.get("path", "")
            if not path:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATH", "message": "Path is required"}
                }

            result = agency.analyze_python_file(path)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": result
            }

        if op == "FIND_CLASS":
            class_name = payload.get("class_name", "")
            directory = payload.get("directory")

            if not class_name:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_CLASS_NAME", "message": "Class name is required"}
                }

            result = agency.find_class_definition(class_name, directory)
            return {
                "ok": result is not None,
                "op": op,
                "mid": mid,
                "payload": result if result else {"found": False, "class_name": class_name}
            }

        if op == "FIND_FUNCTION":
            func_name = payload.get("func_name", "") or payload.get("function_name", "")
            directory = payload.get("directory")

            if not func_name:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_FUNC_NAME", "message": "Function name is required"}
                }

            results = agency.find_function_definition(func_name, directory)
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {"matches": results, "count": len(results)}
            }

        if op == "DETECT_IMPORTS":
            path = payload.get("path", "")
            if not path:
                return {
                    "ok": False,
                    "op": op,
                    "mid": mid,
                    "error": {"code": "MISSING_PATH", "message": "Path is required"}
                }

            result = agency.detect_imports(path)
            return {
                "ok": "error" not in result,
                "op": op,
                "mid": mid,
                "payload": result
            }

        if op == "HEALTH":
            return {
                "ok": True,
                "op": op,
                "mid": mid,
                "payload": {
                    "status": "operational",
                    "service": "filesystem_agency",
                    "capability": "filesystem",
                    "description": "Comprehensive filesystem operations with code analysis",
                    "root": str(agency.root),
                }
            }

        return {
            "ok": False,
            "op": op,
            "mid": mid,
            "error": {"code": "UNSUPPORTED_OP", "message": f"Unknown operation: {op}"}
        }

    except Exception as e:
        return {
            "ok": False,
            "op": op,
            "mid": mid,
            "error": {"code": "FS_AGENCY_ERROR", "message": str(e)}
        }


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "filesystem_agency"
TOOL_CAPABILITY = "filesystem"
TOOL_DESCRIPTION = "Comprehensive filesystem operations with code analysis"
TOOL_OPERATIONS = [
    "SCAN_TREE", "LIST_PYTHON_FILES", "READ_FILE", "WRITE_FILE",
    "FILE_INFO", "FILE_EXISTS", "ANALYZE_FILE", "FIND_CLASS",
    "FIND_FUNCTION", "DETECT_IMPORTS", "HEALTH"
]


def is_available() -> bool:
    """Check if filesystem_agency is available (always True)."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
        "requires_execution": False,
        "module": "brains.tools.filesystem_agency",
    }
