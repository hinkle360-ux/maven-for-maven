#!/usr/bin/env python3
"""
Cleanup script to remove all Python bytecode caches.

This ensures no stale .pyc files cause import issues or run old code.
Run this whenever you want to ensure a clean Python runtime.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def cleanup_caches():
    """Remove all __pycache__ directories and .pyc files."""
    removed_count = 0

    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Remove __pycache__ directories
        if "__pycache__" in dirnames:
            cache_dir = os.path.join(dirpath, "__pycache__")
            print(f"Removing {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed_count += 1

        # Also remove any stray .pyc files
        for filename in filenames:
            if filename.endswith(".pyc"):
                pyc_file = os.path.join(dirpath, filename)
                print(f"Removing {pyc_file}")
                os.remove(pyc_file)
                removed_count += 1

    if removed_count == 0:
        print("No cache files found - already clean!")
    else:
        print(f"\nCleaned {removed_count} cache directories/files")

if __name__ == "__main__":
    cleanup_caches()
