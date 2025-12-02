"""
Memory Librarian Brain - Wrapper Module
========================================
Re-exports service_api from memory_librarian.py for consistent naming.
"""
from brains.cognitive.memory_librarian.service.memory_librarian import service_api

__all__ = ["service_api"]
