"""
Context Management Brain - Wrapper Module
==========================================
Re-exports service_api from context_manager.py for consistent naming.
"""
from brains.cognitive.context_management.service.context_manager import handle as service_api

__all__ = ["service_api"]
