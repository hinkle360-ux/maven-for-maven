"""
Environment Context Brain - Wrapper Module
===========================================
Re-exports service_api from environment_brain.py for consistent naming.
"""
from brains.cognitive.environment_context.service.environment_brain import handle as service_api

__all__ = ["service_api"]
