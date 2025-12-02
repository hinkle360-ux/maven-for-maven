"""
Action Engine Brain - Wrapper Module
=====================================
Re-exports service_api from action_engine.py for consistent naming.
"""
from brains.cognitive.action_engine.service.action_engine import handle as service_api

__all__ = ["service_api"]
