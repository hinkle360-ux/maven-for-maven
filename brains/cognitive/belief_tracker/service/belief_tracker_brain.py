"""
Belief Tracker Brain - Wrapper Module
======================================
Re-exports service_api from belief_tracker.py for consistent naming.
"""
from brains.cognitive.belief_tracker.service.belief_tracker import handle as service_api

__all__ = ["service_api"]
