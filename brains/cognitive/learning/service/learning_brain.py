"""
Learning Brain - Wrapper Module
================================
Re-exports service_api from meta_learning.py for consistent naming.
"""
from brains.cognitive.learning.service.meta_learning import handle as service_api

__all__ = ["service_api"]
