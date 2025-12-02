"""
Attention Brain - Wrapper Module
=================================
Re-exports service_api from attention_service.py for consistent naming.
"""
from brains.cognitive.attention.service.attention_service import handle as service_api

__all__ = ["service_api"]
