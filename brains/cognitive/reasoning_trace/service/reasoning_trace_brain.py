"""
Reasoning Trace Brain - Wrapper Module
=======================================
Re-exports service_api from trace_service.py for consistent naming.
"""
from brains.cognitive.reasoning_trace.service.trace_service import handle as service_api

__all__ = ["service_api"]
