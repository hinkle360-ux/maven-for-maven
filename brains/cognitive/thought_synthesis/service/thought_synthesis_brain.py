"""
Thought Synthesis Brain - Wrapper Module
=========================================
Re-exports service_api from thought_synthesizer.py for consistent naming.
"""
from brains.cognitive.thought_synthesis.service.thought_synthesizer import handle as service_api

__all__ = ["service_api"]
