"""
External Interfaces Brain - Wrapper Module
===========================================
Re-exports service_api from connector.py for consistent naming.
"""
from brains.cognitive.external_interfaces.service.connector import handle as service_api

__all__ = ["service_api"]
