from typing import Dict, Any
from brains.domain_banks.bank_template import bank_service_factory

_service_impl = bank_service_factory('research_reports')

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle research_reports domain bank operations.

    Supported operations mirror the standard domain bank contract:
    - STORE: Store a structured research report entry in STM
    - RETRIEVE: Retrieve reports matching a query
    - COUNT: Get tier counts
    - REBUILD_INDEX: Rebuild search index for topic/token queries
    - COMPACT_ARCHIVE: Compact archive tier
    """
    return _service_impl(msg)

# Standard service contract entry point
service_api = handle
