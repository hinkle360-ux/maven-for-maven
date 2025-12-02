from typing import Dict, Any
from brains.domain_banks.bank_template import bank_service_factory

_service_impl = bank_service_factory('creative')

def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle creative domain bank operations.

    Supported operations:
    - STORE: Store a fact in STM
    - RETRIEVE: Retrieve facts matching query
    - COUNT: Get tier counts
    - REBUILD_INDEX: Rebuild search index
    - COMPACT_ARCHIVE: Compact archive tier

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict from domain bank
    """
    return _service_impl(msg)

# Standard service contract: handle is the entry point
service_api = handle
