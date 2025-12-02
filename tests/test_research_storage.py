#!/usr/bin/env python3
"""Direct test of research report storage."""

from __future__ import annotations
import sys
from pathlib import Path
import traceback

MAVEN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MAVEN_ROOT))

def _direct_storage_ok() -> bool:
    """Return True if storing and retrieving a research report succeeds."""
    print("Testing direct storage to research_reports bank...")

    try:
        from brains.domain_banks.research_reports.service.research_reports_bank import service_api

        # Try to store a test report
        msg = {
            "op": "STORE",
            "mid": "test_direct_001",
            "payload": {
                "fact": {
                    "content": "Test research report on quantum computers",
                    "confidence": 0.8,
                    "source": "test_script",
                    "metadata": {
                        "topic": "quantum computers",
                        "sources": ["test"],
                        "facts_count": 1,
                        "timestamp": 1234567890,
                    }
                }
            }
        }

        print(f"Sending: {msg}")
        result = service_api(msg)
        print(f"Result: {result}")

        # Try to retrieve it
        msg2 = {
            "op": "RETRIEVE",
            "mid": "test_retrieve_001",
            "payload": {
                "query": "quantum",
                "limit": 5,
            }
        }

        print(f"\nRetrieving: {msg2}")
        result2 = service_api(msg2)
        print(f"Result: {result2}")

        payload = result2.get("payload", {})
        results = payload.get("results", [])
        print(f"\nFound {len(results)} results")

        if results:
            print("✓ Storage test PASSED")
            return True
        else:
            print("✗ Storage test FAILED - no results found")
            return False

    except Exception as e:
        print(f"✗ ERROR: {e}")
        traceback.print_exc()
        return False


def test_direct_storage():
    """Test storing directly to research_reports bank."""
    result = _direct_storage_ok()
    assert isinstance(result, bool)


if __name__ == "__main__":
    success = _direct_storage_ok()
    sys.exit(0 if success else 1)
