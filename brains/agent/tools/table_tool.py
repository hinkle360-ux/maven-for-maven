"""
Table Tool
==========

This utility provides basic operations on tabular data represented as
CSV strings or lists of dictionaries.  Currently supported
operations:

* ``PARSE_CSV`` – parse a CSV string into a list of dictionaries
  using the first row as the header.
* ``SUM_COLUMN`` – compute the sum of numeric values in a given
  column.  Accepts either a list of dictionaries (as produced by
  ``PARSE_CSV``) or a raw CSV string with a header row.  Column may
  be specified by name or zero‑based index.

Example:

>>> csv_data = "a,b\n1,2\n3,4"
>>> service_api({"op": "PARSE_CSV", "payload": {"csv": csv_data}})
{"ok": True, "payload": {"rows": [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]}}
>>> service_api({"op": "SUM_COLUMN", "payload": {"table": [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}], "column": "b"}})
{"ok": True, "payload": {"sum": 6.0}}
"""

from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, Any, List, Union, Iterable


def _parse_csv_to_dicts(csv_text: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(StringIO(csv_text))
    return [dict(row) for row in reader]


def _sum_column_from_rows(rows: Iterable[Dict[str, Any]], column: Union[str, int]) -> float:
    total = 0.0
    for row in rows:
        try:
            if isinstance(column, int):
                # convert to list for index access
                vals = list(row.values())
                val = vals[column]
            else:
                val = row.get(column)
            total += float(val)
        except Exception:
            continue
    return total


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "TABLE"

    if op == "PARSE_CSV" or op == "PARSE":
        csv_data = payload.get("csv") or ""
        try:
            rows = _parse_csv_to_dicts(str(csv_data))
            return {"ok": True, "op": op, "mid": mid, "payload": {"rows": rows}}
        except Exception as e:
            return {"ok": False, "op": op, "mid": mid, "error": {"code": "INVALID_CSV", "message": str(e)}}

    if op == "SUM_COLUMN":
        table = payload.get("table")
        column = payload.get("column")
        try:
            rows: List[Dict[str, Any]]
            if table is None:
                # Fallback to parse CSV if provided
                csv_text = payload.get("csv") or ""
                rows = _parse_csv_to_dicts(str(csv_text))
            else:
                # Assume already a list of dictionaries
                rows = list(table)  # type: ignore
            col: Union[str, int] = column
            # If column is numeric string, treat as index
            try:
                if isinstance(col, str) and col.isdigit():
                    col = int(col)
            except Exception:
                pass
            s = _sum_column_from_rows(rows, col)
        except Exception as e:
            return {"ok": False, "op": op, "mid": mid, "error": {"code": "INVALID_TABLE", "message": str(e)}}
        return {"ok": True, "op": op, "mid": mid, "payload": {"sum": s}}

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service": "table",
                "capability": "table",
                "description": "CSV parsing and table operations",
            }
        }

    return {"ok": False, "op": op, "mid": mid, "error": {"code": "UNSUPPORTED_OP", "message": op}}


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "table"
TOOL_CAPABILITY = "table"
TOOL_DESCRIPTION = "CSV parsing and table operations"
TOOL_OPERATIONS = ["PARSE_CSV", "PARSE", "SUM_COLUMN", "HEALTH"]


def is_available() -> bool:
    """Check if table tool is available (always True)."""
    return True


def get_tool_info() -> Dict[str, Any]:
    """Get tool metadata for registry."""
    return {
        "name": TOOL_NAME,
        "capability": TOOL_CAPABILITY,
        "description": TOOL_DESCRIPTION,
        "operations": TOOL_OPERATIONS,
        "available": True,
        "requires_execution": False,
        "module": "brains.agent.tools.table_tool",
    }