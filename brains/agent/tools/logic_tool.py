"""
Logic Tool
==========

This simple tool evaluates boolean expressions composed of
``True``/``False``, the logical operators ``and``, ``or`` and ``not``,
and parentheses.  It deliberately avoids exposing arbitrary Python
execution by restricting the evaluation environment.  The service API
accepts a single operation ``EVAL`` with an ``expression`` payload and
returns the boolean result.

Example:

>>> service_api({"op": "EVAL", "payload": {"expression": "True and not False"}})
{"ok": True, "payload": {"result": True}}
"""

from __future__ import annotations

import ast
from typing import Any, Dict


class _BoolSafeEval(ast.NodeVisitor):
    """
    Strict boolean-expression evaluator.
    Allowed: True/False, names (from context), and/or/not, parentheses,
    ==, !=, <, <=, >, >=, and boolean literals.
    """
    ALLOWED_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def __init__(self, names: Dict[str, Any]):
        self.names = names

    def visit_Module(self, node: ast.Module) -> bool:  # type: ignore[override]
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValueError("Only a single expression is allowed")
        return self.visit(node.body[0].value)

    def visit_Name(self, node: ast.Name):
        if node.id in self.names:
            val = self.names[node.id]
            if isinstance(val, bool):
                return val
            raise ValueError(f"name '{node.id}' must be bool")
        raise ValueError(f"unknown name '{node.id}'")

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        raise ValueError("only boolean literals allowed")

    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = result and bool(self.visit(v))
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = result or bool(self.visit(v))
            return result
        raise ValueError("only 'and'/'or' are allowed")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not bool(self.visit(node.operand))
        raise ValueError("only 'not' is allowed")

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("chained comparisons not allowed")
        op = node.ops[0]
        if not isinstance(op, self.ALLOWED_COMPARE):
            raise ValueError("comparison op not allowed")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        if not (isinstance(left, bool) and isinstance(right, bool)):
            raise ValueError("comparisons must be between booleans")
        if isinstance(op, ast.Eq):
            return left == right
        elif isinstance(op, ast.NotEq):
            return left != right
        elif isinstance(op, ast.Lt):
            return left < right
        elif isinstance(op, ast.LtE):
            return left <= right
        elif isinstance(op, ast.Gt):
            return left > right
        elif isinstance(op, ast.GtE):
            return left >= right
        else:
            raise ValueError("invalid comparison operator")


def _safe_eval(expr: str) -> bool:
    """Evaluate a boolean expression using a restricted environment."""
    expr = str(expr or "").strip()
    names: Dict[str, bool] = {"True": True, "False": False}
    try:
        tree = ast.parse(expr, mode="exec")
        evaluator = _BoolSafeEval(names)
        return bool(evaluator.visit(tree))
    except Exception as exc:
        raise


def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}
    mid = msg.get("mid") or "LOGIC"

    if op == "EVAL" or op == "EVALUATE":
        expr = payload.get("expression")
        try:
            res = _safe_eval(expr)
        except Exception as e:
            return {"ok": False, "op": op, "mid": mid, "error": {"code": "INVALID_EXPRESSION", "message": str(e)}}
        return {"ok": True, "op": op, "mid": mid, "payload": {"result": res}}

    if op == "HEALTH":
        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {
                "status": "operational",
                "service": "logic",
                "capability": "logic",
                "description": "Safe boolean expression evaluation",
            }
        }

    return {"ok": False, "op": op, "mid": mid, "error": {"code": "UNSUPPORTED_OP", "message": op}}


# Standard service contract: handle is the entry point
handle = service_api


# ============================================================================
# Tool Metadata (for registry and capabilities)
# ============================================================================

TOOL_NAME = "logic"
TOOL_CAPABILITY = "logic"
TOOL_DESCRIPTION = "Safe boolean expression evaluation"
TOOL_OPERATIONS = ["EVAL", "EVALUATE", "HEALTH"]


def is_available() -> bool:
    """Check if logic tool is available (always True)."""
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
        "module": "brains.agent.tools.logic_tool",
    }