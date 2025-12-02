"""
Safe Math Expression Evaluator for Maven 2.0

This module provides a secure alternative to eval() for evaluating
mathematical expressions. It uses Python's AST module to parse and
evaluate expressions, only allowing safe mathematical operations.

SECURITY: This replaces dangerous eval() calls that could execute
arbitrary code. Only numeric literals and math operators are allowed.
"""

import ast
import operator
import math
from typing import Union, Optional

# Allowed binary operators
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions (safe subset)
_MATH_FUNCS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'floor': math.floor,
    'ceil': math.ceil,
    'pow': pow,
}

# Allowed constants
_CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
}

# Maximum allowed value to prevent resource exhaustion
_MAX_VALUE = 10 ** 100
_MAX_EXPONENT = 1000


class SafeMathError(Exception):
    """Raised when an unsafe or invalid expression is detected."""
    pass


def _eval_node(node: ast.AST) -> Union[int, float]:
    """
    Recursively evaluate an AST node.

    Only allows:
    - Numeric literals (int, float)
    - Binary operations (+, -, *, /, //, %, **)
    - Unary operations (+, -)
    - Parentheses (implicit in AST structure)
    - Safe math functions and constants
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    elif isinstance(node, ast.Constant):
        # Python 3.8+ uses ast.Constant for all literals
        if isinstance(node.value, (int, float)):
            if abs(node.value) > _MAX_VALUE:
                raise SafeMathError(f"Value too large: {node.value}")
            return node.value
        raise SafeMathError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.Num):
        # Python 3.7 compatibility
        if abs(node.n) > _MAX_VALUE:
            raise SafeMathError(f"Value too large: {node.n}")
        return node.n

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise SafeMathError(f"Unsupported operator: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Special handling for power to prevent resource exhaustion
        if op_type == ast.Pow:
            if abs(right) > _MAX_EXPONENT:
                raise SafeMathError(f"Exponent too large: {right}")
            # Check if result would overflow before computing
            if left != 0 and right > 0:
                import math as _math
                try:
                    log_result = _math.log10(abs(left)) * right
                    if log_result > 100:  # 10^100 limit
                        raise SafeMathError(f"Result would be too large")
                except (ValueError, OverflowError):
                    pass  # Let it compute and check result

        # Prevent division by zero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise SafeMathError("Division by zero")

        result = _BINARY_OPS[op_type](left, right)

        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
            raise SafeMathError("Result is NaN or infinite")

        if abs(result) > _MAX_VALUE:
            raise SafeMathError(f"Result too large: {result}")

        return result

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise SafeMathError(f"Unsupported unary operator: {op_type.__name__}")

        operand = _eval_node(node.operand)
        return _UNARY_OPS[op_type](operand)

    elif isinstance(node, ast.Name):
        # Allow math constants like pi, e
        name = node.id.lower()
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        raise SafeMathError(f"Unknown variable: {node.id}")

    elif isinstance(node, ast.Call):
        # Allow safe math functions
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in _MATH_FUNCS:
                args = [_eval_node(arg) for arg in node.args]
                try:
                    result = _MATH_FUNCS[func_name](*args)
                    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
                        raise SafeMathError("Result is NaN or infinite")
                    return result
                except (ValueError, TypeError) as e:
                    raise SafeMathError(f"Math function error: {e}")
        raise SafeMathError(f"Function calls not allowed")

    else:
        raise SafeMathError(f"Unsupported expression type: {type(node).__name__}")


def safe_math_eval(expression: str) -> Optional[Union[int, float]]:
    """
    Safely evaluate a mathematical expression string.

    Args:
        expression: A string containing a mathematical expression
                   (e.g., "2 + 3 * 4", "sqrt(16)", "2 ** 10")

    Returns:
        The numeric result of the expression, or None if evaluation fails

    Raises:
        SafeMathError: If the expression contains unsafe operations

    Examples:
        >>> safe_math_eval("2 + 3")
        5
        >>> safe_math_eval("10 / 2")
        5.0
        >>> safe_math_eval("2 ** 10")
        1024
        >>> safe_math_eval("sqrt(16)")
        4.0
        >>> safe_math_eval("__import__('os')")  # Raises SafeMathError
    """
    if not expression or not isinstance(expression, str):
        return None

    # Basic sanitization - remove whitespace
    expression = expression.strip()

    if not expression:
        return None

    # Length limit to prevent DoS
    if len(expression) > 1000:
        raise SafeMathError("Expression too long")

    # Block obvious code injection attempts
    dangerous_patterns = [
        '__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
        'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr', 'type', 'class',
        'lambda', 'def ', 'return', 'yield', 'async', 'await',
        'for ', 'while ', 'if ', 'else', 'try', 'except', 'finally',
        'with ', 'assert', 'raise', 'pass', 'break', 'continue',
    ]

    expr_lower = expression.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            raise SafeMathError(f"Dangerous pattern detected: {pattern}")

    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')

        # Evaluate the AST safely
        result = _eval_node(tree)

        return result

    except SyntaxError as e:
        raise SafeMathError(f"Invalid expression syntax: {e}")
    except SafeMathError:
        raise
    except Exception as e:
        raise SafeMathError(f"Evaluation error: {e}")


def safe_math_eval_str(expression: str) -> Optional[str]:
    """
    Safely evaluate a mathematical expression and return result as string.

    This is a convenience wrapper that returns the result as a string,
    matching the original eval() usage pattern in language_brain.py.

    Args:
        expression: A string containing a mathematical expression

    Returns:
        The string representation of the result, or None if evaluation fails
    """
    try:
        result = safe_math_eval(expression)
        if result is not None:
            # Format integers without decimal point
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        return None
    except SafeMathError:
        return None
    except Exception:
        return None
