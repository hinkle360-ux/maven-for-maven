"""
Agency Tool Executor
===================

Executes agency tool calls directly without going through Teacher.
This prevents the echo bug by providing direct tool access.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import importlib
import json


def execute_agency_tool(
    tool_path: str,
    method_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    pattern_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute an agency tool directly.

    Args:
        tool_path: Full import path to tool (e.g., "brains.tools.self_introspection.get_self_introspection")
                   Special values: "browser_runtime" for browser automation
        method_name: Optional method to call on the returned object
        args: Optional arguments to pass to the method
        pattern_info: Optional pattern info dict with not_implemented_message etc.

    Returns:
        Dictionary with execution result
    """
    result = {
        "status": "unknown",
        "tool": tool_path,
        "method": method_name,
        "output": None,
        "error": None
    }

    # Handle NOT_IMPLEMENTED tools
    if tool_path == "NOT_IMPLEMENTED":
        not_impl_msg = "This capability is not implemented yet."
        if pattern_info and pattern_info.get("not_implemented_message"):
            not_impl_msg = pattern_info["not_implemented_message"]
        result["status"] = "not_implemented"
        result["output"] = not_impl_msg
        result["error"] = None
        return result

    # Handle browser_runtime special tool
    if tool_path == "browser_runtime":
        return _execute_browser_tool(method_name, args, result)

    # Handle time_now special tool - provides real-time clock access
    if tool_path == "time_now":
        return _execute_time_tool(method_name, args, result)

    # Handle shell_tool - execute shell commands
    if tool_path == "shell_tool":
        return _execute_shell_tool(method_name, args, result)

    # Handle all other standard Python module tools
    return _execute_standard_tool(tool_path, method_name, args, result)


def _execute_time_tool(
    method_name: Optional[str],
    args: Optional[Dict[str, Any]],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute time_now tool to get current time from system clock.

    CRITICAL: This tool provides the ONLY accurate source of time information.
    The LLM/Teacher cannot provide real-time information.

    Args:
        method_name: Time operation (GET_TIME, GET_DATE, GET_CALENDAR, HEALTH)
        args: Optional arguments (e.g., format_24h, query_type)
        result: Result dict to populate

    Returns:
        Populated result dict with current time/date/calendar info
    """
    try:
        from brains.agent.tools.time_now import service_api as time_api, what_time_is_it

        method = (method_name or "GET_TIME").upper()
        args = args or {}
        query_type = args.get("query_type", "time")

        if method in ("GET_TIME", "GET_DATE", "GET_CALENDAR"):
            # Get full time information (all methods return same comprehensive data)
            time_result = time_api({"op": method, "payload": args})

            if time_result.get("ok"):
                payload = time_result.get("payload", {})
                result["status"] = "success"
                result["output"] = payload

                # Format response based on query type
                if query_type == "date":
                    # Date query: show date and day of week
                    date_str = payload.get("date", "")  # e.g., "Monday, December 01, 2025"
                    day_of_week = payload.get("day_of_week", "")
                    result["formatted_response"] = f"Today is {date_str}."
                elif query_type == "calendar":
                    # Calendar query: show month, year, and date
                    month = payload.get("month", "")
                    year = payload.get("year", "")
                    day = payload.get("day", "")
                    week_number = payload.get("week_number", "")
                    day_of_week = payload.get("day_of_week", "")
                    result["formatted_response"] = f"Today is {day_of_week}, {month} {day}, {year} (week {week_number})."
                else:
                    # Time query: show time
                    result["formatted_response"] = payload.get("formatted", what_time_is_it())

                print(f"[TIME_TOOL] Executed {method}: {result['formatted_response']}")
            else:
                result["status"] = "error"
                result["error"] = time_result.get("error", {}).get("message", "Unknown error")

        elif method == "HEALTH":
            # Health check
            health_result = time_api({"op": "HEALTH"})
            result["status"] = "success" if health_result.get("ok") else "error"
            result["output"] = health_result.get("payload", {})

        else:
            result["status"] = "error"
            result["error"] = f"Unknown time tool method: {method}"

        return result

    except ImportError:
        result["status"] = "error"
        result["error"] = "Time tool module not found. Check brains/agent/tools/time_now.py"
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Time tool execution failed: {str(e)}"
        return result


def _execute_shell_tool(
    method_name: Optional[str],
    args: Optional[Dict[str, Any]],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute shell commands via the shell tool.

    Args:
        method_name: Not used for shell (command comes from args)
        args: Dict with 'command' or 'shell_command' key
        result: Result dict to populate

    Returns:
        Populated result dict
    """
    try:
        from brains.agent.tools.shell_tool import run as shell_run

        # Get command from args
        command = None
        if args:
            command = args.get("command") or args.get("shell_command") or args.get("cmd")

        if not command:
            result["status"] = "error"
            result["error"] = "No command provided for shell execution"
            return result

        print(f"[SHELL_TOOL] Executing: {command}")

        # Execute the command
        shell_result = shell_run(command)

        result["status"] = shell_result.get("status", "unknown")
        result["exit_code"] = shell_result.get("exit_code", -1)
        result["output"] = shell_result.get("stdout", "")
        result["stderr"] = shell_result.get("stderr", "")
        result["error"] = shell_result.get("error")

        # Format output for display - show output even if exit_code != 0
        # Commands like git push can have exit_code=1 but still have useful output
        stdout = result["output"] or ""
        stderr = result["stderr"] or ""

        if result["exit_code"] == 0 or result["status"] in ("success", "completed"):
            # Success - show stdout, or stderr if no stdout
            result["status"] = "success"  # Normalize status
            result["formatted_response"] = stdout or stderr or "(command completed successfully)"
        else:
            # Error - show both stderr and any stdout
            error_msg = result["error"] or stderr or "Command failed"
            if stdout:
                result["formatted_response"] = f"{stdout}\n\nError: {error_msg}"
            else:
                result["formatted_response"] = f"Error: {error_msg}"

        print(f"[SHELL_TOOL] Result: status={result['status']}, exit_code={result.get('exit_code')}")
        return result

    except ImportError as e:
        result["status"] = "error"
        result["error"] = f"Shell tool not available: {e}"
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Shell execution failed: {str(e)}"
        return result


def _execute_browser_tool(
    method_name: Optional[str],
    args: Optional[Dict[str, Any]],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute browser runtime operations via the HTTP API.

    Args:
        method_name: Browser operation (OPEN_URL, SEARCH, etc.)
        args: Arguments for the operation
        result: Result dict to populate

    Returns:
        Populated result dict
    """
    try:
        from optional.browser_runtime.browser_client import is_available, open_url, get_server_url

        if not is_available():
            result["status"] = "error"
            result["error"] = (
                "Browser runtime server not available. "
                "Start it with: python run_browser_server.py"
            )
            return result

        method = (method_name or "").upper()

        if method in ("OPEN_URL", "BROWSE", "FETCH_URL"):
            # Extract URL from args
            url = None
            if args:
                url = args.get("url") or args.get("target_url") or args.get("text")
            if not url:
                result["status"] = "error"
                result["error"] = "No URL provided for browser operation"
                return result

            # Call browser runtime
            browser_result = open_url(url)
            if browser_result.get("error"):
                result["status"] = "error"
                result["error"] = browser_result["error"]
            else:
                result["status"] = "success"
                result["output"] = browser_result
        elif method == "SEARCH":
            # For search, construct a Google search URL
            query = args.get("query") or args.get("text") if args else None
            if not query:
                result["status"] = "error"
                result["error"] = "No search query provided"
                return result
            import urllib.parse
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            browser_result = open_url(search_url)
            if browser_result.get("error"):
                result["status"] = "error"
                result["error"] = browser_result["error"]
            else:
                result["status"] = "success"
                result["output"] = browser_result
        else:
            result["status"] = "error"
            result["error"] = f"Unknown browser method: {method_name}"

    except ImportError as e:
        result["status"] = "error"
        result["error"] = f"Browser runtime not available: {e}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Browser execution failed: {e}"

    return result


def _execute_standard_tool(
    tool_path: str,
    method_name: Optional[str],
    args: Optional[Dict[str, Any]],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a standard Python module-based tool."""

    try:
        # Parse tool path
        parts = tool_path.rsplit('.', 1)
        if len(parts) != 2:
            result["status"] = "error"
            result["error"] = f"Invalid tool path: {tool_path}"
            return result

        module_path, function_name = parts

        # Import module
        module = importlib.import_module(module_path)

        # Check if this is a service_api based brain (e.g., inventory_brain, sensorium_brain, coder_brain)
        if hasattr(module, 'service_api') and method_name and method_name.isupper():
            # This is a brain service API call
            service_api = getattr(module, 'service_api')
            api_msg = {"op": method_name, "payload": args or {}}
            api_result = service_api(api_msg)

            if api_result.get("ok"):
                result["status"] = "success"
                result["output"] = api_result.get("payload", api_result)
            else:
                result["status"] = "error"
                error_info = api_result.get("error", {})
                result["error"] = error_info.get("message", str(error_info))
            return result

        if not hasattr(module, function_name):
            result["status"] = "error"
            result["error"] = f"Function {function_name} not found in {module_path}"
            return result

        # Get function
        func = getattr(module, function_name)

        # Call function (usually returns an instance or result)
        if args and not method_name:
            # Direct function call with args
            output = func(**args)
        else:
            # Get instance/object
            output = func()

        # If method_name specified, call method on the result
        if method_name:
            if not hasattr(output, method_name):
                result["status"] = "error"
                result["error"] = f"Method {method_name} not found on {function_name} result"
                return result

            method = getattr(output, method_name)
            if args:
                output = method(**args)
            else:
                output = method()

        # Format output for response
        result["status"] = "success"
        result["output"] = _format_output(output)

    except ImportError as e:
        result["status"] = "error"
        result["error"] = f"Import error: {str(e)}"
    except TypeError as e:
        result["status"] = "error"
        result["error"] = f"Type error (wrong arguments?): {str(e)}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Execution error: {str(e)}"

    return result


def _format_output(output: Any) -> Any:
    """
    Format tool output for human-readable response.

    Args:
        output: Raw tool output

    Returns:
        Formatted output (dict, str, or original)
    """
    # If it's a dict, return as-is (structured data)
    if isinstance(output, dict):
        return output

    # If it's a list of dicts, return as-is
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return output

    # If it's a simple type, convert to string
    if isinstance(output, (str, int, float, bool)):
        return output

    # Try to convert to dict/json
    try:
        if hasattr(output, '__dict__'):
            return output.__dict__
    except Exception:
        pass

    # Fall back to string representation
    return str(output)


def format_agency_response(query: str, tool_result: Dict[str, Any]) -> str:
    """
    Format an agency tool result into a natural language response.

    Args:
        query: Original user query
        tool_result: Result from execute_agency_tool

    Returns:
        Formatted response string
    """
    # Handle not_implemented status (like browser)
    if tool_result["status"] == "not_implemented":
        return tool_result.get("output", "This capability is not implemented yet.")

    if tool_result["status"] != "success":
        return f"Error executing tool: {tool_result.get('error', 'Unknown error')}"

    output = tool_result["output"]

    # Handle different output types
    if isinstance(output, dict):
        return _format_dict_response(query, output, tool_result)
    elif isinstance(output, list):
        return _format_list_response(query, output, tool_result)
    elif isinstance(output, str):
        return output
    else:
        return str(output)


def _format_dict_response(query: str, output: Dict[str, Any], tool_result: Dict[str, Any]) -> str:
    """Format dictionary output into readable response."""
    tool_name = tool_result.get("method", tool_result.get("tool", "tool"))

    # Special handling for common report types
    if "brains" in output and "total" in output:
        # Inventory brain result
        return _format_inventory_result(output)
    elif "total_brains" in output:
        # Brain analysis report
        return _format_brain_analysis(output)
    elif "total_files" in output and "total_lines" in output:
        # Architecture scan report
        return _format_architecture_scan(output)
    elif "circular_dependencies" in output:
        # Dependency analysis
        return _format_dependency_analysis(output)
    elif "branch" in output and "commit" in output:
        # Git repo info
        return _format_git_info(output)
    elif "normalized_text" in output or "raw_text" in output:
        # Normalization introspection result
        return _format_normalization_result(output)
    else:
        # Generic dict formatting
        return f"Result from {tool_name}:\n{json.dumps(output, indent=2)}"


def _format_list_response(query: str, output: list, tool_result: Dict[str, Any]) -> str:
    """Format list output into readable response."""
    if not output:
        return "No results found."

    tool_name = tool_result.get("method", tool_result.get("tool", "tool"))

    # Check if list of dicts
    if output and isinstance(output[0], dict):
        # Format as table/list
        lines = [f"Found {len(output)} results:"]
        for i, item in enumerate(output[:10], 1):  # Show first 10
            if "name" in item:
                lines.append(f"  {i}. {item['name']}")
            elif "path" in item:
                lines.append(f"  {i}. {item['path']}")
            else:
                lines.append(f"  {i}. {json.dumps(item)}")

        if len(output) > 10:
            lines.append(f"  ... and {len(output) - 10} more")

        return "\n".join(lines)
    else:
        # Simple list
        return f"Results ({len(output)} items):\n" + "\n".join(f"  - {item}" for item in output[:20])


def _format_brain_analysis(output: Dict[str, Any]) -> str:
    """Format brain analysis output."""
    total = output.get("total_brains", 0)
    compliant = output.get("compliant", [])
    non_compliant = output.get("non_compliant", [])

    lines = [
        f"**Brain Analysis Report**",
        f"Total Brains: {total}",
        f"Compliant: {len(compliant)} ({len(compliant)/total*100:.1f}%)",
        f"Non-Compliant: {len(non_compliant)} ({len(non_compliant)/total*100:.1f}%)",
    ]

    if non_compliant:
        lines.append("\n**Non-Compliant Brains (first 10):**")
        missing_methods = output.get("missing_methods", {})
        for brain in non_compliant[:10]:
            missing = missing_methods.get(brain, [])
            lines.append(f"  ❌ {brain}: missing {missing}")

    return "\n".join(lines)


def _format_architecture_scan(output: Dict[str, Any]) -> str:
    """Format architecture scan output."""
    lines = [
        f"**Architecture Scan Complete**",
        f"Total Files: {output.get('total_files', 0)}",
        f"Total Lines: {output.get('total_lines', 0):,}",
        f"Directories: {output.get('dir_count', 0)}",
        f"Total Size: {output.get('total_size', 0) / 1024 / 1024:.2f} MB",
    ]

    if "brains" in output:
        lines.append(f"Brains Found: {len(output['brains'])}")

    return "\n".join(lines)


def _format_dependency_analysis(output: Dict[str, Any]) -> str:
    """Format dependency analysis output."""
    cycles = output.get("circular_dependencies", [])
    external = output.get("external_dependencies", [])

    lines = [
        f"**Dependency Analysis**",
        f"Circular Dependencies: {len(cycles)} cycles detected",
        f"External Dependencies: {len(external)} packages",
    ]

    if cycles:
        lines.append("\n**Circular Dependency Chains (first 5):**")
        for i, cycle in enumerate(cycles[:5], 1):
            chain = " → ".join(cycle)
            lines.append(f"  {i}. {chain}")

    return "\n".join(lines)


def _format_git_info(output: Dict[str, Any]) -> str:
    """Format git repository info output."""
    lines = [
        f"**Git Repository Status**",
        f"Branch: {output.get('branch', 'unknown')}",
        f"Commit: {output.get('commit', 'unknown')[:8]}",
        f"Status: {'Clean' if output.get('is_clean', False) else 'Has changes'}",
    ]

    if output.get("ahead", 0) > 0:
        lines.append(f"Ahead: {output['ahead']} commits")
    if output.get("behind", 0) > 0:
        lines.append(f"Behind: {output['behind']} commits")

    if output.get("staged_files", 0) > 0:
        lines.append(f"Staged: {output['staged_files']} files")
    if output.get("modified_files", 0) > 0:
        lines.append(f"Modified: {output['modified_files']} files")
    if output.get("untracked_files", 0) > 0:
        lines.append(f"Untracked: {output['untracked_files']} files")

    return "\n".join(lines)


def _format_inventory_result(output: Dict[str, Any]) -> str:
    """Format inventory brain result."""
    brains = output.get("brains", [])
    total = output.get("total", len(brains))
    unclassified = output.get("unclassified", [])

    lines = [
        f"**Cognitive Brain Inventory**",
        f"Total brains found: {total}",
        "",
    ]

    if brains:
        lines.append("**Brains:**")
        for i, brain in enumerate(brains, 1):
            name = brain.get("name", "unknown")
            purpose = brain.get("purpose", "")[:60]
            has_memory = "✓" if brain.get("has_memory", False) else "✗"
            lines.append(f"  {i}. {name} (memory: {has_memory})")
            if purpose:
                lines.append(f"      {purpose}")

    if unclassified:
        lines.append("")
        lines.append(f"**Unclassified items:** {len(unclassified)}")
        for item in unclassified[:5]:
            lines.append(f"  - {item.get('name', 'unknown')}")

    return "\n".join(lines)


def _format_normalization_result(output: Dict[str, Any]) -> str:
    """Format normalization introspection result."""
    raw_text = output.get("raw_text", "")
    normalized_text = output.get("normalized_text", "")
    norm_type = output.get("norm_type", "unknown")
    tokens = output.get("tokens", [])

    lines = [
        f"**Last Normalized User Message**",
        f"",
        f"Raw input: {raw_text}",
        f"",
        f"Normalized: {normalized_text}",
        f"",
        f"Classification: {norm_type}",
    ]

    if tokens:
        lines.append(f"")
        lines.append(f"Tokens: {tokens}")

    return "\n".join(lines)
