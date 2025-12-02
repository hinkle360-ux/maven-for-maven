"""
Tool Injection Helper
=====================

This module provides a centralized way to inject host-provided tools into
all brain facade modules. The host runtime calls inject_tools() once at
startup to wire all tools into the brain layer.

Usage (in host runtime):
    from host_tools.factory import create_host_tools
    from brains.tool_injection import inject_tools

    # Create tools
    tools = create_host_tools(
        enable_web=True,
        enable_llm=True,
        enable_shell=True,
        enable_git=True,
        enable_sandbox=True,
        root_dir="/path/to/maven"
    )

    # Inject into all brain facades
    inject_tools(tools)

After injection, all brain facade modules will use the real tool implementations
instead of null/stub implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brains.tools_api import ToolRegistry


def inject_tools(registry: "ToolRegistry") -> None:
    """
    Inject the tool registry into all brain facade modules.

    This function sets the global tool registry in each facade module,
    enabling them to use real tool implementations instead of null stubs.

    Args:
        registry: The ToolRegistry instance with concrete tool implementations
    """
    # Inject into external_interfaces/web_client.py
    try:
        from brains.external_interfaces import web_client
        web_client.set_tool_registry(registry)
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to inject into web_client: {e}")

    # Inject into tools/llm_service.py
    try:
        from brains.tools import llm_service
        llm_service.set_tool_registry(registry)
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to inject into llm_service: {e}")

    # Inject into tools/git_tool.py
    try:
        from brains.tools import git_tool
        git_tool.set_tool_registry(registry)
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to inject into git_tool: {e}")

    # Inject into agent/tools/shell_tool.py
    try:
        from brains.agent.tools import shell_tool
        shell_tool.set_tool_registry(registry)
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to inject into shell_tool: {e}")

    # Inject into cognitive/action_engine/service/action_engine.py
    try:
        from brains.cognitive.action_engine.service import action_engine
        action_engine.set_tool_registry(registry)
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to inject into action_engine: {e}")

    # Inject into cognitive/self_model/continuous_introspector.py
    try:
        from brains.cognitive.self_model import continuous_introspector
        continuous_introspector.set_tool_registry(registry)
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to inject into continuous_introspector: {e}")

    # Register optional browser tools (ChatGPT, Grok, etc.)
    _register_browser_tools(registry)

    print("[TOOL_INJECTION] Tool injection complete")


def _register_browser_tools(registry: "ToolRegistry") -> None:
    """
    Register optional browser-based tools.

    Add your browser tools here:
        from optional.browser_tools.grok_tool import grok_conversation
        registry.register_browser_tool("grok", grok_conversation)
    """
    # General browser URL opener (from browser_runtime)
    try:
        from optional.browser_runtime.browser_client import open_url as browser_open_url
        registry.register_browser_tool("browser_open", browser_open_url)
        print("[TOOL_INJECTION] Registered browser tool: browser_open")
    except ImportError:
        pass  # Browser runtime not available
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register browser_open tool: {e}")

    # Grok tool (async)
    try:
        from optional.browser_tools.grok_tool import grok_conversation
        registry.register_browser_tool("grok", grok_conversation)
        print("[TOOL_INJECTION] Registered browser tool: grok")
    except ImportError:
        pass  # Tool not implemented yet
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register grok tool: {e}")

    # Grok URL opener (sync)
    try:
        from optional.browser_tools.grok_tool import grok_open_url
        registry.register_browser_tool("grok_url", grok_open_url)
        print("[TOOL_INJECTION] Registered browser tool: grok_url")
    except ImportError:
        pass
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register grok_url tool: {e}")

    # Grok message sender
    try:
        from optional.browser_tools.grok_tool import grok_send_message
        registry.register_browser_tool("grok_send", grok_send_message)
        print("[TOOL_INJECTION] Registered browser tool: grok_send")
    except ImportError:
        pass
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register grok_send tool: {e}")

    # Grok page text getter
    try:
        from optional.browser_tools.grok_tool import grok_get_page_text
        registry.register_browser_tool("grok_text", grok_get_page_text)
        print("[TOOL_INJECTION] Registered browser tool: grok_text")
    except ImportError:
        pass
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register grok_text tool: {e}")

    # ChatGPT tool
    try:
        from optional.browser_tools.chatgpt_tool import chatgpt_conversation
        registry.register_browser_tool("chatgpt", chatgpt_conversation)
        print("[TOOL_INJECTION] Registered browser tool: chatgpt")
    except ImportError:
        pass  # Tool not implemented yet
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register chatgpt tool: {e}")

    # Local CAPTCHA solver
    try:
        from optional.browser_tools.local_captcha_solver import solve_captcha
        registry.register_browser_tool("captcha_solver", solve_captcha)
        print("[TOOL_INJECTION] Registered browser tool: captcha_solver")
    except ImportError:
        pass  # Tool not implemented yet
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register captcha_solver tool: {e}")

    # THE ONE X TOOL - does everything on X.com
    try:
        from optional.browser_tools.x import x
        registry.register_browser_tool("x", x)
        print("[TOOL_INJECTION] Registered browser tool: x (unified X.com tool)")
    except ImportError:
        pass
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register x tool: {e}")

    # HUMAN TOOL - real desktop control with pyautogui
    try:
        from optional.browser_tools.human_tool import human
        registry.register_browser_tool("human", human)
        print("[TOOL_INJECTION] Registered browser tool: human (desktop control)")
    except ImportError:
        pass
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register human tool: {e}")

    # PC CONTROL TOOL - system monitoring, cleanup, security (like CCleaner/Afterburner)
    try:
        from optional.browser_tools.pc_control_tool import pc
        registry.register_browser_tool("pc", pc)
        print("[TOOL_INJECTION] Registered browser tool: pc (system control)")
    except ImportError:
        pass
    except Exception as e:
        print(f"[TOOL_INJECTION] Failed to register pc tool: {e}")


def get_available_tools(registry: "ToolRegistry") -> dict:
    """
    Get a summary of available tools in the registry.

    Returns:
        Dict with tool names and their availability status
    """
    result = {
        "web_search": registry.web_search is not None,
        "web_fetch": registry.web_fetch is not None,
        "llm": registry.llm is not None,
        "shell": registry.shell is not None,
        "git": registry.git is not None,
        "python_sandbox": registry.python_sandbox is not None,
    }
    # Add browser tools
    for name in registry.browser_tools:
        result[f"browser:{name}"] = True
    return result
