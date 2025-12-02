"""
ChatGPT Browser Tool - Placeholder
==================================

TODO: Implement browser-based ChatGPT interaction.

This tool would:
1. Connect to the browser runtime on port 8765
2. Navigate to chatgpt.com
3. Handle login/session persistence
4. Send messages and retrieve responses

Dependencies:
- maven_browser_client (from host_tools/browser_runtime/)
- Browser runtime running on port 8765

Usage:
    from optional.browser_tools.chatgpt_tool import chatgpt_conversation

    response = await chatgpt_conversation([
        {"role": "user", "content": "Hello"}
    ])

Session persistence:
- Save cookies/localStorage to chatgpt_session.json
- Restore on subsequent runs to avoid re-login

Implementation notes:
- Check for login page and prompt user if needed
- Wait for response completion before returning
- Handle rate limits and errors gracefully
"""

# Placeholder - implement your browser automation here

async def chatgpt_conversation(messages: list) -> str:
    """
    Send messages to ChatGPT via browser automation.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts

    Returns:
        The assistant's response text

    Raises:
        NotImplementedError: This is a placeholder
    """
    raise NotImplementedError(
        "ChatGPT browser tool not implemented. "
        "See docstring for implementation notes."
    )
