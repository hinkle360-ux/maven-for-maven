# optional/browser_tools/grok_tool.py — FINAL, WORKS WITH YOUR CURRENT RUNTIME
"""
Grok Browser Tool - Uses only the endpoints your browser runtime actually has:
  /open, /type, /click, /get_page

NO /wait_for_selector, /get_text, /press needed.
"""
import json
import time
import urllib.request

BROWSER_URL = "http://127.0.0.1:8765"


def _post(endpoint: str, data: dict) -> dict:
    """Make a POST request to the browser runtime server."""
    req = urllib.request.Request(
        f"{BROWSER_URL}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def grok(message: str) -> str:
    """
    Send a message to Grok and get a reply.

    Uses only /open, /type, /get_page - the endpoints your runtime has.

    Args:
        message: The message to send to Grok

    Returns:
        Grok's response text, or status message
    """
    # 1. Open real Grok chat
    result = _post("/open", {"url": "https://x.com/i/grok"})
    if "error" in result and "page_id" not in result:
        return f"Failed to open Grok: {result.get('error')}"
    page_id = result.get("page_id")
    if not page_id:
        return f"No page_id returned: {result}"

    # 2. Type message + send with \n (this is how X really sends)
    _post("/type", {
        "page_id": page_id,
        "selector": "div[contenteditable='true'][data-testid='dmComposerTextInput']",
        "text": message + "\n"
    })

    # 3. Wait and poll with get_page until Grok replies
    for _ in range(25):
        time.sleep(3)
        page = _post("/get_page", {"page_id": page_id, "include_text": True})
        text = page.get("text", "")
        if "Grok" in text and len(text.split("Grok")) > 1:
            reply = text.split("Grok")[-1].strip()
            if len(reply) > 20:
                return reply

    return "Grok is thinking..."
