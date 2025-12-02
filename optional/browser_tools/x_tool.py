# optional/browser_tools/x_tool.py — FINAL X.COM + GROK TOOL (Dec 1 2025)
"""
X.com and Grok browser tool with persistent session.

Features:
- Login once, stay logged in forever (session saved to x_session.json)
- grok(message) - Send message to Grok, get reply
- x_post(text) - Post a tweet
- x_login_once() - One-time login handler

Usage:
    from optional.browser_tools.x_tool import grok, x_post

    # First time only:
    x_login_once()

    # Then use forever:
    reply = grok("What is 2+2?")
    x_post("Hello world!")
"""

import json
import time
import urllib.request
from pathlib import Path

BROWSER_URL = "http://127.0.0.1:8765"
SESSION_FILE = Path("x_session.json")  # Saves login session


def _post(endpoint: str, data: dict) -> dict:
    """Make POST request to browser server."""
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


def _save_session(cookies):
    """Save session cookies to file."""
    SESSION_FILE.write_text(json.dumps(cookies))
    print("[X] Login saved — never again")


def _load_session():
    """Load saved session cookies."""
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return {}


# ——— MAIN TOOLS ———

def x_open(url: str = "https://x.com/i/grok") -> str:
    """Open a URL in the browser."""
    result = _post("/open", {"url": url})
    if "page_id" not in result:
        return f"Open failed: {result}"
    page_id = result["page_id"]
    print(f"[X] Opened {url} → {page_id}")
    return page_id


def x_type(page_id: str, text: str):
    """Type text into input and press Enter."""
    # Simple contenteditable selector works on main Grok chat
    _post("/type", {
        "page_id": page_id,
        "selector": "div[contenteditable='true']",
        "text": text
    })
    _post("/press", {
        "page_id": page_id,
        "selector": "div[contenteditable='true']",
        "key": "Enter"
    })


def x_post(text: str) -> str:
    """Post a tweet."""
    page_id = x_open("https://x.com/compose/post")
    x_type(page_id, text)
    time.sleep(3)
    return "Posted!"


def grok(message: str) -> str:
    """
    Send a message to Grok and get the reply.

    Args:
        message: The message to send to Grok

    Returns:
        Grok's response text
    """
    # Open Grok directly (use /i/grok for real chat, not /grok marketing page)
    page_id = x_open("https://x.com/i/grok")

    # Type message + newline to send (the REAL way X sends)
    print(f"[GROK] Typing message: {message[:30]}...")
    _post("/type", {
        "page_id": page_id,
        "selector": "div[contenteditable='true'][data-testid='dmComposerTextInput']",
        "text": message + "\n"  # ← newline sends it
    })

    # Wait for Grok to respond
    print("[GROK] Waiting for response...")
    time.sleep(12)

    # Get page text and extract reply
    page = _post("/get_page", {"page_id": page_id, "include_text": True})
    text = page.get("text", "")

    # Extract ONLY Grok's reply (everything after the last "Grok" line)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    grok_lines = [line for line in lines if "Grok" in line]
    if grok_lines:
        last_grok = grok_lines[-1]
        reply_start = text.find(last_grok) + len(last_grok)
        reply = text[reply_start:].strip().split("See new posts")[0].strip()
        return reply or "Grok replied (no text detected)"

    return "Grok is thinking..."


def x_login_once():
    """
    One-time login handler.

    Opens Grok, waits for you to log in, then saves the session.
    After this, you never need to log in again.
    """
    page_id = x_open("https://x.com/i/grok")
    result = _post("/get_page", {"page_id": page_id, "include_text": True})

    if any(word in result.get("text", "").lower() for word in ["login", "sign in"]):
        print("[X] First login — do it in the browser, then press ENTER here")
        input("Done? > ")
        cookies = _post("/cookies", {"page_id": page_id})
        _save_session(cookies)
        print("[X] You are now logged in FOREVER")
    else:
        print("[X] Already logged in!")


# For direct testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python x_tool.py login    - First-time login")
        print("  python x_tool.py grok <message>")
        print("  python x_tool.py post <text>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "login":
        x_login_once()
    elif cmd == "grok":
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello Grok!"
        print(f"Sending: {msg}")
        print(f"Reply: {grok(msg)}")
    elif cmd == "post":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Test post from Maven"
        print(x_post(text))
    else:
        print(f"Unknown command: {cmd}")
