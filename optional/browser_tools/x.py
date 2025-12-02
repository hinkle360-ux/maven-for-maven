# optional/browser_tools/x.py — UNIFIED X.COM TOOL WITH PLAYWRIGHT
"""
One tool for everything on X.com — post, reply, like, DM, Grok.
Properly integrated with Playwright browser runtime.

Endpoints used:
  /open - Open a URL
  /wait_for - Wait for element to appear
  /click - Click element
  /type - Type into element
  /get_page - Get page content
"""
import json
import re
import time
import urllib.request
from pathlib import Path

BROWSER_URL = "http://127.0.0.1:8765"
SESSION_FILE = Path("x_session.json")
DEBUG = True  # Set to True to see what's happening

# Cached page_id for window reuse
_cached_page_id = None


def _post(endpoint: str, data: dict) -> dict:
    """Make POST request to browser runtime."""
    if DEBUG:
        print(f"[X.PY] POST {endpoint}: {json.dumps(data)[:100]}...")

    req = urllib.request.Request(
        f"{BROWSER_URL}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode())
            if DEBUG:
                status = result.get("status", result.get("error", "unknown"))
                print(f"[X.PY] Response: {status}")
            return result
    except Exception as e:
        if DEBUG:
            print(f"[X.PY] Error: {e}")
        return {"error": str(e), "status": "error"}


def _wait_for(page_id: str, selector: str, timeout_ms: int = 10000) -> bool:
    """Wait for an element to appear on the page."""
    result = _post("/wait_for", {
        "page_id": page_id,
        "selector": selector,
        "timeout_ms": timeout_ms
    })
    return result.get("status") == "success"


def x(command: str) -> str:
    """
    One tool for everything on X.com

    Examples:
        x: grok Hello from Maven
        x: post Hello world
        x: reply @user Great post
    """
    cmd = command.strip().lower()

    # ==== 1. GROK CHAT ====
    if "grok" in cmd:
        global _cached_page_id

        # Use case-insensitive split to handle "Grok", "grok", "GROK" etc.
        parts = re.split(r"grok", command, maxsplit=1, flags=re.IGNORECASE)
        msg = parts[1].strip() if len(parts) > 1 else ""
        # Strip leading colon/punctuation from message
        msg = msg.lstrip(":").strip()
        if not msg:
            msg = "Hello from Maven"

        print(f"[X.PY] Sending to Grok: {msg[:50]}...")

        # Try to reuse existing page, otherwise open new one
        page_id = None
        if _cached_page_id:
            print(f"[X.PY] Trying cached page_id: {_cached_page_id}")
            # Test if page is still valid by trying to get it
            test = _post("/get_page", {"page_id": _cached_page_id, "include_text": False})
            if test.get("status") != "error" and "error" not in test:
                page_id = _cached_page_id
                print(f"[X.PY] Reusing existing browser window")
                # Navigate to Grok if not already there
                _post("/open", {"url": "https://x.com/i/grok", "page_id": page_id})
            else:
                print(f"[X.PY] Cached page invalid, opening new window")
                _cached_page_id = None

        if not page_id:
            # Open new Grok page
            result = _post("/open", {"url": "https://x.com/i/grok"})
            page_id = result.get("page_id")
            if not page_id:
                return f"Failed to open Grok: {result}"
            _cached_page_id = page_id
            print(f"[X.PY] Opened new window, page_id: {page_id}")

        # Wait for the chat input to appear
        # PRIORITY ORDER: textarea[placeholder] works best, try it first!
        input_selectors = [
            "textarea[placeholder]",
            "textarea",
            "div[contenteditable='true']",
            "div[role='textbox']",
            "div[data-testid='tweetTextarea_0']",
            "div[data-testid='dmComposerTextInput']",
        ]

        found_selector = None
        for selector in input_selectors:
            print(f"[X.PY] Waiting for selector: {selector}")
            if _wait_for(page_id, selector, timeout_ms=5000):
                found_selector = selector
                print(f"[X.PY] Found input with: {selector}")
                break

        if not found_selector:
            # Get page content to debug
            page = _post("/get_page", {"page_id": page_id, "include_text": True})
            page_text = page.get("text", "")[:500]
            return f"Could not find chat input. Page shows: {page_text}"

        # Click to focus
        click_result = _post("/click", {"page_id": page_id, "selector": found_selector})
        if click_result.get("status") != "success":
            print(f"[X.PY] Click failed: {click_result}")

        time.sleep(0.5)

        # Type the message with submit=True to press Enter
        type_result = _post("/type", {
            "page_id": page_id,
            "selector": found_selector,
            "text": msg,
            "submit": True  # This presses Enter after typing
        })

        if type_result.get("status") != "success":
            return f"Failed to type message: {type_result}"

        print("[X.PY] Message sent, waiting for Grok response...")

        # Wait for response
        time.sleep(12)

        # Get page and extract reply
        page = _post("/get_page", {"page_id": page_id, "include_text": True})
        text = page.get("text", "")

        if DEBUG:
            print(f"[X.PY] Page text length: {len(text)}")
            print(f"[X.PY] Page text preview: {text[:300]}...")

        # Extract Grok's reply - find text after our message
        # Method 1: Find our message and get everything after it
        if msg in text:
            parts = text.split(msg)
            if len(parts) > 1:
                after_msg = parts[-1].strip()
                # Clean up - remove nav elements and suggestion buttons
                junk_patterns = [
                    "Explore", "Home", "Notifications", "Messages", "Premium",
                    "Profile", "Post", "Lists", "Bookmarks", "Communities",
                    "More", "See new posts", "Share xAI", "Discuss SpaceX",
                    "View keyboard", "To view keyboard", "Grok 4.1", "Beta"
                ]
                for junk in junk_patterns:
                    if junk in after_msg:
                        after_msg = after_msg.split(junk)[0]
                after_msg = after_msg.strip()
                # Get all meaningful response lines (not just first line!)
                lines = [l.strip() for l in after_msg.splitlines() if l.strip()]
                if lines:
                    # Return ALL response lines joined together
                    response = "\n".join(lines)
                    if len(response) > 3:
                        print(f"[X.PY] Extracted reply ({len(lines)} lines, {len(response)} chars)")
                        return response

        # Method 2: Look for lines after "Grok" marker
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        grok_idx = -1
        for i, line in enumerate(lines):
            if "Grok" in line:
                grok_idx = i

        if grok_idx >= 0 and grok_idx < len(lines) - 1:
            # Get lines after the last "Grok" line
            reply_lines = []
            for line in lines[grok_idx + 1:]:
                # Stop at navigation elements
                if any(nav in line for nav in ["Explore", "Home", "Notifications", "Messages", "Premium", "Post", "Lists"]):
                    break
                reply_lines.append(line)

            if reply_lines:
                reply = "\n".join(reply_lines).strip()
                if reply and len(reply) > 5:
                    print(f"[X.PY] Extracted reply (method 2): {reply[:100]}...")
                    return reply

        # Method 3: Just return the middle section of text (skip nav)
        if len(lines) > 5:
            # Skip first few lines (nav) and last few lines (nav)
            middle = lines[3:-3]
            reply = "\n".join(middle).strip()
            if reply and len(reply) > 20:
                print(f"[X.PY] Extracted reply (method 3): {reply[:100]}...")
                return reply

        return f"Grok's response (raw): {text[:500]}"

    # ==== 2. POST TWEET ====
    if cmd.startswith("post:") or cmd.startswith("post "):
        # Handle both "post:message" and "post message" formats
        if ":" in command and command.lower().startswith("post:"):
            tweet_text = command.split(":", 1)[1].strip()
        else:
            tweet_text = command[5:].strip()

        if not tweet_text:
            return "No tweet text provided. Use: post <your message>"

        print(f"[X.PY] Posting tweet: {tweet_text[:50]}...")

        result = _post("/open", {"url": "https://x.com/compose/post"})
        page_id = result.get("page_id")
        if not page_id:
            return f"Failed to open composer: {result}"

        # Wait for compose box
        selector = "div[data-testid='tweetTextarea_0']"
        if not _wait_for(page_id, selector, timeout_ms=5000):
            selector = "div[contenteditable='true']"
            _wait_for(page_id, selector, timeout_ms=3000)

        # Type and submit
        _post("/click", {"page_id": page_id, "selector": selector})
        time.sleep(0.3)

        _post("/type", {
            "page_id": page_id,
            "selector": selector,
            "text": tweet_text
        })

        # Click the Post button - try multiple selectors
        time.sleep(1.0)  # Wait for button to become active

        post_selectors = [
            "button[data-testid='tweetButton']",
            "button[data-testid='tweetButtonInline']",
            "div[data-testid='tweetButton']",
            "button[type='button'][tabindex='0']",  # Generic post button
            "button:has-text('Post')",
        ]

        post_btn = None
        for btn_selector in post_selectors:
            post_btn = _post("/click", {
                "page_id": page_id,
                "selector": btn_selector
            })
            if post_btn.get("status") == "success":
                print(f"[X.PY] Clicked Post button with selector: {btn_selector}")
                break

        if post_btn and post_btn.get("status") == "success":
            time.sleep(1.0)  # Wait for post to complete
            return f"Posted: {tweet_text[:60]}..."
        else:
            # Try keyboard shortcut as fallback (Ctrl+Enter to post)
            print("[X.PY] Button click failed, trying Ctrl+Enter...")
            _post("/type", {
                "page_id": page_id,
                "selector": selector,
                "text": "",
                "key": "Control+Enter"
            })
            time.sleep(1.0)
            return f"Attempted to post: {tweet_text[:60]}... (check browser)"

    # ==== DEFAULT ====
    _post("/open", {"url": "https://x.com/home"})
    return "X.com opened — commands: grok <message>, post <text>"


# For CLI testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print(x(cmd))
    else:
        print("Usage: python x.py <command>")
        print("  python x.py 'grok Hello'")
        print("  python x.py 'post Hello world'")
