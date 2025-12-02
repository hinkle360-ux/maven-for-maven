# optional/browser_tools/human_tool.py — REAL HUMAN CONTROL (like ChatGPT agent)
"""
Maven now has real eyes and hands — full desktop control using pyautogui.
No HTTP endpoints, no selectors — just real mouse and keyboard.

Requirements:
    pip install pyautogui opencv-python pillow numpy

Usage:
    human: open Grok and say hello
    human: post on X: Maven is alive
    human: click the chat box and type hello
"""
import time

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.3
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def human(command: str) -> str:
    """
    Maven now has real eyes and hands — full desktop control.

    Examples:
        human: open Grok and say hello
        human: post on X: Maven is alive
        human: click the Grok tab and type We did it
        human: type Hello world
        human: press enter
        human: click 500 300
    """
    if not HAS_PYAUTOGUI:
        return "Error: pyautogui not installed. Run: pip install pyautogui"

    cmd = command.strip().lower()

    # Take screenshot for reference
    if HAS_PIL:
        try:
            screenshot = ImageGrab.grab()
            screenshot.save("screen.png")
        except Exception:
            pass

    # Direct commands
    if cmd.startswith("type "):
        text = command[5:].strip()
        pyautogui.typewrite(text, interval=0.05)
        return f"Typed: {text}"

    if cmd.startswith("press "):
        key = command[6:].strip().lower()
        pyautogui.press(key)
        return f"Pressed: {key}"

    if cmd.startswith("click "):
        parts = command[6:].strip().split()
        if len(parts) >= 2:
            try:
                x, y = int(parts[0]), int(parts[1])
                pyautogui.click(x, y)
                return f"Clicked at ({x}, {y})"
            except ValueError:
                pass
        # Otherwise try to click current position
        pyautogui.click()
        return "Clicked at current position"

    if cmd.startswith("move "):
        parts = command[5:].strip().split()
        if len(parts) >= 2:
            try:
                x, y = int(parts[0]), int(parts[1])
                pyautogui.moveTo(x, y)
                return f"Moved to ({x}, {y})"
            except ValueError:
                pass

    # Grok workflow
    if "grok" in cmd:
        # Extract message after "grok"
        msg = command.split("grok", 1)[1].strip() if "grok" in command.lower() else "Hello from Maven"

        # Open Grok URL
        pyautogui.hotkey("ctrl", "l")  # Focus URL bar
        time.sleep(0.3)
        pyautogui.typewrite("x.com/i/grok", interval=0.02)
        pyautogui.press("enter")
        time.sleep(4)  # Wait for page to load

        # Click in the middle-bottom area where chat input usually is
        screen_width, screen_height = pyautogui.size()
        pyautogui.click(screen_width // 2, screen_height - 150)
        time.sleep(0.5)

        # Type and send
        pyautogui.typewrite(msg, interval=0.03)
        pyautogui.press("enter")

        return f"Sent to Grok: {msg}"

    # Post workflow
    if "post" in cmd:
        msg = command.split("post", 1)[1].strip() if "post" in command.lower() else "Hello from Maven"

        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.3)
        pyautogui.typewrite("x.com/compose/post", interval=0.02)
        pyautogui.press("enter")
        time.sleep(3)

        # Click center of screen for compose box
        screen_width, screen_height = pyautogui.size()
        pyautogui.click(screen_width // 2, screen_height // 2)
        time.sleep(0.5)

        pyautogui.typewrite(msg, interval=0.03)
        pyautogui.press("enter")

        return f"Posted: {msg}"

    # Default: just describe what's possible
    return "Commands: type <text>, press <key>, click <x> <y>, grok <message>, post <message>"


# For CLI testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print(human(cmd))
    else:
        print("Usage: python human_tool.py <command>")
        print("  python human_tool.py 'type Hello world'")
        print("  python human_tool.py 'grok Say hello'")
        print("  python human_tool.py 'click 500 300'")
