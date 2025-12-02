"""
Browser Runtime CLI Entry Point
================================

Allows running browser runtime as a module:
    python -m optional.browser_runtime
"""

from optional.browser_runtime.server import run_server

if __name__ == "__main__":
    run_server()
