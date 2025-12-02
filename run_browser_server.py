import sys
import asyncio

# IMPORTANT: Playwright needs Selector loop on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn


def main():
    uvicorn.run(
        "optional.browser_runtime.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,  # turn off reload for now to avoid extra processes
    )


if __name__ == "__main__":
    main()
