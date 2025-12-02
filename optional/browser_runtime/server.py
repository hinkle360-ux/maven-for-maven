"""
Browser Runtime Server
======================

FastAPI server that exposes browser automation capabilities via HTTP/RPC API.

NOTE: On Windows, the event loop policy must be set BEFORE uvicorn starts.
Use run_browser_server.py to launch this server, which sets the correct
WindowsSelectorEventLoopPolicy for Playwright subprocess support.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from optional.browser_runtime.config import get_config
from optional.browser_runtime.session_manager import get_session_manager
from optional.browser_runtime.models import (
    ActionResult,
    OpenRequest,
    ClickRequest,
    TypeRequest,
    WaitForRequest,
    ScrollRequest,
    ScreenshotRequest,
    GetPageRequest,
    ClosePageRequest,
    PageSnapshot,
    SearchRequest,
)
from optional.browser_runtime.actions import (
    action_open,
    action_click,
    action_type,
    action_wait_for,
    action_scroll,
    action_screenshot,
    create_page_snapshot,
)


# Setup logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("browser_runtime")


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage browser runtime lifecycle."""
    logger.info("Starting browser runtime...")
    session_manager = get_session_manager()
    await session_manager.initialize()
    logger.info("Browser runtime started successfully")

    yield

    logger.info("Shutting down browser runtime...")
    await session_manager.shutdown()
    logger.info("Browser runtime shut down")


# Create FastAPI app
app = FastAPI(
    title="Browser Runtime API",
    description="HTTP API for browser automation using Playwright",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================================
# Health and Status Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {"status": "ok", "service": "browser_runtime", "version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    session_manager = get_session_manager()
    return {
        "status": "healthy",
        "initialized": session_manager._initialized,
        "active_pages": len(session_manager.pages),
    }


# ============================================================================
# Browser Action Endpoints
# ============================================================================


@app.post("/open", response_model=ActionResult)
async def open_url(request: OpenRequest):
    """
    Open a URL in the browser.

    Creates a new page, navigates to the URL, and returns the page ID and snapshot.
    """
    try:
        session_manager = get_session_manager()
        page_id, page = await session_manager.create_page()

        logger.info(f"Opening URL: {request.url} (page_id: {page_id})")

        result = await action_open(page, request)
        result.page_id = page_id

        return result

    except Exception as e:
        logger.error(f"Error opening URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/click", response_model=ActionResult)
async def click(request: ClickRequest):
    """Click an element on the page."""
    try:
        session_manager = get_session_manager()
        page = await session_manager.get_page(request.page_id)

        logger.info(f"Clicking element (page_id: {request.page_id}, selector: {request.selector}, text: {request.text})")

        result = await action_click(page, request)
        result.page_id = request.page_id

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error clicking element: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/type", response_model=ActionResult)
async def type_text(request: TypeRequest):
    """Type text into an input element."""
    try:
        session_manager = get_session_manager()
        page = await session_manager.get_page(request.page_id)

        logger.info(f"Typing text (page_id: {request.page_id}, selector: {request.selector})")

        result = await action_type(page, request)
        result.page_id = request.page_id

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error typing text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wait_for", response_model=ActionResult)
async def wait_for(request: WaitForRequest):
    """Wait for an element or condition."""
    try:
        session_manager = get_session_manager()
        page = await session_manager.get_page(request.page_id)

        logger.info(f"Waiting for element (page_id: {request.page_id}, selector: {request.selector})")

        result = await action_wait_for(page, request)
        result.page_id = request.page_id

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error waiting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scroll", response_model=ActionResult)
async def scroll(request: ScrollRequest):
    """Scroll the page."""
    try:
        session_manager = get_session_manager()
        page = await session_manager.get_page(request.page_id)

        logger.info(f"Scrolling (page_id: {request.page_id}, direction: {request.direction})")

        result = await action_scroll(page, request)
        result.page_id = request.page_id

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error scrolling: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screenshot", response_model=ActionResult)
async def screenshot(request: ScreenshotRequest):
    """Take a screenshot of the page."""
    try:
        session_manager = get_session_manager()
        page = await session_manager.get_page(request.page_id)

        logger.info(f"Taking screenshot (page_id: {request.page_id}, full_page: {request.full_page})")

        result = await action_screenshot(page, request)
        result.page_id = request.page_id

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error taking screenshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_page", response_model=PageSnapshot)
async def get_page(request: GetPageRequest):
    """Get current page state as a snapshot."""
    try:
        session_manager = get_session_manager()
        page = await session_manager.get_page(request.page_id)

        logger.info(f"Getting page snapshot (page_id: {request.page_id})")

        snapshot = await create_page_snapshot(page, include_html=request.include_html, include_text=request.include_text)

        return snapshot

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close_page")
async def close_page(request: ClosePageRequest):
    """Close a page and cleanup resources."""
    try:
        session_manager = get_session_manager()
        await session_manager.close_page(request.page_id)

        logger.info(f"Closed page (page_id: {request.page_id})")

        return {"status": "ok", "message": f"Page {request.page_id} closed successfully"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error closing page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=ActionResult)
async def search(request: SearchRequest):
    """
    Perform a web search using the specified engine.

    Supports DuckDuckGo (default), Bing, and Google.
    DuckDuckGo is recommended as it doesn't use CAPTCHA.
    """
    try:
        from optional.browser_tools.browser_tool import build_search_url

        # Build search URL for the specified engine
        url = build_search_url(request.query, request.engine)

        logger.info(f"Searching: '{request.query}' on {request.engine} -> {url}")

        # Create page and navigate to search URL
        session_manager = get_session_manager()
        page_id, page = await session_manager.create_page()

        open_request = OpenRequest(url=url)
        result = await action_open(page, open_request)
        result.page_id = page_id

        # Add metadata about the search
        result.metadata = {
            "engine": request.engine,
            "query": request.query,
            "search_url": url,
        }

        return result

    except Exception as e:
        logger.error(f"Error performing search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ============================================================================
# Main Entry Point
# ============================================================================


def run_server():
    """Run the browser runtime server."""
    import uvicorn

    config = get_config()
    logger.info(f"Starting server on {config.host}:{config.port}")

    uvicorn.run(
        "optional.browser_runtime.server:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
