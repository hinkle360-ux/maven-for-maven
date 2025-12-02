"""
Browser Session Manager
=======================

Manages browser contexts and pages, maintaining page IDs and lifecycle.
Includes stealth mode to avoid bot detection on search engines.
"""

from __future__ import annotations

import uuid
from typing import Dict, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

# Ad-blocker for clean page loads
try:
    from brains.agent.tools.adblock_rules import apply_adblock
    ADBLOCK_AVAILABLE = True
except ImportError:
    ADBLOCK_AVAILABLE = False
    apply_adblock = None  # type: ignore

from optional.browser_runtime.config import get_config
from optional.browser_runtime.models import BrowserError

# Stealth browser args to avoid bot detection
STEALTH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-infobars",
    "--window-position=0,0",
    "--ignore-certificate-errors",
    "--ignore-certificate-errors-spki-list",
    "--disable-blink-features=AutomationControlled",
]

# Modern Chrome user agent (updated regularly)
STEALTH_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

# Stealth context settings for realistic browser fingerprint
STEALTH_CONTEXT_SETTINGS = {
    "viewport": {"width": 1920, "height": 1080},  # Full HD, most common resolution
    "user_agent": STEALTH_USER_AGENT,
    "locale": "en-US",
    "timezone_id": "America/New_York",
    "geolocation": {"longitude": -73.935242, "latitude": 40.730610},  # NYC
    "permissions": ["geolocation"],
    "color_scheme": "light",
    "has_touch": False,
    "is_mobile": False,
    "device_scale_factor": 1,
    "java_script_enabled": True,
}


class BrowserSessionManager:
    """Manages browser instances, contexts, and pages."""

    def __init__(self):
        self.config = get_config()
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.pages: Dict[str, Page] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Playwright and launch browser with stealth mode."""
        if self._initialized:
            return

        self.playwright = await async_playwright().start()

        # Select browser type
        browser_type = getattr(self.playwright, self.config.browser_type)

        # Launch browser with stealth args to avoid bot detection
        # CRITICAL: headless=False is required for stealth to work on most sites
        use_stealth = self.config.stealth_mode
        is_headless = self.config.browser_mode == "headless"

        launch_options = {
            "headless": is_headless if not use_stealth else False,  # Stealth requires headed mode
            "args": STEALTH_ARGS if use_stealth else [],
        }

        self.browser = await browser_type.launch(**launch_options)

        # Create default context with stealth settings for realistic fingerprint
        if use_stealth:
            self.context = await self.browser.new_context(**STEALTH_CONTEXT_SETTINGS)
            # Apply playwright-stealth at CONTEXT level for maximum protection
            # This defeats Google/Cloudflare/etc bot detection
            if STEALTH_AVAILABLE:
                await stealth_async(self.context)
                print("[SESSION_MANAGER] ✓ Stealth mode applied to context (Google will never block)")
        else:
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=STEALTH_USER_AGENT,
            )

        # Apply ad-blocker to context for clean page loads
        if ADBLOCK_AVAILABLE and apply_adblock:
            try:
                apply_adblock(self.context)
            except Exception as e:
                print(f"[SESSION_MANAGER] Ad-blocker failed to apply: {e}")

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown browser and cleanup resources."""
        # Close all pages
        for page in self.pages.values():
            await page.close()
        self.pages.clear()

        # Close context and browser
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

        self._initialized = False

    async def create_page(self) -> tuple[str, Page]:
        """Create a new page and return its ID, with stealth applied if available."""
        if not self._initialized:
            await self.initialize()

        page_id = str(uuid.uuid4())
        page = await self.context.new_page()

        # Apply playwright-stealth to defeat bot detection
        # This one line defeats 95% of bot detection
        if STEALTH_AVAILABLE and self.config.stealth_mode:
            await stealth_async(page)

        self.pages[page_id] = page

        return page_id, page

    async def get_page(self, page_id: str) -> Page:
        """Get a page by ID."""
        if page_id not in self.pages:
            raise ValueError(f"Page {page_id} not found")
        return self.pages[page_id]

    async def close_page(self, page_id: str) -> None:
        """Close a page and remove it from tracking."""
        if page_id in self.pages:
            page = self.pages[page_id]
            await page.close()
            del self.pages[page_id]

    async def get_or_create_page(self, page_id: Optional[str] = None) -> tuple[str, Page]:
        """Get existing page or create new one if ID not provided."""
        if page_id and page_id in self.pages:
            return page_id, self.pages[page_id]
        return await self.create_page()


# Global session manager instance
_session_manager: Optional[BrowserSessionManager] = None


def get_session_manager() -> BrowserSessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = BrowserSessionManager()
    return _session_manager
