# Browser Runtime Documentation

## Overview

The Browser Runtime system provides Maven with a "browser body" - the ability to interact with web pages, perform searches, and extract information from the internet. It's built on Playwright and designed with safety, human-like behavior, and pattern learning in mind.

## Architecture

### Components

1. **Browser Runtime** (`browser_runtime/`)
   - HTTP/RPC server exposing browser automation API
   - Session management for browser contexts and pages
   - Action implementations (open, click, type, etc.)
   - Configuration and safety guardrails

2. **Maven Browser Client** (`maven_browser_client/`)
   - Python client library for the Browser Runtime API
   - High-level convenience methods
   - Type definitions and schemas

3. **Browser Tool** (`brains/agent/tools/browser/`)
   - Integration with Maven's agent executor
   - Intent resolution (natural language → browser plans)
   - Pattern learning and reuse
   - Task execution and logging

## Quick Start

### 1. Install Dependencies

```bash
cd maven2_fix
pip install -r browser_requirements.txt
playwright install chromium
```

### 2. Configure Environment

Copy the example configuration:

```bash
cp .env.browser.example .env.browser
```

Edit `.env.browser` to customize settings:

```bash
# Browser mode: headless or headed
BROWSER_MODE=headless

# Browser type: chromium, firefox, or webkit
BROWSER_TYPE=chromium

# Safety limits
BROWSER_MAX_STEPS_PER_TASK=20
BROWSER_MAX_DURATION_SECONDS=120
```

### 3. Start Browser Runtime Server

```bash
python -m browser_runtime.server
```

The server will start on `http://127.0.0.1:8765` by default.

### 4. Use from Maven

```python
from brains.agent.tools.browser.browser_tool import simple_google_search, open_url

# Perform a Google search
result = simple_google_search("Python web scraping tutorial")
print(result["final_text"])

# Open a specific URL
result = open_url("https://example.com")
print(result["final_url"])
```

## Usage Examples

### Example 1: Simple Google Search

```python
from brains.agent.tools.browser.browser_tool import simple_google_search

result = simple_google_search("machine learning courses")

if result["status"] == "completed":
    print("Search completed!")
    print(f"Final URL: {result['final_url']}")
    print(f"Page text (first 500 chars): {result['final_text'][:500]}")
else:
    print(f"Search failed: {result['error_message']}")
```

### Example 2: Custom Browser Plan

```python
from brains.agent.tools.browser.browser_tool import run_browser_task
from maven_browser_client.types import ActionType

plan = {
    "goal": "Search for Python tutorials and click first result",
    "max_steps": 10,
    "steps": [
        {
            "action": ActionType.OPEN.value,
            "params": {"url": "https://www.google.com"}
        },
        {
            "action": ActionType.TYPE.value,
            "params": {
                "selector": "textarea[name=q]",
                "text": "Python tutorial",
                "submit": True
            }
        },
        {
            "action": ActionType.WAIT_FOR.value,
            "params": {
                "selector": "#search",
                "timeout_ms": 10000
            }
        },
        {
            "action": ActionType.CLICK.value,
            "params": {
                "selector": "h3",
                "nth": 0
            }
        },
        {
            "action": ActionType.SCREENSHOT.value,
            "params": {"full_page": False}
        }
    ]
}

result = run_browser_task("Search and click first result", plan)
print(f"Task completed in {result['duration_seconds']:.2f} seconds")
```

### Example 3: Using Intent Resolver

```python
from brains.agent.tools.browser.intent_resolver import resolve_intent
from brains.agent.tools.browser.browser_tool import execute_browser_plan
import asyncio

# Resolve natural language intent to a plan
plan = resolve_intent("search for best restaurants in San Francisco")

# Execute the plan
result = asyncio.run(execute_browser_plan(plan))

print(f"Status: {result.status}")
print(f"Final text: {result.final_text[:500]}")
```

### Example 4: Low-Level Client Usage

```python
import asyncio
from maven_browser_client.client import BrowserClient

async def main():
    async with BrowserClient() as client:
        # Open a page
        page_id, snapshot = await client.open_url("https://news.ycombinator.com")
        print(f"Opened: {snapshot.url}")

        # Scroll down
        await client.scroll(page_id, direction="down", amount=500)

        # Take screenshot
        result = await client.screenshot(page_id, full_page=True)
        print(f"Screenshot saved to: {result.snapshot.screenshot_path}")

        # Get page text
        text = await client.get_text(page_id)
        print(f"Page text length: {len(text)} characters")

        # Close page
        await client.close_page(page_id)

asyncio.run(main())
```

## Available Actions

### OPEN
Open a URL in the browser.

**Parameters:**
- `url` (required): URL to navigate to
- `wait_until` (optional): Wait condition ("load", "domcontentloaded", "networkidle")
- `timeout_ms` (optional): Navigation timeout in milliseconds

### CLICK
Click an element on the page.

**Parameters:**
- `selector` (optional): CSS selector for element
- `text` (optional): Text content to search for
- `nth` (optional): Which matching element to click (0-indexed)
- `timeout_ms` (optional): Timeout for finding element

### TYPE
Type text into an input element.

**Parameters:**
- `selector` (required): CSS selector for input element
- `text` (required): Text to type
- `submit` (optional): Whether to press Enter after typing
- `delay_ms` (optional): Delay between keystrokes

### WAIT_FOR
Wait for an element to appear.

**Parameters:**
- `selector` (required): CSS selector to wait for
- `timeout_ms` (optional): Timeout in milliseconds
- `state` (optional): State to wait for ("visible", "attached", "hidden", "detached")

### SCROLL
Scroll the page.

**Parameters:**
- `direction` (required): Scroll direction ("up", "down", "top", "bottom")
- `amount` (optional): Pixels to scroll (for up/down)

### SCREENSHOT
Take a screenshot.

**Parameters:**
- `full_page` (optional): Whether to capture full scrollable page

### EXTRACT_TEXT
Extract text content from the page.

**Parameters:** None (uses current page)

## Safety Features

### Domain Restrictions

Control which domains can be accessed:

```python
# In .env.browser
BROWSER_ALLOWED_DOMAINS=google.com,wikipedia.org,github.com
BROWSER_BLOCKED_DOMAINS=facebook.com,twitter.com
```

### Rate Limiting

Prevent excessive requests to the same domain:

```python
BROWSER_RATE_LIMIT_PER_DOMAIN=5  # Max 5 requests per domain
```

### Step and Time Limits

Prevent runaway tasks:

```python
BROWSER_MAX_STEPS_PER_TASK=20      # Max 20 actions per task
BROWSER_MAX_DURATION_SECONDS=120   # Max 2 minutes per task
```

### Plan Validation

All browser plans are validated before execution:
- Step count limits
- Domain restrictions
- Required parameters
- Action sequence validity

## Pattern Learning

The system learns from successful browsing tasks and can reuse patterns.

### Built-in Patterns

1. **google_search**: Search Google for a query
2. **open_url**: Open a specific URL

### Adding Custom Patterns

```python
from maven_browser_client.types import PatternMatch, BrowserPlan, BrowserAction, ActionType
from brains.agent.tools.browser.pattern_store import get_pattern_store

# Define pattern
pattern = PatternMatch(
    name="stackoverflow_search",
    description="Search StackOverflow for programming questions",
    trigger_keywords=["stackoverflow", "programming question", "code help"],
    domains=["stackoverflow.com"],
    template_plan=BrowserPlan(
        goal="Search StackOverflow",
        max_steps=5,
        steps=[
            BrowserAction(
                action=ActionType.OPEN,
                params={"url": "https://stackoverflow.com"}
            ),
            BrowserAction(
                action=ActionType.TYPE,
                params={"selector": "input[name=q]", "text": "{query}", "submit": True}
            ),
        ]
    )
)

# Add to store
store = get_pattern_store()
store.add_pattern(pattern)
```

### Recording Pattern Success/Failure

```python
from brains.agent.tools.browser.pattern_store import get_pattern_store

store = get_pattern_store()

# After successful execution
store.record_success("google_search")

# After failed execution
store.record_failure("google_search")
```

## Logging

All browser tasks are logged to `logs/browser/tasks/` with the following information:

- Task ID
- Goal
- Plan (steps)
- Start/end time
- Status
- Steps executed
- Errors (if any)
- Final result

View logs:

```bash
ls -la logs/browser/tasks/
cat logs/browser/tasks/<task-id>.json
```

## Testing

### Run Unit Tests

```bash
pytest tests/test_browser_runtime.py -v
```

### Run Smoke Tests

```bash
pytest tests/test_browser_smoke.py -v -m smoke
```

### Run Integration Tests

Requires browser runtime server to be running:

```bash
# Terminal 1: Start server
python -m browser_runtime.server

# Terminal 2: Run tests
pytest tests/test_browser_integration.py -v -m integration
```

## Troubleshooting

### Playwright Installation Issues

If Playwright browsers aren't installed:

```bash
playwright install chromium
```

### Server Connection Issues

Check server is running:

```bash
curl http://127.0.0.1:8765/health
```

### Domain Blocked Errors

Check your domain restrictions in `.env.browser`:

```bash
BROWSER_ALLOWED_DOMAINS=  # Leave empty to allow all
BROWSER_BLOCKED_DOMAINS=  # Leave empty to block none
```

### Timeout Errors

Increase timeouts in configuration:

```bash
BROWSER_MAX_DURATION_SECONDS=300  # Increase to 5 minutes
```

## API Reference

See detailed API documentation in the code docstrings:

- `browser_runtime/actions.py` - Action implementations
- `browser_runtime/server.py` - HTTP API endpoints
- `maven_browser_client/client.py` - Client methods
- `brains/agent/tools/browser/browser_tool.py` - Maven integration

## Future Enhancements

Planned features for future releases:

1. **LLM-based Plan Generation**: Use LLMs to generate complex multi-step plans
2. **Visual Element Selection**: Click elements by visual description
3. **Form Auto-fill**: Intelligent form filling
4. **Multi-page Navigation**: Handle complex navigation flows
5. **Screenshot Analysis**: Use vision models to understand page content
6. **Session Persistence**: Save and restore browser sessions
7. **Proxy Support**: Route traffic through proxies
8. **Cookie Management**: Save and load cookies
9. **Advanced Pattern Mining**: Automatically discover new patterns from successful tasks
10. **Collaborative Browsing**: Multiple agents working together
