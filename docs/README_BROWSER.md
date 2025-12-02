# Maven Browser Runtime

> Give Maven a browser body to interact with the web

## Overview

The Maven Browser Runtime gives Maven the ability to:
- 🔍 Search the web (Google, etc.)
- 🌐 Navigate to URLs
- 🖱️ Click, type, and interact with web pages
- 📸 Take screenshots
- 📄 Extract text content
- 🧠 Learn patterns from successful browsing tasks

## Quick Start

### 1. Install

```bash
# Install dependencies
pip install -r browser_requirements.txt

# Install Playwright browsers
playwright install chromium
```

### 2. Configure

```bash
# Copy example config
cp .env.browser.example .env.browser

# Edit as needed (optional)
nano .env.browser
```

### 3. Start Server

```bash
# Start browser runtime server
python -m browser_runtime.server
```

The server runs on `http://127.0.0.1:8765` by default.

### 4. Use

```python
from brains.agent.tools.browser.browser_tool import simple_google_search

# Search the web
result = simple_google_search("Python tutorial")
print(result["final_text"])
```

## Examples

### Simple Google Search

```python
from brains.agent.tools.browser.browser_tool import simple_google_search

result = simple_google_search("machine learning")
```

### Open a URL

```python
from brains.agent.tools.browser.browser_tool import open_url

result = open_url("https://github.com")
```

### Custom Browser Plan

```python
from brains.agent.tools.browser.browser_tool import run_browser_task
from maven_browser_client.types import ActionType

plan = {
    "goal": "Search Wikipedia",
    "max_steps": 5,
    "steps": [
        {"action": "open", "params": {"url": "https://wikipedia.org"}},
        {"action": "type", "params": {"selector": "input[name=search]", "text": "Python", "submit": True}},
        {"action": "screenshot", "params": {"full_page": False}},
    ]
}

result = run_browser_task("Search Wikipedia", plan)
```

### Natural Language Intent

```python
from brains.agent.tools.browser.intent_resolver import resolve_intent
import asyncio

plan = resolve_intent("search for best restaurants in Tokyo")
# Plan is automatically created from the intent!
```

## Architecture

```
┌─────────────────────────────────────┐
│         Maven Cognition             │
│    (Natural Language Intents)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Intent Resolver                │
│   (Intent → Browser Plan)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Browser Tool                   │
│   (Plan Execution + Logging)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Maven Browser Client             │
│   (High-level Python API)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Browser Runtime Server           │
│   (HTTP API + Playwright)           │
└─────────────────────────────────────┘
```

## Features

### ✅ Implemented (Phases 0-6)

- ✅ **Phase 0**: Configuration and setup
- ✅ **Phase 1**: Basic browser runtime (open, get page)
- ✅ **Phase 2**: Human-like interactions (click, type, scroll, screenshot)
- ✅ **Phase 3**: Maven integration (client + tool)
- ✅ **Phase 4**: Browser plans with validation
- ✅ **Phase 5**: Safety, sandboxing, logging
- ✅ **Phase 6**: Pattern learning and reflection

### 🚀 Available Actions

| Action | Description | Example |
|--------|-------------|---------|
| `OPEN` | Navigate to URL | `{"action": "open", "params": {"url": "..."}}` |
| `CLICK` | Click element | `{"action": "click", "params": {"selector": "..."}}` |
| `TYPE` | Type text | `{"action": "type", "params": {"selector": "...", "text": "..."}}` |
| `WAIT_FOR` | Wait for element | `{"action": "wait_for", "params": {"selector": "..."}}` |
| `SCROLL` | Scroll page | `{"action": "scroll", "params": {"direction": "down"}}` |
| `SCREENSHOT` | Take screenshot | `{"action": "screenshot", "params": {"full_page": true}}` |
| `EXTRACT_TEXT` | Extract page text | `{"action": "extract_text", "params": {}}` |

### 🔒 Safety Features

- **Domain Restrictions**: Whitelist/blacklist domains
- **Rate Limiting**: Prevent excessive requests
- **Step Limits**: Max steps per task (default: 20)
- **Time Limits**: Max duration per task (default: 120s)
- **Plan Validation**: Validates all plans before execution
- **Comprehensive Logging**: All tasks logged to disk

### 🧠 Pattern Learning

The system learns from successful tasks and can reuse patterns:

```python
from brains.agent.tools.browser.pattern_store import get_pattern_store

store = get_pattern_store()

# List learned patterns
for pattern in store.list_patterns():
    print(f"{pattern.name}: {pattern.description}")

# Record success/failure
store.record_success("google_search")
```

## Testing

```bash
# Unit tests
pytest tests/test_browser_runtime.py -v

# Smoke tests (launches real browser)
pytest tests/test_browser_smoke.py -v -m smoke

# Integration tests (requires server running)
pytest tests/test_browser_integration.py -v -m integration
```

## Configuration

Edit `.env.browser` to customize:

```bash
# Browser settings
BROWSER_MODE=headless              # headless or headed
BROWSER_TYPE=chromium              # chromium, firefox, webkit

# Server settings
BROWSER_RUNTIME_HOST=127.0.0.1
BROWSER_RUNTIME_PORT=8765

# Safety limits
BROWSER_MAX_STEPS_PER_TASK=20
BROWSER_MAX_DURATION_SECONDS=120
BROWSER_RATE_LIMIT_PER_DOMAIN=5

# Domain restrictions
BROWSER_ALLOWED_DOMAINS=           # Comma-separated, empty = allow all
BROWSER_BLOCKED_DOMAINS=           # Comma-separated, empty = block none

# Logging
BROWSER_LOG_LEVEL=INFO
BROWSER_LOG_DIR=./logs/browser

# Human-like behavior
BROWSER_MIN_DELAY_MS=100
BROWSER_MAX_DELAY_MS=500
```

## Documentation

- 📚 **Full Documentation**: [`docs/BROWSER_RUNTIME.md`](docs/BROWSER_RUNTIME.md)
- 💡 **Examples**: [`docs/examples/browser_examples.py`](docs/examples/browser_examples.py)
- 🧪 **Tests**: [`tests/test_browser_*.py`](tests/)

## Troubleshooting

**Server won't start?**
```bash
# Check Playwright is installed
playwright install chromium
```

**Domain blocked?**
```bash
# Check .env.browser
BROWSER_ALLOWED_DOMAINS=  # Leave empty to allow all
```

**Timeout errors?**
```bash
# Increase timeout in .env.browser
BROWSER_MAX_DURATION_SECONDS=300
```

**Connection refused?**
```bash
# Check server is running
curl http://127.0.0.1:8765/health
```

## Future Enhancements

- [ ] LLM-based plan generation
- [ ] Visual element selection
- [ ] Form auto-fill
- [ ] Multi-page navigation flows
- [ ] Screenshot analysis with vision models
- [ ] Session persistence
- [ ] Proxy support
- [ ] Cookie management
- [ ] Advanced pattern mining
- [ ] Multi-agent collaborative browsing

## License

Same as Maven project.

---

**Ready to give Maven web access? Start the server and let Maven explore!** 🚀🌐
