# Deep Research Mode - Implementation Documentation

## Overview

Maven's Deep Research Mode is a fully implemented system that enables structured research workflows using existing knowledge (domain banks), Teacher LLM integration, and optional web access. The implementation follows Maven's architectural principles: Python 3.11 only, no `__init__.py`, tiered memory system, and offline-first design.

## Status: ✅ FULLY OPERATIONAL (Offline Mode)

All core components are implemented and tested:
- ✅ `research_manager` cognitive brain
- ✅ `research_reports` domain bank
- ✅ `web_client` tool (offline-safe stub, ready for web implementation)
- ✅ Intent detection patterns (in `language_brain`)
- ✅ Routing logic (in `memory_librarian`)
- ✅ Teacher integration and fact extraction
- ✅ Truth classification and storage
- ✅ Research report synthesis and retrieval

## Architecture

### Components

#### 1. Research Manager Brain
**Location**: `/brains/cognitive/research_manager/service/research_manager_brain.py`

**Operations**:
- `RUN_RESEARCH` - Execute a research task
- `FETCH_REPORT` - Retrieve a stored research report

**Key Functions**:
- `_summarize_memory(topic)` - Query all domain banks for existing knowledge
- `_baseline_teacher(topic)` - Get Teacher LLM briefing
- `_web_research(topic, depth)` - Web search (when enabled)
- `_store_fact_record(...)` - Store facts with truth classification
- `_store_report(...)` - Store structured research reports

**Workflow**:
```
1. Parse research request (topic, depth, sources)
2. Query existing memory via memory_librarian
3. Get Teacher baseline briefing
4. Optionally: Web research (if enabled)
5. Extract and classify facts
6. Store facts in appropriate domain banks
7. Synthesize final report
8. Store report in research_reports bank
```

#### 2. Research Reports Domain Bank
**Location**: `/brains/domain_banks/research_reports/`

**Purpose**: Store structured research reports with metadata

**Storage Structure**:
```json
{
  "content": "Topic: X\nSummary: ...",
  "confidence": 0.8,
  "source": "research_manager",
  "metadata": {
    "topic": "topic_name",
    "sources": ["memory", "teacher", "web"],
    "facts_count": 10,
    "timestamp": 1234567890.0
  }
}
```

**Operations**:
- `STORE` - Store a new research report
- `RETRIEVE` - Query reports by topic
- `COUNT` - Get tier counts
- `REBUILD_INDEX` - Rebuild search index
- `COMPACT_ARCHIVE` - Compact archive tier

#### 3. Web Client Tool
**Location**: `/tools/web_client.py`

**Status**: Stubbed (offline-safe, ready for implementation)

**Functions**:
- `search_web(query, max_results=5)` - Web search (returns empty list when disabled)
- `fetch_page(url, max_chars=8000)` - Fetch page content (returns empty when disabled)
- `_web_enabled()` - Check if web research is enabled

**Configuration**:
```python
# Via environment variable (highest priority)
export MAVEN_ENABLE_WEB_RESEARCH=1

# Via CFG in api/utils.py
CFG["web_research"]["enabled"] = True  # Default: False
CFG["web_research"]["max_results"] = 5
CFG["web_research"]["max_chars"] = 8000
```

#### 4. Intent Detection
**Location**: `/brains/cognitive/language/service/language_brain.py:1448-1503`

**Research Request Patterns**:
```python
"deep research <topic>"     → depth=3, intent="research_request"
"research <topic>"          → depth=2, intent="research_request"
"learn deeply about <topic>" → depth=3, intent="research_request"
"learn about <topic>"       → depth=2, intent="research_request"
"study <topic>"             → depth=2, intent="research_request"
"run a research task on <topic>" → depth=2, intent="research_request"
```

**Follow-up Patterns**:
```python
"what did you learn about <topic>"      → intent="research_followup"
"what have you learned about <topic>"   → intent="research_followup"
"summarize your research on <topic>"    → intent="research_followup"
"tell me what you learned about <topic>" → intent="research_followup"
"what do you know about <topic> now"    → intent="research_followup"
```

#### 5. Routing Logic
**Location**: `/brains/cognitive/memory_librarian/service/memory_librarian.py`

**Bid Creation** (line 4282):
```python
if stage3_intent in {"research_request", "research_followup"}:
    bids.append({
        "brain_name": "research_manager",
        "priority": 0.9,  # High priority
        "reason": "research_intent",
        "evidence": {...}
    })
```

**Brain Invocation** (line 5233):
```python
if stage3_intent == "research_followup":
    research_manager.service_api({"op": "FETCH_REPORT", "payload": {...}})
elif stage3_intent == "research_request":
    research_manager.service_api({"op": "RUN_RESEARCH", "payload": {...}})
```

## Usage

### Basic Research (Offline Mode)

```python
# Direct API call
from brains.cognitive.research_manager.service.research_manager_brain import service_api

result = service_api({
    "op": "RUN_RESEARCH",
    "payload": {
        "topic": "quantum computers",
        "depth": 2  # 1=basic, 2=medium, 3=deep
    }
})

print(result["payload"]["summary"])
```

### Via Chat Interface

```bash
$ ./run_chat.sh

You: research quantum computers

Maven: [Executes research using memory and Teacher]
Here's what I learned about quantum computers:
...

You: what did you learn about quantum computers

Maven: [Retrieves stored research report]
...
```

### Programmatic Usage

```python
# Research with specific configuration
result = service_api({
    "op": "RUN_RESEARCH",
    "payload": {
        "topic": "machine learning",
        "depth": 3,  # Deep research
        "deliverable": "detailed_report"
    }
})

# Retrieve previous research
report = service_api({
    "op": "FETCH_REPORT",
    "payload": {
        "topic": "machine learning"
    }
})
```

## Configuration

### Depth Levels

| Depth | Description | Memory Queries | Teacher Calls | Web Queries (if enabled) |
|-------|-------------|----------------|---------------|--------------------------|
| 1     | Basic       | 5 facts        | 1 briefing    | 1-2 queries, 2-3 pages   |
| 2     | Medium      | 5 facts        | 1 briefing    | 3-5 queries, 5-8 pages   |
| 3     | Deep        | 5 facts        | 1 briefing    | 5-10 queries, max pages  |

### Web Research (Optional)

**Enable via environment variable**:
```bash
export MAVEN_ENABLE_WEB_RESEARCH=1
python3 -m ui.maven_chat
```

**Enable via code**:
```python
# In api/utils.py
CFG["web_research"]["enabled"] = True
```

**Web Client Implementation** (stub currently):
```python
# tools/web_client.py
def search_web(query: str, max_results: int = 5) -> List[WebResult]:
    # TODO: Implement actual web search
    # Options: DuckDuckGo API, Google Custom Search, Bing API
    if not _web_enabled():
        return []
    # Implementation here
    return results
```

## Logging

All research operations produce detailed logs with `[RESEARCH]` prefix:

```
[RESEARCH] Starting research task: topic='computers', depth=2, sources=['memory','teacher']
[RESEARCH] Querying existing memory for topic='computers'...
[RESEARCH] Memory summary: 3 existing facts found
[RESEARCH] Calling Teacher for baseline understanding of 'computers'...
[RESEARCH] Teacher response: 500 chars, extracting facts...
[RESEARCH] Extracted 8 facts from Teacher response
[RESEARCH] Stored fact to 'technology': A computer is a programmable machine...
[RESEARCH] Stored fact to 'factual': Computers process data using binary...
[RESEARCH] Stored research report for topic='computers' (id=3cd91b720a5f08ab)
```

Web research logging (when enabled):
```
[RESEARCH_WEB] Starting web research for topic='computers', depth=2
[RESEARCH_WEB] Query 1: 'computers'
[RESEARCH_WEB] Found 5 search results
[RESEARCH_WEB] Fetched 7500 chars from https://example.com/computers
[RESEARCH_WEB] Web research complete: 12 findings
```

## Testing

### Run Test Suite

```bash
cd /home/user/maven/maven2_fix
python3 test_research_mode.py
```

**Expected Output**:
```
✓ PASS   Basic Research
✓ PASS   Fetch Report
✓ PASS   Reports Bank
✓ PASS   Web Config

Total: 4/4 tests passed
🎉 All tests passed! Deep Research Mode is operational.
```

### Manual Testing

```python
# Test research request
python3 -c "
from brains.cognitive.research_manager.service.research_manager_brain import service_api
result = service_api({'op': 'RUN_RESEARCH', 'payload': {'topic': 'AI', 'depth': 2}})
print(result['payload']['summary'])
"

# Test fetch report
python3 -c "
from brains.cognitive.research_manager.service.research_manager_brain import service_api
result = service_api({'op': 'FETCH_REPORT', 'payload': {'topic': 'AI'}})
print(result['payload']['summary'])
"
```

## Bug Fixes Applied

### 1. MAVEN_ROOT Path Bug
**Issue**: `MAVEN_ROOT = HERE.parents[3]` pointed to `/brains/` instead of `/maven2_fix/`

**Fix**: Changed to `HERE.parents[4]`

**Impact**: `_bank_module()` and `_call_memory_librarian()` now correctly locate modules

### 2. Silent Error Handling
**Issue**: Bare `except Exception: pass` swallowed all errors

**Fix**: Added detailed logging and error messages:
```python
except Exception as e:
    print(f"[RESEARCH] Error: {e}")
```

### 3. Missing Visibility
**Issue**: No logging made debugging difficult

**Fix**: Added comprehensive logging at each step:
- Memory queries
- Teacher calls
- Fact extraction
- Storage operations
- Report synthesis

## Integration with Existing Systems

### Teacher Learning Mode
**Location**: `/brains/teacher_contracts.py:166`

```python
"research_manager": {
    "operation": "TEACH",
    "prompt_mode": "world_question",
    "store_internal": True,
    "store_domain": True,
    "enabled": True,
    "description": "Learns how to run structured research tasks and capture facts"
}
```

### Truth Classification
All facts extracted during research pass through `TruthClassifier`:

```python
from brains.cognitive.reasoning.truth_classifier import TruthClassifier

classification = TruthClassifier.classify(content, confidence, evidence={"source": source})
if TruthClassifier.should_store_in_memory(classification):
    # Store to appropriate domain bank
```

**Verdicts**:
- `FACT` → Store in domain bank
- `EDUCATED_GUESS` → Store in domain bank
- `WORKING_THEORY` → Store in `working_theories` bank
- `RANDOM` → Skip storage

### Memory Librarian
Research uses `UNIFIED_RETRIEVE` operation to query all domain banks:

```python
res = memory_librarian.service_api({
    "op": "UNIFIED_RETRIEVE",
    "payload": {"query": topic, "k": 5}
})
```

## Future Enhancements

### Web Search Implementation Options

1. **DuckDuckGo (Privacy-focused, no API key)**:
```python
import requests
def search_web(query, max_results=5):
    response = requests.get(f"https://html.duckduckgo.com/html/?q={query}")
    # Parse HTML for results
    return results
```

2. **Google Custom Search (Requires API key)**:
```python
from googleapiclient.discovery import build
def search_web(query, max_results=5):
    service = build("customsearch", "v1", developerKey=API_KEY)
    result = service.cse().list(q=query, cx=CX, num=max_results).execute()
    return [WebResult(...) for item in result.get('items', [])]
```

3. **Bing Search API (Requires API key)**:
```python
import requests
def search_web(query, max_results=5):
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers=headers,
        params={"q": query, "count": max_results}
    )
    return [WebResult(...) for r in response.json()["webPages"]["value"]]
```

### Enhanced Research Features

1. **Multi-stage refinement**: Iteratively refine research based on findings
2. **Source citation**: Track and display source URLs in reports
3. **Contradictions detection**: Flag conflicting information from different sources
4. **Visual reports**: Generate markdown/HTML formatted reports
5. **Research scheduling**: Background research tasks
6. **Collaborative research**: Multiple topics in parallel

## Troubleshooting

### Issue: Teacher returns no result
**Cause**: Teacher LLM not available or not configured
**Solution**: Research still works with memory-only mode. Configure Teacher LLM for enhanced results.

### Issue: Web research returns no results
**Cause**: `web_client.py` is stubbed (offline-safe default)
**Solution**: Implement actual web search in `web_client.py` or set `MAVEN_ENABLE_WEB_RESEARCH=0` to disable.

### Issue: Facts not being stored
**Cause**: Truth classification rejecting low-confidence facts
**Solution**: Check `[RESEARCH]` logs for "Skipping storage (classification rejected)" messages. Increase confidence scores if appropriate.

### Issue: Reports not found
**Cause**: Topic mismatch or report not created
**Solution**: Check exact topic string. Reports are stored with exact topic text from request.

## Performance Considerations

### Memory Usage
- Each research task stores facts in domain banks (STM → MTM → LTM → Archive rotation)
- Research reports stored separately in `research_reports` bank
- Automatic tier rotation prevents overflow

### Speed
- **Memory-only research**: ~100-500ms
- **With Teacher**: ~1-5 seconds (depends on LLM response time)
- **With web (when implemented)**: ~5-30 seconds (depends on network and pages)

### Scalability
- Depth 1-2: Suitable for interactive use
- Depth 3: Better for background/scheduled research
- Web research: Rate-limit aware implementation recommended

## Compliance

✅ **Python 3.11 only** - No dependencies on newer Python features
✅ **No `__init__.py`** - Module loading via `importlib.util`
✅ **Offline-first** - Web is optional, works fully offline
✅ **Tiered memory** - All storage via BrainMemory API
✅ **No global state** - All state in domain banks and brain memory
✅ **Windows 10 compatible** - Path handling via `pathlib.Path`

## Summary

Deep Research Mode is **fully operational** in offline mode with comprehensive logging, error handling, and testing. The system integrates seamlessly with Maven's existing architecture:

- **Intent detection** via language brain
- **Routing** via memory librarian
- **Teacher integration** for learning
- **Truth classification** for fact validation
- **Domain banks** for knowledge storage
- **Tiered memory** for automatic rotation

Web search can be optionally enabled by implementing the stubbed functions in `web_client.py`.

---

**Last Updated**: 2025-11-20
**Implementation Status**: Production Ready (Offline Mode)
**Test Status**: ✅ All tests passing
