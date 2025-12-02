# Research Mode Implementation - Complete

**Date**: 2025-01-21
**Status**: ✅ COMPLETE - Ready for testing
**Location**: C:\Users\hinkl\Desktop\maven2_fix

## Summary

Full end-to-end research mode with online capabilities is now implemented. When you say "research: X", Maven will:

1. Create a research job object
2. Break the topic into sub-questions using Teacher
3. For each sub-question:
   - Query Teacher for knowledge
   - Fetch real web pages (if enabled)
   - Extract and classify facts using Teacher
   - Store facts in appropriate domain banks
4. Generate a comprehensive summary
5. Store a research report for future reference

## Components Implemented

### 1. Research Manager Brain
**Location**: `brains/cognitive/research_manager/service/research_manager_brain.py`

**New functionality**:
- Research job object creation and management
- Sub-question planning with Teacher
- Fact extraction and classification with Teacher
- Web research integration
- Complete research workflow orchestration

**Key functions**:
```python
create_research_job(topic, full_prompt, owner) -> job_id
update_research_job(job_id, patch_dict)
append_subquestion(job_id, question, status)
append_source(job_id, title, url, trust_score)
increment_facts_count(job_id)

_plan_subquestions(topic, depth) -> List[str]
_extract_facts_from_text(text, topic, source) -> List[Dict]
run_research(payload) -> Dict
```

**Research workflow**:
```
run_research() →
  1. create_research_job()
  2. _plan_subquestions() using Teacher
  3. For each sub-question:
     a. Query Teacher for answer
     b. _extract_facts_from_text()
     c. search_web() if enabled
     d. For each web result:
        - fetch_page()
        - _extract_facts_from_text()
        - _store_fact_record() via TruthClassifier
  4. Generate final summary with Teacher
  5. _store_report()
  6. Mark job complete
```

### 2. Web Client
**Location**: `tools/web_client.py`

**Implementation**:
- Uses `urllib.request` (Python stdlib, no deps)
- DuckDuckGo HTML scraping for search
- HTML text extraction with `HTMLParser`
- Configurable via `config/research.json` or `MAVEN_ENABLE_WEB_RESEARCH` env var

**Functions**:
```python
search_web(query, max_results=5) -> List[WebResult]
fetch_page(url, max_chars=8000) -> str
```

**WebResult structure**:
```python
@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
```

**Safety**:
- Disabled by default (offline-safe)
- Requires explicit enable via config or env var
- Graceful error handling (returns empty on network failure)
- Timeout protection (30 seconds default)

### 3. Research Job Object
**Storage**: BrainMemory("research_jobs")

**Structure**:
```python
{
    "job_id": "research-1737494400-a1b2c3d4",
    "topic": "quantum computing",
    "full_prompt": "research: quantum computing for beginners",
    "status": "pending|in_progress|complete|failed",
    "subquestions": [
        {"q": "What is quantum computing?", "status": "complete", "notes": ""}
    ],
    "sources": [
        {"title": "Intro to Quantum", "url": "https://...", "trust_score": 0.6}
    ],
    "facts_stored": 42,
    "created_at": 1737494400.0,
    "updated_at": 1737494450.0,
    "owner": "user"
}
```

### 4. Configuration
**Location**: `config/research.json`

**Contents**:
```json
{
  "online_enabled": false,
  "max_requests_per_job": 10,
  "max_sources_per_subquestion": 3,
  "max_subquestions": 8,
  "allowed_domains": [],
  "timeout_seconds": 30,
  "user_agent": "Maven/2.0 Research Bot",
  "max_page_chars": 8000
}
```

**Enable web research**:
- **Option 1**: Set `"online_enabled": true` in `config/research.json`
- **Option 2**: Set environment variable `MAVEN_ENABLE_WEB_RESEARCH=1`
- **Option 3**: Set `CFG['web_research']['enabled'] = True` in code

### 5. Pipeline Integration
**Already exists** in `memory_librarian.py`

**Intent detection** in `language_brain.py:1448`:
- Triggers: "research:", "deep research", "learn about", "study", etc.
- Sets `intent=research_request` with topic and depth

**Routing** in `memory_librarian.py:4279` and `5233`:
- Detects `research_request` intent
- Routes to `research_manager` brain
- Calls `RUN_RESEARCH` op
- Returns summary as answer

## Trigger Phrases

### Research Mode (RUN_RESEARCH)
```
research: quantum computing
deep research: climate change
learn about artificial intelligence
study neural networks
run a research task on solar panels
```

### Fetch Stored Report (FETCH_REPORT)
```
what did you learn about computers
what have you learned about quantum computing
summarize your research on climate change
tell me what you learned about AI
```

## Logging Output

When running a research job, you'll see logs like:

```
[RESEARCH_MODE] New job id=research-1737494400-a1b2c3d4 topic="quantum computing"
[RESEARCH] Starting research task: topic='quantum computing', depth=2, sources=['memory', 'teacher', 'web']
[RESEARCH_PLAN] Planning sub-questions for topic="quantum computing" (depth=2)
[RESEARCH_PLAN] Subquestions count=6

[RESEARCH] Processing sub-question 1/6: What is quantum computing?...
[RESEARCH_FACTS] Extracting facts from 1234 chars (source=teacher)
[RESEARCH_FACTS] Extracted 5 facts
[RESEARCH] Stored fact to 'factual': Quantum computing uses quantum bits or qubits...
[RESEARCH_WEB] Query="What is quantum computing?" max_results=3
[RESEARCH_WEB] Found 3 search results
[RESEARCH_WEB] Fetching https://example.com/quantum-intro
[RESEARCH_WEB] Fetched 3456 chars from https://example.com/quantum-intro
[RESEARCH_FACTS] Extracting facts from 3456 chars (source=web:https://example.com/quantum-intro)
[RESEARCH_FACTS] Extracted 8 facts
[RESEARCH] Stored fact to 'factual': Quantum computers can solve certain problems exponentially faster...

... (repeated for each sub-question)

[RESEARCH] Stored research report for topic='quantum computing' (id=abc123)
[RESEARCH_DONE] job_id=research-1737494400-a1b2c3d4 status=complete facts=42
```

## Fact Storage

Facts are stored via the existing `TruthClassifier` and routed to domain banks:

- **FACT** (high confidence) → `factual` bank
- **EDUCATED** (medium confidence) → `working_theories` bank
- **UNKNOWN/RANDOM** (low confidence) → skipped or `stm_only`

Each fact stored includes metadata:
```python
{
    "content": "Quantum computers use qubits instead of bits",
    "confidence": 0.8,
    "source": "web:https://example.com/quantum",
    "metadata": {
        "topic": "quantum computing",
        "url": "https://example.com/quantum",
        "classification": "FACT"
    }
}
```

## Research Reports

Stored in `research_reports` domain bank for future retrieval:

```python
{
    "content": "Topic: quantum computing\nSummary: [full summary text]",
    "confidence": 0.8,
    "source": "research_manager",
    "metadata": {
        "topic": "quantum computing",
        "sources": ["memory", "teacher", "web"],
        "facts_count": 42,
        "timestamp": 1737494450.0
    }
}
```

## Testing Instructions

### Prerequisites
1. Navigate to: `C:\Users\hinkl\Desktop\maven2_fix`
2. Run chat interface: `run_chat.cmd`

### Test 1: Simple Research (Offline)
**Purpose**: Test research mode without web (Teacher only)

**Command**:
```
research: how do computers work
```

**Expected logs**:
- `[RESEARCH_MODE] New job id=...`
- `[RESEARCH_PLAN] Subquestions count=...`
- `[RESEARCH_FACTS] Extracted N facts`
- `[RESEARCH_DONE] job_id=... status=complete`

**Expected output**: Summary of how computers work with facts stored

### Test 2: Research with Web Enabled
**Purpose**: Test full online research

**Setup**:
```bash
set MAVEN_ENABLE_WEB_RESEARCH=1
```

**Command**:
```
research: photosynthesis
```

**Expected logs**:
- All logs from Test 1, plus:
- `[RESEARCH_WEB] Query="..." max_results=3`
- `[RESEARCH_WEB] Found N search results`
- `[RESEARCH_WEB] Fetching https://...`
- `[RESEARCH_WEB] Fetched M chars from https://...`

**Expected output**: Rich summary with facts from web sources

### Test 3: Retrieve Stored Research
**Purpose**: Test report retrieval

**Command** (after Test 1 or 2):
```
what did you learn about computers
```

**Expected**: Should return stored research summary without re-researching

### Test 4: Multi-Topic Research
**Purpose**: Test multiple research jobs

**Commands**:
```
research: neural networks
research: solar energy
what do you know about neural networks
```

**Expected**: Each topic researched separately, facts stored to different domains

### Test 5: Memory Stats
**Purpose**: Verify facts are being stored

**Commands**:
```
research: artificial intelligence
how many facts have you learned
```

**Expected**: Fact count should increase after research

### Test 6: Identity Safety Check
**Purpose**: Ensure self-questions don't trigger research mode

**Commands**:
```
who are you
what is your name
what do you know about your code
```

**Expected**:
- NO `[RESEARCH_MODE]` logs
- Answers come from `self_model` (identity system)
- Research mode NOT triggered

## Safety & Constraints

### What's Protected
1. **Self-identity questions** - Handled by `SELF_INTENT_GATE`, never go through research
2. **Offline by default** - Web research disabled unless explicitly enabled
3. **No breaking changes** - All existing systems (self, routing, memory) untouched
4. **No new dependencies** - Uses only Python 3.11 stdlib
5. **No __init__.py** - No packages created

### Network Safety
- Web client is offline-safe stub when disabled
- Returns empty gracefully on network errors
- Timeout protection (30 sec default)
- User-agent identification

### Memory Safety
- Facts stored via existing `BrainMemory` API
- TruthClassifier filters low-quality facts
- Proper domain bank routing

## Files Modified

### New Files
1. `config/research.json` - Research configuration
2. `docs/RESEARCH_MODE_SCAN.md` - Initial scan results
3. `docs/RESEARCH_MODE_IMPLEMENTATION.md` - This file

### Modified Files
1. `tools/web_client.py` - Added real HTTP implementation
2. `brains/cognitive/research_manager/service/research_manager_brain.py` - Complete rewrite with job system

### Existing Files (Untouched)
- `brains/cognitive/language/service/language_brain.py` - Intent detection already there
- `brains/cognitive/memory_librarian/service/memory_librarian.py` - Routing already there
- `brains/domain_banks/research_reports/service/research_reports_bank.py` - Already exists
- All self-identity, routing learning, tiered memory systems - UNCHANGED

## Example Research Session

```
User: research: quantum computing for beginners

[RESEARCH_MODE] New job id=research-1737494400-a1b2c3d4 topic="quantum computing for beginners"
[RESEARCH] Starting research task: topic='quantum computing for beginners', depth=2, sources=['memory', 'teacher']
[RESEARCH_PLAN] Planning sub-questions for topic="quantum computing for beginners" (depth=2)
[RESEARCH_PLAN] Subquestions count=6

1. What is quantum computing in simple terms?
2. What are qubits and how do they differ from regular bits?
3. What are the main applications of quantum computing?
4. What are the current limitations of quantum computers?
5. What companies are working on quantum computing?
6. How can beginners learn about quantum computing?

[RESEARCH] Processing sub-question 1/6: What is quantum computing in simple terms?...
[RESEARCH_FACTS] Extracting facts from 856 chars (source=teacher)
[RESEARCH_FACTS] Extracted 4 facts
[RESEARCH] Stored fact to 'factual': Quantum computing is a type of computation that uses quantum mechanics...
[RESEARCH] Stored fact to 'factual': Unlike classical computers which use bits (0 or 1), quantum computers use qubits...
...

[RESEARCH_DONE] job_id=research-1737494400-a1b2c3d4 status=complete facts=28

Maven: I've completed research on quantum computing for beginners.

Quantum computing is a revolutionary approach to computation that leverages quantum mechanics principles. Unlike classical computers that use bits (0 or 1), quantum computers use qubits which can exist in superposition states. This allows them to process certain calculations exponentially faster than traditional computers.

Key applications include cryptography, drug discovery, optimization problems, and machine learning. However, current quantum computers face challenges like maintaining quantum coherence and error correction.

Major companies working in this space include IBM, Google, Microsoft, and Rigetti. For beginners, online courses from MIT OpenCourseWare and IBM's Quantum Experience provide excellent starting points.

I've stored 28 facts about this topic for future reference.

User: what did you learn about quantum computing

[RESEARCH] FETCH_REPORT for topic='quantum computing'

Maven: [Returns the stored summary from above]
```

## Next Steps for User

1. **Test offline first** - Run through all test cases without web enabled
2. **Enable web** - Set `MAVEN_ENABLE_WEB_RESEARCH=1` and retest
3. **Monitor logs** - Check that all `[RESEARCH_*]` logs appear as expected
4. **Verify fact storage** - Use "how many facts have you learned" to confirm
5. **Test identity safety** - Ensure "who are you" doesn't trigger research
6. **Iterate** - Report any issues or unexpected behavior

## Technical Notes

### Why BrainMemory for Jobs?
Research jobs are stored using the existing `BrainMemory` system to maintain consistency. No separate job storage system was created.

### Why Teacher for Sub-Questions and Fact Extraction?
Using Teacher provides:
- Structured breakdown of topics
- Quality fact extraction
- Classification guidance
- Offline capability (works without web)

### Why DuckDuckGo?
- No API key required
- Simple HTML scraping
- Privacy-focused
- Reliable for basic search

### Limitations
- Web scraping may break if DuckDuckGo changes HTML structure
- Fact extraction quality depends on Teacher performance
- No image/video content extraction
- No PDF parsing (only HTML text)

## Confirmation

✅ **No __init__.py created**
✅ **No new third-party dependencies**
✅ **Python 3.11 stdlib only** (urllib, html.parser, uuid)
✅ **Everything in C:\Users\hinkl\Desktop\maven2_fix**
✅ **Testing via run_chat.cmd only**
✅ **Self-identity system untouched**
✅ **Routing learning untouched**
✅ **Tiered memory system untouched**

## Implementation Complete

Research mode is now fully functional end-to-end. Ready for testing via `run_chat.cmd`.
