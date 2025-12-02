# Canonical Maven Runtime Setup

**Status**: ✅ Complete
**Date**: 2025-11-18
**Commit**: 94b1cfe
**Branch**: claude/consolidate-maven-runtime-01MwAoNYSrkeV5R7Yzm4iq2o

---

## Summary

This document describes the canonical Maven runtime setup that ensures:

1. ✅ **Single runtime path** - Only one codebase can execute
2. ✅ **No legacy confusion** - Legacy paths are guarded with warnings
3. ✅ **Memory + routing testable** - Self-test command proves it works
4. ✅ **Version identification** - Every startup shows commit/branch/features

---

## Part 1: Repo + Runtime Cleanup

### 1.1 Canonical Maven Folder

**Location**:
- Linux/macOS: `/home/user/maven/maven2_fix`
- Windows: `C:\Users\hinkl\Desktop\maven2_fix`

**Structure**:
```
maven2_fix/
  api/
  brains/
  config/
  docs/
  ui/
  run_chat.cmd          # Windows entry point
  run_chat.sh           # Linux/macOS entry point (NEW)
  run_maven.py          # Main CLI
  maven_version.txt     # Version identification (NEW)
  cleanup_caches.py     # Cache cleanup utility (NEW)
  ...
```

**Verified**:
- ✅ No duplicate Maven folders with runnable code
- ✅ Only one entry point per platform

---

### 1.2 Entry Scripts

#### Windows: run_chat.cmd
- ✅ Has `cd /d "%~dp0"` as second line (mandatory)
- ✅ Sets PYTHONPATH to project root
- ✅ Runs via `python -m ui.maven_chat`
- ✅ Lives in project root

#### Linux/macOS: run_chat.sh (NEW)
- ✅ Changes to script directory via `cd "$(dirname "$0")"`
- ✅ Sets PYTHONPATH to project root
- ✅ Runs via `python3 -m ui.maven_chat`
- ✅ Executable permission set

---

### 1.3 Cache Cleanup

**Script**: `cleanup_caches.py`

**Purpose**: Remove all `__pycache__` directories and stale `.pyc` files

**Usage**:
```bash
cd maven2_fix
python cleanup_caches.py
```

**Result**: Ensures no old bytecode can cause import issues

---

### 1.4 Runtime Self-Identification

**Version File**: `maven_version.txt`

**Contents**:
```
commit=94b1cfe
branch=claude/consolidate-maven-runtime-01MwAoNYSrkeV5R7Yzm4iq2o
features=teacher_learning,routing_learning,tier_memory,canonical_pipeline
```

**Startup Print**: When you run `run_chat.cmd` or `run_chat.sh`, you'll see:
```
Welcome to the Maven chat interface. Type 'exit' or 'quit' to leave.
MAVEN BUILD: commit=94b1cfe branch=... features=...
```

**Verification**: If you see different commit/branch info, you're running the wrong build.

---

## Part 2: Memory + Routing Sanity Tests

### 2.1 Integration Test

**File**: `tests/integration/test_routing_memory_duck.py`

**What it tests**:
1. First query: "what sound does a duck make"
   - ✅ Teacher is called
   - ✅ verdict == "LEARNED"
   - ✅ teacher_facts_stored > 0
   - ✅ Routing rule is learned
2. Second query: Same question
   - ✅ Teacher is NOT called
   - ✅ Memory retrieval returns facts
   - ✅ Routing rule is matched
   - ✅ Correct banks are used

**How to run**:
```bash
cd maven2_fix/tests/integration
python test_routing_memory_duck.py
```

**Expected output**: `✓ PASS: Routing + memory loop works correctly!`

---

### 2.2 Self-Test Command

**Command**: `python run_maven.py --self-test-routing`

**What it does**:
1. Runs duck question twice
2. Prints detailed report:
   - Teacher called on first query: YES/NO
   - Teacher called on second query: YES/NO
   - Learned routing rule applied: YES/NO
   - Banks used on second query: [list]
3. Exits with result: PASS or FAIL

**Example output**:
```
============================================================
MAVEN ROUTING + MEMORY SELF-TEST
============================================================

Testing with question: 'what sound does a duck make'

[1/2] First query (should call teacher)...
  Verdict: LEARNED
  Teacher facts stored: 3
  [SELF-TEST] Teacher called on first query: YES

[2/2] Second query (should use memory)...
  Verdict: TRUE
  Banks used: ['science', 'factual']
  [SELF-TEST] Teacher called on second query: NO
  [SELF-TEST] Learned routing rule applied: YES
  [SELF-TEST] Banks used on second query: science, factual

============================================================
[SELF-TEST] RESULT: PASS
============================================================
✓ Teacher called on first query
✓ Memory used on second query (teacher not called)
✓ Routing selected banks: science, factual
```

---

### 2.3 Legacy Path Status

**See**: `LEGACY_PATHS_STATUS.md` for full details

**Summary**:
- ✅ Default pipeline: `canonical` (via PipelineExecutor)
- ⚠️ Legacy pipeline: Available via `--pipeline legacy` but **DEPRECATED** with warning
- ✅ No direct memory file operations for cognitive state (use BrainMemory API)

**Verification**: The default behavior uses only canonical paths. Legacy mode is opt-in with warnings.

---

## Quick Start Verification

Run these commands to verify everything works:

```bash
# 1. Check version identification
cd maven2_fix
python run_maven.py
# Should show: MAVEN BUILD: commit=94b1cfe ...

# 2. Clean caches
python cleanup_caches.py

# 3. Run self-test
python run_maven.py --self-test-routing
# Should show: [SELF-TEST] RESULT: PASS

# 4. Run integration test
cd tests/integration
python test_routing_memory_duck.py
# Should show: ✓ PASS: Routing + memory loop works correctly!
```

If all four steps succeed, you have a verified canonical runtime with no legacy confusion.

---

## What Changed

### New Files
1. `run_chat.sh` - Linux/macOS entry script
2. `cleanup_caches.py` - Cache cleanup utility
3. `maven_version.txt` - Version identification file
4. `tests/integration/test_routing_memory_duck.py` - Routing+memory integration test
5. `LEGACY_PATHS_STATUS.md` - Legacy path documentation
6. `CANONICAL_RUNTIME_SETUP.md` - This file

### Modified Files
1. `run_maven.py` - Added `--self-test-routing` command and legacy deprecation warning
2. `ui/maven_chat.py` - Added version printing on startup

### Removed Files
None (cleanup approach: guard legacy, don't delete)

---

## Troubleshooting

### Problem: Version shows wrong commit/branch

**Cause**: Running from wrong directory or old clone

**Fix**:
1. Verify you're in the canonical path: `pwd` should show `maven2_fix`
2. Check git status: `git branch` should show `claude/consolidate-maven-runtime-...`
3. Delete any other maven folders that might be in your PATH

---

### Problem: Self-test fails

**Cause**: Various (teacher unavailable, memory issues, routing issues)

**Fix**:
1. Check the specific failure message in the output
2. Run the integration test for more details: `python tests/integration/test_routing_memory_duck.py`
3. Check if LLM service is available (teacher needs LLM)
4. Review logs for errors

---

### Problem: Legacy pipeline warning appears

**Cause**: You or a script is using `--pipeline legacy` flag

**Fix**:
1. Remove the `--pipeline legacy` flag from your command
2. Use default (canonical) pipeline instead
3. If you need legacy for debugging, understand it's deprecated

---

## Maintainer Notes

**Single source of truth**:
- Entry point: `run_chat.cmd` (Windows) or `run_chat.sh` (Linux/macOS)
- Pipeline: `brains/pipeline/pipeline_runner.py` → `PipelineExecutor`
- Memory: `brains/memory/brain_memory.py` → `BrainMemory` API
- Version: `maven_version.txt`

**Testing**:
- Quick test: `python run_maven.py --self-test-routing`
- Full test: `python tests/integration/test_routing_memory_duck.py`

**When adding features**:
1. Use canonical pipeline (not legacy)
2. Use BrainMemory API (not direct file I/O for cognitive state)
3. Add tests to prove it works
4. Update `maven_version.txt` features list if needed

---

**End of Document**
