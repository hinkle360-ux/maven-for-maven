# Maven Capability System Implementation Progress

**Date:** 2025-11-26
**Branch:** claude/add-agency-filesystem-access-01QtoZQNA8UCLtpmcSLqtx6W
**Status:** Part 1 Complete (Foundation), Part 2 In Progress

---

## Overview

Implementing comprehensive capability-aware system to make Maven honest about what it can and cannot do, eliminate hallucinations, and provide persistent execution configuration.

---

## ✅ COMPLETED (Part 1 - Foundation)

### 1. Capability Registry (`capabilities.py`) ✅
- **Lines:** 340
- **Purpose:** Single source of truth for Maven's capabilities
- **Functions:**
  - `get_capabilities()` - Returns dict of all capabilities with available/enabled/reason
  - `is_capability_enabled(name)` - Quick check if capability is enabled
  - `get_capability_reason(name)` - Get human-readable reason for status
  - `describe_capabilities()` - Generate full human-readable description

**Capabilities Tracked:**
- filesystem_agency
- git_agency
- hot_reload
- self_introspection
- execution_guard
- teacher_learning
- routing_learning
- web_research
- pattern_recognition
- memory_consolidation

**Features:**
- Checks feature flags from `config/features.json`
- Checks execution status from env vars or config file
- Checks module availability via importlib
- Returns structured data: `{"available": bool, "enabled": bool, "reason": str}`

### 2. Feature Flags (`config/features.json`) ✅
- **Purpose:** Control which capabilities are available
- **Format:** JSON with boolean flags
- **Benefits:**
  - Enable/disable features without code changes
  - Clear configuration in one place
  - Basis for dynamic banner and capability reporting

### 3. Dynamic Version Info (`version_utils.py`) ✅
- **Lines:** 200
- **Purpose:** Get version info from git, not stale files
- **Functions:**
  - `get_version_info()` - Returns commit, branch, features, source
  - `get_version_banner()` - Returns formatted string: "commit=X, branch=Y, features=Z"

**Precedence:**
1. Git commands (if repo available)
2. maven_version.txt (fallback)
3. "unknown" (final fallback)

### 4. Persistent Execution Config (execution_guard.py) ✅
- **Modified:** `check_execution_enabled()` function
- **New Functions:**
  - `enable_execution_via_config()` - Write to ~/.maven/config.json
  - `disable_execution_via_config()` - Disable in config
  - `get_execution_status()` - Get current status with source/reason

**New Precedence Order:**
1. **Environment variables** (highest) - MAVEN_EXECUTION_ENABLED + USER_CONFIRMED_EXECUTION
2. **Config file** (~/.maven/config.json) - Persistent per-user settings
3. **Default** (disabled) - Safe default requiring explicit enablement

**Config Format:**
```json
{
  "execution_enabled": true,
  "user_confirmed_execution": true
}
```

**Clear Error Messages:**
- Before: "MAVEN_EXECUTION_ENABLED flag is not enabled"
- After: "Execution disabled by default. Use 'enable execution' command to enable, or set MAVEN_EXECUTION_ENABLED=1 and USER_CONFIRMED_EXECUTION=YES."

### 5. Dynamic Banner (ui/maven_chat.py) ✅
- **Modified:** `repl()` function startup banner
- **Now:** Calls `get_version_banner()` for dynamic info
- **Shows:** Real git commit, branch, and enabled features
- **Fallback:** maven_version.txt if git unavailable

**Example Output:**
```
Welcome to the Maven chat interface. Type 'exit' or 'quit' to leave.
MAVEN BUILD: commit=aa3b04d, branch=claude/add-agency-filesystem-access-01QtoZQNA8UCLtpmcSLqtx6W, features=fs+git+reload+introspection+exec+teacher+routing+patterns+memory
```

---

## 🔄 IN PROGRESS (Part 2 - Integration)

### 6. Enable Execution Chat Command
- **Status:** Not started
- **Location:** Will add to maven_chat.py or as pipeline command
- **Function:** Detect "enable execution" query, prompt for confirmation, call `enable_execution_via_config()`
- **Response:** "Execution enabled. This allows Maven to read/write files and run tools. Settings saved to ~/.maven/config.json"

### 7. Capability-Aware Routing
- **Status:** Not started
- **Location:** brains/cognitive/integrator/service/integrator_brain.py or pipeline_runner.py
- **Purpose:** Route tool queries to actual tools, not Teacher
- **Logic:**
  ```python
  if query_is_tool_call("filesystem"):
      if is_capability_enabled("filesystem_agency"):
          route_to_filesystem_service()
      else:
          return f"Filesystem operations disabled: {get_capability_reason('filesystem_agency')}"
  ```

**Tool Routing Map:**
```python
TOOL_ROUTE_MAP = {
    "fs_tool": "filesystem_agency",
    "git_tool": "git_agency",
    "reload_tool": "hot_reload",
    "introspect": "self_introspection"
}
```

### 8. Self-Model Capabilities
- **Status:** Not started
- **Location:** brains/cognitive/self_model/service/self_model_brain.py
- **Add Functions:**
  - `describe_capabilities()` - Call `capabilities.describe_capabilities()`
  - `describe_limitations()` - Honest about what Maven cannot do
  - `get_system_status()` - Include execution status, root path, etc.

### 9. Route Meta-Questions to Self-Model
- **Status:** Not started
- **Location:** Pipeline or integrator
- **Detect Patterns:**
  - "what can you do"
  - "do you have access to X"
  - "how are you"
  - "scan self"
  - "describe yourself"
- **Action:** Route to self_model with capability data, NOT Teacher

### 10. Fix Final-Answer Fixer
- **Status:** Not started
- **Location:** Find where "I'm not sure how to help" responses are generated
- **Change:** Respect `verdict=CAPABILITY_DISABLED` from pipeline
- **Logic:**
  ```python
  if verdict == "CAPABILITY_DISABLED":
      return payload["reason"]  # Don't overwrite with generic text
  ```

---

## 🔜 PENDING (Part 3 - Testing)

### 11. Unit Tests
- **File:** tests/test_capabilities.py
  - Test `get_capabilities()` with different configs
  - Test execution guard precedence (env > config > default)
  - Test `get_version_info()` with/without git

- **File:** tests/test_version_utils.py
  - Mock git commands
  - Test fallback to maven_version.txt
  - Test feature abbreviation mapping

- **File:** tests/test_execution_config.py
  - Test `enable_execution_via_config()`
  - Test config file creation
  - Verify precedence order

### 12. Integration Tests
- **Chat Interface:**
  - Banner shows current commit/branch
  - `enable execution` command works
  - Execution persists across restarts

- **Capability Routing:**
  - Tool queries route to actual tools when enabled
  - Clear "disabled" messages when not enabled
  - Meta-questions route to self_model

- **Self-Model Honesty:**
  - `scan self` shows real capabilities
  - No hallucinations about non-existent features
  - Execution status included in output

---

## Implementation Statistics

### Code Written (Part 1)
| File | Lines | Status |
|------|-------|--------|
| capabilities.py | 340 | ✅ Complete |
| version_utils.py | 200 | ✅ Complete |
| config/features.json | 12 | ✅ Complete |
| execution_guard.py (modified) | +90 | ✅ Complete |
| maven_chat.py (modified) | +15 | ✅ Complete |
| **Total** | **~660 lines** | **✅ Part 1 Done** |

### Code Remaining (Parts 2-3)
| Task | Est. Lines | Priority |
|------|-----------|----------|
| Enable execution command | ~100 | High |
| Capability-aware routing | ~200 | High |
| Self-model integration | ~150 | High |
| Meta-question routing | ~100 | Medium |
| Final-answer fix | ~50 | Medium |
| Tests | ~400 | High |
| **Total** | **~1,000 lines** | **In Progress** |

---

## Testing Checklist

### Manual Tests (Part 1) ✅
- [x] `capabilities.get_capabilities()` returns all capabilities
- [x] `capabilities.describe_capabilities()` generates readable text
- [x] `version_utils.get_version_banner()` shows current commit/branch
- [x] Banner displays at chat startup
- [x] Execution guard respects env vars
- [x] Execution guard checks ~/.maven/config.json
- [x] Clear error messages when execution disabled

### Manual Tests (Part 2) ⏳
- [ ] `enable execution` command creates config file
- [ ] Execution persists across chat restarts
- [ ] Tool queries route to actual tools when enabled
- [ ] Clear "disabled" messages when tools are disabled
- [ ] `scan self` shows real capabilities
- [ ] Meta-questions go to self_model, not Teacher
- [ ] No hallucinations about capabilities

### Automated Tests ⏳
- [ ] pytest tests/test_capabilities.py
- [ ] pytest tests/test_version_utils.py
- [ ] pytest tests/test_execution_config.py
- [ ] All existing tests still pass

---

## Key Benefits

### Before
- ❌ Stale hardcoded banner (wrong branch/commit)
- ❌ Fragile env-only execution gates
- ❌ No persistent execution settings
- ❌ Hallucinations about capabilities
- ❌ Tool queries sent to Teacher → generic responses
- ❌ "I don't know anything about Maven" responses

### After (Part 1 Complete)
- ✅ Dynamic banner from git (current commit/branch/features)
- ✅ Config file precedence (persistent across restarts)
- ✅ Clear error messages explaining how to enable
- ✅ Capability registry as single source of truth
- ✅ Execution status includes source and reason

### After (All Parts Complete)
- ✅ Tool queries execute actual tools (when enabled)
- ✅ Clear "disabled" messages (when not enabled)
- ✅ Self-model describes real capabilities
- ✅ Meta-questions route to self_model
- ✅ No hallucinations about non-existent features
- ✅ One-time "enable execution" command

---

## Next Steps

**Immediate (High Priority):**
1. Implement `enable execution` chat command
2. Add capability-aware routing to integrator
3. Integrate capabilities into self_model

**Short-term (Medium Priority):**
4. Route meta-questions to self_model
5. Fix final-answer fixer to respect capability_disabled
6. Add comprehensive tests

**Testing:**
7. Manual testing of all chat commands
8. Automated test suite
9. Verify no regressions in existing functionality

---

## Commit History

**Commit 1:** aa3b04d - "Add dynamic capabilities, version info, and persistent execution config"
- Added capabilities.py, version_utils.py, config/features.json
- Updated execution_guard.py with config precedence
- Updated maven_chat.py banner to use dynamic info

**Status:** ✅ Foundation complete, integration in progress

---

**Last Updated:** 2025-11-26
**Author:** Claude (Sonnet 4.5)
