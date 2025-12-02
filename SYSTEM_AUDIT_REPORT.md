# Maven 2.0 Full System Audit Report

**Date:** 2025-12-01
**Auditor:** Claude Code (Opus 4)
**Codebase:** Maven 2.0 Multi-Brain Cognitive AI System
**Branch:** claude/full-system-audit-015yWUn9DAnZH9Lp2u2VZMLx

---

## Executive Summary

Maven is a sophisticated multi-brain cognitive AI system featuring a 9-stage pipeline, 33+ specialized cognitive brains, 19 domain knowledge banks, and tiered memory management. While the architecture demonstrates impressive design principles, this audit has identified **5 critical security vulnerabilities**, **3 high-severity security issues**, and **numerous code quality concerns** that require immediate attention.

### Key Findings Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Security | 5 | 3 | 2 | - |
| Code Quality | 2 | 4 | 6 | 3 |
| Architecture | - | 2 | 3 | 2 |

### Test Suite Status

| Metric | Value |
|--------|-------|
| Total Test Cases | 573 |
| Tests Passing | 364 (63.5%) |
| Tests Failing | 205 (35.8%) |
| Tests with Errors | 2 (0.4%) |
| Tests Skipped | 2 (0.4%) |

---

## 1. Architecture Overview

### 1.1 Project Structure

```
maven2_fix/
├── brains/                    # Core cognitive architecture (stdlib-only)
│   ├── cognitive/             # 33+ cognitive brain modules
│   ├── memory/                # Tiered memory management (STM/MTM/LTM/Archive)
│   ├── domain_banks/          # 19 knowledge domain stores
│   ├── pipeline/              # 9-stage execution pipeline
│   ├── governance/            # Safety & control systems
│   ├── tools/                 # Tool execution & control
│   ├── personal/              # Personal identity system
│   └── agent/                 # Autonomous agent system
├── host_tools/                # External I/O implementations
├── optional/                  # Optional features (browser)
├── ui/                        # User interface
├── api/                       # Utility APIs
├── config/                    # 23 JSON configuration files
├── tests/                     # Test suite (573 tests)
└── scripts/                   # Utility scripts
```

### 1.2 Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ (stdlib-only core) |
| Storage | JSONL files (tiered: STM/MTM/LTM/Archive) |
| Configuration | JSON (23 config files) |
| LLM Provider | Ollama (default), OpenAI (optional) |
| Web Search | DuckDuckGo (default), SerpAPI (optional) |
| Browser | Playwright (optional) |

### 1.3 Code Metrics

| Metric | Value |
|--------|-------|
| Total Python Files | 288+ |
| Total Lines of Code | ~132,500 |
| Test Cases | 573 |
| Configuration Files | 23 JSON files |
| Brain Modules | 33+ |
| Domain Banks | 19 |

### 1.4 Pipeline Architecture (9 Stages)

1. **NLU** - Natural Language Understanding
2. **PATTERN_RECOGNITION** - Pattern detection and classification
3. **MEMORY** - Memory retrieval and context building
4. **REASONING** - Fact evaluation and inference
5. **VALIDATION** - Truth validation and safety checks
6. **GENERATION** - Response generation
7. **FINALIZATION** - Output formatting and cleanup
8. **HISTORY** - Conversation history update
9. **AUTONOMY** - Autonomous action consideration

---

## 2. Security Vulnerabilities

### 2.1 CRITICAL Vulnerabilities

#### VULN-001: Unsafe eval() Execution
- **Location:** `brains/cognitive/language/service/language_brain.py:4011, 4918`
- **CWE:** CWE-95 (Improper Neutralization of Directives)
- **Risk Level:** CRITICAL
- **Description:** Direct `eval()` execution on mathematical expressions without proper sandboxing
- **Code:**
  ```python
  val = str(eval(k))           # Line 4011
  result = str(eval(_math_key)) # Line 4918
  ```
- **Impact:** Arbitrary code execution if attacker-controlled input reaches these functions
- **Recommendation:** Replace with `ast.literal_eval()` or dedicated math expression parser

#### VULN-002: FULL_AGENCY Authorization Bypass
- **Location:** `brains/tools/execution_guard.py:47, 378-396`
- **CWE:** CWE-269 (Improper Access Control)
- **Risk Level:** CRITICAL
- **Description:** FULL_AGENCY mode disables all authorization checks except a weak string-matching deny list
- **Code:**
  ```python
  if status.mode == ExecMode.FULL_AGENCY:
      # FULL_AGENCY allows everything else, including CRITICAL operations
      return True, ""
  ```
- **Impact:** Complete circumvention of security controls
- **Recommendation:** Remove FULL_AGENCY mode; implement capability-based authorization

#### VULN-003: Unrestricted exec() Execution
- **Location:** `brains/agent/tools/python_exec.py:267, 158-160`
- **CWE:** CWE-95 (Eval Injection)
- **Risk Level:** CRITICAL
- **Description:** In FULL_AGENCY mode, exec() runs with unrestricted access to all Python builtins
- **Code:**
  ```python
  if cfg.get("unrestricted", False):
      return dict(vars(builtins))  # ALL builtins available
  exec(compile(code, "<user_code>", "exec"), globals_dict, locals_dict)
  ```
- **Impact:** Arbitrary Python code execution including subprocess, os, socket modules
- **Recommendation:** Remove unrestricted mode; use RestrictedPython or subprocess isolation

#### VULN-004: Self-Modifying Code Execution
- **Location:** `brains/tools/self_upgrade_tool.py:127, 215, 267`
- **CWE:** CWE-95 (Code Injection)
- **Risk Level:** CRITICAL
- **Description:** System can write and commit arbitrary Python code to its own codebase
- **Impact:** Persistent backdoor injection; changes survive restarts
- **Recommendation:** Implement code review workflow; add code signing

#### VULN-005: Dynamic Module Hot Reload
- **Location:** `brains/tools/hot_reload.py:84, 237-238`
- **CWE:** CWE-95 (Dynamic Code Evaluation)
- **Risk Level:** CRITICAL
- **Description:** Modules can be reloaded without integrity verification
- **Code:**
  ```python
  reloaded_module = importlib.reload(old_module)  # No verification
  ```
- **Impact:** Runtime code injection into running process
- **Recommendation:** Implement code signing verification before reload

### 2.2 HIGH Vulnerabilities

#### VULN-006: Weak Command Injection Protection
- **Location:** `host_tools/shell_executor/executor.py:148-157`
- **CWE:** CWE-78 (OS Command Injection)
- **Risk Level:** HIGH
- **Description:** Deny patterns use simple substring matching, easily bypassed
- **Bypass Examples:**
  - `rm  -rf /` (extra spaces)
  - `safe_cmd; rm -rf /` (command chaining)
  - URL encoding variations
- **Recommendation:** Use AST-based command parsing; implement whitelist approach

#### VULN-007: Git Command Injection
- **Location:** `brains/tools/self_upgrade_tool.py:230-234`
- **CWE:** CWE-78 (Command Injection)
- **Risk Level:** HIGH
- **Description:** Commit messages and branch names passed without validation
- **Impact:** Git configuration injection
- **Recommendation:** Validate arguments; use `--` separator before arguments

#### VULN-008: Unrestricted Module Imports
- **Location:** `brains/agent/tools/python_exec.py:105-110, 183-188`
- **CWE:** CWE-94 (Code Injection)
- **Risk Level:** HIGH
- **Description:** FULL_AGENCY mode allows importing any Python module
- **Dangerous Modules Available:** subprocess, os, socket, pickle, importlib
- **Recommendation:** Maintain strict import allowlist always

### 2.3 MEDIUM Vulnerabilities

#### VULN-009: Insufficient URL Validation
- **Location:** `host_tools/web_client/client.py:125-131, 203`
- **CWE:** CWE-918 (SSRF)
- **Risk Level:** MEDIUM
- **Description:** No URL validation before making requests; potential SSRF
- **Recommendation:** Implement URL validation and domain whitelist

#### VULN-010: Missing Operation-Level Authorization
- **Location:** `brains/agent/tools/intent_resolver_tools.py:176-200`
- **CWE:** CWE-269 (Improper Access Control)
- **Risk Level:** MEDIUM
- **Description:** Only global mode checked, not specific operation authorization
- **Recommendation:** Implement per-operation authorization checks

### 2.4 Security Positives

| Feature | Status | Location |
|---------|--------|----------|
| Path traversal protection | IMPLEMENTED | `brains/maven_paths.py` |
| No hardcoded credentials | VERIFIED | Config files only |
| Audit logging | IMPLEMENTED | `~/.maven/execution_audit.jsonl` |
| Tool injection pattern | IMPLEMENTED | `brains/tools_api.py` |

---

## 3. Code Quality Issues

### 3.1 CRITICAL Quality Issues

#### 3.1.1 Excessive File Sizes

| File | Lines | Recommendation |
|------|-------|----------------|
| `memory_librarian.py` | 11,288 | Split into 5-6 focused modules |
| `language_brain.py` | 8,217 | Split into parsing/generation/utils |
| `reasoning_brain.py` | 4,123 | Extract domain logic |
| `self_model_brain.py` | 2,521 | Modularize introspection |
| `research_manager_brain.py` | 2,003 | Extract search logic |

#### 3.1.2 Silent Exception Handling
- **Count:** 456+ instances of bare `except Exception: pass`
- **Impact:** Errors silently swallowed, debugging extremely difficult
- **Primary Locations:**
  - `memory_librarian.py` - 456 instances
  - `language_brain.py` - 273 instances
  - `personal_brain.py` - 109 instances
- **Recommendation:** Add specific exception types and logging

### 3.2 HIGH Quality Issues

#### 3.2.1 Functions Exceeding Reasonable Length

| File | Function | Lines | Start |
|------|----------|-------|-------|
| memory_librarian.py | `_numeric_score` | 2,692 | L6743 |
| memory_librarian.py | `_norm_title` | 1,504 | L9436 |
| memory_librarian.py | `_planner_required` | 1,172 | L5570 |

#### 3.2.2 Circular Dependency Workarounds

Multiple lazy imports to avoid circular dependencies:
- `memory.py` - lazy import of utils
- `memory_librarian.py:2715` - lazy import of message_bus
- `reasoning_brain.py:3432` - lazy import of send function
- `self_dmn/service/self_critique.py` - circular import protection

#### 3.2.3 Missing Type Hints

Critical functions without type annotations:
- `_diag_enabled(ctx=None)` - No types
- `_diag_log(tag, rec)` - No types
- `_mem_call(payload)` - No types
- Multiple handler functions in language_brain.py

#### 3.2.4 Magic Numbers (50+ instances)

Examples requiring named constants:
```python
ttl = 300.0           # DEFAULT_TTL_SECONDS
max_records = 5000    # WM_MAX_RECORDS
threshold = 3         # MIN_PATTERN_THRESHOLD
"418" in error_lower  # HTTP_IM_A_TEAPOT
```

### 3.3 MEDIUM Quality Issues

#### TODO/FIXME Comments Indicating Technical Debt

| File | Count | Examples |
|------|-------|----------|
| reasoning_brain.py | 8 | "TODO: Improve synthesis logic" |
| learning_daemon.py | 1 | "TODO: needs task harness" |
| pattern_coder.py | 3 | Template placeholders |
| browser_client.py | 1 | "TODO: Implement selector extraction" |

#### Debug Print Statements in Production

| File | Count |
|------|-------|
| memory_librarian.py | 6+ `[DEBUG_STAGE8]` |
| language_brain.py | 12+ `[DEBUG]` |
| web_search_tool.py | 4+ `[WEB_SEARCH_DEBUG]` |

### 3.4 Quality Positives

- **Build Rules Documented:** `BUILD_RULES.md` defines clear architectural constraints
- **Service Contract Pattern:** All brains follow `handle(context)` contract
- **Tool Injection Pattern:** Clean separation of host tools from core brains
- **Memory API Invariants:** Clear tier management rules (STM → MTM → LTM → Archive)

---

## 4. Configuration Review

### 4.1 Security Configuration Issues

**config/tool_policy.json:**

| Policy | Current Value | Risk Level | Recommended |
|--------|---------------|------------|-------------|
| shell_policy.mode | `full_agency` | HIGH | `restricted` |
| filesystem_policy.mode | `unrestricted` | HIGH | `project_only` |
| python_policy.mode | `unrestricted` | CRITICAL | `sandbox` |
| web_policy.mode | `unrestricted` | MEDIUM | `whitelist` |
| agent_policy.mode | `full_agency` | HIGH | `supervised` |

### 4.2 Feature Flags (config/features.json)

| Feature | Current | Recommended | Reason |
|---------|---------|-------------|--------|
| full_agency_mode | true | **false** | Security risk |
| hot_reload | true | **false** | Production risk |
| autonomous_agents | true | **false** | Supervision needed |
| teacher_direct_fact_write | false | false | Keep disabled |
| teacher_proposal_mode | true | true | Safe |

### 4.3 API Keys Exposure

**config/api_keys.json:**
- Contains SerpAPI key in plaintext
- **Recommendation:** Move to environment variables or secrets manager

---

## 5. Test Failure Analysis

### 5.1 Category Breakdown

| Category | Failing | Root Cause |
|----------|---------|------------|
| Browser/Playwright | 144 | Optional dependency not installed |
| Self-Critique | 16 | Module API changes |
| Identity Inferencer | 9 | Service API changes |
| Execution Guard | 7 | API signature changes |
| Synonyms | 6 | Import path issues |
| Agency Tools | 8 | Permission/import issues |
| Action Engine | 5 | Execution guard integration |
| Context Management | 3 | Pattern object handling |
| Teacher Blocking | 4 | Question detection logic |
| Version Banner | 2 | Git integration |

### 5.2 Critical Fixes Required

1. **Missing `is_capability_enabled` function** - `capabilities.py:697`
2. **Self-critique module API mismatch** - 16 failing tests
3. **Execution guard API compatibility** - Multiple failures

---

## 6. OWASP Top 10 Compliance

| Vulnerability | Status | Details |
|---------------|--------|---------|
| A01 Broken Access Control | **VULNERABLE** | FULL_AGENCY bypass |
| A02 Cryptographic Failures | N/A | No crypto operations |
| A03 Injection | **VULNERABLE** | eval, exec, shell |
| A04 Insecure Design | **CONCERN** | Mode-based security |
| A05 Security Misconfiguration | **VULNERABLE** | Default unrestricted |
| A06 Vulnerable Components | REVIEW | Dependency audit needed |
| A07 Auth Failures | PARTIAL | No user auth system |
| A08 Software/Data Integrity | **VULNERABLE** | No code signing |
| A09 Logging/Monitoring | PARTIAL | Audit log exists |
| A10 SSRF | **VULNERABLE** | No URL validation |

---

## 7. Recommendations

### 7.1 Immediate Actions (Critical - 48 hours)

1. **Disable FULL_AGENCY mode by default**
   ```json
   // config/features.json
   "full_agency_mode": false
   ```

2. **Remove or sandbox eval() calls**
   - Replace with `ast.literal_eval()` in language_brain.py
   - Implement proper math expression parser

3. **Add import restrictions**
   - Never allow unrestricted imports
   - Maintain strict allowlist always

### 7.2 Short-Term Actions (High - 1 week)

4. **Improve command injection protection**
   - Replace substring matching with command parsing
   - Use shlex for proper argument analysis

5. **Add code signing for self-upgrade**
   - Implement signature verification
   - Require approval for code changes

6. **Fix silent exception handling**
   - Add logging to all exception handlers
   - Use specific exception types

### 7.3 Medium-Term Actions (2-4 weeks)

7. **Refactor large files**
   - Split `memory_librarian.py` (11,288 lines)
   - Split `language_brain.py` (8,217 lines)

8. **Resolve circular dependencies**
   - Restructure imports to eliminate lazy loading
   - Consider dependency injection pattern

9. **Add comprehensive type hints**
   - Complete type annotations for public APIs
   - Enable mypy strict mode

10. **Extract magic numbers to constants**

### 7.4 Long-Term Actions (1-3 months)

11. **Implement capability-based security**
    - Replace mode-based with granular capability grants
    - Require per-operation approval

12. **Add integration tests**
    - Improve test coverage
    - Add security-focused test cases

13. **Implement proper sandboxing**
    - Use subprocess for code execution
    - Consider container isolation

---

## 8. Conclusion

### 8.1 Overall Assessment

Maven 2.0 demonstrates sophisticated AI architecture with its multi-brain cognitive pipeline and tiered memory system. The codebase follows many good practices including:
- Clean separation of core brains from host tools
- Well-documented build rules
- Tiered memory management
- Audit logging

However, the FULL_AGENCY mode creates **significant security risks** by effectively disabling authorization controls. The system should not be deployed in production without addressing the critical vulnerabilities.

### 8.2 Risk Summary

| Area | Risk Level |
|------|------------|
| Overall Security | **HIGH** |
| Code Quality | MEDIUM |
| Architecture | LOW-MEDIUM |
| Test Coverage | MEDIUM |

### 8.3 Priority Actions

1. **IMMEDIATE:** Disable FULL_AGENCY mode by default
2. **IMMEDIATE:** Sandbox all eval()/exec() calls
3. **SHORT-TERM:** Improve command injection protection
4. **SHORT-TERM:** Add code signing for self-upgrades
5. **MEDIUM-TERM:** Refactor large files and fix circular dependencies

---

## Appendix A: Files Reviewed

- `brains/tools/execution_guard.py`
- `brains/tools/hot_reload.py`
- `brains/tools/self_upgrade_tool.py`
- `brains/agent/tools/python_exec.py`
- `host_tools/shell_executor/executor.py`
- `host_tools/web_client/client.py`
- `host_tools/git_client/client.py`
- `brains/cognitive/language/service/language_brain.py`
- `brains/cognitive/memory_librarian/service/memory_librarian.py`
- `brains/cognitive/reasoning/service/reasoning_brain.py`
- `config/*.json` (all 23 configuration files)
- `BUILD_RULES.md`
- `requirements*.txt` (all 4 requirements files)

## Appendix B: Audit Methodology

1. **Architecture Exploration:** Full codebase structure analysis
2. **Security Analysis:** Manual code review for vulnerabilities
3. **Code Quality Review:** Static analysis patterns
4. **Configuration Review:** Security settings analysis
5. **Test Suite Analysis:** Coverage and failure assessment

---

*Report generated by Claude Code (Opus 4) - Full System Audit*
*Date: 2025-12-01*
