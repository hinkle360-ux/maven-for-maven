"""
Agency Routing Patterns
=======================

Routing patterns that map user queries to agency tool capabilities.
This prevents the echo bug by routing tool-requiring queries directly
to the appropriate agency modules instead of falling back to Teacher.

Pattern Format:
{
    "signatures": ["keyword1", "keyword2", ...],
    "tool": "module_path.function_name",
    "confidence": 0.9,
    "bypass_teacher": True
}
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional


# Agency tool routing patterns
AGENCY_PATTERNS = [
    # Self-Introspection Patterns
    {
        "signatures": [
            "scan codebase", "scan code", "scan yourself", "scan your code",
            "introspect", "self introspect", "analyze codebase",
            "analyze your code", "analyze yourself", "scan directory",
            "scan files", "scan python files", "scan brains",
            "scan architecture", "analyze architecture"
        ],
        "tool": "brains.tools.self_introspection.get_self_introspection",
        "method": "generate_self_knowledge_report",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Full codebase introspection and analysis"
    },
    {
        "signatures": [
            "list brains", "show brains", "what brains", "brain list",
            "brain analysis", "analyze brains", "brain compliance",
            "check brains", "brain status"
        ],
        "tool": "brains.tools.self_introspection.get_self_introspection",
        "method": "analyze_all_brains",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Analyze brain compliance and structure"
    },
    {
        "signatures": [
            "circular dependencies", "dependency cycles", "import cycles",
            "check dependencies", "analyze dependencies", "dependency graph"
        ],
        "tool": "brains.tools.self_introspection.get_self_introspection",
        "method": "analyze_dependencies",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Analyze code dependencies"
    },

    # Filesystem Agency Patterns
    {
        "signatures": [
            "scan directory tree", "list directory", "show files",
            "list files", "directory structure", "file tree",
            "show directory", "explore files"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "scan_directory_tree",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Scan filesystem directory structure"
    },
    {
        "signatures": [
            "list python files", "show python files", "find python files",
            "python file list", "py files"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "list_python_files",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "List all Python files"
    },
    {
        "signatures": [
            "read file", "show file", "display file", "file contents",
            "view file", "cat file"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "read_file",
        "confidence": 0.85,
        "bypass_teacher": True,
        "description": "Read file contents",
        "requires_args": True
    },
    {
        "signatures": [
            "analyze python file", "parse python", "analyze code",
            "code structure", "find class", "find function",
            "class definition", "function definition"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "analyze_python_file",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Analyze Python file structure"
    },

    # Git Operations Patterns
    {
        "signatures": [
            "git status", "repo status", "repository status",
            "git state", "check git", "git info"
        ],
        "tool": "brains.tools.git_tool.git_get_repo_info",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Get complete repository information"
    },
    {
        "signatures": [
            "git log", "commit log", "commit history",
            "show commits", "recent commits"
        ],
        "tool": "brains.tools.git_tool.git_log",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Show git commit log"
    },
    {
        "signatures": [
            "git branches", "list branches", "show branches",
            "branch list", "all branches"
        ],
        "tool": "brains.tools.git_tool.git_list_branches",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "List git branches"
    },
    {
        "signatures": [
            "current branch", "which branch", "branch name",
            "active branch"
        ],
        "tool": "brains.tools.git_tool.git_current_branch",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Get current branch name"
    },

    # Hot-Reload Patterns
    {
        "signatures": [
            "reload module", "hot reload", "reload brain",
            "refresh module", "reload code"
        ],
        "tool": "brains.tools.hot_reload.get_hot_reload_manager",
        "method": "reload_module",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Reload Python module at runtime",
        "requires_args": True
    },
    {
        "signatures": [
            "loaded modules", "list modules", "show modules",
            "module list", "maven modules"
        ],
        "tool": "brains.tools.hot_reload.get_hot_reload_manager",
        "method": "get_loaded_maven_modules",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "List loaded Maven modules"
    },

    # Coder Patterns - Code Generation
    {
        "signatures": [
            "coder:", "use coder:", "coder write", "coder generate",
            "write a function", "write a python function",
            "generate code", "generate a function", "create function",
            "write code", "code a function"
        ],
        "tool": "brains.cognitive.coder.service.coder_brain",
        "method": "GENERATE",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Generate code using the coder brain",
        "return_code_directly": True
    },
    {
        "signatures": [
            "return only the code", "return only code", "just the code",
            "show me the code", "give me the code", "code only",
            "return only the full updated function", "return only the function"
        ],
        "tool": "brains.cognitive.coder.service.coder_brain",
        "method": "EXTEND",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Return extended code output directly",
        "return_code_directly": True,
        "is_code_follow_up": True
    },
    # Coder Patterns - Function Extension
    {
        "signatures": [
            "extend that function", "extend the function",
            "modify that function", "update that function",
            "change that function", "add to that function"
        ],
        "tool": "brains.cognitive.coder.service.coder_brain",
        "method": "EXTEND",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Extend the last generated function with new specs",
        "return_code_directly": True,
        "is_extension": True
    },
    # Coder Patterns - Bullet Point Specs (accumulate for extension)
    {
        "signatures": [
            "- if both", "- if one", "if both are lists",
            "if one is a scalar", "broadcast it"
        ],
        "tool": "brains.cognitive.coder.service.coder_brain",
        "method": "ADD_SPEC",
        "confidence": 0.85,
        "bypass_teacher": True,
        "description": "Add a specification for function extension",
        "is_spec_accumulation": True
    },

    # Normalization Introspection Patterns
    {
        "signatures": [
            "last normalized", "normalized message", "last normalized user message",
            "show me your last normalized", "normalized input",
            "what did you normalize", "your last normalized"
        ],
        "tool": "brains.cognitive.sensorium.service.sensorium_brain",
        "method": "GET_LAST_NORMALIZED",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Get the last normalized user message",
        "direct_memory_query": True
    },

    # Inventory Patterns
    {
        "signatures": [
            "inventory:", "inventory: list", "list all cognitive brains",
            "list cognitive brains", "brain inventory", "inventory brains",
            "count brains", "how many brains", "enumerate brains",
            "inventory: list all cognitive brains",
            "report the total count"
        ],
        "tool": "brains.cognitive.inventory.service.inventory_brain",
        "method": "LIST",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "List all cognitive brains with count"
    },

    # Browser Patterns - Web Browsing Automation
    # NOTE: tool name should match what's in system_capabilities (browser_runtime)
    {
        "signatures": [
            "browser:", "browser: open", "open https://", "open http://",
            "fetch url", "browse to", "open website", "visit website",
            "browser open", "browser fetch", "open url", "navigate to",
            "go to website", "load page", "load website", "open google",
            "go to google", "open example", "go to example"
        ],
        "tool": "browser_runtime",
        "method": "OPEN_URL",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Open a URL in the browser"
    },
    # Default search (DuckDuckGo - no CAPTCHA)
    {
        "signatures": [
            "search for", "search", "look up", "find", "web search",
            "search the web", "find online", "look up online",
            "search online", "look for", "query"
        ],
        "tool": "browser_runtime",
        "method": "SEARCH",
        "engine": "ddg",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Perform a web search (default: DuckDuckGo)"
    },
    # Google-specific search
    {
        "signatures": [
            "google", "google search", "search google", "google for",
            "search on google", "use google", "google it",
            "open google search"
        ],
        "tool": "browser_runtime",
        "method": "SEARCH",
        "engine": "google",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Search using Google (may require CAPTCHA)"
    },
    # Bing-specific search
    {
        "signatures": [
            "bing", "bing search", "search bing", "bing for",
            "search on bing", "use bing", "open bing search"
        ],
        "tool": "browser_runtime",
        "method": "SEARCH",
        "engine": "bing",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Search using Bing"
    },
    # DuckDuckGo-specific search
    {
        "signatures": [
            "ddg", "duckduckgo", "duck duck go", "ddg search",
            "search ddg", "search duckduckgo", "use duckduckgo",
            "open duckduckgo search"
        ],
        "tool": "browser_runtime",
        "method": "SEARCH",
        "engine": "ddg",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Search using DuckDuckGo (recommended, no CAPTCHA)"
    },
    {
        "signatures": [
            "browse for", "browse the web for", "browser: search",
            "browser: browse", "use browser to", "use the browser"
        ],
        "tool": "browser_runtime",
        "method": "BROWSE",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Execute a browsing task"
    },

    # =========================================================================
    # WEB SEARCH TOOL - Direct web search with result synthesis
    # =========================================================================
    # These patterns go directly to web_search_tool for synthesized answers
    {
        "signatures": [
            "what new games", "what games are", "new games coming out",
            "latest games", "upcoming games", "recent games",
            "games releasing", "game releases"
        ],
        "tool": "brains.agent.tools.web_search_tool.web_search_tool",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Search for game release information",
        "is_web_search": True
    },
    {
        "signatures": [
            "search the web", "web search for", "search online for",
            "look up online", "find online", "search internet for",
            "look online for", "online search"
        ],
        "tool": "brains.agent.tools.web_search_tool.web_search_tool",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Perform web search with answer synthesis",
        "is_web_search": True
    },
    {
        "signatures": [
            "news about", "latest news", "recent news",
            "what's happening with", "updates on", "current news"
        ],
        "tool": "brains.agent.tools.web_search_tool.web_search_tool",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Search for news and current events",
        "is_web_search": True
    },
    {
        "signatures": [
            "what is the latest", "what are the newest",
            "what's new in", "what's coming in", "upcoming"
        ],
        "tool": "brains.agent.tools.web_search_tool.web_search_tool",
        "confidence": 0.85,
        "bypass_teacher": True,
        "description": "Search for latest/upcoming information",
        "is_web_search": True
    },

    # Diagnostic Patterns
    {
        "signatures": [
            "diagnose", "diagnostic", "system check", "health check",
            "system status", "check system", "troubleshoot",
            "find issues", "detect problems"
        ],
        "tool": "brains.tools.self_introspection.get_self_introspection",
        "method": "generate_self_knowledge_report",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Run system diagnostics"
    },
    {
        "signatures": [
            "missing functions", "missing methods", "detect missing",
            "find missing", "function detection"
        ],
        "tool": "brains.tools.self_introspection.get_self_introspection",
        "method": "detect_missing_functions",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Detect missing functions in brains"
    },

    # =========================================================================
    # FULL_AGENCY MODE - Shell and Command Execution Patterns
    # =========================================================================
    {
        "signatures": [
            "run command", "execute command", "run shell", "shell command",
            "terminal command", "run in terminal", "execute in shell",
            "run this command", "execute this", "do command"
        ],
        "tool": "brains.agent.tools.intent_resolver_tools.maybe_execute_tool",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Execute a shell command",
        "requires_args": True
    },
    {
        "signatures": [
            "install package", "pip install", "npm install", "apt install",
            "install with pip", "install with npm", "add package"
        ],
        "tool": "brains.agent.tools.intent_resolver_tools.maybe_execute_tool",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Install a package",
        "requires_args": True
    },

    # Python Execution Patterns
    {
        "signatures": [
            "run python", "execute python", "run this python",
            "python code", "run python code", "execute python code",
            "run script", "execute script"
        ],
        "tool": "brains.agent.tools.intent_resolver_tools.maybe_execute_tool",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Execute Python code",
        "requires_args": True
    },

    # Web Operations Patterns
    {
        "signatures": [
            "fetch url", "get url", "download from", "open url",
            "browse to url", "visit url", "navigate to url",
            "http get", "http fetch"
        ],
        "tool": "browser_runtime",
        "method": "FETCH_URL",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Fetch content from a URL",
        "requires_args": True
    },
    {
        "signatures": [
            "search online", "web search", "internet search",
            "look up online", "find online", "search internet"
        ],
        "tool": "browser_runtime",
        "method": "SEARCH",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Perform a web search"
    },

    # Advanced Git Operations
    {
        "signatures": [
            "git commit", "commit changes", "make commit",
            "commit with message", "save changes to git"
        ],
        "tool": "brains.tools.git_tool.git_commit",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Commit changes to git",
        "requires_args": True
    },
    {
        "signatures": [
            "git push", "push changes", "push to remote",
            "push to origin", "upload changes"
        ],
        "tool": "brains.tools.git_tool.git_push",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Push changes to remote"
    },
    {
        "signatures": [
            "git pull", "pull changes", "pull from remote",
            "pull from origin", "download changes", "sync repo"
        ],
        "tool": "brains.tools.git_tool.git_pull",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Pull changes from remote"
    },
    {
        "signatures": [
            "git clone", "clone repo", "clone repository",
            "download repo", "get repo"
        ],
        "tool": "brains.tools.git_tool.git_clone",
        "confidence": 0.95,
        "bypass_teacher": True,
        "description": "Clone a git repository",
        "requires_args": True
    },

    # File Operation Patterns (Natural Language)
    {
        "signatures": [
            "delete file", "remove file", "rm file",
            "delete the file", "remove the file"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "delete_file",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Delete a file",
        "requires_args": True
    },
    {
        "signatures": [
            "copy file", "cp file", "duplicate file",
            "copy the file", "make a copy"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "copy_file",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Copy a file",
        "requires_args": True
    },
    {
        "signatures": [
            "move file", "mv file", "rename file",
            "move the file", "rename the file"
        ],
        "tool": "brains.tools.filesystem_agency.get_filesystem_agency",
        "method": "move_file",
        "confidence": 0.9,
        "bypass_teacher": True,
        "description": "Move or rename a file",
        "requires_args": True
    }
]


def match_agency_pattern(query: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """
    Match a user query against agency routing patterns.

    Args:
        query: User query text (lowercased)
        threshold: Minimum confidence threshold

    Returns:
        Matched pattern dict or None if no match
    """
    import re
    query_lower = query.lower()
    # Remove punctuation and split into words
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    best_match = None
    best_score = 0.0

    for pattern in AGENCY_PATTERNS:
        # Calculate match score
        matches = 0
        for signature in pattern["signatures"]:
            sig_words = set(re.findall(r'\b\w+\b', signature))
            # Check if all signature words are in the query (order-independent)
            if sig_words.issubset(query_words):
                matches += 1

        if matches > 0:
            # Score based on pattern confidence (any signature match gives full confidence)
            # Use a bonus for multiple matches but don't penalize patterns with many alternatives
            match_bonus = min(matches * 0.05, 0.2)  # Up to 20% bonus for multiple matches
            score = pattern["confidence"] + match_bonus

            if score > best_score and score >= threshold:
                best_score = score
                best_match = pattern.copy()
                best_match["match_score"] = score

    return best_match


def _is_tool_execution_allowed() -> bool:
    """Check if tool execution is allowed based on current profile/mode."""
    try:
        from brains.tools.execution_guard import get_execution_status, ExecMode
        status = get_execution_status()

        # SAFE_CHAT mode - no tools allowed
        if status.mode == ExecMode.SAFE_CHAT:
            return False

        # DISABLED mode - no tools allowed
        if status.mode == ExecMode.DISABLED:
            return False

        # FULL_AGENCY or FULL mode with effective=True - tools allowed
        if status.mode in (ExecMode.FULL_AGENCY, ExecMode.FULL) and status.effective:
            return True

        # READ_ONLY mode - only read operations allowed (partial)
        if status.mode == ExecMode.READ_ONLY:
            return True

        return False
    except Exception:
        return False


def should_bypass_teacher(query: str) -> bool:
    """
    Determine if a query should bypass Teacher and go directly to agency tools.

    This function respects the current execution profile:
    - In SAFE_CHAT mode, always returns False (no tool access)
    - In FULL_AGENCY mode, routes to tools as appropriate
    - In other modes, checks execution status

    Args:
        query: User query text

    Returns:
        True if query should use agency tools directly
    """
    # Check if tool execution is allowed
    if not _is_tool_execution_allowed():
        return False

    pattern = match_agency_pattern(query)
    return pattern is not None and pattern.get("bypass_teacher", False)


def get_tool_for_query(query: str) -> Optional[str]:
    """
    Get the tool path for a query.

    Args:
        query: User query text

    Returns:
        Tool import path or None
    """
    pattern = match_agency_pattern(query)
    return pattern["tool"] if pattern else None


def get_method_for_query(query: str) -> Optional[str]:
    """
    Get the method name for a query.

    Args:
        query: User query text

    Returns:
        Method name or None
    """
    pattern = match_agency_pattern(query)
    return pattern.get("method") if pattern else None


def explain_routing_decision(query: str) -> Dict[str, Any]:
    """
    Explain why a query was routed a particular way.

    Args:
        query: User query text

    Returns:
        Dictionary explaining routing decision
    """
    pattern = match_agency_pattern(query)

    if pattern:
        return {
            "routed_to": "agency_tool",
            "tool": pattern["tool"],
            "method": pattern.get("method"),
            "confidence": pattern["match_score"],
            "description": pattern["description"],
            "bypass_teacher": pattern["bypass_teacher"]
        }
    else:
        return {
            "routed_to": "teacher_fallback",
            "reason": "No agency pattern matched",
            "suggestion": "Add routing pattern or improve query specificity"
        }


# Export all patterns for external use
def get_all_patterns() -> List[Dict[str, Any]]:
    """Get all agency routing patterns."""
    return AGENCY_PATTERNS.copy()
