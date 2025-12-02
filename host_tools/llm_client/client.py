"""
Host LLM Client Implementation
==============================

Concrete implementation of LLM completion tool using local Ollama
or other LLM providers. This module performs actual network I/O
and should NOT be imported by core brains.

The host runtime creates instances of this tool and injects them
into the brain context via ToolRegistry.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from brains.tools_api import LLMCompletionResult


class HostLLMTool:
    """
    Host implementation of LLM completion using Ollama.

    Satisfies the LLMTool protocol from brains.tools_api.
    Supports pattern learning to reduce LLM calls over time.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.base_url: str = self.config.get("ollama_url", "http://localhost:11434")
        self.model: str = self.config.get("model", "llama3.2")
        self._enabled: bool = bool(self.config.get("enabled", True))

        # Pattern learning paths
        self._patterns_dir: Optional[Path] = None
        self._templates_file: Optional[Path] = None
        self._interaction_log: Optional[Path] = None
        self._init_learning_paths()

    def _init_learning_paths(self) -> None:
        """Initialize pattern learning storage paths."""
        try:
            from brains.maven_paths import get_brains_path
            self._patterns_dir = get_brains_path("personal", "memory", "learned_patterns")
            self._patterns_dir.mkdir(parents=True, exist_ok=True)
            self._templates_file = self._patterns_dir / "learned_templates.json"
            self._interaction_log = self._patterns_dir / "llm_interactions.jsonl"
        except Exception:
            pass

    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load LLM configuration from config file."""
        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    return json.load(fh) or {}
            except Exception:
                pass

        # Try default location
        try:
            from brains.maven_paths import get_maven_root
            cfg_path = get_maven_root() / "config" / "llm.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    return json.load(fh) or {}
        except Exception:
            pass

        # Default configuration
        return {
            "enabled": True,
            "provider": "ollama",
            "ollama_url": "http://localhost:11434",
            "model": "llama3.2",
            "learning": {
                "enabled": True,
                "min_interactions_to_learn": 10,
                "similarity_threshold": 0.8,
                "min_pattern_occurrences": 3,
            },
        }

    @property
    def enabled(self) -> bool:
        """Whether the LLM service is enabled."""
        return self._enabled

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMCompletionResult:
        """
        Generate a completion for the given prompt.

        Attempts to use a learned template first. If no reliable
        template exists, calls the Ollama API.
        """
        if not self._enabled:
            return LLMCompletionResult(
                ok=False,
                text="",
                source="disabled",
                llm_used=False,
                error="LLM disabled"
            )

        prompt_hash = self._hash_prompt(prompt)

        # Try a learned template first
        template_res = self._try_template(prompt_hash, context)
        if template_res:
            return LLMCompletionResult(
                ok=True,
                text=template_res["text"],
                source="learned_template",
                llm_used=False,
                confidence=template_res["confidence"]
            )

        # Fall back to calling the LLM via HTTP
        try:
            parts = urllib.parse.urlparse(self.base_url)
            conn = http.client.HTTPConnection(
                parts.hostname,
                parts.port or 80,
                timeout=30
            )

            path = parts.path.rstrip("/")
            if not path.endswith("/api"):
                path = path + "/api"
            path = path.rstrip("/") + "/generate"

            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            })

            headers = {"Content-Type": "application/json"}
            conn.request("POST", path, body=payload, headers=headers)
            resp = conn.getresponse()

            if resp.status == 200:
                data = resp.read()
                try:
                    result = json.loads(data.decode())
                except Exception:
                    result = {}
                text = result.get("response", "")

                # Log the interaction for learning
                self._log_interaction(prompt, prompt_hash, text, context)

                return LLMCompletionResult(
                    ok=True,
                    text=text,
                    source="ollama",
                    llm_used=True
                )

            return LLMCompletionResult(
                ok=False,
                text="",
                source="ollama",
                llm_used=False,
                error=f"Status {resp.status}"
            )

        except ConnectionRefusedError:
            return LLMCompletionResult(
                ok=False,
                text="",
                source="ollama",
                llm_used=False,
                error="Ollama connection refused. Is Ollama running? Try: 'ollama serve'",
                error_type="connection_refused"
            )
        except Exception as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                error_msg = f"Ollama connection timed out: {error_msg}. Is Ollama running?"
            return LLMCompletionResult(
                ok=False,
                text="",
                source="ollama",
                llm_used=False,
                error=error_msg,
                error_type=type(e).__name__
            )

    def _hash_prompt(self, prompt: str) -> str:
        """Create a stable hash for prompt pattern matching."""
        normalized = " ".join(str(prompt or "").lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _try_template(
        self,
        prompt_hash: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Try to return a response from a learned template."""
        templates = self._load_templates()
        tpl = templates.get(prompt_hash)
        if tpl and tpl.get("success_count", 0) >= 3:
            tpl["use_count"] = tpl.get("use_count", 0) + 1
            tpl["last_used"] = datetime.now().isoformat()
            self._save_templates(templates)
            return self._apply_template(tpl, context)
        return None

    def _apply_template(
        self,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply a learned template with context substitution."""
        text = template.get("response_template", "")
        if context:
            user = context.get("user") or {}
            user_name = user.get("name")
            if user_name and "{user_name}" in text:
                text = text.replace("{user_name}", str(user_name))
        return {
            "text": text,
            "confidence": template.get("confidence", 0.8),
        }

    def _log_interaction(
        self,
        prompt: str,
        prompt_hash: str,
        response: str,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Log an interaction for pattern learning."""
        if not self._interaction_log:
            return
        record = {
            "timestamp": datetime.now().isoformat(),
            "prompt_hash": prompt_hash,
            "prompt": prompt[:200],
            "response": response,
            "context": {
                "query_type": (context or {}).get("query_type"),
                "user_name": (context or {}).get("user", {}).get("name"),
            },
        }
        try:
            with open(self._interaction_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _load_templates(self) -> Dict[str, Any]:
        """Load learned templates from disk."""
        if not self._templates_file or not self._templates_file.exists():
            return {}
        try:
            with open(self._templates_file, "r", encoding="utf-8") as fh:
                return json.load(fh) or {}
        except Exception:
            return {}

    def _save_templates(self, templates: Dict[str, Any]) -> None:
        """Save templates to disk."""
        if not self._templates_file:
            return
        try:
            with open(self._templates_file, "w", encoding="utf-8") as fh:
                json.dump(templates, fh, indent=2)
        except Exception:
            pass

    def learn_patterns(self) -> None:
        """Analyze logged interactions and create/update templates."""
        if not self.config.get("learning", {}).get("enabled", True):
            return
        if not self._interaction_log or not self._interaction_log.exists():
            return

        interactions = self._load_interactions()
        min_occ = int(
            self.config.get("learning", {}).get("min_pattern_occurrences", 3)
        )
        if len(interactions) < min_occ:
            return

        threshold = float(
            self.config.get("learning", {}).get("similarity_threshold", 0.8)
        )

        # Group by prompt hash
        groups: Dict[str, List[Dict]] = {}
        for rec in interactions:
            h = rec.get("prompt_hash")
            if h:
                groups.setdefault(h, []).append(rec)

        templates = self._load_templates()
        for h, recs in groups.items():
            if len(recs) < min_occ:
                continue
            responses = [str(r.get("response", "")) for r in recs]
            sim = self._compute_similarity(responses)
            if sim >= threshold:
                template_text = self._generalize_responses(responses)
                templates[h] = {
                    "prompt_pattern": recs[0].get("prompt"),
                    "response_template": template_text,
                    "confidence": sim,
                    "success_count": len(recs),
                    "created_at": datetime.now().isoformat(),
                    "use_count": templates.get(h, {}).get("use_count", 0),
                    "examples": recs[:3],
                }
        self._save_templates(templates)

    def _load_interactions(self) -> List[Dict[str, Any]]:
        """Load logged interactions."""
        if not self._interaction_log or not self._interaction_log.exists():
            return []
        interactions: List[Dict] = []
        try:
            with open(self._interaction_log, "r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        try:
                            interactions.append(json.loads(ln))
                        except Exception:
                            continue
        except Exception:
            pass
        return interactions

    def _compute_similarity(self, responses: List[str]) -> float:
        """Compute Jaccard similarity across responses."""
        if len(responses) < 2:
            return 0.0
        term_sets = [set(str(resp).lower().split()) for resp in responses]
        common = set.intersection(*term_sets) if term_sets else set()
        all_terms = set.union(*term_sets) if term_sets else set()
        return len(common) / len(all_terms) if all_terms else 0.0

    def _generalize_responses(self, responses: List[str]) -> str:
        """Return the most frequent response from a list."""
        counter = Counter(responses)
        return counter.most_common(1)[0][0] if counter else ""
