from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any, List
from brains.memory.brain_memory import BrainMemory

HERE = Path(__file__).resolve().parent
STORE = HERE / "router_memory"
STORE.mkdir(exist_ok=True)

VOCAB_FILE = STORE / "bank_vocab.json"
DEFS_FILE = STORE / "definitions.json"

STOP = {
    "a","an","the","and","or","but","is","are","was","were","be","being","been",
    "of","to","in","on","for","by","with","as","at","from","that","this","it",
    "its","their","there","then","than","so","if","into","about","over","under"
}

BANKS = ["arts","science","history","economics","geography","language_arts","law","math","philosophy","technology"]

# Initialize BrainMemory for reasoning brain
_memory = BrainMemory("reasoning")

def _load_json(p: Path, default):
    """Load data from BrainMemory instead of file system"""
    try:
        # Determine kind from file path
        if "bank_vocab" in str(p):
            kind = "bank_vocab"
        elif "definitions" in str(p):
            kind = "definitions"
        else:
            kind = "router_data"

        results = _memory.retrieve(query=f"kind:{kind}", limit=1)
        if results:
            return results[0].get("content", default)
    except Exception:
        pass
    return default

def _save_json(p: Path, data):
    """Save data to BrainMemory instead of file system"""
    try:
        # Determine kind from file path
        if "bank_vocab" in str(p):
            kind = "bank_vocab"
        elif "definitions" in str(p):
            kind = "definitions"
        else:
            kind = "router_data"

        _memory.store(
            content=data,
            metadata={"kind": kind, "source": "reasoning", "confidence": 0.9}
        )
    except Exception:
        pass

def _tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in STOP and len(t) > 1]

def _top_vocab(vocab: Dict[str,int], top_k=200):
    return dict(sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])

def _defs_match(tokens: List[str], defs: Dict[str, Dict[str,int]]):
    # returns list of "term->klass" signals if term and class tokens both present
    sigs = []
    for term, m in defs.items():
        klasses = m or {}
        for klass, cnt in klasses.items():
            if term in tokens or klass in tokens:
                # soft match: if either appears, signal
                sigs.append(f"definition:{term}->{klass}")
    return sigs

def _handle_impl(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = (msg or {}).get("op","").upper()
    payload = (msg or {}).get("payload") or {}

    if op == "LEARN":
        text = str(payload.get("text",""))
        bank = str(payload.get("bank","")).strip().lower()
        if bank not in BANKS:
            return {"ok": True, "payload": {"skipped": True, "reason": "unknown_bank"}}
        vocab = _load_json(VOCAB_FILE, {b:{} for b in BANKS})
        toks = _tokens(text)
        bv = vocab.get(bank) or {}
        for t in toks:
            bv[t] = int(bv.get(t,0)) + 1
        vocab[bank] = bv
        _save_json(VOCAB_FILE, vocab)
        return {"ok": True, "payload": {"learned": len(toks)}}

    if op == "LEARN_DEFINITION":
        # expects term and klass extracted upstream
        term = str(payload.get("term","")).lower().strip()
        klass = str(payload.get("klass","")).lower().strip()
        if not term or not klass:
            return {"ok": True, "payload": {"skipped": True, "reason": "empty"}}
        defs = _load_json(DEFS_FILE, {})
        mp = defs.get(term) or {}
        mp[klass] = int(mp.get(klass,0)) + 1
        defs[term] = mp
        _save_json(DEFS_FILE, defs)
        return {"ok": True, "payload": {"term": term, "klass": klass}}

    if op == "ROUTE":
        text = str(payload.get("text",""))
        toks = _tokens(text)
        vocab = _load_json(VOCAB_FILE, {b:{} for b in BANKS})
        defs = _load_json(DEFS_FILE, {})
        scores = {}
        for bank in BANKS:
            bank_vocab = _top_vocab(vocab.get(bank) or {})
            # simple overlap score
            overlap = sum(bank_vocab.get(t,0) for t in set(toks))
            norm = max(len(toks), 1)
            scores[bank] = overlap / norm
        # pick max score
        target = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "arts"
        signals = _defs_match(toks, defs)
        return {"ok": True, "payload": {"target_bank": target, "scores": scores, "signals": signals}}

    if op == "HEALTH":
        vocab = _load_json(VOCAB_FILE, {b:{} for b in BANKS})
        defs = _load_json(DEFS_FILE, {})
        return {"ok": True, "payload": {"banks": {b: len(vocab.get(b) or {}) for b in BANKS}, "defs": len(defs)}}

    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}


def handle(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Brain contract entry point.

    Args:
        context: Standard brain context dict

    Returns:
        Result dict
    """
    return _handle_impl(context)


# Brain contract alias
service_api = handle
