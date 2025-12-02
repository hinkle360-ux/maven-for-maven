# Theories & Contradictions Domain Bank
Path: `brains/domain_banks/theories_and_contradictions/service/theories_and_contradictions_bank.py`

Ops:
- STORE_THEORY
- STORE_CONTRADICTION
- RETRIEVE
- COUNT
- HEALTH

Persistence: JSONL tiers created at runtime via `api.memory.ensure_dirs`.
(No `__init__.py` anywhere.)
