# Governance–Repair–Reasoning Contract (Authoritative)

## Scope
Governance controls **system operations** only. It never makes decisions about truth or storage.

## Authority (ALLOW/DENY)
- **MUST approve** before execution:
  - INIT_GOLDEN_FROM_BASELINE
  - LOAD_CANDIDATE
  - CANARY_TEST
  - PROMOTE_TEMPLATE
  - ROLLBACK_TEMPLATE
  - ROLLBACK_TO_GOLDEN
  - REPAIR_BRAIN
  - REPAIR_HEALTHCHECK
  - Any write/delete under `brains/**` or `templates/**`

- **MUST deny**:
  - Any write/delete under `templates/golden/**`

- **AUDIT-ONLY (never blocks)**:
  - Reasoning verdicts and Memory Librarian routing
  - Personality tone/bias summaries
  - **Working Memory operations** including `WM_PUT`, `WM_GET`, `WM_DUMP` and `CONTROL_TICK`.
    These operations enable cross-module communication via the shared working memory and
    are always logged for audit.  Governance does not block them but records each
    invocation in the governance ledger for traceability.

## Separation of Concerns
- Repair executes changes **after** Governance ALLOW.
- Reasoning is final on truth/inference.
- Librarian routes/stores per Reasoning, without Governance checks.
- Governance never alters facts or storage; it only authorizes operations and logs them.
