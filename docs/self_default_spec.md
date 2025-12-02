# Self‑Default Dissent & Adjudication Specification

This document describes the Self‑Default brain introduced in the Maven architecture to provide an internal mechanism for dissent, skepticism, and claim adjudication.  It outlines roles, data structures, state transitions, and guardrails.

## Purpose

Self‑Default catches groupthink and stale “facts,” forces explicit uncertainty handling, and creates an audit trail of why the system’s confidence changed.

## Composition

| Component | Function |
|----------|----------|
| **Advocates** | Existing cognitive brains (arts, math, history, etc.) produce claims and evidence. |
| **Skeptic** | Challenges each claim by finding counterevidence or logical gaps. |
| **Judge** | Applies deterministic status rules and manages the recompute queue. |
| **Orchestrator** | Owns thresholds, budgets, TTLs, and writes the audit trail. |

## Claim record schema

```
{
  "claim_id": "uuid",
  "proposition": "normalized text",
  "support": ["reasoning: ...", "history: ..."],
  "consensus_score": 0.83,
  "skeptic_score": 0.42,
  "status": "undisputed",  // one of: undisputed | disputed | recompute
  "expiry": "2025-11-04T10:30Z",
  "rationale": "no strong counterevidence; fresh sources"
}
```

## Status logic

Let \(\tau_1\) and \(\tau_2\) be thresholds (configured in CFG["self_default"]).  The status transitions are:

1. If \(\text{skeptic_score} - \text{consensus_score} \ge \tau_1\), set status to **recompute**.
2. Else if \(\text{skeptic_score} \ge \tau_2\), set status to **disputed**.
3. Otherwise, set status to **undisputed**.

Default thresholds: \(\tau_1 = 0.25\), \(\tau_2 = 0.60\).

## Trigger conditions for Skeptic

- **Conflict**: Two or more brains assert incompatible propositions.
- **Novelty**: New high‑impact evidence arrives (higher source score or more recent).
- **Staleness**: Claim has passed its expiry time.
- **Criticality**: Claim gates a risky action.
- **Drift**: Current context embedding drifts beyond a threshold from the context of the claim.

## Recompute routine

When a claim enters the **recompute** state, the Judge triggers a targeted re‑reasoning process:

1. **Adversarial prompts**: Ask for counterexamples and alternative hypotheses.
2. **Source re‑ranking**: Down‑weight stale or low‑reliability sources.
3. **Counterfactual search**: Explore scenarios that would falsify the claim.
4. **Re‑score**: Compute new consensus and skeptic scores; update status accordingly.

## Guardrails

- **Budget cap per claim**: Limit the number of recompute steps, tokens, and time spent.
- **Backoff**: If repeated recomputes yield no status change, suspend recompute for a TTL.
- **Escalation**: Only alert humans when a high‑risk claim remains disputed.

## Surfacing and badges

Claims are annotated with badges wherever they are used:

| Badge | Meaning |
|-------|---------|
| ✓ | Undisputed (with timestamp) |
| ⚠ | Disputed (show rationale) |
| ⟳ | Recompute (work queued) |

## Scoring recipe

- **Consensus score**: Calibrated mean of brain confidences plus a diversity bonus (to discourage correlated agreement).
- **Skeptic score**: Maximum of counterevidence strength, logical flaw likelihood, and out‑of‑distribution signal.
- **Entropy prior**: Slowly decays scores toward uncertainty over time unless reaffirmed.

## Outputs and persistence

- Claim records are appended to `reports/self_default/claims.jsonl`.
- Audit logs of status changes are written to `reports/self_default/audit.jsonl`.

## Governance boundary

Governance may set budgets and TTLs, but only Self‑Default adjudicates truth values.  Governance enforces structure and authorization, never factual correctness.