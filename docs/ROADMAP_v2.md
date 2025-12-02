# Maven Roadmap (Phase 2)

This document outlines the next phase of development for the Maven cognitive
platform.  The goals below extend Maven’s core competencies beyond the
foundational human‑cognition upgrade, adding multi‑modal perception,
adaptive performance tuning, deployment tools and new research tracks.  The
outlined actions are intended as guidelines for incremental implementation; not
all features need to be completed at once.

## 1 Cognitive Expansion

| Goal | Description | Implementation Notes |
|------|-------------|----------------------|
| **Multi‑Modal Perception** | Add visual and audio processing capabilities to the Sensorium stage, enabling basic image and sound feature extraction. | Create optional modules `vision_brain.py` and `hearing_brain.py` in `brains/cognitive/sensorium/service` that expose simple service APIs (e.g. `ANALYZE_IMAGE` / `ANALYZE_AUDIO`).  These stubs will return placeholder features until real embeddings are added. |
| **Context‑Aware Dialogue Memory** | Improve persistent conversation memory with relevance weighting to prioritise recent or important information. | Integrate a `context_relevance_score()` helper inside `memory_librarian`; add decay and recall bias when merging context snapshots. |
| **Imagination Sandbox v2** | Allow the imaginer to run nested hypothetical roll‑outs (“what‑if chains”) for deeper creative exploration. | Increase the maximum number of roll‑outs per query (e.g. up to 10) and add recursion control flags.  Continue to generate proof logs for each internal simulation. |
| **Collaborative Agent Network** | Enable multiple Maven instances to share context or delegate tasks securely. | Extend `peer_connection_brain.py` to maintain session state and synchronise messages via a file‑based message bus. |
| **Ethical and Safety Reasoning** | Formalise moral reasoning by maintaining a separate ethics rule set.  Evaluate actions not only for factual correctness but also moral implications. | Add an `ethics_rules.json` file in the reports directory (initially empty).  Create a new filter in the reasoning stage to scan queries against ethical patterns and return cautionary responses when triggered. |
| **Autonomy Reinforcement Loop** | Allow Maven to self‑assign learning or repair goals when it detects poor performance on a domain. | Extend Stage 18 (self‑review) to create “Improve domain X” goals automatically using the existing goal memory infrastructure. |

## 2 Performance & Efficiency

| Focus | Action | Path |
|-------|-------|------|
| **Pipeline Optimisation** | Cache expensive retrieval operations and lazily load domain banks on demand. | Modify the `memory_librarian.py` service to memoize results per run and avoid redundant computation. |
| **Threaded Task Execution** | Allow planning and reasoning stages to run concurrently where safe. | Use Python’s `concurrent.futures.ThreadPoolExecutor` around planning and reasoning calls, ensuring shared state is thread‑safe. |
| **Incremental Savepoints** | Periodically save context snapshots between runs to support crash recovery. | Add periodic checkpoints for `context_snapshot.json` during Stage 10 finalisation. |
| **Adaptive Confidence Model** | Replace fixed confidence thresholds with a rolling statistical model based on recent success and failure rates. | Introduce a `dynamic_confidence.py` module with a function to compute adjustments from a sequence of success metrics. |
| **Data Pruning & Compression** | Implement retention policies for reflection logs and QA history to prevent unbounded growth. | Extend the memory librarian with configurable retention thresholds (exposed via `config/memory.json`) and compress or discard old entries beyond the limit. |

## 3 Deployment Readiness

| Objective | Description | Task |
|-----------|-------------|-----|
| **Cross‑Platform Bundling** | Package Maven as a portable Python 3.11 executable for distribution. | Use tools like `zipapp` or `PyInstaller` to bundle the application with minimal dependencies. |
| **System Monitor Dashboard** | Provide real‑time reporting of memory usage, active goals, mood and proof logs. | Build a lightweight dashboard (`ui/dashboard.py`) using a text‑based interface (e.g. `curses`) to display metrics. |
| **Governance Logging Framework** | Standardise governance proof and error logs for audits. | Extend `/reports/governance/` schema to include fields such as `type`, `timestamp`, `action` and `verdict`. |
| **Automated Regression Harness** | Schedule continuous self‑testing after upgrades to detect regressions. | Set up nightly runs of the existing `tools/regression_harness.py` via a cron‑like mechanism or script wrapper. |
| **Offline Installer** | Provide an offline installer that reconstructs the necessary directory structure and config. | Create `install_offline.sh` to copy core modules, configs, and docs into a self‑contained distribution. |

## 4 Experimental Research Tracks

| Track | Goal | Research Area |
|------|-----|--------------|
| **Affective AI** | Refine the valence–arousal model and integrate mood more deeply into dialogue generation. | Explore the circumplex model of affect and its application to text generation【128125055361614†L343-L349】. |
| **Self‑Reflection & CoT** | Enhance the chain‑of‑thought self‑evaluation loop to provide richer error detection. | Study CoT strategies and multi‑pass reasoning to further reduce hallucinations and improve robustness【695710332883996†L55-L70】. |
| **Narrative Identity Growth** | Extend the identity journal into a richer narrative self that evolves over long sessions. | Apply narrative identity theory to accumulate experiences and maintain a coherent, evolving self‑story【459801191566567†L145-L150】. |
| **Meta‑Learning for Safety** | Develop mechanisms for Maven to learn new safety rules from reflection logs. | Add a function to `safety_rules.py` that proposes new patterns based on repeated reflection failures or flagged content. |
| **Knowledge Graph Lifespan** | Model fact decay and reinforcement to simulate human memory consolidation. | Investigate memory consolidation research and incorporate forgetting curves or spaced repetition【642301385388938†L146-L158】. |

## 5 Immediate Action Plan (Next 30 Days)

1. **Implement** the ethical and safety rule expansion by creating an empty `ethics_rules.json` file and wiring a simple check in the reasoning brain.
2. **Deploy** context snapshot checkpoints and memory pruning logic in the memory librarian to support retention policies.
3. **Run** nightly regression tests using `tools/regression_harness.py` to detect any drift in answers or unexpected behaviour.
4. **Add** a preliminary dashboard script that reports key metrics such as mood, memory counts and goal statistics.
5. **Document** the interactions between Stage 15–18 (autonomy, regression, consolidation and self‑review) and update this roadmap as new findings arise.
6. **Test** the expanded imagination sandbox with deeper roll‑outs to evaluate performance and safety before enabling by default.
7. **Create** a reinforcement goal generator that automatically adds improvement tasks for domains with low success ratios.