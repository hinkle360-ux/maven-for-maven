# 🧠 MAVEN HUMAN‑COGNITION UPGRADE OVERVIEW

**Revision:** 2025‑11  
**Scope:** Current 14‑stage architecture  
**Focus:** Human‑like reasoning, reflection, imagination, affect, creativity, and self‑identity — all governed, transparent, and offline.

---

## 1 · Context

Maven’s cognitive framework now spans **fourteen sequential stages**, from initial perception through long‑term reflection and affect learning.  Each stage still obeys the broadcast rule — every input moves linearly through the pipeline — but the tail end now includes **System History**, **Self‑DMN**, **Affect‑Learn**, and **Autonomy‑Governance** checkpoints.

This cycle does **not** alter that backbone.  Instead, we’re **augmenting Maven’s cognitive realism** by introducing new subsystems that make it reason, imagine, and self‑reflect more like a person — while preserving all baseline rules:

| Core Rule           | Still Enforced |
|---------------------|----------------|
| Python 3.11 only    | ✅             |
| Stdlib only / offline | ✅             |
| No `__init__.py` files | ✅             |
| Sequential broadcast order | ✅       |
| Governance proof on every autonomous act | ✅ |

---

## 2 · Human‑Cognition Additions

### 2.1 Dual‑Process Reasoning — “System 1 / System 2”
**File:** `brains/cognitive/reasoning/service/dual_router.py`  
**Purpose:** Give Maven a *fast intuitive channel* and a *slow deliberate channel* within the Reasoning stage.  The fast path routes directly based on learned vocabulary; the slow path triggers when the margin between top banks is low.  The outcome balances *speed vs. depth* — Maven now “thinks fast or slow” like a human.

### 2.2 Per‑Turn Self‑Reflection
**File:** `brains/cognitive/self_dmn/service/self_critique.py`  
**Purpose:** Embed a reflective loop inside every cycle.  After Stage 10 Finalize, Maven generates a short self‑critique, logs it to `reports/reflection/turn_*.jsonl`, adjusts local strategy weights (clarity, caution, verbosity) and feeds reflection summaries into Affect‑Learn (Stage 14).  Maven now *learns from itself* each interaction instead of only by external correction.

### 2.3 Imagination Sandbox
**File:** `brains/cognitive/imaginer/service/imaginer_brain.py` (extended)  
**Purpose:** Add a safe internal “what‑if” simulator.  The imaginer can run up to five hypothetical rollouts per query, score candidates for internal consistency and novelty, and keep all simulations sandboxed — no direct memory writes.  Approved results carry a Governance proof.  This grants *creativity + foresight* without unsafe side effects.

### 2.4 Affective Modulation
**Files:** `planner_brain.py`, `reasoning_brain.py`  
**Purpose:** Let emotion values bias cognition.  The Planner and Reasoning brains read Affect‑Priority outputs (`valence`, `arousal`) and dynamically adjust thresholds: negative valence → cautious, slower deliberation; positive valence → faster routing, warmer tone.  This adds believable *mood dynamics* that influence tone and persistence.

### 2.5 Narrative Self‑Model → Personal Brain
**File:** `brains/personal/service/identity_journal.py`  
**Purpose:** Move self‑identity generation out of System History into Maven’s dedicated Personal Brain.  The journal aggregates facts, preferences, and recurring motives into `identity_snapshot.json`, maintains an evolving “self narrative” (beliefs, opinions and style trends), and shares it with Planner, Reasoning and Affect‑Learn to keep behaviour coherent.  This establishes a persistent, adaptive *personality and worldview*.

### 2.6 Creative Divergence / Convergence
**File:** `brains/cognitive/language/service/language_brain.py` (Stage 6 Generate Candidates)  
**Purpose:** Implement structured creativity.  In Stage 6 the language brain now diverges by producing multiple textual variants using lightweight perturbation, then converges by re‑ranking them via Reasoning and Imagination scores.  Maven now brainstorms, then self‑selects the best idea — **creativity with discipline**.

### 2.7 System‑2 Tool Interfaces (Stdlib Only)
**Files:**
- `brains/agent/tools/logic_tool.py`  
- `brains/agent/tools/math_tool.py`  
- `brains/agent/tools/table_tool.py`  
**Purpose:** Give the Reasoning brain precise computation aids.  These tools provide logic evaluation, arithmetic calculation and simple table manipulation via pure Python.  They are invoked through `service_api({"op":"RUN","payload":{"task":…}})` and log every call in `reports/agent/tool_calls.jsonl`.  This extends Maven’s analytical reach without external dependencies.

### 2.9 Greeting Detection & Social Interaction
**File:** `brains/cognitive/language/service/language_brain.py` (Stage 3 & Stage 6)  
**Purpose:** Endow Maven with basic social awareness.  The language brain now recognises common greetings (e.g. “hi”, “hello”, “good morning”) and marks them as **SOCIAL** intents in Stage 3.  Stage 6 responds with a friendly greeting (“Hello! How can I help you today?”) instead of the generic acknowledgement.  Social inputs bypass memory search and storage, and governance always allows them.  This provides a courteous user experience and prevents needless retrieval work.

### 2.10 Cross‑Episode Memory & Self‑Repair
**Files:** `reasoning_brain.py`, `language_brain.py`, `brains/personal/memory/qa_memory.jsonl`, `reports/self_repair.jsonl`  
**Purpose:** Enable Maven to remember definitive answers across sessions and notice contradictions.  The language brain writes each question/answer pair to `reports/qa_memory.jsonl` when the response is certain (not speculative).  The reasoning brain checks this file first when evaluating questions; if a match is found, it returns `KNOWN_ANSWER` immediately, raising confidence and bypassing expensive searches.  When a new answer disagrees with a stored one, the finalization stage logs the conflict to `reports/self_repair.jsonl` for later review.  This lays the groundwork for automatic regression testing and self‑repair loops.

### 2.11 Goal Memory & Autonomy Brain
**Files:** `brains/personal/memory/goal_memory.py`, `brains/cognitive/autonomy/service/autonomy_brain.py`, `config/autonomy.json`, `memory_librarian.py` (Stage 15)  
**Purpose:** Support long‑horizon planning and execution.  The planner decomposes multi‑step commands into a list of steps (Stage 2) and writes each as a **goal** via the goal memory module.  The autonomy brain executes goals one at a time on each pipeline run (`TICK` op) and marks them complete in the goal file.  Stage 15 in the memory librarian coordinates self‑DMN ticks, opportunity scoring via the motivation brain, autonomy execution and goal introspection.  A simple configuration file (`config/autonomy.json`) controls whether autonomy is enabled and how many goals/ticks to run per call.  This infrastructure is an essential bridge to full agentic autonomy.

### 2.12 Topic Statistics & Cross‑Episode Learning
**Files:** `brains/personal/memory/topic_stats.py`, `language_brain.py`, `reasoning_brain.py`  
**Purpose:** Track recurring question themes and adjust behaviour.  The language brain updates a topic statistics file by storing the first two words of each answered question.  The reasoning brain reads this file to compute a *topic familiarity* bias: frequent topics yield a small confidence boost, while novel topics nudge valence downward to encourage caution.  The personal brain exposes `TOPIC_STATS` and `TOPIC_TRENDS` operations so users can inspect which topics Maven has seen most.  Both return the top‑N topics by frequency; `TOPIC_TRENDS` is an alias for convenience.  This is an early form of cross‑episode learning.

### 2.13 Replanner & Compound Goal Splitting
**Files:** `brains/cognitive/planner/service/replanner_brain.py`, `memory_librarian.py` (Stage 15)  
**Purpose:** Divide compound tasks into atomic actions.  The replanner takes existing goals and splits titles on conjunctions (“and”, “then” or commas), writing each sub‑goal back to the goal memory with a `REP-` prefix.  Stage 15 invokes the replanner on all active goals **before** autonomy ticks so that the executor never processes a long, compound command directly.  Remaining sub‑goals are surfaced to the context for inspection or further planning.

### 2.14 Regression Harness & Self‑Repair Testing
**File:** `tools/regression_harness.py`  
**Purpose:** Provide a tool for automated knowledge regression.  The harness reads all stored question/answer pairs from the QA memory, re-asks them through the reasoning brain, compares the current answers with the stored ones, and writes a JSON report to `reports/regression/results.json`.  Mismatches indicate factual drift and point to entries in `self_repair.jsonl` that require attention.  This tool is optional but important for maintaining accuracy as Maven evolves.

### 2.15 Multi‑Agent Capability & Peer Connection
**Files:** `brains/cognitive/language/service/language_brain.py` (Stage 6), `brains/agent/service/peer_connection_brain.py`  
**Purpose:** Lay the groundwork for collaboration with other agents.  Stage 6 now recognises commands like “connect to peer <id>” when the parsed intent is a **REQUEST**.  It delegates to a peer connection brain, which simulates establishing a real‑time communication channel and returns a confirmation message.  While the current implementation is a stub, it demonstrates how Maven can spawn or connect to specialised sub‑agents.

### 2.16 Autonomy Scheduler & Rate Limiting
**Files:** `brains/cognitive/autonomy/service/autonomy_brain.py`, `config/autonomy.json`, `memory_librarian.py` (Stage 15)  
**Purpose:** Provide dynamic scheduling of autonomous actions. The autonomy brain now consults additional configuration fields: `priority_strategy` chooses whether to sort goals by inferred priority or in FIFO order, and `rate_limit_minutes` throttles how often ticks may occur. A helper `_goal_priority` ranks goals (AUTO_REPAIR > delegated tasks > others). Before executing any goals, the brain checks a timestamp in `reports/autonomy/last_tick.json`; if not enough minutes have elapsed, the tick is skipped. After executing goals it updates this timestamp. This prevents runaway autonomy loops while still prioritising urgent self‑repair tasks.

### 2.17 Peer Delegation
**Files:** `brains/cognitive/peer_connection/service/peer_connection_brain.py`, `brains/cognitive/language/service/language_brain.py` (Stage 6)  
**Purpose:** Lay a foundation for cooperative multi‑agent workflows. Stage 6 now recognises commands like “delegate *task* to peer <ID>”. It calls the peer connection brain’s `DELEGATE` operation, which writes a delegated goal into the personal goal memory with a description prefix `DELEGATED_TO:`. The autonomy scheduler can then execute these delegated tasks in future ticks. The peer brain responds with a confirmation message, enabling simple delegation chains without executing any external network requests.

### 2.18 Regression Harness Integration & Self‑Repair Goals
**Files:** `memory_librarian.py` (Stage 16), `brains/personal/memory/goal_memory.py`  
**Purpose:** Close the loop between regression testing and autonomy. When the regression harness (Stage 16) detects mismatches between the current answer and the stored QA memory, the memory librarian automatically generates *self‑repair goals* titled “Verify QA: <question>” with the description `AUTO_REPAIR`. These goals are persisted in goal memory and subsequently prioritised by the autonomy scheduler. This turns factual drift into actionable tasks, enabling Maven to proactively verify and correct its knowledge base.

### 2.19 Dynamic Re‑planning of Stale Goals
**Files:** `memory_librarian.py` (Stage 15), `config/autonomy.json`, `brains/cognitive/planner/service/replanner_brain.py`  
**Purpose:** Automatically break down long‑standing goals into new sub‑tasks. Stage 15 now inspects any remaining active goals **after** autonomy ticks. If a goal has been pending longer than the configured `replan_age_minutes` threshold (24 hours by default), the memory librarian calls the replanner brain’s `REPLAN` operation on that goal. The original goal is marked completed and its title is split on “and/then/, ” into separate tasks, which are persisted as new goals. These new goals are surfaced in `stage_15_replanned_stale_goals`, giving the autonomy scheduler fresh items to execute while preventing goals from languishing indefinitely.

### 2.20 Peer Query (ASK) & Multi‑Agent Question Routing
**Files:** `brains/cognitive/peer_connection/service/peer_connection_brain.py`, `brains/cognitive/language/service/language_brain.py` (Stage 6)  
**Purpose:** Enable Maven to route questions to peers. The peer connection brain now supports an `ASK` operation, and Stage 6 recognises commands like “ask peer <id> <question>” or “ask <question> to peer <id>”. When matched, the language brain invokes the peer brain’s `ASK` op and returns the peer’s stubbed response immediately, bypassing normal candidate generation. Each query is logged to `reports/peer_queries.jsonl` for audit. This lays the groundwork for a collaborative multi‑agent architecture where different agents can answer specialised questions.

### 2.21 Execution Budget & Resource Constraints
**Files:** `brains/cognitive/autonomy/service/autonomy_brain.py`, `config/autonomy.json`, `reports/autonomy/budget.json`

**Purpose:** Limit the total number of autonomous goal completions and prevent unbounded resource consumption. A new `execution_budget` field in `config/autonomy.json` specifies how many goals may be executed across all runs. The autonomy brain tracks remaining budget in `reports/autonomy/budget.json`; each tick decrements this count by the number of goals completed. If the budget is exhausted, Stage 15 skips autonomy ticks and surfaces a `budget_exhausted` reason. Operators can adjust the budget to tune how much autonomous work is performed between human interactions.
### 2.22 Semantic Memory — Knowledge Graph

**File:** `brains/personal/memory/knowledge_graph.py`  
**Operations:** `ADD_FACT`, `QUERY_FACT`, `LIST_FACTS`

To move beyond a simple log of Q/A pairs, Maven now includes a **semantic memory** in the form of a lightweight knowledge graph. Facts are stored as triples *(subject, relation, object)* in `reports/knowledge_graph.json`. Developers or peer agents can persist new facts via the personal brain API (`ADD_FACT`) and retrieve them later (`QUERY_FACT`, `LIST_FACTS`). The reasoning brain consults this graph before using heuristics or external retrieval — when a user asks “What is X?” or “Who is X?”, Maven first looks for a matching `(X, is, ?)` fact in the graph and returns it immediately if present. This enables Maven to build its own structured world knowledge over time.

The knowledge graph does not yet perform inference or reasoning across chains of facts; it is simply an associative lookup. However, it lays the groundwork for richer semantic memory modules in future iterations.  To monitor growth of the graph, the personal brain exposes a **FACT_COUNT** operation that returns the total number of facts stored.  This helps developers understand how Maven’s structured knowledge is expanding over time.  Additionally, the meta‑confidence helper is now accessible via the `META_STATS` alias, which returns the same domain confidence table as `META_CONFIDENCE`, giving a quick overview of which topics Maven succeeds or struggles with.

**Stage flow impact:** While the knowledge graph itself is not a new stage, it plugs into **Stage 8 (Reasoning)**. After affect and familiarity adjustments, the reasoner checks the graph for direct answers to simple definition questions. If found, the result is returned with high confidence and the pipeline skips retrieval, heuristics and tool invocation. In addition, the finalization stage now automatically **assimilates** simple facts into the knowledge graph: when the system answers a “What is X?” or “Who is X?” question definitively, the triple `(X, is, answer)` is persisted to `knowledge_graph.json`. This allows Maven to learn new facts over time without explicit API calls and gradually expand its structured memory.

### 2.23 Domain Confidence & Meta‑Learning

**File:** `brains/personal/memory/meta_confidence.py`  
**Operation:** `META_CONFIDENCE`

Maven now keeps track of how well it has answered questions across *domains*. A domain is defined by the first one or two words of the user’s question (e.g. “what is”, “how many”). Each time Maven finalises an answer, it logs whether that domain produced a definitive answer or not. 

**Weighted counts & recency:** In addition to simply counting successes and failures, each event is weighted by the *complexity of the question*. Longer or more complex queries carry more weight than very short queries. These weighted counts allow Maven to bias confidence more strongly towards difficult topics where recent success indicates genuine mastery. Both weighted and unweighted counts are **decay‑weighted by time**: older successes and failures gradually fade, so recent performance weighs more heavily than long‑ago outcomes. The decay uses an exponential curve (5% per day by default) to ensure that confidence remains responsive to new evidence.

When the reasoning brain evaluates a new question, it computes a small positive or negative adjustment to its affective valence based on the weighted success rate for that domain. Domains with many recent successes boost confidence, while domains with frequent or recent failures dampen it. These adjustments are small (±0.1) but encourage Maven to be more cautious where it tends to struggle and more decisive where it has been performing well.

The personal brain exposes a `META_CONFIDENCE` operation to retrieve a table of the top domains with their success/failure counts and computed adjustments. This can be used by developers to understand Maven’s learning trajectory across topics.

In addition, the API now provides a **`META_TRENDS`** operation. This call returns two lists of domain records: those with the highest positive adjustments (*improved*) and those with the most negative adjustments (*declined*). These lists are sorted by the magnitude of the adjustment and limited to a configurable number of domains (five by default). By examining these trends, developers can quickly see which topics Maven is excelling at and where it is struggling based on recent history. Each record includes the domain key, success/failure counts, total attempts and computed adjustment. This high‑level overview complements the detailed statistics returned by `META_CONFIDENCE`.

**Stage flow impact:** Domain confidence modulation occurs in **Stage 8 (Reasoning)** after the affect and topic‑familiarity adjustments. Success/failure updates are written in **Stage 10 (Finalize)** whenever a question’s answer is committed to the QA memory.

### 2.24 User Profile & Personalised Responses

**File:** `brains/personal/memory/user_profile.py`  
**Operations:** `UPDATE_PROFILE`, `GET_PROFILE`, `GET_ATTRIBUTE`

To enable a more personalised dialogue, Maven now maintains a simple **user profile**. This record stores key–value pairs about the user (e.g. preferred language, location, or interests). The profile lives in `reports/user_profile.json` and can be updated via the personal brain API (`UPDATE_PROFILE`). Developers or agents can fetch the entire profile (`GET_PROFILE`) or retrieve individual attributes (`GET_ATTRIBUTE`). All keys are normalised to lower case and values stored as strings. The profile is non‑sensitive and intended for contextual adjustments only; it should not contain private data.

The profile is now used during greeting generation and beyond. Stage 6 reads the user’s *name*, *timezone* and *language* attributes (if present) to craft personalised salutations. For example, if the user’s profile contains `name="Alice"` and `timezone="America/New_York"`, a morning query will elicit “Good morning Alice! How can I help you today?” and an evening query will produce “Good evening Alice! How can I help you today?”. If a `language` attribute is present and matches a supported code (currently **es**, **fr**, or **de**), the greeting is localised using built‑in translations (e.g. Spanish *“¡Buenos días”*, French *“Bonjour”*, German *“Guten Morgen”*). Unrecognised languages default to English.

Beyond greetings, the user profile can now influence **tone** and **verbosity** for Maven’s responses. Profile attributes like `tone`, `formality` or `style` override the default tone (e.g. setting `tone="formal"` will make responses more formal, while `tone="casual"` keeps them friendly). Similarly, attributes such as `verbosity`, `verbosity_preference`, `detail` or `level` allow the user to adjust how concise or elaborate Maven’s answers should be. These may be specified as keywords (`low`, `normal`, `high`, `verbose`) or numeric multipliers (e.g. `1.3` for slightly more detail). Such settings apply during the finalisation stage to tailor verbosity and tone before the answer is returned. Unrecognised languages or undefined preferences default to neutral behaviour.

Storing user attributes separately from Maven’s own identity keeps the self‑model and user model distinct, a crucial part of meta‑cognition.

**Stage flow impact:** The user profile is managed by the **Personal Brain**; it is not a separate stage. It is accessed via API operations outside the core pipeline, but future revisions may reference the profile during language generation or planning.

### 2.25 Synonym Import / Export & Grouping

**Files:** `brains/personal/memory/synonyms.py`, `brains/personal/service/personal_brain.py`   
**Operations:** `IMPORT_SYNONYMS`, `EXPORT_SYNONYMS`, `REMOVE_SYNONYM`, `LIST_SYNONYM_GROUPS`

While Section 2.26 introduced a persistent synonym mapping, developers often need to manage many mappings at once or clean up obsolete terms. Maven now supports **bulk import and export** of synonym mappings as well as removal and grouping functions. The personal brain exposes:

* **IMPORT_SYNONYMS** — Accepts a dictionary or list of `(epithet, canonical)` pairs and merges them into the existing mapping. Duplicate entries are ignored. Returns the number of new mappings added. Useful for seeding Maven with domain‑specific aliases from an external file.
* **EXPORT_SYNONYMS** — Returns the entire synonym mapping so developers can back up or inspect all defined epithets.
* **REMOVE_SYNONYM** — Deletes a specific mapping by its epithet. This allows correction of erroneous entries.
* **LIST_SYNONYM_GROUPS** — Returns a dictionary keyed by canonical terms with a list of all epithets (including the canonical term itself). This grouping makes it easy to see how many aliases refer to each concept.

These operations extend the synonym infrastructure without introducing new stages. They operate entirely within the personal brain and can be invoked between pipeline runs. Automatic updates (Section 2.26) remain in place: whenever a new definitional fact is assimilated, answer phrases are mapped to the subject. Bulk import/export simply adds or extracts additional mappings.

**Stage flow impact:** None. Synonym import/export occurs outside the main pipeline and does not alter cognition. However, the expanded API makes it easier to curate synonyms, which in turn improves the hit rate for semantic recall.

### 2.26 User Mood Tracking & Affect Integration

**Files:** `brains/personal/memory/user_mood.py`, `brains/personal/service/personal_brain.py`, `language_brain.py`

Human dialogue is coloured by mood. To simulate this, Maven now maintains a **user mood score** — a single floating‑point value in the range [‑1, 1] — that evolves over time. The mood captures the general emotional tone of recent interactions (positive values denote optimism/happiness; negative values indicate sadness/frustration). The mood subsystem includes the following operations:

* **GET_MOOD** — Returns the current mood value (0.0 if unset). Useful for diagnostics or for peer agents to tailor their behaviour.
* **UPDATE_MOOD** — Adds a new valence sample (e.g. +0.5 for positive, ‑0.2 for negative). The stored mood decays by 5 % per day and is updated via a weighted average. When called frequently, this function slowly steers the mood toward the mean of supplied valences.
* **RESET_MOOD** — Resets the mood to neutral (0.0) by clearing the stored file. This is helpful when starting a new interaction sequence.

The mood value influences Maven’s output. During **Stage 10 Finalize**, the finalisation logic now reads the affect stage’s valence (from Stage 5) and calls `UPDATE_MOOD`. It then retrieves the current mood and adjusts the response tone accordingly: a strongly positive mood (≥ 0.3) yields a **friendly** tone; a strongly negative mood (≤ ‑0.3) triggers a **caring** tone; neutral moods leave the tone unchanged. By gradually integrating the user’s emotional state, Maven exhibits a more compassionate conversational style over time.

**Stage flow impact:** Mood updates occur within **Stage 10 Finalize**. The mood module is part of the personal brain; it does not introduce a new pipeline stage. However, mood tracking provides additional feedback for affective modulation and contributes to a more human‑like dialogue.

### 2.25 Memory Consolidation & QA Pruning

**File:** `brains/cognitive/memory_librarian/service/memory_librarian.py` (Stage 17), `config/memory.json`

As Maven answers more questions over many sessions, its **QA memory** can balloon. A growing log slows down lookups and consumes storage. To address this, the pipeline now includes a **memory consolidation** routine that automatically prunes and summarises the QA log. After regression and repair (Stage 16), Maven checks `/reports/qa_memory.jsonl`: if it contains more than a configurable number of entries (default 100), the oldest lines are removed. Before deletion, Maven examines each old Q/A entry and extracts simple definitional facts into the **semantic knowledge graph**. Questions of the form “what is X?” or “who is X?” with short answers (≤80 characters and no uncertainties) are converted into triples `(subject, "is", answer)` and stored permanently in `reports/knowledge_graph.json`.

Statistics about the pruning run — total entries before pruning, number pruned, number of facts assimilated, and entries retained — are recorded in the context under `stage_17_memory_pruning`. Developers can tune the maximum size by editing `config/memory.json` (`qa_memory_max_entries`). This ensures Maven’s episodic memory stays manageable while preserving essential semantic knowledge.

### 2.26 Synonym Mapping & Canonical Terms

**File:** `brains/personal/memory/synonyms.py`   
**Operations:** `ADD_SYNONYM`, `GET_CANONICAL`, `LIST_SYNONYMS`

Users often refer to the same entity using different words (for example, *“the red planet”* instead of *Mars*). To resolve such variations and improve semantic recall, Maven introduces a persistent **synonym mapping**. Developers or peer agents can map informal terms or epithets to a canonical name via the personal brain API (`ADD_SYNONYM`). Mappings are stored in `config/synonyms.json` and persist across sessions. When the reasoning brain encounters a question of the form “what is X?” or “who is X?”, it normalises `X` using the synonym mapping before consulting the knowledge graph. This ensures that queries like “What is the red planet?” will resolve correctly if a fact `(mars, is, the red planet)` exists. Likewise, the `GET_CANONICAL` operation returns the canonical form of a term (or the lower‑cased term itself if no mapping exists), and the entire mapping can be inspected via `LIST_SYNONYMS`.

**Stage flow impact:** The synonym mapping is leveraged within **Stage 8 (Reasoning)** during semantic memory lookup. Candidate subjects extracted from definition questions are passed through the mapping before searching the knowledge graph. If a canonical form is found, the knowledge graph uses that instead of the raw user phrase. No additional pipeline stage is required; the mapping is a lightweight helper that increases the hit rate for stored facts without altering other behaviours.

When new definition facts are assimilated (e.g. in Stage 10 Finalize), the system also automatically updates the synonym mapping: answer phrases become synonyms for the canonical subject. Both the full answer and its form without leading articles (such as “the red planet” and “red planet”) are mapped to the subject. This continuous enrichment of synonyms improves recall across sessions without manual updates.

### 2.27 Cross‑Episode Memory Search

**File:** `brains/personal/service/personal_brain.py`  
**Operation:** `SEARCH_QA`

To approach **near‑perfect retention** across sessions, Maven now provides a way to search its entire QA memory for previous questions or answers. The cross‑episode memory is stored in a JSONL file (`reports/qa_memory.jsonl`). The new `SEARCH_QA` operation lets a developer or peer agent query this log by substring. When `SEARCH_QA` is called with a `query` string, the personal brain scans the QA memory for any entries whose question or answer contains that substring (case‑insensitive) and returns up to a configurable number of matches. Each match includes the original question, its answer and the timestamp when it was recorded. This operation does not modify the memory; it simply exposes a read‑only search capability to aid recall and debugging.

**Stage flow impact:** QA memory search is not part of the main cognitive pipeline. It is an auxiliary API exposed by the **Personal Brain** for manual inspection, troubleshooting and context enrichment. Developers can use it to fetch prior answers that might inform new reasoning or test for consistency.

### 2.28 Diagnostics & System Introspection

**File:** `brains/personal/service/personal_brain.py`  
**Operation:** `INTROSPECT`

As Maven’s cognitive machinery grows, developers need visibility into its internal state. The **INTROSPECT** operation provides a summary of key memory structures and counters in one call. When invoked, the personal brain aggregates counts of:

* **QA memory entries:** how many question/answer pairs have been stored across sessions.
* **Facts:** the number of (subject, relation, object) triples in the semantic knowledge graph.
* **Synonyms:** the number of synonym mappings currently defined.
* **User profile attributes:** how many key‑value pairs are stored about the user.
* **Active goals:** how many uncompleted goals are present in the goal memory.
* **Domains tracked:** the number of domain entries recorded by the meta‑confidence module.
* **Topics tracked:** the number of topics logged in the topic statistics file.

The operation returns a dictionary of these metrics in a `stats` field. All errors are handled gracefully, so missing modules simply yield zero counts. This diagnostic call helps monitor Maven’s footprint and supports debugging or performance tuning.

**Stage flow impact:** The INTROSPECT call is available via the **Personal Brain API** and does not add a new pipeline stage. It is intended for offline inspection or periodic monitoring rather than end‑user interaction.  

### 2.29 Planner Enhancements

**File:** `brains/cognitive/planner/service/planner_brain.py`

The planner has been augmented with broader heuristics for splitting complex commands into sub‑goals. Previously, only **and/then** conjunctions triggered segmentation. The new pattern now recognises additional sequencing words and phrases such as **after**, **before**, and **once you have / once you've** (case insensitive), as well as commas. For example, a request like “Clean the data, then train the model after you fix the schema” produces the segments `["Clean the data", "train the model", "fix the schema"]`. These segments are recorded as separate goals for the autonomy scheduler to execute in order.

**Stage flow impact:** This enhancement affects **Stage 2 (Planner)**. It yields more granular steps for goal decomposition, enabling Maven to tackle multi‑phase instructions systematically. No new pipeline stage is introduced; the segmentation logic runs within the existing PLAN operation.

### 2.30 Knowledge Graph & Synonym Search

**File:** `brains/personal/service/personal_brain.py`    
**Operations:** `SEARCH_KG`, `SEARCH_SYNONYMS`

To support near‑perfect retention and aid debugging, the personal brain now exposes two searchable views into Maven’s long‑term memory:

* **SEARCH_KG** – Performs a substring search over the **semantic knowledge graph** stored in `reports/knowledge_graph.json`.  When invoked with a `query`, the personal brain loads all `(subject, relation, object)` triples and returns those where the query appears in any part of the triple (case‑insensitive).  Results include the full triple and respect a `limit` parameter.  This call is read‑only and does not alter the graph.

* **SEARCH_SYNONYMS** – Searches the **synonym mapping** defined in `config/synonyms.json`.  Given a `query` string, the operation returns pairs of original terms and their canonical forms whenever the query matches either the term or the canonical value (case‑insensitive).  A `limit` parameter bounds the number of returned mappings.  This is also a diagnostic operation, not part of the main pipeline.

These searches help developers and peer agents quickly inspect Maven’s stored facts and canonical terms, boosting recall across sessions and supporting knowledge debugging without modifying internal state.

**Stage flow impact:** Both search operations live in the **Personal Brain API** and do not introduce new pipeline stages. They are meant for manual exploration and programmatic retrieval of stored knowledge, complementing the existing `SEARCH_QA` operation.

### 2.31 Canonical QA Memory Search

**File:** `brains/personal/service/personal_brain.py`    
**Operation:** `SEARCH_QA_CANONICAL`

While `SEARCH_QA` lets developers scan the raw QA memory by substring, it does not account for synonyms or nicknames. The new **SEARCH_QA_CANONICAL** operation bridges this gap by canonicalising the query before searching. When called with a `query`, the personal brain uses the synonym mapping to compute the canonical form and then looks for either the original or canonical phrase in past questions and answers. Matches are returned in the same format as `SEARCH_QA` (question, answer, timestamp), and the number of results can be limited via a `limit` parameter.

For example, if the synonym mapping contains "the red planet" → "mars" and the QA memory has an entry for “What is Mars?”, a developer can call `SEARCH_QA_CANONICAL` with `query="the red planet"` and retrieve the stored answer. This operation enhances Maven’s recall across paraphrased questions and helps verify that synonyms are being honoured in practice.

**Stage flow impact:** `SEARCH_QA_CANONICAL` is an auxiliary **Personal Brain API** call. It is not part of the main reasoning pipeline; instead, it provides a convenient way to interrogate the QA memory using canonicalised terms and supports near‑perfect retention through synonym‑aware search.

### 2.32 QA Memory Summarisation

**File:** `brains/personal/service/personal_brain.py`    
**Operation:** `SUMMARIZE_QA`

As Maven’s QA memory grows, it becomes useful to get a high‑level overview of what has been learned. The new **SUMMARIZE_QA** operation groups all stored question/answer pairs by their *domain key* (the first two words of the question) and returns a concise summary. For each domain it reports:

* **count** – how many Q/A pairs fall under that domain;
* **last_answer** – the most recent answer given for that domain;
* **unique_answers** – up to five unique answers observed for that domain.

An optional `limit` parameter controls how many domains are returned (default 10). This summary provides a snapshot of Maven’s knowledge coverage and highlights areas with many or few examples. It supports developers in auditing memory retention and identifying topics that may need more training or pruning.

**Stage flow impact:** `SUMMARIZE_QA` is purely an administrative helper in the **Personal Brain API**. It does not alter any memory structures or affect the main reasoning pipeline. It reads `reports/qa_memory.jsonl` and returns aggregated statistics on demand.

### 2.33 Goal Summary

**File:** `brains/personal/service/personal_brain.py`    
**Operation:** `GOAL_SUMMARY`

To monitor Maven’s long‑term goals at a glance, a new `GOAL_SUMMARY` operation aggregates goal memory statistics. It returns the total number of goals, how many are active vs. completed, and a breakdown of counts by *category* (using prefixes such as `AUTO_REPAIR`, `DELEGATED_TO`, or `GENERAL` derived from each goal’s description or title). The operation also returns the list of currently active goals so that schedulers or developers can see what remains to be done.  This summary helps ensure that autonomous tasks remain manageable and visible as Maven takes on more responsibility.

**Stage flow impact:** `GOAL_SUMMARY` is part of the **Personal Brain API** and has no impact on the main cognitive pipeline. It reads the persistent goals file (`goals.jsonl`) and computes statistics on demand without modifying any data.

### 2.34 Semantic Memory CRUD & Relation Queries

**File:** `brains/personal/memory/knowledge_graph.py`, `brains/personal/service/personal_brain.py`  
**Operations:** `UPDATE_FACT`, `REMOVE_FACT`, `QUERY_RELATION`

The semantic knowledge graph is no longer write‑only. Developers or peer agents can now **update** or **delete** existing triples and query for all objects of a relation:

- **UPDATE_FACT** – Given a `subject`, `relation` and new `object`, this operation searches for a matching `(subject, relation)` pair. If found, it replaces the stored object with the provided one. If no match exists, it appends a new fact. The operation returns `updated: true` on success.
- **REMOVE_FACT** – Removes the first triple matching `subject` and `relation`. Returns `removed: true` if a fact was deleted, `false` otherwise.
- **QUERY_RELATION** – Returns all subject→object pairs for a given relation. An optional `limit` caps the number of results. For example, `QUERY_RELATION` with `relation="is"` might return a list of facts like `{subject: "mars", object: "the red planet"}` and `{subject: "einstein", object: "the father of relativity"}`.

These operations allow developers to maintain the semantic memory (correcting or removing bad facts) and extract related groups of knowledge. They complement existing operations (`ADD_FACT`, `LIST_FACTS`, `LIST_RELATIONS`, `GROUP_KG_BY_RELATION`, `SEARCH_KG`) to provide near‑full CRUD over Maven’s simple knowledge store.

**Stage flow impact:** All semantic memory CRUD operations are accessed via the **Personal Brain API** and do not introduce new pipeline stages. They may, however, indirectly influence reasoning by changing the facts available to Stage 8.

### 2.35 Self‑Review & Improvement Goal Creation

**File:** `brains/cognitive/memory_librarian/service/memory_librarian.py` (Stage 18), optional `config/self_review.json`

After pruning and assimilating QA memory, Maven now performs a **self‑assessment** to identify weak areas. The memory librarian reads the **meta‑confidence** statistics to find domains where recent adjustments are strongly negative (below a threshold, default −0.05). For each underperforming domain, it automatically creates a new goal titled “Improve domain: <domain>” with a description prefixed `SELF_REVIEW:`. These goals are stored in the goal memory and surfaced in the pipeline context as `stage_18_self_review`. The threshold can be customised via `config/self_review.json` (`{"threshold": −0.03}` for example).

This mechanism encourages Maven to **self‑improve** by allocating time to study topics where it performs poorly. The goals will be executed by the autonomy scheduler in subsequent runs according to their priority.

**Stage flow impact:** This addition inserts a new **Stage 18** after memory consolidation. The pipeline now performs: regression testing (Stage 16), QA pruning & knowledge assimilation (Stage 17), and finally self‑review & improvement goal creation (Stage 18). Contexts may now include a `stage_18_self_review` entry listing created goals.

### 2.36 Domain Performance & Classification

**File:** `brains/personal/service/personal_brain.py`    
**Operation:** `DOMAIN_STATS`

To gauge Maven’s **topic‑level expertise**, a new **DOMAIN_STATS** operation returns detailed performance metrics for each tracked domain.  Domains are defined by the first few words of questions (the same keys used in meta‑confidence).  The personal brain gathers success and failure counts from the meta‑confidence file, computes the overall success ratio and the current confidence adjustment, and then classifies each domain into one of three tiers:

* **expert** – success ratio ≥ 80 %;
* **intermediate** – success ratio between 60 % and 79 %;
* **novice** – success ratio < 60 %.

The operation accepts an optional `limit` parameter (default 10) and sorts domains by the number of attempts and success ratio.  Each entry in the returned list includes the domain key, success/failure counts, total attempts, adjustment, success ratio (0–1) and the classification.  This high‑level view complements `META_STATS` and `META_TRENDS` by offering a straightforward overview of where Maven excels and where it needs improvement.

**Stage flow impact:** `DOMAIN_STATS` is part of the **Personal Brain API** and does not affect the main pipeline stages.  It simply reads the meta‑confidence data and returns aggregated metrics on demand.

### 2.37 Goal Introspection & Dependency Queries

**File:** `brains/personal/service/personal_brain.py`    
**Operations:** `GET_GOAL`, `GOAL_DEPENDENCIES`

As Maven takes on increasingly complex task graphs with nested dependencies, developers need tools to inspect individual goals and their prerequisite chains. Two new operations address this need:

* **GET_GOAL** – Given a `goal_id`, return the full goal record, including its title, description, creation and completion timestamps, dependency list, condition and success flag. This allows manual inspection of any stored task.
* **GOAL_DEPENDENCIES** – Given a `goal_id`, traverse the `depends_on` fields backwards and return an ordered list of all ancestor goals. The list is built until no further dependencies exist. Cycles are ignored. The result helps visualise the chain of tasks required before a goal becomes eligible for execution.

These operations are informational only; they do not modify goal memory. They complement existing goal management functions (`ADD_GOAL`, `GET_GOALS`, `COMPLETE_GOAL`, `GOAL_SUMMARY`) by exposing the structure of complex plans.

**Stage flow impact:** `GET_GOAL` and `GOAL_DEPENDENCIES` belong to the **Personal Brain API** and do not affect the cognitive pipeline. They read from the persistent goals file (`goals.jsonl`) and return requested data.

### 2.38 User Knowledge Model & Familiarity‑Driven Verbosity

**Files:** `brains/personal/memory/user_knowledge.py`, `brains/cognitive/language/service/language_brain.py`, `brains/personal/service/personal_brain.py`    
**Operations:** `USER_KNOWLEDGE_STATS`, `RESET_USER_KNOWLEDGE`

To personalise responses based on how well a user knows a topic, Maven now tracks per‑domain *familiarity* via a **User Knowledge Model**.  A domain is defined by the first one or two words of a normalised question (e.g. “what is” or “how do”).  Each time Maven answers a question it updates the count for that domain, decaying older counts by 5 % per day.  The counts are stored in `reports/user_knowledge.json` and categorised into **expert** (≥ 10 counts), **familiar** (≥ 5 counts) or **novice** (< 5 counts).  During Stage 10 (Finalize), the language brain reads this level and adjusts the response’s verbosity: experts receive shorter answers (×0.75), familiar users get slightly shorter answers (×0.9) and novices get more detailed explanations (×1.1).  This helps Maven tailor its answers to the user’s needs without asking redundant questions.

Two personal‑brain operations expose this data:

* **USER_KNOWLEDGE_STATS** – Returns a list of the most frequently encountered domains along with their decayed counts and inferred familiarity levels.  An optional `limit` parameter (default 10) bounds the number of entries.  This operation helps developers understand what topics a user discusses most and can be used for custom personalisation.  It does not modify any data.

* **RESET_USER_KNOWLEDGE** – Clears the user knowledge store by overwriting `reports/user_knowledge.json` with an empty object.  This resets all familiarity counts and timestamps.  It returns `reset: true` on success.  This operation is useful for debugging or when starting a fresh user session.

**Stage flow impact:** The user knowledge model runs inside Stage 10 (Finalize) of the main pipeline and does not introduce new stages. It simply adjusts verbosity based on familiarity and persists domain counts. The administrative operations live in the **Personal Brain API** and have no effect on the reasoning pipeline. They can be called by developers or peer agents to inspect or reset the familiarity data.

### 2.39 Canonical Knowledge Graph Search

**File:** `brains/personal/service/personal_brain.py`    
**Operation:** `SEARCH_KG_CANONICAL`

While the **SEARCH_KG** operation returns facts matching a raw substring, it does not leverage Maven’s synonym mapping. The new **SEARCH_KG_CANONICAL** operation fills this gap by normalising the query and all facts using the synonym mapping before matching. When invoked with a `query` string, the personal brain computes the canonical form via `synonyms.get_canonical()` and then scans the knowledge graph for any triple where either the subject or object canonicalises to the same value. Matches include the full triple and are capped by a `limit` parameter (default 10). This ensures that queries like “red planet” will return facts about `mars` even if the stored fact uses a different phrasing.

For example, suppose the synonym mapping contains “the red planet” → “mars” and the knowledge graph stores `(mars, is, the red planet)`. Calling `SEARCH_KG_CANONICAL` with `query="the red planet"` or `query="red planet"` will find that fact and return it. This behaviour mirrors the `SEARCH_QA_CANONICAL` operation for Q/A memory, providing synonym‑aware search across Maven’s structured knowledge.

**Stage flow impact:** `SEARCH_KG_CANONICAL` is an auxiliary call in the **Personal Brain API**. It does not affect any cognitive stages or alter the knowledge graph. It is intended for debugging and developer use when investigating how synonyms map to stored facts.

### 2.8 Governance Permits & Proof Hooks
**File:** `brains/governance/policy_engine/service/permits.py`  
**Purpose:** Formalize light‑weight permissions for new autonomy types.  Actions request permits such as `IMAGINE(n ≤ 5)`, `CRITIQUE(write)` or `OPINION(update)`.  Each response logs an allow/deny decision with a proof ID to `reports/governance/proofs/`.  This ensures all self‑directed acts remain *audited and reversible*.

---

## 3 · Updated Stage Flow (Simplified)

| # | Stage               | Function (Post‑Upgrade)                              |
|---|--------------------|------------------------------------------------------|
| 1 | Sensorium          | Normalize inputs                                     |
| 2 | Planner            | Goal decomposition + affect bias                    |
| 3 | Language (Parse)   | Natural language understanding                       |
| 4 | Pattern Recognition | Feature mapping                                     |
| 5 | Memory Librarian   | Retrieval + parallel domain search                  |
| 6 | Language (Generate) | Diverge/Converge creative loop                      |
| 7 | Reasoning          | Truth gate + dual‑process router                    |
| 8 | Affect‑Priority    | Emotional weighting                                 |
| 9 | Personality        | Style & tone modulation                             |
| 10 | Language (Finalize)| Response synthesis                                  |
| 11 | System History     | Run summaries, metrics                              |
| 12 | Self‑DMN           | Reflection & critique injection                     |
| 13 | Governance         | Policy enforcement / proofs                          |
| 14 | Affect‑Learn       | Consolidate mood, reflection, identity              |
| 15 | Autonomy & Replan  | Self‑DMN tick, motivation scoring, goal execution and re‑planning |
| 16 | Regression Harness | Run QA memory regression checks and surface mismatches |
| 17 | Memory Consolidation | Prune QA memory, assimilate simple facts into the knowledge graph |

*(Pipeline length may expand in future, but all upgrades respect this order.)*

---

## 4 · Expected Impact

| Dimension          | Effect                                                                    |
|--------------------|---------------------------------------------------------------------------|
| **Reasoning Depth** | Dual‑router + self‑reflection increase factual accuracy and context awareness. |
| **Creativity**      | Divergence/convergence + sandbox allow safe exploration.                    |
| **Emotional Realism** | Affect modulation yields natural tone shifts.                              |
| **Self‑Consistency** | Personal Brain’s identity journal maintains coherent opinions and style.    |
| **Safety & Proofing** | Governance permits keep all new autonomy within logged, reversible boundaries. |

---

## 5 · Verification Plan
1. Baseline regression → ensure 14‑stage integrity.  
2. Dual‑router test → confirm slow path triggers on low confidence.  
3. Reflection log test → verify self‑critique appears per turn.  
4. Sandbox cap test → max 5 rollouts; all produce proof file.  
5. Affect modulation test → observe tone and pacing shifts.  
6. Identity journal test → confirm updates in Personal Brain snapshot.  
7. Governance proof audit → no new stage bypasses allowed.

---

## 6 · Deliverables

| Item                    | Path                                                        |
|------------------------|-------------------------------------------------------------|
| New/updated source files | `maven/brains/...`                                          |
| Config                 | `config/autonomy.json`                                      |
| Reflection logs        | `/reports/reflection/`                                      |
| Identity snapshot      | `/brains/personal/memory/identity_snapshot.json`            |
| Proof logs             | `/reports/governance/proofs/`                               |
| Developer doc          | `/docs/MAVEN_UPGRADE_OVERVIEW.md` (this file)               |
| QA memory file         | `/reports/qa_memory.jsonl`                                   |
| Self‑repair log        | `/reports/self_repair.jsonl`                                  |
| Goal memory file       | `/brains/personal/memory/goals.jsonl`                         |
| Topic stats file       | `/reports/topic_stats.json`                                   |
| User profile file      | `/reports/user_profile.json`                                   |
| Regression harness     | `/maven/tools/regression_harness.py`                          |
| Regression report      | `/reports/regression/results.json`                            |
| Autonomy config        | `/maven/config/autonomy.json`                                 |
| Replanner brain        | `/brains/cognitive/planner/service/replanner_brain.py`        |
| Autonomy brain         | `/brains/cognitive/autonomy/service/autonomy_brain.py`        |
| Autonomy last tick    | `/reports/autonomy/last_tick.json`                            |
| Peer queries log      | `/reports/peer_queries.jsonl`                                  |
| Autonomy budget file   | `/reports/autonomy/budget.json`                                |
| Knowledge graph file    | `/reports/knowledge_graph.json`                               |
| Meta confidence file    | `/reports/meta_confidence.json`                               |
| Memory config file      | `/maven/config/memory.json`                                    |
| Synonyms config file   | `/maven/config/synonyms.json`                                  |
| User knowledge file     | `/reports/user_knowledge.json`                                |

---

## 7 · Summary

These upgrades mark Maven’s transition from a purely logical, memory‑centric system into a **living cognitive architecture** — capable of:

- Balancing intuition and deliberation,
- Reflecting and learning from its own outputs,
- Imagining and planning before acting,
- Exhibiting affective tone and personality continuity,
- All under full governance proof and offline safety.

Maven now **reasons like a person, remembers like a machine, and evolves like both.**