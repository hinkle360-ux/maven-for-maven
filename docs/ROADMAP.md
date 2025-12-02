# Maven Roadmap – Repository Shape Freeze

This roadmap entry documents the frozen layout of the Maven
repository.  A consistent structure makes it easier to reason
about upgrades, tooling and automation.  **Do not add new
top‑level directories**; all new modules and resources must live
inside one of the existing categories listed below.

## Top‑Level Directories

The following directories are reserved and must exist in every
Maven checkout:

| Directory | Purpose |
|-----------|---------|
| `api/`    | Public service APIs, utilities and helpers. |
| `brains/` | Cognitive and non‑cognitive brain modules, including agent and governance code. |
| `config/` | Static configuration files (JSON/YAML) that alter runtime behaviour. |
| `docs/`   | Markdown documentation, design notes and roadmaps. |
| `reports/`| Generated outputs such as run traces, audits, benchmarks and governance logs. |
| `runbooks/`| YAML or Markdown runbooks used to orchestrate upgrades, repairs and maintenance. |
| `templates/`| Template files used by the agent and planner for generating patch diffs and messages. |
| `tests/`  | Unit tests, smoke packs and benchmark suites. |
| `tools/`  | Helper scripts and command‑line utilities invoked by CI or runbooks. |
| `ui/`     | User‑facing interfaces (CLI and chat) that drive the underlying brains. |

If you need to add new functionality, choose the appropriate
directory above (e.g. place new cognitive components under
`brains/cognitive`, new CLI options under `ui/`, and new helper
scripts under `tools/`).  Keeping the shape of the repository stable
reduces the surface area for regressions and simplifies version control
diffs.

This section should be updated only if the core team decides to
introduce a new category across all deployments.  Such a change must
be reflected in the pre‑upgrade checklist and communicated broadly to
contributors.