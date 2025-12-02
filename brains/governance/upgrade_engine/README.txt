Maven Upgrade Engine
====================

This module contains the **Upgrade Engine** for the Maven project.  It is part
of the governance layer and is responsible for proposing and applying
improvements to the system.  Unlike the Repair Engine, which focuses on
maintaining the health of existing code and data, the Upgrade Engine
analyzes system metadata, performance logs and role charters to suggest
changes that enhance behaviour or extend functionality.

Key responsibilities:

* **Scanning and analysis** – inspect system history and other
  metadata for opportunities to improve.  This is a non‑destructive
  operation and requires no special authorization.

* **Proposal generation** – when a potential improvement is found,
  prepare a structured proposal describing the change.  Proposals are
  stored via the Memory Librarian and reviewed by the human operator
  or governance layer.

* **Application of approved upgrades** – once a proposal has been
  authorized (e.g. via a governance token), apply the change to the
  system.  This is a destructive operation and requires a valid
  authorization token from the Governance brain.

For more details on the upgrade process and how to interact with the
Upgrade Engine, see `service/upgrade_engine.py`.