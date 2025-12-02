"""
Execution State Persistence
===========================

This module defines a helper class for persisting the execution
state of long‑running goals.  When the agent daemon is shut down
unexpectedly (e.g. process crash or manual stop), checkpoints can be
used to resume progress on the next startup.  Checkpoints are stored
as JSON files under ``reports/agent/checkpoints`` relative to the
project root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from brains.maven_paths import get_reports_path


class ExecutionState:
    """Persist checkpoints for autonomous task execution."""

    def __init__(self) -> None:
        # Determine the directory for storing checkpoints.  It is
        # located under ``reports/agent/checkpoints`` relative to the
        # project root and must be confined to the Maven workspace.
        self.checkpoint_dir: Path = get_reports_path("agent", "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, goal_id: str, state: Dict[str, Any]) -> None:
        """Persist a checkpoint for the given goal using atomic writes.

        Args:
            goal_id: Identifier of the goal being executed.
            state: Arbitrary state dictionary to persist.
        """
        if not goal_id:
            return
        cp = {
            "goal_id": goal_id,
            "state": state,
        }
        path = self.checkpoint_dir / f"{goal_id}.json"
        try:
            import tempfile, os
            # Serialize the checkpoint first
            data = json.dumps(cp)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(data)
                os.replace(tmp, path)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            # Ignore persistence errors for now
            pass

    def load_checkpoint(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved checkpoint for a goal.

        Args:
            goal_id: Identifier of the goal to restore.

        Returns:
            The checkpoint dictionary, or None if no checkpoint exists
            or it cannot be read.
        """
        if not goal_id:
            return None
        path = self.checkpoint_dir / f"{goal_id}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None