#!/usr/bin/env python3
"""Agent CLI
============

This script provides a simple command‑line interface for interacting
with the Maven autonomous agent.  Users can start and stop the
background daemon, add goals, list existing goals, and query goal
progress.  The CLI relies on the goal memory and agent daemon
implemented in the ``brains.agent.autonomous`` package.

Running the CLI
---------------

Execute this module directly from the Maven project root::

    python -m maven.brains.agent.autonomous.agent_cli start

To see available commands and usage::

    python -m maven.brains.agent.autonomous.agent_cli --help

This CLI operates in a single process.  Starting the agent will
launch a background thread for autonomous execution.  Stopping the
agent will attempt to join the thread gracefully.
"""

from __future__ import annotations

import argparse
import sys
import json
from typing import Any, Dict

from .agent_daemon import AgentDaemon
from .goal_queue import GoalQueue
from .progress_tracker import ProgressTracker


def _print_json(data: Any) -> None:
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Maven Autonomous Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # Start/stop the agent daemon
    sub.add_parser("start", help="Start the autonomous agent in the background")
    sub.add_parser("stop", help="Stop the autonomous agent")

    # Add goal
    add = sub.add_parser("add", help="Add a new goal")
    add.add_argument("title", type=str, help="Title of the goal")
    add.add_argument("--description", type=str, default=None, help="Optional description")
    add.add_argument("--depends_on", type=str, nargs="*", default=None, help="List of goal IDs this depends on")
    add.add_argument("--priority", type=float, default=None, help="Priority weighting (0..1)")

    # List goals
    ls = sub.add_parser("goals", help="List goals")
    ls.add_argument("--all", action="store_true", help="Include completed goals")

    # Show progress
    prog = sub.add_parser("progress", help="Show progress for a goal")
    prog.add_argument("goal_id", type=str, help="Goal ID to query")

    # Status
    sub.add_parser("status", help="Show agent status and active goals")

    args = parser.parse_args(argv)

    # Create a single daemon instance.  In an interactive environment,
    # this will create at most one running thread.  A production
    # implementation might use a shared process or external service.
    # We rely on process‑global state: starting multiple CLI instances
    # will create separate daemons that operate on the same goal store.
    # Use with care.
    global _daemon
    if "_daemon" not in globals():
        _daemon = AgentDaemon()
    daemon: AgentDaemon = _daemon
    queue = GoalQueue()
    progress = daemon.progress_tracker

    cmd = args.command
    if cmd == "start":
        daemon.start()
        print("Agent daemon started.")
    elif cmd == "stop":
        daemon.stop()
        print("Agent daemon stopped.")
    elif cmd == "add":
        rec = queue.add_goal(
            args.title,
            description=args.description,
            depends_on=args.depends_on,
            priority=args.priority,
        )
        _print_json(rec)
    elif cmd == "goals":
        goals = queue.list_goals(active_only=not args.all)
        _print_json(goals)
    elif cmd == "progress":
        info = progress.get_progress(args.goal_id)
        # Try to fetch goal title for context
        goal_list = queue.load_goals(active_only=False)
        goal = next((g for g in goal_list if g.get("goal_id") == args.goal_id), None)
        if goal:
            report = progress.generate_report(args.goal_id, goal)
        else:
            report = info
        _print_json(report)
    elif cmd == "status":
        # Show number of active goals and running state
        active = queue.load_goals(active_only=True)
        stat: Dict[str, Any] = {
            "running": daemon.running,
            "active_goals": len(active),
            "goals": [g.get("title", g.get("goal_id")) for g in active],
        }
        _print_json(stat)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])