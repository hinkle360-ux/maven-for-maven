"""Background daemon to periodically invoke pattern learning for the LLM.

This module defines ``LearningDaemon``, a small helper class that wakes up
at a configured time each day to run ``llm_service.learn_patterns()``.  It
uses only Python's standard library (``datetime``, ``time`` and
``threading``) to avoid external dependencies.  When run directly, it
executes a single learning cycle immediately for testing.

Extended to support brain-specific learning cycles:
- Reasoning Brain: Load strategies from lessons, optionally run training episodes
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

try:
    # Import the global LLM service.  On import failure the daemon
    # becomes a no‑op.
    from brains.tools.llm_service import llm_service  # type: ignore
except Exception:
    llm_service = None  # type: ignore

# Import LearningMode for training episodes
try:
    from brains.learning.learning_mode import LearningMode
except Exception:
    class LearningMode:  # type: ignore
        TRAINING = "training"
        OFFLINE = "offline"
        SHADOW = "shadow"


class LearningDaemon:
    """Run nightly learning cycles to update LLM response templates."""

    def __init__(self) -> None:
        # Determine if the daemon should run based on configuration.
        # If llm_service is unavailable or disabled, disable the daemon.
        if llm_service is None:
            self.enabled = False
            self.learning_enabled = False
            self.learning_time = "02:00"
            return
        cfg = getattr(llm_service, "config", {}) or {}
        sched = cfg.get("scheduling", {}) or {}
        self.learning_enabled: bool = bool(sched.get("learning_enabled", True))
        self.learning_time: str = str(sched.get("learning_time", "02:00"))
        self.enabled: bool = True

    def run_learning_cycle(self) -> None:
        """Execute a single learning cycle and print a summary to stdout."""
        if llm_service is None or not self.learning_enabled:
            return
        print("\n" + "=" * 50)
        print("LEARNING DAEMON: Starting pattern analysis")
        print("=" * 50)
        try:
            # Trigger pattern learning on the LLM service
            llm_service.learn_patterns()
            stats = llm_service.get_learning_stats()
            total_templates = stats.get("total_templates", 0)
            total_interactions = stats.get("total_interactions", 0)
            print(f"\nLearning Results:")
            print(f"  Total templates: {total_templates}")
            print(f"  Total interactions: {total_interactions}")
            # Compute independence ratio if possible
            try:
                tmpl_hits = sum(stats.get("templates_by_usage", {}).values())
                if total_interactions > 0:
                    independence = (tmpl_hits / total_interactions) * 100.0
                    print(f"  LLM independence: {independence:.1f}%")
            except Exception:
                pass
        except Exception as exc:
            # Log exception to stdout but continue
            print(f"[Learning Daemon] Exception during learning: {exc}")
        print("=" * 50 + "\n")

    def start_scheduled(self) -> None:
        """Start the learning daemon loop.

        This loop computes the next scheduled time based on ``learning_time``
        and sleeps until then.  It runs indefinitely on a background thread
        when invoked from run_maven.py.  A misconfiguration or runtime
        exception will silently stop the loop.
        """
        if not self.enabled or not self.learning_enabled:
            return
        print(f"[Learning Daemon] Scheduled for {self.learning_time} daily")
        while self.enabled and self.learning_enabled:
            try:
                # Compute next run datetime
                now = datetime.now()
                try:
                    hour_str, minute_str = self.learning_time.split(":", 1)
                    target = now.replace(hour=int(hour_str), minute=int(minute_str), second=0, microsecond=0)
                except Exception:
                    # Fallback to 02:00 if parse fails
                    target = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if target <= now:
                    # If target time has passed today, schedule for next day
                    target = target + timedelta(days=1)
                # Sleep in shorter intervals to allow responsive shutdown
                while datetime.now() < target and self.enabled and self.learning_enabled:
                    # Sleep at most 60 seconds at a time
                    delta = (target - datetime.now()).total_seconds()
                    sleep_for = 60.0 if delta > 60.0 else max(delta, 0)
                    time.sleep(sleep_for)
                # Trigger learning cycle if still enabled
                if self.enabled and self.learning_enabled:
                    self.run_learning_cycle()
            except Exception as exc:
                # On any exception, log and break the loop to avoid tight spin
                print(f"[Learning Daemon] Error in loop: {exc}")
                break

    def run_once(self) -> None:
        """Run a single learning cycle immediately (for testing)."""
        self.run_learning_cycle()

    # =========================================================================
    # Brain-Specific Learning Hooks
    # =========================================================================

    def update_reasoning_strategies(self) -> None:
        """
        Refresh the Reasoning Brain's strategy table from stored lessons.

        This should be called periodically to ensure the reasoning brain
        has access to the latest learned strategies.
        """
        try:
            from brains.cognitive.reasoning.service.reasoning_brain import (
                load_strategies_from_lessons,
                STRATEGY_TABLE
            )

            print("[Learning Daemon] Updating Reasoning Brain strategies...")

            # Load strategies from lessons
            load_strategies_from_lessons({})

            print(f"[Learning Daemon] Reasoning Brain has {len(STRATEGY_TABLE)} strategies loaded")

        except Exception as e:
            print(f"[Learning Daemon] Failed to update reasoning strategies: {e}")

    def train_reasoning_brain_once(self, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Run a single reasoning training episode.

        This is the hook for running reasoning training episodes.
        The pipeline must provide tasks and context.

        Args:
            context: Optional context dict for the training episode.
                     Should include:
                     - "learning_mode": LearningMode.TRAINING
                     - "training_task": The task/question to reason about

        TODO: This function needs a task harness to provide training queries.
        Currently a placeholder that logs intent.
        """
        if context is None:
            context = {}

        # Ensure training mode is set
        context["learning_mode"] = LearningMode.TRAINING

        training_task = context.get("training_task", "")

        if not training_task:
            print("[Learning Daemon] No training task provided, skipping reasoning training")
            return

        try:
            from brains.cognitive.reasoning.service.reasoning_brain import (
                reasoning_llm_lesson,
                load_strategies_from_lessons
            )

            print(f"[Learning Daemon] Running reasoning training for: {training_task[:50]}...")

            # Generate lesson from the training task
            lesson = reasoning_llm_lesson(
                context=context,
                question=training_task,
                learning_mode=LearningMode.TRAINING
            )

            if lesson:
                print(f"[Learning Daemon] Generated lesson: {lesson.get('topic', 'unknown')}")

                # Reload strategies to include new lesson
                load_strategies_from_lessons({})
            else:
                print("[Learning Daemon] No lesson generated from training task")

        except Exception as e:
            print(f"[Learning Daemon] Reasoning training failed: {e}")
            import traceback
            traceback.print_exc()

    def run_brain_learning_cycles(self) -> None:
        """
        Run learning cycles for all supported brains.

        Currently supports:
        - Reasoning Brain: Strategy updates from lessons
        """
        print("\n" + "-" * 50)
        print("LEARNING DAEMON: Brain-specific learning cycles")
        print("-" * 50)

        # Update reasoning strategies
        self.update_reasoning_strategies()

        print("-" * 50 + "\n")


if __name__ == "__main__":
    # When executed directly, run one cycle for testing
    daemon = LearningDaemon()
    daemon.run_once()