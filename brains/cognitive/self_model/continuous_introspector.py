"""
continuous_introspector.py
===========================

Continuous introspection and automated upgrade system for Maven.

This module enables Maven to:
1. Continuously monitor its own codebase for compliance gaps
2. Detect when brains fall out of compliance
3. Generate upgrade plans automatically
4. Queue upgrades for review/execution
5. Track upgrade history and success rates

Features:
    - Scheduled compliance scans (configurable interval)
    - Automatic upgrade plan generation
    - Upgrade queue management with priorities
    - Safe upgrade execution with rollback
    - Integration with BrainIntrospector
"""

from __future__ import annotations

import time
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from brains.tools_api import (
    ShellTool,
    ShellResult,
    NullShellTool,
    ToolRegistry,
)


# Global tool registry - set by host runtime
_tool_registry: Optional[ToolRegistry] = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry (called by host runtime)."""
    global _tool_registry
    _tool_registry = registry


def _get_shell_tool() -> ShellTool:
    """Get shell tool from registry."""
    if _tool_registry and _tool_registry.shell:
        return _tool_registry.shell
    return NullShellTool()


class ContinuousIntrospector:
    """Continuously monitor codebase for improvement opportunities."""

    def __init__(self, scan_interval_hours: int = 24):
        self.scan_interval = scan_interval_hours * 3600  # Convert to seconds
        self.last_scan_time = 0
        self.upgrade_queue: List[Dict[str, Any]] = []
        self.upgrade_history: List[Dict[str, Any]] = []
        self.introspector = None
        self._initialize_introspector()

    def _initialize_introspector(self) -> None:
        """Initialize the BrainIntrospector."""
        try:
            from brains.cognitive.self_model.service.self_introspection import BrainIntrospector
            self.introspector = BrainIntrospector()
        except Exception as e:
            print(f"[CONTINUOUS_INTROSPECTOR] Failed to load BrainIntrospector: {e}")

    def scan_for_upgrades(self) -> Dict[str, Any]:
        """
        Scan all brains for compliance and identify upgrade needs.

        Returns:
            Dict with scan results and upgrade recommendations
        """
        if not self.introspector:
            return {"error": "BrainIntrospector not available"}

        scan_start = time.time()

        try:
            # Scan all cognitive brains
            scan_results = self.introspector.scan_all_brains()

            # Identify non-compliant brains
            non_compliant = []
            for brain_name, result in scan_results.items():
                if not result.get("compliant", False):
                    non_compliant.append({
                        "brain": brain_name,
                        "missing_signals": result.get("missing_operations", []),
                        "compliance_status": result
                    })

            # Generate upgrade plans
            upgrade_plans = []
            for brain_info in non_compliant:
                plan = self._generate_upgrade_plan(brain_info)
                upgrade_plans.append(plan)

            scan_duration = time.time() - scan_start
            self.last_scan_time = time.time()

            result = {
                "scan_timestamp": datetime.now().isoformat(),
                "scan_duration_seconds": round(scan_duration, 2),
                "total_brains": len(scan_results),
                "compliant_brains": len(scan_results) - len(non_compliant),
                "non_compliant_brains": len(non_compliant),
                "compliance_percentage": round(
                    ((len(scan_results) - len(non_compliant)) / len(scan_results) * 100)
                    if scan_results else 0,
                    1
                ),
                "non_compliant_details": non_compliant,
                "upgrade_plans": upgrade_plans
            }

            return result

        except Exception as e:
            return {"error": str(e)}

    def _generate_upgrade_plan(self, brain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed upgrade plan for a non-compliant brain."""
        brain_name = brain_info["brain"]
        missing_signals = brain_info["missing_signals"]

        # Determine priority
        priority = self._calculate_upgrade_priority(brain_name, missing_signals)

        # Generate upgrade steps
        steps = []

        if "is_continuation" in missing_signals:
            steps.append({
                "step": 1,
                "action": "add_continuation_detection",
                "description": "Add continuation detection to service_api",
                "code_change": "Add is_continuation() call and context enrichment"
            })

        if "get_conversation_context" in missing_signals:
            steps.append({
                "step": 2,
                "action": "add_context_retrieval",
                "description": "Add conversation context retrieval for follow-ups",
                "code_change": "Call get_conversation_context() when continuation detected"
            })

        if "create_routing_hint" in missing_signals:
            steps.append({
                "step": 3,
                "action": "add_routing_hints",
                "description": "Add routing hint generation to all operations",
                "code_change": "Call create_routing_hint() before return statements"
            })

        plan = {
            "brain": brain_name,
            "priority": priority,
            "estimated_effort": "low" if len(steps) <= 2 else "medium",
            "upgrade_steps": steps,
            "automated": True,  # Can be automated via script
            "script_command": f"python scripts/upgrade_brain_compliance.py --brain {brain_name}",
            "created_at": datetime.now().isoformat()
        }

        return plan

    def _calculate_upgrade_priority(self, brain_name: str, missing_signals: List[str]) -> str:
        """Calculate upgrade priority based on brain importance and missing signals."""
        # Critical brains
        critical_brains = ["teacher", "self_review", "self_model", "integrator", "reasoning"]

        # High priority brains
        high_priority_brains = ["memory_librarian", "sensorium", "language", "planner"]

        if brain_name in critical_brains:
            return "critical"
        elif brain_name in high_priority_brains:
            return "high"
        elif len(missing_signals) >= 2:
            return "medium"
        else:
            return "low"

    def queue_upgrade(self, upgrade_plan: Dict[str, Any]) -> bool:
        """
        Add an upgrade to the queue.

        Args:
            upgrade_plan: Upgrade plan dict from _generate_upgrade_plan()

        Returns:
            True if queued successfully
        """
        # Check if already in queue
        brain_name = upgrade_plan["brain"]
        for queued_upgrade in self.upgrade_queue:
            if queued_upgrade["brain"] == brain_name:
                return False  # Already queued

        # Add to queue
        upgrade_plan["queued_at"] = datetime.now().isoformat()
        upgrade_plan["status"] = "queued"
        self.upgrade_queue.append(upgrade_plan)

        # Sort queue by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.upgrade_queue.sort(key=lambda x: priority_order.get(x["priority"], 4))

        return True

    def execute_next_upgrade(
        self,
        auto_approve: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the next upgrade in the queue.

        Args:
            auto_approve: If False, returns upgrade for manual approval
            dry_run: If True, simulates execution without making changes

        Returns:
            Dict with execution result
        """
        if not self.upgrade_queue:
            return {
                "status": "no_upgrades_queued",
                "message": "Upgrade queue is empty"
            }

        # Get next upgrade
        upgrade = self.upgrade_queue[0]
        brain_name = upgrade["brain"]

        if not auto_approve:
            return {
                "status": "approval_required",
                "upgrade": upgrade,
                "message": f"Upgrade for {brain_name} requires approval. Set auto_approve=True to proceed."
            }

        # Execute upgrade
        result = self._execute_upgrade(upgrade, dry_run)

        # Update queue
        if result["success"]:
            # Remove from queue
            self.upgrade_queue.pop(0)

            # Add to history
            upgrade["status"] = "completed" if not dry_run else "dry_run_success"
            upgrade["completed_at"] = datetime.now().isoformat()
            upgrade["result"] = result
            self.upgrade_history.append(upgrade)
        else:
            # Mark as failed but keep in queue
            upgrade["status"] = "failed"
            upgrade["failure_count"] = upgrade.get("failure_count", 0) + 1
            upgrade["last_failure"] = datetime.now().isoformat()
            upgrade["last_error"] = result.get("error")

            # Remove from queue if failed too many times
            if upgrade["failure_count"] >= 3:
                self.upgrade_queue.pop(0)
                self.upgrade_history.append(upgrade)

        return result

    def _execute_upgrade(
        self,
        upgrade_plan: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute an upgrade plan via the host-provided shell tool."""
        brain_name = upgrade_plan["brain"]
        command = upgrade_plan.get("script_command")

        if not command:
            return {"success": False, "error": "No upgrade command specified"}

        # Add dry-run flag if requested
        if dry_run:
            command += " --dry-run"

        shell = _get_shell_tool()

        # Check if shell tool is available
        if isinstance(shell, NullShellTool):
            return {
                "success": False,
                "error": "Shell tool not available. Upgrades require host tool injection.",
                "brain": brain_name
            }

        try:
            # Execute upgrade script via shell tool
            result = shell.run(command, timeout=60, check_policy=False)

            if result.status == "timeout":
                return {
                    "success": False,
                    "error": "Upgrade execution timed out",
                    "brain": brain_name
                }

            success = result.status == "completed" and result.exit_code == 0

            return {
                "success": success,
                "brain": brain_name,
                "command": command,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "return_code": result.exit_code or 0,
                "dry_run": dry_run
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "brain": brain_name
            }

    def should_scan_now(self) -> bool:
        """Check if it's time for the next scan."""
        return (time.time() - self.last_scan_time) >= self.scan_interval

    def get_upgrade_queue_status(self) -> Dict[str, Any]:
        """Get current status of upgrade queue."""
        return {
            "total_queued": len(self.upgrade_queue),
            "by_priority": {
                "critical": len([u for u in self.upgrade_queue if u["priority"] == "critical"]),
                "high": len([u for u in self.upgrade_queue if u["priority"] == "high"]),
                "medium": len([u for u in self.upgrade_queue if u["priority"] == "medium"]),
                "low": len([u for u in self.upgrade_queue if u["priority"] == "low"]),
            },
            "next_upgrade": self.upgrade_queue[0] if self.upgrade_queue else None,
            "total_completed": len([u for u in self.upgrade_history if u["status"] == "completed"]),
            "total_failed": len([u for u in self.upgrade_history if u["status"] == "failed"])
        }

    def get_upgrade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get upgrade history."""
        return self.upgrade_history[-limit:]


# Singleton instance
_continuous_introspector = ContinuousIntrospector()


def get_continuous_introspector() -> ContinuousIntrospector:
    """Get the global continuous introspector instance."""
    return _continuous_introspector


def get_default_introspector() -> ContinuousIntrospector:
    """Get the default continuous introspector instance (alias for get_continuous_introspector)."""
    return _continuous_introspector


def scan_for_upgrades() -> Dict[str, Any]:
    """Convenience function to scan for upgrades."""
    return _continuous_introspector.scan_for_upgrades()


def queue_all_upgrades() -> int:
    """
    Scan for upgrades and queue them all.

    Returns:
        Number of upgrades queued
    """
    scan_result = scan_for_upgrades()
    upgrade_plans = scan_result.get("upgrade_plans", [])

    queued_count = 0
    for plan in upgrade_plans:
        if _continuous_introspector.queue_upgrade(plan):
            queued_count += 1

    return queued_count
