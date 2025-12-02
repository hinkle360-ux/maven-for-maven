"""
Browser Task Reflection
======================

Reflection and learning hooks for browser task execution.
Analyzes completed tasks to improve future performance.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from optional.maven_browser_client.types import (
    BrowserPlan,
    BrowserTaskResult,
    TaskStatus,
    PatternMatch,
    BrowserAction,
    ActionType,
)
from optional.browser_tools.pattern_store import get_pattern_store


@dataclass
class StepAnalysis:
    """Analysis of a single step in task execution."""

    step_index: int
    action: str
    success: bool
    duration_ms: float = 0
    error_type: Optional[str] = None
    notes: str = ""


@dataclass
class TaskReflection:
    """Reflection summary for a completed task."""

    task_id: str
    goal: str
    goal_met: bool
    steps_analysis: List[StepAnalysis] = field(default_factory=list)
    pattern_used: Optional[str] = None
    pattern_effective: bool = True
    unnecessary_steps: List[int] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    new_pattern_suggested: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "goal_met": self.goal_met,
            "steps_analysis": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "success": s.success,
                    "duration_ms": s.duration_ms,
                    "error_type": s.error_type,
                    "notes": s.notes,
                }
                for s in self.steps_analysis
            ],
            "pattern_used": self.pattern_used,
            "pattern_effective": self.pattern_effective,
            "unnecessary_steps": self.unnecessary_steps,
            "failed_steps": self.failed_steps,
            "suggested_improvements": self.suggested_improvements,
            "new_pattern_suggested": self.new_pattern_suggested,
            "timestamp": self.timestamp.isoformat(),
        }


class TaskReflector:
    """Analyzes completed tasks and generates reflections."""

    def __init__(self):
        self.pattern_store = get_pattern_store()

    def reflect(
        self,
        plan: BrowserPlan,
        result: BrowserTaskResult,
        step_results: Optional[List[Dict[str, Any]]] = None,
        pattern_used: Optional[str] = None,
    ) -> TaskReflection:
        """
        Generate reflection for a completed task.

        Args:
            plan: The browser plan that was executed
            result: The task execution result
            step_results: Optional list of per-step results
            pattern_used: Name of pattern used (if any)

        Returns:
            TaskReflection with analysis and suggestions
        """
        # Determine if goal was met
        goal_met = result.status == TaskStatus.COMPLETED

        # Analyze steps
        steps_analysis = self._analyze_steps(plan, step_results or [])

        # Identify failures
        failed_steps = [s.step_index for s in steps_analysis if not s.success]

        # Identify potentially unnecessary steps
        unnecessary_steps = self._identify_unnecessary_steps(plan, steps_analysis)

        # Check if pattern was effective
        pattern_effective = True
        if pattern_used and not goal_met:
            pattern_effective = False

        # Generate improvement suggestions
        suggested_improvements = self._generate_suggestions(
            plan, result, steps_analysis, failed_steps
        )

        # Determine if this could become a new pattern
        new_pattern_suggested = (
            goal_met
            and not pattern_used
            and len(plan.steps) >= 2
            and result.duration_seconds < 30  # Quick success
        )

        return TaskReflection(
            task_id=result.task_id,
            goal=plan.goal,
            goal_met=goal_met,
            steps_analysis=steps_analysis,
            pattern_used=pattern_used,
            pattern_effective=pattern_effective,
            unnecessary_steps=unnecessary_steps,
            failed_steps=failed_steps,
            suggested_improvements=suggested_improvements,
            new_pattern_suggested=new_pattern_suggested,
        )

    def _analyze_steps(
        self,
        plan: BrowserPlan,
        step_results: List[Dict[str, Any]]
    ) -> List[StepAnalysis]:
        """Analyze individual steps."""
        analyses = []

        for i, step in enumerate(plan.steps):
            # Get result for this step if available
            step_result = step_results[i] if i < len(step_results) else {}

            success = step_result.get("status") == "success"
            error_type = step_result.get("error", {}).get("error_type") if not success else None

            analysis = StepAnalysis(
                step_index=i,
                action=step.action.value if hasattr(step.action, 'value') else str(step.action),
                success=success,
                error_type=error_type,
            )
            analyses.append(analysis)

        return analyses

    def _identify_unnecessary_steps(
        self,
        plan: BrowserPlan,
        steps_analysis: List[StepAnalysis]
    ) -> List[int]:
        """Identify steps that might be unnecessary."""
        unnecessary = []

        # Multiple consecutive scrolls might be unnecessary
        scroll_streak = 0
        for i, step in enumerate(plan.steps):
            action = step.action.value if hasattr(step.action, 'value') else str(step.action)
            if action == "scroll":
                scroll_streak += 1
                if scroll_streak > 2:
                    unnecessary.append(i)
            else:
                scroll_streak = 0

        # Screenshots in the middle of plans might be unnecessary
        for i, step in enumerate(plan.steps):
            action = step.action.value if hasattr(step.action, 'value') else str(step.action)
            if action == "screenshot" and i < len(plan.steps) - 1:
                unnecessary.append(i)

        return unnecessary

    def _generate_suggestions(
        self,
        plan: BrowserPlan,
        result: BrowserTaskResult,
        steps_analysis: List[StepAnalysis],
        failed_steps: List[int]
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        # Suggest based on failures
        for step_idx in failed_steps:
            if step_idx < len(steps_analysis):
                analysis = steps_analysis[step_idx]

                if analysis.error_type == "selector_not_found":
                    suggestions.append(
                        f"Step {step_idx}: Selector not found. Consider using more robust selectors or text matching."
                    )
                elif analysis.error_type == "timeout":
                    suggestions.append(
                        f"Step {step_idx}: Timeout. Consider increasing timeout or adding explicit wait steps."
                    )

        # Suggest based on duration
        if result.duration_seconds > 60:
            suggestions.append(
                "Task took over 60 seconds. Consider optimizing step sequence or adding parallelization."
            )

        # Suggest based on step count
        if result.steps_executed > 10:
            suggestions.append(
                "Task required many steps. Consider if intermediate steps can be combined or removed."
            )

        return suggestions

    def update_patterns(self, reflection: TaskReflection) -> None:
        """
        Update pattern store based on reflection.

        Args:
            reflection: TaskReflection to learn from
        """
        # Update pattern success/failure counts
        if reflection.pattern_used:
            if reflection.goal_met:
                self.pattern_store.record_success(reflection.pattern_used)
            else:
                self.pattern_store.record_failure(reflection.pattern_used)

    def suggest_new_pattern(
        self,
        plan: BrowserPlan,
        result: BrowserTaskResult
    ) -> Optional[PatternMatch]:
        """
        Suggest a new pattern based on successful task.

        Args:
            plan: Successful plan
            result: Task result

        Returns:
            Suggested PatternMatch or None
        """
        if result.status != TaskStatus.COMPLETED:
            return None

        # Extract keywords from goal
        keywords = self._extract_keywords(plan.goal)
        if len(keywords) < 2:
            return None

        # Extract domains from plan
        domains = self._extract_domains(plan)

        # Create pattern name
        name = f"auto_{result.task_id[:8]}"

        # Create template plan (replace specific values with placeholders)
        template_plan = self._generalize_plan(plan)

        return PatternMatch(
            name=name,
            description=f"Auto-generated from: {plan.goal}",
            trigger_keywords=keywords,
            domains=domains,
            template_plan=template_plan,
            success_count=1,
            failure_count=0,
        )

    def _extract_keywords(self, goal: str) -> List[str]:
        """Extract keywords from goal description."""
        # Simple keyword extraction
        stopwords = {"the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "is", "it"}
        words = goal.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords[:5]  # Limit to top 5

    def _extract_domains(self, plan: BrowserPlan) -> List[str]:
        """Extract domains from plan."""
        from urllib.parse import urlparse

        domains = []
        for step in plan.steps:
            if step.action == ActionType.OPEN or (hasattr(step.action, 'value') and step.action.value == "open"):
                url = step.params.get("url", "")
                if url:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        domains.append(parsed.netloc)
        return list(set(domains))

    def _generalize_plan(self, plan: BrowserPlan) -> BrowserPlan:
        """Create a generalized template from a specific plan."""
        # For now, return a copy of the plan
        # In a more sophisticated implementation, this would replace
        # specific values with placeholders like {query}, {url}, etc.
        return plan.model_copy(deep=True)


# Global reflector instance
_reflector: Optional[TaskReflector] = None


def get_reflector() -> TaskReflector:
    """Get the global task reflector instance."""
    global _reflector
    if _reflector is None:
        _reflector = TaskReflector()
    return _reflector


def reflect_on_task(
    plan: BrowserPlan,
    result: BrowserTaskResult,
    step_results: Optional[List[Dict[str, Any]]] = None,
    pattern_used: Optional[str] = None,
    auto_update_patterns: bool = True,
) -> TaskReflection:
    """
    Reflect on a completed task.

    Args:
        plan: The browser plan that was executed
        result: The task execution result
        step_results: Optional list of per-step results
        pattern_used: Name of pattern used (if any)
        auto_update_patterns: Whether to automatically update pattern store

    Returns:
        TaskReflection with analysis
    """
    reflector = get_reflector()
    reflection = reflector.reflect(plan, result, step_results, pattern_used)

    if auto_update_patterns:
        reflector.update_patterns(reflection)

    return reflection
