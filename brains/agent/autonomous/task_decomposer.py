"""
Task Decomposer - Real Goal Decomposition
==========================================

This module implements a comprehensive task decomposer for high-level goals.
Given a goal record, the decomposer breaks it down into ordered concrete
actions for the action engine, browser tool, or coder brain.

Decomposition Process:
1. Parse the natural language goal
2. Classify the goal type (gather_info, transform, output)
3. Generate ordered list of concrete actions
4. Enforce max steps and risk budget constraints
5. Validate actions against available tools

Output Format:
Each action is a structured dict with:
- step_id: int - Sequential step identifier
- tool: str - Target tool (action_engine, browser_tool, coder_brain)
- action: str - Specific action type
- params: dict - Action parameters
- depends_on: List[int] - Step dependencies
- risk_level: str - LOW, MEDIUM, HIGH
- critical: bool - Whether step failure should abort
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class StepType(Enum):
    """Types of decomposed steps."""
    GATHER_INFO = "gather_info"  # Information gathering
    TRANSFORM = "transform"      # Data transformation/processing
    OUTPUT = "output"            # Output generation
    VERIFY = "verify"            # Verification step
    CLEANUP = "cleanup"          # Cleanup/finalization


class ToolType(Enum):
    """Available tool targets."""
    ACTION_ENGINE = "action_engine"
    BROWSER_TOOL = "browser_tool"
    CODER_BRAIN = "coder_brain"
    MEMORY = "memory"
    REASONING = "reasoning"


@dataclass
class DecomposedStep:
    """A single decomposed step in an execution plan."""
    step_id: int
    tool: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    risk_level: str = "LOW"
    critical: bool = True
    step_type: str = "transform"
    description: str = ""
    estimated_duration_ms: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecompositionResult:
    """Result of goal decomposition."""
    goal: str
    steps: List[DecomposedStep]
    total_steps: int
    estimated_duration_ms: int
    risk_budget_used: int
    max_risk_level: str
    valid: bool
    validation_errors: List[str] = field(default_factory=list)


class TaskDecomposer:
    """
    Break high-level goals into executable sub-tasks.

    The decomposer analyzes natural language goals and produces
    ordered sequences of concrete actions that can be executed
    by the tool orchestrator.
    """

    # Risk weights for budget calculation
    RISK_WEIGHTS = {"LOW": 1, "MEDIUM": 3, "HIGH": 5, "CRITICAL": 10}

    # Default constraints
    DEFAULT_MAX_STEPS = 20
    DEFAULT_RISK_BUDGET = 30

    # Keywords for goal classification
    RESEARCH_KEYWORDS = {"research", "find", "search", "look up", "investigate", "learn about"}
    ANALYSIS_KEYWORDS = {"analyze", "analyse", "evaluate", "test", "check", "compare", "measure"}
    CREATION_KEYWORDS = {"create", "build", "make", "develop", "write", "implement", "generate"}
    FILE_KEYWORDS = {"file", "read", "write", "list", "directory", "folder"}
    CODE_KEYWORDS = {"code", "function", "class", "python", "script", "program"}

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        risk_budget: int = DEFAULT_RISK_BUDGET
    ):
        """
        Initialize the task decomposer.

        Args:
            max_steps: Maximum number of steps in a decomposition
            risk_budget: Maximum total risk score allowed
        """
        self.max_steps = max_steps
        self.risk_budget = risk_budget

    def decompose(
        self,
        goal: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> DecompositionResult:
        """
        Decompose a goal into a list of executable steps.

        Args:
            goal: A goal record with 'title', 'description', etc.
            constraints: Optional constraints (max_steps, risk_budget)

        Returns:
            DecompositionResult with ordered steps
        """
        if not goal:
            return DecompositionResult(
                goal="",
                steps=[],
                total_steps=0,
                estimated_duration_ms=0,
                risk_budget_used=0,
                max_risk_level="LOW",
                valid=False,
                validation_errors=["Empty goal provided"]
            )

        # Extract goal text
        title = str(goal.get("title", "")).strip()
        description = str(goal.get("description", "")).strip()
        goal_text = f"{title} {description}".strip()

        # Apply constraints
        max_steps = constraints.get("max_steps", self.max_steps) if constraints else self.max_steps
        risk_budget = constraints.get("risk_budget", self.risk_budget) if constraints else self.risk_budget

        # Classify and decompose
        goal_type = self._classify_goal(goal_text)
        steps = self._generate_steps(goal_text, goal_type, goal)

        # Enforce constraints
        steps, validation_errors = self._enforce_constraints(steps, max_steps, risk_budget)

        # Calculate totals
        total_duration = sum(s.estimated_duration_ms for s in steps)
        risk_used = sum(self.RISK_WEIGHTS.get(s.risk_level, 1) for s in steps)
        max_risk = max((s.risk_level for s in steps), default="LOW", key=lambda r: self.RISK_WEIGHTS.get(r, 0))

        return DecompositionResult(
            goal=goal_text,
            steps=steps,
            total_steps=len(steps),
            estimated_duration_ms=total_duration,
            risk_budget_used=risk_used,
            max_risk_level=max_risk,
            valid=len(validation_errors) == 0,
            validation_errors=validation_errors
        )

    def _classify_goal(self, goal_text: str) -> str:
        """
        Classify a goal into broad categories.

        Args:
            goal_text: The goal text to classify

        Returns:
            Category string: research, analysis, creation, file_operation, code_task, generic
        """
        text_lower = goal_text.lower()

        # Check keywords in order of specificity
        if any(kw in text_lower for kw in self.CODE_KEYWORDS):
            return "code_task"

        if any(kw in text_lower for kw in self.FILE_KEYWORDS):
            return "file_operation"

        if any(kw in text_lower for kw in self.RESEARCH_KEYWORDS):
            return "research"

        if any(kw in text_lower for kw in self.ANALYSIS_KEYWORDS):
            return "analysis"

        if any(kw in text_lower for kw in self.CREATION_KEYWORDS):
            return "creation"

        return "generic"

    def _generate_steps(
        self,
        goal_text: str,
        goal_type: str,
        goal: Dict[str, Any]
    ) -> List[DecomposedStep]:
        """Generate steps based on goal type."""
        if goal_type == "research":
            return self._decompose_research(goal_text, goal)
        elif goal_type == "analysis":
            return self._decompose_analysis(goal_text, goal)
        elif goal_type == "creation":
            return self._decompose_creation(goal_text, goal)
        elif goal_type == "file_operation":
            return self._decompose_file_operation(goal_text, goal)
        elif goal_type == "code_task":
            return self._decompose_code_task(goal_text, goal)
        else:
            return self._decompose_generic(goal_text, goal)

    def _decompose_research(self, goal_text: str, goal: Dict[str, Any]) -> List[DecomposedStep]:
        """Decompose a research goal into concrete steps."""
        topic = goal.get("title", goal_text)
        steps = [
            DecomposedStep(
                step_id=1,
                tool="memory",
                action="search_memory",
                params={"query": topic, "limit": 10},
                step_type="gather_info",
                description=f"Search memory for: {topic}",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=2,
                tool="reasoning",
                action="analyze_results",
                params={"topic": topic},
                depends_on=[1],
                step_type="transform",
                description="Analyze search results",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=3,
                tool="browser_tool",
                action="web_search",
                params={"query": topic, "max_results": 5},
                step_type="gather_info",
                description=f"Web search for: {topic}",
                risk_level="MEDIUM",
                critical=False,
            ),
            DecomposedStep(
                step_id=4,
                tool="reasoning",
                action="synthesize",
                params={"sources": ["memory", "web"]},
                depends_on=[1, 2, 3],
                step_type="transform",
                description="Synthesize findings from all sources",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=5,
                tool="memory",
                action="store_results",
                params={"bank": "factual", "topic": topic},
                depends_on=[4],
                step_type="output",
                description="Store research results",
                risk_level="LOW",
            ),
        ]
        return steps

    def _decompose_analysis(self, goal_text: str, goal: Dict[str, Any]) -> List[DecomposedStep]:
        """Decompose an analysis goal into concrete steps."""
        target = goal.get("data_source", goal.get("title", goal_text))
        steps = [
            DecomposedStep(
                step_id=1,
                tool="action_engine",
                action="read_file",
                params={"path": target, "max_bytes": 1000000},
                step_type="gather_info",
                description=f"Load data from: {target}",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=2,
                tool="reasoning",
                action="analyze",
                params={"method": goal.get("analysis_type", "general")},
                depends_on=[1],
                step_type="transform",
                description="Perform analysis",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=3,
                tool="reasoning",
                action="generate_summary",
                params={"format": "markdown"},
                depends_on=[2],
                step_type="output",
                description="Generate analysis report",
                risk_level="LOW",
            ),
        ]
        return steps

    def _decompose_creation(self, goal_text: str, goal: Dict[str, Any]) -> List[DecomposedStep]:
        """Decompose a creation goal into concrete steps."""
        title = goal.get("title", goal_text)
        steps = [
            DecomposedStep(
                step_id=1,
                tool="reasoning",
                action="plan",
                params={"task": title},
                step_type="gather_info",
                description=f"Plan creation: {title}",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=2,
                tool="coder_brain",
                action="generate",
                params={"spec": title},
                depends_on=[1],
                step_type="transform",
                description="Generate content",
                risk_level="MEDIUM",
            ),
            DecomposedStep(
                step_id=3,
                tool="coder_brain",
                action="verify",
                params={},
                depends_on=[2],
                step_type="verify",
                description="Verify generated content",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=4,
                tool="action_engine",
                action="write_file",
                params={"backup": True},
                depends_on=[3],
                step_type="output",
                description="Save creation",
                risk_level="MEDIUM",
            ),
        ]
        return steps

    def _decompose_file_operation(self, goal_text: str, goal: Dict[str, Any]) -> List[DecomposedStep]:
        """Decompose a file operation goal."""
        text_lower = goal_text.lower()
        steps = []

        if "list" in text_lower or "find" in text_lower:
            steps.append(DecomposedStep(
                step_id=1,
                tool="action_engine",
                action="list_python_files",
                params={"max_files": 100},
                step_type="gather_info",
                description="List files",
                risk_level="LOW",
            ))

        if "read" in text_lower:
            steps.append(DecomposedStep(
                step_id=len(steps) + 1,
                tool="action_engine",
                action="read_file",
                params={"max_bytes": 100000},
                step_type="gather_info",
                description="Read file contents",
                risk_level="LOW",
            ))

        if "write" in text_lower or "save" in text_lower:
            steps.append(DecomposedStep(
                step_id=len(steps) + 1,
                tool="action_engine",
                action="write_file",
                params={"backup": True},
                step_type="output",
                description="Write to file",
                risk_level="MEDIUM",
            ))

        # Add summary step
        steps.append(DecomposedStep(
            step_id=len(steps) + 1,
            tool="reasoning",
            action="summarize",
            params={},
            depends_on=[s.step_id for s in steps],
            step_type="output",
            description="Summarize results",
            risk_level="LOW",
        ))

        return steps

    def _decompose_code_task(self, goal_text: str, goal: Dict[str, Any]) -> List[DecomposedStep]:
        """Decompose a coding task."""
        spec = goal.get("title", goal_text)
        steps = [
            DecomposedStep(
                step_id=1,
                tool="coder_brain",
                action="plan",
                params={"spec": spec},
                step_type="gather_info",
                description="Plan code implementation",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=2,
                tool="coder_brain",
                action="generate",
                params={"spec": spec},
                depends_on=[1],
                step_type="transform",
                description="Generate code",
                risk_level="MEDIUM",
            ),
            DecomposedStep(
                step_id=3,
                tool="coder_brain",
                action="verify",
                params={},
                depends_on=[2],
                step_type="verify",
                description="Verify code (lint + test)",
                risk_level="LOW",
            ),
            DecomposedStep(
                step_id=4,
                tool="coder_brain",
                action="refine",
                params={},
                depends_on=[3],
                step_type="transform",
                description="Refine if tests fail",
                risk_level="MEDIUM",
                critical=False,
            ),
        ]
        return steps

    def _decompose_generic(self, goal_text: str, goal: Dict[str, Any]) -> List[DecomposedStep]:
        """Fallback decomposition for unclassified goals."""
        # Parse goal for conjunctions
        parts = re.split(r'\band\b|[,;]', goal_text, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            # Single task - create a simple gather/transform/output sequence
            return [
                DecomposedStep(
                    step_id=1,
                    tool="reasoning",
                    action="understand",
                    params={"goal": goal_text},
                    step_type="gather_info",
                    description="Understand the goal",
                    risk_level="LOW",
                ),
                DecomposedStep(
                    step_id=2,
                    tool="reasoning",
                    action="process",
                    params={"goal": goal_text},
                    depends_on=[1],
                    step_type="transform",
                    description="Process and respond",
                    risk_level="LOW",
                ),
            ]

        # Multiple parts - create step for each
        steps = []
        for i, part in enumerate(parts, 1):
            sub_type = self._classify_goal(part)
            steps.append(DecomposedStep(
                step_id=i,
                tool="reasoning",
                action=sub_type,
                params={"task": part},
                depends_on=[i - 1] if i > 1 else [],
                step_type="transform",
                description=part[:50],
                risk_level="LOW",
            ))

        return steps

    def _enforce_constraints(
        self,
        steps: List[DecomposedStep],
        max_steps: int,
        risk_budget: int
    ) -> tuple[List[DecomposedStep], List[str]]:
        """
        Enforce max_steps and risk_budget constraints.

        Returns:
            Tuple of (filtered_steps, validation_errors)
        """
        errors = []
        filtered_steps = []
        current_risk = 0

        for step in steps:
            # Check max steps
            if len(filtered_steps) >= max_steps:
                errors.append(f"Step {step.step_id} dropped: max_steps ({max_steps}) exceeded")
                continue

            # Check risk budget
            step_risk = self.RISK_WEIGHTS.get(step.risk_level, 1)
            if current_risk + step_risk > risk_budget:
                if step.critical:
                    errors.append(f"Step {step.step_id} dropped: risk_budget ({risk_budget}) exceeded")
                continue

            filtered_steps.append(step)
            current_risk += step_risk

        # Re-number steps
        for i, step in enumerate(filtered_steps, 1):
            step.step_id = i

        return filtered_steps, errors

    def get_decomposition_summary(self, result: DecompositionResult) -> Dict[str, Any]:
        """Get a summary of a decomposition for logging."""
        return {
            "goal": result.goal[:100],
            "total_steps": result.total_steps,
            "valid": result.valid,
            "risk_used": result.risk_budget_used,
            "tools_used": list(set(s.tool for s in result.steps)),
            "error_count": len(result.validation_errors),
        }
