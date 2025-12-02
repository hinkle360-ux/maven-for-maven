"""
Tests for the Autonomous Agent Modules
======================================

Tests that verify:
1. TaskDecomposer generates proper sequences
2. ToolOrchestrator calls correct tools
3. ExecutionEngine returns correct summary
"""

import pytest
import sys
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestTaskDecomposer:
    """Test suite for the task decomposer."""

    def test_decompose_file_operation(self):
        """Test decomposing a file operation goal."""
        from brains.agent.autonomous.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()

        goal = {
            "title": "List all Python files under sandbox/project",
            "description": "Find Python files and report their sizes"
        }

        result = decomposer.decompose(goal)

        assert result.valid is True
        assert result.total_steps > 0
        assert len(result.steps) > 0

        # Should include action_engine for file operations
        tools_used = {s.tool for s in result.steps}
        assert "action_engine" in tools_used or "reasoning" in tools_used

    def test_decompose_code_task(self):
        """Test decomposing a coding task."""
        from brains.agent.autonomous.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()

        goal = {
            "title": "Write a function that calculates factorial",
            "description": "Create a Python function for factorial"
        }

        result = decomposer.decompose(goal)

        assert result.valid is True
        assert len(result.steps) > 0

        # Should include coder_brain for code tasks
        tools_used = {s.tool for s in result.steps}
        assert "coder_brain" in tools_used

    def test_decompose_research_goal(self):
        """Test decomposing a research goal."""
        from brains.agent.autonomous.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()

        goal = {
            "title": "Research quantum computing basics",
            "description": "Learn about quantum computing"
        }

        result = decomposer.decompose(goal)

        assert result.valid is True
        assert len(result.steps) >= 2

        # Should have gather_info steps
        step_types = {s.step_type for s in result.steps}
        assert "gather_info" in step_types

    def test_enforce_max_steps(self):
        """Test that max_steps constraint is enforced."""
        from brains.agent.autonomous.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer(max_steps=3)

        goal = {
            "title": "Research and implement a complex system",
            "description": "Do many things"
        }

        result = decomposer.decompose(goal)

        assert len(result.steps) <= 3

    def test_enforce_risk_budget(self):
        """Test that risk budget constraint is enforced."""
        from brains.agent.autonomous.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer(risk_budget=5)

        goal = {
            "title": "Write code and modify files",
            "description": "Multiple operations"
        }

        result = decomposer.decompose(goal)

        assert result.risk_budget_used <= 5

    def test_empty_goal(self):
        """Test handling of empty goal."""
        from brains.agent.autonomous.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()

        result = decomposer.decompose({})

        assert result.valid is False
        assert len(result.validation_errors) > 0


class TestToolOrchestrator:
    """Test suite for the tool orchestrator."""

    def test_get_available_tools(self):
        """Test getting list of available tools."""
        from brains.agent.autonomous.tool_orchestrator import ToolOrchestrator

        orchestrator = ToolOrchestrator()
        tools = orchestrator.get_available_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        for tool in tools:
            assert "name" in tool
            assert "available" in tool or "allowed" in tool

    def test_execute_step_unknown_tool(self):
        """Test executing step with unknown tool."""
        from brains.agent.autonomous.tool_orchestrator import (
            ToolOrchestrator, ExecutionContext
        )

        orchestrator = ToolOrchestrator()
        context = ExecutionContext()

        step = {
            "step_id": 1,
            "tool": "nonexistent_tool",
            "action": "do_something",
            "params": {},
        }

        result = orchestrator.execute_step(step, context)

        assert result.success is False
        assert "unknown tool" in result.error.lower()

    def test_execution_context(self):
        """Test execution context stores outputs correctly."""
        from brains.agent.autonomous.tool_orchestrator import ExecutionContext

        context = ExecutionContext()

        # Store output
        context.set_output(1, {"result": "test"})

        # Retrieve output
        output = context.get_step_output(1)
        assert output == {"result": "test"}

        # Store and retrieve variables
        context.set_variable("my_var", "value")
        assert context.get_variable("my_var") == "value"
        assert context.get_variable("nonexistent", "default") == "default"


class TestExecutionEngine:
    """Test suite for the execution engine."""

    def test_decompose_goal_legacy(self):
        """Test legacy decompose_goal interface."""
        from brains.agent.autonomous.execution_engine import ExecutionEngine

        engine = ExecutionEngine()

        subtasks = engine.decompose_goal("List Python files and summarize their sizes")

        assert isinstance(subtasks, list)
        assert len(subtasks) > 0

    def test_plan_execution(self):
        """Test planning execution without running."""
        from brains.agent.autonomous.execution_engine import ExecutionEngine

        engine = ExecutionEngine()

        plan = engine.plan_execution("Analyze code structure")

        assert "goal" in plan
        assert "tasks" in plan
        assert len(plan["tasks"]) > 0

        for task in plan["tasks"]:
            assert "id" in task
            assert "description" in task
            assert "status" in task
            assert task["status"] == "PENDING"

    def test_service_api_plan(self):
        """Test service API PLAN operation."""
        from brains.agent.autonomous.execution_engine import service_api

        result = service_api({
            "op": "PLAN",
            "payload": {"goal": "Test goal"}
        })

        assert result["ok"] is True
        assert "payload" in result
        assert "tasks" in result["payload"]

    def test_service_api_decompose(self):
        """Test service API DECOMPOSE operation."""
        from brains.agent.autonomous.execution_engine import service_api

        result = service_api({
            "op": "DECOMPOSE",
            "payload": {"goal": "Write a test"}
        })

        assert result["ok"] is True
        assert "subtasks" in result["payload"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
