import json
from pathlib import Path

from brains.tools import execution_guard


def test_execution_env_enables(monkeypatch):
    monkeypatch.setenv("MAVEN_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("USER_CONFIRMED_EXECUTION", "YES")
    enabled, reason = execution_guard.check_execution_enabled()
    status = execution_guard.get_execution_status()
    assert enabled is True
    assert status["enabled"] is True and status["source"] == "env"
    assert reason == ""


def test_execution_config_enables(monkeypatch, tmp_path):
    monkeypatch.delenv("MAVEN_EXECUTION_ENABLED", raising=False)
    monkeypatch.delenv("USER_CONFIRMED_EXECUTION", raising=False)

    monkeypatch.setattr(execution_guard.Path, "home", lambda: Path(tmp_path))
    execution_guard.set_execution_config(True, True)

    enabled, reason = execution_guard.check_execution_enabled()
    status = execution_guard.get_execution_status()
    assert enabled is True
    assert status["source"] == "config"


def test_execution_disabled_reason(monkeypatch, tmp_path):
    monkeypatch.delenv("MAVEN_EXECUTION_ENABLED", raising=False)
    monkeypatch.delenv("USER_CONFIRMED_EXECUTION", raising=False)
    monkeypatch.setattr(execution_guard.Path, "home", lambda: Path(tmp_path))
    # Ensure config is disabled
    execution_guard.set_execution_config(False, False)

    enabled, reason = execution_guard.check_execution_enabled()
    status = execution_guard.get_execution_status()

    assert enabled is False
    assert "Execution disabled" in reason
    assert status["enabled"] is False
