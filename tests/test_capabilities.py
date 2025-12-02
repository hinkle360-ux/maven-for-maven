from capabilities import get_capabilities


def test_capabilities_reflect_execution(monkeypatch):
    # Force execution disabled to see propagated state
    monkeypatch.setenv("MAVEN_EXECUTION_ENABLED", "0")
    caps = get_capabilities()
    assert caps["filesystem_agency"]["enabled"] is False
    assert caps["execution_agency"]["enabled"] is False


def test_capabilities_enabled_when_env_on(monkeypatch):
    monkeypatch.setenv("MAVEN_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("USER_CONFIRMED_EXECUTION", "YES")
    caps = get_capabilities()
    assert caps["filesystem_agency"]["enabled"] is True
    assert caps["git_agency"]["enabled"] is True
