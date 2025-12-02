import version_utils


def test_get_version_info_git_success(monkeypatch):
    class DummyResult:
        def __init__(self, output: str):
            self.stdout = output
            self.stderr = ""

    def fake_run(cmd, check, stdout, stderr, text):
        if "--short" in cmd:
            return DummyResult("abc123\n")
        return DummyResult("main\n")

    monkeypatch.setattr(version_utils.subprocess, "run", fake_run)
    info = version_utils.get_version_info()
    assert info["commit"] == "abc123"
    assert info["branch"] == "main"
    assert info["features"] != "unknown"


def test_get_version_info_fallback(monkeypatch):
    monkeypatch.setattr(version_utils, "_run_git_command", lambda args: "")
    monkeypatch.setattr(
        version_utils,
        "_read_version_file",
        lambda root: {"commit": "test123", "branch": "dev", "features": "f1,f2"},
    )
    info = version_utils.get_version_info()
    assert info["commit"] == "test123"
    assert info["branch"] == "dev"
    assert info["features"] == "f1,f2"
