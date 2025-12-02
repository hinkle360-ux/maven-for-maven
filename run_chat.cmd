@echo off
cd /d "%~dp0"
rem Simple launcher for the Maven chat interface on Windows.
rem
rem This script locates a suitable Python 3.11 interpreter, clears stale cache,
rem adds the Maven project root to PYTHONPATH so that package imports work,
rem and then launches the chat interface via Python's module system.

setlocal

rem Attempt to use the py launcher for Python 3.11; fall back to plain python.
set "PYEXE=py -3.11"
%PYEXE% --version >nul 2>&1 || set "PYEXE=python"

rem Compute the absolute path to this script's directory (the Maven project root).
set "SCRIPT_DIR=%~dp0"

rem Clear Python bytecode cache to ensure latest code runs
echo Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul

rem Clear stale response cache from previous sessions
echo Clearing stale response cache...
if exist "reports\system" del /q "reports\system\*.json" 2>nul
if exist "reports\context_snapshot.json" del /q "reports\context_snapshot.json" 2>nul

rem Ensure that the Maven project root is on PYTHONPATH so imports like 'api'
rem resolve correctly when launching modules below.
set "PYTHONPATH=%SCRIPT_DIR%"

echo Starting Maven chat interface...
echo.

rem Launch the chat interface as a module.  There is no nested 'maven'
rem package in this build, so run the ``ui.maven_chat`` package directly.
%PYEXE% -m ui.maven_chat %*

rem Keep the console window open after execution so the user can read the output.
pause