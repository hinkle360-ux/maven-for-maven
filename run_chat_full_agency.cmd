@echo off
cd /d "%~dp0"
rem Launcher for Maven FULL AGENCY mode (unrestricted tool access)
rem
rem This script launches Maven in FULL_AGENCY profile where:
rem - Read/write files anywhere on disk (OS permissions apply)
rem - Run shell commands
rem - Run Python code
rem - Browse the web
rem - Full git operations (clone, push, etc.)
rem - Run autonomous agents
rem
rem Only truly destructive commands (rm -rf /, mkfs, etc.) are blocked.

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

rem Ensure that the Maven project root is on PYTHONPATH
set "PYTHONPATH=%SCRIPT_DIR%"

rem Set the capability profile to FULL_AGENCY
set "MAVEN_CAPABILITIES_PROFILE=FULL_AGENCY"

echo.
echo ==========================================
echo   MAVEN CHAT - FULL AGENCY MODE
echo   Unrestricted access to all tools
echo ==========================================
echo.
echo WARNING: This mode allows Maven to execute shell commands,
echo access files anywhere, and run code. Use with care!
echo.

rem Launch the chat interface as a module
%PYEXE% -m ui.maven_chat %*

rem Keep the console window open after execution
pause
