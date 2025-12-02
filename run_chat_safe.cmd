@echo off
cd /d "%~dp0"
rem Launcher for Maven SAFE CHAT mode (no tools, pure conversation)
rem
rem This script launches Maven in SAFE_CHAT profile where:
rem - No file access (read/write)
rem - No shell/Python execution
rem - No web access
rem - No git operations
rem - Pure conversation only

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

rem Set the capability profile to SAFE_CHAT
set "MAVEN_CAPABILITIES_PROFILE=SAFE_CHAT"

echo.
echo ==========================================
echo   MAVEN CHAT - SAFE MODE
echo   No tools, pure conversation
echo ==========================================
echo.

rem Launch the chat interface as a module
%PYEXE% -m ui.maven_chat %*

rem Keep the console window open after execution
pause
