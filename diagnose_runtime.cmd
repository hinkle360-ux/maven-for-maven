@echo off
REM Diagnostic script to identify runtime environment issues
REM This will show you EXACTLY where your code is running from

echo ================================================================================
echo MAVEN RUNTIME DIAGNOSTICS
echo ================================================================================
echo.

echo 1. CURRENT WORKING DIRECTORY:
cd
echo.

echo 2. SCRIPT LOCATION (where this .cmd file is):
echo %~dp0
echo.

echo 3. CHECKING IF RUNNING FROM CORRECT LOCATION:
if exist "%~dp0ui\maven_chat.py" (
    echo    [OK] maven_chat.py found in ui\ subdirectory
) else (
    echo    [ERROR] maven_chat.py NOT FOUND in ui\ subdirectory
    echo    This means you're NOT running from the maven2_fix folder!
)
echo.

echo 4. CHECKING FOR UPDATED ROUTING CODE:
if exist "%~dp0brains\cognitive\memory_librarian\service\librarian_memory.py" (
    echo    [OK] librarian_memory.py found
    findstr /C:"def learn_routing_for_question" "%~dp0brains\cognitive\memory_librarian\service\librarian_memory.py" >nul
    if errorlevel 1 (
        echo    [ERROR] learn_routing_for_question NOT FOUND in librarian_memory.py
        echo    You're running OLD CODE!
    ) else (
        echo    [OK] learn_routing_for_question function FOUND
    )
) else (
    echo    [ERROR] librarian_memory.py NOT FOUND
    echo    Directory structure is wrong!
)
echo.

echo 5. CHECKING REASONING BRAIN FOR ROUTING LOGS:
if exist "%~dp0brains\cognitive\reasoning\service\reasoning_brain.py" (
    echo    [OK] reasoning_brain.py found
    findstr /C:"[ROUTING_LEARNING]" "%~dp0brains\cognitive\reasoning\service\reasoning_brain.py" >nul
    if errorlevel 1 (
        echo    [ERROR] [ROUTING_LEARNING] log marker NOT FOUND
        echo    You're running OLD CODE!
    ) else (
        echo    [OK] [ROUTING_LEARNING] log marker FOUND
    )
) else (
    echo    [ERROR] reasoning_brain.py NOT FOUND
)
echo.

echo 6. PYTHON VERSION CHECK:
py -3.11 --version 2>nul
if errorlevel 1 (
    python --version
) else (
    py -3.11 --version
)
echo.

echo 7. RUNNING PYTHON VERIFICATION SCRIPT:
echo.
if exist "%~dp0verify_runtime_version.py" (
    py -3.11 -m verify_runtime_version 2>nul || python verify_runtime_version.py
) else (
    echo    [ERROR] verify_runtime_version.py not found!
)

echo.
echo ================================================================================
echo DIAGNOSTICS COMPLETE
echo ================================================================================
echo.
echo If you see any [ERROR] messages above, your runtime environment is NOT
echo using the updated code from maven2_fix folder.
echo.
echo Common issues:
echo   1. Running from C:\Windows\System32 instead of maven2_fix folder
echo   2. Double-clicking run_chat.cmd from a different location
echo   3. Old Python bytecode (.pyc) files cached
echo   4. PYTHONPATH pointing to wrong directory
echo.
echo Solution: Navigate to maven2_fix folder in File Explorer, then
echo           double-click run_chat.cmd from INSIDE that folder.
echo.
pause
