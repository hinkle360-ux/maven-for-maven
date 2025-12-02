@echo off
rem Cleanup script for removing junk and temporary files from the Maven project.
rem
rem This script removes:
rem - Python bytecode files (*.pyc)
rem - __pycache__ directories
rem - Log files (*.log)
rem - Temporary test files (tmp_*.txt, *_tmp.txt)
rem - Patch test files (patch_test*.txt)
rem - Other temporary files
rem
rem You can simply double-click this file to clean up junk files.

setlocal

rem Compute the absolute path to this script's directory (the Maven project root).
set "SCRIPT_DIR=%~dp0"

echo.
echo ============================================
echo Maven Project Cleanup Utility
echo ============================================
echo.
echo Cleaning up junk files in: %SCRIPT_DIR%
echo.

rem Count files before cleanup
set /a COUNT=0

rem Remove Python bytecode files
echo [1/5] Removing Python bytecode files (*.pyc)...
for /r "%SCRIPT_DIR%" %%F in (*.pyc) do (
    del /q "%%F" 2>nul && echo   Deleted: %%F && set /a COUNT+=1
)

rem Remove __pycache__ directories
echo.
echo [2/5] Removing __pycache__ directories...
for /d /r "%SCRIPT_DIR%" %%D in (__pycache__) do (
    if exist "%%D" (
        rmdir /s /q "%%D" 2>nul && echo   Deleted: %%D && set /a COUNT+=1
    )
)

rem Remove log files
echo.
echo [3/5] Removing log files (*.log)...
for /r "%SCRIPT_DIR%" %%F in (*.log) do (
    del /q "%%F" 2>nul && echo   Deleted: %%F && set /a COUNT+=1
)

rem Remove temporary test files
echo.
echo [4/5] Removing temporary test files (tmp_*.txt, *_tmp.txt)...
for /r "%SCRIPT_DIR%" %%F in (tmp_*.txt *_tmp.txt) do (
    del /q "%%F" 2>nul && echo   Deleted: %%F && set /a COUNT+=1
)

rem Remove patch test files
echo.
echo [5/5] Removing patch test files (patch_test*.txt)...
for /r "%SCRIPT_DIR%" %%F in (patch_test*.txt) do (
    del /q "%%F" 2>nul && echo   Deleted: %%F && set /a COUNT+=1
)

echo.
echo ============================================
echo Cleanup Complete!
echo ============================================
echo.

rem Keep the console window open so the user can see the results.
pause
