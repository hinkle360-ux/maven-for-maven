@echo off
REM SAFE RESTORE-ALL FROM GOLDEN
REM Place this file at: brains\governance\repair_engine\tools\safe_restore_all_from_golden.cmd
REM Usage (from maven root): brains\governance\repair_engine\tools\safe_restore_all_from_golden baseline_1761871530

setlocal ENABLEEXTENSIONS
cd /d "%~dp0"
REM Go to repo root (..\..\..\.. = from tools/ to maven/)
pushd ..\..\..\..
if "%~1"=="" (
  echo Usage: brains\governance\repair_engine\tools\safe_restore_all_from_golden baseline_1761871530
  popd
  exit /b 2
)

echo =====================================================
echo  DANGER: This will restore ALL brains from GOLDEN.
echo  Version: %1
echo  Location: templates\golden\*\%1\
echo =====================================================
set /p CONFIRM=Type EXACTLY: RESTORE ALL FROM GOLDEN  ^> 
if /I not "%CONFIRM%"=="RESTORE ALL FROM GOLDEN" (
  echo Cancelled.
  popd
  exit /b 1
)

set "PYTHONPATH=%CD%"
python brains\governance\repair_engine\tools\safe_restore_all.py %1
set rc=%ERRORLEVEL%
popd
exit /b %rc%
