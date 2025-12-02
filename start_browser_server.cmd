@echo off
cd /d "%~dp0"
echo ========================================================================
echo Maven Browser Runtime Server
echo ========================================================================
echo.
echo Starting server on http://127.0.0.1:8765 ...
echo Press Ctrl+C to stop the server.
echo.
python run_browser_server.py
pause
