@echo off
title OptiDrug Platform Launcher
echo ===================================================
echo           OptiDrug Platform Launcher
echo ===================================================
echo.

:: Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -m pip install -r requirements.txt

echo.
echo [2/3] Starting OptiDrug Server on http://localhost:5000 ...
echo [3/3] Press CTRL+C in this window anytime to stop the server.
echo.

:: Automatically open default browser after a brief delay
start "" http://localhost:5000

:: Start Flask app
python server.py

pause
