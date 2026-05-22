@echo off
echo ========================================
echo COVID-19 Prediction System - Setup
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Starting Flask server...
echo.
echo The server will run on: http://127.0.0.1:5000
echo.
echo To access:
echo - Main Dashboard: http://127.0.0.1:5000/
echo - Admin Panel: http://127.0.0.1:5000/admin
echo   (Default password: Vikash09)
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Run Flask app
python api\app.py

pause
