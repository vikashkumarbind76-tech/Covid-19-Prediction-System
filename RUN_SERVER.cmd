@echo off
echo ========================================
echo COVID-19 Prediction System - Server
echo ========================================
echo.
echo Starting Flask server...
echo.
echo The server will run on: http://127.0.0.1:5000
echo.
echo To access:
echo - Main Dashboard: http://127.0.0.1:5000/
echo - Admin Panel: http://127.0.0.1:5000/admin
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python...
)

REM Run Flask app
python api\app.py

pause
