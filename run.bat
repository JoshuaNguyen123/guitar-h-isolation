@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create a virtual environment and install requirements first.
    echo See README.md
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m src.app.main
