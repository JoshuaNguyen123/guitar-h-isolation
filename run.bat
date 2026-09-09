@echo off
setlocal
cd /d "%~dp0"

set "APP_PYTHON=%LOCALAPPDATA%\guitar_h_isolation_venv\Scripts\python.exe"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%APP_PYTHON%" if not exist "%VENV_PYTHON%" (
    echo First launch: installing Guitar H Isolation.
    echo.
    call "%~dp0install.bat" nopause
    if errorlevel 1 (
        echo Install did not finish. Fix the error above, then run this file again.
        pause
        exit /b 1
    )
)

set "PYEXE="
if exist "%APP_PYTHON%" (
    set "PYEXE=%APP_PYTHON%"
) else if exist "%VENV_PYTHON%" (
    set "PYEXE=%VENV_PYTHON%"
)

if not defined PYEXE (
    echo Could not find Python after setup.
    echo Tried:
    echo   %APP_PYTHON%
    echo   %VENV_PYTHON%
    pause
    exit /b 1
)

echo Using %PYEXE%
"%PYEXE%" -m src.app.main
if errorlevel 1 pause
