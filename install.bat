@echo off
setlocal
cd /d "%~dp0"

set "NOPAUSE="
if /I "%~1"=="nopause" set "NOPAUSE=1"

echo Guitar H Isolation - one-time setup
echo.

set "VENV=%LOCALAPPDATA%\guitar_h_isolation_venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "BOOT="

where py >nul 2>&1
if not errorlevel 1 (
    set "BOOT=py -3"
) else (
    where python >nul 2>&1
    if not errorlevel 1 set "BOOT=python"
)

if not defined BOOT (
    echo Python 3 was not found.
    echo Install it from https://www.python.org/downloads/
    echo Turn on "Add python.exe to PATH", then double-click this file again.
    if not defined NOPAUSE pause
    exit /b 1
)

if not exist "%PYEXE%" (
    echo Creating the app environment in:
    echo   %VENV%
    %BOOT% -m venv "%VENV%"
    if errorlevel 1 (
        echo Could not create the environment.
        if not defined NOPAUSE pause
        exit /b 1
    )
)

echo Installing packages. The first time can take a few minutes.
"%PYEXE%" -m pip install -U pip
if errorlevel 1 goto :fail

rem Avoid TensorFlow. basic-pitch on Python 3.11 pulls it unless we skip deps.
"%PYEXE%" -m pip install customtkinter "demucs-onnx>=0.3.0" librosa soundfile numpy onnxruntime "huggingface-hub>=0.32.0" pretty-midi resampy scipy scikit-learn soxr tqdm mir-eval
if errorlevel 1 goto :fail
"%PYEXE%" -m pip install --no-deps basic-pitch
if errorlevel 1 goto :fail

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo FFmpeg is missing. Trying winget...
    winget install --accept-package-agreements --accept-source-agreements Gyan.FFmpeg
    echo If ffmpeg is still missing, close every terminal, then run run.bat again.
)

"%PYEXE%" -c "import customtkinter, demucs_onnx, librosa, onnxruntime, basic_pitch, mir_eval; print('Install OK')"
if errorlevel 1 goto :fail

echo.
echo Setup finished. Double-click run.bat to open the app.
if not defined NOPAUSE pause
exit /b 0

:fail
echo Setup failed. See the text above.
if not defined NOPAUSE pause
exit /b 1
