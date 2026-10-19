@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title BatchPersona - Automated Setup and Installation

echo ===========================================================================
echo       BatchPersona - One-Click Environment Setup and Self-Test
echo ===========================================================================
echo.

:: 1. Verify Python availability and version
set "SYSTEM_PYTHON=python"
%SYSTEM_PYTHON% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('%SYSTEM_PYTHON% --version') do set "PY_VER=%%v"
echo [OK] Detected Python %PY_VER%

:: 2. Create Python virtual environment (.venv)
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    %SYSTEM_PYTHON% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment .venv already exists.
)

set "VENV_PYTHON=.venv\Scripts\python.exe"

:: 3. Check and install dependencies
echo.
%VENV_PYTHON% -c "import requests, websocket, PIL, numpy, pytest" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] All required Python dependencies are already installed.
) else (
    echo [INFO] Installing production dependencies and development extras...
    %VENV_PYTHON% -m pip install -r requirements.txt
    %VENV_PYTHON% -m pip install -e ".[dev]"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install dependencies.
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully.
)

:: 4. Verify input dataset directories
echo.
if not exist "data\input_campaign" mkdir "data\input_campaign"
if not exist "data\input_models" mkdir "data\input_models"
if not exist "data\output" mkdir "data\output"
echo [OK] Workspace data directories verified in data\

:: 5. Run automated self-test verification suite
echo.
echo [INFO] Running automated self-test verification suite...
%VENV_PYTHON% -m pytest tests -q
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Test suite reported warnings or failures.
) else (
    echo [OK] All automated tests passed with 100%% success.
)

:: 6. Setup complete banner
echo.
echo ===========================================================================
echo       [SUCCESS] BatchPersona Installation Completed Successfully!
echo ===========================================================================
echo   To launch the interactive model replacement pipeline, simply run:
echo     run_pipeline.bat
echo ===========================================================================
if "%~1"=="" pause
exit /b 0
