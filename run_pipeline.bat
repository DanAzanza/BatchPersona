@echo off
setlocal
chcp 65001 >nul
title BatchPersona - Automated Headless Model Replacement

:: ============================================================================
:: Python & Virtual Environment Auto-Detection
:: ============================================================================
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    echo [NOTICE] Virtual environment .venv not found.
    echo Running automated installation first...
    call install.bat
    if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
)

%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH or .venv directory.
    echo Please run install.bat or install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

:: ============================================================================
:: Forward all execution directly to the Python orchestrator
:: ============================================================================
%PYTHON_EXE% -m scripts.run_pipeline %*
exit /b %ERRORLEVEL%
