@echo off
setlocal enabledelayedexpansion
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

:: If arguments are provided directly via CLI, forward them to batch_swapper.py
if not "%~1"=="" (
    %PYTHON_EXE% scripts\batch_swapper.py %*
    exit /b !ERRORLEVEL!
)

:: Default server fallback from environment variable or 127.0.0.1:8000
set "DEFAULT_SERVER=%COMFYUI_SERVER%"
if "!DEFAULT_SERVER!"=="" set "DEFAULT_SERVER=127.0.0.1:8000"

:: ============================================================================
:: Streamlined Interactive Menu
:: ============================================================================
:MENU
cls
echo ===========================================================================
echo       BatchPersona - Headless ComfyUI Model Replacement Pipeline
echo ===========================================================================
echo   Python runtime: %PYTHON_EXE%
echo   Server address: !DEFAULT_SERVER!
echo ===========================================================================
echo.
echo   [1] Run Commercial Model Swap  (Campaign Lookbooks - Diverse Personas)
echo   [2] Run Quick Dry-Run          (Instant Zero-GPU Verification, ^< 1s)
echo   [3] Run Quality Gate and Tests (Pytest 55/55 + Coverage + Ruff)
echo   [0] Exit
echo.
echo ===========================================================================
set /p "CHOICE=Select an option [1-3, 0]: "

if "%CHOICE%"=="1" goto :SWAP_CAMPAIGN
if "%CHOICE%"=="2" goto :SWAP_COMPOSITE
if "%CHOICE%"=="3" goto :RUN_QUALITY_GATE
if "%CHOICE%"=="0" goto :EXIT

echo [WARNING] Invalid selection. Please enter a number between 0 and 3.
timeout /t 2 >nul
goto :MENU

:: ============================================================================
:: Action Handlers
:: ============================================================================
:SWAP_CAMPAIGN
echo.
echo [RUNNING] Executing Commercial Batch Model Swap across Campaign Lookbooks...
set "PROCESSED_COUNT=0"
for %%C in (data\input_campaign\campaign_fashion_*.png) do (
    set /a PROCESSED_COUNT+=1
    echo.
    echo ===========================================================================
    echo [CAMPAIGN !PROCESSED_COUNT!] Processing Lookbook: %%~nxC
    echo ===========================================================================
    %PYTHON_EXE% scripts\batch_swapper.py ^
        --server !DEFAULT_SERVER! ^
        --campaign "%%C" ^
        --models-dir data\input_models ^
        --output-dir data\output ^
        --workflow workflows\model_swap_qwen21_maskless_api.json ^
        --market-tag global ^
        --timeout 300.0
)
if !PROCESSED_COUNT! EQU 0 (
    echo [WARNING] No campaign images found matching data\input_campaign\campaign_fashion_*.png
)
goto :FINISH

:SWAP_COMPOSITE
echo.
echo [RUNNING] Executing Instant Composite Test Dry-Run (No GPU Required)...
if not exist "data_synthetic\input_campaign\campaign_summer_lookbook.png" (
    echo [INFO] Generating isolated synthetic test dataset for dry-run...
    %PYTHON_EXE% scripts\generate_testdata.py --output-dir data_synthetic --size 896
)
%PYTHON_EXE% scripts\batch_swapper.py ^
    --server !DEFAULT_SERVER! ^
    --campaign data_synthetic\input_campaign\campaign_summer_lookbook.png ^
    --mask data_synthetic\input_campaign\campaign_summer_mask.png ^
    --models-dir data_synthetic\input_models ^
    --output-dir data_synthetic\output ^
    --workflow workflows\model_swap_composite_api.json ^
    --market-tag test ^
    --timeout 30.0
goto :FINISH

:RUN_QUALITY_GATE
echo.
echo [RUNNING] Running Complete Quality Gate and CI Verification locally...
%PYTHON_EXE% scripts\run_ci_locally.py
goto :FINISH

:: ============================================================================
:: Finish & Exit
:: ============================================================================
:FINISH
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if %EXIT_CODE% EQU 0 (
    echo [SUCCESS] Operation completed successfully.
) else (
    echo [FAILED] Operation exited with error code: %EXIT_CODE%
)
echo.
pause
goto :MENU

:EXIT
echo Exiting BatchPersona. Goodbye!
exit /b 0
