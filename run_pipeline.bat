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
)

%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH or .venv directory.
    echo Please install Python 3.10+ and ensure it is added to your PATH.
    pause
    exit /b 1
)

:: If arguments are provided directly via CLI, forward them to batch_swapper.py
if not "%~1"=="" (
    %PYTHON_EXE% scripts\batch_swapper.py %*
    exit /b !ERRORLEVEL!
)

:: ============================================================================
:: Interactive Menu Mode
:: ============================================================================
:MENU
cls
echo ===========================================================================
echo       BatchPersona - Headless ComfyUI Model Replacement Pipeline
echo ===========================================================================
echo   Python runtime: %PYTHON_EXE%
echo   Default server: 127.0.0.1:8000
echo ===========================================================================
echo.
echo   [1] Run Model Swap (Female Lookbook -> Multi-Ethnic Models)
echo   [2] Run Model Swap (Male Lookbook -> Multi-Ethnic Models)
echo   [3] Run Model Swap with Custom Server Address
echo   [4] Generate Synthetic Test Dataset (Zero Downloads)
echo   [5] Standardize Raw Photos to 896x1152 (Aspect-Fit Resampler)
echo   [6] Run Automated Test Suite (Pytest + Coverage)
echo   [7] Run Code Linter & Style Format (Ruff)
echo   [0] Exit
echo.
echo ===========================================================================
set /p "CHOICE=Select an option [1-7, 0]: "

if "%CHOICE%"=="1" goto :SWAP_FEMALE
if "%CHOICE%"=="2" goto :SWAP_MALE
if "%CHOICE%"=="3" goto :SWAP_CUSTOM
if "%CHOICE%"=="4" goto :GEN_DATA
if "%CHOICE%"=="5" goto :PREP_DATA
if "%CHOICE%"=="6" goto :RUN_TESTS
if "%CHOICE%"=="7" goto :RUN_LINT
if "%CHOICE%"=="0" goto :EXIT

echo [WARNING] Invalid selection. Please enter a number between 0 and 7.
timeout /t 2 >nul
goto :MENU

:: ============================================================================
:: Action Handlers
:: ============================================================================
:SWAP_FEMALE
echo.
echo [RUNNING] Executing Batch Model Swap for Female Lookbook...
%PYTHON_EXE% scripts\batch_swapper.py ^
    --server 127.0.0.1:8000 ^
    --campaign data\input_campaign\campaign_fashion_female.png ^
    --models-dir data\input_models ^
    --output-dir data\output ^
    --workflow workflows\model_swap_qwen21_maskless_api.json ^
    --market-tag apac ^
    --timeout 300.0
goto :FINISH

:SWAP_MALE
echo.
echo [RUNNING] Executing Batch Model Swap for Male Lookbook...
%PYTHON_EXE% scripts\batch_swapper.py ^
    --server 127.0.0.1:8000 ^
    --campaign data\input_campaign\campaign_fashion_male.png ^
    --models-dir data\input_models ^
    --output-dir data\output ^
    --workflow workflows\model_swap_qwen21_maskless_api.json ^
    --market-tag global ^
    --timeout 300.0
goto :FINISH

:SWAP_CUSTOM
echo.
set /p "SERVER_ADDR=Enter ComfyUI server address (e.g. 127.0.0.1:8188 or 127.0.0.1:8000): "
if "!SERVER_ADDR!"=="" set "SERVER_ADDR=127.0.0.1:8000"
echo [RUNNING] Executing with server: !SERVER_ADDR!...
%PYTHON_EXE% scripts\batch_swapper.py ^
    --server !SERVER_ADDR! ^
    --campaign data\input_campaign\campaign_fashion_female.png ^
    --models-dir data\input_models ^
    --output-dir data\output ^
    --workflow workflows\model_swap_qwen21_maskless_api.json ^
    --market-tag apac
goto :FINISH

:GEN_DATA
echo.
echo [RUNNING] Generating procedural synthetic test dataset...
%PYTHON_EXE% scripts\generate_testdata.py --output-dir data --size 1024
goto :FINISH

:PREP_DATA
echo.
set /p "RAW_DIR=Enter directory containing raw candidate images (default: data): "
if "!RAW_DIR!"=="" set "RAW_DIR=data"
echo [RUNNING] Standardizing dataset to 896x1152...
%PYTHON_EXE% scripts\prepare_fullbody_dataset.py --source-dir "!RAW_DIR!" --output-dir data
goto :FINISH

:RUN_TESTS
echo.
echo [RUNNING] Running automated pytest test suite and coverage check...
%PYTHON_EXE% -m pytest tests\test_batch_swapper.py --cov=scripts --cov-report=term-missing -v
goto :FINISH

:RUN_LINT
echo.
echo [RUNNING] Running Ruff linter and code formatter...
%PYTHON_EXE% -m ruff check .
%PYTHON_EXE% -m ruff format --check .
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
