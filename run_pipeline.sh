#!/usr/bin/env bash
# ==============================================================================
# BatchPersona - Interactive & CLI Pipeline Runner (Linux / macOS)
# ==============================================================================
set -e

# Auto-detect Python executable
PYTHON_EXE="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
fi

# Forward CLI arguments directly if supplied
if [ "$#" -gt 0 ]; then
    exec $PYTHON_EXE scripts/batch_swapper.py "$@"
fi

# Default server fallback
DEFAULT_SERVER="${COMFYUI_SERVER:-127.0.0.1:8000}"

while true; do
    clear
    echo "==========================================================================="
    echo "      BatchPersona - Headless ComfyUI Model Replacement Pipeline"
    echo "==========================================================================="
    echo "  Python runtime: $PYTHON_EXE"
    echo "  Default server: $DEFAULT_SERVER"
    echo "==========================================================================="
    echo ""
    echo "  [1] Run Model Swap (Female Lookbook -> Multi-Ethnic Models, Qwen DiT)"
    echo "  [2] Run Model Swap (Male Lookbook -> Multi-Ethnic Models, Qwen DiT)"
    echo "  [3] Run Test Dry-Run (Composite Swap - Instant Zero-GPU Verification)"
    echo "  [4] Run Model Swap with Custom Server Address"
    echo "  [5] Generate Synthetic Test Dataset (Zero Downloads)"
    echo "  [6] Standardize Raw Photos to 896x1152 (Aspect-Fit Resampler)"
    echo "  [7] Run Automated Test Suite (Pytest + Coverage)"
    echo "  [8] Run Code Linter & Style Format (Ruff)"
    echo "  [9] Run Full GitHub Actions CI Gate Locally (Lint + Format + Smoke + Coverage)"
    echo "  [0] Exit"
    echo ""
    echo "==========================================================================="
    read -r -p "Select an option [1-9, 0]: " CHOICE

    case "$CHOICE" in
        1)
            echo ""
            echo "[RUNNING] Executing Batch Model Swap for Female Lookbook..."
            $PYTHON_EXE scripts/batch_swapper.py \
                --server "$DEFAULT_SERVER" \
                --campaign data/input_campaign/campaign_fashion_female.png \
                --models-dir data/input_models \
                --output-dir data/output \
                --workflow workflows/model_swap_qwen21_maskless_api.json \
                --market-tag apac \
                --timeout 300.0
            read -r -p "Press Enter to return to menu..."
            ;;
        2)
            echo ""
            echo "[RUNNING] Executing Batch Model Swap for Male Lookbook..."
            $PYTHON_EXE scripts/batch_swapper.py \
                --server "$DEFAULT_SERVER" \
                --campaign data/input_campaign/campaign_fashion_male.png \
                --models-dir data/input_models \
                --output-dir data/output \
                --workflow workflows/model_swap_qwen21_maskless_api.json \
                --market-tag global \
                --timeout 300.0
            read -r -p "Press Enter to return to menu..."
            ;;
        3)
            echo ""
            echo "[RUNNING] Executing Instant Composite Test Dry-Run (No GPU Required)..."
            if [ ! -f "data_synthetic/input_campaign/campaign_summer_lookbook.png" ]; then
                echo "[INFO] Generating isolated synthetic test dataset for dry-run..."
                $PYTHON_EXE scripts/generate_testdata.py --output-dir data_synthetic --size 896
            fi
            $PYTHON_EXE scripts/batch_swapper.py \
                --server "$DEFAULT_SERVER" \
                --campaign data_synthetic/input_campaign/campaign_summer_lookbook.png \
                --mask data_synthetic/input_campaign/campaign_summer_mask.png \
                --models-dir data_synthetic/input_models \
                --output-dir data_synthetic/output \
                --workflow workflows/model_swap_composite_api.json \
                --market-tag test \
                --timeout 30.0
            read -r -p "Press Enter to return to menu..."
            ;;
        4)
            echo ""
            read -r -p "Enter ComfyUI server address (e.g. 127.0.0.1:8188 or 127.0.0.1:8000): " SERVER_INPUT
            TARGET_SERVER="${SERVER_INPUT:-$DEFAULT_SERVER}"
            $PYTHON_EXE scripts/batch_swapper.py \
                --server "$TARGET_SERVER" \
                --campaign data/input_campaign/campaign_fashion_female.png \
                --models-dir data/input_models \
                --output-dir data/output \
                --workflow workflows/model_swap_qwen21_maskless_api.json \
                --market-tag apac
            read -r -p "Press Enter to return to menu..."
            ;;
        5)
            echo ""
            echo "[RUNNING] Generating procedural synthetic test dataset..."
            $PYTHON_EXE scripts/generate_testdata.py --output-dir data_synthetic --size 896
            read -r -p "Press Enter to return to menu..."
            ;;
        6)
            echo ""
            read -r -p "Enter directory containing raw candidate images (default: data): " RAW_DIR
            TARGET_DIR="${RAW_DIR:-data}"
            $PYTHON_EXE scripts/prepare_fullbody_dataset.py --source-dir "$TARGET_DIR" --output-dir data
            read -r -p "Press Enter to return to menu..."
            ;;
        7)
            echo ""
            echo "[RUNNING] Running automated pytest test suite and coverage check..."
            $PYTHON_EXE -m pytest tests --cov=scripts --cov-report=term-missing -v
            read -r -p "Press Enter to return to menu..."
            ;;
        8)
            echo ""
            echo "[RUNNING] Running Ruff linter and code formatter..."
            $PYTHON_EXE -m ruff check .
            $PYTHON_EXE -m ruff format --check .
            read -r -p "Press Enter to return to menu..."
            ;;
        9)
            echo ""
            echo "[RUNNING] Running Complete GitHub Actions CI Gate locally..."
            $PYTHON_EXE scripts/run_ci_locally.py
            read -r -p "Press Enter to return to menu..."
            ;;
        0)
            echo "Exiting BatchPersona. Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid selection."
            sleep 1
            ;;
    esac
done
