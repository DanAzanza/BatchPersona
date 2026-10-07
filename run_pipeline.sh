#!/usr/bin/env bash
# ==============================================================================
# BatchPersona - Streamlined Interactive Pipeline Runner (Linux / macOS)
# ==============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Auto-detect Python runtime
PYTHON_EXE="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
else
    echo "[NOTICE] Virtual environment .venv not found. Running installer first..."
    ./install.sh
    if [ -f ".venv/bin/python" ]; then
        PYTHON_EXE=".venv/bin/python"
    fi
fi

# Pass CLI arguments directly if provided
if [ $# -gt 0 ]; then
    "$PYTHON_EXE" scripts/batch_swapper.py "$@"
    exit $?
fi

# Default server fallback
DEFAULT_SERVER="${COMFYUI_SERVER:-127.0.0.1:8000}"

while true; do
    clear
    echo "==========================================================================="
    echo "      BatchPersona - Headless ComfyUI Model Replacement Pipeline"
    echo "==========================================================================="
    echo "  Python runtime: $PYTHON_EXE"
    echo "  Server address: $DEFAULT_SERVER"
    echo "==========================================================================="
    echo ""
    echo "  [1] Run Commercial Model Swap  (Campaign Lookbooks - Diverse Personas)"
    echo "  [2] Run Quick Dry-Run          (Instant Zero-GPU Verification, < 1s)"
    echo "  [3] Run Quality Gate and Tests (Pytest 55/55 + Coverage + Ruff)"
    echo "  [0] Exit"
    echo ""
    echo "==========================================================================="
    read -r -p "Select an option [1-3, 0]: " CHOICE

    case "$CHOICE" in
        1)
            echo ""
            echo "[RUNNING] Executing Commercial Batch Model Swap across Campaign Lookbooks..."
            for campaign_file in data/input_campaign/campaign_fashion_*.png; do
                if [ -f "$campaign_file" ]; then
                    echo ""
                    echo "==========================================================================="
                    echo "[CAMPAIGN] Processing Lookbook: $(basename "$campaign_file")"
                    echo "==========================================================================="
                    $PYTHON_EXE scripts/batch_swapper.py \
                        --server "$DEFAULT_SERVER" \
                        --campaign "$campaign_file" \
                        --models-dir data/input_models \
                        --output-dir data/output \
                        --workflow workflows/model_swap_qwen21_maskless_api.json \
                        --market-tag global \
                        --timeout 300.0
                fi
            done
            read -r -p "Press Enter to return to menu..."
            ;;
        2)
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
        3)
            echo ""
            echo "[RUNNING] Running Complete Quality Gate and CI Verification locally..."
            $PYTHON_EXE scripts/run_ci_locally.py
            read -r -p "Press Enter to return to menu..."
            ;;
        0)
            echo "Exiting BatchPersona. Goodbye!"
            exit 0
            ;;
        *)
            echo "[WARNING] Invalid selection: $CHOICE. Please choose between 0 and 3."
            sleep 2
            ;;
    esac
done
